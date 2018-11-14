#!/usr/bin/env python
# To do: Point sources, Blank fields, arf_rmf, sb_profiles
# Optimizations: Run the flux, flare, blk ... with just 2 or 3 ccd

import astropy.io.ascii as at
from ciao_contrib.runtool import *

import sys, os
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from cosmo import *
from ancila import *
from mbproj2_sbprofile import *
from rebin_projsb import *
from noise import *

bandkE=[
    [ 0.5, 0.75], [ 0.75,1.0], [1.0,1.25],
    [1.25,1.5], [1.5,2.0], [2.0,3.0],
    [3.0,4.0], [4.0,5.0], [5.0,6.0],
    [6.0,7.0]
    ]
## -------------------------------
## Input Variables
def main():
    parser = argparse.ArgumentParser(
        description='X-ray Pipeline for multi Chandra Observations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('obsids',nargs='+',
                        help='enter the obsid numbers')
    parser.add_argument('Xra',type=float,
                        help='X-ray Ra')
    parser.add_argument('Xdec',type=float,
                        help='X-ray Dec')
    parser.add_argument('R500',type=float,
                        help='R500 [kpc]')
    parser.add_argument('name',
                        help='Name of the cluster')
    parser.add_argument('redshift',
                        help='redshift of the cluster')
    args = parser.parse_args()
    core = 8
    binf = 4
    radec = [args.Xra,args.Xdec]

    ## -------------------------------
    ## Flags
    down=1;reproc=1;flares=1;
    flux=1;ps=1;cen=1;blk=1;
    src=1;arf=0;rprof=1;check=1;
    poisson=1;cshift=1;conce=1;
    save=1;

    t0 = time.time()

    if (args.R500) < 0:
        cen=0;cshift=0;conce=0;
        print("R500 is not valuable number")

    nobs = len(args.obsids)

    ## -------------------------------
    ## Definindo distâncias e raios

    DA = distance(float(args.redshift))     # em kpc
    themax = 300                            # 5' arcmin de abertura
    kpc_arcsec = (1/(3600*180/np.pi))*DA    # kpc/arcsec
    the500kpc = 500/kpc_arcsec              # 500kpc in physical units
    the_R500 = args.R500/kpc_arcsec         # R500 in arcsec
    print("kpc_arcsec:",kpc_arcsec)
    ## -------------------------------
    if 2*the500kpc < 3*60:
        aperture = 'SAM'   # Small aperture mode
        rmax = 4*60/0.492
    else:
        aperture = 'LAM'   # Large aperture mode
        rmax = 8*60/0.492
    ## -------------------------------
    ## Files
    root = os.getcwd()
    merge = os.path.join(root,args.name)
    fluxroot = os.path.join(merge,'flux')
    specroot = os.path.join(merge,'spec')
    profroot = os.path.join(merge,'profiles')
    noiseroot = os.path.join(merge,'noise')


    ## Merged images and event file
    evt = os.path.join(merge,'merged_evt.fits')
    evt2 = os.path.join(merge,'merged_evt_gti.fits')
    ctimg = os.path.join(merge,'broad_thresh.img')
    emap = os.path.join(merge,'broad_thresh.expmap')
    fimg = os.path.join(merge,'broad_thresh.img')
    mask = os.path.join(merge,"no_ps_fov.mask")
    source = os.path.join(merge,"source.reg")

    print("--------------------------------------------")
    print("Cluster :",args.name,"at redshfit ",args.redshift," and R500=",args.R500)
    ids = ','.join(args.obsids)
    print("Chandra observations:",ids)
    print("--------------------------------------------")
    print("Download data:",down)
    if down:
        if not os.path.exists(os.path.join(root,'%s'%(args.obsids[0]))):
            download(ids)
            print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Reprocess data to lvl.2:",reproc)
    if reproc:
        count = 0
        for obsid in args.obsids:
            evt2reproi = os.path.join(root,"%i"%(int(obsid)),"repro","acisf%05i_repro_evt2.fits"%(int(obsid)))
            if not os.path.exists(evt2reproi):
                count += 1
        if count > 0:
            chandra_repro.check_vf_pha = 'yes'
            chandra_repro.set_ardlib = 'no'
            chandra_repro.clobber = True
            chandra_repro(ids,'')
        print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Cleaning flares:",flares)
    if flares:
        for obsid in args.obsids:
            evtgti = os.path.join(root,"%s"%(obsid),"repro","%s_evt_gti.fits"%(obsid))
            if not os.path.exists(evtgti):
                event_file = os.path.join(root,"%i"%(int(obsid)),"repro","acisf%05i_repro_evt2.fits"%(int(obsid)) )
                repro = os.path.join(root,"%s"%(obsid),"repro")
                flare(event_file,obsid,repro)
        print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Filter ccd:")
    evt2lis = ccd_multi(args.obsids,root)
    # print("Event list",evt2lis)
    print("--------------------------------------------")
    print("Exposure map and Fluximage:",reproc)
    if flux:
        radec = ' '.join(str(x) for x in radec)
        if not os.path.exists(ctimg):
            merge_obs(evt2lis,merge+'/',refcoord=radec,binsize=binf,nproc=core,units="time",clobber=True)

        if not os.path.exists(fluxroot):
            os.makedirs(fluxroot)
        #
        # for emin,emax in bandkE:
        #     eband= str(emin)+':'+str(emax)+':'+str(round((emax+emin)/2,2))
        #     saida = os.path.join(fluxroot,'flux_')
        #     if not os.path.exists(os.path.join(fluxroot,'flux_{0}-{1}_thresh.img'.format(emin,emax))):
        #         merge_obs(evt2lis,saida,bands=eband,refcoord=radec,binsize=binf,nproc=core,units='time',clobber=True)
        print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Point sources detection:",ps)
    ctimg_mask = ctimg.split('.img')[0]+'_mask.img'
    if ps:
        if not os.path.exists(mask):
            ps_merge(args.obsids,'expweight',merge)
            mask_ps(ctimg,merge)
            dmcopy(ctimg+"[sky=mask(%s)]"%(mask),ctimg_mask, clobber=True)
            os.chdir(root)
        print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Blank Fields:",blk)
    blkevts = [os.path.join(merge,"%i_blank.evt"%(int(obsid))) for obsid in args.obsids]
    bgimg = os.path.join(merge,"blank_particle_bgnd.img")
    if blk:
        if not os.path.exists(bgimg):
            evt2vec = evt2lis.split(', ')
            blank_files(merge,args.obsids,evt2vec)


        bgimglist = [os.path.join(merge,"{0}_blank_particle_bgnd.img".format(obsid)) for obsid in args.obsids]
        bgimglist = ','.join(bgimglist)

        # if not os.path.exists(bgimg):
        enu = ['(img%i*img%i)'%(i,i+nobs) for i in range(1,nobs+1)]
        div = ['img%i'%(i+nobs) for i in range(1,nobs+1)]
        op = '('+'+'.join(enu)+')/('+'+'.join(div)+')'
        expfile = [os.path.join(dirname,obsid+'_broad_thresh.expmap') for obsid in obsids ]
        emaplist = ','.join(expfile)
        inlist = bgimglist+','+emaplist
        if not os.path.exists(bgimg):
            # dmimgcalc(infile=bgimglist,infile2='none',outfile=bgimg,op='imgout=%s'%(ope),clobber=True)
            dmimgcalc(infile=inlist,infile2='none',outfile=bgimg,op='imgout=%s'%(op),clobber=True)
        ## Producing blank sky images for each obsid and each energy band
        # for obsid in args.obsids:
            # blkevt = os.path.join(merge,"%i_blank.evt"%(int(obsid)))
            # for emin, emax in bandkE:
            #     outname = os.path.join(fluxroot,"bg_{0}_{1}-{2}".format(obsid,emin,emax))
            #     img_in = os.path.join(fluxroot,"flux_{0}_{1}-{2}_thresh.img".format(obsid,emin,emax))
            #     if not os.path.exists(outname):
            #         blank_image(blkevt,img_in,outname)

        # for emin, emax in bandkE:
        #     bgimglist = [os.path.join(fluxroot,"bg_{0}_{1}-{2}_particle_bgnd.img".format(id,emin,emax)) for id in args.obsids]
        #     emaplist = [os.path.join(fluxroot,"flux_{0}_{1}-{2}_thresh.expmap".format(id,emin,emax)) for id in args.obsids]
        #     bgimglist, emaplist = ','.join(bgimglist), ','.join(emaplist)
        #
        #     bgoutname = os.path.join(fluxroot,"bg_{0}-{1}_particle_bgnd.img".format(emin,emax))
        #     if not os.path.exists(bgoutname):
        #         dmimgcalc.punlearn()
        #         dmimgcalc(infile=bgimglist,infile2='none',outfile=bgoutname,op='imgout=%s'%(ope),clobber=True)
        #     # subprocess.run(['dmimgcalc','infile='+bgimglist,'infile2=none','outfile='+bgoutname,'op=imgout=%s'%(ope),'verbose=0','clob+'])

        print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Poisso noise & Smooth:",poisson)
    """ Given a image in counts units, it generates N x images, with random poissonianian pixels.
        output: <cluster_name>/noiseroot/(broad_N.img)
    """
    N = 20
    if poisson:
        if not os.path.exists(noiseroot):
            os.makedirs(noiseroot)
        for i in range(1,N+1):
            outnoise = os.path.join(noiseroot,"broad_%03i.img"%(i))
            if not os.path.exists(outnoise):
                noise(ctimg,outnoise,mask)
    print("--------------------------------------------")
    print("Estimating the center:",cen)
    dmcoords.punlearn()
    dmcoords(ctimg, asol="non", option="cel", ra=args.Xra, dec=args.Xdec, verbose=1)
    xpeak, ypeak = float(dmcoords.x), float(dmcoords.y)
    xc, yc = float(dmcoords.logicalx), float(dmcoords.logicaly) # in image coordinates
    xcen, ycen = xpeak, ypeak
    if cen:
        if not os.path.exists(os.path.join(merge,'xpeak.txt')):
            position = []
            position2 = []
            for i in range(1,N+1):
                cntnoisei = os.path.join(noiseroot,"broad_%03i.img"%(i))
                # res = center(cntnoisei,xpeak,ypeak,the500kpc/0.492,core)
                res = centroX(cntnoisei,xcen,ycen,the_R500/0.492,0)
                res2 = Xray_peak(cntnoisei,xpeak,ypeak,the500kpc/0.492,0.01*the500kpc/0.492)
                position.append(res)
                position2.append(res2)
            position = np.array(position);position2 = np.array(position2)
            xcen, ycen = np.mean(position[:,0]), np.mean(position[:,1])
            # xpeak, ypeak = np.mean(position2[:,0]), np.mean(position2[:,1])
            std_cen = (np.std(position[:,0])**2+np.std(position[:,1])**2)**(1/2)
            std_peak = (np.std(position2[:,0])**2+np.std(position2[:,1])**2)**(1/2)
            print("X-ray center:", xcen,ycen, " +/- ",std_cen)
            print("X-ray Peak:", xpeak,ypeak, " +/- ",std_peak)
            np.savetxt(os.path.join(merge,'center.txt'),position)
            np.savetxt(os.path.join(merge,'xpeak.txt'),position2)
        else:
            position = np.loadtxt(os.path.join(merge,'center.txt')); position2 = np.loadtxt(os.path.join(merge,'xpeak.txt'))
            xcen, ycen = np.mean(position[:,0]), np.mean(position[:,1])
            xpeak, ypeak = np.mean(position2[:,0]), np.mean(position2[:,1])
            std_cen = (np.std(position[:,0])**2+np.std(position[:,1])**2)**(1/2)
            std_peak = (np.std(position2[:,0])**2+np.std(position2[:,1])**2)**(1/2)
    print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Centroid Shift:",cshift)
    if cshift:
        if not os.path.exists(os.path.join(merge,'centroid_shfit.txt')):
            ## Excluindo região central dentro de 30kpc
            core = os.path.join(noiseroot,"core.reg")
            abertura(xcen,ycen,(3/40)*the500kpc/0.492,core)
            w = []
            ## Definindo cascas
            rt = the_R500/0.492
            print("r500:",rt)
            # if rt/rbkg < 1:
            #     rt = rw*rphy500
            for i in range(1,N+1):
                start_time = time.time()
                cntnoisei = os.path.join(noiseroot,"broad_%03i.img"%(i))
                res = centroid_shift(cntnoisei,xcen,ycen,rt)
                w.append(res)
                # print("%s.  <w> : (%.3f)1e-3"%((i),res))
                # print("Time: %s seconds" % (time.time() - start_time))
                wvalue = np.mean(np.array(w))
                werr = np.std(np.array(w))
                # wvalue, werr = 10, 1
                print("<w>, w_err : ( %.3f +/- %.3f )1e-3"%(wvalue, werr))
                np.savetxt(os.path.join(merge,'centroid_shfit.txt'),np.array(w))
        else:
            wvec = np.loadtxt(os.path.join(merge,'centroid_shfit.txt'))
            wvalue = np.mean(wvec)
            werr = np.std(wvec)
            print("<w>, w_err : ( %.3f +/- %.3f )1e-3"%(wvalue, werr))
            # wvalue, werr = 10, 1
    print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Estimating the source region:",src)
    """ It estimates the region were the cluster emission
        is indistiguishable of the bakcground.
        output: <cluster_name>/(source.reg, rbkg)
    """
    if 2*the500kpc < 6*60:
        rmax = 4*the500kpc/0.492   # 1Mpc in physical units
        print("1Mpc of Aperture")
    else:
        print("750kpc of Aperture")
        rmax = 2*the500kpc/0.492   # 750kpc in physical units
    if src:
        region = os.path.join(merge,"rmax.reg")
        rprof_file1 = os.path.join(merge,"broad_rprofile.fits")
        rprof_file = os.path.join(merge,"broad_rprofile_sn.fits")

        # bgimg = os.path.join(merge,"blank_particle_bgnd.img")
        anel(xcen,ycen,1,rmax,region)
        rprofile(ctimg_mask,bgimg,emap,region,rprof_file1)
        nbins = doSNR(rprof_file1,3,xcen,ycen,region.split('.reg')[0]+'_sn.reg')
        if nbins < 10:
            nbins = doSNR(rprof_file1,0.5,xcen,ycen,region.split('.reg')[0]+'_sn.reg')
        rprofile(ctimg_mask,bgimg,emap,region.split('.reg')[0]+'_sn.reg',rprof_file)

        rbkg, bgscale = rsource(rprof_file)
        abertura(xcen,ycen,rbkg,source)
        dmtcalc(rprof_file+'[R<%.1f]'%(rbkg),rprof_file.split('.fits')[0]+'_rmid.fits',expression='rmid=0.5*(r[0]+r[1])',clobber=True)
        print("Bg scale:",bgscale)
        plotSB2(rprof_file,rbkg,args.name)

    print("Tempo total: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Extracting radial profiles",rprof)
    """ It produces radial profiles
    """
    sn = 20
    if rprof:
        sufix = "_sn_%s_rebin.dat"%(sn)
        if not os.path.exists(profroot):
            os.makedirs(profroot)

        infgtempl = os.path.join(profroot,'total_fg_broad_profile.dat')
        inbgtempl = os.path.join(profroot,'total_bg_broad_profile.dat')
        print("rbkg:",rbkg)
        mbextract(ctimg, infgtempl, rbkg, xc, yc, binf, emap, mask)
        mbextract(bgimg, inbgtempl, rbkg, xc, yc, binf, emap, None)
        file_name_binned = infgtempl.split('.dat')[0]+sufix
        bgfile_name_binned = inbgtempl.split('.dat')[0]+sufix
        try:
            SNbinning(infgtempl, inbgtempl,sn,sufix)
        except:
            file_name_binned = infgtempl
            bgfile_name_binned = inbgtempl
        plotname = os.path.join(merge,"sb_profile.png")
        # plotSB(file_name_binned,bgfile_name_binned,bgscale,plotname)

        # for emin, emax in bandkE:
        #     infgtempl = os.path.join(profroot,'total_fg_%04i_%04i_profile.dat'%(1000*emin,1000*emax))
        #     inbgtempl = os.path.join(profroot,'total_bg_%04i_%04i_profile.dat'%(1000*emin,1000*emax))
        #
        #     ctimgi = os.path.join(fluxroot,"flux_{0}-{1}_thresh.img".format(emin,emax))
        #     bgimgi = os.path.join(fluxroot,"bg_{0}-{1}_particle_bgnd.img".format(emin,emax))
        #     emapi = os.path.join(fluxroot,"flux_{0}-{1}_thresh.expmap".format(emin,emax))
        #
        #     mbextract(ctimgi, infgtempl, rbkg, xc, yc, 1, emapi, mask)
        #     mbextract(bgimgi, inbgtempl, rbkg, xc, yc, 1, emapi, None)
        #     SNbinning(infgtempl, inbgtempl,sn,"_sn_%s_rebin.dat"%(sn))
        #     # subprocess.run(["python","/home/johnny/Documents/Brandeis/Xpipe/rebin_projsb.py","%s"%(infgtempl),"--backprof","%s"%(inbgtempl),"--suffix","_20_match.rebin"])
    print("Imaging check:",check)
    """ Create a png image with point sources and source region
        output: .png
    """
    jpgimg = os.path.join(root,'check',"%s_ps.jpg"%(args.name))
    srcimg = os.path.join(root,'check',"%s_source.jpg"%(args.name))
    simg = os.path.join(merge,"broad_smooth.img")
    simgn = simg.split('.img')[0]+'_final.img'
    if check:
        # if not os.path.exists(simg):
        # dmcopy(fimg+'[sky=region(%s)]'%(source),fimg.split('.img')[0]+'_core.img',clobber=True)
        dmimgadapt(fimg,simg,func='tophat',min=0.1, max=10, num=20, counts=30,clobber=True)
        csmooth(fimg,outfile=simgn,outsig=simg.split('.img')[0]+'sig.img', outscl=simg.split('.img')[0]+'scl.img',bkgmode='user', bkgmap=bgimg, sigmin=3, sigmax=5, sclmax=45, clobber=True)
    subprocess.run(['dmimg2jpg','infile=%s'%(simg),'outfile=%s'%(jpgimg),"lutfile=)ximage_lut.purple4","regionfile=mask(%s[opt full])"%(mask),"regioncolor=)colors.green","scalefun=log","mode=h",'clob+'])
    subprocess.run(['punlearn','dmimg2jpg'])

    subprocess.run(['dmimg2jpg','infile=%s'%(simgn),'outfile=%s'%(srcimg),"lutfile=)ximage_lut.purple4","regionfile=region(%s)"%(source),"regioncolor=)colors.green","scalefun=log","mode=h",'clob+'])
    print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("CSB and CSB4:",conce)
    """ It estimates the csb and csb_4 parameters
    """
    if conce:
        the_400kpc = (4/5)*the500kpc  # in arcsec
        csbr500 = csb(rprof_file,0.15*the_R500/0.492,the_R500/0.492)
        csb4 = csb(rprof_file,0.1*the_400kpc/0.492,the_400kpc/0.492)
        print("Csb, Csberr : %.3f"%(csbr500))
        print("Csb4, Csb4err : %.3f"%(csb4))

    plotname = os.path.join(root,'check',"%s_profile.png"%(args.name))
    # plotSB(file_name_binned,bgfile_name_binned,bgscale,plotname)

    print("--------------------------------------------")
    print("Save the output:",save)
    if save:
        outfile = os.path.join(merge,"log.txt")
        with open(outfile, 'w') as fout:
            print( '# name Xra Xdec errX w werr csb csb4 R500 redshift SN',file=fout )
            print(args.name, args.Xra, args.Xdec, std_peak*0.492*kpc_arcsec, wvalue, werr, csbr500, csb4, args.R500, args.redshift, bgscale, file=fout )
        print("The output are saved in: %s/log.txt"%(args.name))
    print("--------------------------------------------")
    print("ARF and RMF files:",arf)
    """ Given a list of event files, it computes the arf and rmf coadded
        In this code the event files the flares were cleaned for each observation
        output: <cluster_name>/spec/(merge.arf, merge.rmf, merge.pi)
    """
    if arf:
        s = [os.path.join(root,"%s"%(args.obsids[i]),"repro","%s_source.reg"%(int(args.obsids[i]))) for i in range(nobs)]
        repro = [os.path.join(root,"%s"%(args.obsids[i]),"repro","%s_evt_gti.fits"%(args.obsids[i])) for i in range(nobs)]
        for i in range(nobs):
            dmcoords(repro[i], asol="non", option="cel", ra=args.Xra, dec=args.Xdec, verbose=1)
            xobs, yobs = float(dmcoords.x), float(dmcoords.y)
            abertura(xobs,yobs,2*rbkg,s[i])

        evtgtilist = [os.path.join(root,"%s"%(args.obsids[i]),"repro","%s_evt_gti.fits[sky=region(%s)]"%(args.obsids[i],s[i])) for i in range(nobs)]
        evtgtilist = ','.join(evtgtilist)
        outroot = [os.path.join(specroot,"%s"%(obsid)) for obsid in args.obsids]
        outlist = ','.join(outroot)
        if not os.path.exists(specroot):
            os.makedirs(specroot)
        arf_rmf(evtgtilist,outlist)
        arflist = [os.path.join(specroot,"%s.pi"%(obsid)) for obsid in args.obsids]
        arflist = ",".join(arflist)
        # combine_spectra(src_spectra=arflist,outroot=os.path.join(specroot,"%s"%(args.name)),clobber=True)
    print("Tempo parcial: %s seconds" % (time.time() - t0))


if __name__ == '__main__':
    main()
