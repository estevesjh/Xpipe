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
    parser.add_argument('obsid',
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

    binf = 2
    core = 8
    sn = 10
    radec = [args.Xra,args.Xdec]
    ## -------------------------------
    ## Flags
    down=1;reproc=1;flares=1;
    flux=1;ps=1;cen=1;blk=1;
    src=1;arf=1;rprof=1;poisson=1;
    check=0;cshift=0;conce=1;
    save=1;fit=0;

    t0 = time.time()

    if (args.R500)<0:
        cen=0;cshift=0;conce=0
        print("R500 is not valuable number")
    ## -------------------------------
    ## Definindo distâncias e raios

    DA = distance(float(args.redshift))     # em kpc
    themax = 300                            # 5' arcmin de abertura
    kpc_arcsec = (1/(3600*180/np.pi))*DA    # kpc/arcsec
    the500kpc = 500/kpc_arcsec              # 500kpc in physical units
    the_R500 = args.R500/kpc_arcsec         # R500 in arcsec
    obsid = str(args.obsid)
    ## -------------------------------
    if 2*the500kpc < 2*60:
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
    evt2repro = os.path.join(root,"%i"%(int(obsid)),"repro","acisf%05i_repro_evt2.fits"%(int(obsid)))
    ctimg = os.path.join(merge,'%s_broad_thresh.img'%(obsid))
    sbimg = os.path.join(merge,'sb_{0}-{1}_thresh.img'.format(0.7,2))

    emap = os.path.join(merge,'%s_broad_thresh.expmap'%(obsid))
    fimg = os.path.join(merge,'%s_broad_flux.img'%(obsid))
    mask = os.path.join(merge,"no_ps_fov.mask")
    psreg = os.path.join(merge,"ps.reg")
    source = os.path.join(merge,"source.reg")

    print("--------------------------------------------")
    print("Cluster :",args.name,"at redshfit ",args.redshift)
    print("Chandra observation:",args.obsid)
    print("--------------------------------------------")
    if down:
        print("Download data:",down)
        if not os.path.exists(os.path.join(root,'%s'%(args.obsid))):
            download(obsid)
            print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Reprocess data to lvl.2:",reproc)
    if reproc:
        if not os.path.exists(evt2repro):
            chandra_repro.check_vf_pha = 'yes'
            chandra_repro.set_ardlib = 'no'
            chandra_repro(args.obsid,'')
            print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Cleaning flares:",flares)
    if flares:
        evtgti = os.path.join(root,"%s"%(obsid),"repro","%s_evt_gti.fits"%(obsid))
        if not os.path.exists(evtgti):
            repro = os.path.join(root,"%s"%(obsid),"repro")
            flare(evt2repro,obsid,repro)
        print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Filter ccd:")
    ccdlis = ccd_single(obsid,root)
    evtgticcd = evtgti+'[ccd_id=%s]'%(ccdlis)
    print("--------------------------------------------")
    print("Exposure map and Fluximage:",reproc)
    if flux:
        if not os.path.exists(ctimg):
            fluximage(evtgticcd,merge+'/%s'%(obsid),binsize=binf,units='time')
        saida = os.path.join(merge,'sb_')
        fluximage(evtgticcd,saida,binsize=binf,band='0.7:2:1.5')
        print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Point sources detection:",ps)
    ctimg_mask = ctimg.split('.img')[0]+'_mask.img'
    sbimg_mask = sbimg.split('.img')[0]+'_mask.img'
    evt_mask = evtgti.split('.fits')[0]+'_bin_mask.fits'
    if ps:
        # if not os.path.exists(mask):
        ps_single(ctimg,emap,merge)
        mask_ps(ctimg,merge)
        dmcopy(ctimg+"[sky=mask(%s)]"%(mask),ctimg_mask, clobber=True)
        os.chdir(root)
        print("Tempo parcial: %s seconds" % (time.time() - t0))
        dmcopy(sbimg+"[exclude sky=region(%s)]"%(psreg),sbimg_mask,clobber=True)
        dmcopy(evtgti+'[exclude sky=region(%s)]'%(psreg),evt_mask,clobber=True)
    print("--------------------------------------------")
    print("Blank Fields:",blk)
    bgimg = os.path.join(merge,"%s_blank_particle_bgnd.img"%(obsid))
    blkevt = os.path.join(merge,"%s_blank.evt"%(obsid))
    if blk:
        if not os.path.exists(bgimg):
            blank_field(evtgticcd,obsid,merge)
        print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("Poisso noise & Smooth:",poisson)
    """ Given a image in counts units, it generates N x images, with random poissonianian pixels.
        output: <cluster_name>/noiseroot/(broad_N.img)
    """
    N = 2
    if poisson:
        if not os.path.exists(noiseroot):
            os.makedirs(noiseroot)
        for i in range(1,N+1):
            outnoise = os.path.join(noiseroot,"broad_%03i.img"%(i))
            if not os.path.exists(outnoise):
                noise(ctimg,outnoise,mask)
    print("--------------------------------------------")
    print("Center:",cen)
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
            xpeak, ypeak = np.mean(position2[:,0]), np.mean(position2[:,1])
            std_cen = (np.std(position[:,0])**2+np.std(position[:,1])**2)**(1/2)
            std_peak = (np.std(position2[:,0])**2+np.std(position2[:,1])**2)**(1/2)
            print("X-ray center:", xcen,ycen, " +/- ",std_cen)
            print("X-ray Peak:", xpeak,ypeak, " +/- ",std_peak)
            np.savetxt(os.path.join(merge,'center.txt'),position)
            np.savetxt(os.path.join(merge,'xpeak.txt'),position2)
        else:
            position = np.loadtxt(os.path.join(merge,'center.txt')); position2 = np.loadtxt(os.path.join(merge,'xpeak.txt'))
            xcen, ycen = np.mean(position[:,0]), np.mean(position[:,1])
            # xpeak, ypeak = np.mean(position2[:,0]), np.mean(position2[:,1])
            std_cen = (np.std(position[:,0])**2+np.std(position[:,1])**2)**(1/2)
            std_peak = (np.std(position2[:,0])**2+np.std(position2[:,1])**2)**(1/2)
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
    if src:
        region = os.path.join(merge,"rmax.reg")
        rprof_file1 = os.path.join(merge,"broad_rprofile.fits")
        rprof_file = os.path.join(merge,"broad_rprofile_grp.fits")
        rprof = rprof_file.split('.fits')[0]+'_rmid.fits'
        grpregion = region.split('.reg')[0]+'_grp.reg'
        bgregion = os.path.join(merge,"bkg.reg")
        dt = 1

        # The Radial Profile with Blank Fields
        if not os.path.exists(rprof_file1):
            anel(xcen,ycen,10,rmax,dt,region)
            rprofile(sbimg_mask,bgimg,emap,region,region,rprof_file1)
            dmtcalc(rprof_file1,rprof_file1.split('.fits')[0]+'_rmid.fits',expression='rmid=0.5*(r[0]+r[1])',clobber=True)
            nbins = doGRP(rprof_file1,200,xcen,ycen,grpregion)
        # anel(xcen,ycen,rbkg,rbkg+2*60/0.492+1,2*60/0.492-1,bgregion)
        # anel(xcen,ycen,rbkg,rbkg+60/0.492+1,10,bgregion)
            rprofile(sbimg_mask,bgimg,emap,grpregion,grpregion,rprof_file)
            dmtcalc(rprof_file,rprof_file.split('.fits')[0]+'_rmid.fits',expression='rmid=0.5*(r[0]+r[1])',clobber=True)
        rbkg, bgscale = rsource(rprof_file)
        abertura(xcen,ycen,rbkg,source)
        dmtcalc(rprof_file+'[R<%.1f]'%(rbkg),rprof,expression='rmid=0.5*(r[0]+r[1])',clobber=True)
    print("Tempo total: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Imaging check:",check)
    """ Create a png image with point sources and source region
        output: .png
    """
    jpgimg = os.path.join(root,'check',"%s_ps.jpg"%(args.name))
    srcimg = os.path.join(merge,"%s_source.jpg"%(args.name))
    simg = os.path.join(merge,"broad_smooth.img")
    simgn = simg.split('.img')[0]+'_final.img'
    if check:
        if not os.path.exists(simg):
        # dmcopy(fimg+'[sky=region(%s)]'%(source),fimg.split('.img')[0]+'_core.img',clobber=True)
            dmimgadapt(fimg,simg,func='tophat',min=0.1, max=10, num=20, counts=30,clobber=True)
        # csmooth(infile=fimg,outfile=simgn,outsig=simg.split('.img')[0]+'_sig.img', outscl=simg.split('.img')[0]+'_scl.img',bkgmode='user', bkgmap=bgimg, sigmin=3, sigmax=5, sclmax=45, clobber=True)
    subprocess.run(['dmimg2jpg','infile=%s'%(simg),'outfile=%s'%(jpgimg),"lutfile=)ximage_lut.purple4","regionfile=mask(%s[opt full])"%(mask),"regioncolor=)colors.green","scalefun=log","mode=h",'clob+'])
    subprocess.run(['dmimg2jpg','infile=%s'%(simg),'outfile=%s'%(srcimg),"lutfile=)ximage_lut.purple4","regionfile=region(%s)"%(bgregion),"regioncolor=)colors.green","scalefun=log","mode=h",'clob+'])
    print("--------------------------------------------")
    print("Tempo parcial: %s seconds" % (time.time() - t0))

    print("--------------------------------------------")
    print("CSB and CSB4:",conce)
    """ It estimates the csb and csb_4 parameters
    """
    if conce:
        the_400kpc = (4/5)*the500kpc # in arcsec
        csbr500 = csb(rprof_file,0.15*the_R500/0.492,the_R500/0.492)
        csb4 = csb(rprof_file,0.1*the_400kpc/0.492,the_400kpc/0.492)
        print("Csb, Csberr : %.3f"%(csbr500))
        print("Csb4, Csb4err : %.3f"%(csb4))

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
        region_arf = os.path.join(merge,"source_arf.reg")
        arf_out = os.path.join(specroot,"%s"%(obsid))
        dmcoords(evt_mask, asol="non", option="cel", ra=args.Xra, dec=args.Xdec, verbose=1)
        xobs, yobs = float(dmcoords.x), float(dmcoords.y)
        abertura(xobs,yobs,rbkg,region_arf)
        if not os.path.exists(specroot):
            os.makedirs(specroot)
        arf_rmf(evt_mask+"[sky=region(%s)]"%(region_arf),arf_out)
        evtin = evt_mask+"[sky=region(%s)]"%(region_arf)
        bgin = blkevt+"[sky=region(%s)]"%(region_arf)
        subprocess.run(['specextract','%s'%(evtin),'%s'%(arf_out),'bkgfile=%s'%(bgin),'bkgresp=no','mode=h','verbose=1'])

    print("Tempo parcial: %s seconds" % (time.time() - t0))
    print("--------------------------------------------")
    print("Fit a Surface Brightness Profile:",arf)
    """
    """
    if fit:
        betampars = fitBetaM(rprof)
        betapars = fitBeta(rprof)
        # print("Betas pars: n0:",betapars[2],'r0:',betapars[0],'beta:',betapars[1],r'\chi^2',betapars[3])
        plotBeta(rprof,betapars,rbkg,args.name)
        # plotBetaM(rprof,rbkg,betampars,args.name)
        print("Tempo parcial: %s seconds" % (time.time() - t0))
if __name__ == '__main__':
    main()
