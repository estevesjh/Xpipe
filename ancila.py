#!/usr/bin/env python
from astropy.io import fits
from astropy.table import Table
import numpy as np
import math
import sys, os, subprocess
from cosmo import *
from region import *

# set_preference("window.display", "false")
# from pychips.all import *  # import chips hlui and advanced commands
from pychips.all import *
from pychips import *
from pycrates import *     # import crates i/o routines
from sherpa_contrib.all import *
from sherpa.astro.ui import *
from ciao_contrib.runtool import *

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def download(obsids):
    subprocess.run(['download_chandra_obsid',obsids])

def flare(evt,obsid,root):
    ''' Clean the flares of a given observation
    input: event-file, obsid and a root file
    output: root(default=<cluster_name>)
    /(obsid_evt_gti.fits, obsid_lc.fits, obsid_lc.gti)
    '''
    evtgti = os.path.join(root,"%s_evt_gti.fits"%(obsid))
    lc, lcgti = os.path.join(root,"%s_lc.fits"%(obsid)), os.path.join(root,"%s_lc.gti"%(obsid))
    dmcopy(evt+"[energy=300:12000]","bla.fits",clobber=True)

    dmextract.punlearn()
    dmextract("bla.fits[bin time=::259.28]",lc,opt="ltc1")

    deflare.punlearn()
    deflare(lc,lcgti,method="sigma")

    dmcopy(evt+'[@{0}]'.format(lcgti),evtgti)
    os.remove('bla.fits')

def ccd_single(obsid,root):
    """ Find the ccd chips used on the observation
    """
    ## -------------------------------
    ## Using just the ccd with an emission
    evtgti = os.path.join(root,"%s"%(obsid),"repro","%s_evt_gti.fits"%(obsid))
    ra0, dec0 = [float(dmkeypar(evtgti,'RA_PNT','echo+')),float(dmkeypar(evtgti,'DEC_PNT','echo+'))]
    dmcoords(evtgti, asol="non", option="cel", ra=ra0, dec=dec0, verbose=1)
    ccd = float(dmcoords.chip_id)

    if ccd<=3:
        ccd_lis = '0:3'
    else:
        bla = dmkeypar(evtgti,"DETNAM",'echo+')
        bla = bla.split('ACIS-')[1]
        if ccd != 4:
            ccd_lis = ','.join(str(int(ccd-1+i)) for i in range(3))
        else:
            ccd_lis = ','.join(str(int(ccd+i)) for i in range(3))
    print("%i - ccd:"%(int(obsid)),ccd_lis)
    return ccd_lis


def ccd_multi(obsids,root):
    ## -------------------------------
    ## Using just the ccd with an emission
    evt2lis = []
    for obsid in obsids:
        evtgti = os.path.join(root,"%i"%(int(obsid)),"repro","%i_evt_gti.fits"%(int(obsid)))
        ra0, dec0 = [float(dmkeypar(evtgti,'RA_PNT','echo+')),float(dmkeypar(evtgti,'DEC_PNT','echo+'))]
        dmcoords(evtgti, asol="non", option="cel", ra=ra0, dec=dec0, verbose=1)
        ccd = float(dmcoords.chip_id)

        if ccd<=3:
            ccd_lis = '0:3'
        else:
            bla = dmkeypar(evtgti,"DETNAM",'echo+')
            bla = bla.split('ACIS-')[1]
            if ccd != 4:
                ccd_lis = ','.join(str(int(ccd-1+i)) for i in range(3))
            else:
                ccd_lis = ','.join(str(int(ccd+i)) for i in range(3))
        print("%i - ccd:"%(int(obsid)),ccd_lis)

        bla = evtgti+"[ccd_id=%s]"%(ccd_lis)
        evt2lis.append(bla)

    return ', '.join(evt2lis)

def mask_ps(img,dirname):
    src=CXCRegion(os.path.join(dirname,"ps.reg"));(field()-src).write(os.path.join(dirname,"ps2.reg"),fits=True,clobber=True);
    dmimgcalc(img,'none','ones.fits',op="imgout=(1+(img1-img1))",clobber=True)
    dmcopy("ones.fits[sky=region(%s)][opt full]"%(os.path.join(dirname,"ps2.reg")),os.path.join(dirname,"ps_one.mask"),clobber=True)
    dmimgthresh(os.path.join(dirname,"ps_one.mask"),out=os.path.join(dirname,"no_ps_fov.mask"),exp="",cut=1,value=0,clobber=True)

    os.remove(os.path.join(dirname,"ps_one.mask"));os.remove("ones.fits")

def ps_single(imgfile,expfile,image_dir):
    os.chdir(image_dir)
    ## Creating the psf map
    psffile = imgfile.split('.img')[0]+'.psfmap'

    # if not os.path.exists(psffile):
    # mkpsfmap.punlearn()
    # mkpsfmap(imgfile,psffile,energy=2.3,spectrum="",ecf=0.9,units="arcsec",clobber=True)

    ## Running wavdetect
    wavdetect.punlearn()
    wavdetect.infile = imgfile
    wavdetect.expfile = expfile
    wavdetect.scales = "2 2.828 4 5.657 8 11.314 16"
    wavdetect.ellsigma = 2.5
    wavdetect.outfile = os.path.join(image_dir,"raw_ps.fits")
    wavdetect.interdir ="./"
    wavdetect.maxiter = 5
    wavdetect.sigthresh = 1e-6
    wavdetect.scellfile = imgfile.split('.img')[0]+'_cell.fits'
    wavdetect.imagefile = imgfile.split('.img')[0]+'_recon.fits'
    wavdetect.defnbkgfile = imgfile.split('.img')[0]+'_nbkg.fits'
    wavdetect.psffile = psffile

    wavdetect(clobber=True)
    # Do mask point source file
    dmcopy(os.path.join(image_dir,"raw_ps.fits")+"[PSFRATIO < 3, SRC_SIGNIFICANCE > 3]",os.path.join(image_dir,"ps.reg"),clobber="True")
    dmcopy(os.path.join(image_dir,"raw_ps.fits")+"[PSFRATIO > 3, SRC_SIGNIFICANCE > 3]",os.path.join(image_dir,"ext.reg"),clobber="True")

def ps_merge(obsids,method,dirname):
    os.chdir(dirname)
    ''' Find the point sources in a broad image merged
    input: broad_thresh.img','broad_thresh.expmap',
    method: expweight; min
    '''

    ## Creating the psf map
    imgfile = [os.path.join(dirname,obsid+'_broad_thresh.img') for obsid in obsids]
    expfile = [os.path.join(dirname,obsid+'_broad_thresh.expmap') for obsid in obsids ]
    psfmap(imgfile,expfile,method)

    merged_img = os.path.join(dirname,'broad_thresh.img')
    merged_emap = os.path.join(dirname,'broad_thresh.expmap')

    ## Running wavdetect
    wavdetect.punlearn()
    wavdetect.infile = merged_img
    wavdetect.expfile = merged_emap
    wavdetect.scales = "2 2.828 4 5.657 8 11.314 16"
    wavdetect.ellsigma = 2.5
    wavdetect.outfile = os.path.join(dirname,"raw_ps.fits")
    wavdetect.interdir ="./"
    wavdetect.maxiter = 5
    wavdetect.sigthresh = 1e-6
    wavdetect.scellfile = merged_img.split('.img')[0]+'_cell.fits'
    wavdetect.imagefile = merged_img.split('.img')[0]+'_recon.fits'
    wavdetect.defnbkgfile = merged_img.split('.img')[0]+'_nbkg.fits'
    wavdetect.psffile = os.path.join(dirname,'merged_'+method+'.psfmap')

    wavdetect(clobber=True)
    # Do the mask point source file
    # Do mask point source file
    dmcopy(os.path.join(dirname,"raw_ps.fits")+"[PSFRATIO < 1.5, SRC_SIGNIFICANCE > 3]",os.path.join(dirname,"ps.reg"),clobber="True")
    dmcopy(os.path.join(dirname,"raw_ps.fits")+"[PSFRATIO > 1.5, SRC_SIGNIFICANCE > 3]",os.path.join(dirname,"ext.reg"),clobber="True")

def psfmap(imgfilename_lis,emapfilname_lis,method):
    ''' Create a merged point source map
    input: img_lis and emap_lis should be a tupple of files paths
    methods: expweight, min; exposure map weighted psf maps and minimum PSF size '''

    nobs, dirname = len(imgfilename_lis), os.path.dirname(imgfilename_lis[0])
    emaplist = ','.join(emapfilname_lis)
    psflist = []

    # Producing psf maps for each observation in the broad band
    for i in range(nobs):
        mkpsfmap.punlearn()
        infile = imgfilename_lis[i]
        outfile = infile.split('_')[0]+'.psfmap'
        if not os.path.exists(outfile):
            mkpsfmap(infile,outfile,energy=2.3,spectrum="",ecf=0.9,units="arcsec",clobber=True)
        psflist.append(outfile)
    psflist = ','.join(psflist)

    if method == 'min':
        for i in range(nobs):
            name = psflist.split(',')[i]
            out = name.split('.psfmap')[0]+'_fov.psfmap'
            if not os.path.exists(out):
                dmimgthresh(name,out,expfile=emapfilname_lis[i],cut=1,value='INDEF',verbose=0)
        subprocess.run(['dmimgfilt',os.path.join(dirname,'*fov.psfmap'),os.path.join(dirname,'merged_min.psfmap'),'min','point(0,0)','verbose=0','clob+'])
    else:
        method = 'expweight'
        enu = ['(img%i*img%i)'%(i,i+nobs) for i in range(1,nobs+1)]
        div = ['img%i'%(i+nobs) for i in range(1,nobs+1)]
        op = '('+'+'.join(enu)+')/('+'+'.join(div)+')'
        if not os.path.exists(os.path.join(dirname,'merged_'+method+'.psfmap')):
            subprocess.run(['dmimgcalc','infile='+psflist+','+emaplist,'infile2=none','outfile='+os.path.join(dirname,'merged_'+method+'.psfmap'),'op=imgout=%s'%(op),'verbose=0','clob+'])
    subprocess.run(['dmhedit',os.path.join(dirname,'merged_'+method+'.psfmap'),'file=','op=add','key=BUNIT','value="arcsec"','verbose=0'])

def blank_files(merge,obsids,evtlist):
    blks = []
    # img = [os.path.join(merge,obsid+'_broad_thresh.img') for obsid in obsids]
    for i in range(len(obsids)):
        evt = evtlist[i]
        img = os.path.join(merge,obsids[i]+'_broad_thresh.img')
        blkevt = os.path.join(merge,"%i_blank.evt"%(int(obsids[i])))

        blank_field(evt,obsids[i],merge)

    blks = ','.join(blks)

def blank_field(evt,obsid,merge):
    blkevt = os.path.join(merge,"%i_blank.evt"%(int(obsid)))
    img = os.path.join(merge,"%i_broad_thresh.img"%(int(obsid)))

    if not os.path.exists(blkevt):
        blanksky.punlearn()
        blanksky.evtfile = evt
        blanksky.outfile = blkevt
        blanksky(verbose=0,clobber=True)

    blanksky_image.bkgfile = blkevt
    blanksky_image.outroot = blkevt.split('.evt')[0]
    blanksky_image.imgfile = img
    blanksky_image()


def blank_image(blkevt,img,outname):
    blanksky_image.bkgfile = blkevt
    blanksky_image.outroot = outname
    blanksky_image.imgfile = img
    blanksky_image()

## -------------------------------
## função centro de luminosidade
def abertura(x0,y0,rphy,outname):
    cmd  = "echo 'circle(%.3f,%.3f,%.2f)'"%(x0,y0,rphy)
    cmd += " > %s"%(outname)
    os.system(cmd)

def centroX(inimg,x0,y0,rphy,Core):
    dirname = os.path.dirname(inimg)
    region = os.path.join(dirname,"aper.reg")
    toto = os.path.join(dirname,"toto.fits")
    totog = os.path.join(dirname,"totog.fits")
    if Core==1:
        core = region.split("aper.reg")[0]+"core.reg"
        coreimg = os.path.join(dirname,"core.fits")
        dmcopy(inimg+"[exclude sky=region(%s)]"%(core),coreimg,clob=True)
        inimg = coreimg
    for i in range(2):
        x, y = x0, y0
        # Extaindo imagem dentro do círculo
        abertura(x,y,rphy,region)
        if Core==1:
            dmcopy(inimg+"[sky=region(%s)]"%(region),toto,clob=True)
            totog = inimg.split('.img')[0]+"_centroid_shift.img"
        else:
            dmcopy(inimg+"[sky=region(%s)]"%(region),toto,clob=True)
        aconvolve(toto,totog,'lib:gaus(2,3,3,5,5)',method="fft",clobber=True)
        # aconvolve(toto,simg,'lib:gaus(2,5,5,%.2f,%.2f)'%(rphy10kpc,rtotophy10kpc),method="fft",clobber=True)
        dmstat.punlearn()
        bla = dmstat(totog, centroid=True)
        pos = (dmstat.out_cntrd_phy)
        pos = pos.split(',')
        ## Definindo x e y
        x0, y0 = float(pos[0]), float(pos[1])
    return x0, y0

def centroid_shift(img,x0,y0,r500):
    ## Achando centroide
    centroid = []
    ri, rfim, dr = 0.15*r500,r500,0.05*r500
    rbin = np.arange(ri,rfim+dr,dr)
    # rbin = np.flip(rbin,axis=0)
    for i in range(len(rbin)):
        # xr, yr = center(img,x0,y0,rbin[i],1)
        # print("rbin:",rbin[i])
        xr, yr = centroX(img,x0,y0,rbin[i],1)
        centroid.append([xr,yr])
    centroid = np.array(centroid)
    offset = ((centroid[:,0]-x0)**2+(centroid[:,1]-y0)**2)**(1/2)
    ## Estimativa do centroid shift
    w = offset.std(ddof=1)/r500
    return w*1e3

def Xray_peak(inimg,x,y,rphy,rphy10kpc):
    dirname = os.path.dirname(inimg)
    region = os.path.join(dirname,"aper.reg")
    toto = os.path.join(dirname,"toto.fits")
    simg = inimg.split('.img')[0]+"_xpeak.img"
    abertura(x,y,rphy,region)
    dmcopy(inimg+"[sky=region(%s)]"%(region),toto,clob=True)
    if rphy10kpc < 3:
        rphy10kpc = 3
    # Fazendo um smooth 2D gaussiano com 5*10kpc em cada direção
    aconvolve(toto,simg,'lib:gaus(2,5,5,%.2f,%.2f)'%(rphy10kpc,rphy10kpc),method="fft",clobber=True)
    dmstat.punlearn()
    dmstat(simg,centroid=True)
    res = dmstat.out_max_loc.split(',')
    x,y = float(res[0]),float(res[1])
    return [x,y]

    return w*1e3

## -------------------------------
## Definindo perfil radial
def rprofile(img,bkg,expmap,region,bgregion,rprof_file):
    dmextract.punlearn()
    dmextract.infile = img+"[bin sky=@%s]"%(region)
    dmextract.outfile = "%s"%(rprof_file)
    dmextract.bkg = bkg+"[bin sky=@%s]"%(bgregion)
    dmextract.exp = expmap
    dmextract.bkgexp = expmap
    dmextract.opt = "generic"
    dmextract.clob = 'yes'
    dmextract()

def anel(x0,y0,r0,rphy,step,region):
    rbin = np.arange(r0,rphy,step)
    # length = int(round( np.log((rphy/r0)-1)/np.log(1.05),0))
    # rbin = [r0*(1.05)**(n-1) for n in range(1, length + 1)]
    # rbin = np.array(rbin)
    for i in range(len(rbin)-1):
        cmd  = "echo 'annulus(%.3f,%.3f,%.2f,%.2f)'"%(x0,y0,rbin[i],rbin[i+1])
        if i == 0:
            cmd += " > %s"%(region)
        else:
            cmd += " >> %s"%(region)
        os.system(cmd)

def rsource(file):
    dirname = os.path.dirname(file)
    rprof = read_file(file)
    # make_figure(infile+"[cols r,CEL_BRI]","histogram")
    r = copy_colvals(rprof,"R")
    y = copy_colvals(rprof,"CEL_BRI")
    dy = copy_colvals(rprof,"CEL_BRI_ERR")
    by = copy_colvals(rprof,"BG_CEL_BRI")
    # bdy = copy_colvals(rprof,"BG_CEL_BRI_ERR")
    x = 0.492 * 0.5*(r[:,0] + r[:,1])
    xmax = np.max(x)
    ## Definindo a escala do bg em relação ao blank field
    bgscale = np.sum(y)/np.sqrt(np.sum(by)+np.sum(y))
    print("SN: {0}".format(bgscale))
    ## Calculando S/N
    SN=y/(by)
    ## Descobrindo o raio do background
    thresh = 0.75
    if len(SN)>0:
        while len(x[SN<thresh])==0:
            thresh += 0.25
        mask = SN<thresh
        rbkg = x[mask][0]/0.492
    else:
        print("Error: The radial profile was not found, rbkg is 500kpc by default")
        rbkg = 3*60/0.492 # 6 arcmin
    print("Maximum Aperture: {0} arcsec".format(xmax))
    print("Background radius: {0} arcsec".format(rbkg*0.492))
    return rbkg, bgscale

def arf_rmf(evt_region,outroot):
    specextract.punlearn()
    specextract.infile = evt_region
    specextract.outroot = outroot
    specextract.grouptype = "NUM_CTS"
    specextract.binspec = 50
    specextract.combine = "no"
    specextract.clobber = 'yes'
    # print(evt_region)
    # print(specextract())
    specextract()
# def smooth(toto,totog,expmap):
#     dmimgcalc(infile=toto+","+expmap,infile2='none',outfile="bla.fits",op="imgout=img1/img2",clobber=True)
#     aconvolve("bla.fits",totog,'lib:gaus(2,10,10,5,5)',method="fft",clobber=True)
    # os.remove("bla.fits")
    # aconvolve(expmap,"bla.fits",'lib:gaus(2,5,5,3,3)',method="fft",clobber=True)
    # dmimgthresh("totog.fits","bla_thresh.fits",expfile="bla.fits",clobber=True)

def plotBeta(infile,pars,rbkg,name):
    dirname = os.path.dirname(infile)
    rprof = read_file(infile)
    # make_figure(infile+"[cols r,CEL_BRI]","histogram")
    r = copy_colvals(rprof,"R")
    y = copy_colvals(rprof,"CEL_BRI")
    dy = copy_colvals(rprof,"CEL_BRI_ERR")
    bgy = copy_colvals(rprof,"BG_CEL_BRI")
    # bdy = copy_colvals(rprof,"BG_CEL_BRI_ERR")
    x = 0.492*0.5*(r[:,0] + r[:,1])

    # Beta Model
    ym = pars[2] * (1 + (x/pars[0]/0.492)**2)**(0.5-3*pars[1])
    add_curve(x,ym,["symbol.style","none"])
    xr = np.append(x, x[::-1])
    yr = np.append(y+dy, (y-dy)[::-1])
    add_region(xr,yr,["fill.style","solid","fill.color","olive","depth",90])

    ## We take the second minimum and maximum value
    limits(Y_AXIS,0.9*np.min(y),1.1*np.max(yr))
    # limits(X_AXIS,np.min(x),np.max(x)+1)
    log_scale()

    bx = [0.1, 1000, 1000, 0.1]
    by = [0.95*np.min(y), 0.95*np.min(y), 1.05*np.min(y), 1.05*np.min(y)]
    add_region(bx,by,["fill.style","solid","fill.color","red","edge.style","noline","depth",80])
    # add_curve(x,bgy,["symbol.style","square","symbol.size",2])
    add_vline(rbkg*0.492)
    set_plot_xlabel("r (arcsec)")
    set_plot_ylabel("Surface brightness (count arcsec^{-2})")
    set_plot_title(name+r"   \chi^2_r = %.2f"%(pars[3]))
    set_plot(["title.size",20])
    add_label(0.1,0.45,r"f(r) = n (1 + (r/r_0)^2)^{0.5-3\beta}",["coordsys",PLOT_NORM,"size",18])
    add_label(0.1,0.35,"n = %.2f"%(pars[2]),["coordsys",PLOT_NORM])
    add_label(0.1,0.3,"r_0 = %.1f"%(pars[0]),["coordsys",PLOT_NORM])
    add_label(0.1,0.25,r"\beta = %.3f"%(pars[1]),["coordsys",PLOT_NORM])

    opts = { "clobber": True, "fittopage": True }
    opts['pagesize'] = 'letter'
    print_window(os.path.join(dirname,"%s.pdf"%(name)),opts)
    clear_plot()
def doGRP(infile, Ncounts, x0, y0, region):
    """Make groups of Ncounts in each bin
    Make a region file.
    """
    dirname = os.path.dirname(infile)
    rprof = read_file(infile)
    r = copy_colvals(rprof,"R")
    cnt = copy_colvals(rprof,"NET_COUNTS")
    dr = r[:,1]-r[:,0]

    nbins = len(cnt)
    outradii = []
    lastidx = 0
    for i in range(1, nbins+1):
        totfgcts = cnt[lastidx:i].sum()
        Csum = totfgcts
        if Csum >= Ncounts or i == nbins:
            outradii.append(i)
            lastidx = i
    print('Produced', len(outradii), '# counts:', Ncounts,'vec:',outradii)

    lastidx = 0
    with open(region, 'w') as fout:
        for i in outradii:
            inrad = r[lastidx,0]
            outrad = r[i-1,1]+dr[lastidx]
            print('annulus(%.3f,%.3f,%.2f,%.2f)'%(x0,y0,inrad,outrad),file=fout )
            lastidx = i
    return len(outradii)

def doSNR(infile, snthresh, x0, y0, region):
    """Bin up to signal noise given and splits its in nbins

    Returns outer bin number for each bin
    """
    dirname = os.path.dirname(infile)
    rprof = read_file(infile)
    # make_figure(infile+"[cols r,CEL_BRI]","histogram")
    r = copy_colvals(rprof,"R")
    dr = r[:,1]-r[:,0]
    cnt = copy_colvals(rprof,"CEL_BRI")
    bg_cnt = copy_colvals(rprof,"BG_CEL_BRI")

    nbins = len(cnt)
    outradii = []
    lastidx = 0
    for i in range(1, nbins+1):
        totfgcts = cnt[lastidx:i].sum()
        totbgcts = bg_cnt[lastidx:i].sum()
        signal = totfgcts
        noise = math.sqrt(totfgcts+totbgcts)
        sn = signal / noise
        if sn >= snthresh or i == nbins:
            outradii.append(i)
            lastidx = i

    print('Produced', len(outradii), 'sn:', snthresh,'vec:',outradii)

    lastidx = 0
    with open(region, 'w') as fout:
        for i in outradii:
            inrad = r[lastidx,0]
            outrad = r[i-1,1]+dr[lastidx]
            print('annulus(%.3f,%.3f,%.2f,%.2f)'%(x0,y0,inrad,outrad),file=fout )
            lastidx = i
    return len(outradii)

def plotSB(infilename,bgfilename,bgscale,outname):
    data = np.loadtxt(infilename)
    radii = data[:,0]
    widths = data[:,1]
    totcts = data[:,2]
    sb = data[:,5]

    bgdata = np.loadtxt(bgfilename)
    bg_radii = bgdata[:,0]
    bg_widths = bgdata[:,1]
    bg_sb = bgscale*bgdata[:,5]

    figsize_x = 10. #[inches]
    figsize_y = 10. #[inches]
    zeropoint = 25.96

    #creating the figure with the surface brightness profile
    fig = plt.figure(figsize=(figsize_x,figsize_y))
    ax  = plt.subplot() #nothing inside because it is the only plot
    ax.set_xlim(0.01,np.max(radii)*1.2)
    # ax.set_ylim(1e-1,1e-9)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    # ax.scatter(radii,sb, marker =".", color = "c")
    # ax.scatter(radii,bg_sb, marker =".", color = "r")
    plt.errorbar(radii,sb, xerr= widths, linestyle = "None", ecolor = "k")
    plt.errorbar(bg_radii,bg_sb, xerr= bg_widths, linestyle = "None", ecolor = "r")
    plt.xlabel('Radius [arcsec]')
    plt.ylabel('Surface brightness [counts/s/arcmin2]')
    plt.grid()
    # ax2 = ax.twiny() #they share the same y-axis
    # ax2.plot(distance_in_arcsec*kpc_per_arcsec.value,surf_bright, alpha = 0)
    # plt.xlabel('Radius [arcmin]')
    plt.savefig(outname)
    # plt.close()
    plt.clf()

def csb(infile,rin,rout):
    dirname = os.path.dirname(infile)
    rprof = read_file(infile)
    # make_figure(infile+"[cols r,CEL_BRI]","histogram")
    r = copy_colvals(rprof,"R")
    y = copy_colvals(rprof,"CEL_BRI")
    dy = copy_colvals(rprof,"CEL_BRI_ERR")
    x = 0.5*(r[:,0] + r[:,1])
    rmax = np.max(r[:,1])
    ### Inner circle
    mask = (x <= rin)
    sb_inner = y[mask]
    if rout/rmax > 1:
        print("R500 is larger than the observation")
        rout = rmax
    ### outer circle
    mask = (x <= rout)
    sb_out = y[mask]
    return (np.sum(sb_inner)/np.sum(sb_out))

def csb2(fgprof,bgprof,rin,rout):
    ''' rin, rout in arcmin
    '''
    data = np.loadtxt(fgprof)
    radii = data[:,0]
    area = data[:,3]
    sb = data[:,5]
    bgdata = np.loadtxt(bgprof)
    bg_radii = bgdata[:,0]
    bg_area = bgdata[:,3]
    bg_sb = bgdata[:,5]

    rmax = np.max(radii)
    net_sb = sb-bg_sb
    ### Inner circle
    mask = (radii <= rin)
    sb_inner = net_sb[mask]

    if rout/rmax < 1:
        print("R500 is larger than the observation")
        rout = rmax
    ### outer circle
    mask = (radii <= rout)
    sb_out = net_sb[mask]

    return (np.sum(sb_inner)/np.sum(sb_out))

def fitBetaM(table):
    dirname = os.path.dirname(table)
    load_data(1,table, 3, ["RMID","CEL_BRI","CEL_BRI_ERR"])
    pars = ["rc","rs","alpha","beta","epsilon","n0","gamma"]

    load_user_model(Eprojected,"mybeta")
    par0 = [14.0947,0.492*239.625,3,1,1.17144,0.0975581,3.0]
    frzpar=6*[False]+[True]

    add_user_pars("mybeta", pars, par0, parfrozen=frzpar)
    set_model("mybeta")

    mybeta.n0.min = 0
    mybeta.rc.min = 0
    mybeta.rs.min = 0
    mybeta.alpha.min = 0
    mybeta.beta.min = 0
    mybeta.epsilon.min = 0
    # mybeta.alpha.max = 1e2
    # mybeta.beta.max = 1e2
    mybeta.epsilon.max = 4.999
    # mybeta.n0.min = 0
    # freeze(mybeta.alpha)
    show_model()
    fit()
    fitr = get_fit_results()
    out = np.array(fitr.parvals)
    ## apend gamma
    out = np.append(out,3)

    set_xlog()
    set_ylog()
    plot_fit()
    opts = { "clobber": True}
    print_window(os.path.join(dirname,"sb_betaM.png"),opts)
    clear_plot()
    return out
def fitBeta(table):
    dirname = os.path.dirname(table)
    load_data(1,table, 3, ["RMID","CEL_BRI","CEL_BRI_ERR"])
    set_source("beta1d.sbr1")
    sbr1.r0 = 105
    sbr1.beta = 4
    sbr1.ampl = 0.00993448
    freeze(sbr1.xpos)
    fit()
    plot_fit()
    log_scale(XY_AXIS)
    fitr = get_fit_results()
    out = np.array(fitr.parvals)
    out = np.append(out,fitr.rstat)
    print(len(out))
    covar()
    ## apend gamma
    # out = np.append(out,3)
    opts = { "clobber": True}
    print_window(os.path.join(dirname,"sb_beta.png"),opts)
    clear_plot()
    return out
def plotSB3(infile,rbkg,pars,name):
    clear_plot()
    dirname = os.path.dirname(infile)
    rprof = read_file(infile)
    # make_figure(infile+"[cols r,CEL_BRI]","histogram")
    r = copy_colvals(rprof,"R")
    y = copy_colvals(rprof,"CEL_BRI")
    dy = copy_colvals(rprof,"CEL_BRI_ERR")
    bgy = copy_colvals(rprof,"BG_CEL_BRI")
    # bdy = copy_colvals(rprof,"BG_CEL_BRI_ERR")
    x = 0.492 * 0.5*(r[:,0] + r[:,1])
    # Beta Model
    epars = [pars[i] for i in range(len(pars))]
    ym = np.array(Eprojected(pars,x/0.492))
    add_curve(x,ym,["symbol.style","none","symbol.size",2])
    xr = np.append(x, x[::-1])
    yr = np.append(y+dy, (y-dy)[::-1])
    add_region(xr,yr,["fill.style","solid","fill.color","olive","depth",90])

    ## We take the second minimum and maximum value
    limits(Y_AXIS,0.9*np.min(y),1.1*np.max(yr))
    # limits(X_AXIS,np.min(x),np.max(x)+1)
    log_scale()

    bx = [0.1, 1000, 1000, 0.1]
    by = [0.95*np.min(y), 0.95*np.min(y), 1.05*np.min(y), 1.05*np.min(y)]
    add_region(bx,by,["fill.style","solid","fill.color","red","edge.style","noline","depth",80])
    # add_curve(x,bgy,["symbol.style","square","symbol.size",2])
    add_vline(rbkg*0.492)
    set_plot_xlabel("r (arcsec)")
    set_plot_ylabel("Surface brightness (count arcsec^{-2})")
    set_plot_title(r"\chi^2_r = 1.04")
    set_plot(["title.size",20])
    opts = { "clobber": True, "fittopage": True }
    # opts['pagesize'] = 'letter'
    print_window(os.path.join(dirname,"%s.png"%(name)),opts)
