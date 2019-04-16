# !/usr/bin/env python
# tasks: mask point sources, centerX, radial profile and ARF-RMF files
# Obs.: don't forget to activate the ciao enviroment!

from astropy.io.fits import getdata
from astropy.table import Table
import astropy.io.ascii as at

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from ciao_contrib.runtool import *
import pycrates                     # import crates ciao routines

import sys, os
import os.path as path
import logging
import time
import subprocess
import numpy as np

#--- cosmologia
h = 0.7
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3)

DEG2RAD=np.pi/180.0

# Funções básicas
def AngularDistance(z):
    DA = float( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA

def kpcToPhy(radius,z,ARCSEC2PHYSICAL=0.492):
    DA = AngularDistance(z)
    radius = radius/1000 # Mpc
    res = (3600*(radius/DA)/DEG2RAD)/ARCSEC2PHYSICAL
    return res


def writeStringToFile(fileName, toBeWritten):
    # create file if it does not exist
    if not os.path.isfile(fileName):
        with open(fileName, 'w') as f:
            f.write( '{toBeWritten}\n'.format(toBeWritten=toBeWritten) )
    # else, append to file
    else:
        with open(fileName, 'a') as f:
            f.write( '{toBeWritten}\n'.format(toBeWritten=toBeWritten) )

def saveOutput(name,value,section=None,out='output.txt'):
    '''Save an ouptut name value into a section in the output file
    '''
    toBeWritten = '{item}: {value}'.format(item=name,value=value)

    # create file if it does not exist
    if not os.path.isfile(out):
        with open(out, 'w') as f:
            f.write('#general ouput of the Xpipe \n')
            if section is not None:
                f.write('[%s]\n'%(section))
            f.write(toBeWritten+'\n')
    # else, append to file
    else:
        with open(out, 'a') as f:
            if section is not None:
                f.write('[%s]\n'%(section))
            writeStringToFile(out,toBeWritten)

def doGRP(infile, Ncounts, x0, y0, region):
    """Make groups of Ncounts in each bin
    Make a region file.
    """

    rprof = pycrates.read_file(infile)
    r = pycrates.copy_colvals(rprof,"R")
    cnt = pycrates.copy_colvals(rprof,"COUNTS")
    dr = r[:,1]-r[:,0]

    nbins = len(cnt)
    outradii = []
    lastidx = 0

    inrad, outrad = np.empty((0,0)),np.empty((0,0))
    for i in range(1, nbins):
        totfgcts = cnt[lastidx:i].sum()
        Csum = totfgcts
        if Csum >= Ncounts or i == nbins:
            inrad, outrad = np.append(inrad,r[lastidx,0]),np.append(outrad,r[i,1])
            outradii.append(i)
            lastidx = i
    
    logging.debug('doGRP(): threshold = {} cnts; Produced: {} bins'.format(Ncounts,len(outradii)) )

    lastidx = 0
    with open(region, 'w') as fout:
        for i in range(1,len(outrad)):
            # inrad = r[lastidx,0]
            # outrad = r[i-1,1]+dr[lastidx]
            print('annulus(%.3f,%.3f,%.2f,%.2f)'%(x0,y0,inrad[i],outrad[i]),file=fout )
            # lastidx = i
    return len(outradii)

## -------------------------------
## auxiliary functions

def findXrayPeak(inimg,x,y,rphy,rSigma=10):

    dirname = os.path.dirname(inimg)
    region = os.path.join(dirname,"aper.reg")
    toto = os.path.join(dirname,"toto.fits")

    simg = inimg.split('.img')[0]+"_xpeak.img"
    abertura(x,y,rphy,region)
    
    dmcopy(inimg+"[sky=region(%s)]"%(region),toto,clob=True)
    if rSigma < 3:
        rSigma = 3

    # Fazendo um smooth 2D gaussiano com 5*10kpc em cada direção
    aconvolve(toto,simg,'lib:gaus(2,5,1,%.2f,%.2f)'%(rSigma,rSigma),method="fft",clobber=True)
    
    dmstat.punlearn()
    dmstat(simg,centroid=True,clip='yes')
    res = dmstat.out_max_loc.split(',')
    
    x,y = float(res[0]),float(res[1])
    os.remove(toto); os.remove(region)
    return [x,y]

def findCentroX(inimg,x0,y0,rphy):
    dirname = path.dirname(inimg)
    region = path.join(dirname,"aper.reg")
    toto = path.join(dirname,"toto.fits"); totog = path.join(dirname,"totog.fits")
    
    count = 0; conv = 100
    
    while conv > 1:
        x, y = x0, y0
        # Extaindo imagem dentro do círculo
        abertura(x,y,rphy,region)
        dmcopy(inimg+"[sky=region(%s)]"%(region),toto,clob=True)
        aconvolve(toto,totog,'lib:gaus(2,5,1,10,10)',method="fft",clobber=True)
        # aconvolve(toto,simg,'lib:gaus(2,5,5,%.2f,%.2f)'%(rphy10kpc,rtotophy10kpc),method="fft",clobber=True)
        dmstat.punlearn()
        bla = dmstat(totog, centroid=True)
        pos = (dmstat.out_cntrd_phy)
        pos = pos.split(',')
        ## Definindo x e y
        x0, y0 = float(pos[0]), float(pos[1])
        conv = ((x-x0)**2+(y-y0)**2)**(1/2)
        # print("The aperture is x=%.3f  y=%.3f and the convergence=%.2f "%(x0,y0,conv))
        count+=1
        if count>20:
            break
            print("The convergence was not reached")
    os.remove(toto); os.remove(totog); os.remove(region)

    return [x0, y0]

def rprofile(img,bkg,expmap,region,out='rprof.fits',bgregion=None):
    if bgregion is None:
        bgregion = region
    dmextract.punlearn()
    dmextract.infile = img+"[bin sky=@%s]"%(region)
    dmextract.outfile = "%s"%(out)
    dmextract.bkg = bkg+"[bin sky=@%s]"%(bgregion)
    dmextract.exp = expmap
    dmextract.bkgexp = expmap
    dmextract.opt = "generic"
    dmextract.clob = 'yes'
    dmextract()
    
    dmtcalc(out,out,expression='rmid=0.5*(r[0]+r[1])',clobber=True)

def getObsid(obsid):
    lis = obsid.split(',')
    res = [int(ij) for ij in lis]
    return res

def checkDirs(dirList):
    '''Check if a list o files exists
    '''
    idx = np.empty(0,dtype=int)
    for i in range(len(dirList)):
        Dir = dirList[i]
        if path.exists(Dir):
            idx = np.append(i,idx)
    return idx

def checkObsid(obsid):
    '''It checks the obsid variable type.
       It returns in two differnt types, list and string.
    '''
    if isinstance(obsid,str):
        res_lis = getObsid(obsid)
        return obsid,res_lis

    elif isinstance(obsid,list):
        res_str = ','.join(obsid)
        return res_str,obsid

    elif isinstance(obsid,int):
        res_str = str(obsid)
        res_lis = [res_str]
        return res_str,res_lis
    else:
        logging.error('Chandra obsid={} format is not valid! Please try the following formats: int, str or list'.format(obsid))
        pass
        
def checkEvtFiles(obsid_lis,pathData='./'):
    '''Check if the event files were created.
    '''
    event_file_lis = [ path.join(pathData,"%i"%(obsid),"repro","acisf%05i_repro_evt2.fits"%(obsid)) for obsid in obsid_lis]
    idx = checkDirs(event_file_lis)
    if idx.size < len(obsid_lis):
        logging.critical('repro files was not created. Please remove the files {} and run the code again.'.format(','.join(obsid_lis)))
        exit()
    else:
        pass
def checkImg(img):
    if not path.isfile(img):
        logging.critical('Image file was not found:{}.'.format(img))
        exit()
    else:
        pass

def getDir(name,path):
    nameDir = os.path.join(path,name)
    if not os.path.exists(nameDir):
        os.makedirs(nameDir)
    return nameDir

def makeProfile(img, xc, yc, rmax, binf):
    """Take image and compute profile. Ignores nan values.

    Returns: array of total values in bins, array of number of pixels in bins

    xc, yc: centre in pixels
    rmax: maximum radius in pixels
    binf: binning factor
    """
    radii = np.fromfunction(
    lambda y, x: (np.sqrt((x-xc)**2 + (y-yc)**2)*binf).astype(np.int32),
    img.shape)

    rmaxdiv = int(rmax / binf)
    radii[ np.logical_not(np.isfinite(img)) | (radii >= rmaxdiv) ] = rmaxdiv

    numpix = np.bincount(radii.ravel())
    ctsums = np.bincount(radii.ravel(), weights=img.ravel())

    return ctsums[:rmaxdiv], numpix[:rmaxdiv]

def getRmax(xc,yc,img,emap,outdir,binf=1,aperMax=600):
    ones = path.join(outdir,'ones.fits')
    ## Define files
    dmimgcalc(img,'none',ones,op="imgout=(1+(img1-img1))",clobber=True)
    dmimgthresh(ones,out=ones,exp=emap,cut=1,value=0,clobber=True)

    import astropy.io.fits as fits
    with fits.open(ones) as f:
        img_array = np.array(f[0].data)

        ## do a radius image vector
        radii = np.fromfunction(lambda y, x: (np.sqrt((x-xc)**2 + (y-yc)**2)*binf).astype(np.int32),img_array.shape)

        ## the nearest outside point
        rmax = np.min(radii[img_array==0])
    
    if rmax>aperMax:
        logging.debug('getRmax(): rmax set to 2Mpc, value is too big to be computed in a fast way')
        rmax = aperMax
    return rmax

def getCenter(img_mask,pos,unitsInput='deg',units='physical',outdir='./'):
    '''Given an image and a position, it returns the position in the other coordinate system [units]
    '''
    dmcoords.punlearn()
    X,Y=pos
    if unitsInput=='deg':
        a = dmcoords(img_mask, asol="non", option="cel", ra=X, dec=Y, celfmt='deg', verbose=1)
    else:
        a = dmcoords(img_mask, asol="non", option="sky", x=X, y=Y, celfmt='deg', verbose=1)
    Xra, Xdec = round(float(dmcoords.ra),6), round(float(dmcoords.dec),6)
    xphy, yphy = float(dmcoords.x), float(dmcoords.y)
    xc, yc = float(dmcoords.logicalx), float(dmcoords.logicaly) # in image coordinates

    if units=='deg':
        out = Xra, Xdec
    if units=='image':
        out = round(xc), round(yc)
    if units=='physical':
        out = round(xphy), round(yphy)
    
    return out

def anel(x0,y0,r0,rphy,step,region,mode='linear'):
    if mode=='linear':
        rbin = np.arange(r0,rphy,step)
    
    else: ##log
        length = int(round( np.log((rphy/r0)-1)/np.log(1.05),0))
        rbin = np.array([r0*(1.05)**(n-1) for n in range(1, length + 1)])

    with open(region, 'w') as fout:
        for i in range(len(rbin)-1):
            inrad, outrad = rbin[i],rbin[i+1]
            print('annulus(%.3f,%.3f,%.2f,%.2f)'%(x0,y0,inrad,outrad),file=fout )

def abertura(x0,y0,rphy,region):
    with open(region, 'w') as fout:
        print('circle(%.2f,%.2f,%.2f)'%(x0,y0,rphy),file=fout )

def findRbkg(rprof_file,rmax=None):
    rprof = pycrates.read_file(rprof_file)
    r = pycrates.copy_colvals(rprof,"RMID")
    
    cts = pycrates.copy_colvals(rprof,"CEL_BRI")
    bg_cts = pycrates.copy_colvals(rprof,"BG_CEL_BRI")
    
    if rmax is None:
        rmax = np.max(r)
    
    ## Calculando SNR
    ratio = cts/bg_cts
    dratio = np.diff(ratio)
    ratio_threshold = np.min(ratio[:-1])+(-np.mean(dratio))
    
    mask = (ratio[:-1]<ratio_threshold)
    dratio_threshold = np.median(dratio[mask])
    if (dratio_threshold+1.)>0:
        print('threshold:',dratio_threshold)
        idx, = np.where((dratio > dratio_threshold)&(ratio[:-1]<ratio_threshold))
        x = r[idx+1]
        rbkg = x[0]
    else:
        rbkg = r[-1]
    print('Background Radius:',rbkg)
    
    return float(rbkg)

def SNR_obs(file,r_aper):
    rprof = pycrates.read_file(file)
    source_counts = pycrates.copy_colvals(rprof,"COUNTS")
    total_counts = source_counts+pycrates.copy_colvals(rprof,"BG_COUNTS")
    r = pycrates.copy_colvals(rprof,"R"); rmid = 0.5*(r[:,0] + r[:,1])
    ## The SNR in a given circle of aperture r_aper
    mask = rmid<r_aper
    SNR = np.sum(source_counts[mask])/np.sqrt(np.sum(total_counts[mask]))
    return SNR

## --------------------------------------------------------
######################## Main taks ########################
## --------------------------------------------------------

def maskPointSources(img,psreg='ps.reg',clobber=False):
    checkImg = path.isfile(img)
    if checkImg or clobber:
        img_mask = path.splitext(img)[0]+'_mask%s'%(path.splitext(img)[-1])
        dmcopy(img+"[exclude sky=region(%s)]"%(psreg),img_mask,clobber=True)

## Find the X-ray centroid and X-ray peak
def centerX(img,evt,z,radius=500,outdir='./'):
    '''If finds the X-ray center in 500kpc radius (default)
    '''
    print('CenterX')
    logging.debug('Starting preAnalysis.centerX(%s)')

    ## Find Arcsec to physical units
    rphy = kpcToPhy(radius,z) ## 500 kpc in physical units (default)
    r10 = kpcToPhy(10,z) # 10kpc in physical units

    ## Find the pointing coordinates
    ra0, dec0 = [float(dmkeypar(evt,'RA_PNT','echo+')),float(dmkeypar(evt,'DEC_PNT','echo+'))]
    dmcoords(img, asol="non", option="cel", ra=ra0, dec=dec0, verbose=1)

    x0, y0 = float(dmcoords.x), float(dmcoords.y)

    ## search the center in all FOV
    xcen,ycen = findCentroX(img,x0,y0,450)
    
    ## find the center in a given aperture
    xcen,ycen = findCentroX(img,xcen,ycen,rphy)
    xpeak, ypeak = findXrayPeak(img,xcen,ycen,rphy,rSigma=r10)
    
    dmcoords(img, asol="non", option="sky", x=ra0, y=dec0, verbose=1)

    Xra, Xdec = getCenter(img,[xcen,ycen],unitsInput='physical',units='deg')
    Xra_peak, Xdec_peak = getCenter(img,[xpeak,ypeak],unitsInput='physical',units='deg')
    
    output = path.join(outdir,'log.txt')
    if path.isfile(output):
        os.remove(output)
    
    saveOutput('RADEC','{ra},{dec}'.format(ra=Xra,dec=Xdec),section='Center',out=output)
    saveOutput('RADEC_PEAK','{ra},{dec}'.format(ra=Xra_peak,dec=Xdec_peak),out=output)

    return Xra, Xdec

def radialProfile(img_mask,bg_img,emap,z,center,binf=1,outdir='./'):

    logging.debug('Starting preAnalysis.radialProfile')

    ## Create profile directory
    proDir = getDir('profile',outdir)

    Xra,Xdec = center

    aperMax = kpcToPhy(2000,z) ## 2 Mpc in physical units (default)

    ## Get the center
    xcen, ycen = getCenter(img_mask,[Xra,Xdec],unitsInput='deg',units='physical')
    xc, yc = getCenter(img_mask,[Xra,Xdec],unitsInput='deg',units='image')
    
    ## Find the maximum radius
    rmax = getRmax(xc,yc,img_mask,emap,outdir,binf=binf,aperMax=aperMax)
    
    rmax_anel = path.join(outdir,'check_rmax.reg')
    abertura(xcen,ycen,rmax,rmax_anel)

    ## Define files
    region = path.join(proDir,"rmax.reg")
    bgregion = path.join(proDir,"bkg.reg")
    grpRegion = region.split('.reg')[0]+'_grp.reg'

    rprof_file1 = path.join(proDir,"broad_rprofile_1pix.fits")
    rprof_file = path.join(proDir,"broad_rprofile_binned.fits")
    rprof = path.join(proDir,"broad_rprofile_binned_cut.fits")

    dt = 1
    # ## Computing binned radial profile
    # anel(xcen,ycen,10,rmax,dt,grpRegion,mode='log')
    # rprofile(img_mask,bg_img,emap,grpRegion,out=rprof_file,bgregion=grpRegion)
    
    # The Radial Profile with Blank Fields
    if not path.isfile(rprof_file):
        # create ring
        anel(xcen,ycen,5,1.25*rmax,dt,region) 
        # compute radial profile
        rprofile(img_mask,bg_img,emap,region,out=rprof_file1,bgregion=region)
        dmcopy(rprof_file1+'[rmid<%.2f]'%(rmax),rprof_file1,clobber=True)

    ## Do binning in number of counts
    ct_thresh = 2000
    nbins = doGRP(rprof_file1,ct_thresh,xcen,ycen,grpRegion)
    while nbins < 40:
        ct_thresh = ct_thresh-25
        nbins = doGRP(rprof_file1,ct_thresh,xcen,ycen,grpRegion)
    rprofile(img_mask,bg_img,emap,grpRegion,out=rprof_file,bgregion=grpRegion)
    
    ## find backgound radius and compute SNR
    rbkg = findRbkg(rprof_file,rmax=rmax)
    SNR = SNR_obs(rprof_file,rbkg)
    
    ## Scale the blank field counts at the source background
    dmcopy(rprof_file+'[rmid<=%.2f]'%(rbkg),rprof,clobber=True)

    ## Define a background region
    anel(xcen,ycen,rbkg,rbkg+2*60/0.492+1,2*60/0.492-1,bgregion)
    
    # logging.debug("Signal to noise ratio: {0}".format(SNR))
    # logging.debug("Background radius: {0} arcsec".format(rbkg*0.492))
    # logging.debug("Maximum radius: {0} arcsec".format(rmax*0.492))

    output = path.join(outdir,'log.txt')
    saveOutput('rbkgPhysical',rbkg,out=output)
    saveOutput('rmaxPhysical',rmax,out=output)
    saveOutput('SNR',SNR,out=output)
    saveOutput('countsThreshold',ct_thresh,out=output)

    print('check: \n rbkg, rmax = {}, {} \n SNR = {}'.format(rbkg,rmax,SNR))

def doArfRmf(evt_region,outroot):
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

def arf_rmf(obsid,center,z,radius=1500,outdir='./',downPath='./'):
    logging.debug('Starting preAnalysis.arf_rmf')

    specDir = getDir('spec',outdir)

    # check the OBSID input
    obsid_str, obsid_lis = checkObsid(obsid)
    nObs = len(obsid_lis)

    # Xray-Center [deg]
    Xra, Xdec = center
    rphy = kpcToPhy(radius,z) ## 1500 kpc in physical units (default)

    s = [os.path.join(downPath,'{}'.format(obsid),'repro',"%s_source.reg"%(obsid))  for obsid in obsid_lis]
    evt_gti_lis = [os.path.join(downPath,'{}'.format(obsid),'repro',"{}_evt_gti.fits".format(obsid)) for obsid in obsid_lis]
    
    for i in range(nObs):
        dmcoords(evt_gti_lis[i], asol="non", option="cel", ra=Xra, dec=Xdec, verbose=1)
        xobs, yobs = float(dmcoords.x), float(dmcoords.y)
        abertura(xobs,yobs,rphy,s[i])
    
    evt_gti_reg_lis = [evt_gti_lis[i]+"[sky=region(%s)]"%(s[i]) for i in range(nObs)]
    evt_gti_reg_str = ','.join(evt_gti_reg_lis)
    outroot_lis = [os.path.join(specDir,'{}'.format(obsid)) for obsid in obsid_lis]
    out_str = ','.join(outroot_lis)
    if not os.path.exists(outroot_lis[-1]+'.pi'):
        doArfRmf(evt_gti_reg_str,out_str)
        arf_lis = [os.path.join(specDir,'{}.pi'.format(obsid)) for obsid in obsid_lis]
        arf_str = ",".join(arf_lis)
        # combine_spectra(src_spectra=arf_str,outroot=outroot,clobber=True)

    
if __name__ == '__main__':
    print('pre-process.py')
    print('author: Johnny H. Esteves')
