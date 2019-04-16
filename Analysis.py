# !/usr/bin/env python
# tasks: fit SB, fit kT, estimate Mass, csb, w, ErrorCenterX
# Obs.: don't forget to activate the ciao enviroment!

from astropy.io.fits import getdata
from astropy.table import Table
import astropy.io.ascii as at
import matplotlib.pyplot as plt
import matplotlib

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
from scipy.interpolate import interp1d
from numpy.random import poisson

import fit
import preAnalysis

#--- cosmologia
h = 0.7
cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3)

#--- constants
Msol = 1.98847e33
DEG2RAD=np.pi/180.0
kpc_cm = 3.086e21

# Funções básicas
def AngularDistance(z):
    DA = float( (cosmo.luminosity_distance(z)/(1+z)**2)/u.Mpc ) # em Mpc
    return DA

#--- Convertion function: kpc to physical
def kpcToPhy(radius,z,ARCSEC2PHYSICAL=0.492):
    DA = AngularDistance(z)
    radius = radius/1000 # Mpc
    res = ( (radius/DA)/DEG2RAD )*3600/ARCSEC2PHYSICAL
    return res

#--- Critical universe density
def rhoc(z):
    rho_c = float(cosmo.critical_density(z)/(u.g/u.cm**3)) # em g/cm**3
    return rho_c

#--- Função da evolução do redshift
def E(z):
    res = cosmo.H(z)/cosmo.H(0)
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

def saveFinalOutput(fileName,values):
    
    values_str = values.split(',')
    if not os.path.isfile(fileName):
        header = '# Name, Xra, Xdec, kT, r500, M500, Mg500'
        writeStringToFile(fileName,header)
        writeStringToFile(fileName,values_str)
    else:
        writeStringToFile(fileName,values_str)

def saveOutput(name,value,section=None,out='output.txt'):
    '''Save an ouptut name value into a section in the output file
    '''
    toBeWritten = '{item}: {value}'.format(item=name,value=value)

    if section is not None:
        writeStringToFile(out,'\n[%s]'%(section))
    writeStringToFile(out,toBeWritten)

def saveBeta(pars,out,model='modBeta'):
    pars_str = ' '.join(str(round(pars[i],3)) for i in range(len(pars)))
    if not os.path.isfile(out):
        with open(out, 'w') as f:
            f.write('#This file is the ouput of the beta sb profile fit \n')
            f.write('#The first line is the best fit \n')
            if model=='Beta':
                f.write('#rc beta n0 chisq\n')
            if model=='modBeta':
                f.write('#rc rs alpha beta epsilon gamma n0 bkg chisq\n')
            writeStringToFile(out,pars_str)
    else:
        writeStringToFile(out,pars_str)

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
            idx = np.appsaveBetaend(i,idx)
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

def anel(x0,y0,r0,rphy,step,region):
    rbin = np.arange(r0,rphy,step)
    with open(region, 'w') as fout:
        for i in range(len(rbin)-1):
            inrad, outrad = rbin[i],rbin[i+1]
            print('annulus(%.3f,%.3f,%.2f,%.2f)'%(x0,y0,inrad,outrad),file=fout )

def abertura(x0,y0,rphy,region):
    with open(region, 'w') as fout:
        print('circle(%.2f,%.2f,%.2f)'%(x0,y0,rphy),file=fout )


def makePlotBeta(infile,betapars,name,rbkg=0,model='modBeta',outdir='./'):
    '''Given a radial profile file and the model parameters it plots the electronic profile
    '''
    dirname = os.path.dirname(infile)
    rprof = pycrates.read_file(infile)
    
    r = pycrates.copy_colvals(rprof,"R")
    y = pycrates.copy_colvals(rprof,"CEL_BRI")
    dy = pycrates.copy_colvals(rprof,"CEL_BRI_ERR")
    
    x = 0.492*0.5*(r[:,0] + r[:,1])
    
    if model=='Beta':
        # Beta Model
        ym = betapars[2] * (1 + (x/(betapars[0]*0.492))**2)**(0.5-3*betapars[1])+betapars[3]
        Label = r'$\beta$-model'
    if model=='modBeta':
        # Beta Model modified Maughan et al. 2008
        rc,rs,alpha,beta,epsilon,n0,gamma,bkg,chisqr = betapars
        ym = fit.S_bkg(x,(rc*0.492),(rs*0.492),alpha,beta,epsilon,n0,gamma,bkg)
        ym = (np.max(y)/np.max(ym))*ym
        Label=r'adapted-$\beta$-model'

    doPlotModel(x,y,ym,y_obs_err=dy,name=name,rbkg=rbkg,label=Label,outdir=outdir)

    return x,y,dy,ym

def doPlotModel(r,y_obs,y_model,y_obs_err=None,name='',rbkg=0,label='adapted-β-model',outdir='./'):

    heights = [6,1]
    gs_kw = dict(height_ratios=heights)

    f, (ax1,ax3)=plt.subplots(ncols=1, nrows=2, sharex=True,gridspec_kw=gs_kw )
    # f.suptitle('Perfil radial de brilho superficial - A2142')
    f.suptitle('radial profile - {}'.format(name))
    #data and fit_opt

    ax1.errorbar(r, y_obs, yerr=y_obs_err, fmt='.', capsize=5, mec='dimgrey', mfc='dimgrey', \
    ms=6, elinewidth=1, ecolor='dimgrey' )
    ax1.plot(r, y_model, label=label, color='indianred')
    ax1.axvline(rbkg,linestyle='--',color='k')
    # ax1.set_xlim(r.min(),100*r.max())
    # ax1.set_ylim(y_obs.min()/10,y_obs.min()*10)
    ax1.set_yscale('log')
    ax1.set_ylabel('Surface Brightness ($counts / arcsec^{2})$')
    ax1.legend(loc='best')

    # chivet = (y_obs-y_model)**2/y_obs_err**2
    # ax2.plot(r,chivet, linestyle='',marker='.', color='indianred')
    # ax2.axhline(y=0, linestyle='--',marker='', color='dimgrey')
    # ax2.set_title('$\chi$²', pad=-10., fontsize=8)
    # # ax2.set_xlim(5,100*r.max())

    resid = (y_obs-y_model)/y_model
    ax3.plot(r,resid, linestyle='',marker='.', color='indianred')
    ax3.axhline(y=0, linestyle='--',marker='', color='dimgrey')
    # ax3.set_xlim(5,10*r.max())
    ax3.set_xscale('log')
    ax3.set_ylim(-0.5,0.5)
    ax3.set_title('Residue', pad=-10., fontsize=8)
    ax3.set_xlabel('Radius (arcsec)')

    ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax3.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    nome_fig= path.join(outdir,name+'_sb.png')
    plt.savefig(nome_fig)

def scaleRelation(Yx,redshift):
    #--- Calculando Massa do Halo (relação de escala) Maughan et al. 2012
    AYM, BYM, CYM = 5.77/(h**(0.5)), 0.57, 3*1e14

    M500 = E(redshift)**(-2/5)*AYM*(Yx/CYM)**(BYM) # em 10e14*Msolares
    
    rho_c = rhoc(redshift)
    r500 = ((1e14*M500*Msol)/(4*np.pi/3)/rho_c/500)**(1/3)/kpc_cm
    
    M500, r500 = round(float(M500),4),round(float(r500),4)
    return M500, r500

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

def computeCsb(r500vec,betapars,model='modBeta'):
    
    if model=='Beta':
        # Beta Model modified Maughan et al. 2008
        rc,b,n0,bkg,chisqr = betapars
        res = fit.SBeta(r500vec,rc,b,1)

    if model=='modBeta':
        # Beta Model modified Maughan et al. 2008
        rc,rs,a,b,e,g,n0,bkg,chisqr = betapars
        res = fit.S(r500vec,rc,rs,a,b,e,g,n0)
    
    mask = r500vec<=0.15*np.max(r500vec)
    SB500 = np.sum(res)
    SBcore = np.sum(res[mask])

    csb = SBcore/SB500
    return csb

def center(img,x0,y0,rphy):
    dirname = os.path.dirname(img)
    region = os.path.join(dirname,"aper.reg")
    toto = os.path.join(dirname,"toto.fits")
    totog = os.path.join(dirname,"totog.fits")
    
    # Extaindo imagem dentro do círculo
    abertura(x0,y0,rphy,region)
    dmcopy(img+"[sky=region(%s)]"%(region),toto,clob=True)
    aconvolve(toto,totog,'lib:gaus(2,5,1,10,10)',method="fft",clobber=True)
    dmstat.punlearn()
    bla = dmstat(totog, centroid=True)
    pos = (dmstat.out_cntrd_phy)
    pos = pos.split(',')
    ## Definindo x e y
    x, y = float(pos[0]), float(pos[1])
    return x,y

def centroid_shift(img,x0,y0,r500):
    ## Achando centroide
    centroid = []
    ri, rfim, dr = 0.15*r500,r500,0.05*r500
    rbin = np.arange(ri,rfim+dr,dr)
    # rbin = np.flip(rbin,axis=0)
    for i in range(len(rbin)):
        xr, yr = center(img,x0,y0,rbin[i])
        # print("rbin:",rbin[i])
        # xr, yr = centroX(img,x0,y0,rbin[i])
        centroid.append([xr,yr])
    centroid = np.array(centroid)
    offset = ((centroid[:,0]-x0)**2+(centroid[:,1]-y0)**2)**(1/2)
    ## Estimativa do centroid shift
    w = offset.std(ddof=1)/r500
    return w*1e3

def noise(infits,outimage,mask=None):
	""" Module to add poisson noise to the image data

	Input: Fits image
	Output: Fits image - An image with possion randon noise

	"""
	ctimg = pycrates.read_file(infits)
	img = ctimg.get_image()
	pix = img.values
	noiseimg = poisson(pix)

	if mask != None:
		bla = pycrates.read_file(mask)
		msk_values = bla.get_image().values
		msk = msk_values == 0
		noiseimg[msk] = msk_values[msk]
	
	img.values = noiseimg
	pycrates.write_file(ctimg,outimage,clobber=True)

def getNoiseImages(img,N=20,mask=None,pointSource=None,outdir='./noise/'):
    ''' it produces N images with a poissonian noise'''
    for i in range(1,N+1):
        img_noise = os.path.join(outdir,"sb_%03i.img"%(i))
        if not os.path.isfile(img_noise):
            noise(img,img_noise,mask=mask)
    
    if pointSource is not None:
        for i in range(1,N+1):
            img_mask = os.path.join(outdir,"sb_%03i_mask.img"%(i))
            img_noise = os.path.join(outdir,"sb_%03i.img"%(i))
            if not os.path.isfile(img_mask):
                dmcopy(img_noise+"[exclude sky=region(%s)]"%(pointSource),img_mask,clobber=True)


## --------------------------------------------------------
######################## Main taks ########################
## --------------------------------------------------------

def fitSB(rprof_file,model='modBeta',name='Abell',outdir='./'):
    '''It fits a SB density profile. There are 3 model.
    model=['Beta','doubleBeta','modBeta']
    '''
    # fitDir = getdata(outdir,'fit')
    out = path.join(outdir,'%s.txt'%(model))

    if model=='Beta':    
        betapars = fit.fitBeta(rprof_file)
        saveBeta(betapars,out,model=model)
        
    if model=='modBeta':
        betapars = fit.fitBetaM(rprof_file)
        
        if betapars[0]>betapars[1]:
            print('rc is less than rs')
            rc, rs, alpha, beta, epsilon, gamma, n0, bkg, chisqr = betapars
            betapars = fit.fitBetaM(rprof_file,par0=[rs, 10*rc, alpha, beta, epsilon, gamma, n0, bkg])

        saveBeta(betapars,out,model=model)

    return betapars

def fitTemperatureX(obsid_lis,z,center,radius=500,name='source',outdir='./',dataDownPath='./'):
    nObs = len(obsid_lis)
    ## Check spec dir
    specroot = getDir('spec',outdir)
    Xra, Xdec = center

    ## Find Arcsec to physical units
    rphy = kpcToPhy(radius,z) ## 500 kpc in physical units (default)

    ## Input files
    evt_mask_lis = [path.join(dataDownPath,'{}'.format(obsid),'repro',"{}_evt_gti_mask.fits".format(obsid)) for obsid in obsid_lis]
    blk_evt_lis = [path.join(outdir,"img","{}_blank.evt".format(obsid)) for obsid in obsid_lis]

    ## Output files
    phafile = path.join(specroot,'%s_src.pi'%(name))
    spec_out = path.join(specroot,'spec.txt')   ## output fit
    core_vec = [path.join(specroot,"%s_core.reg"%(obsid)) for obsid in obsid_lis] ## region files

    for i in range(nObs):
        dmcoords(evt_mask_lis[i], asol="non", option="cel", ra=Xra, dec=Xdec, verbose=1)
        xobs, yobs = float(dmcoords.x), float(dmcoords.y)
        # anel(xobs,yobs,0.05*rphy,rphy+1,0.85*rphy,core_vec[i])
        abertura(xobs,yobs,rphy+1,core_vec[i])
        fit.kT_prep(obsid_lis[i],evt_mask_lis[i],blk_evt_lis[i],core_vec[i],specroot)

    spec_lis = ','.join( os.path.join(specroot,'%s.pi'%(obsid)) for obsid in obsid_lis )
    combine_spectra(src_spectra=spec_lis,outroot=os.path.join(specroot,"%s"%(name)),bscale_method='asca',clobber=True)
    dmhedit(infile=phafile, filelist="", operation='add', key='ANCRFILE', value='%s_src.arf'%(name))
    dmhedit(infile=phafile, filelist="", operation='add', key='RESPFILE', value='%s_src.rmf'%(name))

    norm, kT, ksqr = fit.fit_kT(phafile,5.,z,spec_out)

    return norm, kT, ksqr

def massX(obsid_lis,z,center,radial_profile,kT_0=5,r0=500,rbkg=1000,model='modBeta',name='Abell',outdir='./',dataDownPath='./'):
    """ Given a ...
        it estimates the M500
    """
    ## Check fit dir
    fitDir = getDir(outdir,'fit')
    currentPath = os.getcwd()
    dirCheck = path.join(currentPath,'check')
    sb_plot_dir = getDir('sb',dirCheck)
    
    ## output sb parameters
    out = path.join(fitDir,'{}.txt'.format(model))

    DA = AngularDistance(z)     # em kpc
    ARCSEC2kpc = ( (1/3600)*DEG2RAD )*1000*DA      # kpc/arcsec
    phy2cm = (ARCSEC2kpc*0.492)*kpc_cm             # (kpc/arcsec)/(physical/arcsec)

    ## Convert radius to physical units
    r0phy = kpcToPhy(r0,z)  ## kpc to phsyical units

    ## Fit SB
    ## cut at the background radius
    rprof = radial_profile.split('.fits')[0]+'_cut.fits'
    if model=='Beta':    
        betapars = fitSB(rprof,model=model,name=name,outdir=fitDir)
        saveBeta(betapars,out,model=model)
    if model=='modBeta':
        betapars = fitSB(rprof,model=model,name=name,outdir=fitDir)
        saveBeta(betapars,out,model=model)
    
    ## Make a plot
    makePlotBeta(radial_profile,betapars,name,rbkg=0.492*rbkg,model=model,outdir=sb_plot_dir)

    conv = 100; count = 1
    while (count<20):
        print("step %i"%(count))
        r500,r500phy = r0, r0phy
        norm, kT, ksqr = fitTemperatureX(obsid_lis,z,center,radius=r500,name=name,outdir=fitDir,dataDownPath=dataDownPath)
        
        #--- Calculando n0
        EI_xspec = 1e14*norm*4*np.pi*(DA*1e3*kpc_cm*(1+z))**2
        EI_model = fit.EI(r500phy,betapars,phy2cm,model=model)
        n0 = ( EI_xspec / EI_model )**(1/2)         ## 1/cm^3

        #--- Calculando Mg em R500
        Mg500 = fit.Mgas(r500phy,betapars,n0,phy2cm,model=model)

        #--- Calculando Yx
        Yx = 1e13*Mg500*kT

        M500, r500 = scaleRelation(Yx,z)

        conv = round(100*np.abs(r500-r0)/r0,2)
        r0, r0phy = r500, kpcToPhy(r500,z)

        count += 1
        print(25*'--')
        print('%s'%(name))
        print("n0:",n0,"cm^-3")
        print("Mg500:",Mg500,"10^13 solar masses")
        print("M500:",M500,"10^14 solar masses")
        print("r500:",r500,"kpc")
        print("kT:",round(kT,2),"keV")
        print("The convergence is:",conv,"%")
        print(25*'--')
        if conv<1.0:
            break
    
    output=os.path.join(outdir,'log.txt')
    saveOutput('kT',kT,section='Fit',out=output)
    saveOutput('R500',r500,out=output)
    saveOutput('Mg500',Mg500,out=output)
    saveOutput('M500',M500,out=output)
    saveOutput('n0',n0,out=output)
    
    ## Switch n0
    if model=='modBeta':
        rc,rs,a,b,e,_,g,bkg,chisqr = betapars
        betapars = [rc,rs,a,b,e,n0,g,bkg,chisqr]
    
    if model=='Beta':
        rc,_,bkg,chisqr = betapars
        betapars = [rc,b,n0,bkg,chisqr]

    # out = [Xra,Xdec,r500,Mg500,M500,kT]
    # saveFinalOutput(out)

    return kT, r500, Mg500, M500, betapars

def csb(betapars,r500,z,outdir='./'):
    r500phy = kpcToPhy(r500,z)  ## kpc to phsyical units
    r500vec = np.arange(2,r500phy,1)
    
    csb = computeCsb(r500vec,betapars,model='modBeta')

    output=os.path.join(outdir,'log.txt')
    saveOutput('csb',csb,out=output)
    print('csb:',csb)

    return csb

def centroidShift(img,center_peak,r500,rmax,z,outdir='./'):
    r500phy = kpcToPhy(r500,z)  ## kpc to phsyical units
    r30kpc = kpcToPhy(30,z)
    noiseroot = getDir('noise',outdir)

    ## center
    xpeak, ypeak = getCenter(img,center_peak,unitsInput='deg',units='physical')

    ## Excluindo região central dentro de 30kpc
    core = os.path.join(noiseroot,'core.reg')
    abertura(xpeak, ypeak, r30kpc, core)
    
    ## Check noise images
    N=10
    getNoiseImages(img,N=N,outdir=noiseroot)
    
    w = []
    ## Definindo cascas
    rt = np.min([r500phy,rmax])
    
    for i in range(1,N+1):
        noisei = path.join(noiseroot,"sb_%03i.img"%(i))
        res = centroid_shift(noisei,xpeak,ypeak,rt)
        w.append(res)

    wvalue = np.mean(np.array(w))
    werr = np.std(np.array(w))
    print("<w>, w_err : ( %.3f +/- %.3f )1e-3"%(wvalue, werr))

    output=os.path.join(outdir,'log.txt')
    saveOutput('w',(wvalue,werr),out=output)

    return wvalue,werr

def errorCenterX(img,center,psreg,z,radius=500,outdir='./'):
    '''Estimate the error in the X-ray center and X-ray peak
    '''
    rphy = kpcToPhy(radius,z)  ## kpc to phsyical units
    r10kpc = kpcToPhy(10,z)
    DA = AngularDistance(z)     # em kpc

    noiseroot = getDir('noise',outdir)

    ## Get the new center
    xcen, ycen = getCenter(img,center,unitsInput='deg',units='physical')    # Get an initial center
    xcen, ycen = preAnalysis.findCentroX(img,xcen,ycen,rphy)   ## Find the center at the given radius
    xpeak,ypeak= preAnalysis.findXrayPeak(img,xcen,ycen,rphy)   ## Find the center at the given radius

    ## Check noise images
    N=10
    getNoiseImages(img,N=N,pointSource=psreg,outdir=noiseroot)
    
    position = []
    position2 = []
    for i in range(1,N+1):
        img_mask = os.path.join(noiseroot,"sb_%03i_mask.img"%(i))
        res = preAnalysis.findCentroX(img_mask,xcen,ycen,rphy)
        res2 = preAnalysis.findXrayPeak(img_mask,xcen,ycen,rphy,rSigma=r10kpc)
        position.append(res)
        position2.append(res2)
    
    position = np.array(position);position2 = np.array(position2)
    
    ARCSEC_kpc = 0.492*( (1/3600)*DEG2RAD )*1000*DA    # kpc/arcsec
    std_cen = ARCSEC_kpc*(np.std(position[:,0])**2+np.std(position[:,1])**2)**(1/2)
    std_peak = ARCSEC_kpc*(np.std(position2[:,0])**2+np.std(position2[:,1])**2)**(1/2)

    
    # print("X-ray Peak:", xpeak, ypeak, " +/- ",std_peak)

    img_mask = path.splitext(img)[0]+'_mask'+path.splitext(img)[1]
    Xra, Xdec = getCenter(img_mask,[xcen,ycen],unitsInput='physical',units='deg')
    Xra_peak, Xdec_peak = getCenter(img_mask,[xpeak,ypeak],unitsInput='physical',units='deg')
    
    print("X-ray center:", Xra,Xdec, " +/- ",std_cen,' [kpc]')
    np.savetxt(os.path.join(outdir,'center.txt'),position,fmt='%4f')
    np.savetxt(os.path.join(outdir,'xpeak.txt'),position2,fmt='%4f')

    output=os.path.join(outdir,'log.txt')
    saveOutput('errorCenter',std_cen,out=output)

    return Xra, Xdec, std_cen, Xra_peak, Xdec_peak

if __name__ == '__main__':
    print('Analysis.py')
    print('author: Johnny H. Esteves')
