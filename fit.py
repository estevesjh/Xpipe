# http://cxc.harvard.edu/sherpa4.10/threads/sourceandbg/
from sherpa_contrib.all import *
from sherpa.astro.ui import *
from ciao_contrib.runtool import *
from pychips import *
from pycrates import *                     # import crates ciao routines
import astropy.io.ascii as at
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


####################################################################################################################
####################################### FIT TEMPERATURE ############################################################
####################################################################################################################

def kT_prep(obsid,evt_mask,blkevt,core,dirname):
    pha = os.path.join(dirname,'%s.pi'%(obsid))
    phabkg = os.path.join(dirname,'%s_bkg.pi'%(obsid))
    dmextract.punlearn()
    dmextract(evt_mask+"[sky=region(%s)][bin pi]"%(core),pha,wmap="[energy=300:2000][bin tdet=8]",op="pha1",clobber=True)
    dmextract(blkevt+"[sky=region(%s)][bin pi]"%(core),phabkg,wmap="[energy=300:2000][bin tdet=8]",op="pha1",clobber=True)
    header_arf_rmf(obsid,dirname)

def header_arf_rmf(obsid,dirname):
    pha = os.path.join(dirname,'%s.pi'%(obsid))
    phabkg = os.path.join(dirname,'%s_bkg.pi'%(obsid))
    dmhedit.punlearn()
    dmhedit(infile=pha, filelist="", operation='add', key='BACKFILE', value='%s_bkg.pi'%(obsid))
    dmhedit(infile=pha, filelist="", operation='add', key='ANCRFILE', value='%s.arf'%(obsid))
    dmhedit(infile=pha, filelist="", operation='add', key='RESPFILE', value='%s.rmf'%(obsid))

    # dmhedit(infile=phabkg,operation='add', filelist="", key='ANCRFILE', value='%s.arf'%(obsid))
    # dmhedit(infile=phabkg,operation='add', filelist="", key='RESPFILE', value='%s.rmf'%(obsid))

def fit_kT(phafile,kT,z,out):
    load_pha(1,phafile)
    notice(0.3,8)
    group_counts(30)
    subtract()
    # Modelos
    set_source(xsphabs.abs1*xsmeka.p1)
    # set_source(xsmeka.p1)
    # set_bkg_model(xsphabs.abs1*const1d.bkg)
    # Parametros inciais
    abs1.nH = 0.07
    # p1.nH = 0.07
    p1.redshift = z
    p1.kT = kT
    p1.Abundanc = 0.3
    freeze(p1,abs1)
    thaw(p1.norm,p1.kT)
    # show_model()
    bla = fit(outfile=out,clobber=True)
    fitr = get_fit_results()
    parnames = fitr.parnames
    parvals = np.array(fitr.parvals)
    out = {parnames[i]: parvals[i] for i in range(len(parvals))}
    return out['p1.norm'], out['p1.kT'], fitr.rstat

####################################################################################################################
####################################### FIT SURFACE BRIGHTNESS #####################################################
####################################################################################################################
mp = 1.67262e-24; G = 6.67408e-8; Msol = 1.98847e33; mu=1.151; kpc_cm=3.086e+21;

#--- Função Hipergeometrica
def F1(r,rc,beta):
    a=(3/2); b=(3/2)*beta; c=(5/2)
    d=-(r/rc)**2
    I=sp.hyp2f1(a, b, c, d)
    return I

def F1_EI(r,rc,beta):
    a=(1); b=(5/2) - 3*beta; c=(5/2)
    d=-(r/rc)**2
    I=sp.hyp2f1(a, b, c, d)
    return I

F1vec = np.vectorize(F1)

## Fit SB profile
npne = lambda r,rc,rs,a,b,e,g,n0: (n0**2)*( (r/rc)**(-a) )*( ( 1+(r/rs)**g )**(-e/g) ) /(1+(r/rc)**2)**(3*b-a/2)
argS = lambda r,R,rc,rs,a,b,e,g,n0: 2*npne(r,rc,rs,a,b,e,g,n0)*r/np.sqrt(r**2 - R**2)
def S(x,rc,rs,a,b,e,g,n0):
    out=[]
    for R in x: 
        aux=quad(argS,R,np.inf,args=(R,rc,rs,a,b,e,g,n0),epsabs=1e-9,limit=250,epsrel=1.49e-6,full_output=1)
        out.append(aux[0])
    return np.array(out)
S_bkg = lambda R,rc,rs,a,b,e,g,n0,bkg: S(R,rc,rs,a,b,e,g,n0)+bkg

## arg: Emission measure integral
argnpneSquare = lambda x,rc,rs,a,b,e,g,n0: 4*np.pi*(npne(x,rc,rs,a,b,e,g,n0))*x**2
def EI(r,pars,phy2cm,model='modBeta'):
    '''r should be in physical units
    phy2cm is the conversion factor of arcsec units to kpc
    EI = int (n/n0)^2 dV
    '''
    n0=1
    if model=='Beta':
        rc,beta,_,bkg,chisq=pars
        res = (4*np.pi*(r)**3/3)*F1_EI(r,rc,beta)*(phy2cm)**(3)
    
    if model=='modBeta':
        rc,rs,alpha,beta,epsilon,gamma,_,bkg,chisq = pars
        aux = quad(argnpneSquare,1,r,args=(rc,rs,alpha,beta,epsilon,gamma,n0))
        res = aux[0]*(phy2cm)**(3)
    
    return res

## Gas Mass Estimation
mu = 1.17
ne = lambda r,rc,rs,a,b,e,g,n0: np.sqrt(mu)*(n0)*( (r/rc)**(-a/2) )*( (1+(r/rs)**(g)) )**(-e/(2*g))/(1+(r/rc)**2)**((3*b-a/2)/2)
argnpne = lambda x,rc,rs,a,b,e,g,n0: 4*np.pi*ne(x,rc,rs,a,b,e,g,n0)*x**2

#--- Massa do Gás em função do raio
def Mgas(r,pars,n0,phy2cm,model='modBeta'):
    # r = r*kpc_cm
    mu=1.17
    if model=='Beta':
        rc,beta,_,bkg,chisq=pars
        Integral = n0*(4*np.pi*(r)**3/3)*F1(r,rc,beta)*(phy2cm)**(3)
    
    if model=='modBeta':
        rc,rs,alpha,beta,epsilon,gamma,_,bkg,chisq = pars
        aux = quad(argnpne,2,r,args=(rc,rs,alpha,beta,epsilon,gamma,1),full_output=1)
        Integral = n0*aux[0]*(phy2cm)**(3)

    mgas = float(mu*mp*Integral/Msol)

    return mgas/1e13

def S_bkg2(pars,R):
    (rc,rs,a,b,e,g,n0,bkg) = pars
    return S(R,rc,rs,a,b,e,g,n0)+bkg

def SBeta(r,rc,beta,S0):
    res = S0/(1+(r/rc)**2)**(3*beta-1/2)
    return res

def fitBeta(table):
    # dirname = os.path.dirname(table)
    load_data(1,table, 3, ["RMID","CEL_BRI","CEL_BRI_ERR"])
    set_source("beta1d.sbr1")
    sbr1.r0 = 105
    sbr1.beta = 4
    sbr1.ampl = 0.00993448
    freeze(sbr1.xpos)
    set_source("const1d.bkg")
    bkg.c0 = 0.01
    set_model(sbr1+bkg)
    fit()
    fitr = get_fit_results()
    out = np.array(fitr.parvals)
    out = np.append(out,fitr.rstat)
    # plot_fit()
    # log_scale(XY_AXIS)
    # covar()
    # opts = { "clobber": True}
    # print_window(os.path.join(dirname,"sb_beta.png"),opts)
    # clear_plot()
    return out

def fitmodBeta(infile,par0=None):
    rprof = read_file(infile)
    r_med = copy_colvals(rprof,"RMID")
    s_obs = copy_colvals(rprof,"CEL_BRI")
    s_obs_err = copy_colvals(rprof,"CEL_BRI_ERR")
    
    if par0 is None:    
        rc0,rs0,a0,b0,e0,g0,n0_0,bkg0 = 90,450,0.65,0.7,2.5,3.,0.12,0.003
        par0 = [rc0,rs0,a0,b0,e0,g0,n0_0,bkg0]
        
    elo = 1e-5
    ## g eh fixo - outros parametros sao ajustados
    Bound = ((elo,elo,elo,1/6,elo,3-elo,0.,0.),
            (np.infty,np.infty,3,2, 5, 3+elo,np.infty,1.))

    ## ajuste
    params, params_cov = curve_fit(S_bkg, r_med, s_obs, sigma=s_obs_err,p0=par0,bounds=Bound)
    
    ## ajuste
    params, params_cov = curve_fit(S_bkg, r_med, s_obs, sigma=s_obs_err,p0=par0,bounds=Bound,)

    return params

# with open(fileName, 'w') as f:
#     f.write( '#r,S,Serr \n')
#     for i in range(len(r_med)):
#         f.write('{r},{s:.5f},{serr:.5f}\n'.format(r=r_med[i],s=s_obs[i],serr=s_obs_err[i]))

def dofitBetaM(table,par0=None,parfrozen=None, chisqr=False):
    load_data(1,table, 3, ["RMID","CEL_BRI","CEL_BRI_ERR"])
    pars = ["rc","rs","alpha","beta","epsilon","gamma",'n0',"bkg"]
    load_user_model(S_bkg2,"mybeta")

    if par0 is None:
        rc0,rs0,alpha0,beta0,epsilon0,gamma0,n0,bkg0 = 54.511,250,0.1,0.65,2.,3.,0.04,0.00053
        par0 = [rc0,rs0,alpha0,beta0,epsilon0,gamma0,n0,bkg0]

    if parfrozen is None:
        parfrozen=[False]+[False]+[False]+[False]+[False]+[True]+[False]+[False]
    
    add_user_pars("mybeta", pars, par0, parfrozen=parfrozen)
    set_model("mybeta")

    var = 1e-6
    mybeta.bkg.min = 0
    mybeta.rc.min = 0
    mybeta.rs.min = 0
    mybeta.alpha.min = 0.
    mybeta.beta.min = var
    mybeta.gamma.min = 3-var
    mybeta.epsilon.min = 0.

    mybeta.rc.max = 1e3
    mybeta.rs.max = 1e6
    mybeta.alpha.max = 3-var
    mybeta.beta.max = 2
    mybeta.gamma.max = 3+var
    mybeta.epsilon.max = 5
    mybeta.n0.min = 0

    # freeze(mybeta.alpha)
    try:
        show_model()
        fit()
        fitr = get_fit_results()
        out = fitr.parvals
        par0 = np.array(par0)
        par0[ np.logical_not(parfrozen) ] = out

        if chisqr:
            par0 = np.append(np.array(par0),fitr.rstat)

        return par0

    except:
        print('---'*10)
        print('Fit Error')
        print('---'*10)
        rc0,rs0,alpha0,beta0,epsilon0,gamma0,n0,bkg0 = par0
        rc0,beta0,n0,bkg0,chisqr0 = fitBeta(table)
        out = np.array([rc0,rs0,alpha0,beta0,epsilon0,gamma0,n0,bkg0,chisqr0])
        return out

def fitBetaM(table,par0=None):
    '''
    '''
    if par0 is None:
        rc0,beta0,n0,bkg0,chisqr0 = fitBeta(table)
        rs0,alpha0,epsilon0,gamma0,bkg0 = 1000,0.,0.,3.,1e-4
        par0 = [rc0,rs0,alpha0,beta0,epsilon0,gamma0,n0,bkg0]
    
    Npar = 8
    # # ## First: Fit a beta model
    frzpar=[False]+[True]+[True]+[False]+[True]+[True]+[False]+[True]
    outp1 = dofitBetaM(table,par0=par0,parfrozen=frzpar,chisqr=True)
    par1_lis = [outp1[i] for i in range(Npar)]

    ## Second: Fit a power law(cusp) + beta Model
    frzpar=[False]+[True]+[False]+[False]+[True]+[True]+[False]+[True]
    outp2 = dofitBetaM(table,par0=par1_lis,parfrozen=frzpar,chisqr=True)
    par2_lis = [outp2[i] for i in range(Npar)]

    ## The final final: beta model modified
    frzpar=[False]+[False]+[False]+[False]+[False]+[True]+[False]+[True]
    outp3 = dofitBetaM(table,par0=par2_lis,parfrozen=frzpar,chisqr=True)

    outvec = np.array([outp1,outp2,outp3])
    
    ## Take the fit with the best chisqr
    idx = np.argmin( np.abs( 1-outvec[:,-1] ) )

    return outvec[idx]

def SB(x,n0,rc,beta):
    y = n0*(1+(x/rc)**2)**(0.5-3*beta)
    return y