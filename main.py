import numpy as np
import logging
import os

from time import time
import configparser as ConfigParser
from joblib import Parallel, delayed

#personal modules
import preProcess
import imaging
import preAnalysis
import Analysis

import getCat

def getConfig(section, item, boolean=False,
		userConfigFile="./Xpipe_Config.ini"):

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	# if config item not found, raise log warning
	if (not configFile.has_option(section, item)):
		msg = '{item} from [{section}] NOT found in config file: {userConfigFile}!'.format(
			item=item, section=section,
			userConfigFile=userConfigFile)
		if (section != 'Log'):
			logging.warning(msg)
		else:
			print(msg)
		return ""

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))
	if (section != 'Log'):
		logging.debug(msg)
	else:
		print(msg)

	if (not boolean):
		return configFile.get(section, item)

	else:
		return configFile.getboolean(section, item)

def getOutConfig(section, item, boolean=False,
		userConfigFile="./config.out"):

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	# if config item not found, raise log warning
	if (not configFile.has_option(section, item)):
		msg = '{item} from [{section}] NOT found in config file: {userConfigFile}!'.format(
			item=item, section=section,
			userConfigFile=userConfigFile)
		if (section != 'Log'):
			logging.warning(msg)
		else:
			print(msg)
		return ""

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))
	logging.debug(msg)
	
	if (not boolean):
		return configFile.get(section, item)

	else:
		return configFile.getboolean(section, item)

def createLog():
	logLevel = getConfig("Log","level")
	logFileName = getConfig("Log","logFile")
	myFormat = '[%(asctime)s] [%(levelname)s]\t%(module)s - %(message)s'
	if logLevel == 'DEBUG':
		logging.basicConfig(
			filename=logFileName,
			level=logging.DEBUG,
			format=myFormat)
	else:
		logging.basicConfig(
			filename=logFileName,
			level=logging.INFO,
			format=myFormat)

def isOperationSet(operation,section="Operations"):
	return getConfig(boolean=True, section=section,
		item=operation)

def isModeSet(operation,section="Mode"):
	return getConfig(boolean=True, section=section,
		item=operation)

def getPath(type="inputPath"):
	path = getConfig("Paths",type)
	return path

def getColsName():
	section = 'Columns'
	
	ID = getConfig(section,"ID")
	OBSIDS = getConfig(section,"obsid")
	RA = getConfig(section,"ra")
	DEC = getConfig(section,"dec")
	REDSHIFT = getConfig(section,"redshift")

	colNames = [ID,OBSIDS,RA,DEC,REDSHIFT]
	return colNames

def getCenter(outputFile,peak=False):
	if peak:
		center = getOutConfig('Center', "RADEC_PEAK",userConfigFile=outputFile).split(',')		## Take the luminosity center
	else:
		center = getOutConfig('Center', "RADEC",userConfigFile=outputFile).split(',')		## Take the luminosity center
	Xra,Xdec=center
	center = float(Xra),float(Xdec)
	return center

def writeStringToFile(fileName, toBeWritten):
    # create file if it does not exist
    if not os.path.isfile(fileName):
        with open(fileName, 'w') as f:
            f.write( '{toBeWritten}\n'.format(toBeWritten=toBeWritten) )
    # else, append to file
    else:
        with open(fileName, 'a') as f:
            f.write( '{toBeWritten}\n'.format(toBeWritten=toBeWritten) )

def checkDir(path,name):
	outObjectPath = os.path.join(path,name)
	if not os.path.exists(outObjectPath):
		os.makedirs(outObjectPath)
	return outObjectPath

def checkOutputFile(path,name):
	outFile = os.path.join(path,name)
	if os.path.isfile(outFile):
		os.remove(outFile)
	toBeWritten = '# Xpipe ouput: main.py \n'
	writeStringToFile(outFile, toBeWritten)
	return outFile

def doPreProcess(list_obsids,idx=np.array([0,1],dtype=int)):
	
	# check the tasks
	download = isOperationSet("dowloadChandraObservation",section='PreProcess')
	repro = isOperationSet("reproData",section='PreProcess')
	flare = isOperationSet("flares",section='PreProcess')

	# get the download outdir
	dataDownPath = getPath(type="dataDownPath")
	list_obsid_cut = list_obsids[idx]

	if download:
		## It downloads the observation in a given path with the obsid as folder name
		for obsid in list_obsid_cut: # it runs for one object per time
			obsid.replace(" ","")
			logging.debug('down({})'.format(obsid))
			print('down({})'.format(obsid))
			preProcess.down(obsid,path=dataDownPath)
	
	if repro:
		## It reprocess the primary data
		for obsid in list_obsid_cut: # it runs for one object per time
			logging.debug('repro({})'.format(obsid))
			obsid.replace(" ","")
			print('repro({})'.format(obsid))
			preProcess.repro(obsid,path=dataDownPath)

	if flare:
		## It clean the flares of the event files
		for obsid in list_obsid_cut: # it runs for one object per time
			logging.debug('flare({})'.format(obsid))
			print('flare({})'.format(obsid))
			preProcess.flare(obsid,path=dataDownPath,clobber=False,method='sigma')

def doImaging(names,list_obsids,idx=np.array([0,1],dtype=int)):
	# check the tasks
	fluxImage = isOperationSet("fluxImage",section='Imaging')
	blankField = isOperationSet("blankField",section='Imaging')
	pointSources = isOperationSet("pointSources",section='Imaging')

	# get the download outdir
	dataDownPath = getPath(type="dataDownPath")
	outPath = getPath(type="outputPath")
	
	# get the imaging parameters
	band = getConfig('Imaging', "energy")
	binSize = int(getConfig('Imaging', "binSize"))
	elo, ehi, expMap = band.split(':')

	# take a sub-sample
	list_obsid_cut = list_obsids[idx]
	name_cut = names[idx]

	for obsid,name in zip(list_obsid_cut,name_cut): # it runs for one object per time
		name = 'RM-{}'.format(name)
		outObjectPath = checkDir(outPath,name)

		# tasks
		if fluxImage:	
			imaging.fluxImage(obsid,blankField=blankField,energy=band,binsize=binSize,pathData=dataDownPath,outdir=outObjectPath,clobber=False)
	
		if pointSources:
			imaging.pointSources(obsid,name,energy_low=(elo), energy_high=(ehi), outdir=outObjectPath, clobber=False)
	

def doPreAnalysis(names,list_obsids,redshift,idx=np.array([0,1],dtype=int)):
	# check the tasks
	maskPointSources = isOperationSet("maskPointSources",section='PreAnalysis')
	centerX = isOperationSet("centerX",section='PreAnalysis')
	radialProfile = isOperationSet("radialProfile",section='PreAnalysis')
	arf_rmf = isOperationSet("arf_rmf",section='PreAnalysis')
	
	# get the download outdir
	dataDownPath = getPath(type="dataDownPath")
	outPath = getPath(type="outputPath")

	# get the imaging parameters
	binSize = int(getConfig('Imaging', "binSize"))

	# take a sub-sample
	list_obsid_cut = list_obsids[idx]
	name_cut = names[idx]
	z_cut = redshift[idx]

	for (obsid,name,z) in zip(list_obsid_cut,name_cut,z_cut): # it runs for one object per time
		name = str(name).replace(" ","")		## Replace white spaces
		# name = 'RM-{}'.format(name)
		outObjectPath = checkDir(outPath,name)
		obsid_str, obsid_lis = preAnalysis.checkObsid(obsid)
		print('name {} at redhsift {}'.format(name,z))
		
		outputFile = checkOutputFile(outObjectPath,'log.txt')
		# event files
		evt_lis = [os.path.join(dataDownPath,'{}'.format(obs),'repro',"{}_evt_gti.fits".format(obs)) for obs in obsid_lis]
		# region files
		ps = os.path.join(outObjectPath,'ps.reg')
		# imaging files
		count_img = os.path.join(outObjectPath,'sb_thresh.img')
		count_mask_img = os.path.join(outObjectPath,'sb_thresh_mask.img')
		bg_img = os.path.join(outObjectPath,'sb_blank_particle_bgnd.img')
		emap = os.path.join(outObjectPath,'sb_thresh.expmap')
		
		if maskPointSources:
			preAnalysis.maskPointSources(count_img,psreg=ps,clobber=True)
			for evt in evt_lis:
				preAnalysis.maskPointSources(evt,psreg=ps,clobber=False)

		if centerX:
			'''If finds the X-ray luminosity centroid and the X-ray peak in a 500kpc radius
			'''
			radii=500 #kpc
			center = preAnalysis.centerX(count_mask_img,evt_lis[0],z,radius=radii,outdir=outObjectPath)
		else:
			center = getCenter(outputFile)

		if radialProfile:
			preAnalysis.radialProfile(count_mask_img,bg_img,emap,z,center,binf=binSize,outdir=outObjectPath)

		if arf_rmf:
			preAnalysis.arf_rmf(obsid_str,center,z,radius=1500,outdir=outObjectPath,downPath=dataDownPath)

def doAnalysis(names,list_obsids,redshift,idx=np.array([0,1],dtype=int)):
	# check the tasks
	# fitSB = isOperationSet("fitSB",section='Analysis')
	# temperatureX = isOperationSet("temperatureX",section='Analysis')
	massX = isOperationSet("massX",section='Analysis')
	csb = isOperationSet("csb",section='Analysis')
	centroidShift = isOperationSet("centroidShift",section='Analysis')
	errorCenterX = isOperationSet("errorCenterX",section='Analysis')
	
	# get the download outdir
	dataDownPath = getPath(type="dataDownPath")
	outPath = getPath(type="outputPath")

	# get the analysis parameters
	model = getConfig('Analysis', "Model")

	# take a sub-sample
	list_obsid_cut = list_obsids[idx]
	name_cut = names[idx]
	z_cut = redshift[idx]

	for (obsid,name,z) in zip(list_obsid_cut,name_cut,z_cut): # it runs for one object per time
		name = str(name).replace(" ","")		## Replace white spaces
		# name = 'RM-{}'.format(name)
		obsid_str, obsid_lis = preAnalysis.checkObsid(obsid)
		print('name {} at redhsift {}'.format(name,z))

		outObjectPath = checkDir(outPath,name)
		outputFile = os.path.join(outObjectPath,'log.txt')
		center = getCenter(outputFile)
		center_peak = getCenter(outputFile,peak=True)
		rmaxPhysical = float(getOutConfig('Center', "rmaxPhysical",userConfigFile=outputFile))
		rbkgPhysical = float(getOutConfig('Center', "rbkgPhysical",userConfigFile=outputFile))
		
		#files
		profile = os.path.join(outObjectPath,"profile","broad_rprofile_binned.fits")
		count_img = os.path.join(outObjectPath,'sb_thresh.img')
		ps = os.path.join(outObjectPath,'ps.reg')
		print('rbkg',rbkgPhysical)
		if massX:
			kT, r500, Mg500, M500, betapars  = Analysis.massX(obsid_lis,z,center,profile,kT_0=5,rbkg=rbkgPhysical,r0=500,model=model,name=name,outdir=outObjectPath,dataDownPath=dataDownPath)
		else:
			# betapars = np.loadtxt(os.path.join(outObjectPath,model+'.txt'))
			r500 = float(getOutConfig('Fit', "R500",userConfigFile=outputFile))
		
		if csb:
			csb = Analysis.csb(betapars,r500,z,outdir=outObjectPath)

		if centroidShift:
			w,werr = Analysis.centroidShift(count_img,center_peak,r500,rmaxPhysical,z,outdir=outObjectPath)
		
		if errorCenterX:
			Xra, Xdec, errX, Xra_peak, Xdec_peak = Analysis.errorCenterX(count_img,center,ps,z,radius=r500,outdir=outObjectPath)

		outCatFile = './output.txt'
		header = '#Name, Xra, Xdec, errX, redshift, kT, r500, Mg500, M500, w, werr, csb, Xra_peak, Xdec_peak'
		line = '{n},{ra:.5f},{dec:.5f},{errX:.0f},{z:.3f},{kT:.2f},{r500:.0f},{Mg500:.2f},{M500:.2f},{w:.3f},{werr:.3f},{csb:.2f},{ra_p:.5f},{dec_p:.5f}'.format(
				n=name,ra=Xra,dec=Xdec,errX=errX, z=z, kT=kT, r500=r500, Mg500=Mg500,M500=M500,w=w,werr=werr,csb=csb,ra_p=Xra_peak,dec_p=Xdec_peak
		)
		writeStringToFile(outCatFile,header)
		writeStringToFile(outCatFile,line)

def parallelPreProcess(list_obsids,batchStart=0,batchMax=20, nJobs=10, nCores=4):
	objectsPerJob = (batchMax-batchStart)/nJobs
	batchesList = np.arange(batchStart, batchMax, objectsPerJob, dtype=int)

	logging.info('Calling parallelism inside doPreProcess().')
	Parallel(n_jobs=nCores)(delayed(doPreProcess)
		(list_obsids, idx = np.arange(batch,batch +objectsPerJob,dtype=int))
		for batch in batchesList)

def parallelImaging(names,list_obsids,batchStart=0,batchMax=20, nJobs=10, nCores=4):
	objectsPerJob = int((batchMax-batchStart)/nJobs)
	batchesList = np.arange(batchStart, batchMax, objectsPerJob, dtype=int)

	logging.info('Calling parallelism inside doImaging().')
	Parallel(n_jobs=nCores)(delayed(doImaging)
		(names, list_obsids,idx=np.arange(batch,batch +objectsPerJob,dtype=int))
		for batch in batchesList)

def parallelPreAnalysis(names,list_obsids,redshift,batchStart=0,batchMax=20, nJobs=10, nCores=4):
	objectsPerJob = int((batchMax-batchStart)/nJobs)
	batchesList = np.arange(batchStart, batchMax, objectsPerJob, dtype=int)

	logging.info('Calling parallelism inside doPreAnalysis().')
	Parallel(n_jobs=nCores)(delayed(doPreAnalysis)
		(names, list_obsids,redshift,idx=np.arange(batch,batch +objectsPerJob,dtype=int))
		for batch in batchesList)

def parallelAnalysis(names,list_obsids,redshift,batchStart=0,batchMax=20, nJobs=10, nCores=4):
	objectsPerJob = int((batchMax-batchStart)/nJobs)
	batchesList = np.arange(batchStart, batchMax, objectsPerJob, dtype=int)

	logging.info('Calling parallelism inside doPreAnalysis().')
	Parallel(n_jobs=nCores)(delayed(doAnalysis)
		(names, list_obsids,redshift,idx=np.arange(batch,batch +objectsPerJob,dtype=int))
		for batch in batchesList)

def main():
	# start logging
	createLog()

	logging.info('Starting Chandra Xpipe v3')
	
	# get initial time
	total_t0 = time()

	# check the process mode (pre-process, process and analysis)
	preProcess = isModeSet('preProcess')
	imaging = isModeSet('imaging')
	preAnalysis = isModeSet('preAnalysis')
	analysis = isModeSet('analysis')
	parallel = isOperationSet("parallel",section='Mode')

	# get the catalog input
	inPath = getPath(type="outputPath")
	dataDownPath = getPath(type="dataDownPath")
	catInFile = getConfig("Files","catInputFile")

	# define path name
	if (not inPath or not catInFile):
		logging.critical("Can't continue without either inputPath or catInFile defined! Exiting.")
		exit()

	# read the catalog input
	colNames = getColsName()
	print("ColNames:",colNames)
	input_dict = getCat.read_inputCat(catInFile, colNames=colNames)
	
	# define some variables
	names = input_dict['ID']
	list_obsids = input_dict['obsids']
	redshift = input_dict['redshift']

	# starting parallel setup if the case
	# if parallel:
	section = "Parallel"
	# get parallel information
	batchStart = int(getConfig(section, "batchStart"))
	batchMax   = int(getConfig(section, "batchMax"))
	nJobs 	   = int(getConfig(section, "nJobs"))
	nCores 	   = int(getConfig(section, "nCores"))
	
	## Runing pre-process mode
	if preProcess:
		preProcess_t0 = time()
		
		# switch to download outdir
		currentPath=os.getcwd()
		os.chdir(dataDownPath)

		if parallel:
			logging.info('Starting parallel pre-process operation.')
			
			# run preProcess
			parallelPreProcess(list_obsids,batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)
			
			# save time 
			preProcessTime = time() - preProcess_t0
			preProcessMsg = "Pre-Process (parallel) time: {}s".format(preProcessTime)
		
		else:
			logging.info('Starting single-mode pre-process.')
			indices = np.arange(batchStart,batchMax,dtype=int) ## Run for the first object of the input file
			doPreProcess(list_obsids,idx=indices)
			
			# save time 
			preProcessTime = time() - preProcess_t0
			preProcessMsg = "Pre-Process time: {}s".format(preProcessTime)
		
		os.chdir(currentPath)
		logging.info(preProcessMsg)
	## end pre-process mode

	## Runing imaging mode
	if imaging:
		imaging_t0 = time()
		if parallel:
			logging.info('Starting parallel imaging operation.')

			# run preProcess
			parallelImaging(names,list_obsids,batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)

			# save time to do imaging
			imagingTime = time() - imaging_t0
			imagingMsg = "Imaging (parallel) time: {}s".format(imagingTime)
		else:
			logging.info('Starting single-mode imaging.')
			
			indices = np.arange(batchStart,batchMax,dtype=int) ## Run for the first object of the input file
			doImaging(names,list_obsids,idx=indices)

			# save time to do imaging
			imagingTime = time() - imaging_t0
			imagingMsg = "Imaging time: {}s".format(imagingTime)

		logging.info(imagingMsg)
		logging.info("Don't forget of check the point sources!")
	## end imaging mode
	
	## Runing analysis mode
	if preAnalysis:
		preAnalysis_t0 = time()
		if parallel:
			logging.info('Starting parallel preAnalysis operation.')

			# run preProcess
			parallelPreAnalysis(names,list_obsids,redshift,batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)

			# save time to do imaging
			preAnalysisTime = time() - preAnalysis_t0
			preAnalysisMsg = "PreAnalysis (parallel) time: {}s".format(preAnalysisTime)
		else:
			logging.info('Starting single-mode preAnalysis.')

			indices = np.arange(batchStart,batchMax,dtype=int) ## Run for the first object of the input file
			doPreAnalysis(names,list_obsids,redshift,idx=indices)
			
			# save time to do imaging
			preAnalysisTime = time() - preAnalysis_t0
			preAnalysisMsg = "PreAnalysis (single) time: {}s".format(preAnalysisTime)
		
		logging.info(preAnalysisMsg)
	## Runing analysis mode
	if Analysis:
		Analysis_t0 = time()
		if parallel:
			logging.info('Starting parallel Analysis operation.')

			# run preProcess
			parallelAnalysis(names,list_obsids,redshift,batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)

			# save time to do imaging
			AnalysisTime = time() - Analysis_t0
			AnalysisMsg = "Analysis (parallel) time: {}s".format(AnalysisTime)
		else:
			logging.info('Starting single-mode Analysis.')

			indices = np.arange(batchStart,batchMax,dtype=int) ## Run for the first object of the input file
			doAnalysis(names,list_obsids,redshift,idx=indices)

			# save time to do imaging
			AnalysisTime = time() - Analysis_t0
			AnalysisMsg = "Analysis (parallel) time: {}s".format(AnalysisTime)

		logging.info(AnalysisMsg)
	
	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')


if __name__ == "__main__":
    main()