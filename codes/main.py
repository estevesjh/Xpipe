import numpy as np
import logging
import os,sys

from time import time
import configparser as ConfigParser
from joblib import Parallel, delayed
from astropy.table import Table

#personal modules
import preProcess
import imaging
import preAnalysis
import Analysis

import getCat

def getConfig(section, item, boolean=False,
		userConfigFile="/home/johnny/Documents/Xpipe/Xpipe_Config.ini"):

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

def saveOutput(names,values,out='output.txt'):
    '''Save an ouptut name value into a section in the output file
    '''
    if not os.path.isfile(out):
       new_data = Table()
       for col,val in zip(names,values):
           new_data[col] = [val]
    else:
        new_data = Table.read(out,format='ascii',delimiter=',')
        old_cols = new_data.colnames
        
        ## if columns does not exists
        notCommonCols = [element for element in names if element in old_cols]
        print(notCommonCols)
        if len(notCommonCols)>0:
            new_data.add_row(new_data[-1])
            for col,val in zip(names,values):
                new_data[col][-1] = val
        else:
            for col,val in zip(names,values):
                new_data[col] = val
    ## save
    new_data.write(out,format='ascii',delimiter=',',overwrite=True)
	
def getOutput(item, idx=-1, outFile="./log.txt"):
	'''Read the log file and return an item from a given index (idx)
	'''
	if not os.path.isfile(outFile):
		return -1
	else:
		try:
			data = Table.read(outFile,format='ascii',delimiter=',')
			value = float(data[item][idx])
			return value
		except:
			return -1

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

def getCenter(outputFile,peak=False,idx=-1):
	if peak:
		ra = getOutput("RA_PEAK",idx=idx,outFile=outputFile)		## Take the luminosity center
		dec = getOutput("DEC_PEAK",idx=idx,outFile=outputFile)		## Take the luminosity center
	else:
		ra = getOutput("RA",idx=idx,outFile=outputFile)		## Take the luminosity center
		dec = getOutput("DEC",idx=idx,outFile=outputFile)		## Take the luminosity center
	
	center = [ra,dec]
	return center

def writeStringToFile(fileName, toBeWritten):
    # create file if it does not exist
    if not os.path.isfile(fileName):
        with open(fileName, 'w') as f:
            f.write('{toBeWritten}\n'.format(toBeWritten=toBeWritten) )
    # else, append to file
    else:
        with open(fileName, 'a') as f:
            f.write( '{toBeWritten}\n'.format(toBeWritten=toBeWritten) )

def checkDir(path,name):
	outObjectPath = os.path.join(path,name)
	if not os.path.exists(outObjectPath):
		os.makedirs(outObjectPath)
	return os.path.relpath(outObjectPath)

def checkOutputFile(path,name):
	outFile = os.path.join(path,name)
	if not os.path.exists(path):
		os.mkdir(path)
	if not os.path.isfile(outFile):
		toBeWritten = '# Xpipe ouput: main.py \n'
		writeStringToFile(outFile, toBeWritten)
	return outFile

def doPreProcess(list_obsids,idx=np.array([0,1],dtype=int)):
	print(30*'--')
	print('idx:',idx)
	
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
	print(30*'--')
	print('Imaging:',idx)
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
		name = str(name).replace(" ","")		## Replace white spaces
		outObjectPath = checkDir(outPath,name)

		# tasks
		if fluxImage:
			print('fluxImage({})'.format(obsid))
			imaging.fluxImage(obsid,blankField=blankField,energy=band,binsize=binSize,pathData=dataDownPath,outdir=outObjectPath,clobber=False)
	
		if pointSources:
			print('pointSources({})'.format(obsid))
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

	## band
	band = getConfig('Imaging', "energy")
	binSize = int(getConfig('Imaging', "binSize"))
	elo, ehi, expMap = band.split(':')

	for (obsid,name,z) in zip(list_obsid_cut,name_cut,z_cut): # it runs for one object per time
		# name = str(name).replace(" ","")		## Replace white spaces
		outObjectPath = checkDir(outPath,name)
		obsid_str, obsid_lis = preAnalysis.checkObsid(obsid)
		print('name {} at redhsift {}'.format(name,z))
		
		outputFile = os.path.join(outObjectPath,'log.txt')
		
		# event files
		evt_lis = [os.path.join(dataDownPath,'{}'.format(obs),'repro',"{}_evt_gti.fits".format(obs)) for obs in obsid_lis]
		# region files
		ps = os.path.join(outObjectPath,'ps.reg')
		# imaging files
		
		# count_img = os.path.join(outObjectPath,'sb_{}-{}_thresh.img'.format(elo,ehi))
		# count_mask_img = os.path.join(outObjectPath,'sb_{}-{}_thresh_mask.img'.format(elo,ehi))
		# emap = os.path.join(outObjectPath,'sb_{}-{}_thresh.expmap'.format(elo,ehi))
		
		count_img = os.path.join(outObjectPath,'sb_thresh.img')
		count_mask_img = os.path.join(outObjectPath,'sb_thresh_mask.img')
		bg_img = os.path.join(outObjectPath,'sb_blank_particle_bgnd.img')
		emap = os.path.join(outObjectPath,'sb_thresh.expmap')
		
		if maskPointSources:
			preAnalysis.maskPointSources(count_img,psreg=ps,clobber=True)
			for evt in evt_lis:
				preAnalysis.maskPointSources(evt,psreg=ps,clobber=True)

		if centerX:
			'''If finds the X-ray luminosity centroid and the X-ray peak in a 500kpc radius
			'''
			radii=500 #kpc
			center = preAnalysis.centerX(count_mask_img,evt_lis[0],z,radius=radii,outdir=outObjectPath)
		else:
			center = getCenter(outputFile)

		if radialProfile:
			rbkgPhysical = getOutput("rbkgPhysical",outFile=outputFile)
			rmaxPhysical = getOutput("rmaxPhysical",outFile=outputFile)
			if (rbkgPhysical>0)&(rmaxPhysical>0):
				preAnalysis.radialProfile(count_mask_img,bg_img,emap,z,center,rbkg=rbkgPhysical,rmax=rmaxPhysical,binf=binSize,outdir=outObjectPath)
			else:
				preAnalysis.radialProfile(count_mask_img,bg_img,emap,z,center,binf=binSize,outdir=outObjectPath)
			# # proDir = preAnalysis.getDir('profile',outObjectPath)
			# # rprof_file = os.path.join(proDir,"broad_rprofile_binned.fits")
			# if not os.path.isfile(rprof_file):
			

		if arf_rmf:
			# preAnalysis.arf_rmf(obsid_str,center,z,radius=1500,outdir=outObjectPath,downPath=dataDownPath)
			preAnalysis.arf_rmf(obsid_str,center,z,radius=2000,outdir=outObjectPath,downPath=dataDownPath)

def doAnalysis(names,list_obsids,redshift,idx=np.array([0,1],dtype=int)):
	# check the tasks
	# fitSB = isOperationSet("fitSB",section='Analysis')
	# temperatureX = isOperationSet("temperatureX",section='Analysis')
	massX = isOperationSet("massX",section='Analysis')	
	csb = isOperationSet("csb",section='Analysis')
	centroidShift = isOperationSet("centroidShift",section='Analysis')
	errorCenterX = isOperationSet("errorCenterX",section='Analysis' )
	
	# get the download outdir
	dataDownPath = getPath(type="dataDownPath")
	outPath = getPath(type="outputPath")

	# get the analysis parameters
	model = getConfig('Analysis', "Model")

	# take a sub-sample
	list_obsid_cut = list_obsids[idx]
	name_cut = names[idx]
	z_cut = redshift[idx]

	## band
	band = getConfig('Imaging', "energy")
	binSize = int(getConfig('Imaging', "binSize"))
	elo, ehi, expMap = band.split(':')

	for (obsid,name,z) in zip(list_obsid_cut,name_cut,z_cut): # it runs for one object per time
		# name = str(name).replace(" ","")		## Replace white spaces
		obsid_str, obsid_lis = preAnalysis.checkObsid(obsid)
		print('name {} at redhsift {}'.format(name,z))

		outObjectPath = checkDir(outPath,name)
		outputFile = os.path.join(outObjectPath,'log.txt')
		center = getCenter(outputFile)
		center_peak = getCenter(outputFile,peak=True)
		
		rmaxPhysical = getOutput("rmaxPhysical",outFile=outputFile)
		rbkgPhysical = getOutput("rbkgPhysical",outFile=outputFile)
		
		#files
		profile = os.path.join(outObjectPath,"profile","broad_rprofile_binned.fits")
		count_img = os.path.join(outObjectPath,'sb_{}-{}_thresh.img'.format(elo,ehi))
		# count_img = os.path.join(outObjectPath,'sb_thresh.img')
		ps = os.path.join(outObjectPath,'ps.reg')

		betaFile = os.path.join(outObjectPath,model+'.txt')
		if os.path.isfile(betaFile):
			betavec = np.loadtxt(betaFile)
			try:
				betapars = betavec[-1,:]
			except:
				betapars = betavec
			
			kT = getOutput("kT",outFile=outputFile)
			r500 = getOutput("R500",outFile=outputFile)
			Mg500 = getOutput("Mg500",outFile=outputFile)
			M500 = getOutput("M500",outFile=outputFile)

			sb_plot_dir = './check/sb/'
			Analysis.makePlotBeta(profile,betapars,name,rbkg=0.492*rbkgPhysical,model='modBeta',outdir=sb_plot_dir)
		else:
			kT, r500, Mg500, M500, betapars  = Analysis.massX(obsid_lis,z,center,profile,kT_0=5,rbkg=rbkgPhysical,r0=500,model=model,name=name,outdir=outObjectPath,dataDownPath=dataDownPath)
	
		if csb:
			csb = Analysis.csb(betapars,r500,z,outdir=outObjectPath)
		else:
			csb = getOutput("csb",outFile=outputFile)

		w = getOutput("w",outFile=outputFile)
		# if centroidShift & (w<1):
			# if w<0:
		if centroidShift:
			w,werr = Analysis.centroidShift(count_img,center_peak,r500,rmaxPhysical,z,outdir=outObjectPath)
		else:
			w = getOutput("w",outFile=outputFile)
			werr = getOutput("werr",outFile=outputFile)
			
		errX = getOutput("errorCenter",outFile=outputFile)
		if errorCenterX & (errX<0):
			Xra, Xdec, errX, Xra_peak, Xdec_peak = Analysis.errorCenterX(count_img,center,ps,z,radius=r500,outdir=outObjectPath)
		else:
			center_peak = getCenter(outputFile,peak=True)
			Xra_peak, Xdec_peak = center_peak
			Xra, Xdec = center
		
		cols = ['Name', 'Xra', 'Xdec', 'errX', 'redshift', 'kT', 'r500', 'Mg500', 'M500',
			    'w', 'werr', 'csb', 'Xra_peak', 'Xdec_peak']
		
		values = [name,round(Xra,6),round(Xdec,6),round(errX,1),round(z,3),round(kT,1),round(r500),round(Mg500,2),round(M500,2),
				 round(w,4),round(werr,4),round(csb,3),round(Xra_peak,6),round(Xdec_peak,6)]

		outCatFile = os.path.join(outObjectPath,'results.txt')
		saveOutput(cols,values,out=outCatFile)

def doMain(names,list_obsids,redshift,indices=np.arange(0,1)):
	
    # get the catalog input
	dataDownPath = getPath(type="dataDownPath")

	# define path name
	if (not dataDownPath):
		logging.critical("Can't continue without either inputPath defined! Exiting.")
		exit()

    # check the process mode (pre-process, process and analysis)
	preProcess = isModeSet('preProcess')
	imaging = isModeSet('imaging')
	preAnalysis = isModeSet('preAnalysis')
	Analysis = isModeSet('analysis')

    ## Runing pre-process mode
	if preProcess:
		preProcess_t0 = time()
		
		# switch to download outdir
		currentPath=os.getcwd()
		os.chdir(dataDownPath)
		
		logging.info('Starting parallel pre-process operation.')
		doPreProcess(list_obsids,idx=indices)
		
		# save time 
		preProcessTime = time() - preProcess_t0
		preProcessMsg = "Pre-Process (parallel) time: {}s".format(preProcessTime)

		os.chdir(currentPath)
		logging.info(preProcessMsg)
    ## end pre-process mode

    ## Runing imaging mode
	if imaging:
		imaging_t0 = time()
		logging.info('Starting parallel imaging operation.')

		# run preProcess
		doImaging(names,list_obsids,idx=indices)

		# save time to do imaging
		imagingTime = time() - imaging_t0
		imagingMsg = "Imaging (parallel) time: {}s".format(imagingTime)

		logging.info(imagingMsg)
		logging.info("Don't forget of check the point sources!")
		## end imaging mode

    ## Runing analysis mode
	if preAnalysis:
		preAnalysis_t0 = time()
		logging.info('Starting parallel preAnalysis operation.')

		# run preAnalysis
		doPreAnalysis(names,list_obsids,redshift,idx=indices)

		# save time to do imaging
		preAnalysisTime = time() - preAnalysis_t0
		preAnalysisMsg = "PreAnalysis (parallel) time: {}s".format(preAnalysisTime)

		logging.info(preAnalysisMsg)
		## end preAnalysis mode

    ## Runing analysis mode
	if Analysis:
		Analysis_t0 = time()
		logging.info('Starting parallel Analysis operation.')

        # run Analysis
		doAnalysis(names,list_obsids,redshift,idx=indices)

        # save time to do imaging
		AnalysisTime = time() - Analysis_t0
		AnalysisMsg = "Analysis (parallel) time: {}s".format(AnalysisTime)
    
		logging.info(AnalysisMsg)
		## end analysis mode

def parallelTrigger(names,list_obsids,redshift,batchStart=0,batchMax=20, nJobs=10, nCores=4):
	objectsPerJob = int((batchMax-batchStart)/nJobs)
	objectsPerJob = np.where(objectsPerJob>1,objectsPerJob,1)
	batchesList = np.arange(batchStart, batchMax, objectsPerJob, dtype=int)

	logging.info('Calling parallelism')
	Parallel(n_jobs=nCores)(delayed(doMain)
		(names, list_obsids,redshift,indices=np.arange(batch,batch +objectsPerJob,dtype=int))
		for batch in batchesList)

def main(batchStart=0,batchMax=None):
	# start logging
	createLog()
	
	# get initial time
	total_t0 = time()

	catInFile = getConfig("Files","catInputFile")
    # define path name
	if (not catInFile):
		logging.critical("Can't continue without either catInFile defined! Exiting.")
		exit()

	# read the catalog input
	colNames = getColsName()
	print("ColNames:",colNames)
	input_dict = getCat.read_inputCat(catInFile, colNames=colNames)
	
	# define some variables
	names = input_dict['ID']
	list_obsids = input_dict['obsids']
	redshift = input_dict['redshift']

	parallel = isOperationSet("parallel",section='Mode')
	# starting parallel setup if the case
	if parallel:
		section = "Parallel"
		# get parallel information
		batchStart = int(getConfig(section, "batchStart"))
		batchMax   = int(getConfig(section, "batchMax"))
		nJobs 	   = int(getConfig(section, "nJobs"))
		nCores 	   = int(getConfig(section, "nCores"))
	
		parallelTrigger(names,list_obsids,redshift,batchStart=batchStart,
		batchMax=batchMax, nJobs=nJobs, nCores=nCores)
	else:
		if batchMax is None:
			batchMax = len(names)
		doMain(names,list_obsids,redshift,indices=np.arange(batchStart,batchMax,1))

	# # save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')

if __name__ == "__main__":
	try:
		main(batchStart=float(sys.argv[1]),batchMax=float(sys.argv[2]))
	except:
		main()

