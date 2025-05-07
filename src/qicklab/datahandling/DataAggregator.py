import os, glob
import time
import numpy as np

from datetime import datetime
from .datafile_tools import DATETIME_FMT

class DataAggregator:

    def __init__(self, basepath, substudy_list, mindataset=None, maxdataset=None):
        self.basepath = basepath
        self.substudy_list = substudy_list
        self.mindataset = mindataset
        self.maxdataset = maxdataset
        self.mintimestamp = time.mktime(datetime(2000,1,1,0,0,0).timetuple()) if self.mindataset is None else time.mktime(datetime.strptime(self.mindataset, DATETIME_FMT).timetuple())
        self.maxtimestamp = time.mktime(datetime(2100,1,1,0,0,0).timetuple()) if self.maxdataset is None else time.mktime(datetime.strptime(self.maxdataset, DATETIME_FMT).timetuple())

    def load_all(self, verbose=False):
        alldata = {}

        ## Ensure the substudy_list is a list even if only a sigle substudy was provided
        runlistshape = np.shape(self.substudy_list)
        if len(runlistshape)==0:
            self.substudy_list = [self.substudy_list]
        Nsubstudies = len(self.substudy_list)
        if verbose: print("Loading data for '{Nsubstudies} substudies:")

        ## Find the full paths to each substudy
        fullpaths = [os.path.join(self.basepath,substudy) for substudy in self.substudy_list]
        if verbose: [print(" - ", path) for path in fullpaths]

        ## Now for every substudy, find the total list of datasets contained
        runlist = []
        pathlist = []
        for path in fullpaths:

            ## Grab the dataset IDs in this substudy
            runlist += os.listdir(path) 
            pathlist += [path]*len(os.listdir(path))

        sortidx = np.argsort(runlist)
        runlist = np.array(runlist)[sortidx]
        pathlist = np.array(pathlist)[sortidx]
        if verbose: print("Runs:", runlist)
        if verbose: print("Paths:", pathlist)

        print(np.shape(pathlist))

        ## Now get datetimes of everything for comparison
        run_ts = np.array([time.mktime(datetime.strptime(run, DATETIME_FMT).timetuple()) for run in runlist])
        run_idx = np.argwhere( (run_ts>=self.mintimestamp) & (run_ts<=self.maxtimestamp) )
        goodruns = runlist[run_idx]
        goodpaths = np.array([path[0] for path in pathlist[run_idx]])

        print(np.shape(goodpaths))

        return goodruns, goodpaths 




