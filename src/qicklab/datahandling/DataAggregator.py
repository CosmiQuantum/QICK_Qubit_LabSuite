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
        for path in fullpaths:

            ## Grab the dataset IDs in this substudy
            runlist += os.listdir(path) 

        runlist = np.sort(runlist)
        if verbose: print(runlist)

        ## Now get datetimes of everything for comparison
        min_dt = datetime(2000,1,1,0,0,0) if self.mindataset is None else datetime.strptime(self.mindataset, DATETIME_FMT)
        max_dt = datetime(2100,1,1,0,0,0) if self.maxdataset is None else datetime.strptime(self.mindataset, DATETIME_FMT)
        run_dt = [datetime.strptime(run, DATETIME_FMT) for run in runlist]

        run_idx = np.argwhere( (run_dt>=min_dt) & (run_dt<=max_dt) )
        goodruns = runlist[run_idx]

        return goodruns




