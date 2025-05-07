import sys, os
import time, datetime
import numpy as np

class DataAggregator:

    def __init__(self, basepath, substudy_list, mindataset=None, maxdataset=None):
        self.basepath = basepath
        self.substudy_list = substudy_list
        self.mindataset = mindataset
        self.maxdataset = maxdataset

    def load_all(self, verbose=False):

        ## Ensure the substudy_list is a list even if only a sigle substudy was provided
        runlistshape = np.shape(substudy_list)
        if len(runlistshape)==0:
            self.substudy_list = [self.substudy_list]
        Nsubstudies = len(self.substudy_list)
        if verbose: print("Loading data for '{Nsubstudies} substudies:")

        ## Find the full paths to each substudy
        fullpaths = [os.path.join(self.basepath,substudy) for substudy in self.substudy_list]
        if verbose: [print(" - ", path) for path in fullpaths]


