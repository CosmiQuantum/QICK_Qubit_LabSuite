import os
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

    def find_all_datasets(self, verbose=False):
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

        ## Now get datetimes of everything for comparison
        run_ts = np.array([time.mktime(datetime.strptime(run, DATETIME_FMT).timetuple()) for run in runlist])
        run_idx = np.argwhere( (run_ts>=self.mintimestamp) & (run_ts<=self.maxtimestamp) )
        ## Have to de-dimensionalize the data, since this is making a list of single-item lists...
        goodruns = np.array([run[0] for run in runlist[run_idx]])
        goodpaths = np.array([path[0] for path in pathlist[run_idx]])

        ## Save the good runs and their respective paths
        self.datasets = goodruns
        self.datapaths = goodpaths
        self.ndatasets = len(goodruns)
        return goodruns, goodpaths 

    def run_analysis(AnalysisClass, qubit_index, analysis_params={}, datasets=None, datapaths=None, verbose=False):

        ## Check the datasets and paths
        if (datasets is not None) and (datapaths is not None):
            if len(datasets) != len(datapaths):
                raise Exception("ERROR: provided datasets and datapaths are different lenghts.")
        else: ## At least one is None
            datasets = self.datasets
            datapaths = self.datapaths
            ndatasets = self.ndatasets

        ## Confirm the presence of required keys in analysis parameters
        required_keys = ["name"]
        optional_keys = ["folder"]
        for req_key in required_keys:
            if req_key not in analysis_params.keys(): 
                raise KeyError("ERROR: required keys not found in analysis_params: "+",".join(required_keys))

        ## Create a container for the results, one key per dataset
        ana_result = { dataset:{} for dataset in datasets}

        ## Now for each dataset, run the analysis
        for i,(dataset,datapath) in enumerate(zip(datasets,datapaths)):

            ## Now instantiate an analysis instance for this dataset
            anaClass = AnalysisClass(datapath, dataset, qubit_index, 
                folder = "study_data" if ("folder" not in analysis_params.keys()) else analysis_params["folder"], 
                expt_name = analysis_params["name"], 
                ana_params = analysis_params )

            ## Load the required data for this analysis, and run it
            _ = anaClass.load_all() 

            ana_result[dataset] = anaClass.run_analysis(verbose=verbose)

            ## Free some memory
            anaClass.cleanup(clear_results=True)
            del anaClass

        self.result = ana_result
        return ana_result

        






