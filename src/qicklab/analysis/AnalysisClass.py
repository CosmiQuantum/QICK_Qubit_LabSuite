import os
import numpy as np

from ..datahandling.datafile_tools import find_h5_files, load_h5_data, get_data_field


class AnalysisClass:

    required_ana_keys = []
    optional_ana_keys = ['datagroup']


    def __init__(self, data_dir, dataset, qubit_index, folder="study_data", expt_name="qspec_ge", datagroup='QSpec', ana_params={}):
        
        ## Save the arguments to the class
        self.data_dir = data_dir
        self.dataset = dataset
        self.datagroup = datagroup
        self.qubit_index = qubit_index
        self.expt_name = expt_name
        self.folder = folder

        ## Check the analysis params keys against the required keys
        for req_key in self.required_ana_keys:
            if req_key not in ana_params.keys(): 
                raise KeyError("ERROR: required keys not found in ana_params: "+",".join(self.required_ana_keys))
        self.ana_params = ana_params

    def load_all(self, verbose=False):
        ## Create a container for the output
        analysis_data = {}

        ## Find all the H5 files for this dataset
        h5_files, data_path, n = find_h5_files(self.data_dir, self.dataset, self.expt_name, folder=self.folder)

        ## For each file in the dataset, load the data, pull specific data fields,
        ## and save it to the output dictonary.
        for i,h5_file in enumerate(h5_files):
            load_data = load_h5_data(os.path.join(data_path, h5_file), self.datagroup)
            timestamps = get_data_field(load_data, self.datagroup, self.qubit_index, 'Dates')
            
            ## Save a data dictionary to the output dictionary, one key for each piece of data
            ## that will be needed later to perform the analysis
            analysis_data[h5_file] = {
                "timestamps": timestamps,
            }

        ## Save the output dictionary to the class instance and then return it
        self.analysis_data = analysis_data
        return analysis_data


    def run_analysis(self, verbose=False):
        ## Create a container for the output
        analysis_result = {}

        ## For each file in the dataset, use the information saved in self.analysis data to
        ## do something, and save it to the output dictonary.
        for k in self.analysis_data.keys():

            analysis_result[k] = {
                "result":self.analysis_data[k]["timestamps"][0],
            }

        ## Save the output dictionary to the class instance and then return it
        self.analysis_result = analysis_result
        return analysis_result


    def cleanup(self, clear_results=False):
        del self.analysis_data
        if clear_results: del self.analysis_result

