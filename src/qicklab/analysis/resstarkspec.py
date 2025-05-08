import os, datetime
import numpy as np

## QICKLAB methods
from ..datahandling.datafile_tools import find_h5_files, load_h5_data, get_data_field
from ..utils.ana_utils  import rotate_and_threshold
from .plot_tools import plot_stark_simple
from .shot_tools import process_shots
from .stark_tools import gain2freq_resonator
from .AnalysisClass import AnalysisClass


class AnaResStarkSpec(AnalysisClass):

    required_ana_keys = ["theta", "threshold", "stark_constant"]
    optional_ana_keys = ["thresholding"]

    def __init__(self, data_dir, dataset, qubit_index, folder="study_data", expt_name="res_starkspec_ge", datagroup='starkSpec', ana_params={}):
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

        dates = []
        I_shots = []
        Q_shots = []
        P = []

        ## For each file in the dataset, load the data, pull specific data fields,
        ## and save it to the output dictonary.
        for i,h5_file in enumerate(h5_files):

            ## Load the selected H5 data into a dictionary, pull the frequency info from the first one
            load_data = load_h5_data(os.path.join(data_path, h5_file), self.datagroup, save_r=1)

            if i==0:
                gain_sweep = get_data_field(load_data, self.datagroup, self.qubit_index, 'Gain Sweep')
                steps = len(gain_sweep)
                reps = int(len(get_data_field(load_data, self.datagroup, self.qubit_index, 'I')) / steps)

            ## For every file add its I, Q, and P
            timestamps = get_data_field(load_data, self.datagroup, self.qubit_index, 'Dates')
            I_shots.append(get_data_field(load_data, self.datagroup, self.qubit_index, 'I', steps=steps, reps=reps))
            Q_shots.append(get_data_field(load_data, self.datagroup, self.qubit_index, 'Q', steps=steps, reps=reps))
            P.append(get_data_field(load_data, self.datagroup, self.qubit_index, 'P'))

            dates.append(datetime.datetime.fromtimestamp(load_data[self.datagroup][self.qubit_index].get('Dates', [])[0][0]))

        ## Save the necessary data to the dictionary
        analysis_data["dates"] = dates
        analysis_data["n"] = n
        analysis_data["gain_sweep"] = gain_sweep
        analysis_data["steps"] = steps
        analysis_data["reps"] = reps
        analysis_data["I"] = I_shots
        analysis_data["Q"] = Q_shots
        analysis_data["P"] = P

        ## Save the output dictionary to the class instance and then return it
        self.analysis_data = analysis_data
        return analysis_data

    def run_analysis(self, verbose=False):
        ## Create a container for the output
        analysis_result = {}

        ## For each file in the dataset, use the information saved in self.analysis data to
        ## do something, and save it to the output dictonary.

        rstark_p_excited = self.process_shots(
            self.analysis_data["I"] , 
            self.analysis_data["Q"] , 
            self.analysis_data["n"] , 
            self.analysis_data["steps"] , 
            )

        rstark_freqs = self.gain2freq(self.analysis_data["gain_sweep"])

        analysis_result["rstark_p_excited"] = rstark_p_excited
        analysis_result["rstark_freqs"]  = rstark_freqs

        ## Save the output dictionary to the class instance and then return it
        self.analysis_result = analysis_result
        return analysis_result

    def plot_shots(self, I_shots, Q_shots, gains, n, round=0, idx=10):
        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        i_new, q_new, states = rotate_and_threshold(this_I, this_Q, self.ana_params["theta"], self.ana_params["threshold"])

        title = (f'dataset {self.dataset} qubit {self.qubit_index} round {round + 1} of {n}: ' +
                 f'rotated I,Q shots for res_stark_spec at gain: {np.round(gains[idx],2)}')

        _, _ = plot_shots(i_new, q_new, states, rotated=True, title=title)

    def process_shots(self, I_shots, Q_shots, n, steps):
        return process_shots(I_shots, Q_shots, n, steps, 
            self.ana_params["theta"], 
            self.ana_params["threshold"], 
            thresholding=True if "thresholding" not in self.ana_params.keys() else self.ana_params["thresholding"])

    def gain2freq(self, gains):
        return gain2freq_resonator(gains, self.ana_params["stark_constant"])

    def get_p_excited_in_round(self, gains, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round]

        title = (f'dataset {self.dataset} qubit {self.qubit_index + 1} round {round + 1} of {n}: ' + 
                  ' resonator stark spectroscopy')
        if plot: _, _ = plot_stark_simple(gains, self.gain2freq(gains), p_excited_in_round, title=title)

        return p_excited_in_round


def resstarkspec_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, res_stark_constant=-17, threshold=-1285.08904, theta=0.17681, selected_round=[10, 73]):
    ana_params = {
        "theta": theta,
        "threshold": threshold,
        "stark_constant": res_stark_constant,
        "thresholding": False
    }

    rstark = AnaResStarkSpec(data_dir, dataset, QubitIndex, ana_params=ana_params)
    data = rstark.load_all()
    result = rstark.run_analysis(verbose=True)
    rstark.cleanup()
    del rstark
    return data["dates"], data["n"], data["gain_sweep"], result["rstark_p_excited"], result["rstark_freqs"]