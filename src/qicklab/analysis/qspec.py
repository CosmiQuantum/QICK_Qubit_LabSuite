import os, datetime
import numpy as np

from scipy.optimize import curve_fit

## QICKLAB methods
from ..datahandling.datafile_tools import find_h5_files, load_h5_data, get_data_field
from .plot_tools import plot_qspec_simple
from .fit_tools import fit_lorenzian
from .AnalysisClass import AnalysisClass

class AnaQSpec(AnalysisClass):

    required_ana_keys = []
    optional_ana_keys = ["plot_idxs"]

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

        dates = []
        I = []
        Q = []

        ## For each file in the dataset, load the data, pull specific data fields,
        ## and save it to the output dictonary.
        for i,h5_file in enumerate(h5_files):

            ## Load the selected H5 data into a dictionary, pull the frequency info from the first one
            load_data = load_h5_data(os.path.join(data_path, h5_file), self.datagroup, save_r=1)
            if i==0: qspec_probe_freqs = get_data_field(load_data, self.datagroup, self.qubit_index, 'Frequencies')

            ## For every file add its I and Q
            timestamps = get_data_field(load_data, self.datagroup, self.qubit_index, 'Dates')
            I.append(get_data_field(load_data, self.datagroup, self.qubit_index, 'I'))
            Q.append(get_data_field(load_data, self.datagroup, self.qubit_index, 'Q'))

            dates.append(datetime.datetime.fromtimestamp(load_data[self.datagroup][self.qubit_index].get('Dates', [])[0][0]))

        ## Save the necessary data to the dictionary
        analysis_data["dates"] = dates
        analysis_data["n"] = n
        analysis_data["qspec_probe_freqs"] = qspec_probe_freqs
        analysis_data["I"] = I
        analysis_data["Q"] = Q

        ## Save the output dictionary to the class instance and then return it
        self.analysis_data = analysis_data
        return analysis_data

    def run_analysis(self, verbose=False):
        ## Create a container for the output
        analysis_result = {}

        ## For each file in the dataset, use the information saved in self.analysis data to
        ## do something, and save it to the output dictonary.

        qspec_freqs, qspec_errs, qspec_fwhms = self.get_all_qspec_freq(
            self.analysis_data["qspec_probe_freqs"] , 
            self.analysis_data["I"] , 
            self.analysis_data["Q"] , 
            self.analysis_data["n"] , 
            plot_idxs=False if "plot_idxs" not in self.ana_params.keys() else self.ana_params["plot_idxs"], 
            )

        analysis_result["qspec_freqs"] = qspec_freqs
        analysis_result["qspec_errs"]  = qspec_errs
        analysis_result["qspec_fwhms"] = qspec_fwhms

        ## Save the output dictionary to the class instance and then return it
        self.analysis_result = analysis_result
        return analysis_result


    def fit_qspec(self, I, Q, freqs):
        mag = np.sqrt(np.square(I) + np.square(Q))
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(mag)]
        qfreq, qfreq_err, fwhm, qspec_fit, qspec_fit_amp = fit_lorenzian(mag, freqs, freq_q)
        return qfreq, qfreq_err, fwhm, qspec_fit

    def get_qspec_freq_in_round(self, qspec_probe_freqs, I, Q, round, n, plot=False):
        thisI = I[round]
        thisQ = Q[round]

        qfreq, qfreq_err, fwhm, qspec_fit = self.fit_qspec(thisI, thisQ, qspec_probe_freqs)

        if plot:

            title = (f'dataset {self.dataset} qubit {self.qubit_index} round {round + 1} of {n}: ' +
                     f'rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[round],2)} us')

            _, _ = plot_qspec_simple(qspec_probe_freqs, thisI, thisQ, fitcurve=qspec_fit, title=title)

        return qfreq, qfreq_err, fwhm, qspec_fit

    def get_all_qspec_freq(self, qspec_probe_freqs, I, Q, n, plot_idxs=[]):
        qspec_freqs = np.zeros(n) #[]
        qspec_errs = np.zeros(n) #[]
        fwhms = np.zeros(n) #[]

        for round in np.arange(n):
            qfreq, qfreq_err, fwhm, _ = self.get_qspec_freq_in_round(qspec_probe_freqs, I, Q, round, n, plot=(round in plot_idxs))
            qspec_freqs[round] = qfreq
            qspec_errs[round] = qfreq_err
            fwhms[round] = fwhm

        return qspec_freqs.tolist(), qspec_errs.tolist(), fwhms.tolist()

def qspec_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, selected_round=[10, 73]):

    ana_params = {
        "plot_idxs": selected_round,
    }

    qspec_ge = AnaQSpec(data_dir, dataset, QubitIndex, ana_params=ana_params)
    data = qspec_ge.load_all()
    result = qspec_ge.run_analysis(verbose=True)
    qspec_ge.cleanup()
    del qspec_ge
    return data["dates"], result["qspec_freqs"]