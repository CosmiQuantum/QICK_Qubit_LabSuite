import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

## QICKLAB methods
from ..datahandling.datafile_tools import find_h5_files, load_h5_data, process_h5_data, get_data_field
from ..utils.ana_utils  import rotate_and_threshold
from .plot_tools import plot_shots
from .AnalysisClass import AnalysisClass

class AnaAutoThreshold(AnalysisClass):

    required_ana_keys = []
    optional_ana_keys = ["idx", "plot"]

    def __init__(self, data_dir, dataset, qubit_index, folder="study_data", expt_name="qspec_ge", datagroup='starkSpec', ana_params={}):
        
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

        ## Determine which file we care aboout
        idx = 0 if "idx" not in self.ana_params.keys() else self.ana_params["idx"]

        ## Load the selected H5 data into a dictionary
        load_data = load_h5_data(os.path.join(data_path, h5_files[idx]), self.datagroup, save_r=1)

        print(load_data[self.datagroup].keys(), ":", load_data[self.datagroup][self.qubit_index].keys())

        ## Pull the gain sweep info and determine how many steps there are
        gain_sweep = get_data_field(load_data, self.datagroup, self.qubit_index, 'Gain Sweep')

        # gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        step_idx = int(steps/2)
        del gain_sweep

        ## Load the I and Q data
        I_shots = get_data_field(load_data, self.datagroup, self.qubit_index, 'I')
        Q_shots = get_data_field(load_data, self.datagroup, self.qubit_index, 'Q')
        reps = int(len(I_shots) / steps)

        I_shots = I_shots.reshape([steps, reps])
        Q_shots = Q_shots.reshape([steps, reps])

        ## Save the necessary data to the dictionary
        analysis_data[h5_files[idx]] = {
                "I": I_shots[step_idx],
                "Q": Q_shots[step_idx],
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

            theta, threshold, i_new, q_new = self.get_threshold(
                self.analysis_data[k]["I"] , 
                self.analysis_data[k]["Q"] , 
                plot=False if "plot" not in self.ana_params.keys() else self.ana_params["plot"], 
                verbose=verbose)

            analysis_result[k] = {
                "theta": theta,
                "threshold": threshold,
                "I_new": i_new,
                "Q_new": q_new,
            }

        ## Save the output dictionary to the class instance and then return it
        self.analysis_result = analysis_result
        return analysis_result

    def get_threshold(self, I_shots, Q_shots, plot=True, verbose=False):
        ## Setup and evaluate the clustering algorithm to find the unrotated centroids
        kmeans = KMeans(n_clusters=2).fit(np.transpose([I_shots, Q_shots]))
        ye = kmeans.cluster_centers_[1,1]
        xe = kmeans.cluster_centers_[1,0]
        yg = kmeans.cluster_centers_[0,1]
        xg = kmeans.cluster_centers_[0,0]
        if verbose: print(kmeans.cluster_centers_[:,0])

        ## Find and apply the rotation angle to optimize readout in the "i_new" direction
        theta = -np.arctan2((ye - yg), (xe - xg))
        i_new, q_new, _ = rotate_and_threshold(I_shots, Q_shots, theta, 0.0)

        ## Setup and evaluate the clustering algorithm to find the rotated centroids
        ## Use that to evaluate the threshold and classify the states
        kmeans_new = KMeans(n_clusters=2).fit(np.transpose([i_new, q_new]))
        threshold = np.mean([kmeans_new.cluster_centers_[1,0],kmeans_new.cluster_centers_[0,0]])
        state = kmeans_new.predict(np.transpose([i_new, q_new]))

        if plot:
            fig, ax = plt.subplots(1,2, layout='constrained')

            plot_shots(I_shots, Q_shots, state, title='unrotated I,Q', rotated=False, ax=ax[0])
            ax[0].scatter(xg, yg, c='k')
            ax[0].scatter(xe, ye, c='k')

            plot_shots(i_new, q_new, state, title=f'rotated I,Q; theta={np.round(theta,2)}, threshold={np.round(threshold,2)}', 
                rotated=True, ax=ax[1])
            ax[1].plot([threshold, threshold], [np.min(q_new), np.max(q_new)], 'k:')

        return theta, threshold, i_new, q_new


def auto_threshold_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, selected_round=[10, 73]):
    
    ana_params = {
        "idx" : 0,
        "plot": True,
    }

    auto = AnaAutoThreshold(data_dir, dataset, QubitIndex, ana_params=ana_params)
    data = auto.load_all()
    result = auto.run_analysis(verbose=True)
    auto.cleanup()
    del auto
    return result
