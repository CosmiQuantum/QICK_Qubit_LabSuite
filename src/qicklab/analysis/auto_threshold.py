import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from ..utils.ana_utils  import rotate_and_threshold
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .plot_tools import plot_shots

class auto_threshold:
    def __init__(self, data_dir, dataset, QubitIndex, folder = "study_data", expt_name = "starkspec_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_sample(self, idx=0):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()

        load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_files[idx]), 'starkSpec', save_r=1)
        gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        step_idx = int(steps/2)
        reps = int(len(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())) / steps)
        I_shots = np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())).reshape([steps, reps])
        Q_shots = np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape([steps, reps])

        return I_shots[step_idx], Q_shots[step_idx]

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
    auto = auto_threshold(data_dir, dataset, QubitIndex)
    I_shots, Q_shots = auto.load_sample()
    return auto.get_threshold(I_shots, Q_shots, plot=True)
