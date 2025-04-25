import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata

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

    def get_threshold(self, I_shots, Q_shots, plot=True):
        kmeans = KMeans(n_clusters=2).fit(np.transpose([I_shots, Q_shots]))
        print(kmeans.cluster_centers_[:,0])
        ye = kmeans.cluster_centers_[1,1]
        xe = kmeans.cluster_centers_[1,0]
        yg = kmeans.cluster_centers_[0,1]
        xg = kmeans.cluster_centers_[0,0]

        theta = -np.arctan2((ye - yg), (xe - xg))
        i_new = I_shots * np.cos(theta) - Q_shots * np.sin(theta)
        q_new = I_shots * np.sin(theta) + Q_shots * np.cos(theta)

        kmeans_new = KMeans(n_clusters=2).fit(np.transpose([i_new, q_new]))
        threshold = np.mean([kmeans_new.cluster_centers_[1,0],kmeans_new.cluster_centers_[0,0]])
        state = kmeans_new.predict(np.transpose([i_new, q_new]))

        if plot:
            fig, ax = plt.subplots(1,2, layout='constrained')

            plot = ax[0]
            plot.scatter(I_shots, Q_shots, c=state)
            plot.set_xlabel('I [a.u.]')
            plot.set_ylabel('Q [a.u.]')
            plot.scatter(xg, yg, c='k')
            plot.scatter(xe, ye, c='k')
            plot.set_aspect('equal')
            plot.set_title('unrotated I,Q')

            plot = ax[1]
            plot.scatter(i_new, q_new, c=state)
            plot.set_xlabel('I [a.u.]')
            plot.set_ylabel('Q [a.u.]')
            plot.set_aspect('equal')
            plot.set_title(f'rotated I,Q; theta={np.round(theta,2)}, threshold={np.round(threshold,2)}')
            plot.plot([threshold, threshold], [np.min(q_new), np.max(q_new)], 'k:')

            #plt.show(block=False)


        return theta, threshold, i_new, q_new