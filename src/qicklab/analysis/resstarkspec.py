import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata


class resstarkspec:

    def __init__(self, data_dir, dataset, QubitIndex, stark_constant, theta, threshold, folder = "study_data", expt_name = "res_starkspec_ge", thresholding=True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.stark_constant = stark_constant
        self.folder = folder
        self.expt_name = expt_name
        self.theta = theta
        self.threshold = threshold
        self.thresholding = thresholding

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_shots = []
        Q_shots = []
        P = []

        load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_files[0]), 'starkSpec', save_r=1)
        gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        reps = int(len(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())) / steps)

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'starkSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['starkSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())).reshape(
                    [reps, steps]))
            Q_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape(
                    [reps, steps]))
            P.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('P', [])[0][0].decode())))

        return dates, n, gain_sweep, steps, reps, I_shots, Q_shots, P

    def plot_shots(self, I_shots, Q_shots, gains, n, round=0, idx=10):

        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
        q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)

        states = (i_new > self.threshold)

        fig, ax = plt.subplots()
        ax.scatter(i_new, q_new, c=states)
        ax.set_xlabel('I [a.u.]')
        ax.set_ylabel('Q [a.u.]')
        ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex +1} round {round + 1} of {n}: rotated I,Q shots for res_stark_spec at gain: {np.round(gains[idx],2)} us')
        #plt.show(block=False)

    def process_shots(self, I_shots, Q_shots, n, steps):

        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][:,idx]
                this_Q = Q_shots[round][:,idx]

                i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
                q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)
                if self.thresholding:
                    states = (i_new > self.threshold)
                else:
                    states = np.mean(i_new)
                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def gain2freq(self, gains):
        freqs = np.square(gains) * self.stark_constant
        return freqs

    def get_p_excited_in_round(self, gains, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round]

        if plot:
            fig, ax = plt.subplots(2,1, layout='constrained')
            fig.suptitle(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n} resonator stark spectroscopy')

            ax[0].plot(gains, p_excited_in_round)
            ax[0].set_xlabel('resonator stark gain [a.u.]')
            ax[0].set_ylabel('P(e)')

            ax[1].plot(self.gain2freq(gains), p_excited_in_round)
            ax[1].set_xlabel('stark shift [MHz]')
            ax[1].set_ylabel('P(e)')
            #plt.show(block=False)

        return p_excited_in_round

