import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .plotting import plot_resstark_simple


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
        ## >TO DO< This is duplicated from t1 -- how to condense in a sensical way?
        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        i_new, q_new, states = rotate_and_threshold(this_I, this_Q, self.theta, self.threshold)

        title = (f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: ' +
                 f'rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[idx],2)} us')

        _, _ = plot_shots(i_new, q_new, states, rotated=True, title=title)

    def process_shots(self, I_shots, Q_shots, n, steps):
        ## >TO DO< This is duplicated from t1 -- how to condense in a sensical way?
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

        title = (f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n}: ' + 
                  ' resonator stark spectroscopy')
        if plot: _, _ = plot_stark_simple(gains, self.gain2freq(gains), p_excited_in_round, title=title)

        return p_excited_in_round



def resstarkspec_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, res_stark_constant=-17, threshold=1285.08904, theta=-2.96478, selected_round=[10, 73]):
    rstark = resstarkspec(data_dir, dataset, QubitIndex, res_stark_constant, theta, threshold)
    rstark_dates, rstark_n, rstark_gains, rstark_steps, rstark_reps, rstark_I_shots, rstark_Q_shots, rstark_P = rstark.load_all()
    rstark_p_excited = rstark.process_shots(rstark_I_shots, rstark_Q_shots, rstark_n, rstark_steps)
    rstark_freqs = rstark.gain2freq(rstark_gains)

    outdata = {}
    for rnd in selected_round:
        outdata[rnd] = rstark.get_p_excited_in_round(rstark_gains, rstark_p_excited, rstark_n, rnd, plot=True)
    return outdata