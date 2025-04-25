import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from .fit_functions import exponential
from ..utils.ana_utils  import rotate_and_threshold
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .plotting import plot_shots

class t1:

    def __init__(self,data_dir, dataset, QubitIndex, theta, threshold, folder="study_data", expt_name ="t1_ge", thresholding = True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder
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

        load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_files[0]), 'T1', save_r=1)
        delay_times = process_h5_data(load_data['T1'][self.QubitIndex].get('Delay Times', [])[0][0].decode())
        steps = len(delay_times)
        reps = int(len(process_h5_data(load_data['T1'][self.QubitIndex].get('I', [])[0][0].decode()))/steps)

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'T1', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['T1'][self.QubitIndex].get('Dates', [])[0][0]))

            I_shots.append(np.array(process_h5_data(load_data['T1'][self.QubitIndex].get('I', [])[0][0].decode())).reshape([reps, steps]))
            Q_shots.append(np.array(process_h5_data(load_data['T1'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape([reps, steps]))

        return dates, n, delay_times, steps, reps, I_shots, Q_shots

    def plot_shots(self, I_shots, Q_shots, delay_times, n, round=0, idx=10):
        # print(np.shape(I_shots))

        this_I = I_shots[round][:,idx]
        this_Q = Q_shots[round][:,idx]

        # print(np.shape(this_I))

        i_new, q_new, states = rotate_and_threshold(this_I, this_Q, self.theta, self.threshold)

        infodict = {
            'dataset': self.dataset,
            'qubit': self.QubitIndex,
            'round': round,
            'nrounds': n,
            'delay_time': np.round(delay_times[idx],2)
        }

        fig, ax = plot_shots(i_new, q_new, states, param_dict=infodict)
        del this_I, this_Q, i_new, q_new, states, infodict

    def process_shots(self, I_shots, Q_shots, n, steps):

        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][:, idx]
                this_Q = Q_shots[round][:, idx]

                i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
                q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)
                if self.thresholding:
                    states = (i_new > self.threshold)
                else:
                    states = np.mean(i_new)
                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def t1_fit(self, signal, delay_times, round, n, plot=False):
        ## >TO DO< Replace this (or integrate) with fitting.t1_fit
        # Initial guess for parameters
        q1_a_guess = np.max(signal) - np.min(signal)  # Initial guess for amplitude (a)
        q1_b_guess = 0  # Initial guess for time shift (b)
        q1_c_guess = (delay_times[-1] - delay_times[0]) / 5  # Initial guess for decay constant (T1)
        q1_d_guess = np.min(signal)  # Initial guess for baseline (d)

        # Form the guess array
        q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

        # Define bounds to constrain T1 (c) to be positive, but allow amplitude (a) to be negative
        lower_bounds = [-np.inf, -np.inf, 0, -np.inf]  # Amplitude (a) can be negative/positive, but T1 (c) > 0
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # No upper bound on parameters

        # Perform the fit using the 'trf' method with bounds
        q1_popt, q1_pcov = curve_fit(exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
        q1_fit_exponential = exponential(delay_times, *q1_popt)

        # Extract T1 and its error
        T1_est = q1_popt[2]  # Decay constant T1
        T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

        if plot:
            fig, ax = plt.subplots()
            ax.plot(delay_times, signal, label='data')
            ax.plot(delay_times, q1_fit_exponential, label='exponential')
            ax.set_xlabel('Delay Time [us]')
            ax.set_ylabel('P(e)')
            ax.legend()
            ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n}: t1_ge = {T1_est:.3f} +/- {T1_err} us')
            #plt.show(block=False)


        return q1_fit_exponential, T1_err, T1_est

    def get_all_t1(self, delay_times, p_excited, n):

        t1s = []
        t1_errs = []

        for round in np.arange(n):
            p_excited_in_round = p_excited[round][:]
            q1_fit_exponential, T1_err, T1_est = self.t1_fit(p_excited_in_round, delay_times, round, n, plot=False)
            t1s.append(T1_est)
            t1_errs.append(T1_err)

        return t1s, t1_errs

    def get_t1_in_round(self, delay_times, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round][:]
        q1_fit_exponential, T1_err, T1_est = self.t1_fit(p_excited_in_round, delay_times, round, n, plot=plot)

        return q1_fit_exponential, T1_err, T1_est

