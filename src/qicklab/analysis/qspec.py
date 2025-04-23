import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from .fit_functions import lorentzian
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata

class qspec:
    def __init__(self, data_dir, dataset, QubitIndex, folder="study_data", expt_name="qspec_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def fit_qspec(self, I, Q, freqs):
        mag = np.sqrt(np.square(I) + np.square(Q))
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(mag)]
        qfreq, qfreq_err, fwhm, qspec_fit = self.fit_lorenzian(mag, freqs, freq_q)
        return qfreq, qfreq_err, fwhm, qspec_fit

    def max_offset_difference_with_x(self,x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, mag, freqs, freq_q):
        ## >TO DO< Replace this (or integrate) with fitting.fit_lorenzian
        # Initial guesses for I and Q
        initial_guess = [freq_q, 1, np.max(mag), np.min(mag)]

        # First round of fits (to get rough estimates)
        params, _ = curve_fit(lorentzian, freqs, mag, p0=initial_guess)


        # Use these fits to refine guesses
        x_max_diff, max_diff = self.max_offset_difference_with_x(freqs, mag, params[3])
        initial_guess = [x_max_diff, 1, np.max(mag), np.min(mag)]

        # Second (refined) round of fits, this time capturing the covariance matrices
        params, cov = curve_fit(lorentzian, freqs, mag, p0=initial_guess)

        # Create the fitted curves
        fit = lorentzian(freqs, *params)

        # Calculate errors from the covariance matrices
        fit_err = np.sqrt(np.diag(cov))[0]

        # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
        mean = params[0]
        fwhm = 2 * params[1]

        # Calculate the amplitude differences from the fitted curves
        amp_fit = abs(np.max(fit) - np.min(fit))

        return mean, fit_err, fwhm, fit

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I = []
        Q = []

        load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_files[0]), 'QSpec', save_r=1)
        qspec_probe_freqs = process_h5_data(load_data['QSpec'][self.QubitIndex].get('Frequencies', [])[0][0].decode())

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['QSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('I', [])[0][0].decode())))
            Q.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('Q', [])[0][0].decode())))

        return dates, n, qspec_probe_freqs, I, Q

    def get_all_qspec_freq(self, qspec_probe_freqs, I, Q, n):
        qspec_freqs = []
        qspec_errs = []
        fwhms = []

        for round in np.arange(0,n):
            qfreq, qfreq_err, fwhm, qspec_fit = self.fit_qspec(I[round], Q[round], qspec_probe_freqs)
            qspec_freqs.append(qfreq)
            qspec_errs.append(qfreq_err)
            fwhms.append(fwhm)

        return qspec_freqs, qspec_errs, fwhms

    def get_qspec_freq_in_round(self, qspec_probe_freqs, I, Q, round, n, plot=False):
        thisI = I[round]
        thisQ = Q[round]

        qfreq, qfreq_err, fwhm, qspec_fit = self.fit_qspec(thisI, thisQ, qspec_probe_freqs)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(qspec_probe_freqs, np.sqrt(np.square(thisI) + np.square(thisQ)), label='data')
            ax.plot(qspec_probe_freqs, qspec_fit, label='lorentzian')
            ax.legend()
            ax.set_xlabel('qubit probe frequency [MHz]')
            ax.set_ylabel('I,Q magnitude [a.u.]')
            ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n} low-gain qspec: {np.round(qfreq,2)} +/- {np.round(qfreq_err,2)} MHz, fwhm: {np.round(fwhm,2)} MHz')
            #plt.show(block=False)

        return qfreq, qfreq_err, fwhm, qspec_fit
