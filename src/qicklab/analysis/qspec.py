import os, datetime
import numpy as np

from scipy.optimize import curve_fit

from ..datahandling.datafile_tools import load_h5_data
from ..utils.data_utils import process_h5_data
# from ..utils.file_utils import load_from_h5_with_shotdata
from .plot_tools import plot_qspec_simple
from .fit_tools import fit_lorenzian

class qspec:
    def __init__(self, data_dir, dataset, QubitIndex, folder="study_data", expt_name="qspec_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I = []
        Q = []

        load_data = load_h5_data(os.path.join(data_path, h5_files[0]), 'QSpec', save_r=1)
        qspec_probe_freqs = process_h5_data(load_data['QSpec'][self.QubitIndex].get('Frequencies', [])[0][0].decode())

        for h5_file in h5_files:
            load_data = load_h5_data(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['QSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('I', [])[0][0].decode())))
            Q.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('Q', [])[0][0].decode())))

        return dates, n, qspec_probe_freqs, I, Q

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

            title = (f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: ' +
                     f'rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[idx],2)} us')

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

    qspec_ge = qspec(data_dir, dataset, QubitIndex)
    qspec_dates, qspec_n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_ge.load_all()
    qspec_freqs, qspec_errs, qspec_fwhms = qspec_ge.get_all_qspec_freq(qspec_probe_freqs, qspec_I, qspec_Q, qspec_n, plot_idxs=selected_round)
    start_time = qspec_dates[0]

    return qspec_dates, qspec_freqs