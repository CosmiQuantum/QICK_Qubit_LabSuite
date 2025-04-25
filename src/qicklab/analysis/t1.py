import os, datetime
import numpy as np

from scipy.optimize import curve_fit

from .fit_functions import exponential
from ..utils.ana_utils  import rotate_and_threshold
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .plotting import plot_shots, plot_t1_simple
from .fitting import fit_t1

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

        title = (f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: ' +
                 f'rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[idx],2)} us')

        _, _ = plot_shots(i_new, q_new, states, rotated=True, title=title)

    def process_shots(self, I_shots, Q_shots, n, steps):
        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][:, idx]
                this_Q = Q_shots[round][:, idx]

                i_new, q_new, states = rotate_and_threshold(this_I, this_Q, self.theta, self.threshold)

                if not self.thresholding:
                    states = np.mean(i_new)

                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def t1_fit(self, signal, delay_times, round, n, plot=False):

        T1_est, T1_err, q1_fit_exponential = fit_t1(signal, delay_times)

        title = (f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n}: ' +
                 f't1_ge = {T1_est:.3f} +/- {T1_err} us')

        if plot: plot_t1_simple(signal, delay_times, q1_fit_exponential, title=title)
            
        return q1_fit_exponential, T1_err, T1_est

    def get_t1_in_round(self, delay_times, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round][:]
        q1_fit_exponential, T1_err, T1_est = self.t1_fit(p_excited_in_round, delay_times, round, n, plot=plot)

        return q1_fit_exponential, T1_err, T1_est

    def get_all_t1(self, delay_times, p_excited, n, plot_idxs=[]):

        t1s = np.zeros(n) #[]
        t1_errs = np.zeros(n) #[]

        for round in np.arange(n):
            _, t1_errs[round], t1s[round] = self.get_t1_in_round(delay_times, p_excited, n, round, plot=(round in plot_idxs))

        return t1s.tolist(), t1_errs.tolist()


def t1_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, threshold=-1285.08904, theta=0.17681, selected_round=[10, 73]):
    # selected_round = [10, 73]
    # threshold = 0 #overwritten when get_threshold flag is set to True
    # theta = 0 #overwritten when get_threshold flag is set to True

    t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold)
    t1_dates, t1_n, delay_times, t1_steps, t1_reps, t1_I_shots, t1_Q_shots = t1_ge.load_all()
    t1_p_excited = t1_ge.process_shots(t1_I_shots, t1_Q_shots, t1_n, t1_steps)
    t1s, t1_errs = t1_ge.get_all_t1(delay_times, t1_p_excited, t1_n, plot_idxs=selected_round)

    return t1_dates, t1s, t1_errs