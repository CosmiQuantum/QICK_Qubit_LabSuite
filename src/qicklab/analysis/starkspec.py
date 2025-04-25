import os, datetime
import numpy as np

from ..utils.ana_utils  import rotate_and_threshold
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .plot_tools import plot_stark_simple
from .shot_tools import process_shots
from .stark_tools import gain2freq_detuning

class starkspec:

    def __init__(self, data_dir, dataset, QubitIndex, duffing_constant, theta, threshold, anharmonicity, detuning, folder = "study_data", expt_name = "starkspec_ge", thresholding=True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.duffing_constant = duffing_constant
        self.folder = folder
        self.expt_name = expt_name
        self.theta = theta
        self.threshold = threshold
        self.thresholding = thresholding
        self.anharmonicity = anharmonicity
        self.detuning = detuning

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
                    [steps, reps]))
            Q_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape(
                    [steps, reps]))
            P.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('P', [])[0][0].decode())))

        return dates, n, gain_sweep, steps, reps, I_shots, Q_shots, P

    def plot_shots(self, I_shots, Q_shots, gains, n, round=0, idx=10):
        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        i_new, q_new, states = rotate_and_threshold(this_I, this_Q, self.theta, self.threshold)

        title = (f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: ' +
                 f'rotated I,Q shots for stark_spec at gain: {np.round(gains[idx],2)}')

        _, _ = plot_shots(i_new, q_new, states, rotated=True, title=title)

    def process_shots(self, I_shots, Q_shots, n, steps):
        return process_shots(I_shots, Q_shots, n, steps, self.theta, self.threshold, thresholding=self.thresholding, axis=1)

    def gain2freq(self, gains):
        steps = int(len(gains)/2)
        gains_pos_detuning = gains[steps:]
        gains_neg_detuning = gains[:steps]

        freq_posneg = gain2freq_detuning(gains_pos_detuning, gains_neg_detuning, self.duffing_constant, self.anharmonicity, self.detuning)

        freqs = np.concatenate(freq_posneg)
        return freqs

    def get_p_excited_in_round(self, gains, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round][:]

        title = (f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n}: ' + 
                  ' stark spectroscopy')
        if plot: _, _ = plot_stark_simple(gains, self.gain2freq(gains), p_excited_in_round, title=title)

        return p_excited_in_round

def starkspec_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, duffing_constant=220, threshold=-1285.08904, theta=0.17681, selected_round=[10, 73]):
    stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant, theta, threshold, anharmonicity[QubitIndex], detuning[QubitIndex])
    stark_dates, stark_n, stark_gains, stark_steps, stark_reps, stark_I_shots, stark_Q_shots, stark_P = stark.load_all()
    stark_p_excited = stark.process_shots(stark_I_shots, stark_Q_shots, stark_n, stark_steps)
    stark_freqs = stark.gain2freq(stark_gains)

    outdata = {}
    for rnd in selected_round:
        outdata[rnd] = stark.get_p_excited_in_round(rstark_gains, rstark_p_excited, rstark_n, rnd, plot=True)
    return outdata

