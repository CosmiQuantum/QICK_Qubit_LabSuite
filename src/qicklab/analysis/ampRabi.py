import os, datetime
import numpy as np

from scipy.optimize import curve_fit

from .fit_functions import cosine
from .fit_tools import fit_amp_rabi
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .data_tools import get_h5_for_qubit


class ampRabi:

    def __init__(self,data_dir, dataset, QubitIndex, folder="study_data", expt_name ="rabi_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files_all_qubits = os.listdir(data_path)
        h5_files = get_h5_for_qubit(data_path, h5_files_all_qubits, self.QubitIndex, 'Rabi')
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I = []
        Q = []

        load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_files[0]), 'Rabi', save_r=1)
        gains = process_h5_data(load_data['Rabi'][self.QubitIndex].get('Gains', [])[0][0].decode())

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'Rabi', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['Rabi'][self.QubitIndex].get('Dates', [])[0][0]))
            I.append(process_h5_data(load_data['Rabi'][self.QubitIndex].get('I', [])[0][0].decode()))
            Q.append(process_h5_data(load_data['Rabi'][self.QubitIndex].get('Q', [])[0][0].decode()))

        return dates, n, gains, I, Q

    def get_all_pi_amp(self, gains, I, Q, n):
        pi_amps = []
        for round in np.arange(n):
            mag = np.sqrt(np.square(I[round] + np.square(Q[round])))
            pi_amp, cosine_fit = fit_amp_rabi(mag, gains)
            pi_amps.append(pi_amp)

        return pi_amps

    def get_pi_amp_in_round(self, gains, I, Q, round):
        mag = np.sqrt(np.square(I[round] + np.square(Q[round])))
        pi_amp, cosine_fit = fit_amp_rabi(mag, gains)

        return pi_amp, cosine_fit, mag
