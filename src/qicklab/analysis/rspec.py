import os, datetime
import numpy as np

from scipy.optimize import curve_fit
from ..datahandling.datafile_tools import find_h5_files, load_h5_data, process_h5_data, get_data_field
from .data_tools import get_h5_for_qubit

class rspec:
    def __init__(self, data_dir, dataset, QubitIndex, folder="study_data", expt_name="res_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files_all_qubits = os.listdir(data_path)
        h5_files = get_h5_for_qubit(data_path, h5_files_all_qubits, self.QubitIndex, 'Res')
        h5_files.sort()
        n = len(h5_files)

        dates = []
        rspec_freqs = []
        rspec_freq_centers = []
        rspec_mags = []
        load_data = load_h5_data(os.path.join(data_path, h5_files[0]), 'Res', save_r=1)
        rspec_probe_freqs = process_h5_data(load_data['Res'][self.QubitIndex].get('freq_pts', [])[0][0].decode())
        mag_idx = np.array(np.arange(0, len(rspec_probe_freqs)) + self.QubitIndex * len(rspec_probe_freqs))

        for h5_file in h5_files:
            load_data = load_h5_data(os.path.join(data_path, h5_file), 'Res', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['Res'][self.QubitIndex].get('Dates', [])[0][0]))
            rspec_mag0 = process_h5_data(load_data['Res'][self.QubitIndex].get('Amps', [])[0][0].decode())
            rspec_mags.append(np.array(rspec_mag0)[mag_idx])
            rspec_freq_centers.append(process_h5_data(load_data['Res'][self.QubitIndex].get('freq_center', [])[0][0].decode())[self.QubitIndex])
            rspec_freqs.append(process_h5_data(load_data['Res'][self.QubitIndex].get('Found Freqs', [])[0][0].decode())[self.QubitIndex])

        return dates, n, rspec_probe_freqs, rspec_mags, rspec_freqs, rspec_freq_centers

    def get_rspec_freq_in_round(self, rspec_freqs, rspec_mags, round, n, plot=False):
        rspec_mag = rspec_mags[round]
        rspec_freq = rspec_freqs[round]

        #if plot:

            #title = (f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: ' +
            #         f'rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[idx],2)} us')

            #_, _ = plot_rspec_simple(qspec_probe_freqs, thisI, thisQ, fitcurve=qspec_fit, title=title)

        return rspec_mag, rspec_freq
