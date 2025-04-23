import os, sys
import re
import datetime
import h5py

import numpy as np
import matplotlib.pyplot as plt

from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata

class ssf:
    def __init__(self,data_dir, dataset, QubitIndex, folder="study_data", expt_name ="ss_ge"):
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
        I_g = []
        Q_g = []
        I_e = []
        Q_e = []
        fid = []
        theta = []

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'SS', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['SS'][self.QubitIndex].get('Dates', [])[0][0]))

            I_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_g', [])[0][0].decode()))
            Q_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_g', [])[0][0].decode()))
            I_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_e', [])[0][0].decode()))
            Q_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_e', [])[0][0].decode()))
            fid.append(load_data['SS'][self.QubitIndex].get('Fidelity', [])[0])
            theta.append(load_data['SS'][self.QubitIndex].get('Angle', [])[0])

        return dates, n, I_g, Q_g, I_e, Q_e, fid, theta

    def get_ssf_in_round(self, I_g, Q_g, I_e, Q_e, round):
        ig = np.array(I_g[round])
        qg = np.array(Q_g[round])
        ie = np.array(I_e[round])
        qe = np.array(Q_e[round])

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        return ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye

