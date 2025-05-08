import os, datetime

import numpy as np

from ..utils.ana_utils  import rotate_and_threshold
from ..utils.data_utils import process_h5_data
from ..utils.file_utils import load_from_h5_with_shotdata
from .data_tools import get_h5_for_qubit

class ssf:
    def __init__(self,data_dir, dataset, QubitIndex, folder="study_data", expt_name ="ss_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files_all_qubits = os.listdir(data_path)
        h5_files = get_h5_for_qubit(data_path, h5_files_all_qubits, self.QubitIndex, 'SS')
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_g = []
        Q_g = []
        I_e = []
        Q_e = []

        for h5_file in h5_files:
            load_data = load_from_h5_with_shotdata(os.path.join(data_path, h5_file), 'SS', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['SS'][self.QubitIndex].get('Dates', [])[0][0]))

            I_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_g', [])[0][0].decode()))
            Q_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_g', [])[0][0].decode()))
            I_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_e', [])[0][0].decode()))
            Q_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_e', [])[0][0].decode()))

        return dates, n, I_g, Q_g, I_e, Q_e #fid, theta

    def get_ssf_in_round(self, I_g, Q_g, I_e, Q_e, round, numbins=100):
        ig = np.array(I_g[round])
        qg = np.array(Q_g[round])
        ie = np.array(I_e[round])
        qe = np.array(Q_e[round])

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        ## Compute the rotation angle
        theta = -np.arctan2((ye - yg), (xe - xg))

        ## Apply the rotation angle to the IQ data
        ig_new, qg_new, _ = rotate_and_threshold(ig, qg, theta, 0.0)
        ie_new, qe_new, _ = rotate_and_threshold(ie, qe, theta, 0.0)
        xlims = [np.min(ig_new), np.max(ie_new)]

        ## New means of each blob
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        ## compute threshold
        #threshold = np.mean([xg, xe])

        ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        ## compute fidelity using overlap of histograms
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]

        return theta, threshold, fid, ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye

    def get_all_ssf(self, I_g, Q_g, I_e, Q_e, n): 
        thetas = []
        thresholds = []
        fids = []
        for round in np.arange(n):
            theta, threshold, fid,_,_,_,_,_,_,_,_ = self.get_ssf_in_round(I_g, Q_g, I_e, Q_e, round)
            thetas.append(theta)
            thresholds.append(threshold)
            fids.append(fid)

        return thetas, thresholds, fids

def ssf_demo(data_dir, dataset='2025-04-15_21-24-46', QubitIndex=0, selected_round=[10, 73]):
    ssf_ge = ssf(data_dir, dataset, QubitIndex)
    ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e = ssf_ge.load_all()
    
    outdata = {}
    for rnd in selected_round:
        outdata[rnd] = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, rnd)
    return outdata