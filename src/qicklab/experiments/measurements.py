import os, logging, datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..config.ExperimentConfig import expt_cfg
#from ..config.round_robin_config import save_figs
from ..hardware.build_state import all_qubit_state
from ..hardware.qick_programs import SingleToneSpectroscopyProgram
from ..utils.file_helpers import create_folder_if_not_exists, extract_resonator_frequencies
from ..analysis.plotting import plot_spectroscopy

class ResonanceSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, save_figs,
                 experiment=None, verbose=False, logger=None, qick_verbose = True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.q_config = all_qubit_state(experiment, self.number_of_qubits)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: {self.config}')
        if self.verbose:
            print(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: ', self.config)

    def run(self):
        fpts = self.exp_cfg["start"] + self.exp_cfg["step_size"] * np.arange(self.exp_cfg["steps"])
        fcenter = self.config['res_freq_ge']
        amps = np.zeros((len(fcenter), len(fpts)))

        for index, f in enumerate(tqdm(fpts)):
            self.config["res_freq_ge"] = fcenter + f

            prog = SingleToneSpectroscopyProgram(
                self.experiment.soccfg,
                reps=self.exp_cfg["reps"],
                final_delay=0.5,
                cfg=self.config
            )
            iq_list = prog.acquire(self.experiment.soc, soft_avgs=self.exp_cfg["rounds"], progress=self.qick_verbose)
            for i in range(len(self.config['res_freq_ge'])):
                amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])

        amps = np.array(amps)

        if self.save_figs:
            res_freqs = plot_spectroscopy(
                qubit_index=self.QubitIndex,
                fpts=fpts,
                fcenter=fcenter,
                amps=amps,
                round_num=self.round_num,
                config=self.config,
                outerFolder=self.outerFolder,
                expt_name=self.expt_name,
                experiment=self.experiment,
                save_figs=self.save_figs,
                fig_quality=100,
                title_word="Resonator",
                xlabel="Frequency (MHz)",
                ylabel="Amplitude (a.u.)",
                find_min=True,
                plot_min_line=True,
                include_min_in_title=True,
                return_min=True,
                fig_filename=None
            )
        else:
            res_freqs = extract_resonator_frequencies(
                    fpts,  # dict: keys are qubit indices (0 means qubit 1) and values are (x_data, y_data)
                    process_offset=True,
                    offsets=fcenter  # dict: keys matching those in data, with offset frequency values
            )
        return res_freqs, fpts, fcenter, amps, self.config

