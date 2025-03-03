from qicklab.experiments.measurements import ResonanceSpectroscopy
from qicklab.config.expt_config import list_of_all_qubits, FRIDGE
from qicklab.config.system_config import QICK_experiment
from qicklab.utils.file_helpers import create_data_dict, mask_gain_res, create_folder_if_not_exists, configure_logging
import logging
import datetime
import numpy as np
from qicklab.config.round_robin_config import (
    n, save_r, signal, save_figs, live_plot, fit_data, save_data_h5,
    verbose, debug_mode, thresholding, increase_qubit_reps,
    qubit_to_increase_reps_for, multiply_qubit_reps_by, Qs_to_look_at,
    run_flags, outerFolder, res_leng_vals, res_gain, freq_offsets,
    res_data, log_file, qick_verbose,
    tot_num_of_qubits, list_of_all_qubits
)



####################################################### RR #############################################################
rr_logger = configure_logging(log_file)
batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        #Get the config for this qubit
        experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 5, DAC_attenuator2 = 10, ADC_attenuator = 10,
                                     fridge=FRIDGE)
        create_folder_if_not_exists(outerFolder)

        #Mask out all other resonators except this one
        res_gains = mask_gain_res(QubitIndex, IndexGain=res_gain[QubitIndex], num_qubits=tot_num_of_qubits)
        experiment.readout_cfg['res_gain_ge'] = res_gains
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]

        ################################################## Res spec ####################################################
        if run_flags["res_spec"]:
            try:
                res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, outerFolder, j, save_figs,
                                                 experiment=experiment, verbose=verbose, logger=rr_logger,
                                                 qick_verbose=qick_verbose)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
                experiment.readout_cfg['res_freq_ge'] = res_freqs
                offset = freq_offsets[QubitIndex]  # use optimized offset values
                offset_res_freqs = [r + offset for r in res_freqs]
                experiment.readout_cfg['res_freq_ge'] = offset_res_freqs
                del res_spec

            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue  # skip the rest of this qubit

