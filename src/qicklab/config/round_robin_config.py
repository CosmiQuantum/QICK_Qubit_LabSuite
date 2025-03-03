from qicklab.experiments.measurements import ResonanceSpectroscopy
from .expt_config import list_of_all_qubits, tot_num_of_qubits
from qicklab.utils.file_helpers import create_data_dict, check_visdom_connection, mask_gain_res
import visdom
import os
import logging
import datetime
import numpy as np

n= 100000
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                     # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True                      # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose =True                      # print everything to the console in real time, good for debugging, bad for memory
qick_verbose=True
debug_mode = True                   # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = True                 #use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False           # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
Qs_to_look_at = [0,1,2,3,4,5]   # only list the qubits you want to do the RR for

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True,
             "t1": True, "t2r": False, "t2e": False}

#outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30/", str(datetime.date.today()))
outerFolder = os.path.join("/data/QICK_data/6transmon_run6/", str(datetime.date.today()))

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [4.3, 9, 3.8, 6, 4.5, 9]
res_gain = [0.833, 0.9167, 0.8, 0.6, 0.9167, 0.6]
freq_offsets = [-0.0667, 0.2, -0.1333, 0.3, -0.2667, -0.75]

################################################## Configure logging ###################################################
if not os.path.exists(outerFolder): os.makedirs(outerFolder)
log_file = os.path.join(outerFolder, "RR_script.log")

################################################# to Store Data ########################################################
# Define what to save to h5 files
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']

# initialize a dictionary to store those values
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
# check if there's a connection to visdom depending on how the boolean is set
check_visdom_connection(live_plot)

#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits



