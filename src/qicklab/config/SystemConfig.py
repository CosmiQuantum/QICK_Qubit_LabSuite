from qick import *
import sys
import os
sys.path.append(os.path.abspath("/home/nexusadmin/Documents/GitHub/tprocv2_demos"))
from tprocv2_demos.qick_tprocv2_experiments_mux.socProxy import makeProxy
import os
import datetime
import numpy as np
from qicklab.utils.file_helpers import create_folder_if_not_exists

class QICK_experiment:
    def __init__(self, folder, DAC_attenuator1 = 5, DAC_attenuator2 = 15, ADC_attenuator = 15, fridge = None):
        if fridge == "QUIET":
            # Where do you want to save data
            self.outerFolder = folder
            create_folder_if_not_exists(self.outerFolder)

            # attenuation settings
            self.DAC_attenuator1 = DAC_attenuator1
            self.DAC_attenuator2 = DAC_attenuator2
            self.ADC_attenuator = ADC_attenuator

            # Make proxy to the QICK
            self.soc, self.soccfg = makeProxy()
            #print(self.soccfg)

            self.FSGEN_CH = 0 # 0 for "old QICK", 6 for RF board
            self.MIXMUXGEN_CH = 4 # Readout resonator DAC channel
            self.MUXRO_CH = [2, 3, 4, 5, 6, 7]
            self.MUXRO_CH_RF = 5  # New variable that we need for QICK box

            self.TESTCH_DAC = 5 # loopback channel for RF board
            self.TESTCH_ADC = 0  # loopback channel for RF board
            self.TESTCH_ADC_RF = 4  # New variable that we need for QICK box

            ### NEW for the RF board
            self.qubit_center_freq = 4400  # To be in the middle of the qubit freqs.
            self.res_center_freq = 6330  # To be in the middle of the res freqs. 3000-5000 see nothing,6000 and 7000 see something, 8000+ see nothing
            # self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_ro_filter(self.MUXRO_CH_RF, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # # Set attenuator on DAC.
            # self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
            # self.soc.rfb_set_gen_rf(self.FSGEN_CH, 5, 4)  # Verified 30->25 see increased gain in loopback
            # # Set attenuator on ADC.
            # ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
            # self.soc.rfb_set_ro_rf(self.MUXRO_CH_RF, self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback


            # Qubit you want to work with
            self.QUBIT_INDEX = 5

            # Hardware Configuration
            self.hw_cfg = {
                # DAC
                "qubit_ch": [self.FSGEN_CH] * 6,  # Qubit Channel Port, Full-speed DAC
                "res_ch": [self.MIXMUXGEN_CH] * 6,  # Single Tone Readout Port, MUX DAC
                # "qubit_ch_ef": [GEN_CH5]*6, # Qubit ef Channel, Full-speed DAC
                "nqz_qubit": 1,
                "nqz_res": 2,
                # ADC
                "ro_ch": [self.MUXRO_CH] * 6,  # MUX readout channel
                "list_of_all_qubits": [0, 1, 2, 3, 4, 5]
            }

            # Readout Configuration
            self.readout_cfg = {
                "trig_time": 0.75,  # [Clock ticks] - get this value from TOF experiment

                # Changes related to the resonator output channel
                "mixer_freq": 6000,  # [MHz]
                "res_freq_ge": [6217, 6276, 6335, 6407, 6476, 6538],  # MHz, run 5
                # "res_freq_ge": [6191.419, 6216.1, 6292.361, 6405.77, 6432.759, 6468.481],  # MHz, run 4a
                # "res_gain_ge": [1] + [0]*5,
                "res_gain_ge": [1, 1, 1, 1, 1, 1],
                # set_res_gain_ge(QUBIT_INDEX), #utomatically sets all gains to zero except for the qubit you are observing
                # "res_gain_ge": [1,1,0.7,0.7,0.7,1], #[0.4287450656184295, 0.4903077560386716, 0.4903077560386716, 0.3941941738241592, 0.3941941738241592, 0.4903077560386716],  # DAC units
                # "res_freq_ef": [7149.44, 0, 0, 0, 0, 0], # [MHz]
                # "res_gain_ef": [0.6, 0, 0, 0, 0, 0], # [DAC units]
                "res_length": 9.0,  # [us] (1.0 for res spec)
                "res_phase": [(2.2684+1.6806) * 180/np.pi, (-0.9415+1.8203) * 180/np.pi, (1.5575-2.8090-0.2) * 180/np.pi,
                              (-1.3211+2.7114-0.2-0.69+0.777-1.9) * 180/np.pi, (-0.1448+0.2798) * 180/np.pi, (-0.3309+0.6677+2.56) * 180/np.pi],#(2.265+1.68) * 180/np.pi
                # [-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
                # "res_phase": [0]*6,#[-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
                "ro_phase": [0, 0, 0, 0, 0, 0],  # Rotation Angle From QICK Function
                "threshold": [0, 0, 0, 0, 0, 0], # Threshold for Distinguish g/e, from QICK Function
            }

            # Qubit Configuration
            self.qubit_cfg = {
                "qubit_freq_ge": [4184.14, 3821.149, 4156.53, 4459.20, 4471.12, 4997.86],  # new
                #"qubit_freq_ge": [4184.14, 3821.144, 4156.57, 4459.19, 4471.12, 4997.86], #old
                #"qubit_freq_ge": [4184.13, 3821.142, 4156.58, 4459.19, 4471.10, 4997.87], #old
                #"qubit_freq_ge": [4184.15, 3821.156, 4156.88, 4459.12, 4471.18, 4998.04],  # Freqs of Qubit g/e Transition, old
                "qubit_gain_ge": [0.05] * 6, #[1] * 6,
                # [0.4287450656184295, 0.4287450656184295, 0.4903077560386716, 0.6, 0.4903077560386716, 0.4287450656184295], # For spec pulse
                "qubit_length_ge": 15,  # [us] for spec Pulse
                "qubit_phase": 0,  # [deg]
                #"sigma": [0.15]*6,  # [us] for Gaussian Pulse (5+10 DAC atten for qubit)
                "sigma": [0.13, 0.15, 0.21, 0.14, 0.19, 0.19],  # [us] for Gaussian Pulse (5+10 DAC atten for qubit)
                #"sigma": [0.05, 0.09, 0.07, 0.065, 0.09, 0.3],  # Goal: cut sigma in half [us] for Gaussian Pulse (5+4 DAC atten for qubit)
                # "pi_amp": [0.92, 0.87, 0.75, 0.73, 0.77, 0.78], # old RR values
                "pi_amp": [1.0, 0.93, 0.77, 0.8, 0.81, 0.9], # Eyeballed by Sara today (5+10 DAC atten for qubit)
                #"pi_amp": [0.7, 0.95, 0.75, 0.78, 0.77, 0.8],  # With shorter sigma (5+4 DAC instead of 5+5 DAC atten for qubit)
                "qubit_mixer_freq": 4300,  # [MHz]
            }

        elif fridge == "NEXUS":
            # Where do you want to save data
            self.outerFolder = folder
            self.create_folder_if_not_exists(self.outerFolder)

            # attenuation settings
            self.DAC_attenuator1 = DAC_attenuator1
            self.DAC_attenuator2 = DAC_attenuator2
            self.ADC_attenuator = ADC_attenuator

            # Make proxy to the QICK
            self.soc, self.soccfg = makeProxy()
            print(self.soccfg)

            self.FSGEN_CH = 10  # set to 8 for bias spectroscopy, and 10 for everything else (pi pulses, RR)
            self.MIXMUXGEN_CH = 4  # Readout resonator DAC channel
            self.MUXRO_CH = [2, 3, 4, 5]
            # self.MUXRO_CH_RF = 5  # New variable that we need for QICK box

            # self.TESTCH_DAC = 5 # loopback channel for RF board
            # self.TESTCH_ADC = 0  # loopback channel for RF board
            # self.TESTCH_ADC_RF = 4  # New variable that we need for QICK box

            # From mux_simultaneous
            # GEN_CH8 = 8
            # GEN_CH10 = 10
            # GEN_CH12 = 12
            # GEN_CH14 = 14
            # MIXMUXGEN_CH = 4
            # MUXRO_CH = [2, 3, 4, 5]
            # # Qubit you want to work with
            # QUBIT_INDEX = 0

            ### NEW for the RF board
            # self.qubit_center_freq = 4400  # To be in the middle of the qubit freqs.
            # self.res_center_freq = 6330  # To be in the middle of the res freqs. 3000-5000 see nothing,6000 and 7000 see something, 8000+ see nothing
            # self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.0)
            # self.soc.rfb_set_ro_filter(self.MUXRO_CH_RF, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
            # # Set attenuator on DAC.
            # self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
            # self.soc.rfb_set_gen_rf(self.FSGEN_CH, 5, 4)  # Verified 30->25 see increased gain in loopback
            # # Set attenuator on ADC.
            # ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
            # self.soc.rfb_set_ro_rf(self.MUXRO_CH_RF, self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback

            # Qubit you want to work with
            self.QUBIT_INDEX = 0

            # Hardware Configuration
            self.hw_cfg = {
                # DAC
                "qubit_ch": [self.FSGEN_CH] * 4,  # Qubit Channel Port, Full-speed DAC
                "res_ch": [self.MIXMUXGEN_CH] * 4,  # Single Tone Readout Port, MUX DAC
                # "qubit_ch_ef": [GEN_CH5]*6, # Qubit ef Channel, Full-speed DAC
                "nqz_qubit": 2,
                "nqz_res": 2,
                # ADC
                "ro_ch": [self.MUXRO_CH] * 4,  # MUX readout channel,
                "list_of_all_qubits": [0, 1, 2, 3]
            }

            # Readout Configuration
            self.readout_cfg = {
                "trig_time": 0.75,  # [Clock ticks] - get this value from TOF experiment
                # Changes related to the resonator output channel
                "mixer_freq": 5500,  # [MHz]
                "res_freq_ge": [6187.191, 5827.678, 6074.095, 5958.453],  # MHz #5958.8 (Grace)
                "res_gain_ge": [0.4, 0.4, 0.4, 0.3875],  # [0.15]*4, #[1, 1, 1, 1],
                "res_length": 4.6,  # 10,  # [us] (1.0 for res spec)
                "res_phase": [0] * 4,
                "ro_phase": [0, 0, 0, 0]  # [0] * 4,  # Rotation Angle From QICK Function
            }

            # Qubit Configuration
            self.qubit_cfg = {
                "qubit_mixer_freq": 4300,  # [MHz]
                "qubit_freq_ge": [4909, 4749.4, 4569, 4756],  # Freqs of Qubit g/e Transition
                "qubit_gain_ge": [0.2] * 4,  # [0.008] * 4,#[0.2, 0.2, 0.2, 0.01], #[0.2] * 4,  #0.07
                "qubit_length_ge": 20,  # [us] for spec Pulse
                "qubit_phase": 0,  # [deg]
                "sigma": [0.04, 0.025, 0.04, 0.03],  # [0.08, 0.15, 0.11, 0.09], # TO DO CHANGE THIS (11/26)
                "pi_amp": [1.0, 0.93, 0.77, 0.846],  # TO DO CHANGE THIS (11/26)

                # "qubit_freqs_ge": [4909, 4749.4, 4569, 4759],  # Freqs of Qubit g/e Transition
                # "qubit_gains_ge": [1] * 4,  # [0.05] * 4
                # "qubit_phases": [0] * 4,  # [deg]
            }
        else:
            print("fridge variable is None or something else, please configure for your fridge or "
                  "change fridge to \"NEXUS\" or \"QUIET\" ")






