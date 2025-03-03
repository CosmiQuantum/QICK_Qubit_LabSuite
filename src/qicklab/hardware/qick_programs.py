import copy
import logging
import numpy as np
from qick.asm_v2 import AveragerProgramV2


class SingleToneSpectroscopyProgram(AveragerProgramV2):
    """
    A QICK program for single-tone resonator spectroscopy.
    Inherits from qick.asm_v2.AveragerProgramV2.
    """

    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']

        self.declare_gen(
            ch=res_ch,
            nqz=cfg['nqz_res'],
            ro_ch=ro_chs[0],
            mux_freqs=cfg['res_freq_ge'],
            mux_gains=cfg['res_gain_ge'],
            mux_phases=cfg['res_phase'],
            mixer_freq=cfg['mixer_freq']
        )

        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch,
                length=cfg['res_length'],
                freq=f,
                phase=ph,
                gen_ch=res_ch
            )

        self.add_pulse(
            ch=res_ch,
            name="mymux",
            style="const",
            length=cfg["res_length"],
            mask=cfg["list_of_all_qubits"]
        )

    def _body(self, cfg):
        self.trigger(ros=cfg['ro_ch'], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)
