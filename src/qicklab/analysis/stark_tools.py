import numpy as np

def gain2freq_resonator(gains, stark_constant):
    freqs = np.square(gains) * stark_constant
    return freqs

def gain2freq_detuning(gains_pos_detuning, gains_neg_detuning, duffing_constant, anharmonicity, detuning):
    # positive detuning, negative frequency shift
    freq_pos_detuning = duffing_constant * (anharmonicity * np.square(gains_pos_detuning)) / (-1*detuning * (anharmonicity - detuning))

    # negative detuning, positive frequency shift
    freq_neg_detuning = duffing_constant * (anharmonicity * np.square(gains_neg_detuning)) / (   detuning * (anharmonicity + detuning))

    return (freq_neg_detuning, freq_pos_detuning)