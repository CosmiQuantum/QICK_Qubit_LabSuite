import numpy as np

def gain2freq_resonator(gain, stark_constant):
    ## Return the frequency shift due to [resonator] Stark effect at the given gain(s)
    return np.square(gains) * stark_constant

def gain2freq_Duffing(gain, duffing_constant, anharmonicity, detuning):
    ## Return the frequency shift due to [constant detuning] Stark effect at the given gains(s)
    ## For detuning < 0 --> freq shift > 0
    ## For detuning > 0 --> freq shift < 0
    return duffing_constant * (anharmonicity * np.square(gain)) / (   detuning * (anharmonicity + detuning))

# def gain2freq_detuning(gains_pos_detuning, gains_neg_detuning, duffing_constant, anharmonicity, detuning):
#     ## positive detuning, negative frequency shift
#     freq_pos_detuning = gain2freq_Duffing(gains_pos_detuning, duffing_constant, anharmonicity, -1*detuning)
#     #duffing_constant * (anharmonicity * np.square(gains_pos_detuning)) / (-1*detuning * (anharmonicity - detuning))

#     ## negative detuning, positive frequency shift
#     freq_neg_detuning = gain2freq_Duffing(gains_neg_detuning, duffing_constant, anharmonicity,    detuning)
#     #duffing_constant * (anharmonicity * np.square(gains_neg_detuning)) / (   detuning * (anharmonicity + detuning))

#     return (freq_neg_detuning, freq_pos_detuning)