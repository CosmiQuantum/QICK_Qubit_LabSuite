import numpy as np

from ..utils.ana_utils  import rotate_and_threshold

def process_shots(I_shots, Q_shots, n, steps, theta, threshold, thresholding=True, axis=0):
    p_excited = []
    for round in np.arange(n):
        p_excited_in_round = []
        for idx in np.arange(steps):
            if   axis==0:
                this_I = I_shots[round][:, idx]
                this_Q = Q_shots[round][:, idx]
            elif axis==1:
                this_I = I_shots[round][idx,:]
                this_Q = Q_shots[round][idx,:]

            i_new, q_new, states = rotate_and_threshold(this_I, this_Q, theta, threshold)

            if not thresholding:
                states = np.mean(i_new)

            p_excited_in_round.append(np.mean(states))

        p_excited.append(p_excited_in_round)

    return p_excited