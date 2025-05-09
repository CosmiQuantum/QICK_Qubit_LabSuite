'''
fit_functions.py
    These methods are the functional forms used to do fitting in various parts of the codebase
'''

import numpy as np

def cosine( x, a, b, c, d):
    """
       Cosine function for curve fitting.
    """
    x = np.array(x)
    return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

def exponential(x, a, b, c, d):
    """
    Exponential function for curve fitting.

    The model is defined as:
        y = a * exp(- (x - b) / c) + d

    Parameters:
        x (array_like): Independent variable.
        a (float): Amplitude.
        b (float): Time shift.
        c (float): Decay constant (must be > 0).
        d (float): Baseline offset.

    Returns:
        array_like: Calculated exponential function values.
    """
    return a * np.exp(- (x - b) / c) + d

def lorentzian(f, f0, gamma, A, B):
    """
    Lorentzian function used for curve fitting.

    The model is defined as:
        L(f) = A * gamma^2 / ((f - f0)^2 + gamma^2) + B

    Parameters:
        f (array_like): Frequency values.
        f0 (float): Center frequency.
        gamma (float): Half-width at half-maximum (HWHM).
        A (float): Amplitude scaling factor.
        B (float): Baseline offset.

    Returns:
        array_like: Calculated Lorentzian function values.
    """
    return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

def allan_deviation_model(tau, h0, h_m1, h_m2, A, tau0):
    # White noise term: ∝ τ^(-1/2)
    term_white = np.sqrt(h0 / 2) * tau**(-0.5)
    # Flicker noise (1/ƒ noise) term: constant with τ
    term_flicker = np.sqrt(2 * np.log(2) * h_m1)
    # Random walk noise term: ∝ τ^(1/2)
    term_randomwalk = np.sqrt((4 * np.pi**2 / 6) * h_m2) * tau**0.5
    # Lorentzian noise term (exponentially correlated noise)
    term_lorentz = np.sqrt(A * tau0 / tau * (4 * np.exp(-tau/tau0) - np.exp(-2*tau/tau0) - 3 + 2*tau/tau0))
    return term_white + term_flicker + term_randomwalk + term_lorentz