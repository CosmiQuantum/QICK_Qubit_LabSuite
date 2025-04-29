"""
fitting.py

This module provides functions for performing curve fitting on experimental data.
It includes functions to fit an exponential decay (for T1 measurements) and Lorentzian curves,
as well as helper routines used in the fitting process.

Dependencies:
    - numpy
    - matplotlib
    - scipy.optimize.curve_fit
    - create_folder_if_not_exists from this package's utils.file_helpers module

Usage Example:
    from your_package.fitting import t1_fit, get_lorentzian_fits

    # Prepare your data arrays for I, Q and delay_times/frequencies
    T1_est, T1_err = t1_fit(I, Q, delay_times, signal='I')
    largest_amp_curve_mean, I_fit, Q_fit, fit_err = get_lorentzian_fits(I, Q, freqs)
"""

import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from ..utils.ana_utils import max_offset_difference_with_x
from ..utils.file_utils import create_folder_if_not_exists
from  .fit_functions import exponential, lorentzian, allan_deviation_model

def fit_t1(signal, delay_times):
    # Initial guess for parameters: amplitude (a), time shift (b), decay constant (c), baseline (d)
    q1_a_guess = np.max(signal) - np.min(signal)
    q1_b_guess = 0
    q1_c_guess = (delay_times[-1] - delay_times[0]) / 5
    q1_d_guess = np.min(signal)
    q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

    # Define bounds: T1 (c) must be positive, amplitude (a) can be any value.
    lower_bounds = [-np.inf, -np.inf, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]   # No upper bound on parameters

    # Fit the exponential function to the data
    q1_popt, q1_pcov = curve_fit(exponential, delay_times, signal,
        p0=q1_guess, bounds=(lower_bounds, upper_bounds), method='trf', maxfev=10000)

    # # Generate the fitted exponential curve
    # fit_exponential = exponential(delay_times, *q1_popt)

    # Extract T1 (decay constant) and its error
    T1_est = q1_popt[2]
    T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')

    return T1_est, T1_err, exponential(delay_times, *q1_popt)

def fit_t1_IQ(I, Q, delay_times, signal='None'):
    """
    Fit an exponential decay curve to I and Q signal data to estimate the T1 decay constant.

    The function selects a signal based on the provided indicator or based on the
    magnitude of signal change if 'None' is passed. It then fits the exponential model
    to the chosen signal.

    Parameters:
        I (array_like): In-phase signal data.
        Q (array_like): Quadrature signal data.
        delay_times (array_like): Array of delay times corresponding to the measurements.
        signal (str): Signal indicator ('I', 'Q', or 'None'). If 'None', the signal with the larger
                      overall amplitude change is used.
        return_everything (bool): If True, returns the complete fitted exponential curve along with the
                                  decay constant, error, and signal indicator used. Otherwise, only the
                                  decay constant and its error are returned.

    Returns:
        tuple: If return_everything is False:
                   (T1_est, T1_err)
               If return_everything is True:
                   (q1_fit_exponential, T1_err, T1_est, plot_sig)
               where:
                   T1_est (float): Estimated decay constant.
                   T1_err (float): Error associated with the decay constant.
                   q1_fit_exponential (array_like): Fitted exponential curve.
                   plot_sig (str): Indicator of which signal was used ('I' or 'Q').
    """
    if 'I' in signal:
        signal_data = I
        plot_sig = 'I'
    elif 'Q' in signal:
        signal_data = Q
        plot_sig = 'Q'
    else:
        # Determine which signal has a larger overall change
        if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
            signal_data = I
            plot_sig = 'I'
        else:
            signal_data = Q
            plot_sig = 'Q'

    return fit_t1(signal_data, delay_times)

def fit_lorenzian(signal, freqs, freq_q):
    # Initial guesses for whatever signal channel
    initial_guess = [freq_q, 1, np.max(signal), np.min(signal)]

    # First round of fits (to get rough estimates)
    params, _ = curve_fit(lorentzian, freqs, signal, p0=initial_guess)

    # Use these fits to refine guesses
    x_max_diff, max_diff = max_offset_difference_with_x(freqs, signal, params[3])
    initial_guess = [x_max_diff, 1, np.max(signal), np.min(signal)]

    # Second (refined) round of fits, this time capturing the covariance matrices
    params, cov = curve_fit(lorentzian, freqs, signal, p0=initial_guess)

    # Create the fitted curves
    fit = lorentzian(freqs, *params)

    # Calculate errors from the covariance matrices
    fit_err = np.sqrt(np.diag(cov))[0]

    # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
    mean = params[0]
    fwhm = 2 * params[1]

    # Calculate the amplitude differences from the fitted curves
    amp_fit = abs(np.max(fit) - np.min(fit))

    return mean, fit_err, fwhm, fit, amp_fit

def fit_lorenzian_IQ(I, Q, freqs, metric_freq, signal='None', verbose=False):
    """
    Perform Lorentzian fits on I and Q signals over frequency.

    This function fits Lorentzian curves to both I and Q data in two rounds:
    an initial rough fit and a refined fit. The best-fit curve is chosen based on
    the amplitude difference of the fitted curves.

    Parameters:
        I (array_like): In-phase signal data.
        Q (array_like): Quadrature signal data.
        freqs (array_like): Frequency values corresponding to the signals.
        metric_freq (float): Frequency value used as an initial estimate (typically the frequency where I is maximized).
        signal (str): Signal indicator ('I', 'Q', or 'None'). If 'None', the function selects the curve
                      with the larger amplitude difference.
        verbose (bool): If True, prints error messages and additional debugging information.

    Returns:
        tuple:
            (mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err)
            where:
                - mean_I (float): Fitted center frequency for I.
                - mean_Q (float): Fitted center frequency for Q.
                - I_fit (array_like): Fitted Lorentzian curve for I.
                - Q_fit (array_like): Fitted Lorentzian curve for Q.
                - largest_amp_curve_mean (float): Center frequency of the curve with the largest amplitude difference.
                - largest_amp_curve_fwhm (float): Full width at half maximum (FWHM) of the chosen curve.
                - fit_err (float): Fitting error for the chosen curve's center frequency.
        If an error occurs during the fit, returns a tuple of None values.
    """
    try:
        mean_I, fit_err_I, fwhm_I, I_fit, amp_I_fit = fit_lorenzian(I, freqs, freq_q)
        mean_Q, fit_err_Q, fwhm_Q, Q_fit, amp_Q_fit = fit_lorenzian(Q, freqs, freq_q)

        # Choose which curve to report based on the signal indicator or amplitude difference
        if 'None' in signal:
            if amp_I_fit > amp_Q_fit:
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                fit_err = fit_err_I[0]
            else:
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                fit_err = fit_err_Q[0]
        elif 'I' in signal:
            largest_amp_curve_mean = mean_I
            largest_amp_curve_fwhm = fwhm_I
            fit_err = fit_err_I[0]
        elif 'Q' in signal:
            largest_amp_curve_mean = mean_Q
            largest_amp_curve_fwhm = fwhm_Q
            fit_err = fit_err_Q[0]
        else:
            if verbose: print('Invalid signal passed, please choose "I", "Q", or "None".')
            return None, None, None, None, None

        return I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err

    except Exception as e:
        if verbose: print("Error during Lorentzian fit:", e)
        return None, None, None, None, None

def get_lorentzian_fits(I, Q, freqs, verbose=False):
    """
    Perform Lorentzian fits on I and Q data based on frequency values and choose the best fit.

    This is a wrapper around the 'fit_lorenzian' function that selects the metric frequency
    based on the maximum value in the I data.

    Parameters:
        I (array_like): In-phase signal data.
        Q (array_like): Quadrature signal data.
        freqs (array_like): Array of frequency values.
        verbose (bool): If True, prints additional debugging information.

    Returns:
        tuple: (largest_amp_curve_mean, I_fit, Q_fit, fit_err) where:
            - largest_amp_curve_mean (float): Center frequency of the curve with the largest amplitude difference.
            - I_fit (array_like): Fitted Lorentzian curve for I.
            - Q_fit (array_like): Fitted Lorentzian curve for Q.
            - fit_err (float): Fitting error for the chosen curve's center frequency.
    """
    freqs = np.array(freqs)
    metric_freq = freqs[np.argmax(I)]

    I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = fit_lorenzian(
        I, Q, freqs, metric_freq, verbose=verbose
    )

    return I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err

