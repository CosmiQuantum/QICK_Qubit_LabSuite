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

from ..utils.file_utils import create_folder_if_not_exists
from  .fit_functions import exponential, lorentzian, allan_deviation_model

def t1_fit(I, Q, delay_times=None, signal='None', return_everything=False):
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

    # Initial guess for parameters: amplitude, time shift, decay constant, baseline
    q1_a_guess = np.max(signal_data) - np.min(signal_data)
    q1_b_guess = 0
    q1_c_guess = (delay_times[-1] - delay_times[0]) / 5
    q1_d_guess = np.min(signal_data)
    q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

    # Define bounds: T1 (c) must be positive, amplitude (a) can be any value.
    lower_bounds = [-np.inf, -np.inf, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    # Fit the exponential function to the data
    q1_popt, q1_pcov = curve_fit(
        exponential,
        delay_times,
        signal_data,
        p0=q1_guess,
        bounds=(lower_bounds, upper_bounds),
        method='trf',
        maxfev=10000
    )

    # Generate the fitted exponential curve
    fit_exponential = exponential(delay_times, *q1_popt)

    # Extract T1 (decay constant) and its error
    T1_est = q1_popt[2]
    T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')

    if return_everything:
        return fit_exponential, T1_err, T1_est, plot_sig
    else:
        return T1_est, T1_err

def max_offset_difference_with_x(x_values, y_values, offset):
    """
    Find the x-value corresponding to the maximum average absolute difference from a given offset.

    The function calculates the average absolute difference from the offset for triplets of consecutive y-values
    and returns the x-value (from the middle point of each triplet) that corresponds to the maximum difference.

    Parameters:
        x_values (array_like): Array of x-values.
        y_values (array_like): Array of y-values.
        offset (float): The offset value to compare against.

    Returns:
        tuple:
            corresponding_x: The x-value corresponding to the maximum average difference.
            max_average_difference: The maximum average absolute difference calculated.
    """
    max_average_difference = -1
    corresponding_x = None

    for i in range(len(y_values) - 2):
        y_triplet = y_values[i:i + 3]
        average_difference = sum(abs(y - offset) for y in y_triplet) / 3

        if average_difference > max_average_difference:
            max_average_difference = average_difference
            corresponding_x = x_values[i + 1]

    return corresponding_x, max_average_difference

def fit_lorenzian(I, Q, freqs, metric_freq, signal='None', verbose=False):
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
        # Initial guesses for the parameters of I and Q
        initial_guess_I = [metric_freq, 1, np.max(I), np.min(I)]
        initial_guess_Q = [metric_freq, 1, np.max(Q), np.min(Q)]

        # First round of fits to obtain rough estimates
        params_I, _ = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, _ = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Refine the guesses using the maximum offset difference method
        x_max_diff_I, _ = max_offset_difference_with_x(freqs, I, params_I[3])
        x_max_diff_Q, _ = max_offset_difference_with_x(freqs, Q, params_Q[3])
        initial_guess_I = [x_max_diff_I, 1, np.max(I), np.min(I)]
        initial_guess_Q = [x_max_diff_Q, 1, np.max(Q), np.min(Q)]

        # Second round of fits (refined) capturing covariance matrices
        params_I, cov_I = curve_fit(lorentzian, freqs, I, p0=initial_guess_I)
        params_Q, cov_Q = curve_fit(lorentzian, freqs, Q, p0=initial_guess_Q)

        # Create the fitted curves
        I_fit = lorentzian(freqs, *params_I)
        Q_fit = lorentzian(freqs, *params_Q)

        # Calculate parameter errors from the covariance matrices
        fit_err_I = np.sqrt(np.diag(cov_I))
        fit_err_Q = np.sqrt(np.diag(cov_Q))

        # Extract fitted center frequencies and calculate FWHM
        mean_I = params_I[0]
        mean_Q = params_Q[0]
        fwhm_I = 2 * params_I[1]
        fwhm_Q = 2 * params_Q[1]

        # Calculate amplitude differences of the fitted curves
        amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
        amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

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
            if verbose:
                print('Invalid signal passed, please choose "I", "Q", or "None".')
            return None, None, None, None, None

        return I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err

    except Exception as e:
        if verbose:
            print("Error during Lorentzian fit:", e)
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

