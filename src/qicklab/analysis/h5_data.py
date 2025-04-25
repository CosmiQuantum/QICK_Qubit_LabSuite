"""
h5_utils.py

This module provides functions to handle HDF5 file operations including saving, loading,
processing, and printing data. It includes utility functions to convert data types,
process strings into numeric formats, smooth numerical arrays, and handle configuration strings.

Dependencies:
    - ast, datetime, glob, os, re, h5py, numpy

Usage:
    Import the module and use the functions such as save_to_h5, load_from_h5,
    load_and_process_all_data, and others as needed.
"""

import time, datetime
import os, glob, re
import h5py
import ast

import numpy as np

from ..utils.data_utils import ensure_str, unwrap_singleton_list
from .plotting import *
from .fitting import *


def mask_gain_res(QUBIT_INDEX, IndexGain = 1, num_qubits=6):
    """Sets the gain for the selected qubit to 1, others to 0."""
    filtered_gain_ge = [0] * num_qubits  # Initialize all gains to 0
    if 0 <= QUBIT_INDEX < num_qubits: #makes sure you are within the range of options
        filtered_gain_ge[QUBIT_INDEX] = IndexGain  # Set the gain for the selected qubit
    return filtered_gain_ge

def extract_resonator_frequencies(data, process_offset=False, offsets=None):
    """
    Extracts the resonance frequencies for each qubit based on the provided data.

    Parameters
    ----------
    data : dict
        Dictionary where each key is a qubit index (0 means qubit 1) and each value is a tuple or list
        containing the x_data and y_data arrays for that qubit.
        keys are qubit indices (0 means qubit 1) and values are (x_data, y_data)
    process_offset : bool, optional
        If True, each qubit's x_data is shifted by an offset provided in the offsets dictionary.
    offsets : dict, optional
        Dictionary of offset frequencies to add to each qubit's x_data (keys must match those in data)
        when process_offset is True.
        keys matching those in data, with offset frequency values

    Returns
    -------
    dict
        Dictionary where each key is a qubit index and each value is the computed resonance frequency
        (x value corresponding to the minimum y value), rounded to 4 decimals.
    """
    res_freqs = {}
    for qkey, (x_vals, y_vals) in data.items():
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        # Apply offset if required.
        if process_offset:
            if offsets is None or qkey not in offsets:
                raise ValueError(
                    "Offsets must be provided as a dict with keys matching those in data when process_offset is True.")
            x_vals = x_vals + offsets[qkey]

        # Compute the resonance frequency: x value at the minimum y value.
        min_index = np.argmin(y_vals)
        res_freq = x_vals[min_index]
        res_freqs[qkey] = round(res_freq, 4)

    return res_freqs


def process_freq_pts(data):
    """
    Process a string representing frequency points and convert it to a NumPy array.

    The function replaces whitespace with commas, cleans the string to remove redundant commas,
    and then evaluates the cleaned string to create a NumPy array of frequency points.

    Parameters:
        data (str): String containing frequency points.

    Returns:
        np.ndarray or None: A NumPy array of frequency points if successful; otherwise, None.
    """
    # Remove newlines and extra spaces.
    data_str = data.replace('\n', ' ')
    # Replace multiple spaces with commas.
    formatted_str = data_str.replace('  ', ',').replace(' ', ',')
    formatted_str = formatted_str.replace(',]', ']').replace('],[', '],[')
    formatted_str = re.sub(r",,", ",", formatted_str)
    formatted_str = re.sub(r",\s*([\]])", r"\1", formatted_str)
    formatted_str = re.sub(r"(\d+)\.,", r"\1.0,", formatted_str)
    try:
        freq_points = np.array(eval(formatted_str))
    except Exception as e:
        print("Error processing freq_pts:", e)
        freq_points = None
    return freq_points


def process_config(x):
    """
    Process a configuration string safely.

    The function replaces any non-literal object representations with a placeholder string,
    then attempts to parse the string using ast.literal_eval for safety. If that fails,
    it falls back to eval using a restricted globals dictionary.

    Parameters:
        x (str): The configuration string.

    Returns:
        The evaluated configuration, or the original string if processing fails.
    """
    if not isinstance(x, str):
        return x
    x = x.strip()
    # Replace object representations with a placeholder.
    x = re.sub(r"<[^>]*object at [^>]*>", "'<object>'", x)
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return eval(x, safe_globals)
        except Exception as e:
            print("Error processing config:", e)
            return x


# ------------------ Process Map and Safe Globals ------------------

# safe_globals provides a restricted evaluation context for configuration strings.
safe_globals = {"np": np, "array": np.array, "__builtins__": {}}

# Map dataset keys to their corresponding processing functions.
# The lambdas assume the input value has already been passed through ensure_str.
process_map = {
    'Dates': lambda x: [] if (x is None or len(x) == 0)
             else [safe_convert_timestamp(ts) for ts in np.array(x).tolist()],
    'freq_pts': lambda x: process_freq_pts(x) if isinstance(x, str) else x,
    'freq_center': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Amps': lambda x: process_string_of_nested_lists(x) if isinstance(x, str) else x,
    'Found Freqs': lambda x: string_to_float_list(x) if isinstance(x, str) else x,
    'I': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Q': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Frequencies': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'I Fit': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Q Fit': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Recycled QFreq': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Gains': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Fit': lambda x: x,
    'Angle': lambda x: x,
    'Fidelity': lambda x: x,
    'I_g': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Q_g': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'I_e': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'Q_e': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'T1': lambda x: x,
    'Errors': lambda x: x,
    'Delay Times': lambda x: process_h5_data(x) if isinstance(x, str) else x,
    'T2': lambda x: x,
    'T2E': lambda x: x,
    'Exp Config': lambda x: process_config(x),
    'Syst Config': lambda x: process_config(x),
    'Round Num': lambda x: float(x) if isinstance(x, str) and is_numeric_string(x) else x,
    'Batch Num': lambda x: float(x) if isinstance(x, str) and is_numeric_string(x) else x,
}


# ------------------ Main Data Loader ------------------

def load_and_process_all_data(h5_filepath, save_r=1):
    loaded_data = {}

    with h5py.File(h5_filepath, 'r') as f:
        for qubit_group in f.keys():
            try:
                # Convert group names like "Q1" to a zero-based index.
                qubit_index = int(qubit_group[1:]) - 1
            except ValueError:
                print(f"Warning: Could not parse qubit index from group name '{qubit_group}'. Skipping.")
                continue

            qubit_data = {}
            group = f[qubit_group]
            for dataset_name in group.keys():
                try:
                    raw_value = group[dataset_name][()]
                except Exception as e:
                    print(f"Warning: Could not load dataset '{dataset_name}' in group '{qubit_group}'. Error: {e}")
                    continue

                # Decode and process the dataset value.
                decoded_value = ensure_str(raw_value)
                if dataset_name in process_map:
                    try:
                        processed_value = process_map[dataset_name](decoded_value)
                    except Exception as e:
                        print(f"Warning: Processing dataset '{dataset_name}' in group '{qubit_group}' failed with error: {e}")
                        processed_value = decoded_value
                else:
                    processed_value = decoded_value

                # For config keys, store the processed value directly.
                if dataset_name in ['Exp Config', 'Syst Config']:
                    qubit_data[dataset_name] = processed_value
                else:
                    replicated = [processed_value] * save_r
                    qubit_data[dataset_name] = unwrap_singleton_list(replicated)

            loaded_data[qubit_index] = qubit_data

    return loaded_data

def get_data_for_key(loaded_data, key, qubit_index=None):
    """
    Extract data associated with a given key from the processed loaded_data dictionary.

    Parameters:
        loaded_data (dict): The nested dictionary returned from load_and_process_all_data.
        key (str): The dataset key to retrieve data for.
        qubit_index (int, optional): Specific qubit index to extract data from.
            If None, the function returns data for the key across all qubits.

    Returns:
        If qubit_index is provided:
            The list of processed data for that qubit and key, or None if the key is not found.
        If qubit_index is None:
            A dictionary mapping each qubit index to the corresponding data for that key.
            Quibit groups without the specified key are omitted.
    """
    if qubit_index is not None:
        qubit_data = loaded_data.get(qubit_index, {})
        return qubit_data.get(key, None)
    else:
        result = {}
        for q_index, data in loaded_data.items():
            if key in data:
                result[q_index] = data[key]
        return result

def get_res_freqs_and_dates(number_of_qubits, filepaths, verbose=False, plot_title_prefix="", save_figs=False, outer_folder=None, expt_name=None, round_num=None, fig_quality=50):
    all_q_min_res_freqs = {i: [] for i in range(number_of_qubits)}
    all_q_res_freqs_time = {i: [] for i in range(number_of_qubits)}

    for path_name in filepaths:
        for q_key in range(number_of_qubits):
            # Load and process the data for this file and qubit
            data = load_and_process_all_data(path_name, save_r=1)
            freq_pts = get_data_for_key(data, key='freq_pts', qubit_index=q_key)
            freq_center = get_data_for_key(data, key='freq_center', qubit_index=q_key)
            Amps = get_data_for_key(data, key='Amps', qubit_index=q_key)
            date = get_data_for_key(data, key='Dates', qubit_index=q_key)
            config = get_data_for_key(data, key='Exp Config', qubit_index=q_key)


            if Amps[0] is not None:
                filename = path_name.split('Res_ge/')[-1].split('_Res_results')[0]
                if save_figs:
                    min_freqs = plot_spectroscopy(q_key, freq_pts, freq_center, Amps, round_num=0, config=config,
                                      outerFolder=outer_folder, fig_quality=fig_quality,
                                          expt_name="res_spec", experiment=None, save_figs=True, reloaded_config=None, fig_filename=filename)
                else:
                    try:
                        center = freq_center[q_key]
                    except (TypeError, IndexError):
                        center = freq_center
                    freqs = np.asarray(freq_pts) + center
                    freqs = freqs[0]
                    idx_min = np.argmin(Amps[q_key])
                    min_freqs = freqs[idx_min]

                if isinstance(min_freqs, list):
                    all_q_min_res_freqs[q_key].extend(min_freqs)
                else:
                    all_q_min_res_freqs[q_key].append(min_freqs)
                if isinstance(date, list):
                    all_q_res_freqs_time[q_key].extend(date)
                else:
                    all_q_res_freqs_time[q_key].append(date)

    #return dict with all qubits and a list of the frequencies for the resonator for that qubit and measurement times
    return all_q_min_res_freqs, all_q_res_freqs_time

def get_ss_info_and_dates(number_of_qubits, filepaths, verbose=False, plot_title_prefix="", save_figs=False, outer_folder=None, expt_name=None, round_num=None, fig_quality=50):
    fidelities = {i: [] for i in range(number_of_qubits)}
    thresholds = {i: [] for i in range(number_of_qubits)}
    angles = {i: [] for i in range(number_of_qubits)}
    date_times = {i: [] for i in range(number_of_qubits)}

    for path_name in filepaths:
        for q_key in range(number_of_qubits):
            # Load and process the data for this file and qubit
            data = load_and_process_all_data(path_name, save_r=1)
            I_g = get_data_for_key(data, key='I_g', qubit_index=q_key)
            Q_g = get_data_for_key(data, key='Q_g', qubit_index=q_key)
            I_e = get_data_for_key(data, key='I_e', qubit_index=q_key)
            Q_e = get_data_for_key(data, key='Q_e', qubit_index=q_key)
            angle = get_data_for_key(data, key='Angle', qubit_index=q_key)
            fidelity = get_data_for_key(data, key='Fidelity', qubit_index=q_key)
            date = get_data_for_key(data, key='Dates', qubit_index=q_key)
            config = get_data_for_key(data, key='Exp Config', qubit_index=q_key)

            if I_g[0] is not None:
                filename = path_name.split('SS_ge/')[-1].split('_SS_results')[0]
                if save_figs:
                    fid, threshold, theta, ig_new, ie_new = plot_ssf_histogram(I_g, Q_g, I_e, Q_e, config, outerFolder=outer_folder, qubit_index=0, round_num=0,
                                           expt_name="ss_repeat_meas", plot=True, fig_quality=fig_quality, fig_filename=filename)
                    theta_deg = theta*180/np.pi
                else:
                    fid, threshold, theta = compute_ssf_metrics(I_g, Q_g, I_e, Q_e, config)
                    theta_deg = theta * 180 / np.pi
                if isinstance(date, list):
                    date_times[q_key].extend(date)
                else:
                    date_times[q_key].append(date)


                fidelities[q_key].append(fid)
                angles[q_key].append(theta_deg)
                thresholds[q_key].append(threshold)

    #return dict with all qubits and a list of the frequencies for the resonator for that qubit and measurement times
    return fidelities, angles, thresholds, date_times

def get_freqs_and_dates(number_of_qubits, filepaths, verbose=False, plot_title_prefix="", save_figs=False, outer_folder=None, expt_name=None, round_num=None, fig_quality=50, fit_err_threshold=0):
    """
    Processes data files to extract spectroscopy frequencies, fitting errors, and dates.

    Optionally, if `plot` is True, plots the I/Q data as they are loaded by calling
    plot_spec_results_individually.

    Parameters:
      number_of_qubits (int): Number of qubits to process.
      filepaths (list): List of file paths to load data from.
      verbose (bool): Verbosity flag for fitting.
      plot (bool): If True, plot the data as it is loaded.
      fit_func (callable): Function to perform the Lorentzian fit; if provided, spectroscopic fits
                           will be plotted.
      plot_title_prefix (str): Optional prefix to add to the plot title.
      config (dict): Optional configuration dictionary containing keys such as 'reps' and 'rounds'.
      save_figs (bool): If True, the plots will be saved.
      outer_folder (str): Folder path to save the figures (required if save_figs is True).
      expt_name (str): Experiment name (required if save_figs is True).
      round_num (int): Round number (required if save_figs is True).
      h5_filename (str): H5 file name (required if save_figs is True).
      fig_quality (int): DPI quality for saving the figure.

    Returns:
      frequencies (dict): Dictionary mapping each qubit index to a list of measured center frequencies.
      fit_errs (dict): Dictionary mapping each qubit index to a list of fit errors.
      date_times (dict): Dictionary mapping each qubit index to a list of date/time stamps.
    """
    if save_figs:
        create_folder_if_not_exists(outer_folder)

    frequencies = {i: [] for i in range(number_of_qubits)}
    fit_errs = {i: [] for i in range(number_of_qubits)}
    date_times = {i: [] for i in range(number_of_qubits)}

    for path_name in filepaths:
        for q_key in range(number_of_qubits):
            # Load and process the data for this file and qubit
            data = load_and_process_all_data(path_name, save_r=1)
            I = get_data_for_key(data, key='I', qubit_index=q_key)
            Q = get_data_for_key(data, key='Q', qubit_index=q_key)
            freqs = get_data_for_key(data, key='Frequencies', qubit_index=q_key)
            date = get_data_for_key(data, key='Dates', qubit_index=q_key)
            config = get_data_for_key(data, key='Exp Config', qubit_index=q_key)

            if I is not None:
                # Check that data are valid numerical arrays (assuming the first element is float)
                if isinstance(I[0], float) and isinstance(Q[0], float) and freqs is not None:
                    # Perform the Lorentzian fit and gather results.
                    I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = get_lorentzian_fits(I, Q, freqs, verbose=verbose)
                    if I_fit is not None:
                        if fit_err < fit_err_threshold:
                            frequencies[q_key].append(largest_amp_curve_mean)
                            fit_errs[q_key].append(fit_err)
                            if isinstance(date, list):
                                date_times[q_key].extend(date)
                            else:
                                date_times[q_key].append(date)
                            # If plotting is enabled, call the plotting routine.
                            if save_figs:
                                # Create a title using an optional prefix, file basename, and qubit index.

                                plot_spec_results_individually(I, Q, freqs,
                                                               largest_amp_curve_mean=largest_amp_curve_mean,
                                                               largest_amp_curve_fwhm= largest_amp_curve_fwhm,
                                                               I_fit=I_fit,
                                                               Q_fit=Q_fit, title_start=plot_title_prefix,
                                                               spec=True,
                                                               qubit_index=q_key, config=config,
                                                               outer_folder=outer_folder, expt_name=expt_name,
                                                               round_num=round_num, h5_filename=path_name,
                                                               fig_quality=fig_quality)

                        # Clean up variables for this iteration.
                        del I, Q, freqs, date, data, largest_amp_curve_mean, I_fit, Q_fit, fit_err

    return frequencies, fit_errs, date_times

def get_decoherence_time_and_dates(number_of_qubits, filepaths, decoherence_type='T1', discard_values_over=None, fit_err_threshold=0, save_figs=False, outer_folder=None, expt_name=None, round_num=None, fig_quality=100, plot_title_prefix="", thresholding=False, discard_low_signal_values = True):
    """
    Processes data files to extract T1 decay constants (decoherence times), fit errors, and dates.

    If save_figs is True, individual T1 fit plots are saved to disk.

    Parameters:
      number_of_qubits (int): Number of qubits to process.
      filepaths (list): List of data file paths.
      decoherence_type (str): Type of decoherence ('T1').
      discard_values_over (float): If provided, discard T1 values above this threshold.
      fit_err_threshold (float): Discard fits with error above this threshold.
      save_figs (bool): If True, save individual T1 plots.
      outer_folder (str): Folder to save figures (required if save_figs is True).
      expt_name (str): Experiment name (required if save_figs is True).
      round_num (int): Round number (required if save_figs is True).
      fig_quality (int): DPI for saved figures.
      plot_title_prefix (str): Optional title prefix for plots.

    Returns:
      decoherence_times (dict): Mapping qubit index to list of T1 values.
      fit_errs (dict): Mapping qubit index to list of fit errors.
      date_times (dict): Mapping qubit index to list of date/time stamps.
    """
    # If saving figures, ensure required parameters are provided.
    if save_figs:
        if outer_folder is None or expt_name is None:
            raise ValueError("outer_folder and expt_name must be provided if save_figs is True")
        # Assume create_folder_if_not_exists is defined elsewhere.
        if not os.path.exists(outer_folder):
            os.makedirs(outer_folder)

    decoherence_times = {i: [] for i in range(number_of_qubits)}
    fit_errs = {i: [] for i in range(number_of_qubits)}
    date_times = {i: [] for i in range(number_of_qubits)}

    decoherence_times_good_data = {i: [] for i in range(number_of_qubits)}
    fit_errs_good_data = {i: [] for i in range(number_of_qubits)}
    date_times_good_data = {i: [] for i in range(number_of_qubits)}

    for path_name in filepaths:
        for q_key in range(number_of_qubits):
            data = load_and_process_all_data(path_name, save_r=1)
            I = get_data_for_key(data, key='I', qubit_index=q_key)
            Q = get_data_for_key(data, key='Q', qubit_index=q_key)
            delay_times = get_data_for_key(data, key='Delay Times', qubit_index=q_key)
            date = get_data_for_key(data, key='Dates', qubit_index=q_key)

            if I is not None and len(I) > 0:
                # Check that data are valid numerical arrays.
                if isinstance(I[0], float) or isinstance(Q[0], float):
                    if 'T1' in decoherence_type:
                        try:
                            t1_fit_curve, T1_err, T1_est, plot_sig = t1_fit(I, Q, delay_times, return_everything=True)
                        except:
                            continue
                        # Optionally discard values above thresholds.
                        if discard_values_over is not None:
                            if T1_est > discard_values_over or T1_err > fit_err_threshold:
                                continue
                        decoherence_time = T1_est
                        fit_err = T1_err

                    decoherence_times[q_key].append(decoherence_time)
                    fit_errs[q_key].append(fit_err)

                    # Flatten the date list.
                    if isinstance(date, list):
                        date_times[q_key].extend(date)
                    else:
                        date_times[q_key].append(date)

                    # Save the plot if requested.
                    if save_figs:
                        title = plot_title_prefix + f" Qubit Q{q_key + 1}"
                        plot_t1_results_individually(
                            I, Q, delay_times, title_start=title,
                            t1_fit_curve=t1_fit_curve, T1_est=T1_est, T1_err=T1_err,
                            plot_sig=plot_sig, qubit_index=q_key,
                            outer_folder=outer_folder, expt_name=expt_name,
                            round_num=round_num, h5_filename=path_name,
                            fig_quality=fig_quality, thresholding=thresholding
                        )

                    if discard_low_signal_values:
                        mag_I = abs(min(I) - max(I))
                        mag_Q = abs(min(Q) - max(Q))
                        if mag_I < 0.3:
                            if mag_Q < 0.3:
                                continue
                        else:
                            decoherence_times_good_data[q_key].append(decoherence_time)
                            fit_errs_good_data[q_key].append(fit_err)

                            # Flatten the date list.
                            if isinstance(date, list):
                                date_times_good_data[q_key].extend(date)
                            else:
                                date_times_good_data[q_key].append(date)




                    del data, decoherence_time, fit_err, date
    if discard_low_signal_values:
        return decoherence_times, fit_errs, date_times, decoherence_times_good_data, fit_errs_good_data, date_times_good_data
    else:
        return decoherence_times, fit_errs, date_times