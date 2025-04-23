"""
plotting.py
===========

This module contains functions for plotting various types of analysis data,
including spectroscopy, scatter plots, histograms with Gaussian fits, cumulative
distributions, error vs. value scatter plots, Allan deviation, and Welch spectral
density. The module is designed for use with experimental data analysis and
provides several helper routines to create high-quality figures.

Dependencies:
    - numpy
    - matplotlib
    - scipy.stats (for norm)
    - scipy.signal (for welch)
    - scipy.optimize.curve_fit
    - allantools (for Allan deviation)
    - datetime, os, math
    - create_folder_if_not_exists from ..utils.file_helpers

Usage Example:
    from qicklab.analysis.plotting import plot_spectroscopy, plot_allan_deviation
    # Prepare your data and then call the desired plotting functions.
"""

import os
import math
import datetime
import allantools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from matplotlib.ticker import StrMethodFormatter

from scipy.stats import norm
from scipy.signal import welch, lombscargle
from scipy.optimize import curve_fit

from ..utils.ana_utils import convert_datetimes_to_seconds, split_into_continuous_segments, get_longest_continuous_segment, sort_date_time_data
from ..utils.data_utils import remove_none_values
from ..utils.file_utils import create_folder_if_not_exists
from  .fit_functions import allan_deviation_model


def plot_ssf_histogram(ig, qg, ie, qe, cfg, outerFolder=None, qubit_index=0, round_num=0, expt_name="ss_repeat_meas", plot=True, fig_quality=100, fig_filename=None):
    """
    Plot single-shot fidelity (SSF) histogram from IQ data.

    This function creates three subplots:
      1. Unrotated IQ scatter plot for ground (g) and excited (e) states.
      2. Rotated IQ scatter plot, computed using the rotation angle that aligns the blobs along the I–axis.
      3. A histogram of the rotated I data for both states with the fidelity computed from the
         cumulative histogram contrast.

    Parameters
    ----------
    ig : array-like
        I data for the ground state.
    qg : array-like
        Q data for the ground state.
    ie : array-like
        I data for the excited state.
    qe : array-like
        Q data for the excited state.
    cfg : dict
        Configuration dictionary. Must include key "steps" for histogram bin calculation.
    outerFolder : str or None, optional
        Top-level directory in which to save the figure. If None, the figure is not saved.
    qubit_index : int, optional
        Qubit index used in naming the saved figure (default is 0).
    round_num : int, optional
        Experiment round number used in naming the saved figure (default is 0).
    expt_name : str, optional
        Experiment name used in naming the saved figure (default is "ss_repeat_meas").
    plot : bool, optional
        If True, the plots are generated and the figure is saved; if False, only the computations are done.
    fig_quality : int, optional
        DPI used when saving the figure (default is 100).
    fig_filename : str or None, optional
        If provided, this filename is used to save the figure (overrides generated filename).

    Returns
    -------
    tuple
        (fid, threshold, theta, ig_new, ie_new) where:
          fid : float
              Computed fidelity from the histogram contrast.
          threshold : float
              I value corresponding to the maximum contrast.
          theta : float
              Rotation angle (in radians) used to align the IQ data.
          ig_new : array-like
              Rotated I data for the ground state.
          ie_new : array-like
              Rotated I data for the excited state.
    """
    ig = np.asarray(ig)
    qg = np.asarray(qg)
    ie = np.asarray(ie)
    qe = np.asarray(qe)

    # Determine the number of bins for the histogram.
    numbins = round(math.sqrt(float(cfg['Readout_Optimization']["steps"])))

    # Compute medians for the unrotated data.
    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
        fig.tight_layout()

        # Plot unrotated IQ data.
        axs[0].scatter(ig, qg, label='g', marker='*', color='b')
        axs[0].scatter(ie, qe, label='e', marker='*', color='r')
        axs[0].scatter(xg, yg, marker='o', color='k')
        axs[0].scatter(xe, ye, marker='o', color='k')
        axs[0].set_xlabel('I (a.u.)')
        axs[0].set_ylabel('Q (a.u.)')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Unrotated')
        axs[0].axis('equal')

    # Compute the rotation angle to align the two blobs.
    theta = -np.arctan2((ye - yg), (xe - xg))

    # Rotate the IQ data.
    ig_new = ig * np.cos(theta) - qg * np.sin(theta)
    qg_new = ig * np.sin(theta) + qg * np.cos(theta)
    ie_new = ie * np.cos(theta) - qe * np.sin(theta)
    qe_new = ie * np.sin(theta) + qe * np.cos(theta)

    # Compute new medians for the rotated data.
    xg_new, yg_new = np.median(ig_new), np.median(qg_new)
    xe_new, ye_new = np.median(ie_new), np.median(qe_new)

    # Define I-axis limits for the histogram.
    xlims = [np.min(ig_new), np.max(ie_new)]

    if plot:
        # Plot rotated IQ data.
        axs[1].scatter(ig_new, qg_new, label='g', marker='*', color='b')
        axs[1].scatter(ie_new, qe_new, label='e', marker='*', color='r')
        axs[1].scatter(xg_new, yg_new, marker='o', color='k')
        axs[1].scatter(xe_new, ye_new, marker='o', color='k')
        axs[1].set_xlabel('I (a.u.)')
        axs[1].legend(loc='lower right')
        axs[1].set_title(f'Rotated Theta: {round(theta, 5)}')
        axs[1].axis('equal')

        # Plot histogram of rotated I data.
        ng, binsg, _ = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.5)
        ne, binse, _ = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.5)
        axs[2].set_xlabel('I (a.u.)')
    else:
        ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

    # Compute fidelity from the overlap of cumulative histograms.
    contrast = np.abs((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * np.sum(ng) + 0.5 * np.sum(ne)))
    tind = contrast.argmax()
    threshold = binsg[tind]
    fid = contrast[tind]

    if plot:
        # Update the histogram subplot title with fidelity.
        axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
        # Save the figure if an outerFolder is provided.
        if outerFolder is not None:
            create_folder_if_not_exists(outerFolder)
            outerFolder_expt = os.path.join(outerFolder, "ss_repeat_meas")
            create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(qubit_index + 1))
            create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{round_num}_Q_{qubit_index + 1}_{formatted_datetime}_{expt_name}.png")
            if fig_filename is not None:
                file_name = os.path.join(outerFolder_expt, fig_filename)
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

    return fid, threshold, theta, ig_new, ie_new


def compute_ssf_metrics(ig, qg, ie, qe, cfg):
    """
    Compute single-shot fidelity (SSF) metrics from IQ data without plotting.

    This function calculates the rotation angle required to align the IQ data, rotates
    the ground (g) and excited (e) state data accordingly, and then computes the fidelity
    based on the overlap of the cumulative histograms of the rotated I data.

    Parameters
    ----------
    ig : array-like
        I data for the ground state.
    qg : array-like
        Q data for the ground state.
    ie : array-like
        I data for the excited state.
    qe : array-like
        Q data for the excited state.
    cfg : dict
        Configuration dictionary containing at least the key "steps" to determine the number
        of histogram bins (e.g., steps in the measurement).

    Returns
    -------
    tuple
        (fid, threshold, theta) where:
          fid : float
              Computed fidelity from the histogram contrast.
          threshold : float
              I value corresponding to the maximum histogram contrast.
          theta : float
              Rotation angle (in radians) used to align the IQ data.
    """
    # Determine the number of histogram bins.

    ig= np.asarray(ig)
    qg = np.asarray(qg)
    ie = np.asarray(ie)
    qe = np.asarray(qe)
    numbins = round(math.sqrt(float(cfg['Readout_Optimization']["steps"])))

    # Compute medians for the unrotated data.
    xg, yg = np.median(ig), np.median(qg)
    xe, ye = np.median(ie), np.median(qe)

    # Compute the rotation angle to align the IQ blobs.
    theta = -np.arctan2((ye - yg), (xe - xg))

    # Rotate the IQ data.
    ig_new = ig * np.cos(theta) - qg * np.sin(theta)
    ie_new = ie * np.cos(theta) - qe * np.sin(theta)

    # Define I-axis limits for the histogram based on rotated I data.
    xlims = [np.min(ig_new), np.max(ie_new)]

    # Compute histograms for rotated I data.
    ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
    ne, _ = np.histogram(ie_new, bins=numbins, range=xlims)

    # Compute the cumulative contrast between the histograms.
    contrast = np.abs((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * np.sum(ng) + 0.5 * np.sum(ne)))
    tind = contrast.argmax()
    threshold = binsg[tind]
    fid = contrast[tind]

    return fid, threshold, theta


def plot_spec_results_individually(I, Q, freqs, title_start='', spec=False, largest_amp_curve_mean=None, largest_amp_curve_fwhm=None, I_fit=None, Q_fit=None, qubit_index=None, config=None, outer_folder=None, expt_name=None, round_num=None, h5_filename = None, fig_quality=100):
    """
    Plots I and Q data with optional Lorentzian fitting for qubit spectroscopy.

    Parameters:
      I, Q: Arrays of measured amplitudes.
      freqs: Frequency values (list or numpy array).
      title: Base title for the plot.
      spec (bool): If True, perform Lorentzian fit and add spectroscopy info to title.
      fit_func (callable): Function to perform the Lorentzian fit. Should accept (I, Q, freqs, freq_q)
                           and return (mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err).
                           Required if spec is True.
      qubit_index (int): Qubit index (0-indexed) to be used in the title when spec is True.
      config (dict): Optional configuration dictionary containing 'reps' and 'rounds' keys.
      save_figs (bool): If True, save the figure.
      outer_folder (str): Folder path to save the figure (required if save_figs is True).
      expt_name (str): Experiment name (required if save_figs is True).
      round_num (int): Round number (required if save_figs is True).
      fig_quality (int): DPI for saving the figure.

    Returns:
      If spec is True, returns (largest_amp_curve_mean, I_fit, Q_fit). Otherwise, returns None.
    """
    freqs = np.array(freqs)

    if spec:
        if qubit_index is None:
            raise ValueError("When spec=True, qubit_index must be provided.")
        if largest_amp_curve_mean is None:
            raise ValueError("When spec=True, largest_amp_curve_mean must be provided.")
        if I_fit is None:
            raise ValueError("When spec=True, I_fit must be provided.")

    else:
        # No fitting is done if spec is False. replace later with rabi params or something
        I_fit = Q_fit = largest_amp_curve_mean = largest_amp_curve_fwhm = None

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.rcParams.update({'font.size': 18})

    # Plot the I data
    ax1.plot(freqs, I, label='I', linewidth=2)
    ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend()

    # Plot the Q data
    ax2.plot(freqs, Q, label='Q', linewidth=2)
    ax2.set_xlabel("Frequency (MHz)", fontsize=20)
    ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend()

    if spec:
        # Plot the fitted curves and mark the center frequency
        ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
        ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)
        ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
        ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

    # Determine the final title text
    if spec:
        spec_details = f" Qubit Q{qubit_index + 1}, {largest_amp_curve_mean:.5f} MHz, FWHM: {round(largest_amp_curve_fwhm, 1)}"
        if config is not None:
            spec_details += f", {config['qubit_spec_ge']['reps']}*{config['qubit_spec_ge']['rounds']} avgs"
        final_title = title_start + spec_details
    else:
        final_title = title_start

    # Center the title above the plot area
    plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
    fig.text(plot_middle, 0.98, final_title, fontsize=24, ha='center', va='top')

    # Adjust layout and margins
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the figure if requested

    if outer_folder is None:
        raise ValueError("outer_folder must be provided to save the figure")
    outerFolder_expt = os.path.join(outer_folder, expt_name)
    os.makedirs(outerFolder_expt, exist_ok=True)
    outerFolder_expt = os.path.join(outerFolder_expt, f'Q{qubit_index}')
    os.makedirs(outerFolder_expt, exist_ok=True)
    outerFolder_expt = os.path.join(outerFolder_expt, 'QSpec_ge')
    os.makedirs(outerFolder_expt, exist_ok=True)

    h5_filename = h5_filename.split('/')[-1].split('.')[0]
    file_name = os.path.join(outerFolder_expt,
                             f"{h5_filename}_plot.png")
    fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')

    plt.close(fig)

    if spec:
        return largest_amp_curve_mean, I_fit, Q_fit
    else:
        return None


def plot_t1_results_individually(I, Q, delay_times, title_start='', t1_fit_curve=None, T1_est=None, T1_err=None, plot_sig=None, qubit_index=None, outer_folder=None, expt_name=None, round_num=None, h5_filename=None, fig_quality=100, thresholding = False):
    """
    Plots I and Q data versus delay time and overlays the T1 exponential fit on the
    appropriate subplot.

    The plot is saved to a subfolder (outer_folder/expt_name) with a filename based on h5_filename.
    """
    if thresholding:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.rcParams.update({'font.size': 18})

    # Plot I data.
    ax1.plot(delay_times, I, label='I', linewidth=2)
    if thresholding:
        ax1.set_ylabel("First excited state / Ground State", fontsize=20)
    else:
        ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend()

    if not thresholding:
        # Plot Q data.
        ax2.plot(delay_times, Q, label='Q', linewidth=2)
        ax2.set_xlabel("Delay time (us)", fontsize=20)
        ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.legend()

    # Overlay the fitted exponential curve on the correct subplot.
    if t1_fit_curve is not None and T1_est is not None:
        if plot_sig == 'I':
            ax1.plot(delay_times, t1_fit_curve, 'r--', label='Exponential Fit')
        elif plot_sig == 'Q' and not thresholding:
            ax2.plot(delay_times, t1_fit_curve, 'r--', label='Exponential Fit')

    # Create and set the title.
    final_title = title_start
    if T1_est is not None:
        final_title += f", T1={T1_est:.2f} us, err={T1_err:.2f}"
    plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
    fig.text(plot_middle, 0.98, final_title, fontsize=24, ha='center', va='top')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if outer_folder is None:
        raise ValueError("outer_folder must be provided to save the figure")
    # Create the experiment folder if needed.
    outerFolder_expt = os.path.join(outer_folder, expt_name)
    os.makedirs(outerFolder_expt, exist_ok=True)
    outerFolder_expt = os.path.join(outerFolder_expt, f'Q{qubit_index}')
    os.makedirs(outerFolder_expt, exist_ok=True)
    outerFolder_expt = os.path.join(outerFolder_expt, 'T1_ge')
    os.makedirs(outerFolder_expt, exist_ok=True)

    # Use the base name of the h5 file (if provided) for naming the plot.
    if h5_filename is not None:
        base_name = os.path.basename(h5_filename).split('.')[0]
    else:
        base_name = "plot"
    file_name = os.path.join(outerFolder_expt, f"{base_name}_Q{qubit_index + 1}_T1_plot.png")
    fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
    plt.close(fig)


def plot_spectroscopy(qubit_index, fpts, fcenter, amps, round_num=0, config=None, outerFolder=None, expt_name="res_spec", experiment=None, save_figs=False, reloaded_config=None, fig_quality=100, title_word="Res", xlabel="Freq (MHz)", ylabel="Amplitude (a.u.)", find_min=True, plot_min_line=True, include_min_in_title=True, return_min=True, fig_filename=None):
    """
    Plot spectroscopy data and optionally return the extracted resonance (minimum) frequencies.

    The function creates subplots (one per dataset in `amps`) that display amplitude versus
    frequency. For each subplot, the function calculates the frequency axis by offsetting the
    provided frequency points (`fpts`) by a center frequency. It also optionally finds the minimum
    amplitude (resonance frequency) and marks it on the plot.

    Parameters
    ----------
    qubit_index : int
        Legacy parameter used in generating a file name when saving figures.
    fpts : np.array
        Array of frequency offsets used in the sweep.
    fcenter : array-like or scalar
        Center frequency (or frequencies) for the datasets. If an array, its length should match
        the number of datasets; if scalar, it is applied to all datasets.
    amps : np.array
        2D array with shape (n_datasets, n_points) containing amplitude data.
    round_num : int, optional
        Experiment round (or iteration) number (default is 0).
    config : dict or None, optional
        Configuration dictionary used to build the overall title.
    outerFolder : str or None, optional
        Top-level directory in which to save figures.
    expt_name : str, optional
        Experiment name (used in file naming; default is "res_spec").
    experiment : object or None, optional
        If provided, indicates a live experiment; used to generate title text.
    save_figs : bool, optional
        If True, the figure is saved to disk.
    reloaded_config : dict or None, optional
        Alternative configuration if no live experiment is provided.
    fig_quality : int, optional
        DPI for saving the figure (default is 100).
    title_word : str, optional
        Word used as the prefix in each subplot’s title (default is "Resonator").
    xlabel : str, optional
        Label for the x–axis (default is "Frequency (MHz)").
    ylabel : str, optional
        Label for the y–axis (default is "Amplitude (a.u.)").
    find_min : bool, optional
        If True, for each subplot the minimum amplitude is computed (default is True).
    plot_min_line : bool, optional
        If True (and find_min is True), a vertical line is plotted at the minimum point.
    include_min_in_title : bool, optional
        If True (and find_min is True), the computed minimum frequency is included in the subplot title.
    return_min : bool, optional
        If True, returns a list of computed minimum frequencies for each dataset; otherwise, returns None.
    fig_filename : str or None, optional
        If provided, this name (within `outerFolder`) is used to save the figure.

    Returns
    -------
    list or None
        If both `find_min` and `return_min` are True, returns a list of extracted resonance (minimum)
        frequencies for each dataset. Otherwise, returns None.
    """
    create_folder_if_not_exists(outerFolder)

    # Determine the number of datasets from the amplitude data.
    n_datasets = np.asarray(amps).shape[0]

    # Calculate a reasonable grid size for subplots.
    n_cols = math.ceil(math.sqrt(n_datasets))
    n_rows = math.ceil(n_datasets / n_cols)

    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    # Update plot font sizes.
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    min_vals = []  # to store computed minimum frequencies (if any)

    # Loop over each dataset.
    for i in range(n_datasets):
        # Use the provided center frequency for this dataset; if fcenter is scalar, use it for all.
        try:
            center = fcenter[i]
        except (TypeError, IndexError):
            center = fcenter

        # Create frequency axis for this dataset.
        freqs = np.asarray(fpts) + center
        freqs = freqs[0]


        # Select the subplot.
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(freqs, amps[i], '-', linewidth=1.5)


        # Optionally find and mark the minimum.
        if find_min:
            idx_min = np.argmin(amps[i])
            freq_min = freqs[idx_min]
            if plot_min_line:
                plt.axvline(freq_min, linestyle='--', color='orange', linewidth=1.5)
            if include_min_in_title:
                title = f"{title_word} {i + 1} {freq_min:.3f} MHz"
            else:
                title = f"{title_word} {i + 1}"
            min_vals.append(freq_min)
        else:
            title = f"{title_word} {i + 1}"

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        # Add a little extra space below the minimum amplitude.
        ylo, yhi = plt.ylim()
        plt.ylim(ylo - 0.05 * (yhi - ylo), yhi)

    # Set a suptitle based on provided configuration.
    if experiment is not None:
        plt.suptitle(f"MUXed {title_word.lower()} spectroscopy {config['reps']}*{config['rounds']} avgs",
                     fontsize=24, y=0.95)
    elif reloaded_config is not None:
        plt.suptitle(
            f"MUXed {title_word.lower()} spectroscopy {reloaded_config['reps']}*{reloaded_config['rounds']} avgs",
            fontsize=24, y=0.95)
    else:
        plt.suptitle(f"MUXed {title_word.lower()} spectroscopy", fontsize=24, y=0.95)

    plt.tight_layout()

    # Optionally save the figure.
    if save_figs and outerFolder is not None:
        if fig_filename is None:
            # Create a subfolder based on expt_name.
            # Create the experiment folder if needed.
            outerFolder_expt = os.path.join(outerFolder, expt_name)
            os.makedirs(outerFolder_expt, exist_ok=True)
            outerFolder_expt = os.path.join(outerFolder_expt, f'Q{qubit_index}')
            os.makedirs(outerFolder_expt, exist_ok=True)
            outerFolder_expt = os.path.join(outerFolder_expt, 'SS_ge')
            os.makedirs(outerFolder_expt, exist_ok=True)

            if not os.path.exists(outerFolder_expt):
                os.makedirs(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            # File name incorporates the round number and qubit_index.
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{round_num}_Q_{qubit_index + 1}_{formatted_datetime}_{expt_name}.png")
        else:
            file_name = os.path.join(outerFolder, fig_filename)
        plt.savefig(file_name, dpi=fig_quality)

    plt.close()

    if return_min and find_min:
        return min_vals
    else:
        return None


def scatter_plot_vs_time_with_fit_errs(date_times, y_data, number_of_qubits, y_data_name='', save_name='', save_folder_path='', y_label='', show_legends=False, fit_err=None, final_figure_quality=100):
    """
    Create scatter plots versus time for multiple qubits, including error bars.

    This function plots data for each qubit on separate subplots arranged dynamically based on
    the number of qubits that have non-empty data. The function sorts the data based on time,
    adds error bars, and optionally saves the figure.

    Parameters
    ----------
    date_times : list of lists
        Each element is a list of date/time values for a qubit.
    y_data : list of lists
        Each element is a list of numerical data values for a qubit.
    fit_err : list of lists
        Each element is a list of error values corresponding to the y_data for a qubit.
    number_of_qubits : int
        Total number of qubits provided (some may have no data).
    y_data_name : str, optional
        Name of the data (used in the overall title).
    save_name : str, optional
        Base name for the saved figure file.
    save_folder_path : str, optional
        Folder path where the figure will be saved.
    y_label : str, optional
        Label for the y–axis.
    show_legends : bool, optional
        If True, displays legends on the subplots.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 100).

    Returns
    -------
    None
    """
    create_folder_if_not_exists(save_folder_path)

    font = 14
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

    # Determine which qubits actually have data.
    valid_indices = []
    for i in range(number_of_qubits):
        if len(date_times[i]) > 0 and len(y_data[i]) > 0:
            valid_indices.append(i)

    effective_qubits = len(valid_indices)
    if effective_qubits == 0:
        print("No data available to plot.")
        return

    # Dynamically set grid dimensions; for example, up to 3 columns.
    n_cols = min(3, effective_qubits)
    n_rows = math.ceil(effective_qubits / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Ensure axes is a flat list.
    if effective_qubits == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plt.suptitle(f'{y_data_name} vs Time', fontsize=font)

    # Loop over each valid qubit’s data.
    for j, i in enumerate(valid_indices):
        ax = axes[j]
        ax.set_title(f"Qubit {i + 1}", fontsize=font)

        x = date_times[i]  # List of date/time values.
        y = y_data[i]
        if fit_err is not None:
            err = fit_err[i]

        # Combine data and sort by time.
        if fit_err is not None:
            combined = list(zip(x, y, err))
        else:
            combined = list(zip(x, y))
        combined.sort(key=lambda tup: tup[0])

        # If after sorting there is no data, hide the subplot.
        if len(combined) == 0:
            ax.set_visible(False)
            continue

        # Unpack the sorted data and remove any None values.
        if fit_err is not None:
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            sorted_y, sorted_x, sorted_err = remove_none_values(sorted_y, list(sorted_x), sorted_err)
        else:
            sorted_x, sorted_y = zip(*combined)
            sorted_x = np.array(sorted_x)
            sorted_y, sorted_x = remove_none_values(sorted_y, list(sorted_x))

        # Plot error bars if provided.
        if fit_err is not None:
            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i % len(colors)],
                elinewidth=1,
                capsize=0
            )

        # Plot the scatter data.
        ax.scatter(
            sorted_x, sorted_y,
            s=10,
            color=colors[i % len(colors)],
            alpha=0.5
        )

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis='x', rotation=45)
        ax.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time (Days)', fontsize=font - 2)
        ax.set_ylabel(y_label, fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide any remaining unused subplots in the grid.
    for j in range(len(valid_indices), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_vs_time.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close()


def plot_histogram_with_gaussian(data_dict, date_dict, save_name='', save_folder_path='', data_type='', x_label='', y_label='', title=None, show_legends=False, final_figure_quality=300, n_cols=3, bin_count=50):
    """
    Plot a histogram for each dataset with an overlaid Gaussian fit.

    [Docstring shortened for brevity]
    """
    create_folder_if_not_exists(save_folder_path)

    # Filter out entries with empty data
    filtered_data = {k: v for k, v in data_dict.items() if len(v) > 0}
    # Optionally filter date_dict if needed; here we assume keys are consistent.
    filtered_date = {k: date_dict[k] for k in filtered_data if k in date_dict}

    n_datasets = len(filtered_data)
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if title is None:
        plt.suptitle(f'{data_type} Histograms', fontsize=16)
    else:
        plt.suptitle(title, fontsize=16)

    # If only one subplot exists, wrap it in a list.
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink', 'red', 'cyan', 'magenta', 'yellow']
    gaussian_fit_data = {}
    mean_values = {}
    std_values = {}

    for i, (key, data) in enumerate(filtered_data.items()):
        ax = axes[i]

        # Use the first label from filtered_date if available.
        date_label = filtered_date.get(key, [''])[0] if filtered_date.get(key) else ''

        # Fit a Gaussian if more than one data point is available.
        if len(data) > 1:
            mu, sigma = norm.fit(data)
            mean_values[f"{data_type} {key + 1}"] = mu
            std_values[f"{data_type} {key + 1}"] = sigma

            # Generate points for the Gaussian curve.
            x_vals = np.linspace(min(data), max(data), bin_count)
            pdf_vals = norm.pdf(x_vals, mu, sigma)

            # Obtain histogram data for scaling.
            hist_data, bins = np.histogram(data, bins=bin_count)
            scale_factor = np.diff(bins) * hist_data.sum()
            ax.plot(x_vals, pdf_vals * scale_factor, linestyle='--', linewidth=2,
                    color=colors[i % len(colors)])

            # Plot the histogram.
            ax.hist(data, bins=bin_count, alpha=0.7, color=colors[i % len(colors)],
                    edgecolor='black', label=date_label)

            # For smoother plotting (e.g., cumulative curves), compute more points.
            x_full = np.linspace(min(data), max(data), 2000)
            pdf_full = norm.pdf(x_full, mu, sigma)
            gaussian_fit_data[key] = (x_full, pdf_full)

            title_str = f"Qubit {key + 1}  μ: {mu:.2f} σ: {sigma:.2f}"
        else:
            # If only one data point exists, only a basic histogram is plotted.
            ax.hist(data, bins=10, alpha=0.7, color=colors[i % len(colors)],
                    edgecolor='black', label=date_label)
            title_str = f"{data_type} {key + 1}"

        if show_legends:
            ax.legend()
        ax.set_title(title_str, fontsize=14)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused axes in the grid.
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_hist.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

    return gaussian_fit_data, mean_values, std_values


def plot_cumulative_distribution(data_dict, gaussian_fit_data, save_name='', save_folder_path='', data_type='', x_label='', y_label='', final_figure_quality=300):
    """
    Plot cumulative distributions for each dataset by overlaying the empirical CDF and the cumulative Gaussian fit.

    For each dataset in `data_dict`, the function computes the empirical CDF and, if available,
    overlays the cumulative Gaussian fit derived from `gaussian_fit_data`.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys mapping to arrays/lists of numerical data.
    gaussian_fit_data : dict
        Gaussian fit data for each dataset as returned by plot_histogram_with_gaussian.
    save_name : str, optional
        Base name for saving the figure.
    save_folder_path : str, optional
        Folder path where the figure will be saved.
    data_type : str, optional
        String used in the plot title (e.g., "T1").
    x_label : str, optional
        Label for the x–axis.
    y_label : str, optional
        Label for the y–axis.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 300).

    Returns
    -------
    None
    """
    create_folder_if_not_exists(save_folder_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink', 'red', 'cyan', 'magenta', 'yellow']
    n_datasets = len(data_dict)

    for i in range(n_datasets):
        data = data_dict.get(i, [])
        if len(data) < 1:
            continue
        # Compute the empirical CDF.
        data_sorted = np.sort(data)
        n = len(data_sorted)
        empirical_cdf = np.linspace(1, n, n) / n

        # Plot the cumulative Gaussian fit if available.
        if i in gaussian_fit_data:
            x_full, pdf_full = gaussian_fit_data[i]
            cumulative_gaussian = np.cumsum(pdf_full) / np.sum(pdf_full)
            ax.plot(x_full, cumulative_gaussian, linestyle='--', color=colors[i % len(colors)],
                    label=f'Gauss Fit {data_type} {i + 1}')

        ax.scatter(data_sorted, empirical_cdf, color=colors[i % len(colors)],
                   s=5, label=f'{data_type} {i + 1}')

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f'Cumulative Distribution of {data_type}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(edgecolor='black', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_cumulative.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_error_vs_value(data_dict, error_dict, save_name='', save_folder_path='', data_type='', x_label='', y_label='', show_legends=False, final_figure_quality=300, n_cols=3):
    """
    Create scatter plots of error versus value for each dataset.

    This function creates a grid of subplots where each subplot shows a scatter plot of values
    against their corresponding fit errors for a given dataset. Only datasets with non-empty data
    and error lists are plotted.

    Parameters
    ----------
    data_dict : dict
        Dictionary with keys mapping to arrays/lists of numerical data.
    error_dict : dict
        Dictionary with keys mapping to arrays/lists of error measurements corresponding to the data.
    save_name : str, optional
        Base name for saving the figure.
    save_folder_path : str, optional
        Folder path where the figure will be saved.
    data_type : str, optional
        String used in subplot titles (e.g., "T1").
    x_label : str, optional
        Label for the x–axis.
    y_label : str, optional
        Label for the y–axis.
    show_legends : bool, optional
        If True, display legends on the subplots.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 300).
    n_cols : int, optional
        Number of columns in the subplot grid.

    Returns
    -------
    None
    """
    create_folder_if_not_exists(save_folder_path)

    # Filter keys that exist in both dictionaries and have non-empty data
    valid_keys = [k for k in data_dict if k in error_dict and len(data_dict[k]) > 0 and len(error_dict[k]) > 0]
    n_datasets = len(valid_keys)
    if n_datasets == 0:
        print("No valid data available to plot.")
        return

    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink', 'red', 'cyan', 'magenta', 'yellow']
    plt.suptitle(f'{data_type} Value vs Fit Error', fontsize=16)

    for j, key in enumerate(valid_keys):
        ax = axes[j]
        data = data_dict[key]
        errors = error_dict[key]

        ax.scatter(data, errors, color=colors[j % len(colors)], label=f'{data_type} {key + 1}')
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_title(f'Qubit {key + 1}', fontsize=14)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Hide any unused axes in the grid.
    for j in range(len(valid_keys), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_errs.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_allan_deviation_largest_continuous_sample(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="", save_folder_path='', final_figure_quality=100, plot_all_data_segments=False, stack_segments_yaxis=False, stack_segments_xaxis=False, resample=False, fit=False):
    """
    Plot the overlapping Allan deviation for each qubit, excluding large gaps in the data.
    In addition, also plot and save the segmented raw data versus time on the x-axis,
    using the same colormap and segmentation logic.

    [Docstring shortened for brevity]
    """
    # Ensure the folder exists.
    create_folder_if_not_exists(save_folder_path)
    font = 14

    # Filter out qubits with no data.
    valid_indices = [i for i in range(number_of_qubits)
                     if len(date_times[i]) > 0 and len(vals[i]) > 0]
    effective_n = len(valid_indices)
    if effective_n == 0:
        print("No valid data available to plot.")
        return

    # Create titles for valid qubits.
    titles = [f"Qubit {i + 1}" for i in valid_indices]

    # --- Allan Deviation Plot ---
    n_cols = min(3, effective_n)
    n_rows = math.ceil(effective_n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=False, sharey=False)
    if plot_all_data_segments:
        fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations across Data Sets', fontsize=font)
    else:
        fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations', fontsize=font)
    if effective_n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # List to store segmentation info (for later raw data plotting).
    segmentation_info = []
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

    for j, i in enumerate(valid_indices):
        ax = axes[j]
        ax.set_title(titles[j], fontsize=font)
        data = vals[i]
        dt_objs = date_times[i]

        # Sort the data.
        sorted_times, sorted_vals = sort_date_time_data(dt_objs, data)
        if not sorted_times:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            segmentation_info.append(([], [], [], [], [], []))
            continue

        # Convert sorted times to seconds relative to the first measurement.
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)
        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center',
                    transform=ax.transAxes)
            segmentation_info.append(([], [], [], [], [], []))
            continue

        # Split data into continuous segments.
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=10)
        segmentation_info.append((segments_time, segments_vals, sorted_times, sorted_vals, time_sec, vals_array))

        # ----- Plot Allan Deviation -----
        if not plot_all_data_segments:
            # Use only the largest continuous segment.
            time_sec_cont, vals_cont = get_longest_continuous_segment(segments_time, segments_vals)
            if len(time_sec_cont) <= 1:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center',
                        transform=ax.transAxes)
                continue
            if resample:
                dt_diffs_cont = np.diff(time_sec_cont)
                uniform_dt = np.median(dt_diffs_cont)
                new_time = np.arange(time_sec_cont[0], time_sec_cont[-1], uniform_dt)
                uniform_vals = np.interp(new_time, time_sec_cont, vals_cont)
                if len(new_time) < 2:
                    ax.text(0.5, 0.5, "Not enough resampled points", ha='center', va='center',
                            transform=ax.transAxes)
                    continue
            else:
                dt_diffs = np.diff(time_sec_cont)
                uniform_dt = np.median(dt_diffs)
                uniform_vals = vals_cont

            rate = 1.0 / uniform_dt
            min_tau = uniform_dt
            total_time = time_sec_cont[-1] - time_sec_cont[0]
            max_tau = total_time / 3
            tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), 50)
            taus_out, ad, ade, ns = allantools.oadev(uniform_vals, rate=rate, data_type='freq', taus=tau_values)
            if fit:
                initial_guess = [1e-14, 1e-14, 1e-14, 1e-14, 1e3]
                popt, pcov = curve_fit(allan_deviation_model, taus_out, ad, sigma=ade, p0=initial_guess)
                taus_fit = np.linspace(np.min(taus_out), np.max(taus_out), 200)
                ad_fit = allan_deviation_model(taus_fit, *popt)
                ax.plot(taus_fit, ad_fit, 'r-', label='Noise model fit')

            ax.set_xscale('log')
            ax.scatter(taus_out, ad, marker='o', color=colors[j % len(colors)], label=f"Qubit {i + 1}")
            ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[j % len(colors)])
            ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
            ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)
        else:
            # Plot all segments separately.
            valid_segments = [(t, v) for t, v in zip(segments_time, segments_vals) if len(t) > 1]
            if not valid_segments:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center',
                        transform=ax.transAxes)
                continue

            if stack_segments_xaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(1, len(valid_segments),
                                                            subplot_spec=ax.get_subplotspec(),
                                                            wspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                combined_taus = []
                first_sub_ax = None
                last_sub_ax = None
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig.add_subplot(gs_inner[0, k])
                        first_sub_ax = sub_ax
                        sub_ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
                    else:
                        sub_ax = fig.add_subplot(gs_inner[0, k], sharey=first_sub_ax)
                        sub_ax.set_yticklabels([])
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 2:
                            continue
                    else:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        uniform_vals = seg_vals

                    rate = 1.0 / uniform_dt
                    min_tau = uniform_dt
                    total_time = seg_time[-1] - seg_time[0]
                    max_tau = total_time / 3
                    tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), 50)
                    taus_out, ad, ade, ns = allantools.oadev(uniform_vals, rate=rate, data_type='freq', taus=tau_values)
                    if fit:
                        initial_guess = [1e-14, 1e-14, 1e-14, 1e-14, 1e3]
                        popt, pcov = curve_fit(allan_deviation_model, taus_out, ad, sigma=ade, p0=initial_guess)
                        taus_fit = np.linspace(np.min(taus_out), np.max(taus_out), 200)
                        ad_fit = allan_deviation_model(taus_fit, *popt)
                        sub_ax.plot(taus_fit, ad_fit, 'r-', label='Noise model fit')

                    combined_taus.extend(taus_out)
                    sub_ax.set_xscale('log')
                    sub_ax.scatter(taus_out, ad, marker='o', color=segment_colors[k])
                    sub_ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                    last_sub_ax = sub_ax
                if last_sub_ax is not None:
                    last_sub_ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
                    if combined_taus:
                        combined_taus = np.array(combined_taus)
                        last_sub_ax.set_xlim(max(min(combined_taus[combined_taus > 0]) * 0.8, 1e-3),
                                             max(combined_taus) * 1.5)
            elif stack_segments_yaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(len(valid_segments), 1,
                                                            subplot_spec=ax.get_subplotspec(),
                                                            hspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                combined_taus = []
                first_sub_ax = None
                last_sub_ax = None
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig.add_subplot(gs_inner[k])
                        first_sub_ax = sub_ax
                        sub_ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
                    else:
                        sub_ax = fig.add_subplot(gs_inner[k], sharex=first_sub_ax)
                        sub_ax.set_yticklabels([])
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 2:
                            continue
                    else:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        uniform_vals = seg_vals

                    rate = 1.0 / uniform_dt
                    min_tau = uniform_dt
                    total_time = seg_time[-1] - seg_time[0]
                    max_tau = total_time / 3
                    tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), 50)
                    taus_out, ad, ade, ns = allantools.oadev(uniform_vals, rate=rate, data_type='freq', taus=tau_values)
                    if fit:
                        initial_guess = [1e-14, 1e-14, 1e-14, 1e-14, 1e3]
                        popt, pcov = curve_fit(allan_deviation_model, taus_out, ad, sigma=ade, p0=initial_guess)
                        taus_fit = np.linspace(np.min(taus_out), np.max(taus_out), 200)
                        ad_fit = allan_deviation_model(taus_fit, *popt)
                        sub_ax.plot(taus_fit, ad_fit, 'r-', label='Noise model fit')

                    combined_taus.extend(taus_out)
                    sub_ax.set_xscale('log')
                    sub_ax.scatter(taus_out, ad, marker='o', color=segment_colors[k])
                    sub_ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                    last_sub_ax = sub_ax
                if last_sub_ax is not None:
                    last_sub_ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
                    if combined_taus:
                        combined_taus = np.array(combined_taus)
                        last_sub_ax.set_xlim(max(min(combined_taus[combined_taus > 0]) * 0.8, 1e-3),
                                             max(combined_taus) * 1.5)
            else:
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments_time)))
                combined_taus = []
                for k, (seg_time, seg_vals) in enumerate(zip(segments_time, segments_vals)):
                    if len(seg_time) <= 4:
                        continue
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 4:
                            continue
                    else:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        uniform_vals = seg_vals

                    rate = 1.0 / uniform_dt
                    min_tau = uniform_dt
                    total_time = seg_time[-1] - seg_time[0]
                    max_tau = total_time / 3
                    tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), 50)
                    taus_out, ad, ade, ns = allantools.oadev(uniform_vals, rate=rate, data_type='freq', taus=tau_values)
                    if fit:
                        initial_guess = [1e-14, 1e-14, 1e-14, 1e-14, 1e3]
                        popt, pcov = curve_fit(allan_deviation_model, taus_out, ad, sigma=ade, p0=initial_guess)
                        taus_fit = np.linspace(np.min(taus_out), np.max(taus_out), 200)
                        ad_fit = allan_deviation_model(taus_fit, *popt)
                        ax.plot(taus_fit, ad_fit, 'r-', label='Noise model fit')

                    combined_taus.extend(taus_out)
                    ax.scatter(taus_out, ad, marker='o', color=segment_colors[k], label=f"D{k + 1}")
                    ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=segment_colors[k])
                if combined_taus:
                    combined_taus = np.array(combined_taus)
                    ax.set_xlim(max(min(combined_taus[combined_taus > 0]) * 0.8, 1e-3),
                                max(combined_taus) * 1.5)
                ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
                ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.set_xscale('log')

        if show_legends and (not plot_all_data_segments or (
                plot_all_data_segments and not (stack_segments_yaxis or stack_segments_xaxis))):
            ax.legend(loc='best', edgecolor='black')

    # Hide any unused axes.
    for j in range(effective_n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if plot_all_data_segments:
        if stack_segments_yaxis:
            save_label = save_label + 'all_segments_stacked'
        elif stack_segments_xaxis:
            save_label = save_label + 'all_segments_xstacked'
        else:
            save_label = save_label + 'all_segments'
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_allan_deviation_continuous.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

    # --- Raw Data Segments vs. Time ---
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                               sharex=False, sharey=False)
    fig2.suptitle(f"Segmented {label} Data", fontsize=font)
    if effective_n == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()

    for j, i in enumerate(valid_indices):
        ax = axes2[j]
        ax.set_title(titles[j], fontsize=font)
        segments_time, segments_vals, sorted_times, sorted_vals, time_sec, vals_array = segmentation_info[j]

        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        if not plot_all_data_segments:
            time_sec_cont, vals_cont = get_longest_continuous_segment(segments_time, segments_vals)
            if len(time_sec_cont) <= 1:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center',
                        transform=ax.transAxes)
                continue
            ax.plot(time_sec_cont, vals_cont, marker='o', color=colors[j % len(colors)], label=f"Qubit {i + 1}")
        else:
            valid_segments = [(t, v) for t, v in zip(segments_time, segments_vals) if len(t) > 1]
            if not valid_segments:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center',
                        transform=ax.transAxes)
                continue

            if stack_segments_xaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(1, len(valid_segments),
                                                            subplot_spec=ax.get_subplotspec(),
                                                            wspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig2.add_subplot(gs_inner[0, k])
                        sub_ax.set_ylabel(f"{label} Value", fontsize=font - 2)
                    else:
                        sub_ax = fig2.add_subplot(gs_inner[0, k], sharey=sub_ax)
                        sub_ax.set_yticklabels([])
                    sub_ax.plot(seg_time, seg_vals, marker='o', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                if sub_ax is not None:
                    sub_ax.set_xlabel("Time (s)", fontsize=font - 2)
            elif stack_segments_yaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(len(valid_segments), 1,
                                                            subplot_spec=ax.get_subplotspec(),
                                                            hspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig2.add_subplot(gs_inner[k])
                        sub_ax.set_ylabel(f"{label} Value", fontsize=font - 2)
                    else:
                        sub_ax = fig2.add_subplot(gs_inner[k], sharex=sub_ax)
                        sub_ax.set_yticklabels([])
                    sub_ax.plot(seg_time, seg_vals, marker='o', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                if sub_ax is not None:
                    sub_ax.set_xlabel("Time (s)", fontsize=font - 2)
            else:
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments_time)))
                for k, (seg_time, seg_vals) in enumerate(zip(segments_time, segments_vals)):
                    if len(seg_time) <= 1:
                        continue
                    ax.plot(seg_time, seg_vals, marker='o', color=segment_colors[k], label=f"D{k + 1}")
                ax.set_xlabel("Time (s)", fontsize=font - 2)
                ax.set_ylabel(f"{label} Value", fontsize=font - 2)
                ax.tick_params(axis='both', which='major', labelsize=8)
        if show_legends and (not plot_all_data_segments or (
                plot_all_data_segments and not (stack_segments_yaxis or stack_segments_xaxis))):
            ax.legend(loc='best', edgecolor='black')

    for j in range(effective_n, len(axes2)):
        axes2[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_data_segments.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig2)


def plot_welch_spectral_density_largest_continuous_sample(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="", save_folder_path='', final_figure_quality=100, plot_all_data_segments=False, stack_segments_yaxis=False, stack_segments_xaxis=False, resample=False):
    """
    Plot the spectral density of fluctuations using Welch's method for each qubit,
    using only the largest continuous segment of data resampled onto a uniform grid
    when plot_all_data_segments is False. If plot_all_data_segments is True, each continuous
    data segment is processed individually and its spectral density calculated and plotted on the same axis.
    When stack_segments_yaxis is True, each segment is plotted on its own vertically stacked subplot
    (sharing the x-axis but with independent y-axes, with only one full y-label shown).
    When stack_segments_xaxis is True, the segments are arranged side-by-side (stacked along the x-axis)
    sharing a common y-axis.
    """
    create_folder_if_not_exists(save_folder_path)
    font = 14

    # Filter out qubits that do not have any data.
    valid_indices = [i for i in range(number_of_qubits) if len(date_times[i]) > 0 and len(vals[i]) > 0]
    effective_n = len(valid_indices)
    if effective_n == 0:
        print("No valid data available to plot.")
        return

    # Dynamically determine subplot grid.
    n_cols = min(3, effective_n)
    n_rows = math.ceil(effective_n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=False, sharey=False)
    if plot_all_data_segments:
        fig.suptitle(f'Welch-method Spectral Density of {label} Fluctuations across Data Sets', fontsize=font)
    else:
        fig.suptitle(f'Welch-method Spectral Density of {label} Fluctuations', fontsize=font)
    if effective_n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in valid_indices]

    # Process each valid qubit's data.
    for j, i in enumerate(valid_indices):
        ax = axes[j]
        ax.set_title(titles[j], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        sorted_times, sorted_vals = sort_date_time_data(dt_objs, data)
        if not sorted_times:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            continue

        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)
        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=10)

        if not plot_all_data_segments:
            time_sec_cont, vals_cont = get_longest_continuous_segment(segments_time, segments_vals)
            if len(time_sec_cont) <= 1:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center', transform=ax.transAxes)
                continue
            if resample:
                dt_diffs_cont = np.diff(time_sec_cont)
                uniform_dt = np.median(dt_diffs_cont)
                new_time = np.arange(time_sec_cont[0], time_sec_cont[-1], uniform_dt)
                uniform_vals = np.interp(new_time, time_sec_cont, vals_cont)
                if len(new_time) < 2:
                    ax.text(0.5, 0.5, "Not enough resampled points", ha='center', va='center', transform=ax.transAxes)
                    continue
                dt_uniform = new_time[1] - new_time[0]
            else:
                dt_uniform = time_sec_cont[1] - time_sec_cont[0]
                uniform_vals = vals_cont
            fs = 1.0 / dt_uniform

            freq, psd = welch(uniform_vals, fs=fs, nperseg=500, scaling='density')

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(freq, psd, marker='o', linestyle='none', color=colors[j % len(colors)], label=f"Qubit {i + 1}")
            ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
            ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)
        else:
            valid_segments = [(t, v) for t, v in zip(segments_time, segments_vals) if len(t) > 1]
            if not valid_segments:
                ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center', transform=ax.transAxes)
                continue

            if stack_segments_xaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(1, len(valid_segments),
                                                            subplot_spec=ax.get_subplotspec(),
                                                            wspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                combined_freqs = []
                first_sub_ax = None
                last_sub_ax = None
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig.add_subplot(gs_inner[0, k])
                        first_sub_ax = sub_ax
                        sub_ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
                    else:
                        sub_ax = fig.add_subplot(gs_inner[0, k], sharey=first_sub_ax)
                        sub_ax.set_yticklabels([])
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 2:
                            continue
                        dt_uniform = new_time[1] - new_time[0]
                    else:
                        dt_uniform = seg_time[1]-seg_time[0]
                        uniform_vals = seg_vals
                    fs = 1.0 / dt_uniform
                    freq, psd = welch(uniform_vals, fs=fs, nperseg=500, scaling='density')
                    combined_freqs.extend(freq)
                    sub_ax.set_xscale('log')
                    sub_ax.set_yscale('log')
                    sub_ax.plot(freq, psd, marker='o', linestyle='none', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                    sub_ax.ticklabel_format(useOffset=False, style='plain', axis='both')
                    last_sub_ax = sub_ax
                if last_sub_ax is not None:
                    last_sub_ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
                    if combined_freqs:
                        combined_freqs = np.array(combined_freqs)
                        last_sub_ax.set_xlim(max(min(combined_freqs[combined_freqs > 0]) * 0.8, 1e-3),
                                              max(combined_freqs) * 1.5)
            elif stack_segments_yaxis:
                ax.set_visible(False)
                gs_inner = gridspec.GridSpecFromSubplotSpec(len(valid_segments), 1,
                                                            subplot_spec=ax.get_subplotspec(),
                                                            hspace=0.1)
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_segments)))
                combined_freqs = []
                first_sub_ax = None
                last_sub_ax = None
                for k, (seg_time, seg_vals) in enumerate(valid_segments):
                    if k == 0:
                        sub_ax = fig.add_subplot(gs_inner[k])
                        first_sub_ax = sub_ax
                        sub_ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
                    else:
                        sub_ax = fig.add_subplot(gs_inner[k], sharex=first_sub_ax)
                        sub_ax.set_yticklabels([])
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 2:
                            continue
                        dt_uniform = new_time[1] - new_time[0]
                    else:
                        dt_uniform = seg_time[1]-seg_time[0]
                        uniform_vals = seg_vals
                    fs = 1.0 / dt_uniform
                    freq, psd = welch(uniform_vals, fs=fs, nperseg=500, scaling='density')
                    combined_freqs.extend(freq)
                    sub_ax.set_xscale('log')
                    sub_ax.set_yscale('log')
                    sub_ax.plot(freq, psd, marker='o', linestyle='none', color=segment_colors[k])
                    sub_ax.tick_params(axis='both', which='major', labelsize=8)
                    sub_ax.ticklabel_format(useOffset=False, style='plain', axis='both')
                    last_sub_ax = sub_ax
                if last_sub_ax is not None:
                    last_sub_ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
                    if combined_freqs:
                        combined_freqs = np.array(combined_freqs)
                        last_sub_ax.set_xlim(max(min(combined_freqs[combined_freqs > 0]) * 0.8, 1e-3),
                                              max(combined_freqs) * 1.5)
            else:
                segment_colors = plt.cm.viridis(np.linspace(0, 1, len(segments_time)))
                combined_freqs = []
                for j_seg, (seg_time, seg_vals) in enumerate(zip(segments_time, segments_vals)):
                    if len(seg_time) <= 1:
                        continue
                    if resample:
                        dt_diffs = np.diff(seg_time)
                        uniform_dt = np.median(dt_diffs)
                        new_time = np.arange(seg_time[0], seg_time[-1], uniform_dt)
                        uniform_vals = np.interp(new_time, seg_time, seg_vals)
                        if len(new_time) < 2:
                            continue
                        dt_uniform = new_time[1] - new_time[0]
                    else:
                        dt_uniform = seg_time[1]-seg_time[0]
                        uniform_vals = seg_vals
                    fs = 1.0 / dt_uniform
                    freq, psd = welch(uniform_vals, fs=fs, nperseg=500, scaling='density')
                    combined_freqs.extend(freq)
                    ax.plot(freq, psd, marker='o', linestyle='none', color=segment_colors[j_seg], label=f"D{j_seg+1}")
                if combined_freqs:
                    combined_freqs = np.array(combined_freqs)
                    ax.set_xlim(max(min(combined_freqs[combined_freqs > 0]) * 0.8, 1e-3),
                                max(combined_freqs) * 1.5)
                ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
                ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.set_xscale('log')
                ax.set_yscale('log')

        if show_legends and (not plot_all_data_segments or (plot_all_data_segments and not (stack_segments_yaxis or stack_segments_xaxis))):
            ax.legend(loc='best', edgecolor='black')

    # Hide any extra axes.
    for j in range(effective_n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if plot_all_data_segments:
        if stack_segments_yaxis:
            save_label = save_label + 'all_segments_stacked'
        elif stack_segments_xaxis:
            save_label = save_label + 'all_segments_xstacked'
        else:
            save_label = save_label + 'all_segments'
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_welch_spectral_density_continuous_sample.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_lomb_scargle_spectral_density(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="", save_folder_path='', final_figure_quality=100, log_freqs=False):
    """
    Plot the spectral density of fluctuations using the Lomb–Scargle periodogram for each qubit.

    This function processes irregularly sampled data without resampling by directly computing
    the Lomb–Scargle periodogram. It converts the date/time strings into seconds, defines a logarithmically
    spaced frequency grid based on the total time span and median sampling interval, and then estimates the
    power spectral density (PSD). The results are plotted on a log–log scale.
    """
    create_folder_if_not_exists(save_folder_path)
    font = 14

    # Filter valid qubits (those with non-empty date/time and value lists)
    valid_indices = [i for i in range(number_of_qubits) if len(date_times[i]) > 0 and len(vals[i]) > 0]
    effective_n = len(valid_indices)
    if effective_n == 0:
        print("No valid data available to plot.")
        return

    # Dynamically set up the subplot grid.
    n_cols = min(3, effective_n)
    n_rows = math.ceil(effective_n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=False, sharey=False)
    fig.suptitle(f'Lomb–Scargle Spectral Density of {label} Fluctuations', fontsize=font)
    if effective_n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in valid_indices]

    # Process each valid qubit’s data.
    for j, i in enumerate(valid_indices):
        ax = axes[j]
        ax.set_title(titles[j], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        # Sort the data by time.
        combined = list(zip(dt_objs, data))
        combined.sort(key=lambda x: x[0])
        if combined:
            sorted_times, sorted_vals = zip(*combined)
        else:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            continue

        # Convert sorted times to seconds relative to the first measurement.
        t0 = sorted_times[0]
        time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
        vals_array = np.array(sorted_vals, dtype=float)

        if len(time_sec) < 2:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        # ---------------- Define Frequency Grid ------------------
        total_time = time_sec[-1] - time_sec[0]
        f_min = 1.0 / total_time if total_time > 0 else 1.0
        median_dt = np.median(np.diff(time_sec))
        f_max = 0.5 / median_dt if median_dt > 0 else 1.0
        n_freq = 5000
        if log_freqs:
            frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
        else:
            frequencies = np.linspace(f_min, f_max, n_freq)
        angular_frequencies = 2 * np.pi * frequencies
        # ---------------------------------------------------------

        # --------------- Compute Lomb–Scargle Periodogram ---------------
        power = lombscargle(time_sec, vals_array, angular_frequencies)
        # ----------------------------------------------------------------

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(frequencies, power, marker='o', linestyle='none', color=colors[j % len(colors)],
                label=f"Qubit {i + 1}")

        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
        ax.set_ylabel(rf"$S_{{{label}}}$", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide any extra axes.
    for j in range(effective_n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_lomb_scargle_spectral_density.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)
