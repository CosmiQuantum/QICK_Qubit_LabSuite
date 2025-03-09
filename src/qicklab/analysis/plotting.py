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

from scipy.stats import norm
from scipy.signal import welch, lombscargle
from scipy.optimize import curve_fit
import os
import datetime
import allantools
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates
from ..utils.file_helpers import create_folder_if_not_exists
from .data_processing import *


def plot_spec_results_individually(I, Q, freqs, title_start='', spec=False, largest_amp_curve_mean=None,
                                   largest_amp_curve_fwhm=None,
                                   I_fit=None, Q_fit=None,
                                   qubit_index=None, config=None, outer_folder=None,
                                   expt_name=None, round_num=None, h5_filename = None, fig_quality=100):
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
    h5_filename = h5_filename.split('/')[-1].split('.')[0]
    file_name = os.path.join(outerFolder_expt,
                             f"{h5_filename}_plot.png")
    fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')

    plt.close(fig)

    if spec:
        return largest_amp_curve_mean, I_fit, Q_fit
    else:
        return None

def plot_spectroscopy(
        qubit_index,
        fpts,
        fcenter,
        amps,
        round_num=0,
        config=None,
        outerFolder=None,
        expt_name="res_spec",
        experiment=None,
        save_figs=False,
        reloaded_config=None,
        fig_quality=100,
        title_word="Resonator",
        xlabel="Frequency (MHz)",
        ylabel="Amplitude (a.u.)",
        find_min=True,
        plot_min_line=True,
        include_min_in_title=True,
        return_min=True,
        fig_filename=None
):
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
    # Determine the number of datasets from the amplitude data.
    n_datasets = amps.shape[0]

    # Calculate a reasonable grid size for subplots.
    n_rows = math.ceil(math.sqrt(n_datasets))
    n_cols = math.ceil(n_datasets / n_rows)

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
        freqs = np.array(fpts) + center

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
        plt.title(title, pad=10)
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

    plt.tight_layout(pad=2.0)

    # Optionally save the figure.
    if save_figs and outerFolder is not None:
        if fig_filename is None:
            # Create a subfolder based on expt_name.
            outerFolder_expt = os.path.join(outerFolder, expt_name)
            if not os.path.exists(outerFolder_expt):
                os.makedirs(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            # File name incorporates the round number and (legacy) qubit_index.
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


def remove_none_values(y_vals, x_vals, dates):
    """
    Remove None values from a list along with their corresponding x-values and dates.

    Parameters
    ----------
    y_vals : list
        List of y-values which may include None entries.
    x_vals : list
        List of x-values corresponding to y_vals.
    dates : list
        List of date values corresponding to y_vals.

    Returns
    -------
    tuple of lists
        Three lists corresponding to y_vals, x_vals, and dates with the None values removed.

    Raises
    ------
    ValueError
        If the input lists do not have the same length.
    """
    if not (len(y_vals) == len(x_vals) == len(dates)):
        raise ValueError("All lists must have the same length")

    # Filter out None values and their corresponding elements.
    filtered_data = [(x, y, z) for x, y, z in zip(y_vals, x_vals, dates) if x is not None]

    # Unzip to separate the lists.
    filtered_list1, filtered_list2, filtered_list3 = zip(*filtered_data) if filtered_data else ([], [], [])

    return list(filtered_list1), list(filtered_list2), list(filtered_list3)


def scatter_plot_vs_time_with_fit_errs(date_times, y_data, fit_err, number_of_qubits, y_data_name='',
                                       save_name='', save_folder_path='', y_label='',
                                       show_legends=False, final_figure_quality=100):
    """
    Create scatter plots versus time for multiple qubits, including error bars.

    This function plots data for each qubit on separate subplots arranged in a 2x3 grid.
    The function sorts the data based on time, adds error bars, and optionally saves the figure.

    Parameters
    ----------
    date_times : list of lists
        Each element is a list of date/time values for a qubit.
    y_data : list of lists
        Each element is a list of numerical data values for a qubit.
    fit_err : list of lists
        Each element is a list of error values corresponding to the y_data for a qubit.
    number_of_qubits : int
        Number of qubits (determines the number of subplots).
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
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.suptitle(f'{y_data_name} vs Time', fontsize=font)
    axes = axes.flatten()

    # Loop over each qubit’s data.
    for i, ax in enumerate(axes):
        if i >= number_of_qubits:  # Hide extra subplots.
            ax.set_visible(False)
            continue

        ax.set_title(titles[i], fontsize=font)

        x = date_times[i]  # List of date/time values.
        y = y_data[i]
        err = fit_err[i]  # Corresponding error values.

        # Combine date_times, y_data, and error values, then sort by time.
        combined = list(zip(x, y, err))
        combined.sort(key=lambda tup: tup[0])
        if len(combined) == 0:
            ax.set_visible(False)
            continue

        # Unpack the sorted data.
        sorted_x, sorted_y, sorted_err = zip(*combined)
        sorted_x = np.array(sorted_x)
        sorted_y, sorted_x, sorted_err = remove_none_values(sorted_y, sorted_x, sorted_err)

        ax.errorbar(
            sorted_x, sorted_y, yerr=sorted_err,
            fmt='none',
            ecolor=colors[i],
            elinewidth=1,
            capsize=0
        )

        ax.scatter(
            sorted_x, sorted_y,
            s=10,
            color=colors[i],
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_vs_time.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close()


def plot_histogram_with_gaussian(data_dict, date_dict, save_name='', save_folder_path='', data_type='',
                                 x_label='', y_label='',
                                 show_legends=False, final_figure_quality=300, n_cols=3, bin_count=50):
    """
    Plot a histogram for each dataset with an overlaid Gaussian fit.

    For each dataset (keyed in `data_dict`), this function fits a Gaussian distribution,
    overlays the Gaussian on the histogram, and returns data useful for generating smoother
    Gaussian curves for cumulative plots.

    Parameters
    ----------
    data_dict : dict
        Keys (e.g., 0, 1, ...) mapping to arrays/lists of numerical data.
    date_dict : dict
        Keys mapping to arrays/lists of date strings or labels used in the legend.
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
        If True, display legends on subplots.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 300).
    n_cols : int, optional
        Number of columns in the subplot grid.

    Returns
    -------
    tuple
        A tuple containing:
            - gaussian_fit_data (dict): For each dataset, a tuple (x_full, pdf_full) for smoother Gaussian plotting.
            - mean_values (dict): Computed mean for each dataset.
            - std_values (dict): Computed standard deviation for each dataset.
    """
    n_datasets = len(data_dict)
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.suptitle(f'{data_type} Histograms', fontsize=16)
    # Handle the case of a single subplot.
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink', 'red', 'cyan', 'magenta', 'yellow']
    gaussian_fit_data = {}
    mean_values = {}
    std_values = {}

    for i, ax in enumerate(axes):
        if i not in data_dict:
            ax.set_visible(False)
            continue
        data = data_dict[i]
        if len(data) == 0:
            ax.set_visible(False)
            continue

        # Use the first label from date_dict if available.
        date_label = date_dict.get(i, [''])[0] if date_dict.get(i) else ''

        # Fit a Gaussian if more than one data point is available.
        if len(data) > 1:
            mu, sigma = norm.fit(data)
            mean_values[f"{data_type} {i + 1}"] = mu
            std_values[f"{data_type} {i + 1}"] = sigma

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
            gaussian_fit_data[i] = (x_full, pdf_full)

            title_str = f"Qubit {i + 1}  μ: {mu:.2f} σ: {sigma:.2f}"
        else:
            # If only one data point exists, only a basic histogram is plotted.
            ax.hist(data, bins=10, alpha=0.7, color=colors[i % len(colors)],
                    edgecolor='black', label=date_label)
            title_str = f"{data_type} {i + 1}"

        if show_legends:
            ax.legend()
        ax.set_title(title_str, fontsize=14)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_hist.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

    return gaussian_fit_data, mean_values, std_values


def plot_cumulative_distribution(data_dict, gaussian_fit_data, save_name='', save_folder_path='', data_type='',
                                 x_label='', y_label='',
                                 final_figure_quality=300):
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
    ax.legend(edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_cumulative.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_error_vs_value(data_dict, error_dict, save_name='', save_folder_path='', data_type='',
                        x_label='', y_label='', show_legends=False, final_figure_quality=300, n_cols=3):
    """
    Create scatter plots of error versus value for each dataset.

    This function creates a grid of subplots where each subplot shows a scatter plot of values
    against their corresponding fit errors for a given dataset.

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
    n_datasets = len(data_dict)
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink', 'red', 'cyan', 'magenta', 'yellow']
    plt.suptitle(f'{data_type} Value vs Fit Error', fontsize=16)
    for i, ax in enumerate(axes):
        if i not in data_dict or i not in error_dict:
            ax.set_visible(False)
            continue
        data = data_dict[i]
        errors = error_dict[i]
        if len(data) == 0 or len(errors) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(data, errors, color=colors[i % len(colors)], label=f'{data_type} {i + 1}')
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_title(f'Qubit {i + 1}', fontsize=14)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_name}_errs.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

def plot_allan_deviation_largest_continuous_sample(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="",
                                                   save_folder_path='', final_figure_quality=100):
    """
    Plot the overlapping Allan deviation for each qubit, excluding large gaps in the data.

    For each qubit, the function:
      1. Sorts timestamps and measurement values.
      2. Converts timestamps to seconds relative to the first measurement.
      3. Splits the data into continuous segments based on a gap threshold.
      4. Selects the largest continuous segment.
      5. Resamples the data in that segment onto a uniform grid.
      6. Computes the overlapping Allan deviation using allantools.
      7. Plots the Allan deviation on a log–log scale.
    """
    # Ensure the folder exists.
    create_folder_if_not_exists(save_folder_path)

    font = 14
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations', fontsize=font)
    axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]

    # Process each qubit's data individually.
    for i, ax in enumerate(axes):
        if i >= number_of_qubits:
            ax.set_visible(False)
            continue

        ax.set_title(titles[i], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        # Sort the data using the helper function.
        sorted_times, sorted_vals = sort_date_time_data(dt_objs, data)
        if not sorted_times:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            continue

        # Convert sorted times to seconds relative to the first measurement.
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)

        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Split the data into continuous segments using the helper function.
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=5)

        # Choose the largest continuous segment using the helper.
        time_sec_cont, vals_cont = get_longest_continuous_segment(segments_time, segments_vals)
        if len(time_sec_cont) <= 1:
            ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Resample the continuous segment onto a uniform grid.
        dt_diffs_cont = np.diff(time_sec_cont)
        uniform_dt = np.median(dt_diffs_cont)
        new_time = np.arange(time_sec_cont[0], time_sec_cont[-1], uniform_dt)
        uniform_vals = np.interp(new_time, time_sec_cont, vals_cont)
        if len(new_time) < 2:
            ax.text(0.5, 0.5, "Not enough resampled points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Calculate the uniform sample rate.
        rate = 1.0 / uniform_dt

        # Compute the overlapping Allan deviation.
        taus_out, ad, ade, ns = allantools.oadev(
            uniform_vals,
            rate=rate,
            data_type='freq',
            taus='decade'
        )

        ax.set_xscale('log')
        ax.scatter(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")
        ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlim(min(taus_out[taus_out > 0]) * 0.8, max(taus_out) * 1.5)
        ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
        ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_allan_deviation_continuous.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_welch_spectral_density_largest_continuous_sample(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="",
                                                            save_folder_path='', final_figure_quality=100):
    """
    Plot the spectral density of fluctuations using Welch's method for each qubit,
    using only the largest continuous segment of data resampled onto a uniform grid.
    """
    create_folder_if_not_exists(save_folder_path)

    font = 14
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    fig.suptitle(f'Welch-method Spectral Density of {label} Fluctuations', fontsize=font)
    axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]

    # Process each qubit's data.
    for i, ax in enumerate(axes):
        if i >= number_of_qubits:
            ax.set_visible(False)
            continue

        ax.set_title(titles[i], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        # Sort the data using the helper function.
        sorted_times, sorted_vals = sort_date_time_data(dt_objs, data)
        if not sorted_times:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            continue

        # Convert times to seconds relative to the first measurement.
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)

        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Split the data into continuous segments.
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=5)

        # Choose the largest continuous segment using the helper.
        time_sec_cont, vals_cont = get_longest_continuous_segment(segments_time, segments_vals)
        if len(time_sec_cont) <= 1:
            ax.text(0.5, 0.5, "Not enough continuous points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Resample the continuous segment onto a uniform grid.
        dt_diffs_cont = np.diff(time_sec_cont)
        uniform_dt = np.median(dt_diffs_cont)
        new_time = np.arange(time_sec_cont[0], time_sec_cont[-1], uniform_dt)
        uniform_vals = np.interp(new_time, time_sec_cont, vals_cont)
        if len(new_time) < 2:
            ax.text(0.5, 0.5, "Not enough resampled points", ha='center', va='center', transform=ax.transAxes)
            continue

        dt_uniform = new_time[1] - new_time[0]
        fs = 1.0 / dt_uniform

        # Compute the PSD using Welch's method.
        freq, psd = welch(uniform_vals, fs=fs, nperseg=None, scaling='density')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(freq, psd, marker='o', linestyle='none', color=colors[i],
                label=f"Qubit {i + 1}")

        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
        ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_welch_spectral_density_continuous_sample.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

def plot_allan_deviation(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="",
                         save_folder_path='', final_figure_quality=100):
    """
    Plot the overlapping Allan deviation for each qubit.

    For each qubit, the function sorts timestamps and measurement values, resamples the irregularly
    spaced data onto a uniform grid, computes the overlapping Allan deviation using allantools, and
    plots the result on a log–log scale.

    Parameters
    ----------
    date_times : list of lists
        Each element is a list of date/time strings (format "YYYY-MM-DD HH:MM:SS") for a qubit.
    vals : list of lists
        Each element is a list of measurement values for a qubit.
    number_of_qubits : int
        Number of qubits (each will be plotted in a separate subplot).
    show_legends : bool, optional
        If True, display legends in the subplots.
    label : str, optional
        Label for the measurement (used in the title and y–axis; default is "T1").
    save_folder_path : str, optional
        Folder path where the plot PDF will be saved.
    final_figure_quality : int, optional
        DPI for the saved plot (default is 100).

    Returns
    -------
    None
    """
    # Ensure the folder exists.
    create_folder_if_not_exists(save_folder_path)

    font = 14
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations', fontsize=font)
    axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]

    # Process each qubit's data individually.
    for i, ax in enumerate(axes):
        if i >= number_of_qubits:
            ax.set_visible(False)
            continue

        ax.set_title(titles[i], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        # Zip and sort the data by time (ascending).
        combined = list(zip(dt_objs, data))
        combined.sort(key=lambda x: x[0])
        if combined:
            sorted_times, sorted_vals = zip(*combined)
        else:
            sorted_times, sorted_vals = [], []
            continue

        # Convert sorted times to seconds (relative to the first measurement).
        t0 = sorted_times[0]
        time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
        vals_array = np.array(sorted_vals, dtype=float)

        if len(time_sec) <= 1:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        # Resample onto a uniform grid.
        dt_diffs = np.diff(time_sec)
        uniform_dt = np.median(dt_diffs)
        new_time = np.arange(time_sec[0], time_sec[-1], uniform_dt)
        uniform_vals = np.interp(new_time, time_sec, vals_array)

        # Calculate the uniform sample rate.
        rate = 1.0 / uniform_dt

        # Compute the overlapping Allan deviation.
        taus_out, ad, ade, ns = allantools.oadev(
            uniform_vals,
            rate=rate,
            data_type='freq',
            taus='decade'
        )

        ax.set_xscale('log')
        ax.scatter(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")
        ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlim(min(taus_out[taus_out > 0]) * 0.8, max(taus_out) * 1.5)
        ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
        ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_allan_deviation.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)


def plot_welch_spectral_density(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="",
                                save_folder_path='', final_figure_quality=100):
    """
    Plot the spectral density of fluctuations using Welch's method for each qubit.

    The function converts irregularly sampled date/time strings into a uniform time grid, computes the
    Power Spectral Density (PSD) using Welch's method, and plots the PSD on a log–log scale for each qubit.

    Parameters
    ----------
    date_times : list of lists
        Each element is a list of date/time strings (format "YYYY-MM-DD HH:MM:SS") for a qubit.
    vals : list of lists
        Each element is a list of measurement values for a qubit.
    number_of_qubits : int
        Number of qubits (determines the number of subplots).
    show_legends : bool, optional
        If True, displays legends in the subplots.
    label : str, optional
        Label for the measurement (used in axis labels; default is "T1").
    save_folder_path : str, optional
        Folder path where the figure will be saved.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 100).

    Returns
    -------
    None
    """
    create_folder_if_not_exists(save_folder_path)

    font = 14
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    fig.suptitle(f'Welch-method Spectral Density of {label} Fluctuations', fontsize=font)
    axes = axes.flatten()

    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]

    # Process each qubit’s data.
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=font)

        data = vals[i]
        dt_objs = date_times[i]

        # Sort data by time.
        combined = list(zip(dt_objs, data))
        combined.sort(key=lambda x: x[0])
        if combined:
            sorted_times, sorted_vals = zip(*combined)
        else:
            sorted_times, sorted_vals = [], []
            continue

        t0 = sorted_times[0]

        time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
        vals_array = np.array(sorted_vals, dtype=float)

        # Resample irregular data onto a uniform grid.
        n_points = len(time_sec)
        time_uniform = np.linspace(time_sec[0], time_sec[-1], n_points)
        vals_uniform = np.interp(time_uniform, time_sec, vals_array)

        dt_uniform = time_uniform[1] - time_uniform[0] if n_points > 1 else 1.0
        fs = 1.0 / dt_uniform

        # Compute the PSD using Welch's method.
        freq, psd = welch(vals_uniform, fs=fs, nperseg=None, scaling='density')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(freq, psd, marker='o', linestyle='none', color=colors[i],
                label=f"Qubit {i + 1}")

        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlabel(r"Frequency (Hz)", fontsize=font - 2)
        ax.set_ylabel(rf"$S_{{{label}}}$ ($\mu s^2$/Hz)", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_welch_spectral_density.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)

def plot_lomb_scargle_spectral_density(date_times, vals, number_of_qubits, show_legends=False, label="", save_label="",
                                       save_folder_path='', final_figure_quality=100, log_freqs=False):
    """
    Plot the spectral density of fluctuations using the Lomb–Scargle periodogram for each qubit.

    This function processes irregularly sampled data without resampling by directly computing
    the Lomb–Scargle periodogram. It converts the date/time strings into seconds, defines a logarithmically
    spaced frequency grid based on the total time span and median sampling interval, and then estimates the
    power spectral density (PSD). The results are plotted on a log–log scale.

    Parameters
    ----------
    date_times : list of lists
        Each element is a list of date/time strings (format "YYYY-MM-DD HH:MM:SS") for a qubit.
    vals : list of lists
        Each element is a list of measurement values for a qubit.
    number_of_qubits : int
        Number of qubits (determines the number of subplots).
    show_legends : bool, optional
        If True, displays legends in the subplots.
    label : str, optional
        Label for the measurement (used in axis labels; default is "T1").
    save_folder_path : str, optional
        Folder path where the figure will be saved.
    final_figure_quality : int, optional
        DPI for saving the figure (default is 100).

    Returns
    -------
    None
    """
    # Create folder if it doesn't exist
    create_folder_if_not_exists(save_folder_path)

    # Set up the figure and axes (assuming at most 6 qubits)
    font = 14
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
    fig.suptitle(f'Lomb–Scargle Spectral Density of {label} Fluctuations', fontsize=font)
    axes = axes.flatten()

    # Define colors for each qubit
    colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
    titles = [f"Qubit {i + 1}" for i in range(number_of_qubits)]

    # Process each qubit’s data individually.
    for i, ax in enumerate(axes):
        # Hide extra subplots if there are fewer qubits than available axes.
        if i >= number_of_qubits:
            ax.set_visible(False)
            continue

        ax.set_title(titles[i], fontsize=font)

        # Retrieve the date/time strings and measurement values for this qubit.
        data = vals[i]
        dt_objs = date_times[i]

        # Sort the data in ascending order of time.
        combined = list(zip(dt_objs, data))
        combined.sort(key=lambda x: x[0])
        if combined:
            sorted_times, sorted_vals = zip(*combined)
        else:
            sorted_times, sorted_vals = [], []
            continue

        # Convert sorted times to seconds relative to the first measurement.
        t0 = sorted_times[0]
        time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
        vals_array = np.array(sorted_vals, dtype=float)

        # Skip qubits with insufficient data.
        if len(time_sec) < 2:
            ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
            continue

        # ----------------- Define Frequency Grid ------------------
        # Total time span of the measurements.
        total_time = time_sec[-1] - time_sec[0]
        # Set the minimum frequency as 1 divided by the total time span.
        f_min = 1.0 / total_time if total_time > 0 else 1.0
        # Estimate maximum frequency using the median time difference.
        median_dt = np.median(np.diff(time_sec))
        f_max = 0.5 / median_dt if median_dt > 0 else 1.0

        n_freq = 1000  # number of frequency points
        if log_freqs:
            #log spaced frequency grid to give better resolution at low frequencies (compare to without doing this later)
            frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
        else:
            frequencies = np.linspace(f_min, f_max, n_freq)
        # Convert frequencies (Hz) to angular frequencies (rad/s) for lombscargle.
        angular_frequencies = 2 * np.pi * frequencies
        # -----------------------------------------------------------

        # ---------------- Compute Lomb–Scargle Periodogram ----------------
        # The lombscargle function computes the raw power at each angular frequency.
        # 'precenter=True' subtracts the mean from the data before processing.
        power = lombscargle(time_sec, vals_array, angular_frequencies) #, precenter=True?
        # --------------------------------------------------------------------

        # Plot the periodogram on a log–log scale.
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(frequencies, power, marker='o', linestyle='none', color=colors[i],
                label=f"Qubit {i + 1}")

        # Optionally display the legend.
        if show_legends:
            ax.legend(loc='best', edgecolor='black')

        ax.set_xlabel("Frequency (Hz)", fontsize=font - 2)
        # The y-axis label here is generic; adjust units/labels as needed.
        ax.set_ylabel(rf"$S_{{{label}}}$", fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    # Save the figure to the specified folder.
    plt.savefig(os.path.join(save_folder_path, f'{save_label}_lomb_scargle_spectral_density.png'),
                transparent=False, dpi=final_figure_quality)
    plt.close(fig)