# src/qicklab/analysis/plotting.py

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from qicklab.utils.file_helpers import create_folder_if_not_exists

def plot_resonance_spectroscopy(
    qubit_index,
    fpts,
    fcenter,
    amps,
    number_of_qubits,
    round_num,
    config,
    outerFolder=None,
    expt_name="res_spec",
    experiment=None,
    save_figs=False,
    reloaded_config=None,
    fig_quality=100
):
    """
    Plots the resonator spectroscopy results and returns the resonance frequencies.

    Parameters
    ----------
    fpts : np.array
        The frequency offsets used in the sweep (relative to fcenter).
    fcenter : np.array or list
        The center frequencies for each qubit/resonator.
    amps : np.array
        2D array of amplitude data: shape = (#qubits, #points).
    number_of_qubits : int
        Total number of qubits in the experiment (for subplots, etc.).
    round_num : int
        Which round (or iteration) of the experiment this data is from.
    config : dict
        The dictionary of settings used in the experiment.
    outerFolder : str
        Path to top-level directory where results may be saved.
    expt_name : str
        Name of the experiment (for file naming, etc.).
    experiment : object or None
        If not None, indicates a live experiment object; used to generate title text.
    save_figs : bool
        Whether or not to save the figure to disk.
    reloaded_config : dict or None
        An alternative config if no live experiment is provided.
    fig_quality : int
        DPI for saved figures.

    Returns
    -------
    res_freqs : list
        List of the extracted resonance frequencies for each qubit.
    """
    res_freqs = []

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })

    for i in range(number_of_qubits):
        plt.subplot(2, 3, i + 1)
        # Shift each offset by that qubit's center frequency:
        freqs = [f + fcenter[i] for f in fpts]
        plt.plot(freqs, amps[i], '-', linewidth=1.5)

        # Mark the minimum amplitude as the "resonance" freq
        freq_r = freqs[np.argmin(amps[i])]
        res_freqs.append(freq_r)

        # Vertical line for the resonance
        plt.axvline(freq_r, linestyle='--', color='orange', linewidth=1.5)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(f"Resonator {i + 1} {freq_r:.3f} MHz", pad=10)
        # Add a little buffer below the min amplitude
        ylo, yhi = plt.ylim()
        plt.ylim(ylo - 0.05 * (yhi - ylo), yhi)

    if experiment is not None:
        plt.suptitle(
            f"MUXed resonator spectroscopy {config['reps']}*{config['rounds']} avgs",
            fontsize=24, y=0.95
        )
    else:
        # If no live experiment object, maybe rely on reloaded_config
        plt.suptitle(
            f"MUXed resonator spectroscopy {reloaded_config['reps']}*{reloaded_config['rounds']} avgs",
            fontsize=24, y=0.95
        )

    plt.tight_layout(pad=2.0)

    # Optionally save the figure
    if save_figs and outerFolder is not None:
        outerFolder_expt = os.path.join(outerFolder, expt_name)
        create_folder_if_not_exists(outerFolder_expt)

        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(
            outerFolder_expt,
            f"R_{round_num}_Q_{qubit_index + 1}_{formatted_datetime}_{expt_name}.png"
        )
        # You can pass QubitIndex or other info if you want to vary the filename
        plt.savefig(file_name, dpi=fig_quality)

    plt.close()

    # Round frequencies to 4 decimals
    res_freqs = [round(x, 4) for x in res_freqs]
    return res_freqs
