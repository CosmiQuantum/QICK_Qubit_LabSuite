import os, visdom, logging
import numpy as np

def create_folder_if_not_exists(folder):
    """
    Creates a folder at the given path if it doesn't already exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

def check_visdom_connection(live_plot=False):
    if live_plot:
        # Check if visdom is connected right away, otherwise, throw an error
        if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                               "http://localhost:8097/ on firefox")

def mask_gain_res(QUBIT_INDEX, IndexGain = 1, num_qubits=6):
    """Sets the gain for the selected qubit to 1, others to 0."""
    filtered_gain_ge = [0] * num_qubits  # Initialize all gains to 0
    if 0 <= QUBIT_INDEX < num_qubits: #makes sure you are within the range of options
        filtered_gain_ge[QUBIT_INDEX] = IndexGain  # Set the gain for the selected qubit
    return filtered_gain_ge

def configure_logging(log_file):
    ''' We need to create a custom logger and disable propagation like this
    to remove the logs from the underlying qick from saving to the log file for RR'''

    rr_logger = logging.getLogger("custom_logger_for_rr_only")
    rr_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    rr_logger.addHandler(file_handler)
    rr_logger.propagate = False  # dont propagate logs from underlying qick package

    return rr_logger


def extract_resonator_frequencies(
        data,  # dict: keys are qubit indices (0 means qubit 1) and values are (x_data, y_data)
        process_offset=False,
        offsets=None  # dict: keys matching those in data, with offset frequency values
):
    """
    Extracts the resonance frequencies for each qubit based on the provided data.

    Parameters
    ----------
    data : dict
        Dictionary where each key is a qubit index (0 means qubit 1) and each value is a tuple or list
        containing the x_data and y_data arrays for that qubit.
    process_offset : bool, optional
        If True, each qubit's x_data is shifted by an offset provided in the offsets dictionary.
    offsets : dict, optional
        Dictionary of offset frequencies to add to each qubit's x_data (keys must match those in data)
        when process_offset is True.

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