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