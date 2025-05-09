
from ..datahandling.datafile_tools import find_h5_files, load_h5_data, process_h5_data, get_data_field
import os
import numpy as np

#temporary file, will change with DJT data loading restructure

def get_h5_for_qubit(data_path, h5_files, QubitIndex, data_type):
    # excludes h5 files that have no data for the specified Qubit Index
    h5_files_new = []

    for h5_file in h5_files:
        load_data = load_h5_data(os.path.join(data_path, h5_file), data_type, save_r=1)
        if not np.isnan(load_data[data_type][QubitIndex].get('Dates', [])[0][0]).any():
                h5_files_new.append(h5_file)

    return h5_files_new
