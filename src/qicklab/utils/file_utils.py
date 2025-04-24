import os
import h5py
import numpy as np

from .data_utils import convert_non_floats_to_strings, unwrap_singleton_list

##  ================= General Path & File Handling Methods ================= ##

def create_folder_if_not_exists(folder):
    """
    Creates a folder at the given path if it doesn't already exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_h5_files_in_dirs(folder_paths):
    """
    Get the pathnames of all .h5 files in the given list of directories.

    Parameters:
        folder_paths (list of str): A list of directory paths to search.

    Returns:
        list: A list of full pathnames of .h5 files found in all directories.
    """
    h5_files = []
    for folder_path in folder_paths:
        h5_files.extend(glob.glob(os.path.join(folder_path, "*.h5")))
    return h5_files

##  ================= Interfacing with H5 Files Methods ================= ##

def print_h5_contents(filename):
    """
    Print the contents of an HDF5 file including file attributes and dataset values.

    For each group in the file, the function prints the dataset names and, if the dataset is large,
    only prints a sample of the data.

    Parameters:
        filename (str): Path to the HDF5 file.
    """
    try:
        with h5py.File(filename, 'r') as f:
            print("File Attributes:")
            # Print each file-level attribute.
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            print("\nDataset Contents:")
            # Iterate over each group in the file.
            for key in f.keys():
                group = f[key]
                print(f"\nGroup: {key}")
                # Iterate over each dataset in the group.
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    try:
                        data = dataset[()]
                        print(f"  Dataset: {dataset_name}")
                        # If the dataset is large, only show a sample.
                        if isinstance(data, np.ndarray) and data.size > 100:
                            print(f"    Data (sample): {data[:50]} ... (truncated)")
                        else:
                            print(f"    Data: {data}")
                    except Exception:
                        print("    Data: Could not print data")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_h5_dataset(name, value, group):
    """
    Create a dataset within an HDF5 group based on the given name and value.

    The function handles different cases:
      - If the dataset is named "Dates", it attempts to convert the value to a float64 array.
      - If the value is a list and its first element indicates 'None', an empty array is created.
      - Otherwise, it first attempts to convert non-floats to strings and then convert
        to a float64 array. If that fails, it converts the value to a string array.
      - If value is None, an empty dataset is created.

    Parameters:
        name (str): Name of the dataset.
        value (any): The data to store.
        group (h5py.Group): The HDF5 group in which to create the dataset.
    """
    if value is not None:
        if name == "Dates":
            try:
                value = np.array(value, dtype=np.float64)
            except ValueError:
                value = np.array(value, dtype='S')
        elif isinstance(value, list) and 'None' in str(value[0]):
            # If the list appears to represent None values, store an empty array.
            value = np.array([])
        else:
            try:
                # Convert any non-numeric entries to strings, then to float64.
                value = convert_non_floats_to_strings(value)
                value = np.array(value, dtype=np.float64)
            except ValueError:
                # If conversion to float fails, store as a string array.
                value = np.array(value, dtype='S')
        group.create_dataset(name, data=value)
    else:
        # Create an empty dataset if no value is provided.
        group.create_dataset(name, data=np.array([]))


## Note: these are specific to a data format and probably don't belong here

def save_to_h5(data, outer_folder_expt, data_type, batch_num, save_r):
    """
    Save data to an HDF5 file with a timestamped filename.

    The function creates the destination folder if it does not exist. It generates a filename
    containing the current date/time, data type, batch number, and a replication factor. It then
    iterates over the provided data (organized per qubit) and saves each dataset within its respective
    group.

    Parameters:
        data (dict): Nested dictionary where each key corresponds to a qubit index and the value is a 
                     dictionary of dataset names and data.
        outer_folder_expt (str): Path to the folder where the HDF5 file should be saved.
        data_type (str): A descriptor for the type of data (used in filename and attributes).
        batch_num (int or str): Batch number to include in the filename and file attributes.
        save_r (int or str): Replication factor for the number of items per batch (included in filename and attributes).
    """
    # Ensure the output directory exists.
    os.makedirs(outer_folder_expt, exist_ok=True)
    # Create a formatted datetime string for uniqueness.
    formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Construct the filename with provided parameters.
    h5_filename = os.path.join(outer_folder_expt,
                               f"{formatted_datetime}_{data_type}_results_batch_{batch_num}_Num_per_batch{save_r}.h5")

    # Write the data to the HDF5 file.
    with h5py.File(h5_filename, 'w') as f:
        # Store file-level attributes.
        f.attrs['datestr'] = formatted_datetime
        f.attrs[f'{data_type}_results_batch'] = batch_num
        f.attrs['num_per_batch'] = save_r
        # Iterate over each qubit's data.
        for QubitIndex, qubit_data in data.items():
            # Create a group for each qubit (named Q1, Q2, etc.)
            group = f.create_group(f'Q{QubitIndex + 1}')
            # Create datasets within the qubit group.
            for key, value in qubit_data.items():
                create_h5_dataset(key, value, group)

def load_from_h5(filename, data_type, save_r=1):
    """
    Load data from an HDF5 file and replicate each dataset value a specified number of times.

    The function reads each group (assumed to represent a qubit) and for every dataset within,
    it replicates the read value save_r times in a list.

    Parameters:
        filename (str): Path to the HDF5 file.
        data_type (str): A key to wrap the loaded data under.
        save_r (int, optional): Number of times to replicate each loaded dataset. Defaults to 1.

    Returns:
        dict: Nested dictionary with structure:
              { data_type: { qubit_index: { dataset_name: [data]*save_r, ... }, ... } }
    """
    data = {data_type: {}}
    with h5py.File(filename, 'r') as f:
        for qubit_group in f.keys():
            # Convert group name (e.g., "Q1") to a zero-based index.
            qubit_index = int(qubit_group[1:]) - 1
            qubit_data = {}
            group = f[qubit_group]
            for dataset_name in group.keys():
                # Read the dataset value and replicate it save_r times.
                replicated = [group[dataset_name][()]] * save_r
                qubit_data[dataset_name] = unwrap_singleton_list(replicated)
            data[data_type][qubit_index] = qubit_data
    return data


def load_from_h5_with_shotdata(filename, data_type, save_r=1):  # Added save_r as parameter.

    data = {data_type: {}}  # Initialize the main dictionary with the data_type.

    with h5py.File(filename, 'r') as f:
        for qubit_group in f.keys():
            qubit_index = int(qubit_group[1:]) - 1
            qubit_data = {}
            group = f[qubit_group]

            for dataset_name in group.keys():
                # Attempt to map HDF5 keys to the target dictionaries' keys.
                if data_type == 'Res':
                    target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'Ext_QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies','Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                elif data_type == 'Rabi':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'SS':
                    target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T1':
                    target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2':
                    target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2E':
                    target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'stark2D':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q', 'Qu Frequency Sweep':'Qu Frequency Sweep',
                                   'Res Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type =='starkSpec':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q','P': 'P', 'shots':'shots','Gain Sweep':'Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                else:
                    raise ValueError(f"Unsupported data_type: {data_type}")

                try:
                    mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                    qubit_data[mapped_key] = [group[dataset_name][()]] * save_r  # Expand to match the desired length.

                except KeyError:
                    print(f"Warning: Key '{dataset_name}' not found in target dictionary for data_type '{data_type}'. Skipping.")
                    pass

            data[data_type][qubit_index] = qubit_data

    return data


def load_h5_data(filename, data_type, save_r=1):
    """
    Save data to an HDF5 file with a timestamped filename.
    Hybrid version of Joyce and Olivias two methods, authored by Dylan

    The function creates the destination folder if it does not exist. It generates a filename
    containing the current date/time, data type, batch number, and a replication factor. It then
    iterates over the provided data (organized per qubit) and saves each dataset within its respective
    group.

    Parameters:
        data (dict): Nested dictionary where each key corresponds to a qubit index and the value is a 
                     dictionary of dataset names and data.
        outer_folder_expt (str): Path to the folder where the HDF5 file should be saved.
        data_type (str): A descriptor for the type of data (used in filename and attributes).
        batch_num (int or str): Batch number to include in the filename and file attributes.
        save_r (int or str): Replication factor for the number of items per batch (included in filename and attributes).
    """

    data = {data_type: {}}  # Initialize the main output dictionary with the data_type.

    ## Define the target data fields, necessary if by-shot data is saved
    global_fields = ['Dates', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
    target_fields = {
        "Res": ['freq_pts', 'freq_center', 'Amps', 'Found Freqs']+global_fields,
        "QSpec": ['I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Recycled QFreq']+global_fields,
        "Ext_QSpec": ['I', 'Q', 'Frequencies']+global_fields,
        "Rabi": ['I', 'Q', 'Gains', 'Fit']+global_fields,
        "SS": ['Fidelity', 'Angle', 'I_g', 'Q_g', 'I_e', 'Q_e']+global_fields,
        "T1": ['T1', 'Errors', 'I', 'Q', 'Delay Times', 'Fit']+global_fields,
        "T2": ['T2', 'Errors', 'I', 'Q', 'Delay Times', 'Fit']+global_fields,
        "T2E": ['T2E', 'Errors', 'I', 'Q', 'Delay Times', 'Fit']+global_fields,
        "stark2D": ['I', 'Q', 'Qu Frequency Sweep', 'Res Gain Sweep']+global_fields,
        "starkSpec": ['I', 'Q', 'P', 'shots', 'Gain Sweep']+global_fields,
    }

    ## Open the file for pulling the data
    with h5py.File(filename, 'r') as f:
        
        ## Loop over all the top-level keys in the file (i.e., Qubit index)
        for qubit_group in f.keys():
            
            ## Convert group name (e.g., "Q1") to a zero-based index.
            qubit_index = int(qubit_group[1:]) - 1
            
            qubit_data = {} ## temporary container for output
            
            ## If this is a data type that is not defined, throw an error
            if data_type not in target_fields.keys():
                raise ValueError(f"Unsupported data_type: {data_type}")

            ## Now check all the keys in this data group (i.e., for this qubit)
            for dataset_name in f[qubit_group].keys():

                if dataset_name not in target_fields[data_type]:
                    print(f"Warning: Key '{dataset_name}' not found in target data field list for data_type '{data_type}'. Skipping.")

                else:
                    ## Copy the H5 dataset into the temporary dictionary for this qubit
                    qubit_data[dataset_name] = unwrap_singleton_list([f[qubit_group][dataset_name][()]] * save_r)

            ## Save this qubit's full data dict to our output container
            data[data_type][qubit_index] = qubit_data

    return data

