import os
import h5py
import numpy as np

from .data_utils import convert_non_floats_to_strings

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
