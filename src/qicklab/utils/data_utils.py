'''
data_utils.py
    Low-level methods to parse data and handle changing data types.
    This should have no dependencies on any files elsewhere in the package.
'''

import re, ast
import numpy as np




## ================= String Conversion Methods ================= ##

def is_numeric_string(s):
    """
    Check if a string can be converted to a float.

    Parameters:
        s (str): The string to check.

    Returns:
        bool: True if the string is numeric, False otherwise.
    """
    try:
        float(s)
        return True
    except Exception:
        return False

def ensure_str(x):
    """
    Decode byte arrays or bytes objects into Python strings.

    If the decoded string (or strings in an array) equals "None" (after stripping whitespace),
    return None (or a list with None values). Otherwise, return the decoded value.

    Parameters:
        x (bytes, np.ndarray, or str): The input value to decode.

    Returns:
        str, None, or list: The decoded string or list of strings, or None if the string equals "None".
    """
    # Handle numpy arrays of byte strings.
    if isinstance(x, np.ndarray) and x.dtype.kind == 'S':
        if x.size == 1:
            s = x.item().decode()
            return None if s.strip() == "None" else s
        else:
            decoded = [s.decode() for s in x]
            return [None if s.strip() == "None" else s for s in decoded]
    # Handle individual bytes objects.
    elif isinstance(x, bytes):
        s = x.decode()
        return None if s.strip() == "None" else s
    elif isinstance(x, str):
        return None if x.strip() == "None" else x
    return x

def convert_non_floats_to_strings(data_list):
    """
    Convert non-numeric items in a list to strings.

    Parameters:
        data_list (list): A list of items.

    Returns:
        list: A list where each item that is not an int, float, or np.float64
              is converted to a string.
    """
    # Check each element's type and convert non-floats to string.
    return [str(x) if not isinstance(x, (int, float, np.float64)) else x for x in data_list]

def string_to_float_list(input_string):
    """
    Convert a string representation of a list into an actual list of floats.

    The function cleans the string by removing occurrences of 'np.float64' and then uses ast.literal_eval
    for safe evaluation.

    Parameters:
        input_string (str): String representation of a list (e.g., "[1, 2, 3]").

    Returns:
        list or None: A list of floats if successful, otherwise None.
    """
    try:
        # Remove 'np.float64(' and ')' from the string.
        cleaned_string = input_string.replace('np.float64(', '').replace(')', '')
        # Safely evaluate the string into a Python list.
        float_list = ast.literal_eval(cleaned_string)
        return [float(x) for x in float_list]
    except Exception as e:
        print("Error: Invalid input string format. It should be a string representation of a list of numbers.", e)
        return None

def process_string_of_nested_lists(data):
    """
    Convert a string representing nested lists of numbers into a list of lists of floats.

    The function cleans the string by removing newline characters and extra whitespace,
    then uses regular expressions to extract the nested lists.

    Parameters:
        data (str): String representing nested lists (e.g., "[[1.0, 2.0], [3.0, 4.0]]").

    Returns:
        list: A list of lists of floats.
    """
    # Remove newline characters.
    data = data.replace('\n', '')
    # Remove extra whitespace within the brackets.
    data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
    data = data.replace('[ ', '[').replace('[ ', '[').replace('[ ', '[')
    # Keep only allowed characters.
    cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
    # Pattern to match content within square brackets.
    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, cleaned_data)
    result = []
    for match in matches:
        try:
            # Convert space-separated numbers to floats.
            numbers = [float(x.strip('[]').replace("'", "").replace("  ", ""))
                       for x in match.split() if x]
        except Exception as e:
            print("Error parsing nested list:", e)
            numbers = []
        result.append(numbers)
    return result

# def process_string_of_nested_lists(data):
#     ## From Joyce
#     # Remove extra whitespace and non-numeric characters.
#     data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
#     data = data.replace('[ ', '[')
#     data = data.replace('[ ', '[')
#     data = data.replace('[ ', '[')
#     cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
#     pattern = r'\[(.*?)\]'  # Regular expression to match data within brackets
#     matches = re.findall(pattern, cleaned_data)
#     result = []
#     for match in matches:
#         numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in
#                     match.split()]  # Convert strings to integers
#     result.append(numbers)

#     return result

##  ================= List & Dict Manipulation Methods ================= ##

def unwrap_singleton_list(val):
    """
    If val is a list containing exactly one element that is also a list,
    return that inner list. Otherwise, return val unchanged.
    """
    if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
        return val[0]
    return val

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# ------------------ Helper Functions for Data Processing ------------------




# def process_h5_data(data):
#     """
#     Process a string containing numeric data by filtering out unwanted characters.

#     The function removes newline characters and keeps only digits, minus signs, dots, spaces, and the letter 'e'
#     (for exponential notation), then converts the resulting tokens to floats.

#     Parameters:
#         data (str): String containing numeric data.

#     Returns:
#         list: A list of floats extracted from the string.
#     """
#     # Replace newline characters with a space.
#     data = data.replace('\n', ' ')
#     # Keep only valid numeric characters.
#     cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e'])
#     # Split the cleaned string and convert each part to float.
#     numbers = [float(x) for x in cleaned_data.split() if x]
#     return numbers

def process_h5_data(data):
    ## From Joyce
    # Check if the data is a byte string; decode if necessary.
    if isinstance(data, bytes):
        data_str = data.decode()
    elif isinstance(data, str):
        data_str = data
    else:
        raise ValueError("Unsupported data type. Data should be bytes or string.")

    data_str = data_str.strip().replace('\n', ' ')

    # Remove extra whitespace and non-numeric characters.
    cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])

    # Split into individual numbers, removing empty strings.
    numbers = [float(x) for x in cleaned_data.split() if x]
    return numbers




