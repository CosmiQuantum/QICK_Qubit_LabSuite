'''
ana_utils.py
    Collection of general processing methods that are broadly useful.
    These methods should have no dependencies to other files in the codebase.
'''

import numpy as np

def rotate_and_threshold(Ivals, Qvals, theta, threshold):
    i_new = Ivals * np.cos(theta) - Qvals * np.sin(theta)
    q_new = Ivals * np.sin(theta) + Qvals * np.cos(theta)

    states = (i_new > threshold)
    return i_new, q_new, states

def roll(data: np.ndarray) -> np.ndarray:
    """
    Smooth a 1D numpy array using a simple moving average filter with a window size of 5.

    The function computes the convolution of the data with a uniform kernel. It then pads the smoothed 
    data to maintain the original length.

    Parameters:
        data (np.ndarray): 1D array of numerical data.

    Returns:
        np.ndarray: Smoothed array with the same length as the input.
    """
    # Create a simple averaging kernel of size 5.
    kernel = np.ones(5) / 5
    # Convolve the input data with the kernel.
    smoothed = np.convolve(data, kernel, mode='valid')
    # Calculate the padding size required to match the original length.
    pad_size = (len(data) - len(smoothed)) // 2
    # Concatenate the unprocessed beginning and end segments with the smoothed data.
    return np.concatenate((data[:pad_size], smoothed, data[-pad_size:]))

def split_into_continuous_segments(time_sec, values, gap_threshold_factor=5):
    """
    Split the data into continuous segments based on a gap threshold.

    The gap threshold is defined as gap_threshold_factor * median(time differences).

    Parameters
    ----------
    time_sec : numpy.ndarray
        Array of time values in seconds.
    values : numpy.ndarray
        Array of corresponding measurement values.
    gap_threshold_factor : float, optional
        Factor to multiply the median of time differences to define a gap (default is 5).

    Returns
    -------
    segments_time : list of numpy.ndarray
        List of continuous time segments.
    segments_vals : list of numpy.ndarray
        List of corresponding measurement value segments.
    """
    dt_diffs = np.diff(time_sec)
    gap_threshold = gap_threshold_factor * np.median(dt_diffs)
    split_indices = np.where(dt_diffs > gap_threshold)[0] + 1
    segments_time = np.split(time_sec, split_indices)
    segments_vals = np.split(values, split_indices)
    return segments_time, segments_vals

def sort_date_time_data(date_times, data):
    """
    Sort the date/time objects and corresponding data values in ascending order.

    Parameters
    ----------
    date_times : list
        List of date/time objects.
    data : list
        List of measurement values.

    Returns
    -------
    sorted_times : tuple
        Sorted date/time objects.
    sorted_vals : tuple
        Sorted data values.
    """
    combined = list(zip(date_times, data))
    combined.sort(key=lambda x: x[0])
    if combined:
        sorted_times, sorted_vals = zip(*combined)
    else:
        sorted_times, sorted_vals = [], []
    return sorted_times, sorted_vals

def get_longest_continuous_segment(segments_time, segments_vals):
    """
    Returns the longest continuous segment from lists of continuous segments.

    Parameters
    ----------
    segments_time : list of arrays/lists
        Each element is a segment of time data.
    segments_vals : list of arrays/lists
        Each element is a segment of corresponding measurement data.

    Returns
    -------
    longest_time_segment : array/list
        The time data from the longest continuous segment.
    longest_vals_segment : array/list
        The measurement values from the longest continuous segment.
    """
    if not segments_time:
        return [], []
    lengths = [len(seg) for seg in segments_time]
    max_idx = np.argmax(lengths)
    return segments_time[max_idx], segments_vals[max_idx]

def max_offset_difference_with_x(x_values, y_values, offset):
    """
    Find the x-value corresponding to the maximum average absolute difference from a given offset.

    The function calculates the average absolute difference from the offset for triplets of consecutive y-values
    and returns the x-value (from the middle point of each triplet) that corresponds to the maximum difference.

    Parameters:
        x_values (array_like): Array of x-values.
        y_values (array_like): Array of y-values.
        offset (float): The offset value to compare against.

    Returns:
        tuple:
            corresponding_x: The x-value corresponding to the maximum average difference.
            max_average_difference: The maximum average absolute difference calculated.
    """
    max_average_difference = -1
    corresponding_x = None

    for i in range(len(y_values) - 2):
        y_triplet = y_values[i:i + 3]
        average_difference = sum(abs(y - offset) for y in y_triplet) / 3

        if average_difference > max_average_difference:
            max_average_difference = average_difference
            corresponding_x = x_values[i + 1]

    return corresponding_x, max_average_difference