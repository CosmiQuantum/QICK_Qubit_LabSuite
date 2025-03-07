import numpy as np

def convert_datetimes_to_seconds(dt_objs):
    """
    Convert a sorted list of datetime objects to seconds relative to the first timestamp.

    Parameters
    ----------
    dt_objs : list
        Sorted list of datetime objects.

    Returns
    -------
    numpy.ndarray
        Array of time values in seconds relative to the first timestamp.
    """
    t0 = dt_objs[0]
    return np.array([(t - t0).total_seconds() for t in dt_objs])


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