import time, datetime
import numpy as np

def datetime_to_unix(dt):
    ''' Convert python datetime to Unix timestamp '''
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp

def unix_to_datetime(unix_timestamp):
    ''' Convert the Unix timestamp to a datetime object '''
    dt = datetime.datetime.fromtimestamp(unix_timestamp)
    return dt

def safe_convert_timestamp(ts):
    try:
        # If ts is NaN, return None instead of trying to convert it.
        if np.isnan(ts):
            return None
        return datetime.datetime.fromtimestamp(ts)
    except Exception:
        return None

def get_abs_min(start_time, dates):
    ''' returns absolute time in minutes '''
    abs_min = []
    for date in dates:
        abs_min.append(np.array((date - start_time).total_seconds()) / 60)
    return abs_min

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
