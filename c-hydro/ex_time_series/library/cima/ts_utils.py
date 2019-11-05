# ----------------------------------------------------------------------------------------------------------------------
# Libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from library.pytesmo_mod.temporal_matchers import BasicTemporalMatching
# ----------------------------------------------------------------------------------------------------------------------

def df_period_selection(df, time_start, time_end):

    if time_start and time_end:

        try:
            datetime.strptime(time_start, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValueError("Incorrect data format, should be %Y-%m-%d %H:%M")
        try:
            datetime.strptime(time_end, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValueError("Incorrect data format, should be %Y-%m-%d %H:%M")

        date_start = datetime.strptime(time_start, '%Y-%m-%d %H:%M')
        date_end = datetime.strptime(time_end, '%Y-%m-%d %H:%M')

        df_time = df.iloc[df.index.get_loc(date_start, method='nearest'):
                          df.index.get_loc(date_end, method='nearest')]

    else:
        df_time = df

    return df_time

# Method to find time in a dataset
def df_time_matching(df, date, window=24, frequency="H"):
    try:
        datetime.strptime(date, '%Y-%m-%d %H:%M')
    except ValueError:
        raise ValueError("Incorrect data format, should be %Y-%m-%d %H:%M")

    date_cur = datetime.strptime(date, '%Y-%m-%d %H:%M')

    win_hours = float(window)/2
    date_to = date_cur + timedelta(hours=int(win_hours))
    date_from = date_cur - timedelta(hours=int(win_hours))

    time_to = date_to.strftime('%Y-%m-%d %H:%M')
    time_from = date_from.strftime('%Y-%m-%d %H:%M')

    time_steps = pd.date_range(start=time_from, end=time_to, freq=frequency)

    df_period = df.loc[date_from:time_to]
    df_time = df_period.iloc[df_period.index.get_loc(date_cur, method='nearest')]

    return df_time, df_period


# ----------------------------------------------------------------------------------------------------------------------
# Method to find temporal matching between two dataset(s)
def df_temporal_matching(df_ref, df_k, name_ref='ASCAT', name_k='ERA5', window=72, drop_duplicates=False):

    window_match = float(window) / 24

    df_dict = {}
    df_dict[name_k] = df_k
    df_dict.update({name_ref: df_ref})

    oTempMatch = BasicTemporalMatching(window=window_match, drop_duplicates=drop_duplicates)
    results_matched = oTempMatch.combinatory_matcher(df_dict, name_ref, 2)

    data = results_matched[name_ref, name_k]

    ts_ref_match = data[name_ref]
    ts_k_match = data[name_k]

    return ts_ref_match, ts_k_match
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Method to compute dates group by month and year
def compute_times(startDate, endDate, split_unit='month', split_n=1,  startAnalysis=None, endAnalysis=None):

    if startAnalysis is None and endAnalysis is None:
        cur_date = start = datetime.strptime(startDate, '%Y-%m-%d').date()
        end = datetime.strptime(endDate, '%Y-%m-%d').date()
    else:
        cur_date = start = datetime.strptime(startAnalysis, '%Y-%m-%d').date()
        end = datetime.strptime(endAnalysis, '%Y-%m-%d').date()

    dates_start = []
    dates_end = []
    if split_unit:
        dates_start.append(start)
        while cur_date < end:
            if split_unit == 'month':
                cur_date += relativedelta(months=split_n)
            elif split_unit == 'year':
                cur_date += relativedelta(years=split_n)
            dates_end.append(cur_date)
            dates_start.append(cur_date)

        del dates_start[-1]
    else:
        dates_start.append(start)
        dates_end.append(end)

    return dates_start, dates_end
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Function to check bit setting
def get_bit(a, bit_pos):
    """
    Returns 1 or 0 if bit is set or not.

    Parameters
    ----------
    a : int or numpy.ndarray
      Input array.
    bit_pos : int
      Bit position. First bit position is right.

    Returns
    -------
    b : numpy.ndarray
      1 if bit is set and 0 if not.
    """
    return np.clip(np.bitwise_and(a, 2 ** (bit_pos-1)), 0, 1)

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Method to get dataset name(s)
def get_dataset_names(ref_key, datasets):
    """
    Get dataset names in correct order as used in the validation framework
        -) reference dataset = ref
        -) first other dataset = k1
        -) second other dataset = k2
    This is important to correctly iterate through the H-SAF metrics and to
    save each metric with the name of the used datasets

    Parameters
    ----------
    ref_key: basestring
        Name of the reference dataset
    datasets: dict
        Dictionary of dictionaries as provided to the validation framework
        in order to perform the validation process.

    Returns
    -------
    dataset_names: list
        List of the dataset names in correct order

    """
    ds_dict = {}
    for ds in datasets.keys():
        ds_dict[ds] = datasets[ds]['columns']
    ds_names = get_result_names(ds_dict, ref_key, n=3)
    dataset_names = []
    for name in ds_names[0]:
        dataset_names.append(name[0])

    return dataset_names

# ----------------------------------------------------------------------------------------------------------------------
