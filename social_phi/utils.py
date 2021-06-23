import numpy as np
from tqdm.notebook import tqdm

def sliding_window(arr, agg_func=np.mean, window=10, hop=10):
    res = []
    n = (arr.shape[-1] - window) // hop
    for i in range(n):
        res += [agg_func(arr[..., i*hop : i*hop + window], axis=1)]
    return np.stack(res, axis=1)

def nan_mean_value(t_arr, nan_idx):
    arr = t_arr.copy()
    idx1 = idx2 = None
    if np.isnan(arr[0]):
        arr = np.append(0, arr)
        nan_idx += 1
    if np.isnan(arr[-1]):
        arr = np.append(arr, 0)
    for i in range(nan_idx-1,-1,-1):
        if not np.isnan(arr[i]):
            idx1 = i
            break
            
    for i in range(nan_idx+1, len(arr)):
        if not np.isnan(arr[i]):
            idx2 = i
            break 

    m = (arr[idx2] - arr[idx1]) / (idx2 - idx1)
    return arr[idx1] + m*(nan_idx - idx1)

def fill_nans_with_mean(arr):
    arr = arr.copy()
    nans = np.where(np.isnan(arr))[0]
    for nan in nans:
        arr[nan] = nan_mean_value(arr, nan)
    return arr