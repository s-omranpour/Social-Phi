from typing import Dict
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from .signal import get_signal, get_single_signal
from .phi import phi_ar, calc_phi_for_signal
from .utils import fill_nans_with_mean, sliding_window

def phi_for_act_dict(
    acts : Dict[str, list], 
    time_scale : int = 24*3600,
    window : int = 30, 
    hop : int = 10,
    binarize : bool = False,
    base : float = np.e,
    filter_resol : int = 20,
    nan_policy : str = 'zero',
    silent : bool = False):
    '''
    ** parameters **

    acts = a dictionary with keys equal to user_id and values equal to a list of timebins in which the corresponding user was active
    time_scale = in seconds. used as the scale of time for future calculations
    window = window length used for distribution estimation
    hop = time delay for phi calculation
    binarize = whether to binarize the activity signal (instead of number of activites per timebin use 1 as active and 0 as inactive)
    base = base of the logarithms used in calculating phi. base=e means phi is in nats and base=2 means phi is in bits.
    nan_policy = {
        'zero' : fill nans with zeros
        'none' : do nothing
        'linear' : fill nans with linear interpolation
    }
    silent = disable progress bar



    ** outputs **
    phis = list of phi values for each time step
    n_users = list of int indicating number of users used for calculating phi in the k-th timestep.

    '''
    
    sig = get_signal(acts, time_scale=time_scale, binarize=binarize)
    return phi_for_act_sig(
        sig, base=base, window=window, hop=hop, filter_resol=filter_resol, nan_policy=nan_policy, silent=silent
    )


def phi_for_act_sig(
    sig : np.ndarray, 
    window : int = 30, 
    hop : int = 10,
    base : float = np.e, 
    filter_resol : int = 20,
    nan_policy : str = 'zero',
    silent : bool = False):
    '''
    ** parameters **

    sig = a 2-D numpy array with shape=(n_user, n_timesteps)
    window = window length used for distribution estimation
    hop = time delay for phi calculation
    base = base of the logarithms used in calculating phi. base=e means phi is in nats and base=2 means phi is in bits.
    nan_policy = {
        'zero' : fill nans with zeros
        'none' : do nothing
        'linear' : fill nans with linear interpolation
    }
    silent = disable progress bar
    

    ** outputs **
    phis = list of phi values for each time step
    n_users = list of int indicating number of users used for calculating phi in the k-th timestep.

    '''

    phis, n_users = calc_phi_for_signal(sig, win_len=window, hop_len=hop, base=base, filter_resol=filter_resol, silent=silent)
    if nan_policy == 'zero':
        phis[np.isnan(phis)] = 0.
    elif nan_policy == 'linear':
        phis = fill_nans_with_mean(phis)
    return phis, n_users

def experiment_hop_range(
    sig : np.ndarray, 
    window : int = 30, 
    min_hop : int = 1,
    max_hop : int = 7, 
    filter_resol : int = 20,
    silent : bool = False):

    res = {}
    prog = lambda x: x if silent else tqdm(x)
    for hop in prog(range(min_hop, max_hop+1)):
        phis, _ = calc_phi_for_signal(sig, win_len=window, hop_len=hop, base=2, filter_resol=filter_resol, silent=True)
        nans = np.isnan(phis).sum()
        zero_phis = phis.copy()
        zero_phis[np.isnan(zero_phis)] = 0.
        
        res[hop] = {
            'hop' : hop,
            'num_nans' : nans, 
            'num_valids': len(phis) - nans, 
            'vnr': (len(phis) - nans)/nans, 
            'raw_avg_phi': np.mean(phis[~np.isnan(phis)]),
            'zeroed_avg_phi': np.mean(zero_phis),
            'linear_avg_phi' : np.mean(fill_nans_with_mean(phis))
        }
    return pd.DataFrame(res).T
