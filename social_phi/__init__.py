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
    silent : bool = False):
    '''
    ** parameters **

    acts = a dictionary with keys equal to user_id and values equal to a list of timebins in which the corresponding user was active
    time_scale = in seconds. used as the scale of time for future calculations
    fill_nans = whether to fill nan values occuring during calculation of phi with means.
    window = time delay used in calculating phi
    binarize = whether to binarize the activity signal (instead of number of activites per timebin use 1 as active and 0 as inactive)
    base = base of the logarithms used in calculating phi. base=e means phi is in nats and base=2 means phi is in bits.
    var_threshold = threshold to filter less active users for more numerical stability. 
                    threshold = var_threshold * sum(vars)



    ** outputs **
    phis = list of phi values for each time step
    n_users = list of int indicating number of users used for calculating phi in the k-th timestep.

    '''
    
    sig = get_signal(acts, time_scale=time_scale, binarize=binarize)
    return phi_for_act_sig(
        sig, base=base, window=window, hop=hop, fill_nans=fill_nans, silent=silent
    )


def phi_for_act_sig(
    sig : np.ndarray, 
    window : int = 30, 
    hop : int = 10,
    base : float = np.e, 
    silent : bool = False):
    '''
    ** parameters **

    sig = a 2-D numpy array with shape=(n_user, n_timesteps)
    fill_nans = whether to fill nan values occuring during calculation of phi with means.
    window = time delay used in calculating phi
    base = base of the logarithms used in calculating phi. base=e means phi is in nats and base=2 means phi is in bits.
    var_threshold = threshold to filter less active users for more numerical stability. 
                    threshold = var_threshold * sum(vars)



    ** outputs **
    phis = list of phi values for each time step
    n_users = list of int indicating number of users used for calculating phi in the k-th timestep.

    '''

    phis, n_users = calc_phi_for_signal(sig, win_len=window, hop_len=hop, base=base, silent=silent)
    return fill_nans_with_mean(phis), phis, n_users

def experiment_hop_range(
    sig : np.ndarray, 
    window : int = 30, 
    min_hop : int = 1,
    max_hop : int = 7, 
    silent : bool = False):

    res = {}
    prog = lambda x: x if silent else tqdm
    for hop in prog(range(min_hop, max_hop+1)):
        phis, _ = calc_phi_for_signal(sig, win_len=window, hop_len=hop, base=2, silent=True)
        nans = np.isnan(phis).sum()
        avg_phi = np.mean(phis[~np.isnan(phis)])
        res[hop] = {'num_nans' : nans, 'num_valids': len(phis) - nans, 'vnr': (len(phis) - nans)/nans, 'avg_phi': avg_phi}
    return pd.DataFrame(res).T
