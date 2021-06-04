from typing import Dict
import numpy as np

from .signal import get_signal
from .phi import phi_ar, calc_phi_for_signal, fill_nans_with_mean

def phi_for_act_dict(
    acts : Dict[str, list], 
    fill_nans: bool = True, 
    window : int = 30, 
    binarize : bool = False,
    base : float = np.e,
    var_threshold : float = 0.01):
    '''
    ** parameters **

    acts = a dictionary with keys equal to user_id and values equal to a list of timebins in which the corresponding user was active
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

    sig = get_signal(acts, binarize)
    phis, n_users = calc_phi_for_signal(sig, win_len=window, min_var=var_threshold, base=base)
    if fill_nans:
        return fill_nans_with_mean(phis), n_users
    return phis, n_users


def phi_for_act_sig(
    sig : np.ndarray, 
    fill_nans: bool = True, 
    window : int = 30, 
    base : float = np.e,
    var_threshold : float = 0.01):
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

    phis, n_users = calc_phi_for_signal(sig, win_len=window, min_var=var_threshold, base=base)
    if fill_nans:
        return fill_nans_with_mean(phis), n_users
    return phis, n_users