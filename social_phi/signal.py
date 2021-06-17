import numpy as np
from collections import Counter


def get_single_signal(act, time_scale=24*3600, binarize=False):
    act = dict(Counter([int(t // time_scale) for t in act]))
    steps = sorted(act.keys()) #act
    a,b = steps[-1], steps[0]
    sig = np.zeros(shape=(a-b+1,))
    for step in steps:
        v = 1 if binarize else act[step]
        sig[step - b] = v
    return sig, a, b

def get_signal(activities, time_scale=24*3600, binarize=False):
    min_step = 999999999999999
    max_step = 0
    users_acts = {}
    
    for user, act in activities.items():
        sig, M, m = get_single_signal(act, time_scale, binarize)
        min_step = min(m, min_step)
        max_step = max(M, max_step)
        users_acts[user] = {'min': m, 'max': M, 'sig': sig}

    for user in users_acts:
        users_acts[user]['sig'] = np.pad(users_acts[user]['sig'], 
                                         pad_width=(users_acts[user]['min'] - min_step, 
                                                    max_step - users_acts[user]['max']))

    return np.stack([users_acts[user]['sig'] for user in users_acts], axis = 0)