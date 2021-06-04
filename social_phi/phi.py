import numpy as np

def regression(X, Y):
    X_p = np.concatenate([np.ones(shape=(X.shape[0],1)), X], axis=1)
    beta, _, _ , _ = np.linalg.lstsq(X_p, Y, rcond=None)
    E = Y - X_p@beta
    return E


def phi_ar(X1, X2, base=np.e):
    assert X1.shape == X2.shape
    prod = lambda x: x[0]*x[1]
    
    S_X1 = np.cov(X1)
    S_part_X1_X2 = np.cov(regression(X2, X1))
    numerator = prod(np.linalg.slogdet(S_X1))
    denominator = prod(np.linalg.slogdet(S_part_X1_X2))
    term1 = 0.5*(numerator-denominator)

    S_M1 = [S_X1[i,i] for i in range(X1.shape[0])]
    S_part_M1_M2 = [np.cov(regression(X2[i:i+1], X1[i:i+1])) for i in range(X1.shape[0])]
    term2 = 0.5*np.sum([np.log(s_m1+1e-10) - np.log(s_part_m1_m2+1e-10) if s_part_m1_m2 > 0 else 0 for s_m1, s_part_m1_m2 in zip(S_M1, S_part_M1_M2)])

    return (term1 - term2)/ np.log(base)


def calc_phi_for_signal(sig, win_len=100, min_var=0.01, base=np.e):
    phis = []
    num_users = []
    X = sig
    hop_len = win_len
    n = (X.shape[1] - win_len) // hop_len    
    for i in range(n):
        X1 = X[:, i*hop_len : i*hop_len + win_len]
        X2 = X[:, (i+1)*hop_len : (i+1)*hop_len + win_len]
        
        var = np.var(X1, axis=1)
        threshold = max(sum(var) * min_var, 0.)
        valid_users = list(set(np.where(np.var(X1, axis=1) > threshold)[0]).intersection(set(np.where(np.var(X2, axis=1) > threshold)[0])))
        if len(valid_users) > 1:
            phi = phi_ar(X1[valid_users], X2[valid_users], base=base)
            if np.isnan(phi) or phi < 0:
                phi = np.nan
        else:
            phi = np.nan
        phis += [phi]
        num_users += [len(valid_users)]
    return np.array(phis), np.array(num_users)

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