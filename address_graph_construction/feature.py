import numpy as np
import statsmodels.robust as rb
from scipy import stats

def statistics(values):
    maxium = max(values)
    minium = min(values)
    range_ = maxium - minium 
    midrange = (maxium+minium)*0.5 
    mean = np.mean(values)
    std = np.std(values)
    var = np.var(values)
    be_mid,mid,fomid = np.percentile(values, [25, 50, 75])
    MAD = rb.scale.mad(values) 
    variation = np.std(values)/np.mean(values)
    sum_ = sum(values)
    number = len(values)
    kurtosis = stats.kurtosis(values)    
    skew = stats.skew(values)
    tilt = 0 if mean>mid else 1
    return [maxium, minium, range_, midrange, mean, std, var, be_mid, mid,
            fomid, MAD, variation, sum_, number, kurtosis, skew, tilt]