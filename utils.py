import numpy as np


# Autocorrelation function
def autocorrelation(data, t_max, burn_in=0):
    '''
    Function to compute the autocorrelation of a time series data up to a lag t_max.
    '''
    considered = data[burn_in:]
    considered -= np.mean(considered)
    result = np.correlate(considered, considered, mode='full')
    result = result[result.size // 2:]
    result /= result[0]
    return result[:t_max]

# Autocorrelation time
def get_autocorrelation_time(autocorr):
    '''
    Function to compute the autocorrelation time.
    '''
    smaller = np.where(autocorr < np.exp(-1)*autocorr[0])[0]
    return smaller[0] if len(smaller) > 0 else len(autocorr)
