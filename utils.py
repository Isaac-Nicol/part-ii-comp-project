import numpy as np


# Autocorrelation function
def autocorrelation(data, t_max):
    '''
    Function to compute the autocorrelation of a time series data up to a lag t_max.
    '''
    data -= np.mean(data)
    result = np.correlate(data, data, mode='full')
    result = result[result.size // 2:]
    result /= result[0]
    return result[:t_max]

# Autocorrelation time
def get_autocorrelation_time(autocorr, burn_in=0):
    '''
    Function to compute the autocorrelation time.
    '''
    considered = autocorr[burn_in:]
    smaller = np.where(considered < np.exp(-1)*considered[0])[0]
    return smaller[0] if len(smaller) > 0 else len(considered)