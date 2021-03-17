import numpy as np
from scipy.signal import butter, filtfilt
from numpy.lib.stride_tricks import as_strided as ast


def remove_conditions(df, subinfo, condition):
    """
    REMOVE_CONDITIONS is a helper function that will remove condtions for *ALL* subjects/shoe conditions in a dataframe.
    :param df: Pandas df or tuple of dfs with same number of rows as `sub_info` df.
    :param subinfo: Pandas dataframe containing information about subject and trial conditions. Has the following column
    headers: 'Sub', 'Shoe', 'Condition', 'Sex', 'Age', 'Height', 'Mass'.
    :param condition: String or tuple of strings of condition describing speed/slope/step freq combination to remove
    REGARDLESS OF SHOE (e.g. 'C10').
    :return: Pandas df with indicated conditions removed for ALL SUBJECTS.
    """
    if type(condition) is tuple:
        for c in condition:
            df = df.drop(df[(subinfo['Condition'] == c)].index)
            subinfo = subinfo.drop(subinfo.index[(subinfo['Condition'] == c)])  # update subinfo to match df shape
    else:
        df = df.drop(df[(subinfo['Condition'] == condition)].index)
    return df


def define_conditions(subinfo):
    """
    DEFINE_CONDITIONS converts the "S#C#" format of trial conditions into the numerical values for running speed,
    slope, and step frequency. Ignores shoe.

    :param subinfo: pandas dataframe containing information about subject and trial conditions. Has the following column
    headers: 'Sub', 'Shoe', 'Condition', 'Sex', 'Age', 'Height', 'Mass'.
    :return: Dictionary containing the numerical speeds, slopes, and step frequencies relating to the shoe/condition
    combinations in `subinfo`.
    """
    speed = [np.nan] * len(subinfo)
    slope = [np.nan] * len(subinfo)
    sf = [999] * len(subinfo)

    # loop through all trials :'(
    for n, s in enumerate(subinfo['Shoe']):
        c = subinfo['Condition'].iloc[n]
        if s == 'S1' and c == 'C1':
            speed[n] = 2.5
            slope[n] = 10
            sf[n] = 0
        elif s == 'S1' and c == 'C2':
            speed[n] = 3.33
            slope[n] = 10
            sf[n] = 0
        elif s == 'S1' and c == 'C3':
            speed[n] = 3.33
            slope[n] = 10
            sf[n] = 10
        elif s == 'S1' and c == 'C4':
            speed[n] = 3.33
            slope[n] = 10
            sf[n] = -10
        elif s == 'S1' and c == 'C5':
            speed[n] = 4.17
            slope[n] = 10
            sf[n] = 0
        elif s == 'S1' and c == 'C6':
            speed[n] = 2.5
            slope[n] = 5
            sf[n] = 0
        elif s == 'S1' and c == 'C7':
            speed[n] = 3.33
            slope[n] = 5
            sf[n] = 0
        elif s == 'S1' and c == 'C8':
            speed[n] = 3.33
            slope[n] = 5
            sf[n] = 10
        elif s == 'S1' and c == 'C9':
            speed[n] = 3.33
            slope[n] = 5
            sf[n] = -10
        elif s == 'S1' and c == 'C10':
            speed[n] = 4.17
            slope[n] = 5
            sf[n] = 0
        elif s == 'S1' and c == 'C11':
            speed[n] = 2.5
            slope[n] = 0
            sf[n] = 0
        elif s == 'S1' and c == 'C12':
            speed[n] = 3.33
            slope[n] = 0
            sf[n] = 0
        elif s == 'S1' and c == 'C13':
            speed[n] = 3.33
            slope[n] = 0
            sf[n] = 10
        elif s == 'S1' and c == 'C14':
            speed[n] = 3.33
            slope[n] = 0
            sf[n] = -10
        elif s == 'S1' and c == 'C15':
            speed[n] = 4.17
            slope[n] = 0
            sf[n] = 0
        elif s == 'S1' and c == 'C16':
            speed[n] = 2.5
            slope[n] = -5
            sf[n] = 0
        elif s == 'S1' and c == 'C17':
            speed[n] = 3.33
            slope[n] = -5
            sf[n] = 0
        elif s == 'S1' and c == 'C18':
            speed[n] = 3.33
            slope[n] = -5
            sf[n] = 10
        elif s == 'S1' and c == 'C19':
            speed[n] = 3.33
            slope[n] = -5
            sf[n] = -10
        elif s == 'S1' and c == 'C20':
            speed[n] = 3.33
            slope[n] = -5
            sf[n] = 0
        elif s == 'S1' and c == 'C21':
            speed[n] = 4.17
            slope[n] = -5
            sf[n] = 0
        elif s == 'S1' and c == 'C22':
            speed[n] = 2.5
            slope[n] = -10
            sf[n] = 0
        elif s == 'S1' and c == 'C23':
            speed[n] = 3.33
            slope[n] = -10
            sf[n] = 0
        elif s == 'S1' and c == 'C24':
            speed[n] = 3.33
            slope[n] = -10
            sf[n] = 10
        elif s == 'S1' and c == 'C25':
            speed[n] = 3.33
            slope[n] = -10
            sf[n] = -10
        elif s == 'S1' and c == 'C26':
            speed[n] = 4.17
            slope[n] = -10
            sf[n] = 0
        elif s == 'S1' and c == 'C27':  # WALKING
            speed[n] = np.nan
            slope[n] = np.nan
            sf[n] = np.nan
        elif s == 'S1' and c == 'C28':
            speed[n] = 1.8
            slope[n] = 10
            sf[n] = 0
        elif s == 'S1' and c == 'C29':  # WALKING
            speed[n] = np.nan
            slope[n] = np.nan
            sf[n] = np.nan
        elif s == 'S1' and c == 'C30':
            speed[n] = 1.6
            slope[n] = 15
            sf[n] = 0
        # shoe 2
        elif s == 'S2' and c == 'C1':
            speed[n] = 3.33
            slope[n] = 10
            sf[n] = 0
        elif s == 'S2' and c == 'C2':
            speed[n] = 3.33
            slope[n] = 5
            sf[n] = 0
        elif s == 'S2' and c == 'C3':
            speed[n] = 3.33
            slope[n] = 0
            sf[n] = 0
        elif s == 'S2' and c == 'C4':
            speed[n] = 3.33
            slope[n] = -5
            sf[n] = 0
        elif s == 'S2' and c == 'C5':
            speed[n] = 3.33
            slope[n] = -10
            sf[n] = 0

    d = dict()
    d['speed'] = speed
    d['slope'] = slope
    d['step_freq'] = sf
    return d


def buttfilt(x, fs, fc, order, axis=1):
    """
    BUTTFILT applies a lowpass butterworth filter with a correction factor described in
    Research Methods in Biomechanics (2ed) and explained further here:
    https://github.com/alcantarar/dryft/issues/22#issuecomment-557771825
    :param x: signal to filter
    :param fs: sampling frequency (Hz)
    :param fc: lowpass cutoff frequency (Hz). If tuple, will do bandpass (low, high).
    :param order: final desired filter order. Must be even number.
    :param axis: axis to filter along. Default is `axis=1`.
    :return:
    """
    n_pass = 2  # two passes (one forward, one backward to be zero-lag)
    if (order % 2) != 0:
        raise ValueError('order must be even integer')
    else:
        order = order / 2
        fn = fs / 2
        # Correction factor per Research Methods in Biomechanics (2e) pg 288
        c = (2 ** (1 / n_pass) - 1) ** (1 / (2 * order))
        if type(fc) is tuple:
            # bandpass filter:
            wn_low = (np.tan(np.pi*fc[0]/fs))/c  # Apply correction to adjusted cutoff freq (lower boundary)
            wn_up = (np.tan(np.pi*fc[1]/fs))/c  # Apply correction to adjusted cutoff freq (upper boundary)
            fc_corrected_low = np.arctan(wn_low)*fs/np.pi  # Hz
            fc_corrected_up = np.arctan(wn_up)*fs/np.pi  # Hz
            b, a = butter(order, [fc_corrected_low/fn, fc_corrected_up/fn], btype='band')

            x_filt = filtfilt(b, a, x, axis=axis)

        else:
            # lowpass filter:
            wn = (np.tan(np.pi*fc/fs))/c  # Apply correction to adjusted cutoff freq
            fc_corrected = np.arctan(wn)*fs/np.pi  # Hz
            b, a = butter(order, fc_corrected/fn)

            x_filt = filtfilt(b, a, x, axis=axis)

        return x_filt


def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    """
    CHUNK_DATA was made by Matthew Johnson (github username mattjj). Gist URL (accessed 8/2020):
    https://gist.github.com/mattjj/5213172. It is clever. I use this function within window_data_centered().
    :param data: data to be windowed
    :param window_size: size of window
    :param overlap_size: size of overlap between windows
    :param flatten_inside_window: flatten ndim data inside window
    :return: ndarray with shape: (trials, number of windows, window size).
    """
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size * data.shape[1]),
        strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def window_data_centered(data, window_size, verbose=True):
    """
    WINDOW_DATA_CENTERED uses chunk_data() to create windows centered about a given frame. This is accomplished
    by padding the trial with data at start/end. Input of 5 trials, each 1000 frames long (5, 1000) and window_size of 6
    would result in output shape of (5, 1000, 6) because overlap between windows is window_size-1. X at time points t-3,
    t-2, t-1, t, t+1, and t+2 create the 6-frame window that corresponds to Y at time t.
    :param data: data (shape (trials, frames)) to be windowed using chunk_data().
    :param window_size: size of windows. Must be even number.
    :param verbose: If True, prints input/output shapes. Default True.
    :return: ndarray of shape (trials, number of windows, window size)
    """
    if (window_size % 2) != 0:
        raise ValueError('window_size must be divisible by 2')
    else:
        pad = int(window_size/2)
        overlap = window_size - 1
        ds = data.shape
        # pad with nearest value at edges
        data = np.pad(data, [(0, 0), (pad, pad-1)], 'edge')

        # apply chunk_data()
        data = np.apply_along_axis(chunk_data, 1, data, window_size, overlap, True)
        if verbose:
            print('input shape:', ds)
            print('output shape:', data.shape)

        return data


def signal_features(data, fs):
    """
    SIGNAL_FEATURES used in generate_features() and calculates the following features for each window:
    - mean (np.mean)
    - standard deviation (np.std)
    - range (np.ptp)
    - Average 1st Derivative (np.gradient) #commented out
    - Average 2nd Derivative (np.gradient) #commented out
    - Average 1st Integral (np.cumtrapz) #commented out
    - Average 2nd Integral (np.cumtrapz) #commented out

    :param data: ndarray with shape (trials, number of windows, window size)
    :return: ndarray with shape (trials, number of windows, number of features)
    """
    mean = np.mean(data, axis=2)
    std = np.std(data, axis=2)
    rg = np.ptp(data, axis=2)
    # diff = np.mean(np.gradient(data, axis=2), axis=2)
    # diffdiff = np.mean(np.gradient(np.gradient(data, axis=2), axis=2), axis=2)
    # integral = np.mean(cumtrapz(data, np.linspace(0, data.shape[2]/fs, data.shape[2])), axis=2)
    # temp_integral = cumtrapz(data, np.linspace(0, data.shape[2]/fs, data.shape[2]))
    # integralintegral = np.mean(cumtrapz(temp_integral,
    #                                    np.linspace(0,(data.shape[2]-1)/fs, (data.shape[2]-1))
    #                                    ),
    #                            axis=2)
    features = np.concatenate((np.expand_dims(mean, 2),
                               np.expand_dims(std, 2),
                               np.expand_dims(rg, 2),
                               # np.expand_dims(diff, 2),
                               # np.expand_dims(diffdiff, 2),
                               # np.expand_dims(integral, 2),
                               # np.expand_dims(integralintegral, 2)
                               ), axis=2)

    return features


def subject_info_features(subinfo, test_sub_num, data_shape):
    """
    SUBJECT_INFO_FEATURES used in generate_features() and extracts the following features from subinfo df:
    - Subject height
    - Subject mass
    - Running speed
    - Treadmill slope
    - % of steps Rearfoot strike (RF)
    - % of steps Midfoot Strike (MF)
    - % of steps Forefoot Strike (FF)
    :param subinfo: Pandas dataframe containing information about subject and trial conditions. Has the following column
    headers: 'Sub', 'Shoe', 'Condition', 'Sex', 'Age', 'Height', 'Mass'.
    :param test_sub_num: Integer of subject to be used to test model when doing Leave-One-Subject-Out Cross Validation.
    To ONLY calculate features for test_sub_num, make it negative.
    :param data_shape: shape of windowed data to match feature shapes
    :return: concatenated features with shape of data_shape.
    """
    if test_sub_num >= 0:
        ht = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['Height']].to_numpy(), data_shape)
        ms = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['Mass']].to_numpy(), data_shape)
        sp = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['Speed']].to_numpy(), data_shape)
        sl = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['Slope']].to_numpy(), data_shape)
        rf = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['RFS']].to_numpy(), data_shape)
        mf = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['MFS']].to_numpy(), data_shape)
        ff = np.tile(subinfo.loc[subinfo['Sub'] != test_sub_num, ['FFS']].to_numpy(), data_shape)
    else:
        test_sub_num = np.abs(test_sub_num)
        ht = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['Height']].to_numpy(), data_shape)
        ms = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['Mass']].to_numpy(), data_shape)
        sp = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['Speed']].to_numpy(), data_shape)
        sl = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['Slope']].to_numpy(), data_shape)
        rf = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['RFS']].to_numpy(), data_shape)
        mf = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['MFS']].to_numpy(), data_shape)
        ff = np.tile(subinfo.loc[subinfo['Sub'] == test_sub_num, ['FFS']].to_numpy(), data_shape)
    features = np.concatenate((np.expand_dims(ht, 2),
                               np.expand_dims(ms, 2),
                               np.expand_dims(sp, 2),
                               np.expand_dims(sl, 2),
                               np.expand_dims(rf, 2),
                               np.expand_dims(mf, 2),
                               np.expand_dims(ff, 2)), axis=2)

    return features


def generate_features(data, fs, subinfo, test_sub_num, include_sub_info_feats=True):
    """

    :param data: Pandas dataframe containing windowed data (see window_data_centered()).
    :param fs: Sampling frequency of [data] in Hz.
    :param subinfo: Pandas dataframe containing information about subject and trial conditions. Has the following column
    headers: 'Sub', 'Shoe', 'Condition', 'Sex', 'Age', 'Height', 'Mass'.
    :param test_sub_num: Integer of subject to be used to test model when doing Leave-One-Subject-Out Cross Validation.
    :param include_sub_info_feats: Boolean to concatenate signal and subject info/condition features. Default True. If
    False, return only features calculated from `data`.
    :return: concatenated features with shape of data_shape.
    """
    sig_feats = signal_features(data=data, fs=fs)
    if include_sub_info_feats:
        sub_cond_feats = subject_info_features(subinfo, test_sub_num, data.shape[1])
        features = np.concatenate((sig_feats, sub_cond_feats), axis=2)
    else:
        features = sig_feats

    return features
