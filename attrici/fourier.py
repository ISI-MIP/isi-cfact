import numpy as np
import pandas as pd


def series(t, p, modes):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, modes + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    # scale to range [0, 1]
    # x = (x + 1) / 2
    return x


def rescale(df, modes):
    """ This function computes a scaled (0, 1) fourier series for a given input dataset.
    An input vector of dates ("ds") must be available in a datestamp format.
    If the time vector has gaps (due to dropped NA's), the fourier series will
     also contain gaps (jumps in value).
    The output format will be of [len["ds"], 2*modes], where the first
    half of the columns contains the cos(x)-series and die latter half
    contains the sin(x)-series
    """

    # rescale the period, as t is also scaled
    p = 365.25 / (df["ds"].max() - df["ds"].min()).days
    x = series(df["t"], p, modes)
    return x


def get_fourier_valid(df, modes):

    """ Create a pandas Dataframe with all fourier series. They are named
    mode_X_Y with X refering to the position in settings.py modes.
    For example, for modes = [2,1,1,1],
    mode_0_3 is the last (fourth) series for the first mode (2) in the list.
    It is the sine with 2 periods per year.
    """

    x_fourier = pd.DataFrame()
    for i, mode in enumerate(modes):

        xf = rescale(df, mode)
        xff = pd.DataFrame(
            xf, columns=["mode_" + str(i) + "_" + str(j) for j in range(mode * 2)]
        )
        x_fourier = pd.concat([x_fourier, xff], axis=1)

    return x_fourier
