import numpy as np
from scipy import stats


threshold = {
    "tas": (0, None),
    "tasrange": (0.01, None),
    "tasskew": (0.0001, 0.9999),
    "pr": (0.0000011574, None),
    "prsnratio": (0.01, 0.99),
    "hurs": (0.01, 99.99),
    "ps": (0, None),
    "rsds": (0, None),
    "rlds": (0, None),
    "wind": (0.01,),
}

bound = {
    "tas": (0.0, None),
    "tasrange": (0.0, None),
    "tasskew": (0.0, 1.0),
    "pr": (0.0, None),
    "prsnratio": (0.0, 1.0),
    "hurs": (0, 100.0),
    "ps": (0, None),
    "rsds": (0, 501),
    "rlds": (0, None),
    "wind": (0.0, None),
}

# for rsds, the global maximum is 500


def check_bounds(data, variable):

    lower = bound[variable][0]
    upper = bound[variable][1]

    if lower is not None and data.min() < lower:
        raise ValueError(data.min(), "is smaller than lower bound", lower, ".")

    if upper is not None and data.max() > upper:
        raise ValueError(data.max(), "is bigger than upper bound", upper, ".")


def scale_to_unity(data, variable, datamin=None, scale=None):

    """ Take a pandas Series and scale it linearly to
    lie within [0, 1]. Return pandas Series as well as the
    data minimum and the scale. """

    if scale is None: scale = data.max() - data.min()
    if datamin is None: datamin = data.min()
    scaled_data = (data - datamin) / scale

    return scaled_data, datamin, scale


def rescale_to_original(scaled_data, datamin, scale):

    """ Use a given datamin and scale to rescale to original. """

    return scaled_data * scale + datamin


def scale_and_mask(data, variable, datamin=None, scale=None):

    print("Mask", (data <= threshold[variable][0]).sum(), "values below lower bound.")
    data[data <= threshold[variable][0]] = np.nan
    try:
        print(
            "Mask", (data >= threshold[variable][1]).sum(), "values above upper bound."
        )
        data[data >= threshold[variable][1]] = np.nan
    except IndexError:
        pass

    if scale is None: scale = data.max() - data.min()
    if datamin is None: datamin = data.min()
    scaled_data = data / scale
    print("Min, max after scaling:", scaled_data.min(), scaled_data.max())

    return scaled_data, datamin, scale


def mask_and_scale_by_bounds(data, variable, datamin=None, scale=None):

    print("Mask", (data <= threshold[variable][0]).sum(), "values below lower bound.")
    data[data <= threshold[variable][0]] = np.nan
    print("Mask", (data >= threshold[variable][1]).sum(), "values above upper bound.")
    data[data >= threshold[variable][1]] = np.nan

    scale = bound[variable][1] - bound[variable][0]
    scaled_data = data / scale
    print("Scaling by bounds of variable, divide data by", scale)
    print("Min and max are", scaled_data.min(), scaled_data.max())
    return scaled_data, 0, scale


def scale_precip(data, variable, datamin=None, scale=None):

    data = data - threshold[variable][0]

    print("Mask", (data <= 0).sum(), "values below lower bound.")
    data[data <= 0] = np.nan
    fa, floc, fscale = stats.gamma.fit(data[~np.isnan(data)], floc=0)
    # for scipy.gamma: fscale = 1/beta
    # std = sqrt(fa/beta**2)
    if scale is None: scale = fscale * fa ** 0.5
    scaled_data = data / scale

    print("Min, max after scaling:", scaled_data.min(), scaled_data.max())
    return scaled_data, data.min(), scale


def refill_and_rescale(scaled_data, datamin, scale):

    # TODO: implement refilling of values that have been masked before.

    return scaled_data * scale


def rescale_and_offset_precip(scaled_data, datamin, scale):

    # TODO: implement refilling of values that have been masked before.

    return scaled_data * scale + threshold["pr"][0]


mask_and_scale = {
    "gmt": [scale_to_unity, rescale_to_original],
    "tas": [scale_to_unity, rescale_to_original],
    "ps": [scale_to_unity, rescale_to_original],
    "rlds": [scale_to_unity, rescale_to_original],
    "rsds": [mask_and_scale_by_bounds, refill_and_rescale],
    "wind": [scale_and_mask, refill_and_rescale],
    "hurs": [mask_and_scale_by_bounds, refill_and_rescale],
    "prsnratio": [mask_and_scale_by_bounds, refill_and_rescale],
    "tasskew": [mask_and_scale_by_bounds, refill_and_rescale],
    "tasrange": [scale_and_mask, refill_and_rescale],
    "pr": [scale_precip, rescale_and_offset_precip],
}
