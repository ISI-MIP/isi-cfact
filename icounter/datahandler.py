import numpy as np
import pandas as pd
import pathlib
import sys
import netCDF4 as nc
import icounter.const as c
import icounter.fourier as fourier


def create_output_dirs(output_dir):

    """ params: output_dir: a pathlib object """

    for d in ["cfact", "traces", "timeseries"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def make_cell_output_dir(output_dir, sub_dir, lat, lon, variable):

    """ params: output_dir: a pathlib object """

    lat_sub_dir = output_dir / sub_dir / variable / ("lat_" + str(lat))
    lat_sub_dir.mkdir(parents=True, exist_ok=True)

    if sub_dir == "traces":
        #
        return lat_sub_dir / ("lon" + str(lon))
    else:
        return lat_sub_dir


def get_subset(df, subset, seed):

    orig_len = len(df)
    if subset > 1:
        np.random.seed(seed)
        subselect = np.random.choice(orig_len, np.int(orig_len / subset), replace=False)
        df = df.loc[np.sort(subselect), :].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(len(df), "data points used from originally", orig_len, "datapoints.")

    return df


def create_dataframe(nct_array, units, data_to_detrend, gmt, variable):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct_array, unit="D", origin=pd.Timestamp(units.lstrip("days since"))
    )

    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)

    f_scale = c.mask_and_scale["gmt"][0]
    gmt_scaled, _, _ = f_scale(gmt_on_data_cal, "gmt")

    c.check_bounds(data_to_detrend, variable)
    try:
        f_scale = c.mask_and_scale[variable][0]
    except KeyError as error:
        print(
            "Error:",
            variable,
            "is not implement (yet). Please check if part of the ISIMIP set.",
        )
        raise error

    y_scaled, datamin, scale = f_scale(pd.Series(data_to_detrend), variable)

    tdf = pd.DataFrame(
        {
            "ds": ds,
            "t": t_scaled,
            "y": data_to_detrend,
            "y_scaled": y_scaled,
            "gmt": gmt_on_data_cal,
            "gmt_scaled": gmt_scaled,
        }
    )
    if variable == "pr":
        tdf["is_dry_day"] = np.isnan(y_scaled)

    return tdf, datamin, scale

def create_dataframe_extended(nct_array, 
                              units, 
                              dataframe_nonextended,
                              data_extended,
                              last_gmt_value,
                              variable):

    ds = pd.to_datetime(
        nct_array, unit="D", origin=pd.Timestamp(units.lstrip("days since"))
    )

    # take old gmt up to the last index in the old dataframe. Take the last gmt_value in the new dataframe.
    # Interpolate linearly between the last gmt value of the old timeseries
    # and the last gmt value of the new datafram.
    dataframe_extended = pd.DataFrame(data={'gmt': np.nan}, index=ds)
    dataframe_extended['gmt'] = dataframe_nonextended['gmt']
    dataframe_extended.iloc[-1]['gmt'] = last_gmt_value
    dataframe_extended.interpolate(inplace=True)
    # assert gmt is equal up to the extended period
    # todo mv asserts to a unittest
    np.testing.assert_allclose(dataframe_nonextended['gmt'],
                               dataframe_extended.loc[:dataframe_nonextended.index[-1], 'gmt'])

    # rescale gmt
    shift = dataframe_nonextended['gmt'].min()
    scale = dataframe_nonextended['gmt'].max() - dataframe_extended['gmt'].min()
    dataframe_extended['gmt_scaled'] = (dataframe_extended['gmt'] - shift) / scale
    # todo mv asserts to a unittest
    np.testing.assert_allclose(dataframe_nonextended['gmt_scaled'],
                               dataframe_extended.loc[:dataframe_nonextended.index[-1], 'gmt_scaled'])
    # todo test rescaling for all variables
    c.check_bounds(data_extended, variable)
    try:
        f_scale = c.mask_and_scale[variable][0]
    except KeyError as error:
        print(
            "Error:",
            variable,
            "is not implement (yet). Please check if part of the ISIMIP set.",
        )
        raise error
    y_scaled, datamin, scale = f_scale(dataframe_nonextended['y'], variable)
    # todo mv asserts to a unittest
    np.testing.assert_equal(dataframe_nonextended['y_scaled'].to_numpy(), y_scaled)

    if variable == "pr":
        raise NotImplementedError('extension for precipitation is not implemented yet')
        # tdf["is_dry_day"] = np.isnan(y_scaled)

    return dataframe_extended, datamin, scale


def create_ref_df(df, trace_for_qm, ref_period, params):

    df_params = pd.DataFrame(index=df.index)

    for p in params:
        df_params.loc[:, p] = trace_for_qm[p].mean(axis=0)

    df_params.index = df["ds"]

    df_params_ref = df_params.loc[ref_period[0] : ref_period[1]]
    # mean over all years for each day
    df_params_ref = df_params_ref.groupby(df_params_ref.index.dayofyear).mean()

    # write the average values for the reference period to each day of the
    # whole timeseries
    for day in df_params_ref.index:
        for p in params:
            df_params.loc[
                df_params.index.dayofyear == day, p + "_ref"
            ] = df_params_ref.loc[day, p]

    return df_params


def get_source_timeseries(data_dir, dataset, qualifier, variable, lat, lon):

    input_file = (
        data_dir
        / dataset
        / pathlib.Path(variable + "_" + dataset.lower() + "_" + qualifier + ".nc4")
    )
    obs_data = nc.Dataset(input_file, "r")
    nct = obs_data.variables["time"]
    lats = obs_data.variables["lat"][:]
    lons = obs_data.variables["lon"][:]
    i = np.where(lats == lat)[0][0]
    j = np.where(lons == lon)[0][0]
    data = obs_data.variables[variable][:, i, j]
    tm = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    df = pd.DataFrame(data, index=tm, columns=[variable])
    df.index.name = "Time"
    obs_data.close()
    return df

def get_cell_filename(outdir_for_cell, lat, lon, settings):

    return outdir_for_cell / (
        "ts_" + settings.dataset + "_lat" + str(lat) + "_lon" + str(lon) + settings.storage_format
    )

def test_if_data_valid_exists(fname):

    if ".h5" in str(fname):
        pd.read_hdf(fname)
    elif ".csv" in str(fname):
        pd.read_csv(fname)
    else:
        raise ValueError

def save_to_disk(df_with_cfact, fname, lat, lon, storage_format):

    # outdir_for_cell = make_cell_output_dir(
    #     settings.output_dir, "timeseries", lat, lon, settings.variable
    # )

    # fname = outdir_for_cell / (
    #     "ts_" + settings.dataset + "_lat" + str(lat) + "_lon" + str(lon) + dformat
    # )

    if storage_format == ".csv":
        df_with_cfact.to_csv(fname)
    elif storage_format == ".h5":
        df_with_cfact.to_hdf(fname, "lat_" + str(lat) + "_lon_" + str(lon), mode="w")
    else:
        raise NotImplementedError("choose storage format .h5 or csv.")

    print("Saved timeseries to ", fname)
