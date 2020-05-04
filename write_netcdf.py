#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools
from pathlib import Path

import netCDF4 as nc
import xarray as xr

import numpy as np

# import pandas as pd
from datetime import datetime
import subprocess
import attrici
import attrici.postprocess as pp
import settings as s

### options for postprocess
write_netcdf = False
rechunk = False
replace_invalid=True
# cdo_processing needs rechunk
cdo_processing = True

# append later with more variables if needed
vardict = {s.variable: "cfact", s.variable + "_orig": "y"}

cdo_ops = {
    "monmean": "monmean ",
    "yearmean": "yearmean ",
    #    "monmean_valid": "monmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
    #    "yearmean_valid": "yearmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
    "trend": "trend ",
    #    "trend_valid": "trend -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
}


TIME0 = datetime.now()

source_file = Path(s.input_dir) / s.dataset / s.source_file.lower()
ts_dir = s.output_dir / "timeseries" / s.variable
cfact_dir = s.output_dir / "cfact" / s.variable
cfact_file = cfact_dir / s.cfact_file
cfact_rechunked = str(cfact_file).rstrip(".nc4") + "_rechunked.nc4"

if write_netcdf:

    data_gen = ts_dir.glob("**/*" + s.storage_format)
    cfact_dir.mkdir(parents=True, exist_ok=True)

    ### check which data is available
    data_list = []
    lat_indices = []
    lon_indices = []
    for i in data_gen:
        data_list.append(str(i))
        lat_float = float(str(i).split("lat")[-1].split("_")[0])
        lon_float = float(str(i).split("lon")[-1].split(s.storage_format)[0])
        lat_indices.append(int(180 - 2 * lat_float - 0.5))
        lon_indices.append(int(2 * lon_float - 0.5 + 360))

    # adjust indices if datasets are subsets (lat/lon-shapes are smaller than 360/720)
    # TODO: make this more robust
    lat_indices = np.array(np.array(lat_indices) / s.lateral_sub, dtype=int)
    lon_indices = np.array(np.array(lon_indices) / s.lateral_sub, dtype=int)

    #  get headers and form empty netCDF file with all meatdata
    print(data_list[0])

    # write empty outfile to netcdf with all orignal attributes
    source_data = xr.open_dataset(source_file)
    attributes = source_data[s.variable].attrs
    coords = source_data[s.variable].coords

    outfile = source_data.drop_vars(s.variable)

    outfile.to_netcdf(cfact_file)

    # open with netCDF4 for memory efficient writing
    outfile = nc.Dataset(cfact_file, "a")

    for var in s.report_to_netcdf:
        ncvar = outfile.createVariable(
            var,
            "f4",
            ("time", "lat", "lon"),
            chunksizes=(len(coords["time"]), 1, 1),
            fill_value=9.9692e36,
        )
        if var in [s.variable, s.variable + "_orig"]:
            for key, att in attributes.items():
                ncvar.setncattr(key, att)


    outfile.setncattr("cfact_version", attrici.__version__)
    outfile.setncattr("runid", Path.cwd().name)

    for (i, j, dfpath) in itertools.zip_longest(lat_indices, lon_indices, data_list):

        df = pp.read_from_disk(dfpath)
        for var in s.report_to_netcdf:
            ts = df[vardict[var]]
            outfile.variables[var][:, i, j] = np.array(ts)
        print("wrote data from", dfpath, "to", i, j)

    outfile.close()

    print("Successfully wrote", cfact_file, "file. Took")
    print(
        "Writing took {0:.1f} minutes.".format(
            (datetime.now() - TIME0).total_seconds() / 60
        )
)

if rechunk:
    cfact_rechunked = pp.rechunk_netcdf(cfact_file, cfact_rechunked)

if replace_invalid:
    cfact_rechunked = pp.replace_nan_inf_with_orig(s.variable, source_file, cfact_rechunked)


if cdo_processing:

    for cdo_op in cdo_ops:

        outfile = str(cfact_file).rstrip(".nc4") + "_" + cdo_op + ".nc4"
        if "trend" in cdo_op:
            outfile = (
                outfile.rstrip(".nc4") + "_1.nc4 " + outfile.rstrip(".nc4") + "_2.nc4"
            )
        try:
            cmd = "cdo " + cdo_ops[cdo_op] + " " + cfact_rechunked + " " + outfile
            print(cmd)
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            cmd = "module load cdo && " + cmd
            print(cmd)
            subprocess.check_call(cmd, shell=True)
