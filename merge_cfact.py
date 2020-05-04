#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools as it
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
import attrici.postprocess as pp
import settings as s

# options for postprocess
rechunk = False
# cdo_processing needs rechunk
cdo_processing = False

TIME0 = datetime.now()

ts_dir = s.output_dir / "timeseries" / s.variable
cfact_dir = s.output_dir / "cfact" / s.variable

data_gen = ts_dir.glob("**/*" + s.storage_format)
cfact_dir.mkdir(parents=True, exist_ok=True)
cfact_file = cfact_dir / s.cfact_file

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
lat_indices = np.array(lat_indices) / s.lateral_sub
lon_indices = np.array(lon_indices) / s.lateral_sub

### access data from source file
obs = nc.Dataset(Path(s.input_dir) / s.dataset / s.source_file.lower(), "r")
time = obs.variables["time"][:]
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]

# append later with more variables if needed
variables_to_report = {s.variable: "cfact", s.variable + "_orig": "y"}

#  get headers and form empty netCDF file with all meatdata
headers = pp.read_from_disk(data_list[0]).keys()
print(data_list[0])
headers = headers.drop(["t", "ds", "gmt", "gmt_scaled"])

out = nc.Dataset(cfact_file, "w", format="NETCDF4")
pp.form_global_nc(out, time, lat, lon, headers, obs.variables["time"].units)


for (i, j, dfpath) in it.zip_longest(lat_indices, lon_indices, data_list):

    df = pp.read_from_disk(dfpath)
    for head in headers:
        ts = df[head]
        out.variables[head][:, i, j] = np.array(ts)
    print("wrote data from", dfpath, "to", i, j)

out.close()
print("Successfully wrote", cfact_file, "file. Took")
print("It took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60))

if rechunk:
    cfact_rechunked = pp.rechunk_netcdf(cfact_file)

if cdo_processing:
    cdo_ops = {
        "monmean": "monmean -selvar,cfact,y",
        "yearmean": "yearmean -selvar,cfact,y",
        #    "monmean_valid": "monmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
        #    "yearmean_valid": "yearmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
        "trend": "trend -selvar,cfact,y",
        #    "trend_valid": "trend -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
    }

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
