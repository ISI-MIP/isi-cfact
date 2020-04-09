import os
import settings as s
import icounter
import icounter.estimator as est
import icounter.datahandler as dh
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime

# todo logging

print("Version", icounter.__version__)

try:
    submitted = os.environ["SUBMITTED"] == "1"
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    njobarray = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    s.ncores_per_job = 1
    s.progressbar = False
except KeyError:
    submitted = False
    njobarray = 1
    task_id = 0
    s.progressbar = True

dh.create_output_dirs(s.output_dir_extended)

# load files
gmt_file = s.input_dir / s.dataset / s.gmt_file
with nc.Dataset(gmt_file, "r") as ncg:
    gmt = np.squeeze(ncg.variables["tas"][:])

with nc.Dataset(s.input_dir / s.dataset_extended / s.gmt_file_extended,
                "r") as ncg:
    gmt_extended = np.squeeze(ncg.variables["tas"][:])

input_file_extended = s.input_dir / s.dataset_extended / s.source_file_extended.lower()
landsea_mask_file = s.input_dir / s.landsea_file

# load extended data
obs_data = nc.Dataset(input_file_extended, "r")
nc_lsmask = nc.Dataset(landsea_mask_file, "r")
nct = obs_data.variables["time"]
lats = obs_data.variables["lat"][:]
lons = obs_data.variables["lon"][:]

longrid, latgrid = np.meshgrid(lons, lats)
jgrid, igrid = np.meshgrid(np.arange(len(lons)),
                           np.arange(len(lats)))

ls_mask = nc_lsmask.variables["LSM"][0, :]
df_specs = pd.DataFrame()
df_specs["lat"] = latgrid[ls_mask == 1]
df_specs["lon"] = longrid[ls_mask == 1]
df_specs["index_lat"] = igrid[ls_mask == 1]
df_specs["index_lon"] = jgrid[ls_mask == 1]

print("A total of", len(df_specs), "grid cells to estimate.")

# setup the job distribution
if len(df_specs) % njobarray == 0:
    print("Grid cells can be equally distributed to Slurm tasks")
    calls_per_arrayjob = np.ones(njobarray) * len(df_specs) // (njobarray)
else:
    print("Slurm tasks not a divisor of number of grid cells, discard some cores.")
    calls_per_arrayjob = np.ones(njobarray) * len(df_specs) // (njobarray) + 1
    discarded_jobs = np.where(np.cumsum(calls_per_arrayjob) > len(df_specs))
    calls_per_arrayjob[discarded_jobs] = 0
    calls_per_arrayjob[discarded_jobs[0][0]] = len(df_specs) - calls_per_arrayjob.sum()

assert calls_per_arrayjob.sum() == len(df_specs)

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
cum_calls_per_arrayjob = calls_per_arrayjob.cumsum(dtype=int)
start_num = 0 if task_id == 0 else cum_calls_per_arrayjob[task_id - 1]
end_num = cum_calls_per_arrayjob[task_id] - 1
run_numbers = np.arange(start_num, end_num + 1, 1, dtype=np.int)
if len(run_numbers) == 0:
    print("No runs assigned for this SLURM task.")
else:
    print("This is SLURM task", task_id, "which will do runs", start_num, "to", end_num)

# todo implement ext.extend(s)
# extend = ext.extend(s)

TIME0 = datetime.now()
for n in run_numbers[:]:
    sp = df_specs.loc[n, :]
    print(
        f'This is SLURM task {task_id} run number {n} lat,lon {sp["lat"]}, {sp["lon"]}'
    )
    outdir_for_cell = dh.make_cell_output_dir(
        s.output_dir_extended, "timeseries", sp["lat"], sp["lon"], s.variable
    )
    fname_cell = outdir_for_cell / f"ts_{s.dataset_extended}_lat{sp['lat']}_lon{sp['lon']}{s.storage_format}"

    if s.skip_if_data_exists:
        try:
            dh.test_if_data_valid_exists(fname_cell)
            print(f"Existing valid data in {fname_cell} . Skip calculation.")
            continue
        except Exception as e:
            print(e)
            print("No valid data found. Run calculation.")

    data = obs_data.variables[s.variable][:, sp["index_lat"], sp["index_lon"]]
    # load nonextended calculated data
    indir_for_cell = s.output_dir / "timeseries" / s.variable / f"lat_{sp['lat']}"
    infile_fname_cell = dh.get_cell_filename(indir_for_cell,
                                             sp['lat'],
                                             sp['lon'],
                                             s)
    # try:
    df_nonextended = pd.read_hdf(infile_fname_cell)
    # except:
    # log an error message and skip
    # todo skip if file does not exist

    # todo implement create_dataframe_extended
    df, datamin, scale = dh.create_dataframe_extended(nct[:], nct.units, data, gmt, s.variable)
    # todo calculat parameters for old dataset
    # todo assert extension of parameters remain valid (only for wind and pr)
    # todo get timeseries
    # assert timeseries for old data is almost similar
    # save data
obs_data.close()
nc_lsmask.close()

print("Estimation completed for all cells." +
      "It took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60)
      )


