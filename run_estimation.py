import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import attrici
import attrici.estimator as est
import attrici.datahandler as dh
import settings as s
from pymc3.parallel_sampling import ParallelSamplingError
import logging

s.output_dir.mkdir(parents=True,exist_ok=True)
logging.basicConfig(
    filename=s.output_dir / "failing_cells.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
# needed to silence verbose pymc3
pmlogger = logging.getLogger("pymc3")
pmlogger.propagate = False

print("Version", attrici.__version__)

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

dh.create_output_dirs(s.output_dir)

gmt_file = s.input_dir / s.dataset / s.gmt_file
ncg = nc.Dataset(gmt_file, "r")
gmt = np.squeeze(ncg.variables["tas"][:])
ncg.close()

input_file = s.input_dir / s.dataset / s.source_file.lower()
landsea_mask_file = s.input_dir / s.landsea_file

obs_data = nc.Dataset(input_file, "r")
nc_lsmask = nc.Dataset(landsea_mask_file, "r")
nct = obs_data.variables["time"]
lats = obs_data.variables["lat"][:]
lons = obs_data.variables["lon"][:]
longrid, latgrid = np.meshgrid(lons, lats)
jgrid, igrid = np.meshgrid(np.arange(len(lons)), np.arange(len(lats)))

ls_mask = nc_lsmask.variables["LSM"][0, :]
df_specs = pd.DataFrame()
df_specs["lat"] = latgrid[ls_mask == 1]
df_specs["lon"] = longrid[ls_mask == 1]
df_specs["index_lat"] = igrid[ls_mask == 1]
df_specs["index_lon"] = jgrid[ls_mask == 1]

print("A total of", len(df_specs), "grid cells to estimate.")

if len(df_specs) % (njobarray) == 0:
    print("Grid cells can be equally distributed to Slurm tasks")
    calls_per_arrayjob = np.ones(njobarray) * len(df_specs) // (njobarray)
else:
    print("Slurm tasks not a divisor of number of grid cells, discard some cores.")
    calls_per_arrayjob = np.ones(njobarray) * len(df_specs) // (njobarray) + 1
    discarded_jobs = np.where(np.cumsum(calls_per_arrayjob) > len(df_specs))
    calls_per_arrayjob[discarded_jobs] = 0
    calls_per_arrayjob[discarded_jobs[0][0]] = len(df_specs) - calls_per_arrayjob.sum()

assert calls_per_arrayjob.sum() == len(df_specs)
# print(calls_per_arrayjob)

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
cum_calls_per_arrayjob = calls_per_arrayjob.cumsum(dtype=int)
start_num = 0 if task_id == 0 else cum_calls_per_arrayjob[task_id-1]
end_num = cum_calls_per_arrayjob[task_id] - 1
run_numbers = np.arange(start_num, end_num + 1, 1, dtype=np.int)
if len(run_numbers) == 0:
    print ("No runs assigned for this SLURM task.")
else:
    print("This is SLURM task", task_id, "which will do runs", start_num, "to", end_num)

estimator = est.estimator(s)

TIME0 = datetime.now()

for n in run_numbers[:]:
    sp = df_specs.loc[n, :]

    # if lat >20: continue
    print(
        "This is SLURM task", task_id, "run number", n, "lat,lon", sp["lat"], sp["lon"]
    )
    outdir_for_cell = dh.make_cell_output_dir(
        s.output_dir, "timeseries", sp["lat"], sp["lon"], s.variable
    )
    fname_cell = dh.get_cell_filename(outdir_for_cell, sp["lat"], sp["lon"], s)

    if s.skip_if_data_exists:
        try:
            dh.test_if_data_valid_exists(fname_cell)
            print(f"Existing valid data in {fname_cell} . Skip calculation.")
            continue
        except Exception as e:
            print(e)
            print("No valid data found. Run calculation.")

    data = obs_data.variables[s.variable][:, sp["index_lat"], sp["index_lon"]]
    df, datamin, scale = dh.create_dataframe(nct[:], nct.units, data, gmt, s.variable)

    try:
        trace, dff = func_timeout(
            s.timeout, estimator.estimate_parameters, args=(df, sp["lat"], sp["lon"])
        )
    except (FunctionTimedOut, ParallelSamplingError, ValueError) as error:
        if str(error) == "Modes larger 1 are not allowed for the censored model.":
            raise error
        else:
            print("Sampling at", sp["lat"], sp["lon"], " timed out or failed.")
            print(error)
            logger.error(
                str(
                    "lat,lon: "
                    + str(sp["lat"])
                    + " "
                    + str(sp["lon"])
                    + " : "
                    + str(error)
                )
            )
        continue

    df_with_cfact = estimator.estimate_timeseries(dff, trace, datamin, scale)
    dh.save_to_disk(df_with_cfact, fname_cell, sp["lat"], sp["lon"], s.storage_format)

obs_data.close()
nc_lsmask.close()
print(
    "Estimation completed for all cells. It took {0:.1f} minutes.".format(
        (datetime.now() - TIME0).total_seconds() / 60
    )
)
