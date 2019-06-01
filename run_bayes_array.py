import os
import sys
import numpy as np
import pymc3 as pm
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
# from mpi4py.futures import MPIPoolExecutor
import settings as s
import idetrend as idtr
import idetrend.const as c
import idetrend.bayes_detrending as bt

try:
    submitted = os.environ["SUBMITTED"] == "1"
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    njobarray = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
except KeyError:
    submitted = False
    njobarray = 1
    task_id = 0

gmt_file = os.path.join(s.input_dir, s.gmt_file)
ncg = nc.Dataset(gmt_file,"r")
# gmt = bt.get_gmt_on_each_day(gmt_file, s.days_of_year)
# gmt_scaled = bt.y_norm(gmt, gmt)
gmt = np.squeeze(ncg.variables["tas"][:])
ncg.close()

# get data to detrend
to_detrend_file = os.path.join(s.input_dir, s.source_file)
obs_data = nc.Dataset(to_detrend_file, "r")
nct = obs_data.variables["time"]
latsize = obs_data.dimensions["lat"].size
ncells = latsize*obs_data.dimensions["lon"].size

# combine data to first data table
tdf = bt.create_dataframe(nct, obs_data.variables[s.variable][:, 0, 0],
                         gmt)

if not os.path.exists(s.output_dir):
    os.makedirs(s.output_dir)
    os.makedirs(Path(s.output_dir) / "traces")
    os.makedirs(Path(s.output_dir) / "theano")

if ncells%njobarray:
    print("task_id",task_id)
    print("njobarray",njobarray)
    print("ncells",ncells)
    raise ValueError("ncells does not fit into array job, adjust jobarray.")

calls_per_arrayjob = ncells/njobarray

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
start_num = int(task_id * calls_per_arrayjob)
end_num = int((task_id+1)*calls_per_arrayjob -1)

# Print the task and run range
print("This is SLURM task",task_id,"which will do runs", start_num,"to", end_num)

print("Variable is:")
print(s.variable, flush=True)
# Create bayesian regression model instance
bayes = bt.bayes_regression(tdf["gmt_scaled"], s.output_dir)

TIME0 = datetime.now()
# if submitted:

# Run the loop of runs for this task.
futures = []
for n in np.arange(start_num,end_num+1,1, dtype=np.int):
    i=int(n%latsize)
    j=int(n/latsize)
    print("This is SLURM task",task_id,"run number", n, "i,j", i,j)

    futr = bayes.run(bt.mcs_helper(nct, obs_data, gmt, i, j))
    futures.append(futr)

# else:
#     print("serial mode")
#     futures = map(
#         bayes.run,
#         (
#             bt.mcs_helper(nct, data, gmt, i, j)
#             for i in range(data.dimensions["lat"].size)
#             for j in range(data.dimensions["lon"].size)
#         ),
#     )
#     # necessary to trigger serial map() function.
#     futures = list(futures)

print("Estimation completed for all cells. It took {0:.1f} minutes.".format(
            (datetime.now() - TIME0).total_seconds()/60))


# # Run the loop of runs for this task.
# for n in np.arange(start_num,end_num,1, dtype=np.int):
#     print("This is SLURM task",task_id,"run number", n)
#     i=int(n%latsize)
#     j=int(n/latsize)



  #Do your stuff here



# i=task_id%latsize
# j=int(task_id/latsize)
# print(task_id, i, j)
