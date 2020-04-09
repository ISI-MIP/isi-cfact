import getpass
from pathlib import Path

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    conda_path = "/home/mengel/anaconda3/envs/pymc3/"
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
    # data_dir = "/p/tmp/mengel/isimip/isi-cfact"
    log_dir = "./log"

elif user == "sitreu":
    # conda_path = "/home/sitreu/.conda/envs/mpi_py3"
    data_dir = "/home/sitreu/Documents/PIK/CounterFactuals/isi-cfact/"
    # data_dir = "/p/tmp/sitreu/isimip/isi-cfact"
    log_dir = "./log"

input_dir = Path(data_dir) / "input"
# make output dir same as cwd. Helps if running more than one job.
output_dir = Path(data_dir) / "output" / Path.cwd().name
output_dir_extended = Path(str(output_dir) + "-EXTENDED")

# max time in sec for sampler for a single grid cell.
timeout = 240 * 60
# tas, tasrange pr, prsn, prsnratio, ps, rlds, wind, hurs
variable = "tas"  # select variable to detrend

# number of modes for fourier series of model, only relevant if mu or sigma model
# include yearly cycles
modes = [1, 1, 1, 1]
# NUTS or ADVI
inference = "NUTS"

seed = 0  # for deterministic randomisation
subset = 1  # only use every subset datapoint for bayes estimation for speedup

# out of "watch+wfdei", "GSWP3", "GSWP3+ERA5"
dataset = "GSWP3"
dataset_extended = f"{dataset}-EXTENDED"
# use a dataset with only subset spatial grid points for testing
lateral_sub = 80

# start and end date are the time period used to construct
# the reference distribution for quantile mapping.
# take care that period encompasses a leap year
qm_ref_period = ["1901-01-01", "1904-12-31"]

gmt_file = dataset.lower() + "_ssa_gmt.nc4"
gmt_file_extended= dataset_extended.lower() + "_ssa_gmt.nc4"
landsea_file = "ISIMIP2b_landseamask_generic_sub" + str(lateral_sub) + ".nc4"
# source_file = variable + "_" + dataset + "_sub.nc4"
source_file = variable + "_" + dataset + "_sub" + str(lateral_sub) + ".nc4"
source_file_extended = variable + "_" + dataset_extended + "_sub" + str(lateral_sub) + ".nc4"
cfact_file = variable + "_" + dataset + "_cfactual.nc4"
# .h5 or .csv
storage_format = ".h5"
# "all" or list like ["y","y_scaled","mu","sigma"]
# for productions runs, use ["cfact"]
report_variables = "all"
# reporting to netcdf can include all report variables
# "cfact" is translated to variable, and "y" to variable_orig
report_to_netcdf = [variable, variable + "_orig"]

save_trace = True
skip_if_data_exists = True

# model run settings
tune = 500  # number of draws to tune model
draws = 1000  # number of sampling draws per chain
chains = 2  # number of chains to calculate (min 2 to check for convergence)

# number of cores to use for one gridpoint
# submitted jobs will have ncores_per_job=1 always.
ncores_per_job = 2
progressbar = True  # print progress in output (.err file for mpi)

#### settings for create_submit.py
# number of parallel jobs through jobarray
# needs to be divisor of number of grid cells
njobarray = 64
