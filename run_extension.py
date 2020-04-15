import os
import settings as s
import icounter
import icounter.estimator as est
import icounter.datahandler as dh
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime


# helper functions
def get_fourier_series(t, nmodes, gmt):
    tau = lambda t, k: t * k * 2 * np.pi / 365.25
    fourier_series = [np.ones_like(t), gmt[t]]
    for k in range(1, nmodes+1):
        fourier_series += [np.sin(tau(t, k)), gmt[t] * np.sin(tau(t, k)), np.cos(tau(t, k)), gmt[t] * np.cos(tau(t, k))]
    return fourier_series

def infer_model_params(gmt, t, y, n_modes):
    """
    infers model parameters by solving a system of linear equations at random indices.
    This is not deterministic
    """
    x = None
    break_index = 0
    while (x is None):
        n = 2 + n_modes * 4
        time_samples = np.random.randint(low=0, high=len(gmt), size=n)
        a = np.empty((n, n))
        for i, t in enumerate(time_samples):
            a[i, :] = get_fourier_series(t, n_modes, gmt)
            b = np.array([y[t] for t in time_samples])
            try:
                x = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                break_index += 1
                if break_index == 100:
                    raise TimeoutError('more than 100 linalg errors')
    return x


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
with nc.Dataset(s.input_dir / s.dataset_extended / s.gmt_file_extended,
                "r") as ncg:
    last_gmt_value = np.squeeze(ncg.variables["tas"][:])[-1]

input_file_extended = s.input_dir / s.dataset_extended / s.source_file_extended.lower()
landsea_mask_file = s.input_dir / s.landsea_file

# load extended data
obs_data_extended = nc.Dataset(input_file_extended, "r")
nc_lsmask = nc.Dataset(landsea_mask_file, "r")
nct = obs_data_extended.variables["time"]
lats = obs_data_extended.variables["lat"][:]
lons = obs_data_extended.variables["lon"][:]

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

    data_extended = obs_data_extended.variables[s.variable][:, sp["index_lat"], sp["index_lon"]]
    # load nonextended calculated data
    indir_for_cell = s.output_dir / "timeseries" / s.variable / f"lat_{sp['lat']}"
    infile_fname_cell = dh.get_cell_filename(indir_for_cell,
                                             sp['lat'],
                                             sp['lon'],
                                             s)
    # try:
    df_nonextended = pd.read_hdf(infile_fname_cell)
    df_nonextended.set_index(df_nonextended['ds'], inplace=True)
    # except:
    # log an error message and skip
    # todo skip if file does not exist

    # todo implement create_dataframe_extended
    df_extended, datamin, scale = dh.create_dataframe_extended(
        nct_array=nct[:],
        units=nct.units,
        dataframe_nonextended=df_nonextended,
        data_extended=data_extended,
        last_gmt_value=last_gmt_value,
        variable=s.variable)
    # case distinction for variables:
    if s.variable in ['tas',
                      'tasrange',
                      'tasskew',
                      'hurs',
                      'ps',
                      'rsds',
                      'rlds']:
        fourier_coefficients_mu = np.median(
            np.array(
                [infer_model_params(gmt=df_nonextended['gmt'].to_numpy(),
                                    t=np.arange(len(df_nonextended['gmt']), dtype=np.int),
                                    y=df_nonextended['mu'],
                                    n_modes=s.modes[0])
                 for i in range(1000)]),
            axis=0)
        fourier_coeff_matrix = fourier_coefficients_mu[:, np.newaxis].repeat(len(df_extended['gmt']), 1)
        df_extended['mu'] = (
                fourier_coeff_matrix *
                np.array(get_fourier_series(t=np.arange(len(df_extended['gmt']), dtype=np.int),
                                            nmodes=s.modes[0],
                                            gmt=df_extended['gmt'].to_numpy()))
        ).sum(axis=0)
        # todo find a better cutoff
        np.testing.assert_allclose(df_extended.iloc[:len(df_nonextended)]['mu'],
                                   df_nonextended['mu'])
        # keep old mu value for the nonextended timeperiod
        df_extended.iloc[:len(df_nonextended)]['mu'] = df_nonextended['mu']
        foo = 'baa'
    # elif s.variable == 'wind':
    #     # use logit function (inverse of logistic function) first
    #     pass
    else:
        raise NotImplementedError(f'infering parameters from a trained model is not implemented for {s.variable}')

    foo = 'baa'

    # todo assert extension of parameters remain valid (only for wind and pr)
    # todo implement rescaling based on the nonextended data
    # todo get timeseries
    # assert timeseries for old data is almost similar
    # save data
obs_data_extended.close()
nc_lsmask.close()

print("Estimation completed for all cells." +
      "It took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60)
      )

