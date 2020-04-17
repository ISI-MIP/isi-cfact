import os
import settings as s
import icounter
import icounter.datahandler as dh
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime
from icounter.estimator import model_for_var
import icounter.const as c
import logging


# helper functions
def get_fourier_series(t, n_modes, gmt):
    def tau(t, k): return t * k * 2 * np.pi / 365.25
    fourier_series = [np.ones_like(t), gmt[t]]
    for k in range(1, n_modes + 1):
        fourier_series += [np.sin(tau(t, k)), gmt[t] * np.sin(tau(t, k)), np.cos(tau(t, k)), gmt[t] * np.cos(tau(t, k))]
    return fourier_series


def infer_fourier_coefficients(gmt, y, n_modes):
    """
    infers fourier coefficients of the model by solving a system of linear equations at random indices.
    This is not deterministic
    """
    x = None
    break_index = 0
    while x is None:
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


dh.create_output_dirs(s.output_dir_extended)
logging.basicConfig(
    filename=s.output_dir_extended / "failing_cells.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
# needed to silence verbose pymc3
pmlogger = logging.getLogger("pymc3")
pmlogger.propagate = False

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

# load last value of the extended gmt
with nc.Dataset(s.input_dir / s.dataset_extended / s.gmt_file_extended,
                "r") as ncg:
    last_gmt_value = np.squeeze(ncg.variables["tas"][:])[-1]

# load extended data
input_file_extended = s.input_dir / s.dataset_extended / s.source_file_extended.lower()
landsea_mask_file = s.input_dir / s.landsea_file
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
    fname_fouier_coefficients_cell = outdir_for_cell / f"fourier_coefficients_{s.dataset_extended}_lat{sp['lat']}_lon{sp['lon']}{s.storage_format}"

    if s.skip_if_data_exists:
        try:
            dh.test_if_data_valid_exists(fname_cell)
            dh.test_if_data_valid_exists(fname_fouier_coefficients_cell)
            print(f"Existing valid data in {fname_cell} and {fname_fouier_coefficients_cell} . Skip calculation.")
            continue
        except Exception as e:
            print(e)
            print("No valid data found. Run calculation.")

    data_extended = obs_data_extended.variables[s.variable][:, sp["index_lat"], sp["index_lon"]]
    # load nonextended calculated data
    infile_fname_cell = dh.get_cell_filename(
        s.output_dir / "timeseries" / s.variable / f"lat_{sp['lat']}",
        sp['lat'],
        sp['lon'],
        s
    )

    # todo skip if file does not exist
    # try:
    df_nonextended = pd.read_hdf(infile_fname_cell)
    df_nonextended.set_index(df_nonextended['ds'], inplace=True)
    # except:
    # log an error message and skip

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
                [infer_fourier_coefficients(gmt=df_nonextended['gmt'].to_numpy(),
                                            y=df_nonextended['mu'],
                                            n_modes=s.modes[0])
                 for i in range(1000)]),
            axis=0)
        fourier_coeff_matrix = fourier_coefficients_mu[:, np.newaxis].repeat(len(df_extended['gmt']), 1)
        df_extended['mu'] = (
                fourier_coeff_matrix *
                np.array(get_fourier_series(t=np.arange(len(df_extended['gmt']), dtype=np.int),
                                            n_modes=s.modes[0],
                                            gmt=df_extended['gmt'].to_numpy()))
        ).sum(axis=0)
        # todo find a better cutoff, here we can accept some difference
        np.testing.assert_allclose(df_extended.iloc[:len(df_nonextended)]['mu'],
                                   df_nonextended['mu'], rtol=1e-5)
        # keep old mu value for the not extended time period
        df_extended.loc[:df_nonextended.index[-1], 'mu'] = df_nonextended['mu']
        df_extended['sigma'] = df_nonextended['sigma'].mean()
        np.testing.assert_allclose(
            df_extended.loc[:df_nonextended.index[-1], 'sigma'],
            df_nonextended['sigma']
        )
    # elif s.variable == 'wind':
    #     # use logit function (inverse of logistic function) first
    # todo assert extension of parameters remain valid (only for wind and pr)
    else:
        raise NotImplementedError(f'infering parameters from a trained model is not implemented for {s.variable}')


    # estimate timeseries
    ####################
    # create reference dataframe
    statmodel = model_for_var[s.variable](s.modes)
    df_params = df_extended.loc[:, [p for p in statmodel.params]]
    df_params_ref = df_params.loc[s.qm_ref_period[0]: s.qm_ref_period[1]]
    # mean over all years for each day
    df_params_ref = df_params_ref.groupby(df_params_ref.index.dayofyear).mean()

    # write the average values for the reference period to each day of the
    # whole timeseries
    for day in df_params_ref.index:
        for p in statmodel.params:
            df_params.loc[
                df_params.index.dayofyear == day, p + "_ref"
            ] = df_params_ref.loc[day, p]

    cfact_scaled = statmodel.quantile_mapping(df_params, df_extended["y_scaled"])
    print("Done with quantile mapping.")
    # fill cfact_scaled as is from quantile mapping
    # for easy checking later
    df_extended.loc[:, "cfact_scaled"] = cfact_scaled
    np.testing.assert_allclose(
        df_extended.loc[:df_nonextended.index[-1], 'cfact_scaled'],
        df_nonextended['cfact_scaled']
    )
    # rescale all scaled values back to original, invalids included
    df_extended.loc[:, "cfact"] = c.mask_and_scale[s.variable][1](df_extended.loc[:, "cfact_scaled"], datamin, scale)
    # populate invalid values originating from y_scaled with with original values
    invalid_index = df_extended.index[df_extended["y_scaled"].isna()]
    df_extended.loc[invalid_index, "cfact"] = df_extended.loc[invalid_index, "y"]

    # df = df.replace([np.inf, -np.inf], np.nan)
    # if df["y"].isna().sum() > 0:
    yna = df_extended["cfact"].isna()
    yinf = df_extended["cfact"] == np.inf
    yminf = df_extended["cfact"] == -np.inf
    print(f"There are {yna.sum()} NaN values from quantile mapping. Replace.")
    print(f"There are {yinf.sum()} Inf values from quantile mapping. Replace.")
    print(f"There are {yminf.sum()} -Inf values from quantile mapping. Replace.")

    df_extended.loc[yna | yinf | yminf, "cfact"] = df_extended.loc[yna | yinf | yminf, "y"]
    np.testing.assert_allclose(
        df_extended.loc[:df_nonextended.index[-1], 'cfact'],
        df_nonextended['cfact']
    )
    # todo: unifiy indexes so .values can be dropped
    for v in df_params.columns:
        df_extended.loc[:, v] = df_params.loc[:, v].values

    if s.report_variables != "all":
        df_extended = df_extended.loc[:, s.report_variables]
    dh.save_to_disk(df_extended, fname_cell, sp["lat"], sp["lon"], s.storage_format)
    dh.save_fourier_coefficients_to_disk(
        pd.DataFrame({'fc_mu': fourier_coefficients_mu}),
        fname_fouier_coefficients_cell,
        sp["lat"],
        sp["lon"],
        s.storage_format
    )
obs_data_extended.close()
nc_lsmask.close()

print("Estimation completed for all cells." +
      "It took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60)
      )

