# ATTRICI - counterfactual climate for impact attribution

Code implementing the methods of as discussed in Mengel et al. (submitted) [insert link to preprint].

## Summary

Climate has changed over the past century due to anthropogenic greenhouse gas emissions. In parallel, societies and their environment have evolved rapidly. To identify the impacts of climate change on human or natural systems, it is therefore necessary to separate the effect of different drivers. By definition this is done by comparing the observed situation to a counterfactual one in which climate change is absent and other drivers change according to observations. As this counterfactual baseline cannot be observed it has to be estimated by process-based or empirical models. We here present methods to remove the signal of climate change from observational climate data to generate “no-climate change” climate forcing data that can be used to simulate the counterfactual baseline of impact indicators. Our method identifies the interannual and yearly-cycle shifts that are correlated to global mean temperature change. We use quantile mapping to a reference distribution without the global-mean-temperature related shifts to find the counterfactual value for the observed daily climate data. Applied to each variable in the GSWP3 and GSWP3-W5E5 climate datasets, we produce two counterfactual datasets that are made available through ISIMIP along with the original datasets. Our method preserves the internal variability of the observed data in the sense that the applied transfer functions are monotonous. That makes it possible to compare observed impact events and counterfactual impact events in a world that would have been without climate change. Our approach captures the long-term trends associated with global warming but does not address the attribution of climate change to anthropogenic greenhouse gas emissions.

## Approach

Preserving the events of the historical record is a key objective of the climate counterfactual presented here. In summary, we construct the counterfactual by removing shifts in the historical dataset that can be linked to global mean temperature change.

Our method relies on quantile mapping and thus on two statistical distributions: a distribution *A* that captures the evolution of the statistics of a climate variable due to climate change and a reference distribution *B* that approximates such evolution in the absence of climate change. Both distributions are dependent on time. Distribution *A* varies with a long-term global mean temperature trend, the yearly cycle, and a global-mean-temperature related distortion of the yearly cycle. The reference distribution *B* varies with the yearly cycle only. The type of the distribution depends on the climate variable and is the same for *A* and *B*.



### Variables

We model the different climatic variables using the statistical distribution listed in Table 1.


| Variable | Short name | Unit | Statistical model |
| -------- | ---------- | ---- | ----------------- |
| Near-Surface Air Temperature | tas | K | Gaussian |
| Range of daily temperature | tasrange | K | Gaussian |
| Skewness of daily temperature | tasskew | 1 | Gaussian |
| Daily Minimum Near-Surface Air Temperature | tasmin | K | Derived from tas, tasrange and tasskew |
| Daily Maximum Near-Surface Air Temperature | tasmax | K | Derived from tas, tasrange and tasskew |
| Precipitation | pr | kg / m² s | Bernoulli-Gamma |
| Surface Downwelling Longwave Radiation | rlds | W / m² | Gaussian |
| Surface Downwelling Shortwave Radiation | rsds | W / m²| Gaussian |
| Surface Air Pressure | ps | Pa | Gaussian |
| Near-Surface Wind Speed | sfcWind | m / s | Weibull |
| Near-Surface Relative Humidity | hurs | % | Gaussian |
| Near-Surface Specific Humidity | huss | kg / kg | Derived from hurs ps and tas |

*Table 1: Specs of climate variables for the ISIMIP3b counterfactual climate datasets. The variables tasrange and tasskew are auxiliary variables to calculate tasmin and tasmax*

For tasmin and tasmax, we do not estimate counterfactual time series individually to avoid large relative errors in the daily temperature range as pointed out by (Piani et al. 2010). Following (Piani et al. 2010), we estimate counterfactuals of the daily temperature range tasrange = tasmax - tasmin and the skewness of the daily temperature tasskew = (tas - tasmin) / tasrange.

A counterfactual huss is derived from the counterfacual tas, ps and hurs using the equations of Buck (1981) as described in Weedon et al. (2010).
huss = f(tas, pr ,hurs)


## Usage

This code is currently taylored to run on the supercomputer at the Potsdam Institute for Climate Impact Research. Generalizing it into a package is ongoing work. We use the GNU compiler as the many parallel compile jobs through jobarrays and JIT compilation conflict with the few Intel licenses.

`module purge`

`module load compiler/gnu/7.3.0`

`conda activate yourenv`

Override the conda setting with: `export CXX=g++`

Adjust `settings.py`

For estimating parameter distributions (above step 1) and smaller datasets

`python run_estimation.py`

For larger datasets, produce a `submit.sh` file via

`python create_submit.py`

Then submit to the slurm scheduler

`sbatch submit.sh`

For merging the single timeseries files to netcdf datasets

`python merge_cfact.py`

### Running multiple instances at once


`conda activate yourenv`

In the root package directory.

`pip install -e .`

Copy the `settings.py`, `run_estimation.py`, `merge_cfact.py` and `submit.sh` to a separate directory,
for example `myrunscripts`. Adjust `settings.py` and `submit.sh`, in particular the output directoy, and continue as in Usage.

## Install

We use the jobarray feature of slurm to run many jobs in parallel.
The configuration is very much tailored to the PIK supercomputer at the moment. Please do

`conda config --add channels conda-forge`

`conda create -n isi-cfact pymc3==3.7 python==3.7`

`conda activate isi-cfact`

`conda install netCDF4 pytables matplotlib arviz`

`pip install func_timeout`

You may optionally
`cp config/theanorc ~/.theanorc`


## Credits

We rely on the [pymc3](https://github.com/pymc-devs/pymc3) package for probabilistic programming (Salvatier et al. 2016).

The code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 is inspired and adopted from [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.


## License

This code is licensed under GPLv3, see the LICENSE.txt. See commit history for authors.

## References
- Buck, A.L.:New Equations for Computing Vapor Pressure and Enhancement Factor, J. Appl. Meteorol., 20, 1527–1532, 1981.
- Piani, C., Weedon, G. P., Best, M., Gomes, S. M., Viterbo, P.,
Hagemann, S., and Haerter, J. O.: Statistical bias correction
of global simulated daily precipitation and temperature for the
application of hydrological models, J. Hydrol., 395, 199–215,
https://doi.org/10.1016/j.jhydrol.2010.10.024, 2010.
- Weedon, G. P., Gomes, S., Viterbo, P., Österle, H., Adam, J. C., Bellouin, N., Boucher, O., and Best, M.: The WATCH forcing data 1958–2001: A meteorological forcing dataset for land surface and hydrological models, in: Technical Report no 22., available at: http://www.eu-watch.org/publications/technical-reports (last access: July 2016), 2010.
- Salvatier J., Wiecki T.V., Fonnesbeck C. (2016) Probabilistic programming in Python using PyMC3. PeerJ Computer Science 2:e55 DOI: 10.7717/peerj-cs.55.

