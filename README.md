# ISI-CFACT

ISI-CFACT produces counterfactual climate data from past datasets for the ISIMIP project.

## Idea
Counterfactual climate is a hypothetical climate in a world without climate change.
For impact models, such climate should stay as close as possible to the observed past,
as we aim to compare impact events of the past (for which we have data) to the events in the counterfactual. The difference between past impacts and counterfactual impacts is a proxy for the impacts caused by climate change. We run the following steps:

1. We approximate the change in past climate through a model with three parts. Long-term trend, an ever-repeating yearly cycle, and a trend in the yearly cycle. Trends are induced by global mean temperature change. We use a Bayesian approach to estimate all parameters of the model and their dependencies at once, here implemented through pymc3. Yearly cycle and trend in yearly cycles are approximated through a finite number of modes, which are periodic in the year. The parameter distributions tell us which part of changes in the variables can be explained through global mean temperature as a direct driver.

2. We do quantile mapping to map each value from the observed dataset to a value that we expect it would have been without the climate-induced trend. Our hierarchical model approach provides us with a time evolution of our distribution through the time evolution of a gmt-dependent parameter.
We first use this time-evolving distribution to map each value to its quantile in this time evolving distribution.
We then use the distribution from a reference period in the beginning of our dataset where we assume that climate change did not play a role, to remap the quantile to a value of the variable. This value is our counterfactual value. Quantile mapping is different for each day of the year because our model is sensitive to the yearly cycle and the trend in the yearly cycle. This approach is illustrated in _figure 1_

The following graph illustrates the approach. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.

![Counterfactual example](image01.png)
*Figure 1: Model for a climate variable with yearly cycle. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.*

### Variables

| Variable | Short name | Unit | Statistical model |
| -------- | ---------- | ---- | ----------------- |
| Near-Surface Air Temperature | tas | K | Gaussian |
| Auxiliary Variable for Tasmin and Tasmax| tasrange | | Bounded Gaussian |
| Auxiliary Variable for Tasmin and Tasmax| tasskew | | Bounded Gaussian |
| Daily Minimum Near-Surface Air Temperature | tasmin | K | f(tasrange, tasskew) |
| Daily Maximum Near-Surface Air Temperature | tasmax | K | f(tasrange, tasskew) |
| Surface Downwelling Longwave Radiation | rlds | W / m² | Gaussian |
| Surface Downwelling Shortwave Radiation | rsds | W / m²| Censored Gaussian |
| Surface Air Pressure | ps | Pa | Gaussian |
| Near-Surface Wind Speed | sfcWind | m / s | Weibull |
| Precipitation | pr | kg / m² s | Bernoulli-Gamma |
| Near-Surface Relative Humidity | hurs | % | Censored Gaussian |
| Near-Surface Specific Humidity | huss | kg / kg | f(hurs, pr, tas) |

## Model

A global mean temperature timeseries 
without yearly variations is used as predictor for the model.
Tho generate this predictor, global mean temperature is preprocessed using singular spectrum analysis. 

The variables tas, rlds, and ps are modeled with a Gaussian distribution with a time varying mean value. The mean value is a linear function of the global mean temperature change plus a yearly cycle. The fourier coefficients of the yearly cycle are also a linear function of the global mean temperature.

Tasrange and Tasskew are also modeled with a Gaussian distribution as described above. But those variables are bounded. Tasrange is positive and tasskew between 0 and 1. In the quantile mapping step, values that are close to the boundary can get mapped to values outside the defined range. To avoid this, such values are not quantile mapped s.th. the counterfactual value is the same as the historic value in those cases. This happens only rarely, as the value has to be already close to the boundary which is unlikely for both variables.

The variables hurs and rsds are described with a Gaussian distribution. Those variables are bounded, hurs is between 0 and 1 and rsds is always non-negative. To avoid invalid values after quantile mapping, values that are outside the defined range after quantile mapping are reset to the closest boundary value. 

The sfcWind variable is modeled with a Weibull distribution using two parameters. 
The shape parameter _alpha_ is assumed to be free of trend. 
Both parameters need to be positive. 
Therefore, the model output is transformed with the logistic function to produce positive outputs.

Precipitation is modeled with a mixed Bernoulli-gamma distribution.  
This approach enables to model the probability of rain-days and precipitation amounts on rain days with one distribution.
The Bernoulli-gamma distribution has three parameters. All three parameters are modeled as a linear function of the global mean temperature. 
The model does not contain a yearly cycle for any parameter. 

### Tasmin and Tasmax

Following -Lange--Piani et al- we model tasrange = tasmax - tasmin and tasskew = (tas-tasmin)/tasrange instead of tasmin and tasmax. Tasmin and Tasmax are then calculated from those variables with the formulas:

tasmin = 
tasmax = 

### huss

Huss is also not directly modeled, but generated in postprocessing from the variables tas, pr, hurs.
huss = f(tas, pr ,hurs)

## Results

![Trend Maps](isicfact_v1.0.0_gswp3_trend_map.png)
*Figure 2: *


## Example

See [here](examples/tas_example.ipynb) for a notebook leading you through the basic steps.

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

## Comments for each variable

#### daily mean temperature (tas)
Two cells fail in complete dataset.
67418 of 6420 cells.

#### tasskew
Work in progress. See #59

#### tasrange
Calculation alsmost complete on full dataset.
Some cells do not detrend as expected. Need assessment.
See #60.

#### precipiation (pr)
Calculatio complete.
67339 of 67420 work.

#### sea level pressure (ps)
Calculation complete on full dataset.
See #65

#### wind
Calculation complete on full dataset.
Minor issues on the coast of the Arabic Peninsula.
See #66

#### longwave radiation (rlds)
Calculation complete on full dataset.
Need check of trend removal.

#### shortwave radiation (rsds)

#### relative humidity (hurs)



## Comments for datasets

GSWP: needs preprocessing to rename from rhs to hurs, and mask invalid values below zero:

```
ncrename -O -v rhs,hurs fname1.nc fname2.nc

cdo setrtomiss,-1e20,0 fname2.nc fname3.nc
```


## Credits

The code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 is inspired and adopted from [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.

## License

This code is licensed under GPLv3, see the LICENSE.txt. See commit history for authors.

