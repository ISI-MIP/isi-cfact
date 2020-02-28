# ISI-CFACT
## Introduction
ISI-CFACT produces counterfactual climate data from past datasets for the ISIMIP project.
Counterfactual climate is a hypothetical climate in a world without climate change.
For impact models, such climate should stay as close as possible to the observed past,
as we aim to compare impact events of the past (for which we have data) to the events in the counterfactual. The difference between past impacts and counterfactual impacts is a proxy for the impacts caused by climate change. We run the following steps:

1. We approximate the change in past climate through a model with three parts. Long-term trend, an ever-repeating yearly cycle, and a trend in the yearly cycle. Trends are induced by global mean temperature change. We use a Bayesian approach to estimate all parameters of the model and their dependencies at once, here implemented through pymc3. Yearly cycle and trend in yearly cycles are approximated through a finite number of modes, which are periodic in the year. The parameter distributions tell us which part of changes in the variables can be explained through global mean temperature as a direct driver.

2. We do quantile mapping to map each value from the observed dataset to a value that we expect it would have been without the climate-induced trend. Our hierarchical model approach provides us with a time evolution of our distribution through the time evolution of a gmt-dependent parameter. We first use this time-evolving distribution to map each value to its quantile in this time evolving distribution. We then use the distribution from a reference period in the beginning of our dataset where we assume that climate change did not play a role, to remap the quantile to a value of the variable. This value is our counterfactual value. Quantile mapping is different for each day of the year because our model is sensitive to the yearly cycle and the trend in the yearly cycle. This approach is illustrated in Figure 1.

The following graph illustrates the approach. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.

![Counterfactual example](image01.png)

*Figure 1: Model for a climate variable with yearly cycle. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.*

### Variables

To avoid large relative errors in the daily temperature range as pointed out by Piani et al. (2010), we de-trend the daily temperature range tasrange = tasmax - tasmin and the skewness of the daily temperature tasskew = (tas-tasmin)/tasrange and derive tasmin and tasmax from tas, tasrange and tasskew.

A counterfactual huss is derived from the counterfacual tas, ps and hurs using the equations of Buck (1981) as described in Weedon et al. (2010).
huss = f(tas, pr ,hurs)

| Variable | Short name | Unit | Statistical model |
| -------- | ---------- | ---- | ----------------- |
| Near-Surface Air Temperature | tas | K | Gaussian |
| Range of daily temperature | tasrange | K | Bounded Gaussian |
| Skewness of daily temperature | tasskew | 1 | Bounded Gaussian |
| Daily Minimum Near-Surface Air Temperature | tasmin | K | Derived from tas, tasrange and tasskew |
| Daily Maximum Near-Surface Air Temperature | tasmax | K | Derived from tas, tasrange and tasskew |
| Surface Downwelling Longwave Radiation | rlds | W / m² | Gaussian |
| Surface Downwelling Shortwave Radiation | rsds | W / m²| Censored Gaussian |
| Surface Air Pressure | ps | Pa | Gaussian |
| Near-Surface Wind Speed | sfcWind | m / s | Weibull |
| Precipitation | pr | kg / m² s | Bernoulli-Gamma |
| Near-Surface Relative Humidity | hurs | % | Censored Gaussian |
| Near-Surface Specific Humidity | huss | kg / kg | Derived from hurs ps and tas |

*Table 1: Specs of climate variables for the ISIMIP3b counterfactual climate datasets. The variables tasrange and tasskew are auxiliary variables to calculate tasmin and tasmax* 

## Model

A global mean temperature timeseries 
without yearly variations is used as predictor for the model.
To generate this predictor, global mean temperature is preprocessed using singular spectrum analysis. 

The variables tas, rlds, and ps are modeled with a Gaussian distribution with a time varying mean value. The mean value is a linear function of the global mean temperature change plus a yearly cycle. This yearly cycle is modeled with one mode for all variables except tasskew, where two modes are used. The parameters of the yearly cycle are also a linear function of the global mean temperature.

Tasrange and tasskew are modeled with a Gaussian distribution as described above. But those variables are bounded. Tasrange is positive and tasskew between 0 and 1.
 In the quantile mapping step, values that are close to the boundary can get mapped to values outside the defined range. To avoid this, such values are not quantile mapped with the effect that the counterfactual value is the same as the historic value in those cases. This happens only rarely, as the value has to be already close to the boundary which is unlikely for both variables.

The variables hurs and rsds are also bounded, hurs is between 0 and 1 and rsds is always non-negative. Those variables are also modeled with a Gaussian distribution. Values that are outside the defined range after quantile mapping are reset to the closest boundary value. 

The sfcWind variable is modeled with a Weibull distribution using two parameters. 
The shape parameter _alpha_ is assumed to be free of trend. 
Both parameters need to be positive. 
Therefore, the model output is transformed with the logistic function to produce positive outputs.

Precipitation is modeled with a mixed Bernoulli-gamma distribution.  
This approach enables to model the probability of rain-days and precipitation amounts on rain days with one distribution.
The Bernoulli-gamma distribution has three parameters. All three parameters are modeled as a linear function of the global mean temperature. 
The model does not contain a yearly cycle for any parameter. 

## Results
We here present summaries of each variable for both the original data based on GSWP3 and the GSWP3-W5E5, and the corresponding counterfactual. ISI-CFACT removes annual trends as well as trends in the yearly cycle. Hereby the trend is regarded with the global mean temperature as independent variable. As a visual check, we show a map of the slope of a linear trend with time as independent variable, relative to the standard deveation of the slope in all grid-cells (see Figure 2). By construction, our method should reduce this relative slope to a value close to zero. The trend calculated for this visual check is based on a simpler method that disregards the yearly-cycle of the variables.
### GSWP3 
![Trend Maps](isicfact_v1.0.0_gswp3_trend_map.png)

*Figure 2: Maps of linear trends in historical (left) and counterfactual (right) climate for the GSWP3 dataset. The linear trends for hurs, huss, pr, ps, rlds, rsds and sfcWind are calculated with time as independent variable and without a yearly cycle. The grid-cell show the slope of that trend relative to the standard deviation of the slope in all grid-cells* 
## References
- Buck, A.L.:New Equations for Computing Vapor Pressure and Enhancement Factor, J. Appl. Meteorol., 20, 1527–1532, 1981.
- Piani, C., Weedon, G. P., Best, M., Gomes, S. M., Viterbo, P.,
Hagemann, S., and Haerter, J. O.: Statistical bias correction
of global simulated daily precipitation and temperature for the
application of hydrological models, J. Hydrol., 395, 199–215,
https://doi.org/10.1016/j.jhydrol.2010.10.024, 2010.
- Weedon, G. P., Gomes, S., Viterbo, P., Österle, H., Adam, J. C., Bellouin, N., Boucher, O., and Best, M.: The WATCH forcing data 1958–2001: A meteorological forcing dataset for land surface and hydrological models, in: Technical Report no 22., available at: http://www.eu-watch.org/publications/technical-reports (last access: July 2016), 2010.



----
----

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

