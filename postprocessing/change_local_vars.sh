#/bin/bash

data_dir=/p/projects/isimip/isimip/sitreu/data/attrici/20201205_IsimipCounterfactualGSWP3-W5E5-revised/daily/decadal_data 

for file in ${data_dir}/gswp3-w5e5_counterclim_tasmax_global_daily_*.nc
do
  ncatted -a long_name,tasmax,o,c,"Maximum Surface Air Temperature" ${file}
done

for file in ${data_dir}/gswp3-w5e5_counterclim_tasmin_global_daily_*.nc
do
  ncatted -a long_name,tasmin,o,c,"Minimum Surface Air Temperature" ${file}
done
  
for file in ${data_dir}/gswp3-w5e5_counterclim_huss_global_daily_*.nc
do
  ncatted -a standard_name,huss,o,c,"specific_humidity" ${file}
done

for file in ${data_dir}/gswp3-w5e5_counterclim_huss_global_daily_*.nc
do
  ncatted -a long_name,huss,o,c,"Specific Humidity at time of Maximum Temperature" ${file}
done

for file in ${data_dir}/gswp3-w5e5_counterclim_huss_global_daily_*.nc
do
  ncatted -a units,huss,o,c,"kg kg-1" ${file}
done

# for file in ${data_dir}/gswp3-w5e5_counterclim_wind_global_daily_*.nc
# do
#   ncrename -v wind,sfcWind $file
#   mv $file ${file/wind/sfcWind} 
# done
#
