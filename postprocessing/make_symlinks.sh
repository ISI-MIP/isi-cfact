#/bin/bash

output_dir=/p/tmp/sitreu/isimip/isi-cfact/output

for var in tas tasskew tasrange pr hurs sfcwind rlds rsds ps tasmin tasmax
do
  ln -s ${output_dir}/isicf033_gswp3-w5e5_${var}_sub01/cfact/${var}/${var}_GSWP3-W5E5_cfactual_rechunked_valid.nc4 ../gswp3-w5e5_counterclim_${var}_global_daily_1901_2016_intermediate.nc
done

