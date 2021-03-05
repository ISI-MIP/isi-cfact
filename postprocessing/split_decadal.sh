#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=isicf_split_to_decades
#SBATCH --account=isimip
#SBATCH --output=%x.log
#SBATCH --error=%x.log
#SBATCH --ntasks=1

data_dir=/p/projects/isimip/isimip/sitreu/data/attrici/20201205_IsimipCounterfactualGSWP3-W5E5-revised/daily/
mkdir $data_dir/decadal_data

for input_file in ${data_dir}/*1901_2016_intermediate.nc
do
  var=$(echo ${input_file} |  awk -F '_' '{print $4}')
  input_base=$(basename ${input_file})
  output_stub_base=${input_base/_1901_2016_intermediate\.nc/}
  output_stub=${data_dir}/decadal_data/${output_stub_base}
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1901/1910 -selvar,${var} ${input_file} ${output_stub}_1901_1910.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1911/1920 -selvar,${var} ${input_file} ${output_stub}_1911_1920.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1921/1930 -selvar,${var} ${input_file} ${output_stub}_1921_1930.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1931/1940 -selvar,${var} ${input_file} ${output_stub}_1931_1940.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1941/1950 -selvar,${var} ${input_file} ${output_stub}_1941_1950.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1951/1960 -selvar,${var} ${input_file} ${output_stub}_1951_1960.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1961/1970 -selvar,${var} ${input_file} ${output_stub}_1961_1970.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1971/1980 -selvar,${var} ${input_file} ${output_stub}_1971_1980.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1981/1990 -selvar,${var} ${input_file} ${output_stub}_1981_1990.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,1991/2000 -selvar,${var} ${input_file} ${output_stub}_1991_2000.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,2001/2010 -selvar,${var} ${input_file} ${output_stub}_2001_2010.nc
  cdo -f nc4c -z zip setmissval,1.e+20 -selyear,2011/2016 -selvar,${var} ${input_file} ${output_stub}_2011_2016.nc
done

