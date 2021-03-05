#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --account=isimip
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=derive_huss
#SBATCH --output=derive_huss.%j.log
#SBATCH --error=derive_huss.%j.log

function get_cdoexpr_huss_Weedon2010style {
  # returns the cdo expression that calculates specific humidity from
  # relative humidity, air pressure and temperature using the equations of
  # Buck (1981) Journal of Applied Meteorology 20, 1527-1532,
  # doi:10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2 as described in
  # Weedon et al. (2010) WATCH Technical Report 22,
  # url:www.eu-watch.org/publications/technical-reports

  local shum=$1  # name of specific humidity [kg/kg]
  local rhum=$2  # name of relative humidity [1]
  local pres=$3  # name of air pressure [mb]
  local temp=$4  # name of temperature [degC]

  # ratio of the specific gas constants of dry air and water vapor after Weedon2010
  local RdoRv=0.62198

  # constants for calculation of saturation water vapor pressure over water and ice after Weedon2010, i.e.,
  # using Buck1981 curves e_w4, e_i3 and f_w4, f_i4
  local aw=6.1121   # [mb]
  local ai=6.1115   # [mb]
  local bw=18.729
  local bi=23.036
  local cw=257.87   # [degC]
  local ci=279.82   # [degC]
  local dw=227.3    # [degC]
  local di=333.7    # [degC]
  local xw=7.2e-4
  local xi=2.2e-4
  local yw=3.20e-6
  local yi=3.83e-6
  local zw=5.9e-10
  local zi=6.4e-10

  # prepare usage of different parameter values above and below 0 degC
  local a="(($temp>0)?$aw:$ai)"
  local b="(($temp>0)?$bw:$bi)"
  local c="(($temp>0)?$cw:$ci)"
  local d="(($temp>0)?$dw:$di)"
  local x="(($temp>0)?$xw:$xi)"
  local y="(($temp>0)?$yw:$yi)"
  local z="(($temp>0)?$zw:$zi)"

  # saturation water vapor pressure part of the equation
  local saturationpurewatervaporpressure="$a*exp(($b-$temp/$d)*$temp/($temp+$c))"
  local enhancementfactor="1.0+$x+$pres*($y+$z*$temp^2)"
  local saturationwatervaporpressure="($saturationpurewatervaporpressure)*($enhancementfactor)"

  # saturation water vapor pressure -> saturation specific humidity -> specific humidity
  echo "$shum=$rhum*$RdoRv/($pres/($saturationwatervaporpressure)+$RdoRv-1.0);"
  return 0
}


cd ..
prefix=gswp3-w5e5_counterclim_
postfix=_global_daily_1901_2016_intermediate.nc
ofile=${prefix}huss_global_daily_1901_2016_intermediate.nc
years=$(seq 1901 2016)
cdo="cdo -O -f nc4c -z zip"

var_limits_lower=.0000001
var_limits_upper=.1

hfiles=
for clim in counterclim obsclim
do
  echo $clim
  case $clim in
  counterclim)
    infix=;;
  obsclim)
    infix=_orig;;
  esac  # clim

  hfile=$ofile.$clim
  cdoexprlower="huss$infix=(huss$infix<$var_limits_lower)?$var_limits_lower:huss$infix;"
  cdoexprupper="huss$infix=(huss$infix>$var_limits_upper)?$var_limits_upper:huss$infix;"
  cdoexprhuss=$(get_cdoexpr_huss_Weedon2010style "huss$infix" "(hurs$infix*0.01)" "(ps$infix*0.01)" "(tas$infix-273.15)")
  $cdo merge -selname,hurs$infix ${prefix}hurs$postfix -selname,ps$infix ${prefix}ps$postfix -selname,tas$infix ${prefix}tas$postfix $hfile
  for year in $years; do $cdo -expr,"$cdoexprlower" -expr,"$cdoexprupper" -expr,"$cdoexprhuss" -selyear,$year $hfile $hfile$year; done
  $cdo mergetime $hfile???? $hfile
  rm $hfile????
  hfiles="$hfiles $hfile"
  echo
done  # clim
$cdo merge $hfiles $ofile
rm $hfiles
