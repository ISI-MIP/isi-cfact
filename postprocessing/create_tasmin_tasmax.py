import subprocess
from pathlib import Path

""" This code allows to create tasmax and tasmin from tas, tasrange and
    tasskew. It runs independent from the attrici package.
    Copy anywhere and adjust the variables below. """

output_base = Path("/p/tmp/mengel/isimip/isi-cfact/output/")
dataset = "GSWP3"

tas_runid = "isicf022_gswp3_tas_sub20_simplecauchyprior"
tasrange_runid = "isicf022_gswp3_tasrange_sub20_sharperslopepriors"
tasskew_runid = "isicf022_gswp3_tasskew_sub20_widerpriors"

def d(var):
    """ get file strings from cfact calculation """
    tdict = {"tas":tas_runid, "tasrange":tasrange_runid, "tasskew":tasskew_runid}
    dpath = output_base/tdict[var]/"cfact"/var
    return str(dpath)+"/"+var+"_"+dataset+"_cfactual_rechunked_valid.nc4 "

def dout(var):
    """ get file strings for output data """
    dpath = output_base/dataset
    dpath.mkdir(parents=True,exist_ok=True)
    return str(dpath)+"/"+var+"_"+dataset.lower()+"_cfactual_rechunked_valid.nc4 "

p = "module load cdo && cdo -O "

cmd = p+"-chname,tas,tasmin -chname,tas_orig,tasmin_orig -sub "+d("tas")+" -mul "+d("tasskew")+d("tasrange")+dout("tasmin")
print(cmd)
print("")
subprocess.check_call(cmd, shell=True)
cmd = p+"-chname,tasmin,tasmax -chname,tasmin_orig,tasmax_orig -add "+dout("tasmin")+d("tasrange")+dout("tasmax")
print(cmd)
subprocess.check_call(cmd, shell=True)
