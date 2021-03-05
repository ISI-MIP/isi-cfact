# Attrici Postprocessing

## Step 1 - Create folder to calculate outputs

   * create a folder to collect outputs, see e.g. `/p/projects/isimip/isimip/sitreu/data/attrici/20201205_IsimipCounterfactualGSWP3-W5E5-revised/daily`
   
## Step 2 Calculate tasmin, tasmax

* use `create_tasmin_tasmax.py`
    * change *line 8* to match your output
    * change *line 9* to match your dataset
    * change *lines 11-13* to match your runids
* `submit_create_tasmin_tasmax.sh`  
    * change *line 14* to match your conda env

## Step 3 Collect model outputs in a folder

* use `make_symlinks.sh` to make symlinks
    * change *line 3* to match your output base
    * change *line 7* to match the run_id, the dataset and the covered time period

## Step 4 - calculate huss
* use `derive_huss.sh`
    * change *line 65* to cd to where your symlinks are
    * change *line 66* to match the filename before the variable
    * change *line 67* to match the filename after the variable
    * change *line 68* accordingly
    * change *line 69* to match the covered period

## Step 5 - split to decadal
* use `split_decadal.sh` to create decadal netCDF files 
    * change *line 11* to match the base folder of your symlinks
    * change *line 14* and *line 18* to match your covered time period
    * if necessary change the covered time periods
    * make sure *line 16* returns the variable
* decadal data will be saved in `decadal_data`

## Step 3 - Fix metadata
### Variable specific metadata
* Use `change_local_vars.sh` to change variable specific metadata. 
* They need to be fixed for tasmin, tasmax and huss (as they are produced from other variables). 
    * change *line 3* to match the path to your decadal data
* Make sure all variable metadata is correct. 
* If there are fixes necessary, they can be done in `change_local_vars.sh`
### Global metadata
* use `set_global_metadata.sh` to fix global metadata.
* It deletes history and all unnecessary fields. 
    * change *line 3* to match the path to your decadal data
    * change title in *line 17*
    * change summary in *line 21*
    * change version in *line 22*
* describe version in the field XXX
* describe data in the field XXX
* running `change_global` should be the last step of postprocessing. 
