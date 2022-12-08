#!/bin/bash
#SBATCH -J SouthCascade                     # job name
#SBATCH -o log_slurm_SouthCascade.o%j       # output and error file name (%j expands to job$)
#SBATCH -n 28                               # total number of tasks requested
#SBATCH -N 1                                # number of nodes you want to run on
#SBATCH -p defq                             # queue (partition)
#SBATCH -t 24:00:00                         # run time (hh:mm:ss)

## activate conda environment
. ~/.bashrc
conda activate planet-snow

# define paths and file names
site_name="SouthCascade"
base_path="/home/raberle/scratch/snow_cover_mapping/snow-cover-mapping/"
im_path="$base_path../study-sites/$site_name/imagery/PlanetScope/2016-2022/"
AOI_path="$base_path../study-sites/$site_name/glacier_outlines/"
AOI_fn="${site_name}_USGS_glacier_outline*.shp"
DEM_path="$base_path../study-sites/$site_name/DEMs/"
DEM_fn="${site_name}*_DEM_filled.tif"
out_path="$im_path../"

# run your code
python /home/raberle/scratch/snow_cover_mapping/snow-cover-mapping/scripts/snow_classification_pipeline_PlanetScope.py \
-base_path $base_path \
-site_name $site_name \
-im_path $im_path \
-AOI_path $AOI_path \
-AOI_fn $AOI_fn \
-DEM_path $DEM_path \
-DEM_fn $DEM_fn \
-out_path $out_path \
-steps_to_run 3 4 5
