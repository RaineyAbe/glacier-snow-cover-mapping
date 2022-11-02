#!/bin/bash
#SBATCH -J Sperry               # job name
#SBATCH -o log_slurm.o%j        # output and error file name (%j expands to job$)
#SBATCH -n 28                   # total number of tasks requested
#SBATCH -N 1                    # number of nodes you want to run on
#SBATCH -p defq                 # queue (partition)
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 12.0 hours in this exam$

# activate conda environment
. ~/.bashrc
conda activate planet-snow

# define paths and file names
site_name="Sperry"
base_path="/home/raberle/scratch/snow_cover_mapping/snow-cover-mapping/"
im_path="$base_path../study-sites/$site_name/imagery/PlanetScope/2016-2021/"
AOI_path="$im_path../../../glacier_outlines/"
AOI_fn="${site_name}_USGS_glacier_outline*.shp"
DEM_path="$base_path../study-sites/$site_name/DEMs/"
DEM_fn="${site_name}*_DEM_filled.tif"
out_path="$im_path../"

# run your code
python /home/raberle/scratch/snow_cover_mapping/snow-cover-mapping/scripts/PlanetScope_snow_classification_pipeline.py \
-base_path $base_path \
-site_name $site_name \
-im_path $im_path \
-AOI_path $AOI_path \
-AOI_fn $AOI_fn \
-DEM_path $DEM_path \
-DEM_fn $DEM_fn \
-out_path $out_path \
-steps_to_run 1 2 3 4 5
