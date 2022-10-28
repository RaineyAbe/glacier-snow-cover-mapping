#!/bin/bash
#SBATCH -J Gulkana              # job name
#SBATCH -o log_slurm.o%j        # output and error file name (%j expands to job$)
#SBATCH -n 28                   # total number of tasks requested
#SBATCH -N 1                    # number of nodes you want to run on
#SBATCH -p defq                 # queue (partition)
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 12.0 hours in this exam$

# activate conda environment
. ~/.bashrc
conda activate planet-snow

# define paths and file names
site_name="Gulkana"
base_path="/home/raberle/scratch/snow_cover_mapping/snow-cover-mapping/"
im_path="$base_path../study-sites/$site_name/imagery/PlanetScope/2016-2021/"
AOI_path="$base_path../RGI_outlines/"
AOI_fn="Gulkana_RGI.shp"
DEM_path="$base_path../study-sites/$site_name/DEMs/"
DEM_fn="ArcticDEM_clip_$site_name.tif"
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
-steps_to_run 1 2 3 4
