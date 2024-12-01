#!/bin/bash
#SBATCH -J example              # job name
#SBATCH -o log_slurm.o%j        # output and error file name (%j expands to job$)
#SBATCH -n 28                   # total number of tasks requested
#SBATCH -N 1                    # number of nodes you want to run on
#SBATCH -p defq                 # queue (partition)
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 12.0 hours in this exam$

# source your .bashrc file (e.g., for PATH modifications and Conda initialization)
. ~/.bashrc

# activate conda environment
conda activate snow-cover-mapping

# (optional) define site name for convenience
site_name="SITE_NAME"

# run your code
python /bsuhome/raineyaberle/scratch/snow-cover-mapping/scripts/snow_classification_pipeline_pass_arguments.py \
-site_name $site_name \
-base_path "/Research/PhD/snow_cover_mapping/snow-cover-mapping/" \
-AOI_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/AOIs/" \
-AOI_fn "${site_name}_RGI_outline.shp" \
-out_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/" \
-figures_out_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/figures/" \
-date_start "2013-01-01" \
-date_end "2023-12-01" \
-month_start 5 \
-month_end 10 \
-mask_clouds True \
-cloud_cover_max 70 \
-aoi_coverage 70 \
-steps_to_run 1 2 3
