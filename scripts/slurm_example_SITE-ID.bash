#!/bin/bash
#SBATCH -J SITE-ID                          # job name
#SBATCH -o output_SITE-ID.o%j               # output and error file name (%j expands to job$)
#SBATCH -n 48                               # total number of tasks requested
#SBATCH -N 1                                # number of nodes you want to run on
#SBATCH -p bsudfq                           # queue (partition)
#SBATCH -t 48:00:00                         # run time (hh:mm:ss)

# Load apptainer module
module load apptainer/1.2.5

# Pull the Docker image if it doesn't exist in scratch/
cd scratch
FILE="glacier-snow-cover-mapping_latest.sif"
if [ ! -f $FILE ]; then
    apptainer pull docker://raineyabe/glacier-snow-cover-mapping
else
    echo "Docker image exists, skipping pull."
fi

# Define site name here for convenience
SITE_NAME="RGI60-SITE-ID"

# Run the pipeline in the container - CHANGE "GLAC_ID" TO YOUR GLIMS GLAC_ID
apptainer exec $FILE \
/opt/conda/bin/python /app/snow_classification_pipeline_SKL.py \
-site_name $SITE_NAME \
-aoi_path "/bsuhome/raineyaberle/scratch/snow_cover_mapping/study-sites/${SITE_NAME}/AOIs/" \
-aoi_fn "${SITE_NAME}_outline.shp" \
-dem_path "/bsuhome/raineyaberle/scratch/snow_cover_mapping/study-sites/${SITE_NAME}/DEMs/" \
-out_path "/bsuhome/raineyaberle/scratch/snow_cover_mapping/study-sites/${SITE_NAME}/" \
-figures_out_path "/bsuhome/raineyaberle/scratch/snow_cover_mapping/study-sites/${SITE_NAME}/figures/" \
-date_start "2013-05-01" \
-date_end "2023-11-01" \
-month_start 5 \
-month_end 10 \
-mask_clouds True \
-min_aoi_coverage 70 \
-im_download True \
-steps_to_run 1 2 3
