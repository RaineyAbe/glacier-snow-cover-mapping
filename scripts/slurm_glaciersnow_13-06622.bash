#!/bin/bash
#SBATCH -J 13-06622                          # job name
#SBATCH -o output_13-06622.o%j               # output and error file name (%j expands to job$)
#SBATCH -n 48                               # total number of tasks requested
#SBATCH -N 1                                # number of nodes you want to run on
#SBATCH -p bsudfq                           # queue (partition)
#SBATCH -t 48:00:00                         # run time (hh:mm:ss)

# Load apptainer module
module load apptainer/1.2.5

# Pull the Docker image if it doesn't exist in scratch/
cd /bsuhome/ellynenderlin/scratch/
FILE="glacier-snow-cover-mapping_latest.sif"
if [ ! -f $FILE ]; then
    apptainer pull docker://raineyabe/glacier-snow-cover-mapping
else
    echo "Docker image exists, skipping pull."
fi

# Define site name here for convenience
RGI_VERSION="RGI2000-v7.0"
SITE_NAME="13-06622"

#mount relevant file directories in the container
export APPTAINER_BINDPATH="/bsuscratch,/bsuhome,/bsushare"

# Run the pipeline in the container
code_dir="/bsuhome/ellynenderlin/scratch/NASA_HMA-surges/glacier-snow-cover-mapping"
apptainer exec $FILE \
/opt/conda/bin/python $code_dir/scripts/snow_classification_pipeline.py \
-site_name $SITE_NAME \
-aoi_path "/bsuhome/ellynenderlin/scratch/NASA_HMA-surges/study-sites/${SITE_NAME}/AOIs/" \
-aoi_fn "${RGI_VERSION}-G-${SITE_NAME}_outline.shp" \
-dem_path "/bsuhome/ellynenderlin/scratch/NASA_HMA-surges/study-sites/${SITE_NAME}/DEMs/" \
-out_path "/bsuhome/ellynenderlin/scratch/NASA_HMA-surges/study-sites/${SITE_NAME}/" \
-figures_out_path "/bsuhome/ellynenderlin/scratch/NASA_HMA-surges/study-sites/${SITE_NAME}/figures/" \
-date_start "2014-01-01" \
-date_end "2025-12-31" \
-month_start 1 \
-month_end 12 \
-mask_clouds True \
-min_aoi_coverage 70 \
-im_download True \
-steps_to_run 1 2 3
