#!/bin/bash
#SBATCH -J GLAC_ID                          # job name
#SBATCH -o output_GLAC_ID.o%j               # output and error file name (%j expands to job$)
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

# Run the pipeline in the container - CHANGE "GLAC_ID" TO YOUR GLIMS GLAC_ID
apptainer exec $FILE \
/opt/conda/bin/python /app/snow_classification_pipeline_GEE.py \
-glac_id "GLAC_ID" \
-project_id "ee-raineyaberle" \
-out_folder "snow_cover_exports" \
-date_start "2013-06-01" \
-date_end "2024-11-01" \
-month_start 6 \
-month_end 10 \
-mask_clouds True \
-min_aoi_coverage 70 \
-steps_to_run 1 2 3
