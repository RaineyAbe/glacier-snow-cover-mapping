# Run the snow classification pipeline by passing arguments to the Python script.
# Rainey Aberle
# 2025
#
# To run:
# - Make a copy of this script
# - Modify the arguments below with your file paths and settings
# - Open a command window
# - Change directory ("cd") to glacier-snow-cover-mapping/scripts/
# - Activate environment
# - Run the script: "sh snow_classification_pipeline_args_example.sh" replacing with your file name.

# Run snow classification pipeline - modify arguments before running
python snow_classification_pipeline_GEE.py \
-code_path "/Users/raineyaberle/Research/PhD/snow_cover_mapping/glacier-snow-cover-mapping/" \
-out_folder "snow_cover_exports" \
-project_id "ee-raineyaberle" \
-glac_id "G211100E60420N" \
-date_start "2020-06-01" \
-date_end "2025-05-01" \
-month_start 6 \
-month_end 11 \
-mask_clouds True \
-min_aoi_coverage 70 \
-steps_to_run 2
