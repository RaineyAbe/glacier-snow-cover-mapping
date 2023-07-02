# Run the snow classification pipeline by passing arguments to the Python script.
# Rainey Aberle
# 2023
#
# This may be convenient for batch running the workflow for multiple sites.
# Make a copy of this script and simply modify the arguments below.
# To run:
# - Modify the arguments below with your file paths and settings
# - Open a Terminal window
# - Change directory ("cd") to snow_cover_mapping/scripts/
# - Activate conda environment: "conda activate snow-cover-mapping"
# - Run the script: "sh pipeline_args_example.sh"

# Define site name for convenience if desired
site_name="Emmons"

# Run snow classification pipeline - modify arguments before running
python snow_classification_pipeline_pass_arguments.py \
-site_name $site_name \
-base_path "/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/" \
-AOI_path "/Users/raineyaberle/Google Drive/My Drive/Research/PhD/snow_cover_mapping/study-sites/${site_name}/AOIs/" \
-AOI_fn "${site_name}_RGI_outline.shp" \
-out_path "/Users/raineyaberle/Google Drive/My Drive/Research/PhD/snow_cover_mapping/study-sites/${site_name}/imagery/" \
-figures_out_path "/Users/raineyaberle/Google Drive/My Drive/Research/PhD/snow_cover_mapping/study-sites/${site_name}/figures/" \
-date_start "2013-01-01" \
-date_end "2023-06-01" \
-month_start 5 \
-month_end 10 \
-cloud_cover_max 70 \
-mask_clouds True \
-steps_to_run 1 2 3


