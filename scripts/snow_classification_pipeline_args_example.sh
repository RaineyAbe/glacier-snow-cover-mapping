# Run the snow classification pipeline by passing arguments to the Python script.
# Rainey Aberle
# 2023
#
# To run:
# - Make a copy of this script
# - Modify the arguments below with your file paths and settings
# - Open a command window
# - Change directory ("cd") to snow_cover_mapping/scripts/
# - Activate conda environment: "conda activate snow-cover-mapping"
# - Run the script: "sh snow_classification_pipeline_args_example.sh" replacing with your file name.

# Define site name for convenience if desired
site_name="SITE-NAME"

# Run snow classification pipeline - modify arguments before running
python snow_classification_pipeline_pass_arguments.py \
-site_name $site_name \
-base_path "/Research/PhD/snow_cover_mapping/snow-cover-mapping/" \
-AOI_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/AOIs/" \
-AOI_fn "${site_name}_RGI_outline.shp" \
-out_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/imagery/" \
-figures_out_path "/Research/PhD/snow_cover_mapping/study-sites/${site_name}/figures/" \
-date_start "2013-01-01" \
-date_end "2023-12-01" \
-month_start 5 \
-month_end 10 \
-mask_clouds True \
-cloud_cover_max 70 \
-aoi_coverage 70 \
-steps_to_run 1 2 3
