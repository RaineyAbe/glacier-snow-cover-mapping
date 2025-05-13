"""
Estimate glacier snow cover in Sentinel-2, and/or Landsat 8/9: full pipeline
Rainey Aberle
Department of Geosciences, Boise State University
2025

Requirements:
- Google Earth Engine (GEE) account: used to pull DEM over the AOI.
                                     Sign up for a free account [here](https://earthengine.google.com/new_signup/).
- Google Drive folder: where output snow cover statistics will be saved

Outline:
    0. Setup paths in directory, file locations, authenticate GEE
    1. Load Area of Interest (AOI) and digital elevation mode (DEM)
    2. Sentinel-2 Top of Atmosphere (TOA) imagery: full pipeline
    3. Sentinel-2 Surface Reflectance (SR) imagery: full pipeline
    4. Landsat 8/9 Surface Reflectance (SR) imagery: full pipeline
"""
# ----------------- #
# --- 0. Set up --- #
# ----------------- #

import os
import ee
import sys
import argparse

def getparser():
    parser = argparse.ArgumentParser(description="snow_classification_pipeline with arguments passed by the user",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-out_folder', default=None, type=str, help='Path in directory where output images will be saved')
    parser.add_argument('-project_id', default=None, type=str, help='Google Earth Engine project ID, managed on your account.')
    parser.add_argument('-glac_id', default=None, type=str, help="GLIMS glacier ID used to query and clip imagery.")
    parser.add_argument('-date_start', default=None, type=str, help='Start date for image querying: "YYYY-MM-DD"')
    parser.add_argument('-date_end', default=None, type=str, help='End date for image querying: "YYYY-MM-DD"')
    parser.add_argument('-month_start', default=1, type=int, help='Start month for image querying (inclusive), e.g. 5')
    parser.add_argument('-month_end', default=12, type=int, help='End month for image querying (inclusive), e.g. 10')
    parser.add_argument('-mask_clouds', default=True, type=bool, help='Whether to mask clouds in images.')
    parser.add_argument('-min_aoi_coverage', default=70, type=int,
                        help='Minimum percent coverage of the AOI after cloud masking (0-100)')
    parser.add_argument('-steps_to_run', default=None, nargs="+", type=int,
                        help='List of steps to be run, e.g. [1, 2, 3]. '
                             '1=Sentinel-2_TOA, 2=Sentinel-2_SR, 3=Landsat')
    
    return parser


def main():
    # -----Set user arguments as variables
    parser = getparser()
    args = parser.parse_args()
    out_folder = args.out_folder
    project_id = args.project_id
    glac_id = args.glac_id
    date_start = args.date_start
    date_end = args.date_end
    month_start = args.month_start
    month_end = args.month_end
    mask_clouds = args.mask_clouds
    min_aoi_coverage = args.min_aoi_coverage
    steps_to_run = args.steps_to_run

    # -----Import pipeline utilities
    # When running locally, must import from "functions" folder
    script_path = os.path.dirname(os.path.abspath(__file__))
    if "functions" in os.listdir(os.path.join(script_path, '..')):
        sys.path.append(os.path.join(script_path, '..', 'functions'))
    # In Docker image, all files are in "/app" folder
    else:
        sys.path.append(os.path.join(script_path))
    import pipeline_utils_GEE as utils

    # -----Authenticate and initialize GEE
    try:
        ee.Initialize(project=project_id)
    except:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    # -----Load AOI from GLIMS
    glims = ee.FeatureCollection('GLIMS/20230607')
    aoi = glims.filter(ee.Filter.eq('glac_id', glac_id))
    # Merge all geometries to use as the AOI 
    aoi = aoi.union().geometry()

    # -----Load the DEM
    dem = utils.query_gee_for_dem(aoi)

    # -----Split the date range into separate years
    # If the GEE computations take too long (> ~12 h), the final export times out. 
    # Splitting the date range and running each separately helps to mitigate time-out. 
    date_ranges = utils.split_date_range_by_year(date_start, date_end, month_start, month_end)

    # ------------------------- #
    # --- 1. Sentinel-2 TOA --- #
    # ------------------------- #
    if 1 in steps_to_run:
        print('----------')
        print('Sentinel-2 TOA')
        print('----------')

        # Define dataset-specific params
        dataset = "Sentinel-2_TOA"
        resolution = 10

        # Run the workflow for each year in the date range separately 
        for date_range in date_ranges:
            print('\n', date_range)

            # Query GEE for imagery
            image_collection = utils.query_gee_for_imagery(dataset, aoi, date_range[0], date_range[1], month_start, month_end, 
                                                        min_aoi_coverage, mask_clouds)

            # Classify image collection
            classified_collection = utils.classify_image_collection(image_collection, dataset)

            # Calculate snow cover statistics, export to Google Drive
            stats = utils.calculate_snow_cover_statistics(classified_collection, dem, aoi, scale=resolution, out_folder=out_folder,
                                                          file_name_prefix=f"{glac_id}_{dataset}_snow_cover_stats_{date_range[0]}_{date_range[1]}")

    # ------------------------ #
    # --- 2. Sentinel-2 SR --- #
    # ------------------------ #
    if 2 in steps_to_run:

        print('----------')
        print('Sentinel-2 SR')
        print('----------')

        # Define dataset-specific params
        dataset = "Sentinel-2_SR"
        resolution = 10

        # Run the workflow for each year in the date range separately 
        for date_range in date_ranges:
            print('\n', date_range)

            # Query GEE for imagery
            image_collection = utils.query_gee_for_imagery(dataset, aoi, date_range[0], date_range[1], month_start, month_end, 
                                                        min_aoi_coverage, mask_clouds)

            # Classify image collection
            classified_collection = utils.classify_image_collection(image_collection, dataset)

            # Calculate snow cover statistics, export to Google Drive
            stats = utils.calculate_snow_cover_statistics(classified_collection, dem, aoi, scale=resolution, out_folder=out_folder,
                                                        file_name_prefix=f"{glac_id}_{dataset}_snow_cover_stats_{date_range[0]}_{date_range[1]}")


    # ------------------------- #
    # --- 3. Landsat 8/9 SR --- #
    # ------------------------- #
    if 3 in steps_to_run:

        print('----------')
        print('Landsat 8/9 SR')
        print('----------')

        # Define dataset-specific params
        dataset = "Landsat"
        resolution = 30

        # Run the workflow for each year in the date range separately 
        for date_range in date_ranges:
            print('\n', date_range)

            # Query GEE for imagery
            image_collection = utils.query_gee_for_imagery(dataset, aoi, date_range[0], date_range[1], month_start, month_end, 
                                                        min_aoi_coverage, mask_clouds)

            # Classify image collection
            classified_collection = utils.classify_image_collection(image_collection, dataset)

            # Calculate snow cover statistics, export to Google Drive
            stats = utils.calculate_snow_cover_statistics(classified_collection, dem, aoi, scale=resolution, out_folder=out_folder,
                                                          file_name_prefix=f"{glac_id}_{dataset}_snow_cover_stats_{date_range[0]}_{date_range[1]}")


    print('Done!')

if __name__ == '__main__':
    main()
