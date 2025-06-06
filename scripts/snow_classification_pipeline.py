"""
Estimate glacier snow cover in Sentinel-2, Landsat 8/9, and/or PlanetScope imagery: full pipeline
Rainey Aberle
Department of Geosciences, Boise State University
2023

Requirements:
- Area of Interest (AOI) shapefile: where snow will be classified in all available images.
- Google Earth Engine (GEE) account: used to pull DEM over the AOI.
                                     Sign up for a free account [here](https://earthengine.google.com/new_signup/).
- (Optional) Digital elevation model (DEM): used to extract elevations over the AOI and for each snowline.
             If no DEM is provided, the ArcticDEM will be loaded where there is coverage. Otherwise, the NASADEM will be used.
             ArcticDEM elevations are reprojected to the geoid to match the vertical datum of the NASADEM.
- (Optional) Pre-downloaded PlanetScope images.Download images using Planet Explorer (planet.com/explorer) or
             snow-cover-mapping/notebooks/download_PlanetScope_images.ipynb.

Outline:
    0. Setup paths in directory, file locations, authenticate GEE
    1. Sentinel-2 Top of Atmosphere (TOA) imagery: full pipeline
    2. Sentinel-2 Surface Reflectance (SR) imagery: full pipeline
    3. Landsat 8/9 Surface Reflectance (SR) imagery: full pipeline
    4. PlanetScope Surface Reflectance (SR) imagery: full pipeline
"""
# ----------------- #
# --- 0. Set up --- #
# ----------------- #

# -----Import packages
import xarray as xr
import os
import numpy as np
import glob
import geopandas as gpd
import sys
import ee
import json
from tqdm.auto import tqdm
from joblib import load
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


def getparser():
    parser = argparse.ArgumentParser(description="snow_classification_pipeline with arguments passed by the user",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-site_name', default=None, type=str, help='Name of study site')
    parser.add_argument('-project_id', default=None, type=str, help='Google Earth Engine project ID, managed on your account.')
    parser.add_argument('-aoi_path', default=None, type=str, help='Path in directory to area of interest geospatial file')
    parser.add_argument('-aoi_fn', default=None, type=str, help='Area of interest file name')
    parser.add_argument('-dem_path', default=None, type=str,
                        help='(Optional) Path in directory to digital elevation model')
    parser.add_argument('-dem_fn', default=None, type=str, help='(Optional) Digital elevation model file name')
    parser.add_argument('-out_path', default=None, type=str, help='Path in directory where output images will be saved')
    parser.add_argument('-ps_im_path', default=None, type=str, help='Path in directory where PlanetScope raw images '
                                                                    'are located')
    parser.add_argument('-date_start', default=None, type=str, help='Start date for image querying: "YYYY-MM-DD"')
    parser.add_argument('-date_end', default=None, type=str, help='End date for image querying: "YYYY-MM-DD"')
    parser.add_argument('-month_start', default=1, type=int, help='Start month for image querying (inclusive), e.g. 5')
    parser.add_argument('-month_end', default=12, type=int, help='End month for image querying (inclusive), e.g. 10')
    parser.add_argument('-mask_clouds', default=True, type=bool, help='Whether to mask clouds in images.')
    parser.add_argument('-min_aoi_coverage', default=70, type=int,
                        help='Minimum percent coverage of the AOI after cloud masking (0-100)')
    parser.add_argument('-im_download', default=False, type=bool, help='Whether to download intermediary images. '
                                                                       'If images clipped to the AOI exceed the GEE '
                                                                       'user memory limit, images must be ' 
                                                                       'downloaded regardless.')
    parser.add_argument('-delineate_snowline', default=False, type=bool, help='Whether to delineate the snowline from each classified image.')
    parser.add_argument('-steps_to_run', default=None, nargs="+", type=int,
                        help='List of steps to be run, e.g. [1, 2, 3]. '
                             '1=Sentinel-2_TOA, 2=Sentinel-2_SR, 3=Landsat, 4=PlanetScope')
    parser.add_argument('-verbose', default=True, type=bool,
                        help='Whether to print details for each image at each processing step.')

    return parser


def main():
    # -----Set user arguments as variables
    parser = getparser()
    args = parser.parse_args()
    site_name = args.site_name
    project_id = args.project_id
    aoi_path = args.aoi_path
    aoi_fn = args.aoi_fn
    dem_path = args.dem_path
    if dem_path == "None":
        dem_path = None
    dem_fn = args.dem_fn
    if dem_fn == "None":
        dem_fn = None
    out_path = args.out_path
    ps_im_path = args.ps_im_path
    figures_out_path = args.figures_out_path
    date_start = args.date_start
    date_end = args.date_end
    month_start = args.month_start
    month_end = args.month_end
    mask_clouds = args.mask_clouds
    min_aoi_coverage = args.min_aoi_coverage
    im_download = args.im_download
    delineate_snowline = args.delineate_snowline
    steps_to_run = args.steps_to_run
    verbose = args.verbose

    # -----Determine image clipping & plotting settings
    plot_results = True  # = True to plot figures of results for each image where applicable
    skip_clipped = False  # = True to skip PlanetScope images where bands appear "clipped", i.e. max(blue) < 0.8
    save_outputs = True  # = True to save SCAs and snowlines to file

    print(site_name)
    print(' ')

    # -----Set paths for output files
    s2_toa_im_path = os.path.join(out_path, 'Sentinel-2_TOA')
    s2_sr_im_path = os.path.join(out_path, 'Sentinel-2_SR')
    l_im_path = os.path.join(out_path, 'Landsat')
    ps_im_masked_path = os.path.join(out_path, 'PlanetScope', 'masked')
    ps_im_mosaics_path = os.path.join(out_path, 'PlanetScope', 'mosaics')
    im_classified_path = os.path.join(out_path, 'classified')
    snow_cover_stats_path = os.path.join(out_path, 'snow_cover_stats')
    figures_out_path = os.path.join(out_path, 'figures')

    # -----Import pipeline utilities
    # When running locally, must import from "functions" folder
    script_path = os.getcwd()
    if "functions" in os.listdir(os.path.join(script_path, '..')):
        sys.path.append(os.path.join(script_path, '..', 'functions'))
    # In Docker image, all files are in "/app" folder
    else:
        sys.path.append(os.path.join(script_path))
    import pipeline_utils_SKL as utils
    import PlanetScope_preprocessing as psp

    # -----Load dataset dictionary
    dataset_dict_fn = os.path.join(script_path, '..', 'inputs-outputs', 'datasets_characteristics.json')
    dataset_dict = json.load(open(dataset_dict_fn))

    # -----Authenticate and initialize GEE
    try:
        ee.Initialize(project=project_id)
    except:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    # -----Load AOI as gpd.GeoDataFrame
    aoi = gpd.read_file(os.path.join(aoi_path, aoi_fn))
    # reproject the AOI to WGS84 to solve for the optimal utm zone
    aoi_wgs = aoi.to_crs('EPSG:4326')
    aoi_wgs_centroid = [aoi_wgs.geometry[0].centroid.xy[0][0],
                        aoi_wgs.geometry[0].centroid.xy[1][0]]
    # grab the optimal utm zone EPSG code
    epsg_utm = utils.convert_wgs_to_utm(aoi_wgs_centroid[0], aoi_wgs_centroid[1])
    print('Optimal UTM CRS = EPSG:' + str(epsg_utm))
    # reproject AOI to the optimal utm zone
    aoi_utm = aoi.to_crs('EPSG:' + epsg_utm)

    # -----Load DEM as Xarray DataSet
    if dem_fn is None:
        # query GEE for DEM
        dem = utils.query_gee_for_dem(aoi_utm, site_name, dem_path)
    else:
        # load DEM as xarray DataSet
        dem = xr.open_dataset(os.path.join(dem_path, dem_fn))
        dem_crs = dem.rio.crs.to_epsg()
        dem = dem.rename({'band_data': 'elevation'})
        # set no data values to NaN
        dem = xr.where((dem > 1e38) | (dem <= -9999), np.nan, dem)
        # reproject the DEM to the optimal utm zone
        dem = dem.rio.write_crs(f'EPSG:{dem_crs}')
        dem = dem.rio.reproject(f'EPSG:{epsg_utm}')

    # ------------------------- #
    # --- 1. Sentinel-2 TOA --- #
    # ------------------------- #
    if 1 in steps_to_run:
        print('----------')
        print('Sentinel-2 TOA')
        print('----------')

        # -----Query GEE for imagery and run classification pipeline
        # Define dataset
        dataset = 'Sentinel-2_TOA'
        # Load trained image classifier
        clf_fn = os.path.join(script_path, '..', 'inputs-outputs', 'Sentinel-2_TOA_classifier_all_sites.joblib')
        clf = load(clf_fn)
        # Load feature columns (bands to use in classification)
        feature_cols_fn = os.path.join(script_path, '..', 'inputs-outputs', 'Sentinel-2_TOA_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))
        # Run the classification pipeline
        run_pipeline = True
        utils.query_gee_for_imagery_yearly(aoi_utm, dataset, date_start, date_end, month_start, month_end, mask_clouds,
                                       min_aoi_coverage, im_download, s2_toa_im_path, run_pipeline, dataset_dict, site_name,
                                       im_classified_path, snow_cover_stats_path, dem, clf, feature_cols,
                                       figures_out_path, plot_results, verbose, delineate_snowline)
        print(' ')

    # ------------------------ #
    # --- 2. Sentinel-2 SR --- #
    # ------------------------ #
    if 2 in steps_to_run:

        print('----------')
        print('Sentinel-2 SR')
        print('----------')

        # -----Query GEE for imagery and run classification pipeline
        # Define dataset
        dataset = 'Sentinel-2_SR'
        # Load trained image classifier
        clf_fn = os.path.join(script_path, '..','inputs-outputs', 'Sentinel-2_SR_classifier_all_sites.joblib')
        clf = load(clf_fn)
        # Load feature columns (bands to use in classification)
        feature_cols_fn = os.path.join(script_path, '..','inputs-outputs', 'Sentinel-2_SR_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))
        # Run the classification pipeline
        run_pipeline = True
        utils.query_gee_for_imagery_yearly(aoi_utm, dataset, date_start, date_end, month_start, month_end, mask_clouds,
                                       min_aoi_coverage, im_download, s2_sr_im_path, run_pipeline, dataset_dict, site_name,
                                       im_classified_path, snow_cover_stats_path, dem, clf, feature_cols,
                                       figures_out_path, plot_results, verbose, delineate_snowline)
        print(' ')

    # ------------------------- #
    # --- 3. Landsat 8/9 SR --- #
    # ------------------------- #
    if 3 in steps_to_run:

        print('----------')
        print('Landsat 8/9 SR')
        print('----------')

        # -----Query GEE for imagery (and download to l_im_path if necessary)
        # Define dataset
        dataset = 'Landsat'
        # Load trained image classifier
        clf_fn = os.path.join(script_path, '..','inputs-outputs', 'Landsat_classifier_all_sites.joblib')
        clf = load(clf_fn)
        # Load feature columns (bands to use in classification)
        feature_cols_fn = os.path.join(script_path, '..','inputs-outputs', 'Landsat_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))
        # Run the classification pipeline
        run_pipeline = True
        utils.query_gee_for_imagery_yearly(aoi_utm, dataset, date_start, date_end, month_start, month_end, mask_clouds,
                                       min_aoi_coverage, im_download, l_im_path, run_pipeline, dataset_dict, site_name,
                                       im_classified_path, snow_cover_stats_path, dem, clf, feature_cols,
                                       figures_out_path, plot_results, verbose, delineate_snowline)
        print(' ')

    # ------------------------- #
    # --- 4. PlanetScope SR --- #
    # ------------------------- #
    if 4 in steps_to_run:

        print('----------')
        print('PlanetScope SR')
        print('----------')

        # -----Read surface reflectance image file names
        os.chdir(ps_im_path)
        im_fns = sorted(glob.glob('*SR*.tif'))

        # ----Mask clouds and cloud shadows in all images
        plot_results = False
        if mask_clouds:
            print('Masking images using cloud bitmask...')
            for im_fn in tqdm(im_fns):
                psp.planetscope_mask_image_pixels(ps_im_path, im_fn, ps_im_masked_path, save_outputs, plot_results)
        # read masked image file names
        os.chdir(ps_im_masked_path)
        im_masked_fns = sorted(glob.glob('*_mask.tif'))

        # -----Mosaic images captured within same hour
        print('Mosaicking images captured in the same hour...')
        if mask_clouds:
            psp.planetscope_mosaic_images_by_date(ps_im_masked_path, im_masked_fns, ps_im_mosaics_path, aoi_utm)
            print(' ')
        else:
            psp.planetscope_mosaic_images_by_date(ps_im_path, im_fns, ps_im_mosaics_path, aoi_utm)
            print(' ')

        # -----Load trained classifier and feature columns
        clf_fn = os.path.join(script_path, '..','inputs-outputs', 'PlanetScope_classifier_all_sites.joblib')
        clf = load(clf_fn)
        feature_cols_fn = os.path.join(script_path, '..','inputs-outputs', 'PlanetScope_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))
        dataset = 'PlanetScope'

        # -----Iterate over image mosaics
        # read image mosaic file names
        os.chdir(ps_im_mosaics_path)
        im_mosaic_fns = sorted(glob.glob('*.tif'))
        # create polygon(s) of the top and bottom 20th percentile elevations within the AOI
        polygons_top, polygons_bottom = psp.create_aoi_elev_polys(aoi_utm, dem)
        # loop through images
        for im_mosaic_fn in tqdm(im_mosaic_fns):

            # -----Determine image date from image mosaic file name
            im_date = im_mosaic_fn[0:4] + '-' + im_mosaic_fn[4:6] + '-' + im_mosaic_fn[6:8] + 'T' + im_mosaic_fn[
                                                                                                    9:11] + ':00:00'
            im_dt = np.datetime64(im_date)
            if verbose:
                print(im_date)

            # -----Open image mosaic
            im_da = xr.open_dataset(os.path.join(ps_im_mosaics_path, im_mosaic_fn))

            # -----Adjust radiometry
            im_adj, im_adj_method = psp.planetscope_adjust_image_radiometry(im_da, im_dt, polygons_top,
                                                                            polygons_bottom,
                                                                            dataset_dict, skip_clipped)
            if type(im_adj) is str:  # skip if there was an error in adjustment
                continue

            # -----Check if classified image already exists in file
            # check if classified image already exists in file
            im_classified_fn = im_date.replace('-', '').replace(':',
                                                                '') + '_' + site_name + '_' + dataset + '_classified.nc'
            if os.path.exists(os.path.join(im_classified_path, im_classified_fn)):
                if verbose:
                    print('Classified image already exists in file, loading...')
                im_classified = xr.open_dataset(os.path.join(im_classified_path, im_classified_fn))

                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
                # reproject to UTM
                im_classified = im_classified.rio.write_crs('EPSG:4326')
                im_classified = im_classified.rio.reproject('EPSG:' + epsg_utm)
            else:

                # -----Check that image mosaic covers at least min_aoi_coverage % of the AOI
                # Create dummy band for AOI masking comparison
                im_adj['aoi_mask'] = (
                ['time', 'y', 'x'], np.ones(np.shape(im_adj[dataset_dict[dataset]['RGB_bands'][0]].data)))
                im_aoi = im_adj.rio.clip(aoi_utm.geometry, im_adj.rio.crs)
                # Calculate the percentage of real values in the AOI
                perc_real_values_aoi = (
                            len(np.where(~np.isnan(np.ravel(im_aoi[dataset_dict[dataset]['RGB_bands'][0]].data)))[0])
                            / len(np.where(~np.isnan(np.ravel(im_aoi['aoi_mask'].data)))[0]))
                if perc_real_values_aoi < min_aoi_coverage:
                    if verbose:
                        print(f'Less than {min_aoi_coverage}% coverage of the AOI, skipping image...')
                        print(' ')
                    continue

                # -----Classify image
                im_classified = utils.classify_image(im_adj, clf, feature_cols, aoi_utm, dataset_dict,
                                                 dataset, im_classified_fn, im_classified_path, verbose)
                if type(im_classified) == str:
                    continue

            # -----Calculate snow cover stats
            # Check if snow cover stats already exists in file
            snow_cover_stats_fn = (im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset
                                   + '_snow_cover_stats.csv')
            if os.path.exists(os.path.join(snow_cover_stats_path, snow_cover_stats_fn)):
                # No need to load snow cover stats if it already exists
                continue
            else:
                # Calculate snow cover stats
                scs_df = utils.calculate_snow_cover_stats(dataset_dict, dataset, im_date, im_adj, im_classified, dem, aoi_utm, 
                                                          site_name, delineate_snowline, snow_cover_stats_fn, snow_cover_stats_path, 
                                                          figures_out_path, plot_results, verbose)
                plt.close()
            if verbose:
                print(' ')

    print('Done!')

if __name__ == '__main__':
    main()
