"""
Estimate glacier snow cover in Sentinel-2, Landsat 8/9, and/or PlanetScope imagery: full pipeline
Rainey Aberle
Department of Geosciences, Boise State University
2023

Requirements:
- Area of Interest (AOI) shapefile: where snow will be classified in all available images.
- Google Earth Engine (GEE) account: used to query imagery and DEM if no other DEM is specified.
                                     Sign up for a free account [here](https://earthengine.google.com/new_signup/).
- (Optional) Digital elevation model (DEM): used to extract elevations and snow-covered elevations.
             If no DEM is provided, the DEM will be loaded through GEE. Areas with ArcticDEM coverage will use the
             ArcticDEM Mosaic. Otherwise, the NASADEM will be used.
- (Optional) Pre-downloaded PlanetScope images. Download images using Planet Explorer (planet.com/explorer) or
             programmatically using the notebook: snow-cover-mapping/notebooks/download_PlanetScope_images.ipynb.

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

##### MODIFY HERE #####

# -----Paths in directory
site_name = 'RGI60-02.16674'
# path to snow-cover-mapping/ - Make sure you include a "/" at the end
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to AOI
aoi_path = '/Users/raineyaberle/Google Drive/My Drive/Research/CryoGARS-Glaciology/Advising/student-research/Alexandra-Friel/snow_cover_mapping_application/study-sites/' + site_name + '/AOIs/'
# AOI file name including file extension
aoi_fn = site_name + '_outline.shp'
# path to dem
dem_path = aoi_path + '../DEMs/'
# dem file name including file extension
# Note: set dem_fn=None if you want to use dem available via Google Earth Engine
dem_fn = None
# path for output images
out_path = aoi_path + '../imagery/'
# path to PlanetScope images
# Note: set ps_im_path=None if not using PlanetScope
ps_im_path = out_path + 'PlanetScope/raw_images/'
# path for output figures
figures_out_path = aoi_path + '../figures/'

# -----Define steps to run
# Note: 1=Sentinel-2_TOA, 2=Sentinel-2_SR, 3=Landsat, 4=PlanetScope
# Enclose steps in brackets, e.g. steps_to_run = [1,2]
steps_to_run = [1]

# -----Define image search filters
date_start = '2013-05-01'
date_end = '2022-12-01'
month_start = 5
month_end = 11
cloud_cover_max = 70

# -----Determine whether to print details for each image at each processing step
verbose = True

# -----Determine whether to mask clouds using the respective cloud masking data products
# NOTE: Cloud mask products anecdotally are less accurate over glacierized/snow-covered surfaces.
# If the cloud masks are consistently masking large regions or your study site, I suggest setting mask_clouds = False
mask_clouds = True

# -----Determine image download, clipping & plotting settings
# Note: if im_download = False, but images over the AOI exceed GEE limit,
# images must be downloaded regardless.
im_download = True  # = True to download all satellite images by default
plot_results = True  # = True to plot figures of results for each image where applicable
skip_clipped = False  # = True to skip images where bands appear "clipped", i.e. max(blue) < 0.8
crop_to_aoi = True  # = True to crop images to AOI before calculating SCA
save_outputs = True  # = True to save SCAs and snowlines to file
save_figures = True  # = True to save output figures to file

#######################

print(site_name)
print(' ')

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
import dask.bag as db
from dask.diagnostics import ProgressBar
import warnings
warnings.simplefilter("ignore")

# -----Set paths for output files
s2_toa_im_path = os.path.join(out_path, 'Sentinel-2_TOA')
s2_sr_im_path = os.path.join(out_path, 'Sentinel-2_SR')
l_im_path = os.path.join(out_path, 'Landsat')
ps_im_masked_path = os.path.join(out_path, 'PlanetScope', 'masked')
ps_im_mosaics_path = os.path.join(out_path, 'PlanetScope', 'mosaics')
im_classified_path = os.path.join(out_path, 'classified')
snowlines_path = os.path.join(out_path, 'snowlines')

# -----Add path to functions
sys.path.insert(1, os.path.join(base_path, 'functions'))
import pipeline_utils as f

# -----Load dataset dictionary
dataset_dict_fn = os.path.join(base_path, 'inputs-outputs', 'datasets_characteristics.json')
dataset_dict = json.load(open(dataset_dict_fn))

# -----Authenticate and initialize GEE
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# -----Load AOI as gpd.GeoDataFrame
aoi = gpd.read_file(os.path.join(aoi_path, aoi_fn))
# reproject the AOI to WGS84 to solve for the optimal utm zone
aoi_wgs = aoi.to_crs('EPSG:4326')
aoi_wgs_centroid = [aoi_wgs.geometry[0].centroid.xy[0][0],
                    aoi_wgs.geometry[0].centroid.xy[1][0]]
# grab the optimal utm zone EPSG code
epsg_utm = f.convert_wgs_to_utm(aoi_wgs_centroid[0], aoi_wgs_centroid[1])
print('Optimal UTM CRS = EPSG:' + str(epsg_utm))
# reproject AOI to the optimal utm zone
aoi_utm = aoi.to_crs('EPSG:'+epsg_utm)

# -----Load dem as Xarray DataSet
if dem_fn is None:
    # query GEE for dem
    dem = f.query_gee_for_dem(aoi_utm, base_path, site_name, dem_path)
else:
    # load dem as xarray DataSet
    dem = xr.open_dataset(os.path.join(dem_path, dem_fn))
    dem = dem.rename({'band_data': 'elevation'})
    # set no data values to NaN
    dem = xr.where((dem > 1e38) | (dem <= -9999), np.nan, dem)
    # reproject the dem to the optimal utm zone
    dem = dem.rio.reproject('EPSG:'+str(epsg_utm)).rio.write_crs('EPSG:'+str(epsg_utm))


# ------------------------- #
# --- 1. Sentinel-2 TOA --- #
# ------------------------- #
if 1 in steps_to_run:
    print('----------')
    print('Sentinel-2 TOA')
    print('----------')

    # -----Query GEE for imagery (and download to s2_toa_im_path if necessary)
    dataset = 'Sentinel-2_TOA'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, aoi_utm, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, s2_toa_im_path, im_download)

    # -----Check whether images were found
    if type(im_list) == str:
        print('No images found to classify, quitting...')
    else:

        # -----Load trained classifier and feature columns
        clf_fn = os.path.join(base_path, 'inputs-outputs', 'Sentinel-2_TOA_classifier_all_sites.joblib')
        clf = load(clf_fn)
        feature_cols_fn = os.path.join(base_path, 'inputs-outputs', 'Sentinel-2_TOA_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))

        # -----Apply pipeline to list of images
        # Convert list of images to dask bag
        im_bag = db.from_sequence(im_list)
        # Create processor with appropriate function arguments
        def create_processor(im_xr):
            snowline_df = f.apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                          snowlines_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                                          crop_to_aoi, figures_out_path, plot_results, verbose)
            return snowline_df
        # Apply batch processing
        with ProgressBar():
            # prepare bag for mapping
            im_bag_results = im_bag.map(create_processor)
            im_bag_results.compute()


# ------------------------ #
# --- 2. Sentinel-2 SR --- #
# ------------------------ #
if 2 in steps_to_run:

    print('----------')
    print('Sentinel-2 SR')
    print('----------')

    # -----Query GEE for imagery and download to s2_sr_im_path if necessary
    dataset = 'Sentinel-2_SR'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, aoi_utm, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, s2_sr_im_path, im_download)

    # -----Check whether images were found
    if type(im_list) == str:
        print('No images found to classify, quitting...')
    else:

        # -----Load trained classifier and feature columns
        clf_fn = os.path.join(base_path, 'inputs-outputs', 'Sentinel-2_SR_classifier_all_sites.joblib')
        clf = load(clf_fn)
        feature_cols_fn = os.path.join(base_path, 'inputs-outputs', 'Sentinel-2_SR_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))

        # -----Apply pipeline to list of images
        # Convert list of images to dask bag
        im_bag = db.from_sequence(im_list)
        # Create processor with appropriate function arguments
        def create_processor(im_xr):
            snowline_df = f.apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                          snowlines_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                                          crop_to_aoi, figures_out_path, plot_results, verbose)
            return snowline_df
        # Apply batch processing
        with ProgressBar():
            # prepare bag for mapping
            im_bag_results = im_bag.map(create_processor)
            im_bag_results.compute()

# ------------------------- #
# --- 3. Landsat 8/9 SR --- #
# ------------------------- #
if 3 in steps_to_run:

    print('----------')
    print('Landsat 8/9 SR')
    print('----------')

    # -----Query GEE for imagery (and download to l_im_path if necessary)
    dataset = 'Landsat'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, aoi_utm, date_start, date_end, month_start, month_end,
                                      cloud_cover_max, mask_clouds, l_im_path, im_download)

    # -----Check whether images were found
    if type(im_list) == str:
        print('No images found to classify, quitting...')
    else:

        # -----Load trained classifier and feature columns
        clf_fn = os.path.join(base_path, 'inputs-outputs', 'Landsat_classifier_all_sites.joblib')
        clf = load(clf_fn)
        feature_cols_fn = os.path.join(base_path, 'inputs-outputs', 'Landsat_feature_columns.json')
        feature_cols = json.load(open(feature_cols_fn))

        # -----Apply pipeline to list of images
        # Convert list of images to dask bag
        im_bag = db.from_sequence(im_list)
        # Create processor with appropriate function arguments
        def create_processor(im_xr):
            snowline_df = f.apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                          snowlines_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                                          crop_to_aoi, figures_out_path, plot_results, verbose)
            return snowline_df
        # Apply batch processing
        with ProgressBar():
            # prepare bag for mapping
            im_bag_results = im_bag.map(create_processor)
            im_bag_results.compute()


# ------------------------- #
# --- 4. PlanetScope SR --- #
# ------------------------- #
if 4 in steps_to_run:

    print('----------')
    print('PlanetScope SR')
    print('----------')

    # -----Read surface reflectance image file names
    if not ps_im_path:
        print('Variable ps_im_path must be specified to run the PlanetScope classification pipeline, exiting...')
    else:

        dataset = 'PlanetScope'

        # -----Read surface reflectance image file names
        os.chdir(ps_im_path)
        im_fns = sorted(glob.glob('*SR*.tif'))

        # ----Mask clouds and cloud shadows in all images
        plot_results = False
        if mask_clouds:
            print('Masking images using cloud bitmask...')
            for im_fn in tqdm(im_fns):
                f.planetscope_mask_image_pixels(ps_im_path, im_fn, ps_im_masked_path, save_outputs, plot_results)
        # read masked image file names
        os.chdir(ps_im_masked_path)
        im_masked_fns = sorted(glob.glob('*_mask.tif'))

        # -----Mosaic images captured within same hour
        print('Mosaicking images captured in the same hour...')
        if mask_clouds:
            f.planetscope_mosaic_images_by_date(ps_im_masked_path, im_masked_fns, ps_im_mosaics_path, aoi_utm)
            print(' ')
        else:
            f.planetscope_mosaic_images_by_date(ps_im_path, im_fns, ps_im_mosaics_path, aoi_utm)
            print(' ')

            # -----Adjust image radiometry
            im_adj_list = []
            # read mosaicked image file names
            os.chdir(ps_im_mosaics_path)
            im_mosaic_fns = sorted(glob.glob('*.tif'))
            # create polygon(s) of the top and bottom 20th percentile elevations within the aoi
            polygons_top, polygons_bottom = f.create_aoi_elev_polys(aoi_utm, dem)
            # loop through images
            for im_mosaic_fn in tqdm(im_mosaic_fns):

                # -----Open image mosaic
                im_da = xr.open_dataset(ps_im_mosaics_path + im_mosaic_fn)
                # determine image date from image mosaic file name
                im_date = im_mosaic_fn[0:4] + '-' + im_mosaic_fn[4:6] + '-' + im_mosaic_fn[6:8] + 'T' + im_mosaic_fn[
                                                                                                        9:11] + ':00:00'
                im_dt = np.datetime64(im_date)
                print(im_date)

                # -----Adjust radiometry
                im_adj, im_adj_method = f.planetscope_adjust_image_radiometry(im_da, im_dt, polygons_top, polygons_bottom,
                                                                              dataset_dict, skip_clipped)
                if type(im_adj) == str:  # skip if there was an error in adjustment
                    continue
                else:
                    im_adj_list.append(im_adj)

            # -----Load trained classifier and feature columns
            clf_fn = os.path.join(base_path, 'inputs-outputs', 'PlanetScope_classifier_all_sites.joblib')
            clf = load(clf_fn)
            feature_cols_fn = os.path.join(base_path, 'inputs-outputs', 'PlanetScope_feature_columns.json')
            feature_cols = json.load(open(feature_cols_fn))

            # -----Apply pipeline to list of images
            # Convert list of images to dask bag
            im_bag = db.from_sequence(im_list)
            # Create processor with appropriate function arguments
            def create_processor(im_xr):
                snowline_df = f.apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                              snowlines_path,
                                                              aoi_utm, dem, epsg_utm, clf, feature_cols, crop_to_aoi,
                                                              figures_out_path,
                                                              plot_results, verbose)
                return snowline_df
            # Apply batch processing
            with ProgressBar():
                # prepare bag for mapping
                im_bag_results = im_bag.map(create_processor)
                im_bag_results.compute()

print('Done!')
