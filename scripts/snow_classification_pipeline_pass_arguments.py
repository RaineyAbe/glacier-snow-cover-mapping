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
             If no DEM is provided, the ASTER Global DEM will be loaded through GEE.
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
import warnings

warnings.simplefilter("ignore")

# -----Parse user arguments
parser = argparse.ArgumentParser(description="snow_classification_pipeline with arguments passed by the user",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-site_name', default=None, type=str, help='Name of study site')
parser.add_argument('-base_path', default=None, type=str, help='Path in directory to "snow-cover-mapping/"')
parser.add_argument('-AOI_path', default=None, type=str, help='Path in directory to area of interest shapefile')
parser.add_argument('-AOI_fn', default=None, type=str, help='Area of interest file name (.shp)')
parser.add_argument('-DEM_path', default=None, type=str, help='(Optional) Path in directory to digital elevation model')
parser.add_argument('-DEM_fn', default=None, type=str, help='(Optional) Digital elevation model file name (.shp)')
parser.add_argument('-out_path', default=None, type=str, help='Path in directory where output images will be saved')
parser.add_argument('-figures_out_path', default=None, type=str, help='Path in directory where figures will be saved')
parser.add_argument('-date_start', default=None, type=str, help='Start date for image querying: "YYYY-MM-DD"')
parser.add_argument('-date_end', default=None, type=str, help='End date for image querying: "YYYY-MM-DD"')
parser.add_argument('-month_start', default=None, type=int, help='Start month for image querying, e.g. 5')
parser.add_argument('-month_end', default=None, type=int, help='End month for image querying, e.g. 10')
parser.add_argument('-cloud_cover_max', default=None, type=int, help='Max. cloud cover percentage in images, '
                                                                     'e.g. 50 = 50% maximum cloud coverage')
parser.add_argument('-mask_clouds', default=None, type=bool, help='Whether to mask clouds using the respective cloud '
                                                                  'cover masking product of each dataset')
parser.add_argument('-im_download', default=False, type=bool, help='Whether to download intermediary images. '
                                                                   'If im_download=False, but images over the AOI '
                                                                   'exceed the GEE limit, images must be '
                                                                   'downloaded regardless.')
parser.add_argument('-steps_to_run', default=None, nargs="+", type=int,
                    help='List of steps to be run, e.g. [1, 2, 3]. '
                         '1=Sentinel-2_TOA, 2=Sentinel-2_SR, 3=Landsat, 4=PlanetScope')
parser.add_argument('-verbose', action='store_true',
                    help='Whether to print details for each image at each processing step.')
args = parser.parse_args()

# -----Set user arguments as variables
site_name = args.site_name
base_path = args.base_path
AOI_path = args.AOI_path
AOI_fn = args.AOI_fn
DEM_path = args.DEM_path
if DEM_path == "None":
    DEM_path = None
DEM_fn = args.DEM_fn
if DEM_fn == "None":
    DEM_fn = None
out_path = args.out_path
figures_out_path = args.figures_out_path
date_start = args.date_start
date_end = args.date_end
month_start = args.month_start
month_end = args.month_end
cloud_cover_max = args.cloud_cover_max
mask_clouds = args.mask_clouds
im_download = args.im_download
steps_to_run = args.steps_to_run
verbose = args.verbose

# -----Determine image clipping & plotting settings
plot_results = True  # = True to plot figures of results for each image where applicable
skip_clipped = False  # = True to skip images where bands appear "clipped", i.e. max(blue) < 0.8
crop_to_AOI = True  # = True to crop images to AOI before calculating SCA
save_outputs = True  # = True to save SCAs and snowlines to file
save_figures = True  # = True to save output figures to file

print(site_name)
print(' ')

# -----Set paths for output files
# Note: set PS_im_path=None if not using PlanetScope
PS_im_path = out_path + 'PlanetScope/raw_images/'
S2_TOA_im_path = out_path + 'Sentinel-2_TOA/'
S2_SR_im_path = out_path + 'Sentinel-2_SR/'
L_im_path = out_path + 'Landsat/'
PS_im_masked_path = out_path + 'PlanetScope/masked/'
PS_im_mosaics_path = out_path + 'PlanetScope/mosaics/'
im_classified_path = out_path + 'classified/'
snowlines_path = out_path + 'snowlines/'

# -----Add path to functions
sys.path.insert(1, base_path + 'functions/')
import pipeline_utils as f

# -----Load dataset dictionary
dataset_dict = json.load(open(base_path + 'inputs-outputs/datasets_characteristics.json'))

# -----Authenticate and initialize GEE
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# -----Load AOI as gpd.GeoDataFrame
AOI = gpd.read_file(AOI_path + AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs('EPSG:4326')
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
# grab the optimal UTM zone EPSG code
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
# reproject AOI to the optimal UTM zone
AOI_UTM = AOI.to_crs('EPSG:' + epsg_UTM)

# -----Load DEM as Xarray DataSet
if DEM_fn is None:
    # query GEE for DEM
    DEM = f.query_gee_for_dem(AOI_UTM, base_path, site_name, DEM_path)
else:
    # load DEM as xarray DataSet
    DEM = xr.open_dataset(DEM_path + DEM_fn)
    DEM = DEM.rename({'band_data': 'elevation'})
    # reproject the DEM to the optimal UTM zone
    DEM = DEM.rio.reproject('EPSG:' + str(epsg_UTM))
    DEM = DEM.rio.write_crs('EPSG:' + str(epsg_UTM))
# remove unnecessary data (possible extra bands from ArcticDEM or other DEM)
if len(np.shape(DEM.elevation.data)) > 2:
    DEM['elevation'] = DEM.elevation[0]

# ------------------------- #
# --- 1. Sentinel-2 TOA --- #
# ------------------------- #
if 1 in steps_to_run:
    print('----------')
    print('Sentinel-2 TOA')
    print('----------')

    # -----Query GEE for imagery (and download to S2_TOA_im_path if necessary)
    dataset = 'Sentinel-2_TOA'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, AOI_UTM, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, S2_TOA_im_path, im_download)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Sentinel-2_TOA_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Sentinel-2_TOA_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quitting...')
    else:
        print('Classifying images...')
        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]
            if verbose:
                print(im_date)

            # -----Adjust image for image scalar and no data values
            # replace no data values with NaN and account for image scalar
            crs = im_xr.rio.crs.to_epsg()
            if np.nanmean(im_xr['B2']) > 1e3:
                im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan,
                                 im_xr / dataset_dict[dataset]['image_scalar'])
            else:
                im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan, im_xr)
            # add NDSI band
            im_xr['NDSI'] = (
                    (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] - im_xr[dataset_dict[dataset]['NDSI_bands'][1]])
                    / (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] +
                       im_xr[dataset_dict[dataset]['NDSI_bands'][1]]))
            im_xr.rio.write_crs('EPSG:' + str(crs), inplace=True)

            # -----Classify image
            # check if classified image already exists in file
            im_classified_fn = im_date.replace('-', '').replace(':',
                                                                '') + '_' + site_name + '_' + dataset + '_classified.nc'
            if os.path.exists(im_classified_path + im_classified_fn):
                if verbose:
                    print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                if verbose:
                    print('Snowline already exists in file, continuing...')
                    print(' ')
                continue  # no need to load snowline if it already exists
            else:
                snowline_df = f.delineate_snowline(im_xr, im_classified, site_name, AOI_UTM, DEM, dataset_dict, dataset,
                                                   im_date, snowline_fn, snowlines_path, figures_out_path, plot_results,
                                                   verbose)
                if verbose:
                    print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
            if verbose:
                print(' ')
    print(' ')

# ------------------------ #
# --- 2. Sentinel-2 SR --- #
# ------------------------ #
if 2 in steps_to_run:

    print('----------')
    print('Sentinel-2 SR')
    print('----------')

    # -----Query GEE for imagery and download to S2_SR_im_path if necessary
    dataset = 'Sentinel-2_SR'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, AOI_UTM, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, S2_SR_im_path, im_download)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Sentinel-2_SR_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Sentinel-2_SR_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quiting...')
    else:
        print('Classifying images...')
        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]
            if verbose:
                print(im_date)

            # -----Adjust image for image scalar and no data values
            # replace no data values with NaN and account for image scalar
            crs = im_xr.rio.crs.to_epsg()
            if np.nanmean(im_xr['B2']) > 1e3:
                im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan,
                                 im_xr / dataset_dict[dataset]['image_scalar'])
            else:
                im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan, im_xr)
            # add NDSI band
            im_xr['NDSI'] = (
                    (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] - im_xr[dataset_dict[dataset]['NDSI_bands'][1]])
                    / (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] +
                       im_xr[dataset_dict[dataset]['NDSI_bands'][1]]))
            im_xr.rio.write_crs('EPSG:' + str(crs), inplace=True)

            # -----Classify image
            # check if classified image already exists in file
            im_classified_fn = im_date.replace('-', '').replace(':',
                                                                '') + '_' + site_name + '_' + dataset + '_classified.nc'
            if os.path.exists(im_classified_path + im_classified_fn):
                if verbose:
                    print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                if verbose:
                    print('Snowline already exists in file, continuing...')
                continue  # no need to load snowline if it already exists
            else:
                snowline_df = f.delineate_snowline(im_xr, im_classified, site_name, AOI_UTM, DEM, dataset_dict, dataset,
                                                   im_date, snowline_fn, snowlines_path, figures_out_path, plot_results,
                                                   verbose)
                if verbose:
                    print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
            if verbose:
                print(' ')
    print(' ')

# ------------------------- #
# --- 3. Landsat 8/9 SR --- #
# ------------------------- #
if 3 in steps_to_run:

    print('----------')
    print('Landsat 8/9 SR')
    print('----------')

    # -----Query GEE for imagery (and download to L_im_path if necessary)
    dataset = 'Landsat'
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, AOI_UTM, date_start, date_end, month_start, month_end,
                                      cloud_cover_max, mask_clouds, L_im_path, im_download)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Landsat_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Landsat_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quitting...')
    else:
        print('Classifying images...')
        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]
            if verbose:
                print(im_date)

            # -----Adjust image for image scalar and no data values
            # replace no data values with NaN and account for image scalar
            crs = im_xr.rio.crs.to_epsg()
            # add NDSI band
            im_xr['NDSI'] = (
                    (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] - im_xr[dataset_dict[dataset]['NDSI_bands'][1]])
                    / (im_xr[dataset_dict[dataset]['NDSI_bands'][0]]
                       + im_xr[dataset_dict[dataset]['NDSI_bands'][1]]))
            im_xr.rio.write_crs('EPSG:' + str(crs), inplace=True)

            # -----Classify image
            # check if classified image already exists in file
            im_classified_fn = im_date.replace('-', '').replace(':',
                                                                '') + '_' + site_name + '_' + dataset + '_classified.nc'
            if os.path.exists(im_classified_path + im_classified_fn):
                if verbose:
                    print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                if verbose:
                    print('Snowline already exists in file, continuing...')
                continue  # no need to load snowline if it already exists
            else:
                snowline_df = f.delineate_snowline(im_xr, im_classified, site_name, AOI_UTM, DEM, dataset_dict, dataset,
                                                   im_date, snowline_fn, snowlines_path, figures_out_path, plot_results,
                                                   verbose)
                if verbose:
                    print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
            if verbose:
                print(' ')
    print(' ')

# ------------------------- #
# --- 4. PlanetScope SR --- #
# ------------------------- #
if 4 in steps_to_run:

    print('----------')
    print('PlanetScope SR')
    print('----------')

    # -----Read surface reflectance image file names
    os.chdir(PS_im_path)
    im_fns = sorted(glob.glob('*SR*.tif'))

    # ----Mask clouds and cloud shadows in all images
    plot_results = False
    if mask_clouds:
        print('Masking images using cloud bitmask...')
        for im_fn in tqdm(im_fns):
            f.planetscope_mask_image_pixels(PS_im_path, im_fn, PS_im_masked_path, save_outputs, plot_results)
    # read masked image file names
    os.chdir(PS_im_masked_path)
    im_masked_fns = sorted(glob.glob('*_mask.tif'))

    # -----Mosaic images captured within same hour
    print('Mosaicking images captured in the same hour...')
    if mask_clouds:
        f.planetscope_mosaic_images_by_date(PS_im_masked_path, im_masked_fns, PS_im_mosaics_path, AOI_UTM)
        print(' ')
    else:
        f.planetscope_mosaic_images_by_date(PS_im_path, im_fns, PS_im_mosaics_path, AOI_UTM)
        print(' ')

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/PlanetScope_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/PlanetScope_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))
    dataset = 'PlanetScope'

    # -----Adjust image radiometry
    # read mosaicked image file names
    os.chdir(PS_im_mosaics_path)
    im_mosaic_fns = sorted(glob.glob('*.tif'))
    # create polygon(s) of the top and bottom 20th percentile elevations within the AOI
    polygons_top, polygons_bottom = f.create_aoi_elev_polys(AOI_UTM, DEM)
    # loop through images
    for im_mosaic_fn in tqdm(im_mosaic_fns):

        # -----Open image mosaic
        im_da = xr.open_dataset(PS_im_mosaics_path + im_mosaic_fn)
        # determine image date from image mosaic file name
        im_date = im_mosaic_fn[0:4] + '-' + im_mosaic_fn[4:6] + '-' + im_mosaic_fn[6:8] + 'T' + im_mosaic_fn[
                                                                                                9:11] + ':00:00'
        im_dt = np.datetime64(im_date)
        if verbose:
            print(im_date)

        # -----Adjust radiometry
        im_adj, im_adj_method = f.planetscope_adjust_image_radiometry(im_da, im_dt, polygons_top, polygons_bottom,
                                                                      dataset_dict, skip_clipped)
        if type(im_adj) == str:  # skip if there was an error in adjustment
            continue

        # -----Classify image
        im_classified_fn = im_date.replace('-', '').replace(':',
                                                            '') + '_' + site_name + '_' + dataset + '_classified.nc'
        if os.path.exists(im_classified_path + im_classified_fn):
            if verbose:
                print('Classified image already exists in file, loading...')
            im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
            # remove no data values
            im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
        else:
            im_classified = f.classify_image(im_adj, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                             dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
        if type(im_classified) == str:
            continue

        # -----Delineate snowline(s)
        # check if snowline already exists in file
        snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
        if os.path.exists(snowlines_path + snowline_fn):
            if verbose:
                print('Snowline already exists in file, skipping...')
        else:
            plot_results = True
            snowline_df = f.delineate_snowline(im_adj, im_classified, site_name, AOI_UTM, DEM, dataset_dict, dataset, im_date,
                                               snowline_fn, snowlines_path, figures_out_path, plot_results, verbose)
            if verbose:
                print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
        if verbose:
            print(' ')
    print(' ')

print('Done!')
