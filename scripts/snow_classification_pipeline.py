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

##### MODIFY HERE #####

# -----Paths in directory
site_name = 'Hidden'
# path to snow-cover-mapping/ - Make sure you include a "/" at the end
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to AOI including the name of the shapefile
AOI_path = '/Users/raineyaberle/Google Drive/My Drive/Research/PhD/snow_cover_mapping/study-sites/' + site_name + '/AOIs/'
# AOI file name
AOI_fn = 'Hidden_RGI_outline.shp'
# path to DEM including the name of the tif file
# Note: set DEM_path==None and DEM_fn=None if you want to use the ASTER GDEM via Google Earth Engine
DEM_path = AOI_path + '../DEMs/'
# DEM file name
DEM_fn = None
# path for output images
out_path = AOI_path + '../imagery/'
# path to PlanetScope images
# Note: set PS_im_path=None if not using PlanetScope
PS_im_path = out_path + 'PlanetScope/raw_images/'
# path for output figures
figures_out_path = AOI_path + '../figures/'

# -----Define steps to run
# Note: 1=Sentinel-2_TOA, 2=Sentinel-2_SR, 3=Landsat, 4=PlanetScope
# Enclose steps in brackets, e.g. steps_to_run = [1,2]
steps_to_run = [3]

# -----Define image search filters
date_start = '2013-05-01'
date_end = '2022-11-01'
month_start = 5
month_end = 11
cloud_cover_max = 70

# -----Determine whether to mask clouds using the respective cloud masking data products
# NOTE: Cloud mask products anecdotally are less accurate over glacierized/snow-covered surfaces.
# If the cloud masks are consistently masking large regions or your study site, I suggest setting mask_clouds = False
mask_clouds = True

# -----Determine image clipping & plotting settings
plot_results = True  # = True to plot figures of results for each image where applicable
skip_clipped = False  # = True to skip images where bands appear "clipped", i.e. max(blue) < 0.8
crop_to_AOI = True  # = True to crop images to AOI before calculating SCA
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
import geedim as gd
import json
from tqdm.auto import tqdm
from joblib import load

# -----Set paths for output files
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
gd.Initialize()

# -----Load AOI as gpd.GeoDataFrame
AOI = gpd.read_file(AOI_path + AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs('EPSG:4326')
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
# grab the optimal UTM zone EPSG code
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
print('Optimal UTM CRS = EPSG:' + str(epsg_UTM))

# -----Load DEM as xarray.DataSet
if DEM_fn is None:
    # set DEM path if not defined
    DEM_path = AOI_path + '../DEMs/'
    # query GEE for DEM
    DEM, AOI_UTM = f.query_gee_for_dem(AOI, base_path, site_name, DEM_path)
else:
    # reproject AOI to UTM
    AOI_UTM = AOI.to_crs('EPSG:' + str(epsg_UTM))
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
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, AOI, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, S2_TOA_im_path)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Sentinel-2_TOA_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Sentinel-2_TOA_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quiting...')
    else:

        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]

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
                print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                print('Snowline already exists in file, continuing...')
                print(' ')
                continue  # no need to load snowline if it already exists
            else:
                plot_results = True
                # create directory for figures if it doesn't already exist
                if (not os.path.exists(figures_out_path)) & plot_results:
                    os.mkdir(figures_out_path)
                    print('Created directory for output figures: ' + figures_out_path)
                snowline_df = f.delineate_image_snowline(im_xr, im_classified, site_name, AOI_UTM, dataset_dict,
                                                         dataset,
                                                         im_date, snowline_fn, snowlines_path, figures_out_path,
                                                         plot_results)
                print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
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
    im_list = f.query_gee_for_imagery(dataset_dict, dataset, AOI, date_start, date_end, month_start,
                                      month_end, cloud_cover_max, mask_clouds, S2_SR_im_path)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Sentinel-2_SR_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Sentinel-2_SR_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quiting...')
    else:

        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]

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
                print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                print('Snowline already exists in file, continuing...')
                continue  # no need to load snowline if it already exists
            else:
                plot_results = True
                # create directory for figures if it doesn't already exist
                if (not os.path.exists(figures_out_path)) & plot_results:
                    os.mkdir(figures_out_path)
                    print('Created directory for output figures: ' + figures_out_path)
                snowline_df = f.delineate_image_snowline(im_xr, im_classified, site_name, AOI_UTM, dataset_dict,
                                                         dataset,
                                                         im_date, snowline_fn, snowlines_path, figures_out_path,
                                                         plot_results)
                print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
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
                                      cloud_cover_max, mask_clouds, L_im_path)

    # -----Load trained classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/Landsat_classifier_all_sites.joblib'
    clf = load(clf_fn)
    feature_cols_fn = base_path + 'inputs-outputs/Landsat_feature_columns.json'
    feature_cols = json.load(open(feature_cols_fn))

    # -----Loop through images
    if type(im_list) == str:  # check that images were found
        print('No images found to classify, quitting...')
    else:

        for i in tqdm(range(0, len(im_list))):

            # -----Subset image using loop index
            im_xr = im_list[i]
            im_date = str(im_xr.time.data[0])[0:19]

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
                print('Classified image already exists in file, continuing...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                # remove no data values
                im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
            else:
                # classify image
                im_classified = f.classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                                 dataset_dict, dataset, im_classified_fn, im_classified_path)
                if type(im_classified) == str:  # skip if error in classification
                    continue

            # -----Delineate snowline(s)
            # check if snowline already exists in file
            snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
            if os.path.exists(snowlines_path + snowline_fn):
                print('Snowline already exists in file, continuing...')
                continue  # no need to load snowline if it already exists
            else:
                plot_results = True
                # create directory for figures if it doesn't already exist
                if (not os.path.exists(figures_out_path)) & plot_results:
                    os.mkdir(figures_out_path)
                    print('Created directory for output figures: ' + figures_out_path)
                snowline_df = f.delineate_image_snowline(im_xr, im_classified, site_name, AOI_UTM, dataset_dict,
                                                         dataset,
                                                         im_date, snowline_fn, snowlines_path, figures_out_path,
                                                         plot_results)
                print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
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
            print('Classified image already exists in file, loading...')
            im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
            # remove no data values
            im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
        else:
            im_classified = f.classify_image(im_adj, clf, feature_cols, crop_to_AOI, AOI_UTM, DEM,
                                             dataset_dict, dataset, im_classified_fn, im_classified_path)
        if type(im_classified) == str:
            continue

        # -----Delineate snowline(s)
        plot_results = True
        # create directory for figures if it doesn't already exist
        if (not os.path.exists(figures_out_path)) & plot_results:
            os.mkdir(figures_out_path)
            print('Created directory for output figures: ' + figures_out_path)
        # check if snowline already exists in file
        snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
        if os.path.exists(snowlines_path + snowline_fn):
            print('Snowline already exists in file, skipping...')
        else:
            snowline_df = f.delineate_image_snowline(im_adj, im_classified, site_name, AOI_UTM, dataset_dict, dataset,
                                                     im_date, snowline_fn, snowlines_path, figures_out_path,
                                                     plot_results)
            print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))
        print(' ')
    print(' ')

print('Done!')
