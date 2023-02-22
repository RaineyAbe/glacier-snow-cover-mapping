### Classify snow-covered area (SCA) in Sentinel-2, Landsat 8/9, and PlanetScope imagery: full pipeline
# Rainey Aberle
# Department of Geosciences, Boise State University
# 2022
#
### Requirements:
# - Area of Interest (AOI) shapefile: where snow will be classified in all available images.
# - Google Earth Engine (GEE) account: used to pull DEM over the AOI. Sign up for a free account [here](https://earthengine.google.com/new_signup/).
# - Digital elevation model (DEM) (_optional_): used to extract elevations over the AOI and for each snowline. If no DEM is provided, the ASTER Global DEM will be loaded through GEE.
#
### Outline:
#   0. Setup paths in directory, file locations, authenticate GEE -> MODIFY THIS SECTION
#   1. Sentinel-2 Top of Atmosphere (TOA) imagery: full pipeline
#   2. Sentinel-2 Surface Reflectance (SR) imagery: full pipeline
#   3. Landsat 8/9 Surface Reflectance (SR) imagery: full pipeline
#   4. PlanetScope Surface Reflectance (SR) imagery: full pipeline

# ----------------- #
# --- 0. Set up --- #
# ----------------- #

##### MODIFY HERE #####

# -----Paths in directory
site_name = 'Sperry'
# path to snow-cover-mapping/
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to AOI including the name of the shapefile
AOI_fn = base_path + '../study-sites/' + site_name + '/glacier_outlines/' + site_name + '_USGS_*.shp'
# path to DEM including the name of the tif file
# Note: set DEM_fn=None if you want to use the ASTER GDEM on Google Earth Engine
DEM_fn = base_path + '../study-sites/' + site_name + '/DEMs/' + site_name + '*_DEM*.tif'
# path for output images
out_path = base_path + '../study-sites/' + site_name + '/imagery/'
# path to PlanetScope images
# Note: set PS_im_path=None if not using PlanetScope
PS_im_path = out_path + 'PlanetScope/2016-2022/'
# path for output figures
figures_out_path = base_path + '../study-sites/' + site_name + '/figures/'
# define steps to run
# 1 = Sentinel-2 TOA; 2 = Sentinel-2 SR; 3 = Landsat 8/9 SR; 4 = PlanetScope SR
steps_to_run = [1, 2, 3, 4]

# -----Define image search filters
date_start = '2013-05-01'
date_end = '2023-01-01'
month_start = 5
month_end = 11
cloud_cover_max = 100

# -----Determine settings
plot_results = True # = True to plot figures of results for each image where applicable
skip_clipped = False # = True to skip images where bands appear "clipped", i.e. max blue SR < 0.8
crop_to_AOI = True # = True to crop images to AOI before calculating SCA
save_outputs = True # = True to save SCA images to file
save_figures = True # = True to save SCA output figures to file

#######################

# -----Import packages
import xarray as xr
import os
import numpy as np
import glob
from matplotlib import pyplot as plt, dates
import matplotlib
import rasterio as rio
import geopandas as gpd
import pandas as pd
import sys
import ee
import pickle
import time

# -----Set paths for output files
PS_im_masked_path = out_path + 'PlanetScope/masked/'
PS_im_mosaics_path = out_path + 'PlanetScope/mosaics/'
im_classified_path = out_path + 'classified/'
snowlines_path = out_path + 'snowlines/'

# -----Add path to functions
sys.path.insert(1, base_path+'functions/')
import pipeline_utils_REORG as f

# -----Load dataset dictionary
with open(base_path + 'inputs-outputs/datasets_characteristics.pkl', 'rb') as fn:
    dataset_dict = pickle.load(fn)

# -----Authenticate and initialize GEE
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# -----Load AOI and DEM
print('Loading AOI and DEM files...')
print(' ')
# load AOI as gpd.GeoDataFrame
AOI_fn = glob.glob(AOI_fn)[0]
AOI = gpd.read_file(AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs(4326)
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
# load DEM as Xarray DataSet
if DEM_fn==None:
    # query GEE for DEM
    DEM, AOI_UTM = f.query_GEE_for_DEM(AOI)
else:
    # reproject AOI to UTM
    AOI_UTM = AOI.to_crs(str(epsg_UTM))
    # load DEM as xarray DataSet
    DEM_fn = glob.glob(DEM_fn)[0]
    DEM = xr.open_dataset(DEM_fn)
    DEM = DEM.rename({'band_data': 'elevation'})
    # reproject the DEM to the optimal UTM zone
    DEM = DEM.rio.reproject(str('EPSG:'+epsg_UTM))

# ------------------------- #
# --- 1. Sentinel-2 TOA --- #
# ------------------------- #

if 1 in steps_to_run:
    print('Sentinel-2 TOA')
    print('----------')

    # -----Load trained classifier and feature columns
    clf_fn = base_path+'inputs-outputs/S2_TOA_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path+'inputs-outputs/S2_TOA_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))

    # -----Query GEE for imagery
    dataset = 'Sentinel2_TOA'
    im_list = f.query_GEE_for_Sentinel2(dataset, dataset_dict, site_name,
                                         AOI_UTM, date_start, date_end, month_start,
                                         month_end, cloud_cover_max)
    im_list_size = im_list.size().getInfo()

    # -----Loop through images
    if im_list_size==0: # check that images were found
        print('No images found to classify, quiting...')
    else:

        for i in range(0, im_list_size):

            # -----Select image by index
            im = ee.Image(ee.List(im_list).get(i))
            # get image time
            im_date = im.date().format(None, 'GMT').getInfo()
            print(' ')
            print(str(i+1)+'/'+str(im_list_size))
            print(im_date)

            # -----Check if classified image and snowline already exists in file
            im_classified_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_classified.nc'
            snowline_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_snowline.pkl'
            if os.path.exists(im_classified_path + im_classified_fn) & os.path.exists(snowlines_path + snowline_fn):

                print('Classified image already exists in file, loading...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                print('Snowline already exists in file, loading...')
                snowline_df = pd.read_pickle(snowlines_path + snowline_fn)

            else:
                # -----Convert image to xarray.Dataset
                res = dataset_dict[dataset]['resolution_m']
                im_xr = im.wx.to_xarray(scale=res, crs='EPSG:4326')
                # reproject to UTM CRS
                im_xr_UTM = im_xr.rio.reproject('EPSG:'+epsg_UTM)
                # replace no data values with NaN and account for image scalar
                bands = [band for band in dataset_dict[dataset]['bands'] if 'QA' not in band]
                for band in bands:
                    im_xr_UTM[band] = xr.where(im_xr_UTM[band] != dataset_dict[dataset]['no_data_value'],
                                               im_xr_UTM[band] / dataset_dict[dataset]['im_scalar'], np.nan)

                # -----Add NDSI band
                im_xr_UTM['NDSI'] = ((im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] - im_xr_UTM[dataset_dict[dataset]['NDSI'][1]])
                                     / (im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] + im_xr_UTM[dataset_dict[dataset]['NDSI'][1]]))

                # -----Classify image
                im_classified = f.classify_image(im_xr_UTM, clf, feature_cols, crop_to_AOI,
                                                 AOI_UTM, dataset, dataset_dict, site_name,
                                                 im_classified_fn, im_classified_path)
                if type(im_classified)==str:
                    continue

                # -----Delineate snowline(s)
                plot_results = True
                snowline_df = f.delineate_im_snowline(im_xr_UTM, im_classified, site_name, AOI_UTM, DEM,
                                                      dataset_dict, dataset, im_date, snowline_fn,
                                                      snowlines_path, figures_out_path, plot_results)

            print('Median snowline elevation: ' + str(snowline_df['snowlines_elevs_median'][0]) + ' m')

# ------------------------ #
# --- 2. Sentinel-2 SR --- #
# ------------------------ #

if 2 in steps_to_run:
    print('Sentinel-2 SR')
    print('----------')

    # -----Load trained classifier and feature columns
    clf_fn = base_path+'inputs-outputs/S2_SR_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path+'inputs-outputs/S2_SR_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))

    # -----Query GEE for imagery
    dataset = 'Sentinel2_SR'
    im_list = f.query_GEE_for_Sentinel2(dataset, dataset_dict, site_name,
                                           AOI_UTM, date_start, date_end, month_start,
                                           month_end, cloud_cover_max)
    im_list_size = im_list.size().getInfo()

    # -----Loop through images
    if im_list_size==0: # check that images were found
        print('No images found to classify, quiting...')
    else:

        for i in range(0, im_list_size):

            # -----Select image by index
            im = ee.Image(ee.List(im_list).get(i))
            # get image time
            im_date = im.date().format(None, 'GMT').getInfo()
            print(' ')
            print(str(i+1)+'/'+str(im_list_size))
            print(im_date)

            # -----Check if classified image and snowline already exists in file
            im_classified_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_classified.nc'
            snowline_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_snowline.pkl'
            if os.path.exists(im_classified_path + im_classified_fn) & os.path.exists(snowlines_path + snowline_fn):

                print('Classified image already exists in file, loading...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                print('Snowline already exists in file, loading...')
                snowline_df = pd.read_pickle(snowlines_path + snowline_fn)

            else:
                # -----Convert image to xarray.Dataset
                res = dataset_dict[dataset]['resolution_m']
                im_xr = im.wx.to_xarray(scale=res, crs='EPSG:4326')
                # reproject to UTM CRS
                im_xr_UTM = im_xr.rio.reproject('EPSG:'+epsg_UTM)
                # replace no data values with NaN and account for image scalar
                bands = [band for band in dataset_dict[dataset]['bands'] if 'QA' not in band]
                for band in bands:
                    im_xr_UTM[band] = xr.where(im_xr_UTM[band] != dataset_dict[dataset]['no_data_value'],
                                               im_xr_UTM[band] / dataset_dict[dataset]['im_scalar'], np.nan)

                # -----Add NDSI band
                im_xr_UTM['NDSI'] = ((im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] - im_xr_UTM[dataset_dict[dataset]['NDSI'][1]])
                                     / (im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] + im_xr_UTM[dataset_dict[dataset]['NDSI'][1]]))

                # -----Classify image
                im_classified = f.classify_image(im_xr_UTM, clf, feature_cols, crop_to_AOI,
                                                 AOI_UTM, dataset, dataset_dict, site_name,
                                                 im_classified_fn, im_classified_path)
                if type(im_classified)==str:
                    continue

                # -----Delineate snowline(s)
                plot_results = True
                snowline_df = f.delineate_im_snowline(im_xr_UTM, im_classified, site_name, AOI_UTM, DEM,
                                                      dataset_dict, dataset, im_date, snowline_fn,
                                                      snowlines_path, figures_out_path, plot_results)

            print('Median snowline elevation: ' + str(snowline_df['snowlines_elevs_median'][0]) + ' m')


# ------------------------- #
# --- 3. Landsat 8/9 SR --- #
# ------------------------- #

if 3 in steps_to_run:
    print('Landsat 8/9 SR')
    print('----------')

    # -----Load trained classifier and feature columns
    clf_fn = base_path+'inputs-outputs/L_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path+'inputs-outputs/L_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))

    # -----Query GEE for imagery
    dataset = 'Landsat'
    im_list = f.query_GEE_for_Landsat_SR(AOI_UTM, date_start, date_end, month_start, month_end,
                                      cloud_cover_max, site_name, dataset, dataset_dict, out_path)
    im_list_size = im_list.size().getInfo()

    # -----Loop through images
    if im_list_size==0: # check that images were found
        print('No images found to classify, quiting...')
    else:

        for i in range(0, im_list_size):

            # -----Select image by index
            im = ee.Image(ee.List(im_list).get(i))
            # get image time
            im_date = im.date().format(None, 'GMT').getInfo()
            print(' ')
            print(str(i+1)+'/'+str(im_list_size))
            print(im_date)

            # -----Check if classified image and snowline already exists in file
            im_classified_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_classified.nc'
            snowline_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_snowline.pkl'
            if os.path.exists(im_classified_path + im_classified_fn) & os.path.exists(snowlines_path + snowline_fn):

                print('Classified image already exists in file, loading...')
                im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
                print('Snowline already exists in file, loading...')
                snowline_df = pd.read_pickle(snowlines_path + snowline_fn)

            else:
                # -----Convert image to xarray.Dataset
                res = dataset_dict[dataset]['resolution_m']
                im_xr = im.wx.to_xarray(scale=res, crs='EPSG:4326')
                # reproject to UTM CRS
                im_xr_UTM = im_xr.rio.reproject('EPSG:'+epsg_UTM)
                # replace no data values with NaN and account for image scalar
                bands = [band for band in dataset_dict[dataset]['bands'] if 'QA' not in band]
                for band in bands:
                    im_xr_UTM[band] = xr.where(im_xr_UTM[band] != dataset_dict[dataset]['no_data_value'],
                                               im_xr_UTM[band] / dataset_dict[dataset]['im_scalar'], np.nan)

                # -----Add NDSI band
                im_xr_UTM['NDSI'] = ((im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] - im_xr_UTM[dataset_dict[dataset]['NDSI'][1]])
                                     / (im_xr_UTM[dataset_dict[dataset]['NDSI'][0]] + im_xr_UTM[dataset_dict[dataset]['NDSI'][1]]))

                # -----Classify image
                im_classified = f.classify_image(im_xr_UTM, clf, feature_cols, crop_to_AOI,
                                                 AOI_UTM, dataset, dataset_dict, site_name,
                                                 im_classified_fn, im_classified_path)
                if type(im_classified)==str:
                    continue

                # -----Delineate snowline(s)
                plot_results = True
                snowline_df = f.delineate_im_snowline(im_xr_UTM, im_classified, site_name, AOI_UTM, DEM,
                                                      dataset_dict, dataset, im_date, snowline_fn,
                                                      snowlines_path, figures_out_path, plot_results)

            print('Median snowline elevation: ' + str(snowline_df['snowlines_elevs_median'][0]) + ' m')


# ------------------------- #
# --- 4. PlanetScope SR --- #
# ------------------------- #
if 4 in steps_to_run:
    print('PlanetScope SR')
    print('----------')

    # -----Load trained classifier and feature columns
    clf_fn = base_path+'inputs-outputs/PS_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path+'inputs-outputs/PS_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))
    dataset = 'PlanetScope'

    # -----Read surface reflectance file names
    os.chdir(PS_im_path)
    im_fns = glob.glob('*SR*.tif')
    im_fns = sorted(im_fns) # sort chronologically
    plot_results = False

    # ----Mask clouds and cloud shadows in all images
    print('Masking images using cloud bitmask...')
    for i, im_fn in enumerate(im_fns):
        f.PS_mask_im_pixels(PS_im_path, im_fn, PS_im_masked_path, save_outputs, plot_results)

    # -----Mosaic images captured within same hour
    print('Mosaicking images captured in the same hour...')
    # read masked image file names
    os.chdir(PS_im_masked_path)
    im_masked_fns = glob.glob('*_mask.tif')
    im_masked_fns = sorted(im_masked_fns) # sort chronologically
    # mosaic images by date
    f.PS_mosaic_ims_by_date(PS_im_masked_path, im_masked_fns, PS_im_mosaics_path, AOI_UTM, plot_results)
    print(' ')

    # -----Adjust image radiometry
    # read mosaicked image file names
    os.chdir(PS_im_mosaics_path)
    im_mosaic_fns = glob.glob('*.tif')
    im_mosaic_fns = sorted(im_mosaic_fns)
    # create a polygon(s) of the top 20th percentile elevations within the AOI
    plot_results=False
    polygon_top, polygon_bottom, im_mosaic_fn, im_mosaic = f.create_AOI_elev_polys(AOI_UTM, PS_im_mosaics_path, im_mosaic_fns, DEM)
    # loop through images
    for i, im_mosaic_fn in enumerate(im_mosaic_fns):
        print(' ')
        print(str(i+1)+'/'+str(len(im_mosaic_fns)))

        # adjust radiometry
        im_adj, im_adj_method = f.PS_adjust_image_radiometry(im_mosaic_fn, PS_im_mosaics_path, polygon_top,
                                                             polygon_bottom, AOI_UTM, dataset_dict, dataset,
                                                             site_name, skip_clipped, plot_results)
        if type(im_adj)==str: # skip if there was an error in adjustment
            continue

        # -----Determiine image date
        im_date = im_mosaic_fn[0:8] + 'T' + im_mosaic_fn[9:11] + ':00:00'

        # -----Classify image
        im_classified_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_classified.nc'
        if os.path.exists(im_classified_path + im_classified_fn):
            print('Classified image already exists in file, loading...')
            im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
        else:
            im_classified = f.classify_image(im_adj, clf, feature_cols, crop_to_AOI,
                                             AOI_UTM, dataset, dataset_dict, site_name,
                                             im_classified_fn, im_classified_path)
        if type(im_classified)==str:
            continue

        # -----Delineate snowline(s)
        snowline_fn = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_snowline.pkl'
        if os.path.exists(snowlines_path + snowline_fn):
            print('Snowline already exists in file, loading...')
            snowline_df = pd.read_pickle(snowlines_path + snowline_fn)
        else:
            plot_results=True
            snowline_df = f.delineate_im_snowline(im_adj, im_classified, site_name, AOI_UTM, DEM,
                                                  dataset_dict, dataset, im_date, snowline_fn,
                                                  snowlines_path, figures_out_path, plot_results)
        print('Median snowline elevation: ' + str(snowline_df['snowlines_elevs_median'][0]) + ' m')
