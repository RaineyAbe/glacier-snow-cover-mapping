# Classify snow-covered area (SCA) in Sentinel-2 surface reflectance imagery: full pipeline
# Rainey Aberle
# Department of Geosciences, Boise State University
# 2022
#
# Requirements:
# - Area of Interest (AOI) shapefile: where snow will be classified in all available images.
# - Google Earth Engine (GEE) account: used to pull DEM over the AOI. Sign up for a free account [here](https://earthengine.google.com/new_signup/).
#
# Outline:
# 0. Setup paths in directory, file locations, authenticate GEE - _modify this section!_
# 1. Load images over the AOI
# 2. Classify SCA and use the snow elevations distribution to estimate the seasonal snowline
# 3. Delineate snowlines using classified images.


# ---------------- #
# --- 0. SETUP --- #
# ---------------- #

# -----Paths in directory
site_name = 'LemonCreek'
# path to snow-cover-mapping/
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to AOI including the name of the shapefile
AOI_fn = base_path + '../study-sites/' + site_name + '/glacier_outlines/' + site_name + '_USGS_*.shp'
# path to DEM including the name of the tif file
# Note: set DEM_fn=None if you want to use the ASTER GDEM on Google Earth Engine
DEM_fn = base_path + '../study-sites/' + site_name + '/DEMs/' + site_name + '*_DEM_filled.tif'
# path for output images
out_path = base_path + '../study-sites/' + site_name + '/imagery/Landsat/'
# path for output figures
figures_out_path = base_path + '../study-sites/' + site_name + '/figures/'

# -----Define image search filters
date_start = '2019-01-01'
date_end = '2019-12-01'
month_start = 5
month_end = 10
cloud_cover_max = 100

# -----Determine settings
plot_results = True # = True to plot figures of results for each image where applicable
skip_clipped = False # = True to skip images where bands appear "clipped", i.e. max blue SR < 0.8
crop_to_AOI = True # = True to crop images to AOI before calculating SCA
save_outputs = True # = True to save SCA images to file
save_figures = True # = True to save SCA output figures to file

# -----Import packages
import xarray as xr
import rioxarray
import wxee as wx
import os
import numpy as np
import glob
from osgeo import gdal
import matplotlib
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt, dates
import rasterio as rio
import rasterio.features
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Polygon, shape
import shapely.geometry
from scipy.interpolate import interp2d
from scipy import stats
import pandas as pd
import geopandas as gpd
import geemap
import math
import sys
import ee
import fiona
import pickle
import wxee as wx
import time
from PIL import Image as PIL_Image
from IPython.display import Image as IPy_Image

# -----Add path to functions
sys.path.insert(1, base_path+'functions/')
import pipeline_utils_PlanetScope as pf
import pipeline_utils_LSM as lf

# -----Load dataset dictionary
with open(base_path + 'inputs-outputs/datasets_characteristics.pkl', 'rb') as fn:
    dataset_dict = pickle.load(fn)
dataset = 'Sentinel2'
ds_dict = dataset_dict[dataset]

# -----Define output paths
classified_path = out_path + 'classified/'
snowlines_path = out_path + 'snowlines/'

# -----Authenticate & initialize Google Earth Engine (GEE)
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# -----Load AOI and DEM
# load AOI as gpd.GeoDataFrame
AOI_fn = glob.glob(AOI_fn)[0]
AOI = gpd.read_file(AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs(4326)
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
epsg_UTM = lf.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])

# load DEM as Xarray DataSet
if DEM_fn==None:
    # query GEE for DEM
    DEM, AOI_UTM = lf.query_GEE_for_DEM(AOI)
else:
    # reproject AOI to UTM
    AOI_UTM = AOI.to_crs(str(epsg_UTM))
    # load DEM as xarray DataSet
    DEM_fn = glob.glob(DEM_fn)[0]
    DEM_rio = rio.open(DEM_fn) # open using rasterio to access the transform
    DEM = xr.open_dataset(DEM_fn)
    DEM = DEM.rename({'band_data': 'elevation'})
    # reproject the DEM to the optimal UTM zone
    DEM = DEM.rio.reproject(str('EPSG:'+epsg_UTM))

# ---------------------- #
# --- 1. LOAD IMAGES --- #
# ---------------------- #

print('--------------------')
print('1. LOAD IMAGES')
print('--------------------')

# -----Check if masked surface reflectance images already exist in file
S2_masked_fn = dataset + '_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_masked.nc'
if len(glob.glob(out_path + S2_masked_fn)) > 0:
    
    # load masked image collection from file
    print('Masked image collection already exists in file, loading...')
    S2_masked = xr.open_dataset(out_path + S2_masked_fn)
                             
else:

    # query GEE for SR images
    print('Querying GEE for Sentinel-2 SR imagery...')
    S2_masked = lf.query_GEE_for_Sentinel2_SR(AOI, date_start, date_end, month_start, month_end, cloud_cover_max, ds_dict)
    
    # save masked image collection
    if S2_masked is not None:
        S2_masked.to_netcdf(out_path + S2_masked_fn)
        print('Sentinel-2 masked SR image collection saved to file: ' + out_path + S2_masked_fn)

# -------------------------- #
# --- 2. CLASSIFY IMAGES --- #
# -------------------------- #

print('--------------------')
print('2. CLASSIFY IMAGES')
print('--------------------')

# load trained classifier and feature columns
clf_fn = base_path+'inputs-outputs/S2_classifier_all_sites.sav'
clf = pickle.load(open(clf_fn, 'rb'))
feature_cols_fn = base_path+'inputs-outputs/S2_feature_cols.pkl'
feature_cols = pickle.load(open(feature_cols_fn,'rb'))

# classify images
plot_results=False
S2_masked_classified, S2_masked_classified_fn, fig = lf.classify_image_collection(S2_masked, clf, feature_cols,
                                                                                  crop_to_AOI, AOI_UTM, ds_dict,
                                                                                  dataset, site_name, out_path,
                                                                                  date_start, date_end, plot_results,
                                                                                  figures_out_path)

# -----Compile individual figures into a .gif and delete individual figures
# identify the string that is present in all filenames of the figures that you want to compile
#fig_fns_str = 'Sentinel2_' + site_name + '_*SCA.png'
## define the output .gif filename
#gif_fn = 'Sentinel2_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_SCA.gif'
## make a .gif of output images
#os.chdir(figures_out_path)
#fig_fns = glob.glob(fig_fns_str) # load all output figure file names
#fig_fns = sorted(fig_fns) # sort chronologically
## grab figures date range for .gif file name
#frames = [PIL_Image.open(im) for im in fig_fns]
#frame_one = frames[0]
#frame_one.save(figures_out_path + gif_fn, format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
#print('GIF saved to file:' + figures_out_path + gif_fn)
## clean up: delete individual figure files
#for fn in fig_fns:
#    os.remove(os.path.join(figures_out_path, fn))
#print('Individual figure files deleted.')

# ------------------------------ #
# --- 3. DELINEATE SNOWLINES --- #
# ------------------------------ #

print('----------')
print('3. DELINEATE SNOWLINES')
print('----------')
 
# -----Delineate snowlines
plot_results=True
snowlines_df = lf.Landsat_delineate_snow_line(S2_masked, S2_masked_classified, date_start,
                                              date_end, site_name, AOI_UTM, DEM, ds_dict,
                                              dataset, out_path, figures_out_path, plot_results)
                                              
# -----Compile individual figures into a .gif and delete individual figures
# identify the string that is present in all filenames of the figures that you want to compile
fig_fns_str = 'Sentinel2_' + site_name + '_*snowlines.png'
# define the output .gif filename
gif_fn = 'Sentinel2_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_SCA.gif'
# make a .gif of output images
os.chdir(figures_out_path)
fig_fns = glob.glob(fig_fns_str) # load all output figure file names
fig_fns = sorted(fig_fns) # sort chronologically
# grab figures date range for .gif file name
frames = [PIL_Image.open(im) for im in fig_fns]
frame_one = frames[0]
frame_one.save(figures_out_path + gif_fn, format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
print('GIF saved to file:' + figures_out_path + gif_fn)
# clean up: delete individual figure files
for fn in fig_fns:
    os.remove(os.path.join(figures_out_path, fn))
print('Individual figure files deleted.')
