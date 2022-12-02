### Classify snow-covered area (SCA) in PlanetScope imagery: full pipeline
# Rainey Aberle
# Department of Geosciences, Boise State University
# 2022
#
## Requirements:
# - Planet account with access to PlanetScope imagery through the NASA CSDA contract.
#   Sign up here: https://www.planet.com/markets/nasa/
# - PlanetScope 4-band image collection over the AOI. Download images using
#   `planetAPI_image_download.ipynb` or through [PlanetExplorer](https://www.planet.com/explorer/).
# - Google Earth Engine (GEE) account: used to pull DEM over the AOI.
#   Sign up for a free account here: https://earthengine.google.com/new_signup/
#
## Outline:
#   0. Setup paths in directory, AOI file location
#   1. Mask image pixels using associated UDM files for cloud, shadow, and heavy haze
#   2. Mosaic images captured in the same hour
#   3. Adjust image radiometry using median surface reflectance at the top
#       perentile of elevations
#   4. Classify SCA and use the snow elevations distribution to estimate the seasonal snowline
#   5. Estimate snow line and snow line elevations

# -------------------------
# ------  0. Setup  -------
# -------------------------
# -----Import packages
import argparse
import os
import numpy as np
import glob
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import sys
import time
import ee
import pickle
from time import mktime
from matplotlib import pyplot as plt, dates
import matplotlib
from PIL import Image as PIL_Image
from IPython.display import Image as IPy_Image

# -----Parse arguments
def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full snow classification workflow')
    parser.add_argument('-base_path', default=None, type=str, help='path to snow-cover-mapping')
    parser.add_argument('-site_name', default=None, type=str, help='name of study site')
    parser.add_argument('-im_path', default=None, type=str, help='path to raw images')
    parser.add_argument('-AOI_path', default=None, type=str, help='path to AOI shapefile')
    parser.add_argument('-AOI_fn', default=None, type=str, help='file name of AOI shapefile')
    parser.add_argument('-DEM_path', default=None, type=str, help='path to DEM')
    parser.add_argument('-DEM_fn', default=None, type=str, help='file name of DEM (tif or tiff)')
    parser.add_argument('-out_path', default=None, type=str, help='path to output folder to save results in')
    parser.add_argument('-steps_to_run', nargs='*', help='specify steps of workflow to run (e.g., [1, 2, 3, 4, 5])')
    parser.add_argument('-skip_clipped', default=False, type=bool, help='whether to skip images that appear clipped')
    parser.add_argument('-crop_to_AOI', default=True, type=bool, help='whether to crop images to the AOI before classifying')
    return parser
parser = getparser()
args = parser.parse_args()
base_path = args.base_path
site_name = args.site_name
im_path = args.im_path
AOI_path = args.AOI_path
AOI_fn = args.AOI_fn
DEM_path = args.DEM_path
DEM_fn = args.DEM_fn
out_path = args.out_path
steps_to_run = np.array(args.steps_to_run).astype(int)
skip_clipped = args.skip_clipped
crop_to_AOI = args.crop_to_AOI
save_outputs = True

# -----Print job to be run
print('Site name: ' + site_name)
print('Steps to run: '+str(steps_to_run))
print(' ')

# -----Check for input files
# input image files
if 1 in steps_to_run:
    im_list = glob.glob(os.path.join(im_path,'*.tif'))+glob.glob(os.path.join(im_path,'*.tiff'))
    if len(im_list)<1:
        print('< 1 images detected, exiting')
        sys.exit()
# AOI path
if not os.path.exists(AOI_path):
    print('Path to AOI could not be located, exiting')
    sys.exit()
# AOI file
AOI_fn_full = glob.glob(os.path.join(AOI_path, AOI_fn))
if len(AOI_fn_full)<1:
    print('AOI shapefile could not be located in directory, exiting')
    sys.exit()
# DEM path
if not os.path.exists(DEM_path):
    print('Path to DEM could not be located, exiting')
    sys.exit()
# DEM file
DEM_fn_full = glob.glob(os.path.join(DEM_path, DEM_fn))
if len(DEM_fn_full)<1:
    print('DEM file could not be located in directory, exiting')
    sys.exit()

# -----Add path to functions
sys.path.insert(1, base_path+'functions/')
import pipeline_utils_PlanetScope as pf
import pipeline_utils_LSM as lf

# -----Load AOI as GeoPandas.DataFrame
AOI_fn = glob.glob(AOI_path + AOI_fn)[0]
AOI = gpd.read_file(AOI_fn)
# reproject to the optimal UTM zone
AOI_WGS = AOI.to_crs(4326)
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
epsg_UTM = lf.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
AOI_UTM = AOI.to_crs(str(epsg_UTM))

# -----Load DEM as xarray DataSet
DEM_fn = glob.glob(DEM_path + DEM_fn)[0]
DEM = xr.open_dataset(DEM_fn)
DEM = DEM.rename({'band_data': 'elevation'})
# reproject to the optimal UTM zone
DEM = DEM.rio.reproject(str('EPSG:'+epsg_UTM))

# -----Load dataset dictionary
with open(base_path + 'inputs-outputs/datasets_characteristics.pkl', 'rb') as fn:
    dataset_dict = pickle.load(fn)
dataset = 'PlanetScope'
ds_dict = dataset_dict[dataset]

# -----Set paths for output files
im_masked_path = out_path + 'masked/'
im_mosaics_path = out_path + 'mosaics/'
im_adjusted_path = out_path + 'adjusted/'
im_classified_path = out_path + 'classified/'
snowlines_path = out_path + 'snowlines/'
figures_out_path = out_path + '../../figures/'


# ---------------------------------
# ---  1. Mask unusable pixels  ---
# ---------------------------------
plot_results=False
if 1 in steps_to_run:

    print('--------------------------')
    print('1. Masking pixels using UDM/UDM2')
    print('--------------------------')

    # -----Read surface reflectance file names
    os.chdir(im_path)
    im_fns = glob.glob('*SR*.tif')
    im_fns = sorted(im_fns) # sort chronologically

    # ----Mask images
    for im_fn in im_fns:
        
        print(im_fn)

        pf.mask_im_pixels(im_path, im_fn, im_masked_path, save_outputs, plot_results)
        print(' ')

# ----------------------------------
# ---  2. Mosaic images by date  ---
# ----------------------------------
if 2 in steps_to_run:

    print('--------------------------')
    print('2. Mosaicing images by date')
    print('--------------------------')

    # -----Read masked image file names
    os.chdir(im_masked_path)
    im_mask_fns = glob.glob('*_mask.tif')
    im_mask_fns = sorted(im_mask_fns) # sort chronologically

    # ----Mosaic images by date
    pf.mosaic_ims_by_date(im_masked_path, im_mask_fns, im_mosaics_path, AOI_UTM, plot_results)


# ------------------------------------
# ---  3. Adjust image radiometry  ---
# ------------------------------------

# -----Read image mosaic file names
os.chdir(im_mosaics_path)
im_mosaic_fns = glob.glob('*.tif')
im_mosaic_fns = sorted(im_mosaic_fns)
    
if 3 in steps_to_run:

    print('--------------------------')
    print('3. Adjusting images')
    print('--------------------------')

    # -----Create a polygon(s) of the top 20th percentile elevations within the AOI
    polygon_top, polygon_bottom, im_mosaic_fn, im_mosaic = pf.create_AOI_elev_polys(AOI_UTM, im_mosaics_path, im_mosaic_fns, DEM)

    # -----Loop through images
    for im_mosaic_fn in im_mosaic_fns:

        # load image
        print(im_mosaic_fn)
        # adjust radiometry
        plot_results=False
        im_adj_fn, im_adj_method = pf.adjust_image_radiometry(im_mosaic_fn, im_mosaics_path, polygon_top, polygon_bottom, AOI_UTM, ds_dict, dataset, site_name, im_adjusted_path, skip_clipped, plot_results)
        print('image adjustment method = ' + im_adj_method)
        print(' ')

# ----------------------------
# ---  4. Classify images  ---
# ----------------------------
if 4 in steps_to_run:

    print('--------------------------')
    print('4. Classifying images')
    print('--------------------------')

    # -----Load trained classifier and feature columns
    clf_fn = base_path+'inputs-outputs/PS_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path+'inputs-outputs/PS_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))

    # -----Read masked images
    # im_adjusted_fns = glob.glob(im_adjusted_path + '*_adj.nc')
    im_adjusted_fns = glob.glob(im_adjusted_path + '*_adj.nc')

    im_adjusted_fns = sorted(im_adjusted_fns) # sort chronologically

    # -----Loop through masked image files
    for im_adjusted_fn in im_adjusted_fns:
        # load file
        im_adjusted = xr.open_dataset(im_adjusted_fn)
        # classify images
        plot_results=True
        im_classified = lf.classify_image(im_adjusted, clf, feature_cols,
                                          crop_to_AOI, AOI, ds_dict, dataset,
                                          site_name, im_classified_path, plot_results,
                                          figures_out_path)
        print(' ')
    
    
# ---------------------------------
# ---  5. Delineate snow lines  ---
# ---------------------------------
if 5 in steps_to_run:

    print('--------------------------')
    print('5. Delineating snow lines')
    print('--------------------------')

    # -----Read image file names
    # adjusted images
    im_adjusted_fns = glob.glob(im_adjusted_path + '*_adj.nc')
    im_adjusted_fns = sorted(im_adjusted_fns) # sort chronologically
    # classified images
    im_classified_fns = glob.glob(im_classified_path + '*_classified.nc')
    im_classified_fns = sorted(im_classified_fns) # sort chronologically

    # -----Initialize snowlines data frame
    snowlines_df = pd.DataFrame(columns=['study_site', 'datetime', 'snowlines_coords', 'snowlines_elevs', 'snowlines_elevs_median'])
    
    # -----Loop through classified images
    for im_classified_fn in im_classified_fns:
            
        # load classified file
        im_classified = xr.open_dataset(im_classified_fn)
        im_dt = im_classified_fn.split(site_name+'_')[1][0:15]
        print(im_dt)
        
        # check if snowline exists in directory already
        snowline_fn = dataset + '_' + site_name + '_' + im_dt + '_snowline.pkl'
        if os.path.exists(os.path.join(snowlines_path, snowline_fn)):
            print('snowline already exist in file, loading...')
            snowline_df = pickle.load(open(snowlines_path + snowline_fn,'rb'))
        else:
            # load masked image file
            im_adjusted_fn = [x for x in im_adjusted_fns if (im_dt in x)][0]
            im_adjusted = xr.open_dataset(im_adjusted_fn)
            # delineate snowline
            snowline_df = lf.delineate_im_snowline(im_adjusted, im_classified, site_name, AOI_UTM, DEM, ds_dict, dataset, im_dt, snowlines_path, figures_out_path, plot_results)
        # save snowline to file
        snowline_df.to_pickle(snowlines_path + snowline_fn)
        print('snowline saved to file:' + snowlines_path + snowline_fn)
        # concatenate results to snowlines_df
        snowlines_df = pd.concat([snowlines_df, snowline_df])
        print(' ')
    
    # -----Save snowlines_df to file
    date_start = im_classified_fns[0].split(site_name+'_')[1][0:8]
    date_end = im_classified_fns[-1].split(site_name+'_')[1][0:8]
    snowlines_fn = dataset + '_' + site_name + '_' + date_start + '_' + date_end + '_snowlines.pkl'
    snowlines_df = snowlines_df.reset_index(drop=True)
    snowlines_df.to_pickle(snowlines_path + snowlines_fn)
    print('snowline timeseries compiled and saved to file:' + snowlines_path + snowlines_fn)

    # -----Plot median snow line elevations
    if plot_results:
        fig2, ax2 = plt.subplots(figsize=(10,6))
        plt.rcParams.update({'font.size':12, 'font.sans-serif':'Arial'})
        # plot snowlines
        ax2.plot(snowlines_df['datetime'].astype(np.datetime64),
                 snowlines_df['snowlines_elevs_median'], '.b', markersize=10)
        ax2.set_ylabel('Median snow line elevation [m]')
        ax2.grid()
        # format x-axis
        xmin, xmax = np.datetime64('2016-05-01T00:00:00'), np.datetime64('2022-11-01T00:00:00')
        fmt_month = matplotlib.dates.MonthLocator(bymonth=(5, 11)) # minor ticks every month.
        fmt_year = matplotlib.dates.YearLocator() # minor ticks every year.
        ax2.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%b'))
        ax2.xaxis.set_major_locator(fmt_month)
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
        # create a second x-axis beneath the first x-axis to show the year in YYYY format
        sec_xaxis = ax2.secondary_xaxis(-0.1)
        sec_xaxis.xaxis.set_major_locator(fmt_year)
        sec_xaxis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        # Hide the second x-axis spines and ticks
        sec_xaxis.spines['bottom'].set_visible(False)
        sec_xaxis.tick_params(axis='x', length=0, pad=-10)
        fig2.suptitle(site_name + ' Glacier median snow line elevations')
        fig2.tight_layout()
        plt.show()
        # save figure
        fig2_fn = figures_out_path + dataset + '_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','')+ '_snowline_median_elevs.png'
        fig2.savefig(fig2_fn, dpi=300, facecolor='white', edgecolor='none')
        print('figure saved to file:' + fig2_fn)

# -------------------------------------
# ---  *Compile figures into .gif*  ---
# -------------------------------------
# -----SCA figures
#os.chdir(figures_out_path)
#fig_fns = glob.glob('PlanetScope_*_SCA.png') # load all output figure file names
#if fig_fns:
#    print('Compiling individual SCA figures into gif...')
#    fig_fns = sorted(fig_fns) # sort chronologically
#    # grab figures date range for .gif file name
#    fig_start_date = fig_fns[0][3:-8] # first figure date
#    fig_end_date = fig_fns[-1][3:-8] # final figure date
#    frames = [PIL_Image.open(im) for im in fig_fns]
#    frame_one = frames[0]
#    gif_fn = ('PlanetScope_' + fig_start_date[0:8] + '_' + fig_end_date[0:8] + '_SCA.gif' )
#    frame_one.save(figures_out_path + gif_fn, format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
#    print('GIF saved to file:' + figures_out_path + gif_fn)
#    # clean up: delete individual figure files
#    for fig_fn in fig_fns:
#        os.remove(fig_fn)
#    print('Individual figure files deleted.')
#    
## -----Snowline figures
#fig_fns = glob.glob('PlanetScope_*_snowline.png') # load all output figure file names
#if fig_fns:
#    print('Compiling individual snowline figures into .gif...')
#    fig_fns = sorted(fig_fns) # sort chronologically
#    # grab figures date range for .gif file name
#    fig_start_date = fig_fns[0].sp')[3:-8] # first figure date
#    fig_end_date = fig_fns[-1][3:-8] # final figure date
#    frames = [PIL_Image.open(im) for im in fig_fns]
#    frame_one = frames[0]
#    gif_fn = ('PlanetScope_' + fig_start_date[0:8] + '_' + fig_end_date[0:8] + '_snowlines.gif' )
#    frame_one.save(figures_out_path + gif_fn, format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
#    print('GIF saved to file:' + figures_out_path + gif_fn)
#    # clean up: delete individual figure files
#    for fig_fn in fig_fns:
#        os.remove(fig_fn)
#    print('Individual figure files deleted.')
#    print(' ')

print('DONE!')
