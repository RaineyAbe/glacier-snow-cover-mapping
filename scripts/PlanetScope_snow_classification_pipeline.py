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
import subprocess
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt, dates
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from scipy import stats
import pandas as pd
import geopandas as gpd
import sys
import ee
import pickle
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
import ps_pipeline_utils as f

# -----Load AOI as GeoPandas.DataFrame
AOI_fn = glob.glob(AOI_path + AOI_fn)[0]
AOI = gpd.read_file(AOI_fn)
# reproject to the optimal UTM zone
AOI_WGS = AOI.to_crs(4326)
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
AOI_UTM = AOI.to_crs(str(epsg_UTM))

# -----Load DEM as xarray DataSet
DEM_fn = glob.glob(DEM_path + DEM_fn)[0]
DEM_rio = rio.open(DEM_fn) # open using rasterio to access the transform
DEM = xr.open_dataset(DEM_fn)
DEM = DEM.rename({'band_data': 'elevation'})
# reproject to the optimal UTM zone
DEM = DEM.rio.reproject(str('EPSG:'+epsg_UTM))

# -----Set paths for output files
im_mask_path = os.path.join(out_path, 'masked/')
im_mosaic_path = os.path.join(out_path, 'mosaics/')
im_adj_path = os.path.join(out_path, 'adjusted/')
im_classified_path = os.path.join(out_path, 'classified/')
figures_out_path = os.path.join(out_path, '../../figures/')
snowlines_path = os.path.join(out_path, 'snowlines/')


# ---------------------------------
# ---  1. Mask unusable pixels  ---
# ---------------------------------
plot_results=False
if 1 in steps_to_run:

    print('--------------------------')
    print('1. Masking pixels using UDM')
    print('--------------------------')

    # -----Read surface reflectance file names
    os.chdir(im_path)
    im_fns = glob.glob('*SR*.tif')
    im_fns = sorted(im_fns) # sort chronologically

    # ----Mask images
    for im_fn in im_fns:
        
        print(im_fn)

        f.mask_im_pixels(im_path, im_fn, im_mask_path, save_outputs, plot_results)
        print(' ')

# ----------------------------------
# ---  2. Mosaic images by date  ---
# ----------------------------------
if 2 in steps_to_run:

    print('--------------------------')
    print('2. Mosaicing images by date')
    print('--------------------------')

    # -----Read masked image file names
    os.chdir(im_mask_path)
    im_mask_fns = glob.glob('*_mask.tif')
    im_mask_fns = sorted(im_mask_fns) # sort chronologically

    # ----Mosaic images by date
    f.mosaic_ims_by_date(im_mask_path, im_mask_fns, im_mosaic_path, AOI_UTM, plot_results)


# ------------------------------------
# ---  3. Adjust image radiometry  ---
# ------------------------------------

# -----Read image mosaic file names
os.chdir(im_mosaic_path)
im_mosaic_fns = glob.glob('*.tif')
im_mosaic_fns.sort()

# -----Create a polygon(s) of the top 20th percentile elevations within the AOI
polygon_top, polygon_bottom, im_mosaic_fn, im_mosaic = f.create_AOI_elev_polys(AOI_UTM, im_mosaic_path, im_mosaic_fns, DEM, DEM_rio)
    
if 3 in steps_to_run:

    print('--------------------------')
    print('3. Adjusting images')
    print('--------------------------')

    # -----Loop through images
    for im_mosaic_fn in im_mosaic_fns:

        # load image
        print(im_mosaic_fn)
        # adjust radiometry
        plot_results=False
        im_adj_fn, im_adj_method = f.adjust_image_radiometry(im_mosaic_fn, im_mosaic_path, polygon_top, polygon_bottom, im_adj_path, skip_clipped, plot_results)
        print('image adjustment method = ' + im_adj_method)
        print(' ')

# ----------------------------
# ---  4. Classify images  ---
# ----------------------------
if 4 in steps_to_run:

    print('--------------------------')
    print('4. Classifying images')
    print('--------------------------')

    # -----Read adjusted image file names
    os.chdir(im_adj_path)
    im_adj_fns = glob.glob('*.tif')
    im_adj_fns.sort()

    # -----Load image classifier and feature columns
    clf_fn = base_path + 'inputs-outputs/PS_classifier_all_sites.sav'
    clf = pickle.load(open(clf_fn, 'rb'))
    feature_cols_fn = base_path + 'inputs-outputs/PS_feature_cols.pkl'
    feature_cols = pickle.load(open(feature_cols_fn,'rb'))

    # -----Loop through images
    # image datetimes
    im_dts = []
    # DataFrame to hold stats summary
    df = pd.DataFrame(columns=('site_name', 'datetime', 'im_elev_min', 'im_elev_max', 'snow_elev_min', 'snow_elev_max',
                               'snow_elev_median', 'snow_elev_10th_perc', 'snow_elev_90th_perc'))
    for im_adj_fn in im_adj_fns:

        print(im_adj_fn[0:8])

        # extract datetime from image name
        im_dt = np.datetime64(im_adj_fn[0:4] + '-' + im_adj_fn[4:6] + '-' + im_adj_fn[6:8] + 'T' + im_adj_fn[9:11] + ':00:00')
        im_dts = im_dts + [im_dt]

        # classify snow
        im_classified_fn, im_adj = f.classify_image(im_adj_fn, im_adj_path, clf, feature_cols, crop_to_AOI, AOI_UTM, im_classified_path)
        
        print(' ')

# ---------------------------------
# ---  5. Delineate snow lines  ---
# ---------------------------------
if 5 in steps_to_run:

    print('--------------------------')
    print('5. Delineating snow lines')
    print('--------------------------')

    # -----Read classified image file names
    os.chdir(im_classified_path)
    im_classified_fns = glob.glob('*.tif')
    im_classified_fns = sorted(im_classified_fns)

    # -----Create directories for outputs if they do not exist
    # snowlines folder
    if save_outputs and os.path.exists(snowlines_path)==False:
        os.mkdir(snowlines_path)
        print('made directory for output snowlines:' + snowlines_path)
        print(' ')
    # figures folder
    if os.path.exists(figures_out_path)==False:
        os.mkdir(figures_out_path)
        print('made directory for output figures:' + figures_out_path)
        print(' ')

    # -----Loop through classified image filenames
    for im_classified_fn in im_classified_fns:
            
        # extract datetime from image file name
        im_date = im_classified_fn[0:11]
        print(im_date)
        im_dt = np.datetime64(im_classified_fn[0:4] + '-' + im_classified_fn[4:6] + '-' + im_classified_fn[6:8] + 'T' + im_classified_fn[9:11] + ':00:00')

        # check if snowline already exists in file
        snowline_fn = site_name + '_' + im_date + '_snowline.pkl'
        print(snowline_fn)
        if os.path.exists(snowlines_path + snowline_fn):
            print("snowline already exists in file, continuing...")
            continue
        else:
            print("snowline not found")
            
        # load adjusted image from the same date
        os.chdir(im_adj_path)
        im_adj_fn = glob.glob(im_date + '*.tif')[0]

        # delineate estimated snow line
#        try:
        fig, ax, sl_est, sl_est_elev = f.delineate_snow_line(im_adj_fn, im_adj_path, im_classified_fn, im_classified_path, AOI_UTM, DEM, DEM_rio)

        # calculate median snow line elevation
        sl_est_elev_median = np.nanmedian(sl_est_elev)

        # save figure
        fig.savefig(os.path.join(figures_out_path, 'PS_' + im_date + '_SCA.png'), dpi=300, facecolor='white', edgecolor='none')
        print('figure saved to file')

        # compile result in df
        result_df = pd.DataFrame({'study_site': site_name,
                                  'datetime': im_dt,
                                  'snowlines_coords': [sl_est],
                                  'snowlines_elevs': [sl_est_elev],
                                  'snowlines_elevs_median': sl_est_elev_median})
        
        # save snowline to file
        if save_outputs:
            result_df.to_pickle(os.path.join(snowlines_path, snowline_fn))
            print('results data table saved to file')

#        except:
#            print('error in snowline delineation, skipping...')
#            pass

        print(' ')

    # -----Compile result data tables
    # grab file names of all snow line data frames
    df_fns = glob.glob(snowlines_path + '*.pkl')
    # initialize full data frame
    results_df = pd.DataFrame(columns=['study_site', 'datetime', 'snowlines_coords', 'snowlines_elevs', 'snowlines_elevs_median'])
    # loop through data tables
    for df_fn in df_fns:
        # open data frame
        result_df = pd.read_pickle(df_fn)
        # concatenate to results_df
        results_df = pd.concat([results_df, result_df])
    results_df = results_df.reset_index(drop=True) # reset df row indices
    # grab snowline start and end dates for filename
    df_dates = [df_fn.split(site_name)[1][1:].split('_')[0] for df_fn in df_fns]
    df_dates = sorted(df_dates)
    sl_start_date = df_dates[0]
    sl_end_date = df_dates[-1]
    # save full snow lines data frame to file
    results_df_fn = site_name + '_' + sl_start_date + '_' + sl_end_date + '_snowlines.pkl'
    results_df.to_pickle(os.path.join(snowlines_path, results_df_fn))
    print('all snowlines saved to file:' + snowlines_path + results_df_fn)
    # delete individual snowline files
    for df_fn in df_fns:
        os.remove(df_fn)
    print('individual snowline files deleted.')
    
    # -----Plot median snow line elevations
    # plot
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(results_df['datetime'], results_df['snowlines_elevs_median'], '.b')
    ax2.set_xlabel('Image capture date')
    ax2.set_ylabel('Median snow line elevation [m]')
    ax2.grid()
    fig2.suptitle(site_name + ' Glacier snow line elevations')

    # save results
    fig2.savefig(os.path.join(figures_out_path, site_name + '_sl_elevs_median.png'),
                 facecolor='white', edgecolor='none')
    print('figure saved to file')

    # -----Make a .gif of output images
    os.chdir(figures_out_path)
    fig_fns = glob.glob('PS_*_SCA.png') # load all output figure file names
    fig_fns = sorted(fig_fns) # sort chronologically
    
    # grab figures date range for .gif file name
    fig_start_date = fig_fns[0][3:-8] # first figure date
    fig_end_date = fig_fns[-1][3:-8] # final figure date
    frames = [PIL_Image.open(im) for im in fig_fns]
    frame_one = frames[0]
    gif_fn = ('PS_' + fig_start_date[0:8] + '_' + fig_end_date[0:8] + '_SCA.gif' )
    frame_one.save(os.path.join(figures_out_path, gif_fn), format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
    print('GIF saved to file:' + figures_out_path + gif_fn)
    
    # delete individual figure files
    for fig_fn in fig_fns:
        os.remove(os.path.join(figures_out_path, fig_fn))
    print('Individual figure files deleted.')

print('DONE!')
