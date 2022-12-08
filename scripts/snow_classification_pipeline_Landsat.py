# Classify snow-covered area (SCA) in Landsat surface reflectance imagery: full pipeline
# Rainey Aberle
# Department of Geosciences, Boise State University
# 2022
#
# Requirements:
# - Area of Interest (AOI) shapefile: where snow will be classified in all available images.
# - Google Earth Engine (GEE) account: used to search for imagery and DEM over the AOI. Sign up for a free account here: https://earthengine.google.com/new_signup/).
#
# Outline:
# 0. Setup paths in directory, file locations, authenticate GEE - _modify this section!_
# 1. Load images over the AOI
# 2. Classify SCA and use the snow elevations distribution to estimate the seasonal snowline
# 3. Delineate snowlines using classified images.


# ---------------- #
# --- 0. SETUP --- #
# ---------------- #

##### MODIFY HERE #####

# -----Paths in directory
site_name = 'Gulkana'
# path to snow-cover-mapping/
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to AOI including the name of the shapefile
AOI_fn = base_path + '../study-sites/' + site_name + '/glacier_outlines/' + site_name + '_USGS_*.shp'
# path to DEM including the name of the tif file
# Note: set DEM_fn=None if you want to use the ASTER GDEM on Google Earth Engine
DEM_fn = base_path + '../study-sites/' + site_name + '/DEMs/' + site_name + '*_DEM*.tif'
# path for output images
out_path = base_path + '../study-sites/' + site_name + '/imagery/Landsat/'
# path for output figures
figures_out_path = base_path + '../study-sites/' + site_name + '/figures/'

# -----Define image search filters
date_start = '2016-01-01'
date_end = '2022-12-01'
month_start = 5
month_end = 10
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
import matplotlib.pyplot as plt
import matplotlib
import rasterio as rio
import geopandas as gpd
import sys
import ee
import pickle
import pandas as pd

# -----Set paths for output files
im_masked_path = out_path + 'masked/'
im_classified_path = out_path + 'classified/'
snowlines_path = out_path + 'snowlines/'

# -----Add path to functions
sys.path.insert(1, base_path+'functions/')
import pipeline_utils as f

# -----Load dataset dictionary
with open(base_path + 'inputs-outputs/datasets_characteristics.pkl', 'rb') as fn:
    dataset_dict = pickle.load(fn)
dataset = 'Landsat'
ds_dict = dataset_dict[dataset]

# -----Authenticate & initialize Google Earth Engine (GEE)
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# -----Load AOI as gpd.GeoDataFrame
AOI_fn = glob.glob(AOI_fn)[0]
AOI = gpd.read_file(AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs(4326)
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
    
# -----Load DEM as Xarray DataSet
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

# ---------------------- #
# --- 1. LOAD IMAGES --- #
# ---------------------- #

print('--------------------')
print('1. LOAD IMAGES')
print('--------------------')


L_ds_fns = f.query_GEE_for_Landsat_SR(AOI, date_start, date_end, month_start, month_end, cloud_cover_max,
                                       site_name, dataset, ds_dict, im_masked_path, plot_results)
    
# -------------------------- #
# --- 2. CLASSIFY IMAGES --- #
# -------------------------- #

print('--------------------')
print('2. CLASSIFY IMAGES')
print('--------------------')

# load trained classifier and feature columns
clf_fn = base_path+'inputs-outputs/L_classifier_all_sites.sav'
clf = pickle.load(open(clf_fn, 'rb'))
feature_cols_fn = base_path+'inputs-outputs/L_feature_cols.pkl'
feature_cols = pickle.load(open(feature_cols_fn,'rb'))

# read masked images
im_masked_fns = glob.glob(im_masked_path + '*_masked.nc')
im_masked_fns = sorted(im_masked_fns) # sort chronologically

# loop through masked image files
for im_masked_fn in im_masked_fns:
    # load file
    im_masked = xr.open_dataset(im_masked_fn)
    # classify images
    plot_results=False
    im_classified = f.classify_image(im_masked, clf, feature_cols,
                                     crop_to_AOI, AOI_UTM, ds_dict, dataset,
                                     site_name, im_classified_path, plot_results,
                                     figures_out_path)
    print(' ')

# ------------------------------ #
# --- 3. DELINEATE SNOWLINES --- #
# ------------------------------ #

# -----Read image file names
# masked images
im_masked_fns = glob.glob(im_masked_path + '*_masked.nc')
im_masked_fns = sorted(im_masked_fns) # sort chronologically
# classified images
im_classified_fns = glob.glob(im_classified_path + '*_classified.nc')
im_classified_fns = sorted(im_classified_fns) # sort chronologically

# -----Initialize snowlines data frame
snowlines_df = pd.DataFrame(columns=['study_site', 'datetime', 'snowlines_coords', 'snowlines_elevs', 'snowlines_elevs_median'])
    
# -----Loop through classified images
for im_classified_fn in im_classified_fns:
        
    # load classified file
    im_classified = xr.open_dataset(im_classified_fn)
    im_dt = str(im_classified.time.data[0]).replace('-','').replace(':','')[0:15]
    print(im_dt)
    
    # check if snowline exists in directory already
    snowline_fn = site_name + '_' + dataset + '_' + im_dt + '_snowline.pkl'
    if os.path.exists(os.path.join(snowlines_path, snowline_fn)):
        print('snowline already exist in file, loading...')
        snowline_df = pickle.load(open(snowlines_path + snowline_fn,'rb'))
        
    else:
        # load masked image file
        masked_fn = [x for x in im_masked_fns if (im_dt in x)][0]
        im_masked = xr.open_dataset(masked_fn)
        # delineate snowline
        plot_results=True
        snowline_df = f.delineate_im_snowline(im_masked, im_classified, site_name, AOI_UTM, DEM, ds_dict,
                                              dataset, im_dt, snowlines_path, figures_out_path, plot_results)
        
    # save snowline to file
    snowline_df.to_pickle(snowlines_path + snowline_fn)
    print('snowline saved to file:' + snowlines_path + snowline_fn)
    # concatenate results to snowlines_df
    snowlines_df = pd.concat([snowlines_df, snowline_df])
    print(' ')
    
# -----Save snowlines_df to file
date_start = im_classified_fns[0].split(dataset+'_')[1][0:8]
date_end = im_classified_fns[-1].split(dataset+'_')[1][0:8]
snowlines_fn = site_name + '_' + dataset + '_' + date_start + '_' + date_end + '_snowlines.pkl'
snowlines_df = snowlines_df.reset_index(drop=True)
snowlines_df.to_pickle(snowlines_path + snowlines_fn)
print('snowlines saved to file:' + snowlines_path + snowlines_fn)

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
    fig2_fn = figures_out_path + site_name + '_' + dataset + '_' + date_start.replace('-','') + '_' + date_end.replace('-','')+ '_snowline_median_elevs.png'
    fig2.savefig(fig2_fn, dpi=300, facecolor='white', edgecolor='none')
    print('figure saved to file:' + fig2_fn)
    
### Compile individual snowline figures into a GIF ###
# -----Identify the string that is present in all figure file names to be compiled
fig_fns_str = 'Sentinel2_' + site_name + '_*snowline.png'
# define the output .gif filename
gif_fn = 'Sentinel2_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_snowlines.gif'

# -----Make a .gif of output images
os.chdir(figures_out_path)
fig_fns = glob.glob(fig_fns_str) # load all output figure file names
fig_fns = sorted(fig_fns) # sort chronologically
# grab figures date range for .gif file name
frames = [PIL_Image.open(im) for im in fig_fns]
frame_one = frames[0]
frame_one.save(figures_out_path + gif_fn, format="GIF", append_images=frames, save_all=True, duration=2000, loop=0)
print('GIF saved to file:' + figures_out_path + gif_fn)

# -----Clean up: delete individual figure files
for fn in fig_fns:
    os.remove(os.path.join(figures_out_path, fn))
print('Individual figure files deleted.')

