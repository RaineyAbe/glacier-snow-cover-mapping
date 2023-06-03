# Functions for image adjustment and snow classification in Landsat, Sentinel-2, and PlanetScope imagery
# Rainey Aberle
# 2023

import math
import geopandas as gpd
import pandas as pd
import ee
import geedim as gd
from shapely.geometry import MultiPolygon, Polygon, LineString, Point, shape
import os
import wxee as wx
import xarray as xr
import numpy as np
import rasterio as rio
import rioxarray as rxr
from scipy.ndimage import binary_fill_holes
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import glob
from tqdm.auto import tqdm
import re
import datetime
import ipywidgets as widgets
from IPython.display import display, HTML


# --------------------------------------------------
def convert_wgs_to_utm(lon: float, lat: float):
    '''
    Return best UTM epsg-code based on WGS84 lat and lon coordinate pair

    Parameters
    ----------
    lon: float
        longitude coordinate
    lat: float
        latitude coordinate

    Returns
    ----------
    epsg_code: str
        optimal UTM zone
    '''
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code

# --------------------------------------------------
def plot_xr_RGB_image(im_xr, RGB_bands):
    '''Plot RGB image of xarray.DataSet

    Parameters
    ----------
    im_xr: xarray.DataSet
        dataset containing image bands in data variables with x and y coordinates.
        Assumes x and y coordinates are in units of meters.
    RGB_bands: List
        list of data variable names for RGB bands contained within the dataset, e.g. ['red', 'green', 'blue']

    Returns
    ----------
    fig: matplotlib.pyplot.figure
        figure handle for the resulting plot
    ax: matplotlib.pyplot.figure.Axes
        axis handle for the resulting plot
    '''

    # -----Grab RGB bands from dataset
    if len(np.shape(im_xr[RGB_bands[0]].data)) > 2: # check if need to take [0] of data
        red = im_xr[RGB_bands[0]].data[0]
        blue = im_xr[RGB_bands[1]].data[0]
        green = im_xr[RGB_bands[2]].data[0]
    else:
        red = im_xr[RGB_bands[0]].data
        blue = im_xr[RGB_bands[1]].data
        green = im_xr[RGB_bands[2]].data

    # -----Format datatype as float, rescale RGB pixel values from 0 to 1
    red, green, blue = red.astype(float), green.astype(float), blue.astype(float)
    im_min = np.nanmin(np.ravel([red, green, blue]))
    im_max = np.nanmax(np.ravel([red, green, blue]))
    red = ((red - im_min) * (1/(im_max - im_min)))
    green = ((green - im_min) * (1/(im_max - im_min)))
    blue = ((blue - im_min) * (1/(im_max - im_min)))

    # -----Plot
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.imshow(np.dstack([red, green, blue]),
              extent=(np.min(im_xr.x.data)/1e3, np.max(im_xr.x.data)/1e3, np.min(im_xr.y.data)/1e3, np.max(im_xr.y.data)/1e3))
    ax.grid()
    ax.set_xlabel('Easting [km]')
    ax.set_ylabel('Northing [km]')

    return fig, ax

# --------------------------------------------------
def query_GEE_for_DEM(AOI, base_path, site_name, out_path=None):
    '''Query GEE for the ASTER Global DEM, clip to the AOI, and return as a numpy array.

    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM
    base_path: str
        path to 'snow-cover-mapping/' used to load ArcticDEM_Mosaic_coverage.shp
    site_name: str
        name of site used for saving output files
    out_path: str
        path where DEM will be saved (if size exceeds GEE limit). Default = None.

    Returns
    ----------
    DEM_ds: xarray.Dataset
        elevations extracted within the AOI
    AOI_UTM: geopandas.geodataframe.GeoDataFrame
        AOI reprojected to the appropriate UTM coordinate reference system
    '''

    # -----Grab optimal UTM zone, reproject AOI
    # reproject AOI to WGS 84 for compatibility with DEM
    AOI_WGS = AOI.to_crs('EPSG:4326')
    AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                        AOI_WGS.geometry[0].centroid.xy[1][0]]
    epsg_UTM = convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
    AOI_UTM = AOI.to_crs('EPSG:'+str(epsg_UTM))

    # -----Define output image name(s), check if already exists in directory
    ArcticDEM_fn = site_name+'_ArcticDEM_clip.tif'
    NASADEM_fn = site_name+'_NASADEM_clip.tif'
    if os.path.exists(out_path+ArcticDEM_fn):
        print('Clipped ArcticDEM already exists in directory, loading...')
        DEM_ds = xr.open_dataset(out_path+ArcticDEM_fn)
        DEM_ds = DEM_ds.rename({'band_data':'elevation'}) # rename band data to "elevation"
        DEM_ds = DEM_ds.rio.reproject('EPSG:'+str(epsg_UTM)) # reproject to optimal UTM zone
    elif os.path.exists(out_path+NASADEM_fn):
        print('Clipped NASADEM already exists in directory, loading...')
        DEM_ds = xr.open_dataset(out_path+NASADEM_fn)
        DEM_ds = DEM_ds.rename({'band_data':'elevation'}) # rename band data to "elevation"
        DEM_ds = DEM_ds.rio.reproject('EPSG:'+str(epsg_UTM)) # reproject to optimal UTM zone
    else: # if no DEM exists in directory, load from GEE

        # -----Reformat AOI for clipping DEM
        AOI_UTM_buffer = AOI_UTM.copy(deep=True)
        AOI_UTM_buffer.geometry[0] = AOI_UTM_buffer.geometry[0].buffer(1000)
        AOI_WGS_buffer = AOI_UTM_buffer.to_crs('EPSG:4326')
        region = {'type': 'Polygon',
                  'coordinates':[[
                                  [AOI_WGS_buffer.geometry.bounds.minx[0], AOI_WGS_buffer.geometry.bounds.miny[0]],
                                  [AOI_WGS_buffer.geometry.bounds.maxx[0], AOI_WGS_buffer.geometry.bounds.miny[0]],
                                  [AOI_WGS_buffer.geometry.bounds.maxx[0], AOI_WGS_buffer.geometry.bounds.maxy[0]],
                                  [AOI_WGS_buffer.geometry.bounds.minx[0], AOI_WGS_buffer.geometry.bounds.maxy[0]],
                                  [AOI_WGS_buffer.geometry.bounds.minx[0], AOI_WGS_buffer.geometry.bounds.miny[0]]
                                 ]]
                 }

        # -----Check for ArcticDEM coverage over AOI
        # load ArcticDEM_Mosaic_coverage.shp
        ArcticDEM_coverage_fn = 'ArcticDEM_Mosaic_coverage.shp'
        ArcticDEM_coverage = gpd.read_file(base_path+'inputs-outputs/'+ArcticDEM_coverage_fn)
        # reproject to optimal UTM zone
        ArcticDEM_coverage_UTM = ArcticDEM_coverage.to_crs('EPSG:'+str(epsg_UTM))
        # check for intersection with AOI
        intersects = ArcticDEM_coverage_UTM.geometry[0].intersects(AOI_UTM.geometry[0])
        # use ArcticDEM if intersects==True
        if intersects:
            print('ArcticDEM coverage over AOI');
            DEM = gd.MaskedImage.from_id('UMN/PGC/ArcticDEM/V3/2m_mosaic', region=region)
            DEM_fn = ArcticDEM_fn # file name for saving
            res = 2 # spatial resolution [m]
        else:
            print('No ArcticDEM coverage, using NASADEM')
            DEM = gd.MaskedImage.from_id("NASA/NASADEM_HGT/001", region=region)
            DEM_fn = NASADEM_fn # file name for saving
            res = 30 # spatial resolution [m]

        # -----Determine whether DEM must be downloaded
        # # Calculate width and height of AOI bounding box [m]
        # AOI_UTM_bb_width = AOI_UTM.geometry[0].bounds[2] - AOI_UTM.geometry[0].bounds[0]
        # AOI_UTM_bb_height = AOI_UTM.geometry[0].bounds[3] - AOI_UTM.geometry[0].bounds[1]
        # if ((AOI_UTM_bb_width / res) * (AOI_UTM_bb_height / res) >= 1e9):
        #     DEM_download = True
        #     # print('DEM must be downloaded for full spatial resolution')
        # else:
        #     DEM_download = False
        #     # print('DEM size is within GEE limit, no download necessary')

        # -----Download DEM and open as xarray.Dataset
        # if DEM_download:
        print('Downloading DEM to '+out_path)
        # create out_path if it doesn't exist
        if os.path.exists(out_path)==False:
            os.mkdir(out_path)
        # download DEM
        DEM.download(out_path+DEM_fn, region=region, scale=res)
        # read DEM as xarray.Dataset
        DEM_ds = xr.open_dataset(out_path+DEM_fn)
        # reproject to UTM
        DEM_ds = DEM_ds.rio.reproject('EPSG:'+str(epsg_UTM))
        DEM_ds = DEM_ds.rename({'band_data':'elevation'})
        # remove unnecessary data
        if len(np.shape(DEM_ds.elevation.data))>2:
            DEM_ds['elevation'] = DEM_ds.elevation[0]
        # else:
        #     DEM_ee_im = DEM.ee_image.clip(region)
        #     DEM_ds = DEM_ee_im.wx.to_xarray(scale=res, region=region, crs='EPSG:4326')
        #     DEM_ds = DEM_ds.rio.reproject('EPSG:'+str(epsg_UTM))

    return DEM_ds, AOI_UTM


# --------------------------------------------------
def query_GEE_for_imagery(dataset, dataset_dict, AOI, date_start, date_end, month_start, month_end, cloud_cover_max, mask_clouds, out_path=None):
    '''
    Query Google Earth Engine for Landsat 8 and 9 surface reflectance (SR), Sentinel-2 top of atmosphere (TOA) or SR imagery.

    Parameters
    __________
    dataset: str
        name of dataset ('Landsat', 'Sentinel-2_SR', 'Sentinel-2_TOA', 'PlanetScope')
    dataset_dict: dict
        dictionary of parameters for each dataset
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for searching and clipping images
    date_start: str
        start date for image search ('YYYY-MM-DD')
    date_end: str
        end date for image search ('YYYY-MM-DD')
    month_start: int
        starting month for calendar range filtering
    month_end: int
        ending month for calendar range filtering
    cloud_cover_max: int
        maximum image cloud cover percentage (0-100)
    mask_clouds: bool
        whether to mask clouds using geedim masking tools
    out_path: str
        path where images will be downloaded

    Returns
    __________
    im_ds_list: list
        list of xarray.Datasets, masked and filtered using AOI coverage
    '''

    # -----Reformat AOI for image filtering
    # reproject AOI to WGS
    AOI_WGS = AOI.to_crs('EPSG:4326')
    # solve for optimal UTM zone
    AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
    epsg_UTM = convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
    AOI_UTM = AOI_WGS.to_crs('EPSG:'+str(epsg_UTM))

    # -----Prepare AOI for querying geedim
    region = {'type': 'Polygon',
              'coordinates':[[[AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]]
                              ]]}

    # -----Query GEE for imagery
    print('Querying GEE for '+dataset+' imagery...')
    if dataset=='Landsat':
        # Landsat 8
        im_col_gd_8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(start_date=date_start, end_date=date_end, region=region,
                                                                                    cloudless_portion=100-cloud_cover_max, fill_portion=70)
        # Landsat 9
        im_col_gd_9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(start_date=date_start, end_date=date_end, region=region,
                                                                                    cloudless_portion=100-cloud_cover_max, fill_portion=70)
        # check if any images were found
        im_IDs = sorted(list(im_col_gd_8.properties) + list(im_col_gd_9.properties)) # grab list of image IDs
        if len(im_IDs) < 1:
            print('No images found, exiting...')
            return 'N/A'
        # remove images outside the month range
        im_IDs_filt = [im_ID for im_ID in im_IDs if (int(im_ID[-4:-2]) > month_start) and (int(im_ID[-4:-2]) < month_end)]
    elif dataset=='Sentinel-2_TOA':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(start_date=date_start, end_date=date_end, region=region,
                                                                                     cloudless_portion=100-cloud_cover_max, fill_portion=70)
        # check if any images were found
        im_IDs = sorted(list(im_col_gd.properties)) # grab list of image IDs
        if len(im_IDs) < 1:
            print('No images found. Exiting...')
            return 'N/A'
        # remove images outside the month range
        im_IDs_filt = [im_ID for im_ID in im_IDs if (int(im_ID.split('/')[-1][4:6]) > month_start) and (int(im_ID.split('/')[-1][4:6]) < month_end)]
    elif dataset=='Sentinel-2_SR':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(start_date=date_start, end_date=date_end, region=region,
                                                                                     cloudless_portion=100-cloud_cover_max, fill_portion=70)
        # check if any images were found
        im_IDs = sorted(list(im_col_gd.properties)) # grab list of image IDs
        if len(im_IDs) < 1:
            print('No images found. Exiting...')
            return 'N/A'
        # remove images outside the month range
        im_IDs_filt = [im_ID for im_ID in im_IDs if (int(im_ID.split('/')[-1][4:6]) > month_start) and (int(im_ID.split('/')[-1][4:6]) < month_end)]
    else:
        print("'dataset' variable not recognized or accepted. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. Exiting..." )
        return 'N/A'
    # check if any images are left after filtering for months
    print('Number of images found = '+str(len(im_IDs_filt)))
    if len(im_IDs_filt) < 1:
        print('Exiting...')
        return 'N/A'

    # -----Determine whether images must be downloaded (image sizes exceed GEE limit)
    im_download = True
    ### NOTE: wxee has a bug related to new xarray release (xarray.open_rasterio no longer functional) - must download all images
    # Calculate width and height of AOI bounding box [m]
    # AOI_UTM_bb_width = AOI_UTM.geometry[0].bounds[2] - AOI_UTM.geometry[0].bounds[0]
    # AOI_UTM_bb_height = AOI_UTM.geometry[0].bounds[3] - AOI_UTM.geometry[0].bounds[1]
    # # Check if bounding box is larger than GEE image outputs
    # if dataset=='Landsat':
    #     if ((AOI_UTM_bb_width / 30 * 8)*(AOI_UTM_bb_height / 30 * 8)) > 1e9:
    #         im_download = True
    #         print('Landsat mages must be downloaded for full spatial resolution')
    # if (dataset=='Sentinel-2_SR') or (dataset=='Sentinel-2_TOA'):
    #     if ((AOI_UTM_bb_width / 10 * 9)*(AOI_UTM_bb_height / 10 * 9)) > 1e9:
    #         im_download = True
    #         print('Sentinel-2_SR mages must be downloaded for full spatial resolution')

    # -----Convert image collection to list of xarray.Datasets
    # initialize list of xarray.Datasets
    im_ds_list = []
    if im_download:
        # Make directory for outputs (out_path) if it doesn't exist
        if os.path.exists(out_path)==False:
            os.mkdir(out_path)
        os.chdir(out_path)
        print('Downloading images to ' + out_path)
        # loop through image IDs
        for im_ID in tqdm(im_IDs_filt):
            # define image filename for saving
            im_fn = im_ID.split('/')[-1]+'.tif'
            # download if doesn't exist in directory
            if os.path.exists(out_path+im_fn)==False:
                # create gd.MaskedImage from image ID
                im_gd = gd.MaskedImage.from_id(im_ID, mask=mask_clouds)
                im_gd.download(out_path + im_fn, region=region, scale=dataset_dict[dataset]['resolution_m'],
                               bands=im_gd.refl_bands)
            else:
                print(im_fn+' already exists in directory, skipping...')
            # read in xarray.DataArray
            im_da = rxr.open_rasterio(im_fn)
            # reproject to optimal UTM zone (if necessary)
            im_da = im_da.rio.reproject('EPSG:'+str(epsg_UTM))
            # convert to xarray.DataSet
            im_ds = im_da.to_dataset('band')
            band_names = list(dataset_dict[dataset]['refl_bands'].keys())
            im_ds = im_ds.rename({i + 1: name for i, name in enumerate(band_names)})
            # account for image scalar and no data values
            im_ds = xr.where(im_ds != dataset_dict[dataset]['no_data_value'],
                             im_ds / dataset_dict[dataset]['image_scalar'], np.nan)
            # expand dimensions to include time
            im_dt = np.datetime64(datetime.datetime.fromtimestamp(im_da.attrs['system-time_start'] / 1000))
            im_ds = im_ds.expand_dims({'time':[im_dt]})
            # set CRS
            im_ds.rio.write_crs('EPSG:'+str(im_da.rio.crs.to_epsg()), inplace=True)
            # add xarray.Dataset to list
            im_ds_list.append(im_ds)

    else:
        # loop through image IDs
        for im_ID in tqdm(im_IDs_filt):
            # create gd.MaskedImage from image ID
            im_gd = gd.MaskedImage.from_id(im_col_id+'/'+im_ID, mask=mask_clouds, region=region)
            # create ee.Image and select bands
            im_ee = im_gd.ee_image#.select(L.refl_bands)
            # convert to xarray.Dataset
            im_ds = im_ee.wx.to_xarray()
            # remove no data values
            im_ds = xr.where(im_ds < 0, np.nan, im_ds)
            # reproject to optimal UTM zone
            im_ds = im_ds.rio.reproject('EPSG:'+str(epsg_UTM))
            im_ds.rio.write_crs('EPSG:'+str(epsg_UTM), inplace=True)
            # add xarray.Dataset to list
            im_ds_list.append(im_ds)

    return im_ds_list

        # mosaic images captured the same day
        # def merge_by_date(im_col):
        #     # convert image collection to a list
        #     imgList = im_col.toList(im_col.size())
        #     # driver function for mapping the unique dates
        #     def uniqueDriver(image):
        #         return ee.Image(image).date().format("YYYY-MM-dd")
        #     uniqueDates = imgList.map(uniqueDriver).distinct()
        #     # Driver function for mapping mosaics
        #     def mosaicDriver(date):
        #         date = ee.Date(date)
        #         image = (im_col
        #                .filterDate(date, date.advance(1, "day"))
        #                .mosaic())
        #         return image.set(
        #                         "system:time_start", date.millis(),
        #                         "system:id", date.format("YYYY-MM-dd")).clip(AOI_WGS_bb_ee.buffer(1000))
        #     mosaicImgList = uniqueDates.map(mosaicDriver)
        #     return ee.ImageCollection(mosaicImgList)
        # L_clip_mask_mosaic = merge_by_date(L_clip_mask)


# --------------------------------------------------
def PlanetScope_mask_image_pixels(im_path, im_fn, out_path, save_outputs, plot_results):
    ''' Mask PlanetScope 4-band image pixels using the Usable Data Mask (UDM) file associated with each image.

    Parameters
    ----------
    im_path: str
        path in directory to input images.
    im_fn: str
        file name of image to be masked, located in im_path.
    out_path: str
        path in directory where masked image will be saved.
    save_outputs: bool
        whether to save masked image tiff file to out_path
    plot_results: bool
        whether to plot resulting masked image (not saved to file).

    Returns
    ----------
    None

    '''

    # -----Create directory for outputs if it doesn't exist
    if save_outputs and os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print('made directory for output masked images:' + out_path)

    # -----Check if masked image already exists in file
    im_mask_fn = im_fn[0:15] + '_mask.tif'
    if os.path.exists(out_path + im_mask_fn):
        # print('Masked image already exists in directory. Skipping...')
        return

    # -----Open image
    os.chdir(im_path)
    im = rxr.open_rasterio(im_fn)
    # replace no data values with NaN
    im = im.where(im!=im._FillValue)
    # account for band scalar multiplier
    im_scalar = 1e4
    im = im / im_scalar

    # -----Create masked image
    im_string = im_fn[0:20]
    im_mask = im.copy() # copy image
    # determine which UDM file is associated with image
    if len(glob.glob(im_string + '*udm2*.tif')) > 0:
#        print('udm2 detected, applying mask...')
        im_udm_fn = glob.glob(im_string + '*udm2*.tif')[0]
        im_udm = rxr.open_rasterio(im_udm_fn)
        # loop through image bands
        for i in np.arange(0, len(im_mask.data)):
            # create mask (1 = usable pixels, 0 = unusable pixels)
            mask = np.where(((im_udm.data[2]==0) &  # shadow-free
                             (im_udm.data[4]==0) &  # heavy haze-free
                             (im_udm.data[5]==0)),  # cloud-free
                            1, 0)
            # apply mask to image
            im_mask.data[i] = np.where(mask==1, im.data[i], np.nan)

    # -----Save masked raster image to file
    if save_outputs:
        # assign attributes
        im_mask = im_mask.assign_attrs({'NoDataValue': '-9999',
                                        'Bands':{'1':'Blue', '2':'Green', '3':'Red', '4':'NIR'}})
        # reformat bands for saving as int data type
        for i in np.arange(0, len(im_mask.data)):
            # replace NaNs with -9999, multiply real values by image scalar
            im_mask.data[i] = np.where(~np.isnan(im_mask.data[i]), im_mask.data[i] * im_scalar, -9999)
        im_mask.data = im_mask.data.astype(int)
        # write to tiff file
        im_mask.rio.to_raster(out_path + im_mask_fn, dtype='int32')

    # -----Plot results
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(np.dstack([im.data[2], im.data[1], im.data[0]]))
        # set no data values to NaN, divide my im_scalar for plotting
        im_mask = im_mask.where(im_mask!=-9999) / im_scalar
        ax[1].imshow(np.dstack([im_mask.data[2], im_mask.data[1], im_mask.data[0]]))
        plt.show()


# --------------------------------------------------
def PlanetScope_mosaic_images_by_date(im_path, im_fns, out_path, AOI, plot_results):
    '''
    Mosaic PlanetScope images captured within the same hour using gdal_merge.py. Skips images which contain no real data in the AOI. Adapted from code developed by Jukes Liu.

    Parameters
    ----------
    im_path: str
        path in directory to input images.
    im_fns: list of strings
        file names of images to be mosaicked, located in im_path.
    out_path: str
        path in directory where image mosaics will be saved.
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest. If no real data exist within the AOI, function will exit. AOI must be in the same CRS as the images.
    plot_results: bool

    Returns
    ----------
    N/A

    '''

    # -----Check for spaces in file paths, replace with "\ " (spaces not accepted by subprocess commands)
    if (' ' in im_path) and ('\ ' not in im_path):
        im_path_adj = im_path.replace(' ','\ ')
    if (' ' in out_path) and ('\ ' not in out_path):
        out_path_adj = out_path.replace(' ','\ ')

    # -----Create output directory if it does not exist
    if os.path.isdir(out_path)==0:
        os.mkdir(out_path)
        print('Created directory for image mosaics: ' + out_path)

    # ----Grab all unique scenes (images captured within the same hour)
    os.chdir(im_path)
    unique_scenes = sorted(list(set([scene[0:11] for scene in im_fns])))

    # -----Loop through unique scenes
    for scene in tqdm(unique_scenes):

        # define the output file name with correct extension
        out_im_fn = os.path.join(scene + ".tif")

        # check if image mosaic file already exists
        if os.path.exists(out_path + out_im_fn)==False:

            file_paths = [] # files from the same hour to mosaic together
            for im_fn in im_fns: # check all files
                if (scene in im_fn): # if they match the scene datetime
                    # check if real data values exist within AOI
                    im = rio.open(os.path.join(im_path, im_fn)) # open image
                    AOI_reproj = AOI.to_crs('EPSG:'+str(im.crs.to_epsg())) # reproject AOI to image CRS
                    # mask the image using AOI geometry
                    b = im.read(1).astype(float) # blue band
                    mask = rio.features.geometry_mask(AOI_reproj.geometry,
                                                   b.shape,
                                                   im.transform,
                                                   all_touched=False,
                                                   invert=False)
                    b_AOI = b[mask==0] # grab blue band values within AOI
                    # set no-data values to NaN
                    b_AOI[b_AOI==-9999] = np.nan
                    b_AOI[b_AOI==0] = np.nan
                    if (len(b_AOI[~np.isnan(b_AOI)]) > 0):
                        file_paths.append(im_path_adj + im_fn) # add the path to the file

            # check if any filepaths were added
            if len(file_paths) > 0:

                # construct the gdal_merge command
                cmd = 'gdal_merge.py -v -n -9999 -a_nodata -9999 '

                # add input files to command
                for file_path in file_paths:
                    cmd += file_path+' '

                cmd += '-o ' + out_path_adj + out_im_fn

                # run the command
                p = subprocess.run(cmd, shell=True, capture_output=True)
                # print(p)


# --------------------------------------------------
def create_AOI_elev_polys(AOI, DEM):
    '''
    Function to generate a polygon of the top and bottom 20th percentile elevations
    within the defined Area of Interest (AOI).

    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        Area of interest used for masking images. Must be in same coordinate
        reference system (CRS) as the DEM.
    DEM: xarray.DataSet
        Digital elevation model. Must be in the same coordinate reference system
        (CRS) as the AOI.

    Returns
    ----------
    polygons_top: shapely.geometry.MultiPolygon
        Polygons outlining the top 20th percentile of elevations contour in the AOI.
    polygons_bottom: shapely.geometry.MultiPolygon
        Polygons outlining the bottom 20th percentile of elevations contour in the AOI.
    '''

    # -----Clip DEM to AOI
    DEM_AOI = DEM.rio.clip(AOI.geometry, AOI.crs)
    elevations = DEM_AOI.elevation.data

    # -----Calculate the threshold values for the percentiles
    percentile_bottom = np.nanpercentile(elevations, 20)
    percentile_top = np.nanpercentile(elevations, 80)

    # -----Bottom 20th percentile polygon
    mask_bottom = elevations <= percentile_bottom
    # find contours from the masked elevation data
    contours_bottom = find_contours(mask_bottom, 0.5)
    # interpolation functions for pixel to geographic coordinates
    fx = interp1d(range(0,len(DEM_AOI.x.data)), DEM_AOI.x.data)
    fy = interp1d(range(0,len(DEM_AOI.y.data)), DEM_AOI.y.data)
    # convert contour pixel coordinates to geographic coordinates
    polygons_bottom_list = []
    for contour in contours_bottom:
        # convert image pixel coordinates to real coordinates
        coords = (fx(contour[:,1]), fy(contour[:,0]))
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        polygons_bottom_list.append(Polygon(xy))
    # convert list of polygons to MultiPolygon
    polygons_bottom = MultiPolygon(polygons_bottom_list)

    # -----Top 20th percentile polygon
    top_mask = elevations >= percentile_top
    # find contours from the masked elevation data
    top_contours = find_contours(top_mask, 0.5)
    # convert contour pixel coordinates to geographic coordinates
    polygons_top_list = []
    for contour in top_contours:
        coords = (fx(contour[:,1]), fy(contour[:,0]))
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        polygons_top_list.append(Polygon(xy))
    # convert list of polygons to MultiPolygon
    polygons_top = MultiPolygon(polygons_top_list)

    return polygons_top, polygons_bottom


# --------------------------------------------------
def PlanetScope_adjust_image_radiometry(im_xr, im_dt, polygon_top, polygon_bottom, AOI, dataset_dict, dataset, site_name, skip_clipped):
    '''
    Adjust PlanetScope image band radiometry using the band values in a defined snow-covered area (SCA) and the expected surface reflectance of snow.

    Parameters
    ----------
    im_xr: xarray.DataSet
        input image with x and y coordinates and data variables containing bands values
    im_dt: numpy.datetime64
        datetime of image capture
    im_path: str
        path in directory to the input image
    polygon_top: shapely.geometry.polygon.Polygon
        polygon of the top 20th percentile of elevations in the AOI
    polygon_bottom: shapely.geometry.polygon.Polygon
        polygon of the bottom 20th percentile of elevations in the AOI
    AOI: geopandas.dataframe.DataFrame
        area of interest
    skip_clipped: bool
        whether to skip images where bands appear "clipped"

    Returns
    ----------
    im_adj: xarray.DataArray
        adjusted image
    im_adj_method: str
        method used to adjust image ('SNOW' = using the predicted surface reflectance of snow, 'ICE' = using the predicted surface reflectance of ice)
    '''

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Adjust input image values
    # set no data values to NaN
    im = im_xr.where(im_xr!=-9999)
    # account for image scalar multiplier if necessary
    if np.nanmean(np.ravel(im.band_data.data[0])) > 1e3:
        im = im / ds_dict['image_scalar']

    # define bands
    b = im.band_data.data[0]
    g = im.band_data.data[1]
    r = im.band_data.data[2]
    nir = im.band_data.data[3]

    # -----Return if image bands are likely clipped
    if skip_clipped==True:
        if ((np.nanmax(b) < 0.8) or (np.nanmax(g) < 0.8) or (np.nanmax(r) < 0.8)):
            print('image bands appear clipped... skipping.')
            im_adj_fn = 'N/A'
            return im_adj_fn

    # -----Return if image does not contain polygons
    # mask the image using polygon geometries
    mask_top = rio.features.geometry_mask([polygon_top],
                                   np.shape(b),
                                   im.rio.transform(),
                                   all_touched=False,
                                   invert=False)
    mask_bottom = rio.features.geometry_mask([polygon_bottom],
                                   np.shape(b),
                                   im.rio.transform(),
                                   all_touched=False,
                                   invert=False)
    # skip if image does not contain both polygons
    if (0 not in mask_top.flatten()) or (0 not in mask_bottom.flatten()):
        print('image does not contain polygons... skipping.')
        im_adj, im_adj_method = 'N/A', 'N/A'
        return im_adj, im_adj_method

    # -----Return if no real values exist within the polygons
    if (np.nanmean(b)==0) or (np.isnan(np.nanmean(b))):
#            print('image does not contain any real values within the polygon... skipping.')
        im_adj, im_adj_method = 'N/A', 'N/A'
        return im_adj, im_adj_method

    # -----Grab band values in the top elevations polygon
    b_top_polygon = b[mask_top==0]
    g_top_polygon = g[mask_top==0]
    r_top_polygon = r[mask_top==0]
    nir_top_polygon = nir[mask_top==0]

    # -----Grab band values in the bottom elevations polygon
    b_bottom_polygon = b[mask_bottom==0]
    g_bottom_polygon = g[mask_bottom==0]
    r_bottom_polygon = r[mask_bottom==0]
    nir_bottom_polygon = nir[mask_bottom==0]

    # -----Calculate median value for each polygon and the mean difference between the two
    SR_top_median = np.mean([np.nanmedian(b_top_polygon), np.nanmedian(g_top_polygon),
                               np.nanmedian(r_top_polygon), np.nanmedian(nir_top_polygon)])
    SR_bottom_median = np.mean([np.nanmedian(b_bottom_polygon), np.nanmedian(g_bottom_polygon),
                               np.nanmedian(r_bottom_polygon), np.nanmedian(nir_bottom_polygon)])
    difference = np.mean([np.nanmedian(b_top_polygon) - np.nanmedian(b_bottom_polygon),
                            np.nanmedian(g_top_polygon) - np.nanmedian(g_bottom_polygon),
                            np.nanmedian(r_top_polygon) - np.nanmedian(r_bottom_polygon),
                            np.nanmedian(nir_top_polygon) - np.nanmedian(nir_bottom_polygon)])
    if (SR_top_median < 0.45) and (difference < 0.1):
        im_adj_method = 'ICE'
    else:
        im_adj_method = 'SNOW'

    # -----Define the desired bright and dark surface reflectance values
    #       at the top elevations based on the method determined above
    if im_adj_method=='SNOW':
        # define desired SR values at the bright area and darkest point for each band
        # bright area
        bright_b_adj = 0.94
        bright_g_adj = 0.95
        bright_r_adj = 0.94
        bright_nir_adj = 0.78
        # dark point
        dark_adj = 0.0

    elif im_adj_method=='ICE':
        # define desired SR values at the bright area and darkest point for each band
        # bright area
        bright_b_adj = 0.58
        bright_g_adj = 0.59
        bright_r_adj = 0.57
        bright_nir_adj = 0.40
        # dark point
        dark_adj = 0.0

    # -----Adjust surface reflectance values
    # band_adjusted = band*A - B
    # A = (bright_adjusted - dark_adjusted) / (bright - dark)
    # B = (dark*bright_adjusted - bright*dark_adjusted) / (bright - dark)
    # blue band
    bright_b = np.nanmedian(b_top_polygon) # SR at bright point
    dark_b = np.nanmin(b) # SR at darkest point
    A = (bright_b_adj - dark_adj) / (bright_b - dark_b)
    B = (dark_b*bright_b_adj - bright_b*dark_adj) / (bright_b - dark_b)
    b_adj = (b * A) - B
    b_adj = np.where(b==0, np.nan, b_adj) # replace no data values with nan
    # green band
    bright_g = np.nanmedian(g_top_polygon) # SR at bright point
    dark_g = np.nanmin(g) # SR at darkest point
    A = (bright_g_adj - dark_adj) / (bright_g - dark_g)
    B = (dark_g*bright_g_adj - bright_g*dark_adj) / (bright_g - dark_g)
    g_adj = (g * A) - B
    g_adj = np.where(g==0, np.nan, g_adj) # replace no data values with nan
    # red band
    bright_r = np.nanmedian(r_top_polygon) # SR at bright point
    dark_r = np.nanmin(r) # SR at darkest point
    A = (bright_r_adj - dark_adj) / (bright_r - dark_r)
    B = (dark_r*bright_r_adj - bright_r*dark_adj) / (bright_r - dark_r)
    r_adj = (r * A) - B
    r_adj = np.where(r==0, np.nan, r_adj) # replace no data values with nan
    # nir band
    bright_nir = np.nanmedian(nir_top_polygon) # SR at bright point
    dark_nir = np.nanmin(nir) # SR at darkest point
    A = (bright_nir_adj - dark_adj) / (bright_nir - dark_nir)
    B = (dark_nir*bright_nir_adj - bright_nir*dark_adj) / (bright_nir - dark_nir)
    nir_adj = (nir * A) - B
    nir_adj = np.where(nir==0, np.nan, nir_adj) # replace no data values with nan

    # -----Compile adjusted bands in xarray.Dataset
    # create meshgrid of image coordinates
    x_mesh, y_mesh = np.meshgrid(im.x.data, np.flip(im.y.data))
    # create xarray.Dataset
    im_adj = xr.Dataset(
        data_vars = dict(
            blue = (['y', 'x'], b_adj),
            green = (['y', 'x'], g_adj),
            red = (['y', 'x'], r_adj),
            NIR = (['y', 'x'], nir_adj)
        ),
        coords=im.coords,
        attrs = dict(
            no_data_values = np.nan,
            image_scalar = 1
        )
    )
    # add NDSI band
    im_adj['NDSI'] = ((im_adj[ds_dict['NDSI_bands'][0]] - im_adj[ds_dict['NDSI_bands'][1]])
                       / (im_adj[ds_dict['NDSI_bands'][0]] + im_adj[ds_dict['NDSI_bands'][0]]))
    # add time dimension
    im_adj = im_adj.expand_dims(dim={'time':[im_dt]})

    return im_adj, im_adj_method


# --------------------------------------------------
def classify_image(im_xr, clf, feature_cols, crop_to_AOI, AOI, DEM, dataset, dataset_dict, site_name, im_classified_fn, out_path):
    '''
    Function to classify image collection using a pre-trained classifier

    Parameters
    ----------
    im_xr: xarray.Dataset
        stack of images
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier
    crop_to_AOI: bool
        whether to mask everywhere outside the AOI before classifying
    AOI: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked if crop_to_AOI==True. AOI must be in the same CRS as the image.
    DEM: xarray.Dataset
        digital elevation model over the AOI
    dataset: str
        name of dataset ('Landsat', 'Sentinel2_SR', 'Sentinel2_TOA', 'PlanetScope')
    dataset_dict: dict
        dictionary of parameters for each dataset
    site_name: str
        name of study site used for output file names
    im_classified_fn: str
        file name of classified image to be saved
    out_path: str
        path in directory where classified images will be saved

    Returns
    ----------
    im_classified_xr: xarray.Dataset
        classified image
    '''

    # -----Make output directory if it doesn't already exist
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print('Made output directory for classified images:' + out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Define image bands and capture date
    bands = [band for band in ds_dict['refl_bands'] if 'QA' not in band]
    im_date = np.datetime64(str(im_xr.time.data[0])[0:19], 'ns')
    print(im_date)

    # -----Crop image to the AOI
    if crop_to_AOI:
        im_AOI = im_xr.rio.clip(AOI.geometry, im_xr.rio.crs).squeeze(dim='time', drop=True)

    # -----Prepare image for classification
    # find indices of real numbers (no NaNs allowed in classification)
    ix = [np.where((np.isfinite(im_AOI[band].data) & ~np.isnan(im_AOI[band].data)), True, False) for band in bands]
    I_real = np.full(np.shape(im_AOI[bands[0]].data), True)
    for ixx in ix:
        I_real = I_real & ixx
    # create df of image band values
    df = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        df[col] = np.ravel(im_AOI[col].data[I_real])
    df = df.reset_index(drop=True)

    # -----Classify image
    if len(df)>1:
        array_classified = clf.predict(df[feature_cols])
    else:
        print("No real values found to classify, skipping...")
        return 'N/A'
    # reshape from flat array to original shape
    im_classified = np.zeros(im_AOI.to_array().data[0].shape)
    im_classified[:] = np.nan
    im_classified[I_real] = array_classified

    # -----Mask the DEM using the AOI
    DEM_AOI = DEM.rio.clip(AOI.geometry, DEM.rio.crs)

    # -----Convert numpy.array to xarray.Dataset
    # create xarray DataSet
    im_classified_xr = xr.Dataset(data_vars = dict(classified=(['y', 'x'], im_classified)),
                                  coords = im_AOI.coords,
                                  attrs = im_AOI.attrs)
    # set coordinate reference system (CRS)
    im_classified_xr = im_classified_xr.rio.write_crs(im_xr.rio.crs)

    # -----Determine snow covered elevations
    # interpolate DEM to image coordinates
    DEM_AOI_interp = DEM_AOI.interp(x=im_classified_xr.x.data, y=im_classified_xr.y.data, method='linear')
    # create array of elevation for all un-masked pixels
    all_elev = np.ravel(DEM_AOI_interp.elevation.data[~np.isnan(DEM_AOI_interp.elevation.data)])
    # create array of snow-covered pixel elevations
    snow_est_classified_mask = xr.where(im_classified_xr<=2, True, False).classified.data
    snow_est_elev = DEM_AOI_interp.elevation.data[snow_est_classified_mask]

    # -----Preprare classified image for saving
    # add elevation band
    im_classified_xr['elevation'] = (('y', 'x'), DEM_AOI_interp.elevation.data)
    # add time dimension
    im_classified_xr = im_classified_xr.expand_dims(dim={'time':[np.datetime64(im_date)]})
    # add additional attributes for description and classes
    im_classified_xr = im_classified_xr.assign_attrs({'Description':'Classified image',
                                                      'NoDataValues':'-9999',
                                                      'Classes':'1 = Snow, 2 = Shadowed snow, 3 = Firn, 4 = Ice, 5 = Rock, 6 = Water'})
    # replace NaNs with -9999, convert data types to int
    im_classified_xr_int = xr.where(np.isnan(im_classified_xr), -9999, im_classified_xr)
    im_classified_xr_int.classified.data = im_classified_xr_int.classified.data.astype(int)
    im_classified_xr_int.elevation.data = im_classified_xr_int.elevation.data.astype(int)

    # -----Save to file
    if '.nc' in im_classified_fn:
        im_classified_xr_int.to_netcdf(out_path + im_classified_fn)
    elif '.tif' in im_classified_fn:
        # remove time dimension
        # im_classified_xr_adj_int = im_classified_xr_adj_int.drop_dims('time')
        im_classified_xr_int.rio.to_raster(out_path + im_classified_fn)
    print('Classified image saved to file: ' + out_path + im_classified_fn)

    return im_classified_xr


# --------------------------------------------------
def delineate_image_snowline(im_xr, im_classified, site_name, AOI, dataset_dict, dataset, im_date, snowline_fn, out_path, figures_out_path, plot_results):
    '''
    Delineate snowline(s) in classified images. Snowlines will likely not be detected in images with nearly all or no snow.

    Parameters
    ----------
    im_xr: xarray.Dataset
        input image used for plotting
    im_classified: xarray.Dataset
        classified image used to delineate snowlines
    site_name: str
        name of study site used for output file names
    AOI:  geopandas.geodataframe.GeoDataFrame
        area of interest used to crop classified images
    ds_dict: dict
        dictionary of dataset-specific parameters
    dataset: str
        name of dataset ('Landsat', 'Sentinel2', 'PlanetScope')
    im_dt: str
        image capture datetime ('YYYYMMDDTHHmmss')
    snowline_fn: str
        file name of snowline to be saved in out_path
    out_path: str
        path in directory for output snowlines
    figures_out_path: str
        path in directory for figures

    Returns
    ----------
    snowline_gdf: geopandas.GeoDataFrame
        resulting study site name, image datetime, snowline coordinates, snowline elevations, and median snowline elevation
    '''

    # -----Make directory for snowlines (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print("Made directory for snowlines:" + out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Define image bands
    bands = [x for x in im_xr.data_vars]

    # -----Remove time dimension
    im_xr = im_xr.isel(time=0)
    im_classified = im_classified.isel(time=0)

    # -----Create no data mask
    no_data_mask = xr.where(np.isnan(im_classified), 1, 0).to_array().data[0]
    # convert to polygons
    no_data_polygons = []
    for s, value in rio.features.shapes(no_data_mask.astype(np.int16),
                                        mask=(no_data_mask > 0),
                                        transform=im_xr.rio.transform()):
        no_data_polygons.append(shape(s))
    no_data_polygons = MultiPolygon(no_data_polygons)

    # -----Determine snow covered elevations
    all_elev = np.ravel(im_classified.elevation.data)
    all_elev = all_elev[~np.isnan(all_elev)] # remove NaNs
    snow_est_elev = np.ravel(im_classified.where((im_classified.classified <=2))
                                          .where(im_classified.classified !=-9999).elevation.data)
    snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)] # remove NaNs

    # -----Create elevation histograms
    # determine bins to use in histograms
    elev_min = np.fix(np.nanmin(np.ravel(im_classified.elevation.data))/10)*10
    elev_max = np.round(np.nanmax(np.ravel(im_classified.elevation.data))/10)*10
    bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max-elev_min)/10 + 1))
    bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2
    # calculate elevation histograms
    H_elev = np.histogram(all_elev, bins=bin_edges)[0]
    H_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
    H_snow_est_elev_norm = H_snow_est_elev / H_elev

    # -----Make all pixels at elevation bins with >75% snow coverage = snow
    # determine elevation with > 75% snow coverage
    if len(np.where(H_snow_est_elev_norm > 0.75)[0]) > 1:
        elev_75_snow = bin_centers[np.where(H_snow_est_elev_norm > 0.75)[0][0]]
        # make a copy of im_classified for adjusting
        im_classified_adj = im_classified
        # set all pixels above the elev_75_snow to snow (1)
        im_classified_adj['classified']  = xr.where(im_classified_adj['elevation'] > elev_75_snow, 1, im_classified_adj['classified'])
        # H_snow_est_elev_norm[bin_centers >= elev_75_snow] = 1
    else:
        im_classified_adj = im_classified

    # -----Delineate snow lines
    # create binary snow matrix
    im_binary = xr.where(im_classified_adj  > 2, 1, 0)
    # apply median filter to binary image with kernel_size of 1 pixel (~30 m)
    im_binary_filt = im_binary['classified'].data
    # fill holes in binary image (0s within 1s = 1)
    im_binary_filt_no_holes = binary_fill_holes(im_binary_filt)
    # find contours at a constant value of 0.5 (between 0 and 1)
    contours = find_contours(im_binary_filt_no_holes, 0.5)
    # convert contour points to image coordinates
    contours_coords = []
    for contour in contours:
        # convert image pixel coordinates to real coordinates
        fx = interp1d(range(0,len(im_classified_adj.x.data)), im_classified_adj.x.data)
        fy = interp1d(range(0,len(im_classified_adj.y.data)), im_classified_adj.y.data)
        coords = (fx(contour[:,1]), fy(contour[:,0]))
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        contours_coords.append(xy)
    # create snow-free polygons
    c_polys = []
    for c in contours_coords:
        c_points = [Point(x,y) for x,y in c]
        c_poly = Polygon([[p.x, p.y] for p in c_points])
        c_polys = c_polys + [c_poly]
    # only save the largest polygon
    if len(c_polys) > 0:
        # calculate polygon areas
        areas = np.array([poly.area for poly in c_polys])
        # grab top 3 areas with their polygon indices
        areas_max = sorted(zip(areas, np.arange(0,len(c_polys))), reverse=True)[:1]
        # grab indices
        ic_polys = [x[1] for x in areas_max]
        # grab polygons at indices
        c_polys = [c_polys[i] for i in ic_polys]

    # extract coordinates in polygon
    polys_coords = [list(zip(c.exterior.coords.xy[0], c.exterior.coords.xy[1]))  for c in c_polys]
    # extract snow lines (sl) from contours
    # filter contours using no data and AOI masks (i.e., along glacier outline or data gaps)
    sl_est = [] # initialize list of snow lines
    min_sl_length = 100 # minimum snow line length
    for c in polys_coords:
        # create array of points
        c_points =  [Point(x,y) for x,y in c]
        # loop through points
        line_points = [] # initialize list of points to use in snow line
        for point in c_points:
            # calculate distance from the point to the no data polygons and the AOI boundary
            distance_no_data = no_data_polygons.distance(point)
            distance_AOI = AOI.boundary[0].distance(point)
            # only include points more than two pixels away from each mask
            if (distance_no_data > 60) and (distance_AOI > 60):
                line_points = line_points + [point]
        if line_points: # if list of line points is not empty
            if len(line_points) > 1: # must have at least two points to create a LineString
                line = LineString([(p.xy[0][0], p.xy[1][0]) for p in line_points])
                if line.length > min_sl_length:
                    sl_est = sl_est + [line]

    # -----Split lines with points more than 100 m apart and filter by length
    # check if any snow lines were found
    if sl_est:
        sl_est = sl_est[0]
        max_dist = 100 # m
        first_point = Point(sl_est.coords.xy[0][0], sl_est.coords.xy[1][0])
        points = [Point(sl_est.coords.xy[0][i], sl_est.coords.xy[1][i])
                  for i in np.arange(0,len(sl_est.coords.xy[0]))]
        isplit = [0] # point indices where to split the line
        for i, p in enumerate(points):
            if i!=0:
                dist = p.distance(points[i-1])
                if dist > max_dist:
                    isplit.append(i)
        isplit.append(len(points)) # add ending point to complete the last line
        sl_est_split = [] # initialize split lines
        # loop through split indices
        if len(isplit) > 1:
            for i, p in enumerate(isplit[:-1]):
                if isplit[i+1]-isplit[i] > 1: # must have at least two points to make a line
                    line = LineString(points[isplit[i]:isplit[i+1]])
                    if line.length > min_sl_length:
                        sl_est_split = sl_est_split + [line]
        else:
            sl_est_split = [sl_est]

        # -----Interpolate elevations at snow line coordinates
        # compile all line coordinates into arrays of x- and y-coordinates
        xpts, ypts = [], []
        for line in sl_est_split:
            xpts = xpts + [x for x in line.coords.xy[0]]
            ypts = ypts + [y for y in line.coords.xy[1]]
        xpts, ypts = np.array(xpts).flatten(), np.array(ypts).flatten()
        # interpolate elevation at snow line points
        sl_est_elev = [im_classified.sel(x=x, y=y, method='nearest').elevation.data
                       for x, y in list(zip(xpts, ypts))]

    else:
        sl_est_split = None
        sl_est_elev = np.nan

    # -----If no snowline exists and AOI is ~covered in snow, make sl_est_elev = min AOI elev
    if np.size(sl_est_elev)==1:
        if (np.isnan(sl_est_elev)) & (np.nanmedian(H_snow_est_elev_norm) > 0.5):
            sl_est_elev = elev_min
            sl_est_elev_median = elev_min

    # -----Calculate snow-covered area (SCA) and accumulation area ratio (AAR)
    # pixel resolution
    dx = im_classified.x.data[1]-im_classified.x.data[0]
    # snow-covered area
    SCA = len(np.ravel(im_classified.classified.data[im_classified.classified.data<=2]))*(dx**2) # number of snow-covered pixels * pixel resolution [m^2]
    # accumulation area ratio
    total_area = len(np.ravel(im_classified.classified.data[~np.isnan(im_classified.classified.data)]))*(dx**2) # number of pixels * pixel resolution [m^2]
    AAR = SCA / total_area

    # -----Compile results in dataframe
    # calculate median snow line elevation
    sl_est_elev_median = np.nanmedian(sl_est_elev)
    # compile results in df
    if np.size(sl_est_elev)==1:
        snowlines_coords_X = [[]]
        snowlines_coords_Y = [[]]
    else:
        snowlines_coords_X = [[x for x in sl_est.coords.xy[0]]]
        snowlines_coords_Y = [[y for y in sl_est.coords.xy[1]]]
    snowline_df = pd.DataFrame({'study_site': [site_name],
                                'datetime': [im_date],
                                'snowlines_coords_X': snowlines_coords_X,
                                'snowlines_coords_Y': snowlines_coords_Y,
                                'CRS': ['EPSG:'+str(im_xr.rio.crs.to_epsg())],
                                'snowlines_elevs_m': [sl_est_elev],
                                'snowlines_elevs_median_m': [sl_est_elev_median],
                                'SCA_m2': [SCA],
                                'AAR': [AAR],
                                'dataset': [dataset],
                                'geometry': [sl_est]
                               })

    # -----Save snowline df to file
    if 'pkl' in snowline_fn:
        snowline_df.to_pickle(out_path + snowline_fn)
        print('Snowline saved to file: ' + out_path + snowline_fn)
    elif 'csv' in snowline_fn:
        snowline_df.to_csv(out_path + snowline_fn)
        print('Snowline saved to file: ' + out_path + snowline_fn)
    else:
        print('Please specify snowline_fn with extension .pkl or .csv. Exiting...')
        return 'N/A'

    # -----Plot results
    if plot_results:
        contour = None
        fig, ax = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]})
        ax = ax.flatten()
        # define x and y limits
        xmin, xmax = AOI.geometry[0].buffer(100).bounds[0]/1e3, AOI.geometry[0].buffer(100).bounds[2]/1e3
        ymin, ymax = AOI.geometry[0].buffer(100).bounds[1]/1e3, AOI.geometry[0].buffer(100).bounds[3]/1e3
        # define colors for plotting
        colors = list(dataset_dict['classified_image']['class_colors'].values())
        cmp = matplotlib.colors.ListedColormap(colors)
        # RGB image
        ax[0].imshow(np.dstack([im_xr[ds_dict['RGB_bands'][0]].data,
                                im_xr[ds_dict['RGB_bands'][1]].data,
                                im_xr[ds_dict['RGB_bands'][2]].data]),
                     extent=(np.min(im_xr.x.data)/1e3, np.max(im_xr.x.data)/1e3, np.min(im_xr.y.data)/1e3, np.max(im_xr.y.data)/1e3))
        ax[0].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        # classified image
        ax[1].imshow(im_classified['classified'].data, cmap=cmp, vmin=1, vmax=6,
                     extent=(np.min(im_classified.x.data)/1e3, np.max(im_classified.x.data)/1e3,
                             np.min(im_classified.y.data)/1e3, np.max(im_classified.y.data)/1e3))
        # plot dummy points for legend
        ax[1].scatter(0, 0, color=colors[0], s=50, label='Snow')
        ax[1].scatter(0, 0, color=colors[1], s=50, label='Shadowed snow')
        ax[1].scatter(0, 0, color=colors[2], s=50, label='Ice')
        ax[1].scatter(0, 0, color=colors[3], s=50, label='Rock')
        ax[1].scatter(0, 0, color=colors[4], s=50, label='Water')
        ax[1].set_xlabel('Easting [km]')
        # AOI
        for j, geom in enumerate(AOI.geometry[0].boundary.geoms):
            if j==0:
                ax[0].plot([x/1e3 for x in geom.coords.xy[0]], [y/1e3 for y in geom.coords.xy[1]], '-k', linewidth=1, label='AOI')
            else:
                ax[0].plot([x/1e3 for x in geom.coords.xy[0]], [y/1e3 for y in geom.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
            ax[1].plot([x/1e3 for x in geom.coords.xy[0]], [y/1e3 for y in geom.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
        # reset x and y limits
        ax[0].set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        # image bands histogram
        h_b = ax[2].hist(im_xr[ds_dict['RGB_bands'][0]].data.flatten(), color='blue', histtype='step', linewidth=2, bins=100, label="blue")
        h_g = ax[2].hist(im_xr[ds_dict['RGB_bands'][1]].data.flatten(), color='green', histtype='step', linewidth=2, bins=100, label="green")
        h_r = ax[2].hist(im_xr[ds_dict['RGB_bands'][2]].data.flatten(), color='red', histtype='step', linewidth=2, bins=100, label="red")
        ax[2].set_xlabel("Surface reflectance")
        ax[2].set_ylabel("Pixel counts")
        ax[2].legend(loc='best')
        ax[2].grid()
        # normalized snow elevations histogram
        ax[3].bar(bin_centers, H_snow_est_elev_norm, width=(bin_centers[1]-bin_centers[0]), color=colors[0], align='center')
        ax[3].set_xlabel("Elevation [m]")
        ax[3].set_ylabel("Fraction snow-covered")
        ax[3].grid()
        ax[3].set_xlim(elev_min-10, elev_max+10)
        ax[3].set_ylim(0,1)
        # plot estimated snow line coordinates
        if sl_est_split!=None:
            for j, line  in enumerate(sl_est_split):
                if j==0:
                    ax[0].plot([x/1e3 for x in line.coords.xy[0]],
                               [y/1e3 for y in line.coords.xy[1]],
                               '-', color='#f768a1', label='sl$_{estimated}$')
                else:
                    ax[0].plot([x/1e3 for x in line.coords.xy[0]],
                               [y/1e3 for y in line.coords.xy[1]],
                               '-', color='#f768a1', label='_nolegend_')
                ax[1].plot([x/1e3 for x in line.coords.xy[0]],
                           [y/1e3 for y in line.coords.xy[1]],
                           '-', color='#f768a1', label='_nolegend_')
        # determine figure title and file name
        title = im_date.replace('-','').replace(':','') + '_' + site_name + '_' + dataset + '_snow-cover'
        # add legends
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.suptitle(title)
        fig.tight_layout()
        # save figure
        fig_fn = figures_out_path + title + '.png'
        fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
        print('Figure saved to file:' + fig_fn)

    return snowline_df


# --------------------------------------------------
def query_GEE_for_MODIS_SR(AOI, date_start, date_end, month_start, month_end, cloud_cover_max, ds_dict):
    '''
    Query Google Earth Engine for MODIS surface reflectance (SR) imagery from the Terra platform.

    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for searching and clipping images
    date_start: str
        start date for image search ('YYYY-MM-DD')
    date_end: str
        end date for image search ('YYYY-MM-DD')
    month_start: str
        starting month for calendar range filtering
    month_end: str
        ending month for calendar range filtering
    cloud_cover max: float
        maximum image cloud cover percentage (0-100)
    ds_dict: dict
        dictionary of dataset-specific parameters

    Returns
    ----------
    M_xr: xarray.Dataset
        resulting dataset of MODIS image results
    '''

    # reproject AOI to WGS for image searching
    AOI_WGS = AOI.to_crs('EPSG:4326')
    # solve for optimal UTM zone
    AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
    epsg_UTM = convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
    # reformat AOI for clipping images
    AOI_WGS_bb_ee = ee.Geometry.Polygon(
                            [[[AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]]]
                            ])
    def clip_image(im):
        return im.clip(AOI_WGS_bb_ee.buffer(1000))
    # Query GEE for imagery
    M = (ee.ImageCollection('MODIS/061/MOD09GA')
             .filterDate(ee.Date(date_start), ee.Date(date_end))
             .filter(ee.Filter.calendarRange(month_start, month_end, 'month'))
             .filterBounds(AOI_WGS_bb_ee))
    # define band names
    M_band_names = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07', 'state_1km']
    #  clip images to AOI and select bands
    M_clip = M.map(clip_image).select(M_band_names)
    print(M_clip.getInfo())
    # convert image collection to xarray Dataset
    M_xr = M_clip.wx.to_xarray(scale=ds_dict['resolution_m'], crs='EPSG:4326')
    # reproject to UTM CRS
    M_xr = M_xr.rio.reproject('EPSG:'+epsg_UTM)
    # replace no data values with NaN and account for image scalar
    for band in M_band_names:
        M_xr[band] = xr.where(M_xr[band] != ds_dict['no_data_value'],
                              M_xr[band] / ds_dict['SR_scalar'],
                              np.nan)
    # Add NDSI band
    NDSI_bands = ds_dict['NDSI']
    M_xr['NDSI'] = ((M_xr[NDSI_bands[0]] - M_xr[NDSI_bands[1]]) / (M_xr[NDSI_bands[0]] + M_xr[NDSI_bands[1]]))

    return M_xr


# --------------------------------------------------
def reduce_memory_usage(df, verbose=True):
    '''
    Reduce memory usage in pandas.DataFrame, from Bex T (2021): https://towardsdatascience.com/6-pandas-mistakes-that-silently-tell-you-are-a-rookie-b566a252e60d

    Parameters
    ----------
    df: pandas.DataFrame
        input dataframe
    verbose: bool
        whether to output verbage (default=True)

    Returns
    ----------
    df: pandas.DataFrame
        output dataframe with reduced memory usage
    '''
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
#                if (
#                    c_min > np.finfo(np.float16).min
#                    and c_max < np.finfo(np.float16).max
#                ):
#                    df[col] = df[col].astype(np.float16) # float16 not compatible with linalg
                if (#elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# --------------------------------------------------
def manual_snowline_filter_plot(sl_est_df, dataset_dict, L_im_path, PS_im_path, S2_SR_im_path, S2_TOA_im_path):
    '''
    Loop through full snowlines dataframe, plot associated image and snowline, display option to remove snowlines.

    Parameters
    ----------
    sl_est_df: pandas.DataFrame
        full, compiled dataframe of snowline CSV files
    dataset_dict: dict
        dictionary of parameters for each dataset
    L_im_path: str
        path in directory to raw Landsat images
    PS_im_path: str
        path in directory to PlanetScope image mosaics
    S2_SR_im_path: str
        path in directory to raw Sentnel-2 Surface Reflectance (SR) images
    S2_TOA_im_path: str
        path in directory to raw Sentinel-2 Top of Atmosphere reflectance (TOA) images

    Returns
    ----------
    checkboxes: list
        list of ipywidgets.widgets.widget_bool.Checkbox objects associated with each image for user input
    '''

    # -----Set the font size and checkbox size using CSS styling
    style = """
            <style>
            .my-checkbox input[type="checkbox"] {
                transform: scale(2.5); /* Adjust the scale factor as needed */
                margin-right: 20px; /* Adjust the spacing between checkbox and label as needed */
                margin-left: 20px;
            }
            .my-checkbox label {
                font-size: 24px; /* Adjust the font size as needed */
            }
            </style>
            """

    # -----Display instructions message
    print('Scroll through each snowline image and check boxes below "bad" snowlines to remove from time series.')
    print('When finished, proceed to next cell.')

    # -----Loop through snowlines
    checkboxes = [] # initalize list of heckboxes for user input
    for i in np.arange(0,len(sl_est_df)):

        print(' ')
        print(' ')

        # grab snowline coordinates
        if len(sl_est_df.iloc[i]['snowlines_coords_X']) > 2:
            sl_X = [float(x) for x in sl_est_df.iloc[i]['snowlines_coords_X'].replace('[','').replace(']','').split(', ')]
            sl_Y = [float(y) for y in sl_est_df.iloc[i]['snowlines_coords_Y'].replace('[','').replace(']','').split(', ')]
        # grab snowline date
        date = sl_est_df.iloc[i]['datetime']
        # grab snowline dataset
        dataset = sl_est_df.iloc[i]['dataset']
        print(date, dataset)

        # determine snowline image file name
        im_fn=None
        if dataset=='Landsat':
            im_fn = glob.glob(L_im_path + '*' + date.replace('-','')[0:8]+'.tif')
        elif dataset=='PlanetScope':
            im_fn = glob.glob(PS_im_path + date.replace('-','')[0:8]+'.tif')
        elif dataset=='Sentinel-2_SR':
            im_fn = glob.glob(S2_SR_im_path + date.replace('-','')[0:8] + '*.tif')
        elif dataset=='Sentinel-2_TOA':
            im_fn = glob.glob(S2_TOA_im_path + date.replace('-','')[0:8] + '*.tif')

        if im_fn:
            im_fn = im_fn[0]
        else:
            print('No image found in file')
            continue
        print(im_fn)

        # load image
        im_da = rxr.open_rasterio(im_fn)
        im_ds = im_da.to_dataset('band')
        band_names = list(dataset_dict[dataset]['refl_bands'].keys())
        im_ds = im_ds.rename({i + 1: name for i, name in enumerate(band_names)})
        im_ds = xr.where(im_ds!=dataset_dict[dataset]['no_data_value'],
                         im_ds / dataset_dict[dataset]['image_scalar'], np.nan)
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        RGB_bands = dataset_dict[dataset]['RGB_bands']
        ax.imshow(np.dstack([im_ds[RGB_bands[0]], im_ds[RGB_bands[1]], im_ds[RGB_bands[2]]]),
                  extent=(np.min(im_ds.x.data)/1e3, np.max(im_ds.x.data)/1e3,
                          np.min(im_ds.y.data)/1e3, np.max(im_ds.y.data)/1e3))
        if len(sl_est_df.iloc[i]['snowlines_coords_X']) > 2:
            ax.plot([x/1e3 for x in sl_X], [y/1e3 for y in sl_Y], '.m', markersize=2, label='snowline')
            ax.legend(loc='best')
        else:
            print('No snowline coordinates detected')
        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')
        ax.set_title(date)
        plt.show()

        # create and display checkbox
        checkbox = widgets.Checkbox(value=False, description='Remove snowline', indent=False)
        checkbox.add_class('my-checkbox')
        display(HTML(style))
        display(checkbox)

        # add checkbox to list of checkboxes
        checkboxes += [checkbox]

    return checkboxes


# --------------------------------------------------
def fourier_series_symb(x, f, n=0):
    """
    Creates a symbolic fourier series of order 'n'.

    Parameters
    ----------
    n: float
        Order of the fourier series
    x: numpy.array
        Independent variable
    f: float
        Frequency of the fourier series

    Returns
    ----------
    series: str
        symbolic fourier series of order 'n'
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


# --------------------------------------------------
def fourier_model(c, X):
    '''
    Generates a fourier series model using the given coefficients, evaluated at the input X values.

    Parameters
    ----------
    c: numpy.array
        vector containing the coefficients for the Fourier fit
    x: numpy.array
        x-values at which to evaluate the model

    Returns
    ----------
    ymod: numpy.array
        modeled y-values values at each x-value
    '''

    if len(c): # at least a1 and b1 coefficients exist
        # grab a0 and w coefficients
        a0 = c[0]
        w = c[-1]
        # list a and b coefficients in pairs, with a coeffs in one column, b coeffs in the second column
        coeff_pairs = list(zip(*[iter(c[1:])]*2))
        # separate a and b coefficients
        a_coeffs = [y[0] for y in coeff_pairs]
        b_coeffs = [y[1] for y in coeff_pairs]
        # construct the series
        series_a = np.zeros(len(X))
        for i, x in enumerate(X): # loop through x values
            series_a[i] = np.sum([y*np.cos((i+1)*x*w) for i, y in enumerate(a_coeffs)]) # sum the a terms
        series_b = np.zeros(len(X))
        for i, x in enumerate(X): # loop through x values
            series_b[i] = np.sum([y*np.sin((i+1)*x*w) for i, y in enumerate(b_coeffs)]) # sum the a terms
        ymod = [a0+a+b for a, b in list(zip(series_a, series_b))]
    else: # only a0 coefficient exists
        ymod = a0*np.ones(len(X))

    return ymod


# --------------------------------------------------
def optimized_fourier_model(X, Y, nyears, plot_results):
    '''
    Generate a modeled fit to input data using Fourier series. First, identify the ideal number of terms for the Fourier model using 100 Monte Carlo simulations. Then, solve for the mean value for each coefficient using 500 Monte Carlo simulations.

    Parameters
    ----------
    X: numpy.array
        independent variable
    Y: numpy.array
        dependent variable
    nyears: int
        number of years (or estimated periods in your data) used to determine the range of terms to test
    plot_results: bool
        whether to plot results

    Returns
    ----------
    Y_mod: numpy.array
        modeled y values evaluated at each X-value
    '''

    # -----Identify the ideal number of terms for the Fourier model using Monte Carlo simulations
    # set up variables and parameters
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series_symb(x, f=w, n=5)}

    nmc = 100 # number of Monte Carlo simulations
    pTrain = 0.9 # percent of data to use as training
    fourier_ns = [nyears-1, nyears, nyears+1]
    print('Conducting 100 Monte Carlo simulations to determine the ideal number of model terms...')

    # loop through possible number of terms
    df_terms = pd.DataFrame(columns=['fit_minus1_err', 'fit_err', 'fit_plus1_err'])
    for i in np.arange(0,nmc):

        # split into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=pTrain, shuffle=True)
        # fit fourier curves to the training data with varying number of coeffients
        fit_minus1 = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[0])},
                    x=X_train, y=Y_train).execute()
        fit = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[1])},
                    x=X_train, y=Y_train).execute()
        fit_plus1 = Fit({y: fourier_series_symb(x, f=w, n=fourier_ns[2])},
                    x=X_train, y=Y_train).execute()
        # fit models to testing data
        Y_pred_minus1 = fit_minus1.model(x=X_test, **fit_minus1.params).y
        Y_pred = fit.model(x=X_test, **fit.params).y
        Y_pred_plus1 = fit_plus1.model(x=X_test, **fit_plus1.params).y
        # calculate error, concatenate to df
        fit_minus1_err = np.abs(Y_test - Y_pred_minus1)
        fit_err = np.abs(Y_test - Y_pred)
        fit_plus1_err = np.abs(Y_test - Y_pred_plus1)
        result = pd.DataFrame({'fit_minus1_err': fit_minus1_err,
                               'fit_err': fit_err,
                               'fit_plus1_err': fit_plus1_err})
        # add results to df
        df_terms = pd.concat([df_terms, result])

    df_terms = df_terms.reset_index(drop=True)

    # plot results
    if plot_results:
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        ax[0].boxplot(fit_minus1_err);
        ax[0].set_title(str(fourier_ns[0]) + ' terms, N=' + str(len(fit_minus1_err)));
        ax[0].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[0].set_ylabel('Least absolute error')
        ax[1].boxplot(fit_err);
        ax[1].set_title(str(fourier_ns[1]) + ' terms, N=' + str(len(fit_err)));
        ax[1].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[2].boxplot(fit_plus1_err);
        ax[2].set_ylim(np.min([fit_minus1_err, fit_err, fit_plus1_err])-10, np.max([fit_minus1_err, fit_err, fit_plus1_err])+10)
        ax[2].set_title(str(fourier_ns[2]) + ' terms, N=' + str(len(fit_plus1_err)));
        plt.show()

    # calculate mean error for each number of coefficients
    fit_err_mean = [np.nanmean(df_terms['fit_minus1_err']), np.nanmean(df_terms['fit_err']), np.nanmean(df_terms['fit_plus1_err'])]
    # identify best number of coefficients
    Ibest = np.argmin(fit_err_mean)
    fit_best = [fit_minus1, fit, fit_plus1][Ibest]
    fourier_n = fourier_ns[Ibest]
    print('Optimal # of model terms = ' + str(fourier_n))
    print('Mean error = +/- ' + str(np.round(fit_err_mean[Ibest])) + ' m')

    # -----Conduct Monte Carlo simulations to generate 500 Fourier models
    nmc = 500 # number of monte carlo simulations
    # initialize coefficients data frame
    cols = [val[0] for val in fit_best.params.items()]
    X_mod = np.linspace(X[0], X[-1], num=100) # points at which to evaluate the model
    Y_mod = np.zeros((nmc, len(X_mod))) # array to hold modeled Y values
    Y_mod_err = np.zeros(nmc) # array to hold error associated with each model
    print('Conducting Monte Carlo simulations to generate 500 Fourier models...')
    # loop through Monte Carlo simulations
    for i in np.arange(0,nmc):

        # split into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=pTrain, shuffle=True)

        # fit fourier model to training data
        fit = Fit({y: fourier_series_symb(x, f=w, n=fourier_n)},
                    x=X_train, y=Y_train).execute()

#        print(str(i)+ ' '+ str(len(fit.params)))

        # apply fourier model to testing data
        Y_pred = fit.model(x=X_test, **fit.params).y

        # calculate mean error
        Y_mod_err[i] = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)

        # apply the model to the full X data
        c = [c[1] for c in fit.params.items()] # coefficient values
        Y_mod[i,:] = fourier_model(c, X_mod)

    # plot results
    if plot_results:
        Y_mod_iqr = iqr(Y_mod, axis=0)
        Y_mod_median = np.nanmedian(Y_mod, axis=0)
        Y_mod_P25 = Y_mod_median - Y_mod_iqr/2
        Y_mod_P75 = Y_mod_median + Y_mod_iqr/2

        fig, ax = plt.subplots(figsize=(10,6))
        plt.rcParams.update({'font.size':14})
        ax.fill_between(X_mod, Y_mod_P25, Y_mod_P75, facecolor='blue', alpha=0.5, label='model$_{IQR}$')
        ax.plot(X_mod, np.median(Y_mod, axis=0), '.-b', linewidth=1, label='model$_{median}$')
        ax.plot(X, Y, 'ok', markersize=5, label='data')
        ax.set_ylabel('Snowline elevation [m]')
        ax.set_xlabel('Days since first observation date')
        ax.grid()
        ax.legend(loc='best')
        plt.show()

    return X_mod, Y_mod, Y_mod_err
