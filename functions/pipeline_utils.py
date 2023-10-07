"""
Functions for image querying in Google Earth Engine, image adjustment, and snow detection in Landsat, Sentinel-2, and PlanetScope imagery
Rainey Aberle
2023
"""

import math
import geopandas as gpd
import pandas as pd
import ee
import geedim as gd
import requests
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString, Point, shape
import os
import xarray as xr
import numpy as np
import rasterio as rio
import rioxarray as rxr
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import glob
from tqdm.auto import tqdm
import datetime
from sklearn.exceptions import NotFittedError
import PIL
import io
import wxee as wx


# --------------------------------------------------
def convert_wgs_to_utm(lon: float, lat: float):
    """
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
        optimal UTM zone, e.g. "32606"
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


# --------------------------------------------------
def plot_xr_rgb_image(im_xr, rgb_bands):
    """Plot RGB image of xarray.DataSet

    Parameters
    ----------
    im_xr: xarray.DataSet
        dataset containing image bands in data variables with x and y coordinates.
        Assumes x and y coordinates are in units of meters.
    rgb_bands: List
        list of data variable names for RGB bands contained within the dataset, e.g. ['red', 'green', 'blue']

    Returns
    ----------
    fig: matplotlib.pyplot.figure
        figure handle for the resulting plot
    ax: matplotlib.pyplot.figure.Axes
        axis handle for the resulting plot
    """

    # -----Grab RGB bands from dataset
    if len(np.shape(im_xr[rgb_bands[0]].data)) > 2:  # check if need to take [0] of data
        red = im_xr[rgb_bands[0]].data[0]
        blue = im_xr[rgb_bands[1]].data[0]
        green = im_xr[rgb_bands[2]].data[0]
    else:
        red = im_xr[rgb_bands[0]].data
        blue = im_xr[rgb_bands[1]].data
        green = im_xr[rgb_bands[2]].data

    # -----Format datatype as float, rescale RGB pixel values from 0 to 1
    red, green, blue = red.astype(float), green.astype(float), blue.astype(float)
    im_min = np.nanmin(np.ravel([red, green, blue]))
    im_max = np.nanmax(np.ravel([red, green, blue]))
    red = ((red - im_min) * (1 / (im_max - im_min)))
    green = ((green - im_min) * (1 / (im_max - im_min)))
    blue = ((blue - im_min) * (1 / (im_max - im_min)))

    # -----Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(np.dstack([red, green, blue]),
              extent=(np.min(im_xr.x.data) / 1e3, np.max(im_xr.x.data) / 1e3, np.min(im_xr.y.data) / 1e3,
                      np.max(im_xr.y.data) / 1e3))
    ax.grid()
    ax.set_xlabel('Easting [km]')
    ax.set_ylabel('Northing [km]')

    return fig, ax


# --------------------------------------------------
def query_gee_for_dem(aoi_utm, base_path, site_name, out_path=None):
    """
    Query GEE for the ArcticDEM Mosaic (where there is coverage) or the NASADEM,
    clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM, reprojected to the optimal UTM zone
    base_path: str
        path to 'snow-cover-mapping/' used to load ArcticDEM_Mosaic_coverage.shp
    site_name: str
        name of site used for saving output files
    out_path: str
        path where DEM will be saved (if size exceeds GEE limit). Default = None.

    Returns
    ----------
    dem_ds: xarray.Dataset
        elevations extracted within the AOI
    """

    # -----Grab optimal UTM zone from AOI CRS
    epsg_utm = str(aoi_utm.crs.to_epsg())

    # -----Define some functions
    def adjust_data_vars(im_xr):
        if 'band_data' in im_xr.data_vars:
            im_xr = im_xr.rename({'band_data': 'elevation'})
        if 'band' in im_xr.dims:
            elev_data = im_xr.elevation.data[0]
            im_xr = im_xr.drop_dims('band')
            im_xr['elevation'] = (('y', 'x'), elev_data)
        return im_xr

    def ellipsoid_to_geoid_heights(ds, base_path, out_path, out_fn):
        print('Transforming elevations from the ellipsoid to the geoid...')

        # Load EGM96 model from file
        geoid_model_fn = os.path.join(base_path, 'inputs-outputs', 'us_nga_egm96_15.tif')
        geoid_model = xr.open_dataset(geoid_model_fn)

        # Resample geoid model to DEM coordinates
        geoid_model_resampled = geoid_model.interp(x=ds.x, y=ds.y, method='linear')
        geoid_height = geoid_model_resampled.band_data.data[0]

        # Subtract geoid heights from ds heights, add to dataset
        elevation_geoid = ds.elevation.data - geoid_height
        ds['elevation'] = (('y', 'x'), elevation_geoid)
        ds = ds.astype('float32')

        # Re-save to file with updated elevations
        ds.rio.to_raster(out_path + out_fn)
        print('DEM re-saved with elevations referenced to the EGM96 geoid.')

        return ds

    # -----Define output image names, check if already exists in directory
    arcticdem_fn = site_name + '_ArcticDEM_clip.tif'
    arcticdem_geoid_fn = site_name + '_ArcticDEM_clip_geoid.tif'
    nasadem_fn = site_name + '_NASADEM_clip.tif'
    if os.path.exists(os.path.join(out_path, arcticdem_geoid_fn)):
        print('Clipped ArcticDEM referenced to the geoid already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, arcticdem_geoid_fn))
        dem_ds = adjust_data_vars(dem_ds)
    elif os.path.exists(os.path.join(out_path, arcticdem_fn)):
        print('Clipped ArcticDEM already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, arcticdem_fn))
        dem_ds = adjust_data_vars(dem_ds)
        # transform elevations from ellipsoid to geoid, save to file
        dem_ds = ellipsoid_to_geoid_heights(dem_ds, base_path, out_path, arcticdem_geoid_fn)
    elif os.path.exists(os.path.join(out_path, nasadem_fn)):
        print('Clipped NASADEM already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, nasadem_fn))
        dem_ds = adjust_data_vars(dem_ds)
    else:  # if no DEM exists in directory, load from GEE

        # -----Reformat AOI for clipping DEM
        aoi_utm_buffer = aoi_utm.copy(deep=True)
        aoi_utm_buffer.geometry[0] = aoi_utm_buffer.geometry[0].buffer(1000)
        aoi_utm_buffer = aoi_utm_buffer.to_crs('EPSG:4326')
        region = {'type': 'Polygon',
                  'coordinates': [[
                      [aoi_utm_buffer.geometry.bounds.minx[0], aoi_utm_buffer.geometry.bounds.miny[0]],
                      [aoi_utm_buffer.geometry.bounds.maxx[0], aoi_utm_buffer.geometry.bounds.miny[0]],
                      [aoi_utm_buffer.geometry.bounds.maxx[0], aoi_utm_buffer.geometry.bounds.maxy[0]],
                      [aoi_utm_buffer.geometry.bounds.minx[0], aoi_utm_buffer.geometry.bounds.maxy[0]],
                      [aoi_utm_buffer.geometry.bounds.minx[0], aoi_utm_buffer.geometry.bounds.miny[0]]
                  ]]
                  }

        # -----Check for ArcticDEM coverage over AOI
        # load ArcticDEM_Mosaic_coverage.shp
        arcticdem_coverage_fn = 'ArcticDEM_Mosaic_coverage.shp'
        arcticdem_coverage = gpd.read_file(base_path + 'inputs-outputs/' + arcticdem_coverage_fn)
        # reproject to optimal UTM zone
        arcticdem_coverage_utm = arcticdem_coverage.to_crs('EPSG:' + str(epsg_utm))
        # check for intersection with AOI
        intersects = arcticdem_coverage_utm.geometry[0].intersects(aoi_utm.geometry[0])
        # use ArcticDEM if intersects==True
        if intersects:
            print('ArcticDEM coverage over AOI')
            dem = gd.MaskedImage.from_id('UMN/PGC/ArcticDEM/V3/2m_mosaic', region=region)
            dem_fn = arcticdem_fn  # file name for saving
            res = 10  # spatial resolution [m]
            elevation_source = 'ArcticDEM Mosaic (https://developers.google.com/earth-engine/datasets/catalog/UMN_PGC_ArcticDEM_V3_2m_mosaic)'
        else:
            print('No ArcticDEM coverage, using NASADEM')
            dem = gd.MaskedImage.from_id("NASA/NASADEM_HGT/001", region=region)
            dem_fn = nasadem_fn  # file name for saving
            res = 30  # spatial resolution [m]
            elevation_source = 'NASADEM (https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001)'

        # -----Download DEM and open as xarray.Dataset
        print('Downloading DEM to ' + out_path)
        # create out_path if it doesn't exist
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # download DEM
        dem.download(out_path + dem_fn, region=region, scale=res, crs="EPSG:4326")
        # read DEM as xarray.Dataset
        dem_ds = xr.open_dataset(out_path + dem_fn)
        dem_ds = adjust_data_vars(dem_ds)

        # -----If using ArcticDEM, transform elevations with respect to the geoid (rather than the ellipsoid)
        if 'ArcticDEM' in elevation_source:
            dem_ds = ellipsoid_to_geoid_heights(dem_ds, base_path, out_path, arcticdem_geoid_fn)

    # -----Reproject DEM to UTM
    dem_ds = dem_ds.rio.reproject('EPSG:' + epsg_utm)
    dem_ds = xr.where((dem_ds > 1e38) | (dem_ds <= -9999), np.nan, dem_ds)
    dem_ds = dem_ds.rio.write_crs('EPSG:' + epsg_utm)

    return dem_ds


# --------------------------------------------------
def query_gee_for_imagery(dataset_dict, dataset, aoi_utm, date_start, date_end, month_start, month_end,
                          cloud_cover_max, mask_clouds, out_path=None, im_download=False):
    """
    Query Google Earth Engine for Landsat 8 and 9 surface reflectance (SR), Sentinel-2 top of atmosphere (TOA) or SR imagery.
    Images captured within the hour will be mosaicked.

    Parameters
    __________
    dataset_dict: dict
        dictionary of parameters for each image product
    dataset: str
        name of dataset ('Landsat', 'Sentinel-2_SR', 'Sentinel-2_TOA', 'PlanetScope')
    aoi_utm: geopandas.geodataframe.GeoDataFrame
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
    im_download: bool
        whether to download images. Folder for downloads (out_path) must be specified.

    Returns
    __________
    im_xr_list: list
        list of xarray.Datasets over the AOI
    """

    # -----Grab optimal UTM zone from AOI CRS
    epsg_utm = str(aoi_utm.crs.to_epsg())

    # -----Reformat AOI for image filtering
    # reproject CRS from AOI to WGS
    aoi_wgs = aoi_utm.to_crs('EPSG:4326')
    # prepare AOI for querying geedim (AOI bounding box)
    region = {'type': 'Polygon',
              'coordinates': [[[aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]]
                               ]]}

    # -----Query GEE for imagery
    print('Querying GEE for ' + dataset + ' imagery...')
    if dataset == 'Landsat':
        # Landsat 8
        im_col_gd_8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=100 - cloud_cover_max,
                                                                                     mask=mask_clouds,
                                                                                     fill_portion=70)
        # Landsat 9
        im_col_gd_9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=100 - cloud_cover_max,
                                                                                     mask=mask_clouds,
                                                                                     fill_portion=70)
    elif dataset == 'Sentinel-2_TOA':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=100 - cloud_cover_max,
                                                                                     mask=mask_clouds,
                                                                                     fill_portion=70)
    elif dataset == 'Sentinel-2_SR':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(start_date=date_start,
                                                                                        end_date=date_end,
                                                                                        region=region,
                                                                                        cloudless_portion=100 - cloud_cover_max,
                                                                                        mask=mask_clouds,
                                                                                        fill_portion=70)
    else:
        print("'dataset' variable not recognized. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. "
              "Exiting...")
        return 'N/A'

    # -----Create list of image IDs to download, include those that will be composited
    # define function to create list of IDs to be mosaicked
    def image_mosaic_ids(im_col_gd):
        # create lists of image properties
        try:
            properties = im_col_gd.properties
        except Exception as e:
            exc_id = str(e).split('ID=')[1].split(')')[0]
            print('Error accessing image ID: ' + exc_id + '. Exiting...')
            return 'N/A', 'N/A'
        ims = dict(properties).keys()
        im_ids = [properties[im]['system:id'] for im in ims]
        # return if no images found
        if len(im_ids) < 1:
            return 'N/A', 'N/A'
        im_dts = np.array(
            [datetime.datetime.utcfromtimestamp(properties[im]['system:time_start'] / 1000) for im in ims])
        # remove image datetimes and IDs outside the specified month range
        i = [int(ii) for ii in np.arange(0, len(im_dts)) if
             (im_dts[ii].month >= month_start) and (im_dts[ii].month <= month_end)]  # indices of images to keep
        im_dts, im_ids = [im_dts[ii] for ii in i], [im_ids[ii] for ii in i]  # subset of image datetimes and IDs
        # return if no images remain after filtering by month range
        if len(im_dts) < 1:
            return 'N/A', 'N/A'
        # grab all unique hours in image datetimes
        hours = np.array(im_dts, dtype='datetime64[h]')
        unique_hours = sorted(set(hours))
        # create list of IDs for each unique hour
        im_ids_list, im_dts_list = [], []
        for unique_hour in unique_hours:
            i = list(np.ravel(np.argwhere(hours == unique_hour)))
            im_ids_list_hour = [im_ids[ii] for ii in i]
            im_ids_list.append(im_ids_list_hour)
            im_dts_list_hour = [im_dts[ii] for ii in i]
            im_dts_list.append(im_dts_list_hour)

        return im_ids_list, im_dts_list

    # extract list of IDs to be mosaicked
    if dataset == 'Landsat':  # must run for Landsat 8 and 9 separately
        im_ids_list_8, im_dts_list_8 = image_mosaic_ids(im_col_gd_8)
        im_ids_list_9, im_dts_list_9 = image_mosaic_ids(im_col_gd_9)
        # check which collections found images
        if (type(im_ids_list_8) is str) and (type(im_ids_list_9) is str):
            im_ids_list, im_dts_list = 'N/A', 'N/A'
        elif type(im_ids_list_9) is str:
            im_ids_list, im_dts_list = im_ids_list_8, im_dts_list_8
        elif type(im_ids_list_8) is str:
            im_ids_list, im_dts_list = im_ids_list_9, im_dts_list_9
        else:
            im_ids_list = im_ids_list_8 + im_ids_list_9
            im_dts_list = im_dts_list_8 + im_dts_list_9
    else:
        im_ids_list, im_dts_list = image_mosaic_ids(im_col_gd)
    # check if any images were found after filtering by month and determine mosaic IDs
    if type(im_ids_list) is str:
        print('No images found or error in one or more image IDs, exiting...')
        return 'N/A'

    # -----Determine whether images must be downloaded (if image sizes exceed GEE limit)
    # Calculate width and height of AOI bounding box [m]
    aoi_utm_bb_width = aoi_utm.geometry[0].bounds[2] - aoi_utm.geometry[0].bounds[0]
    aoi_utm_bb_height = aoi_utm.geometry[0].bounds[3] - aoi_utm.geometry[0].bounds[1]
    # Check if number of pixels in each image exceeds GEE limit
    res = dataset_dict[dataset]['resolution_m']
    num_bands = len(dataset_dict[dataset]['refl_bands'])
    if ((aoi_utm_bb_width / res * num_bands) * (aoi_utm_bb_height / res * num_bands)) > 1e8:
        im_download = True
        print(dataset + ' images must be downloaded for full spatial resolution')
    else:
        print('No image downloads necessary, ' + dataset + ' images over the AOI are within the GEE limit.')
    if (out_path is None) & im_download:
        print('Variable out_path must be specified to download images. Exiting...')
        return 'N/A'

    # -----Create list of xarray.Datasets from list of image IDs
    im_xr_list = []  # initialize list of xarray.Datasets
    # loop through image IDs
    for i in tqdm(range(0, len(im_ids_list))):

        # subset image IDs and image datetimes
        im_ids, im_dts = im_ids_list[i], im_dts_list[i]

        # if images must be downloaded, use geedim
        if im_download:

            # make directory for outputs (out_path) if it doesn't exist
            if not os.path.exists(out_path):
                os.mkdir(out_path)
                print('Made directory for image downloads: ' + out_path)
            # define filename
            if len(im_ids) > 1:
                im_fn = dataset + '_' + str(im_dts[0]).replace('-', '')[0:8] + '_MOSAIC.tif'
            else:
                im_fn = dataset + '_' + str(im_dts[0]).replace('-', '')[0:8] + '.tif'
            # check file does not already exist in directory, download
            if not os.path.exists(os.path.join(out_path, im_fn)):
                # create list of MaskedImages from IDs
                im_gd_list = [gd.MaskedImage.from_id(im_id) for im_id in im_ids]
                # combine into new MaskedCollection
                im_collection = gd.MaskedCollection.from_list(im_gd_list)
                # create image composite
                im_composite = im_collection.composite(method=gd.CompositeMethod.q_mosaic,
                                                       mask=mask_clouds,
                                                       region=region)
                # download to file
                im_composite.download(os.path.join(out_path, im_fn),
                                      region=region,
                                      scale=dataset_dict[dataset]['resolution_m'],
                                      crs='EPSG:' + epsg_utm,
                                      dtype='float32',
                                      bands=im_composite.refl_bands)
            # load image from file
            im_da = rxr.open_rasterio(os.path.join(out_path, im_fn))
            # convert to xarray.DataSet
            im_ds = im_da.to_dataset('band')
            band_names = list(dataset_dict[dataset]['refl_bands'].keys())
            im_ds = im_ds.rename({i + 1: name for i, name in enumerate(band_names)})
            # account for image scalar and no data values
            im_ds = xr.where(im_ds != dataset_dict[dataset]['no_data_value'],
                             im_ds / dataset_dict[dataset]['image_scalar'], np.nan)
            # add time dimension
            im_dt = np.datetime64(datetime.datetime.fromtimestamp(im_da.attrs['system-time_start'] / 1000))
            im_ds = im_ds.expand_dims({'time': [im_dt]})
            # set CRS
            im_ds.rio.write_crs('EPSG:' + str(im_da.rio.crs.to_epsg()), inplace=True)
            # add xarray.Dataset to list
            im_xr_list.append(im_ds)

        else:  # if no image downloads necessary, use wxee

            # if more than one ID, composite images
            if len(im_dts) > 1:
                # create list of MaskedImages from IDs
                ims_gd = [gd.MaskedImage.from_id(im_id, mask=mask_clouds, region=region) for im_id in im_ids]
                # convert to list of ee.Images
                ims_ee = [ee.Image(im_gd.ee_image).select(im_gd.refl_bands) for im_gd in ims_gd]
                # convert to xarray.Datasets
                ims_xr = [im_ee.wx.to_xarray(scale=res, region=region, crs='EPSG:' + epsg_utm) for im_ee in ims_ee]
                # composite images
                ims_xr_composite = xr.merge(ims_xr, compat='override')
                # account for image scalar
                ims_xr_composite = xr.where(ims_xr_composite != dataset_dict[dataset]['no_data_value'],
                                            ims_xr_composite / dataset_dict[dataset]['image_scalar'], np.nan)
                # set CRS
                ims_xr_composite.rio.write_crs('EPSG:' + epsg_utm, inplace=True)
                # append to list of xarray.Datasets
                im_xr_list.append(ims_xr_composite)
            else:
                # create MaskedImage from ID
                im_gd = gd.MaskedImage.from_id(im_ids[0], mask=mask_clouds, region=region)
                # convert to ee.Image
                im_ee = ee.Image(im_gd.ee_image).select(im_gd.refl_bands)
                # convert to xarray.Datasets
                im_xr = im_ee.wx.to_xarray(scale=res, region=region, crs='EPSG:' + epsg_utm)
                # account for image scalar
                im_xr = xr.where(im_xr != dataset_dict[dataset]['no_data_value'],
                                 im_xr / dataset_dict[dataset]['image_scalar'], np.nan)
                # set CRS
                im_xr.rio.write_crs('EPSG:' + epsg_utm, inplace=True)
                # append to list of xarray.Datasets
                im_xr_list.append(im_xr)

    return im_xr_list


# --------------------------------------------------
def planetscope_mask_image_pixels(im_path, im_fn, out_path, save_outputs, plot_results):
    """
    Mask PlanetScope 4-band image pixels using the Usable Data Mask (UDM) file associated with each image.

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

    """

    # -----Create directory for outputs if it doesn't exist
    if save_outputs and (not os.path.exists(out_path)):
        os.mkdir(out_path)
        print('made directory for output masked images:' + out_path)

    # -----Check if masked image already exists in file
    im_mask_fn = im_fn[0:15] + '_mask.tif'
    if os.path.exists(os.path.join(out_path, im_mask_fn)):
        # print('Masked image already exists in directory. Skipping...')
        return

    # -----Open image
    os.chdir(im_path)
    im = rxr.open_rasterio(im_fn)
    # replace no data values with NaN
    im = im.where(im != im._FillValue)
    # account for band scalar multiplier
    im_scalar = 1e4
    im = im / im_scalar

    # -----Create masked image
    im_string = im_fn[0:20]
    im_mask = im.copy()  # copy image
    # determine which UDM file is associated with image
    if len(glob.glob(im_string + '*udm2*.tif')) > 0:
        #        print('udm2 detected, applying mask...')
        im_udm_fn = glob.glob(im_string + '*udm2*.tif')[0]
        im_udm = rxr.open_rasterio(im_udm_fn)
        # loop through image bands
        for i in np.arange(0, len(im_mask.data)):
            # create mask (1 = usable pixels, 0 = unusable pixels)
            mask = np.where(((im_udm.data[2] == 0) &  # shadow-free
                             (im_udm.data[4] == 0) &  # heavy haze-free
                             (im_udm.data[5] == 0)),  # cloud-free
                            1, 0)
            # apply mask to image
            im_mask.data[i] = np.where(mask == 1, im.data[i], np.nan)

    # -----Save masked raster image to file
    if save_outputs:
        # assign attributes
        im_mask = im_mask.assign_attrs({'NoDataValue': '-9999',
                                        'Bands': {'1': 'Blue', '2': 'Green', '3': 'Red', '4': 'NIR'}})
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
        im_mask = im_mask.where(im_mask != -9999) / im_scalar
        ax[1].imshow(np.dstack([im_mask.data[2], im_mask.data[1], im_mask.data[0]]))
        plt.show()


# --------------------------------------------------
def planetscope_mosaic_images_by_date(im_path, im_fns, out_path, aoi):
    """
    Mosaic PlanetScope images captured within the same hour using gdal_merge.py.
    Skips images which contain no real data in the AOI. Adapted from code developed by Jukes Liu.

    Parameters
    ----------
    im_path: str
        path in directory to input images.
    im_fns: list of strings
        file names of images to be mosaicked, located in im_path.
    out_path: str
        path in directory where image mosaics will be saved.
    aoi: geopandas.geodataframe.GeoDataFrame
        area of interest. If no real data exist within the AOI, function will exit. AOI must be in the same CRS as the images.

    Returns
    ----------
    N/A

    """

    # -----Check for spaces in file paths, replace with "\ " (spaces not accepted by subprocess commands)
    if (' ' in im_path) and ('\ ' not in im_path):
        im_path_adj = im_path.replace(' ', '\ ')
    if (' ' in out_path) and ('\ ' not in out_path):
        out_path_adj = out_path.replace(' ', '\ ')

    # -----Create output directory if it does not exist
    if os.path.isdir(out_path) == 0:
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
        if not os.path.exists(os.path.join(out_path, out_im_fn)):

            file_paths = []  # files from the same hour to mosaic together
            for im_fn in im_fns:  # check all files
                if scene in im_fn:  # if they match the scene datetime
                    # check if real data values exist within AOI
                    im = rio.open(os.path.join(im_path, im_fn))  # open image
                    aoi_reproj = aoi.to_crs('EPSG:' + str(im.crs.to_epsg()))  # reproject AOI to image CRS
                    # mask the image using AOI geometry
                    b = im.read(1).astype(float)  # blue band
                    mask = rio.features.geometry_mask(aoi_reproj.geometry,
                                                      b.shape,
                                                      im.transform,
                                                      all_touched=False,
                                                      invert=False)
                    b_aoi = b[mask == 0]  # grab blue band values within AOI
                    # set no-data values to NaN
                    b_aoi[b_aoi == -9999] = np.nan
                    b_aoi[b_aoi == 0] = np.nan
                    if len(b_aoi[~np.isnan(b_aoi)]) > 0:
                        file_paths.append(im_path_adj + im_fn)  # add the path to the file

            # check if any filepaths were added
            if len(file_paths) > 0:

                # construct the gdal_merge command
                cmd = 'gdal_merge.py -v -n -9999 -a_nodata -9999 '

                # add input files to command
                for file_path in file_paths:
                    cmd += file_path + ' '

                cmd += '-o ' + out_path_adj + out_im_fn

                # run the command
                subprocess.run(cmd, shell=True, capture_output=True)


# --------------------------------------------------
def create_aoi_elev_polys(aoi, dem):
    """
    Function to generate a polygon of the top and bottom 20th percentile elevations
    within the defined Area of Interest (AOI).

    Parameters
    ----------
    aoi: geopandas.geodataframe.GeoDataFrame
        Area of interest used for masking images. Must be in same coordinate
        reference system (CRS) as the DEM.
    dem: xarray.DataSet
        Digital elevation model. Must be in the same coordinate reference system
        (CRS) as the AOI.

    Returns
    ----------
    polygons_top: shapely.geometry.MultiPolygon
        Polygons outlining the top 20th percentile of elevations contour in the AOI.
    polygons_bottom: shapely.geometry.MultiPolygon
        Polygons outlining the bottom 20th percentile of elevations contour in the AOI.
    """

    # -----Clip DEM to AOI
    dem_aoi = dem.rio.clip(aoi.geometry, aoi.crs)
    elevations = dem_aoi.elevation.data

    # -----Calculate the threshold values for the percentiles
    percentile_bottom = np.nanpercentile(elevations, 20)
    percentile_top = np.nanpercentile(elevations, 80)

    # -----Bottom 20th percentile polygon
    mask_bottom = elevations <= percentile_bottom
    # find contours from the masked elevation data
    contours_bottom = find_contours(mask_bottom, 0.5)
    # interpolation functions for pixel to geographic coordinates
    fx = interp1d(range(0, len(dem_aoi.x.data)), dem_aoi.x.data)
    fy = interp1d(range(0, len(dem_aoi.y.data)), dem_aoi.y.data)
    # convert contour pixel coordinates to geographic coordinates
    polygons_bottom_list = []
    for contour in contours_bottom:
        # convert image pixel coordinates to real coordinates
        coords = (fx(contour[:, 1]), fy(contour[:, 0]))
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
        coords = (fx(contour[:, 1]), fy(contour[:, 0]))
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        polygons_top_list.append(Polygon(xy))
    # convert list of polygons to MultiPolygon
    polygons_top = MultiPolygon(polygons_top_list)

    return polygons_top, polygons_bottom


# --------------------------------------------------
def planetscope_adjust_image_radiometry(im_xr, im_dt, polygon_top, polygon_bottom, dataset_dict, skip_clipped):
    """
    Adjust PlanetScope image band radiometry using the band values in a defined snow-covered area (SCA) and the expected surface reflectance of snow.

    Parameters
    ----------
    im_xr: xarray.DataSet
        input image with x and y coordinates and data variables containing bands values
    im_dt: numpy.datetime64
        datetime of image capture
    polygon_top: shapely.geometry.polygon.Polygon
        polygon of the top 20th percentile of elevations in the AOI
    polygon_bottom: shapely.geometry.polygon.Polygon
        polygon of the bottom 20th percentile of elevations in the AOI
    dataset_dict: dict
        dictionary of parameters for each dataset
    skip_clipped: bool
        whether to skip images where bands appear "clipped"

    Returns
    ----------
    im_adj: xarray.DataArray
        adjusted image
    im_adj_method: str
        method used to adjust image ('SNOW' = using the predicted surface reflectance of snow, 'ICE' = using the predicted surface reflectance of ice)
    """

    # -----Subset dataset_dict to dataset
    dataset = "PlanetScope"
    ds_dict = dataset_dict[dataset]

    # -----Adjust input image values
    # set no data values to NaN
    im = im_xr.where(im_xr != -9999)
    # account for image scalar multiplier if necessary
    if np.nanmean(np.ravel(im.band_data.data[0])) > 1e3:
        im = im / ds_dict['image_scalar']

    # define bands
    b = im.band_data.data[0]
    g = im.band_data.data[1]
    r = im.band_data.data[2]
    nir = im.band_data.data[3]

    # -----Return if image bands are likely clipped
    if skip_clipped:
        if (np.nanmax(b) < 0.8) or (np.nanmax(g) < 0.8) or (np.nanmax(r) < 0.8):
            print('Image bands appear clipped... skipping.')
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
        print('Image does not contain polygons... skipping.')
        im_adj, im_adj_method = 'N/A', 'N/A'
        return im_adj, im_adj_method

    # -----Return if no real values exist within the polygons
    if (np.nanmean(b) == 0) or (np.isnan(np.nanmean(b))):
        # print('image does not contain any real values within the polygon... skipping.')
        im_adj, im_adj_method = 'N/A', 'N/A'
        return im_adj, im_adj_method

    # -----Grab band values in the top elevations polygon
    b_top_polygon = b[mask_top == 0]
    g_top_polygon = g[mask_top == 0]
    r_top_polygon = r[mask_top == 0]
    nir_top_polygon = nir[mask_top == 0]

    # -----Grab band values in the bottom elevations polygon
    b_bottom_polygon = b[mask_bottom == 0]
    g_bottom_polygon = g[mask_bottom == 0]
    r_bottom_polygon = r[mask_bottom == 0]
    nir_bottom_polygon = nir[mask_bottom == 0]

    # -----Calculate median value for each polygon and the mean difference between the two
    sr_top_median = np.mean([np.nanmedian(b_top_polygon), np.nanmedian(g_top_polygon),
                             np.nanmedian(r_top_polygon), np.nanmedian(nir_top_polygon)])
    difference = np.mean([np.nanmedian(b_top_polygon) - np.nanmedian(b_bottom_polygon),
                          np.nanmedian(g_top_polygon) - np.nanmedian(g_bottom_polygon),
                          np.nanmedian(r_top_polygon) - np.nanmedian(r_bottom_polygon),
                          np.nanmedian(nir_top_polygon) - np.nanmedian(nir_bottom_polygon)])
    if (sr_top_median < 0.45) and (difference < 0.1):
        im_adj_method = 'ICE'
    else:
        im_adj_method = 'SNOW'

    # -----Define the desired bright and dark surface reflectance values
    #       at the top elevations based on the method determined above
    if im_adj_method == 'SNOW':
        # define desired SR values at the bright area and darkest point for each band
        # bright area
        bright_b_adj = 0.94
        bright_g_adj = 0.95
        bright_r_adj = 0.94
        bright_nir_adj = 0.78
        # dark point
        dark_adj = 0.0

    elif im_adj_method == 'ICE':
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
    bright_b = np.nanmedian(b_top_polygon)  # SR at bright point
    dark_b = np.nanmin(b)  # SR at darkest point
    A = (bright_b_adj - dark_adj) / (bright_b - dark_b)
    B = (dark_b * bright_b_adj - bright_b * dark_adj) / (bright_b - dark_b)
    b_adj = (b * A) - B
    b_adj = np.where(b == 0, np.nan, b_adj)  # replace no data values with nan
    # green band
    bright_g = np.nanmedian(g_top_polygon)  # SR at bright point
    dark_g = np.nanmin(g)  # SR at darkest point
    A = (bright_g_adj - dark_adj) / (bright_g - dark_g)
    B = (dark_g * bright_g_adj - bright_g * dark_adj) / (bright_g - dark_g)
    g_adj = (g * A) - B
    g_adj = np.where(g == 0, np.nan, g_adj)  # replace no data values with nan
    # red band
    bright_r = np.nanmedian(r_top_polygon)  # SR at bright point
    dark_r = np.nanmin(r)  # SR at darkest point
    A = (bright_r_adj - dark_adj) / (bright_r - dark_r)
    B = (dark_r * bright_r_adj - bright_r * dark_adj) / (bright_r - dark_r)
    r_adj = (r * A) - B
    r_adj = np.where(r == 0, np.nan, r_adj)  # replace no data values with nan
    # nir band
    bright_nir = np.nanmedian(nir_top_polygon)  # SR at bright point
    dark_nir = np.nanmin(nir)  # SR at darkest point
    A = (bright_nir_adj - dark_adj) / (bright_nir - dark_nir)
    B = (dark_nir * bright_nir_adj - bright_nir * dark_adj) / (bright_nir - dark_nir)
    nir_adj = (nir * A) - B
    nir_adj = np.where(nir == 0, np.nan, nir_adj)  # replace no data values with nan

    # -----Compile adjusted bands in xarray.Dataset
    # create xarray.Dataset
    im_adj = xr.Dataset(
        data_vars=dict(
            Blue=(['y', 'x'], b_adj),
            Green=(['y', 'x'], g_adj),
            Red=(['y', 'x'], r_adj),
            NIR=(['y', 'x'], nir_adj)
        ),
        coords=im.coords,
        attrs=dict(
            no_data_values=np.nan,
            image_scalar=1
        )
    )
    # add NDSI band
    im_adj['NDSI'] = ((im_adj[ds_dict['NDSI_bands'][0]] - im_adj[ds_dict['NDSI_bands'][1]])
                      / (im_adj[ds_dict['NDSI_bands'][0]] + im_adj[ds_dict['NDSI_bands'][0]]))
    # add time dimension
    im_adj = im_adj.expand_dims(dim={'time': [im_dt]})

    return im_adj, im_adj_method


# --------------------------------------------------
def classify_image(im_xr, clf, feature_cols, crop_to_aoi, aoi, dataset_dict, dataset, im_classified_fn, out_path,
                   verbose=False):
    """
    Function to classify image collection using a pre-trained classifier

    Parameters
    ----------
    im_xr: xarray.Dataset
        input image
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier
    crop_to_aoi: bool
        whether to mask everywhere outside the AOI before classifying
    aoi: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked if crop_to_AOI==True.
        AOI must be in the same coordinate reference system as the image.
    dataset: str
        name of dataset ('Landsat', 'Sentinel2_SR', 'Sentinel2_TOA', 'PlanetScope')
    dataset_dict: dict
        dictionary of parameters for each dataset
    im_classified_fn: str
        file name of classified image to be saved
    out_path: str
        path in directory where classified images will be saved
    verbose: bool
        whether to print details while classifying each image

    Returns
    ----------
    im_classified_xr: xarray.Dataset
        classified image
    """

    # -----Make output directory if it doesn't already exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print('Made output directory for classified images:' + out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Define image bands and capture date
    bands = [band for band in ds_dict['refl_bands'] if 'QA' not in band]
    im_date = np.datetime64(str(im_xr.time.data[0])[0:19], 'ns')

    # -----Crop image to the AOI and remove time dimension
    if crop_to_aoi:
        im_aoi = im_xr.rio.clip(aoi.geometry, im_xr.rio.crs).isel(time=0)
    else:
        im_aoi = im_xr.isel(time=0)

    # -----Prepare image for classification
    # find indices of real numbers (no NaNs allowed in classification)
    ix = [np.where((np.isfinite(im_aoi[band].data) & ~np.isnan(im_aoi[band].data)), True, False) for band in bands]
    ireal = np.full(np.shape(im_aoi[bands[0]].data), True)
    # return if no real numbers in image
    if np.count_nonzero(ireal) == 0:
        if verbose:
            print("No real values found to classify, skipping...")
        return 'N/A'
    for ixx in ix:
        ireal = ireal & ixx
    # create df of image band values
    df = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        df[col] = np.ravel(im_aoi[col].data[ireal])
    df = df.reset_index(drop=True)

    # -----Classify image
    try:
        array_classified = clf.predict(df[feature_cols])
        # reshape from flat array to original shape
        im_classified = np.full(im_aoi.to_array().data[0].shape, np.nan)
        im_classified[ireal] = array_classified
    except NotFittedError:
        if verbose:
            print("Classifier is not fitted, skipping...")
        return 'N/A'
    except Exception as e:
        if verbose:
            print("Error occurred in classification:", str(e))
        return 'N/A'

    # -----Convert numpy.array to xarray.Dataset
    # create xarray DataSet
    im_classified_xr = xr.Dataset(data_vars=dict(classified=(['y', 'x'], im_classified)),
                                  coords=im_aoi.coords,
                                  attrs=im_aoi.attrs)
    # set coordinate reference system (CRS)
    im_classified_xr = im_classified_xr.rio.write_crs(im_xr.rio.crs)

    # -----Prepare classified image for saving
    # add time dimension
    im_classified_xr = im_classified_xr.expand_dims(dim={'time': [np.datetime64(im_date)]})
    # add additional attributes to image before saving
    im_classified_xr = im_classified_xr.assign_attrs({'Description': 'Classified image',
                                                      'Classes': '1 = Snow, 2 = Shadowed snow, 4 = Ice, 5 = Rock, 6 = Water',
                                                      'NoDataValues': '-9999'
                                                      })
    # replace NaNs with -9999, convert data types to int
    im_classified_xr_int = im_classified_xr.fillna(-9999).astype(int)
    # reproject to WGS84 horizontal coordinates for consistency before saving
    im_classified_xr_int = im_classified_xr_int.rio.reproject('EPSG:4326')

    # -----Save to file
    if '.nc' in im_classified_fn:
        im_classified_xr_int.to_netcdf(os.path.join(out_path, im_classified_fn))
    elif '.tif' in im_classified_fn:
        im_classified_xr_int.rio.to_raster(os.path.join(out_path, im_classified_fn))
    if verbose:
        print('Classified image saved to file: ' + os.path.join(out_path, im_classified_fn))

    return im_classified_xr


# --------------------------------------------------
def query_gee_for_image_thumbnail(dataset, dt, aoi_utm):

    # -----Grab datetime from snowline df
    date_start = str(dt - np.timedelta64(1, 'D'))
    date_end = str(dt + np.timedelta64(1, 'D'))

    # -----Buffer AOI by 1km
    aoi_utm_buffer = aoi_utm.buffer(1e3)
    # determine bounds for image plotting
    bounds = aoi_utm_buffer.geometry[0].bounds

    # -----Reformat AOI for image filtering
    # reproject CRS from AOI to WGS
    aoi_wgs = aoi_utm.to_crs('EPSG:4326')
    aoi_buffer_wgs = aoi_utm_buffer.to_crs('EPSG:4326')
    # prepare AOI for querying geedim (AOI bounding box)
    region = {'type': 'Polygon',
              'coordinates': [[[aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]]
                               ]]}
    region_buffer_ee = ee.Geometry.Polygon([[[aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.miny[0]],
                                              [aoi_buffer_wgs.geometry.bounds.maxx[0], aoi_buffer_wgs.geometry.bounds.miny[0]],
                                              [aoi_buffer_wgs.geometry.bounds.maxx[0], aoi_buffer_wgs.geometry.bounds.maxy[0]],
                                              [aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.maxy[0]],
                                              [aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.miny[0]]
                                            ]])

    # -----Query GEE for imagery
    if dataset == 'Landsat':
        # Landsat 8
        im_col_gd_8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     mask=True,
                                                                                     region=region,
                                                                                     fill_portion=50)
        # Landsat 9
        im_col_gd_9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     mask=True,
                                                                                     region=region,
                                                                                     fill_portion=50)
        im_col_ee = im_col_gd_8.ee_collection.merge(im_col_gd_9.ee_collection)

        # apply scaling factors
        def apply_scale_factors(image):
            opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

        im_col_ee = im_col_ee.map(apply_scale_factors)
        # define how to display image
        visualization = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0.0, 'max': 1.0, 'dimensions': 768,
                         'region': region_buffer_ee}
    elif dataset == 'Sentinel-2_TOA':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     mask=True,
                                                                                     region=region,
                                                                                     fill_portion=50)
        im_col_ee = im_col_gd.ee_collection
        # define how to display image
        visualization = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 1e4, 'dimensions': 768,
                         'region': region_buffer_ee}
    elif dataset == 'Sentinel-2_SR':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(start_date=date_start,
                                                                                        end_date=date_end,
                                                                                        mask=True,
                                                                                        region=region,
                                                                                        fill_portion=50)
        im_col_ee = im_col_gd.ee_collection
        # define how to display image
        visualization = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 1e4, 'dimensions': 768,
                         'region': region_buffer_ee}
    else:
        print("'dataset' variable not recognized. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. Exiting...")
        return 'N/A'

    # -----Display image, snowline, and AOI on geemap.Map()
    # Reproject the Earth Engine image to UTM projection
    utm_epsg = aoi_utm.crs.to_epsg()  # Get UTM EPSG code from AOI's CRS
    im_col_ee_utm = im_col_ee.map(lambda img: img.reproject(crs=f'EPSG:{utm_epsg}', scale=30))
    # Fetch the image URL from Google Earth Engine
    image_url = im_col_ee_utm.first().clip(region_buffer_ee).getThumbURL(visualization)
    # Fetch the image and convert it to a PIL Image object
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)
    image = PIL.Image.open(image_bytes)

    return image, bounds


# --------------------------------------------------
def delineate_snowline(im_classified, site_name, aoi, dem, dataset_dict, dataset, im_date, snowline_fn,
                       out_path, figures_out_path, plot_results, im_xr=None, verbose=False):
    """
    Delineate the seasonal snowline in classified images. Snowlines will likely not be detected in images with nearly all or no snow.

    Parameters
    ----------
    im_classified: xarray.Dataset
        classified image, used to delineate snowlines
    site_name: str
        name of study site, used for output file names
    aoi:  geopandas.geodataframe.GeoDataFrame
        area of interest, used to crop classified images
        must be in the same coordinate reference system as the classified image
    dem: xarray.Dataset
        digital elevation model over the aoi, used to calculate the ELA from the AAR
        must be in the same coordinate reference system as the classified image
    dataset_dict: dict
        dictionary of dataset-specific parameters
    dataset: str
        name of dataset ('Landsat', 'Sentinel2', 'PlanetScope')
    im_date: str
        image capture datetime (format: 'YYYYMMDDTHHmmss')
    snowline_fn: str
        file name of snowline to be saved in out_path
    out_path: str
        path to directory where output snowline will be saved
    figures_out_path: str
        path to directory where figure will be saved
    plot_results: bool
        whether to plot RGB image, classified image, and resulting snowline and save figure to file
    im_xr: xarray.Dataset
        input reflectance image, used for plotting
        if no image provided, will query GEE for image thumbnail
    verbose: bool
        whether to print details during the process

    Returns
    ----------
    snowline_gdf: geopandas.GeoDataFrame
        resulting study site name, image datetime, snowline coordinates, snowline elevations, and median snowline elevation
    """

    # -----Make directory for snowlines (if it does not already exist)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print("Made directory for snowlines:" + out_path)

    # -----Make directory for figures (if it does not already exist)
    if (not os.path.exists(figures_out_path)) & plot_results:
        os.mkdir(figures_out_path)
        print('Made directory for output figures: ' + figures_out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Remove time dimension
    im_dt = np.datetime64(im_date[0:10])
    im_classified = im_classified.isel(time=0)

    # -----Create no data mask
    no_data_mask = xr.where(np.isnan(im_classified), 1, 0).classified.data
    # dilate by 30 m
    iterations = int(30 / ds_dict['resolution_m'])  # number of pixels equal to 30 m
    dilated_mask = binary_dilation(no_data_mask, iterations=iterations)
    no_data_mask = np.logical_not(dilated_mask)
    # add no_data_mask variable classified image
    im_classified = im_classified.assign(no_data_mask=(["y", "x"], no_data_mask))

    # -----Clip DEM to AOI and interpolate to classified image coordinates
    dem_aoi = dem.rio.clip(aoi.geometry, aoi.crs)
    dem_aoi_interp = dem_aoi.interp(x=im_classified.x.data, y=im_classified.y.data, method='linear')
    # add elevation as a band to classified image for convenience
    im_classified['elevation'] = (('y', 'x'), dem_aoi_interp.elevation.data)

    # -----Determine snow covered elevations
    all_elev = np.ravel(dem_aoi_interp.elevation.data)
    all_elev = all_elev[~np.isnan(all_elev)]  # remove NaNs
    snow_est_elev = np.ravel(im_classified.where((im_classified.classified <= 2))
                             .where(im_classified.classified != -9999).elevation.data)
    snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)]  # remove NaNs

    # -----Create elevation histograms
    # determine bins to use in histograms
    elev_min = np.fix(np.nanmin(all_elev) / 10) * 10
    elev_max = np.round(np.nanmax(all_elev) / 10) * 10
    bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max - elev_min) / 10 + 1))
    bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2
    # calculate elevation histograms
    hist_elev = np.histogram(all_elev, bins=bin_edges)[0]
    hist_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
    hist_snow_est_elev_norm = hist_snow_est_elev / hist_elev

    # -----Make all pixels at elevation bins with >75% snow coverage = snow
    # determine elevation with > 75% snow coverage
    if np.any(hist_snow_est_elev_norm > 0.75):
        elev_75_snow = bin_centers[np.argmax(hist_snow_est_elev_norm > 0.75)]
        # make a copy of im_classified for adjusting
        im_classified_adj = im_classified.copy()
        # Fill gaps in elevation using linear interpolation along the spatial dimensions
        im_classified_adj['elevation'] = im_classified['elevation'].interpolate_na(dim='x', method='linear')
        # set all pixels above the elev_75_snow to snow (1)
        im_classified_adj['classified'] = xr.where(im_classified_adj['elevation'] > elev_75_snow, 1,
                                                   im_classified_adj['classified'])
        # create a binary mask for everything above the first instance of 25% snow-covered
        sca_perc_threshold = 0.1
        if np.any(hist_snow_est_elev_norm > sca_perc_threshold):
            elev_25_snow = bin_centers[np.argmax(hist_snow_est_elev_norm > sca_perc_threshold)]
            elevation_threshold_mask = xr.where(im_classified.elevation > elev_25_snow, 1, 0)
        else:
            elevation_threshold_mask = None

    else:
        im_classified_adj = im_classified
        elevation_threshold_mask = None

    # -----Delineate snow lines
    # create binary snow matrix
    im_binary = xr.where(im_classified_adj > 2, 1, 0).classified.data
    # fill holes in binary image (0s within 1s = 1)
    im_binary_no_holes = binary_fill_holes(im_binary)
    # find contours at a constant value of 0.5 (between 0 and 1)
    contours = find_contours(im_binary_no_holes, 0.5)
    # convert contour points to image coordinates
    contours_coords = []
    for contour in contours:
        # convert image pixel coordinates to real coordinates
        fx = interp1d(range(0, len(im_classified_adj.x.data)), im_classified_adj.x.data)
        fy = interp1d(range(0, len(im_classified_adj.y.data)), im_classified_adj.y.data)
        coords = (fx(contour[:, 1]), fy(contour[:, 0]))
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        contours_coords.append(xy)

    # convert list of coordinates to list of LineStrings
    # do not include points in the no data mask or points above the elevation threshold
    contour_lines = []
    for contour_coords in contours_coords:
        # use elevation_threshold_mask to filter points if it exists
        if elevation_threshold_mask is not None:
            points_real = [Point(x, y) for x, y in contour_coords
                           if im_classified.sel(x=x, y=y, method='nearest').no_data_mask.data.item()
                           and (elevation_threshold_mask.sel(x=x, y=y, method='nearest').data.item() == 1)
                           ]
        else:
            points_real = [Point(x, y) for x, y in contour_coords
                           if im_classified.sel(x=x, y=y, method='nearest').no_data_mask.data.item()
                           ]

        if len(points_real) > 2:  # need at least 2 points for a LineString
            contour_line = LineString([[point.x, point.y] for point in points_real])
            contour_lines.append(contour_line)

    # proceed if lines were found after filtering
    if len(contour_lines) > 0:

        # -----Use the longest line as the snowline
        lengths = [line.length for line in contour_lines]
        max_length_index = max(range(len(contour_lines)), key=lambda i: lengths[i])
        snowline = contour_lines[max_length_index]

        # -----Interpolate elevations at snow line coordinates
        # compile all line coordinates into arrays of x- and y-coordinates
        xpts = np.ravel([x for x in snowline.coords.xy[0]])
        ypts = np.ravel([y for y in snowline.coords.xy[1]])
        # interpolate elevation at snow line points
        snowline_elevs = [im_classified.sel(x=x, y=y, method='nearest').elevation.data.item()
                          for x, y in list(zip(xpts, ypts))]

    else:

        snowline = []
        snowline_elevs = np.nan

    # -----If AOI is ~covered in snow, set snowline elevation to the minimum elevation in the AOI
    if np.all(np.isnan(snowline_elevs)) and (np.nanmedian(hist_snow_est_elev_norm) > 0.5):
        snowline_elevs = np.nanmin(np.ravel(im_classified.elevation.data))

    # -----Calculate snow-covered area (SCA) and accumulation area ratio (AAR)
    # pixel resolution
    dx = im_classified.x.data[1] - im_classified.x.data[0]
    # snow-covered area
    sca = len(np.ravel(im_classified.classified.data[im_classified.classified.data <= 2])) * (
            dx ** 2)  # number of snow-covered pixels * pixel resolution [m^2]
    # accumulation area ratio
    total_area = len(np.ravel(im_classified.classified.data[~np.isnan(im_classified.classified.data)])) * (
            dx ** 2)  # number of pixels * pixel resolution [m^2]
    aar = sca / total_area

    # -----Calculate the equilibrium line altitude (ELA) from the AAR
    dem_clip = dem.rio.clip(aoi.geometry, aoi.crs)
    elevations = np.ravel(dem_clip.elevation.data)
    ela_from_aar = np.nanquantile(elevations, 1 - aar)

    # -----Compile results in dataframe
    # calculate median snow line elevation
    median_snowline_elev = np.nanmedian(snowline_elevs)
    # compile results in df
    if type(snowline) == LineString:
        snowlines_coords_x = [list(snowline.coords.xy[0])]
        snowlines_coords_y = [list(snowline.coords.xy[1])]
    else:
        snowlines_coords_x = [[]]
        snowlines_coords_y = [[]]
    snowline_df = pd.DataFrame({'site_name': [site_name],
                                'datetime': [im_date],
                                'snowlines_coords_X': snowlines_coords_x,
                                'snowlines_coords_Y': snowlines_coords_y,
                                'HorizontalCRS': ['EPSG:' + str(im_classified.rio.crs.to_epsg())],
                                'VerticalCRS': ['EGM96 geoid (EPSG:5773)'],
                                'snowline_elevs_m': [snowline_elevs],
                                'snowline_elevs_median_m': [median_snowline_elev],
                                'SCA_m2': [sca],
                                'AAR': [aar],
                                'ELA_from_AAR_m': [ela_from_aar],
                                'dataset': [dataset],
                                'geometry': [snowline]
                                })

    # -----Save snowline df to file
    # reduce memory storage of dataframe
    snowline_df = reduce_memory_usage(snowline_df, verbose=False)
    # save using user-specified file extension
    if 'pkl' in snowline_fn:
        snowline_df.to_pickle(os.path.join(out_path, snowline_fn))
        if verbose:
            print('Snowline saved to file: ' + os.path.join(out_path, snowline_fn))
    elif 'csv' in snowline_fn:
        snowline_df.to_csv(os.path.join(out_path, snowline_fn), index=False)
        if verbose:
            print('Snowline saved to file: ' + os.path.join(out_path, snowline_fn))
    else:
        print('Please specify snowline_fn with extension .pkl or .csv. Exiting...')
        return 'N/A'

    # -----Plot results
    if plot_results:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        # define x and y limits
        xmin, xmax = aoi.geometry[0].buffer(100).bounds[0] / 1e3, aoi.geometry[0].buffer(100).bounds[2] / 1e3
        ymin, ymax = aoi.geometry[0].buffer(100).bounds[1] / 1e3, aoi.geometry[0].buffer(100).bounds[3] / 1e3
        # define colors for plotting
        colors = list(dataset_dict['classified_image']['class_colors'].values())
        cmp = matplotlib.colors.ListedColormap(colors)
        # RGB image
        if im_xr is None:  # query GEE for image thumbnail if im_xr=None
            image_thumbnail, bounds = query_gee_for_image_thumbnail(dataset, im_dt, aoi)
            ax[0].imshow(image_thumbnail,
                         extent=(bounds[0] / 1e3, bounds[2] / 1e3, bounds[1] / 1e3, bounds[3] / 1e3))
        else:
            im_xr = im_xr.isel(time=0)
            ax[0].imshow(np.dstack([im_xr[dataset_dict[dataset]['RGB_bands'][0]].values,
                                    im_xr[dataset_dict[dataset]['RGB_bands'][1]].values,
                                    im_xr[dataset_dict[dataset]['RGB_bands'][2]].values]),
                         extent=(xmin, xmax, ymin, ymax))
        # classified image
        ax[1].imshow(im_classified['classified'].data, cmap=cmp, clim=(1, 5),
                     extent=(np.min(im_classified.x.data) / 1e3, np.max(im_classified.x.data) / 1e3,
                             np.min(im_classified.y.data) / 1e3, np.max(im_classified.y.data) / 1e3))
        # snowline coordinates
        if type(snowline) == LineString:
            ax[0].plot(np.divide(snowline.coords.xy[0], 1e3), np.divide(snowline.coords.xy[1], 1e3),
                       '.', color='#f768a1', markersize=2)
            ax[1].plot(np.divide(snowline.coords.xy[0], 1e3), np.divide(snowline.coords.xy[1], 1e3),
                       '.', color='#f768a1', markersize=2)
        # plot dummy points for legend
        ax[1].scatter(0, 0, color=colors[0], s=50, label='Snow')
        ax[1].scatter(0, 0, color=colors[1], s=50, label='Shadowed snow')
        ax[1].scatter(0, 0, color=colors[2], s=50, label='Ice')
        ax[1].scatter(0, 0, color=colors[3], s=50, label='Rock')
        ax[1].scatter(0, 0, color=colors[4], s=50, label='Water')
        if type(snowline) == LineString:
            ax[0].scatter(0, 0, color='#f768a1', s=25, label='Snowline estimate')
            ax[1].scatter(0, 0, color='#f768a1', s=25, label='Snowline estimate')
        ax[0].set_ylabel('Northing [km]')
        ax[0].set_xlabel('Easting [km]')
        ax[1].set_xlabel('Easting [km]')
        # AOI
        label = 'AOI'
        if type(aoi.geometry[0].boundary) == MultiLineString:
            for ii, geom in enumerate(aoi.geometry[0].boundary.geoms):
                if ii > 0:
                    label = '_nolegend_'
                ax[0].plot(np.divide(geom.coords.xy[0], 1e3),
                           np.divide(geom.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
                ax[1].plot(np.divide(geom.coords.xy[0], 1e3),
                           np.divide(geom.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
        elif type(aoi.geometry[0].boundary) == LineString:
            ax[0].plot(np.divide(aoi.geometry[0].boundary.coords.xy[0], 1e3),
                       np.divide(aoi.geometry[0].boundary.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
            ax[1].plot(np.divide(aoi.geometry[0].boundary.coords.xy[0], 1e3),
                       np.divide(aoi.geometry[0].boundary.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
        # reset x and y limits
        ax[0].set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        # normalized snow elevations histogram
        ax[2].bar(bin_centers, hist_snow_est_elev_norm, width=(bin_centers[1] - bin_centers[0]), color=colors[0],
                  align='center')
        ax[2].plot([median_snowline_elev, median_snowline_elev], [0, 1], '-', color='#f768a1',
                   linewidth=3, label='Median snowline elevation')
        ax[2].set_xlabel("Elevation [m]")
        ax[2].set_ylabel("Fraction snow-covered")
        ax[2].grid()
        ax[2].set_xlim(elev_min - 10, elev_max + 10)
        ax[2].set_ylim(0, 1)
        # determine figure title and file name
        title = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snow-cover'
        # add legends
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        ax[2].legend(loc='lower right')
        fig.suptitle(title)
        fig.tight_layout()
        # save figure
        fig_fn = os.path.join(figures_out_path, title + '.png')
        fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
        if verbose:
            print('Figure saved to file:' + fig_fn)

    return snowline_df


# --------------------------------------------------
def apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path, snowlines_path,
                                  aoi_utm, dem, epsg_utm, clf, feature_cols, crop_to_aoi, figures_out_path,
                                  plot_results, verbose):
    """
    Apply the classification and snow delineation pipeline to an image. Batch apply using Dask.

    Parameters
    ----------
    im_xr: xarray.Dataset
        input image
    dataset_dict: dict
        dictionary of dataset-specific parameters
    dataset: str
        name of dataset ('Landsat', 'Sentinel2', 'PlanetScope')
    site_name: str
        name of site, used for output file names
    im_classified_path: str
        path in directory where classified netCDF images will be saved
    snowlines_path: str
        path in directory where snowline CSV files will be saved
    aoi_utm: geopandas.GeoDataFrame
        area of interest with CRS in local UTM zone
    dem: xarray.Dataset
        digital elevation model
    epsg_utm: str
        EPSG code for local UTM zone
    clf: sklearn.Classifier
        image classsifier
    feature_cols: list of str
        list of bands to use to classify image
    crop_to_aoi: bool
        whether to crop images to AOI before classifying image
    figures_out_path: str
        path in directory where figures will be saved
    plot_results: bool
        whether to plot results and save figures
    verbose: bool
        whether to output details during processing steps

    Returns
    -------
    snowline_df: pandas.DataFrame
    """
    # Grab image date string from time variable
    im_date = str(im_xr.time.data[0])[0:19]
    # Adjust image for image scalar and no data values
    crs = im_xr.rio.crs.to_epsg()
    band2 = list(dataset_dict[dataset]['refl_bands'].keys())[1]
    if np.nanmean(im_xr[band2]) > 1e3:
        im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan,
                         im_xr / dataset_dict[dataset]['image_scalar'])
    else:
        im_xr = xr.where(im_xr == dataset_dict[dataset]['no_data_value'], np.nan, im_xr)
    # Add NDSI band
    im_xr['NDSI'] = ((im_xr[dataset_dict[dataset]['NDSI_bands'][0]] - im_xr[dataset_dict[dataset]['NDSI_bands'][1]])
                     / (im_xr[dataset_dict[dataset]['NDSI_bands'][0]] + im_xr[dataset_dict[dataset]['NDSI_bands'][1]]))
    im_xr.rio.write_crs('EPSG:' + str(crs), inplace=True)

    # Check if classified image already exists in file
    im_classified_fn = im_date.replace('-', '').replace(':',
                                                        '') + '_' + site_name + '_' + dataset + '_classified.nc'
    if os.path.exists(os.path.join(im_classified_path, im_classified_fn)):
        # load classified image from file
        im_classified = xr.open_dataset(os.path.join(im_classified_path, im_classified_fn))
        # remove no data values
        im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
        im_classified = im_classified.rio.write_crs('EPSG:4326').rio.reproject('EPSG:' + epsg_utm)
    else:
        # classify image
        im_classified = classify_image(im_xr, clf, feature_cols, crop_to_aoi, aoi_utm,
                                       dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
        if type(im_classified) == str:  # skip if error in classification
            return

    # Check if snowline already exists in file
    snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snowline.csv'
    if os.path.exists(os.path.join(snowlines_path, snowline_fn)):
        # No need to load snowline if it already exists
        return
    else:
        # Delineate snowline
        snowline_df = delineate_snowline(im_classified, site_name, aoi_utm, dem, dataset_dict, dataset, im_date,
                                         snowline_fn, snowlines_path, figures_out_path, plot_results, im_xr,
                                         verbose)
        plt.close()

    return snowline_df


# --------------------------------------------------
def reduce_memory_usage(df, verbose=True):
    """
    Reduce memory usage in pandas.DataFrame
    From Bex T (2021): https://towardsdatascience.com/6-pandas-mistakes-that-silently-tell-you-are-a-rookie-b566a252e60d

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
    """
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
                if (  # elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "pandas.DataFrame memory usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
