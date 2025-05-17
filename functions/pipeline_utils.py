"""
Functions for image querying in Google Earth Engine, image adjustment, and snow detection in Landsat, Sentinel-2, and PlanetScope imagery
Rainey Aberle
"""

import math
import geopandas as gpd
import pandas as pd
import ee
import geedim as gd
import requests
from shapely.geometry import MultiLineString, LineString, Point
import os
import xarray as xr
import numpy as np
import rioxarray as rxr
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from tqdm.auto import tqdm
import datetime
from datetime import UTC
from sklearn.exceptions import NotFittedError
import PIL
import io
import wxee as wx
import rasterio as rio


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


def adjust_dem_data_vars(dem):
    """

    Parameters
    ----------
    dem: xarray.Dataset
        digital elevation model (DEM)

    Returns
    -------
    dem: xarray.Dataset
        digital elevation model (DEM) with one band: "elevation"
    """
    if 'band_data' in dem.data_vars:
        dem = dem.rename({'band_data': 'elevation'})
    if 'band' in dem.dims:
        elev_data = dem.elevation.data[0]
        dem = dem.drop_dims('band')
        dem['elevation'] = (('y', 'x'), elev_data)
    return dem


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
        dataset of elevations over the AOI
    """

    # -----Grab optimal UTM zone from AOI CRS
    epsg_utm = str(aoi_utm.crs.to_epsg())

    # -----Define function to transform ellipsoid to geoid heights
    def ellipsoid_to_geoid_heights(ds, base_path, out_path, out_fn):
        print('Transforming elevations from the ellipsoid to the geoid...')

        # Load EGM96 model from file
        geoid_model_fn = os.path.join(base_path, 'inputs-outputs', 'us_nga_egm96_15.tif')
        geoid_model = xr.open_dataset(geoid_model_fn)

        # Resample geoid model to DEM coordinates
        geoid_model_resampled = geoid_model.interp(x=ds.x, y=ds.y, method='linear')
        geoid_height = geoid_model_resampled.band_data.data[0]

        # Subtract geoid heights from ds heights and update the dataset
        ds['elevation'] -= geoid_height

        # Re-save to file with updated elevations
        ds.rio.to_raster(os.path.join(out_path, out_fn), dtype='float32', zlib=True, compress='deflate')
        print('DEM re-saved with elevations referenced to the EGM96 geoid.')

        return ds

    # -----Define output image names, check if already exists in directory
    arcticdem_geoid_fn = site_name + '_ArcticDEM_clip_geoid.tif'
    arcticdem_fn = site_name + '_ArcticDEM_clip.tif'
    nasadem_fn = site_name + '_NASADEM_clip.tif'
    if os.path.exists(os.path.join(out_path, arcticdem_geoid_fn)):
        print('Clipped ArcticDEM referenced to the geoid already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, arcticdem_geoid_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)
    elif os.path.exists(os.path.join(out_path, arcticdem_fn)):
        print('Clipped ArcticDEM already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, arcticdem_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)
        # transform elevations from ellipsoid to geoid, save to file
        dem_ds = ellipsoid_to_geoid_heights(dem_ds, base_path, out_path, arcticdem_geoid_fn)
    elif os.path.exists(os.path.join(out_path, nasadem_fn)):
        print('Clipped NASADEM already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, nasadem_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)
    else:  # if no DEM exists in directory, load from GEE

        # -----Reformat AOI for clipping DEM
        aoi_wgs = aoi_utm.to_crs("EPSG:4326")
        region = ee.Geometry.Polygon(list(zip(aoi_wgs.geometry[0].exterior.coords.xy[0], 
                                              aoi_wgs.geometry[0].exterior.coords.xy[1])))
        
        # -----Check for ArcticDEM coverage over AOI
        # load ArcticDEM_Mosaic_coverage.shp
        arcticdem_coverage_fn = os.path.join(base_path, 'inputs-outputs', 'ArcticDEM_Mosaic_coverage.shp')
        arcticdem_coverage = gpd.read_file(arcticdem_coverage_fn)
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
        
        # -----Clip to exact region (otherwise, it's a bounding box region)
        dem.ee_image = dem.ee_image.clip(region)

        # -----Download DEM and open as xarray.Dataset
        print('Downloading DEM to ', out_path)
        # create out_path if it doesn't exist
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # download DEM
        dem.download(os.path.join(out_path, dem_fn), region=region, scale=res, crs="EPSG:4326")
        # read DEM as xarray.Dataset
        dem_ds = xr.open_dataset(os.path.join(out_path, dem_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)

        # -----If using ArcticDEM, transform elevations with respect to the geoid (rather than the ellipsoid)
        if 'ArcticDEM' in elevation_source:
            dem_ds = ellipsoid_to_geoid_heights(dem_ds, base_path, out_path, arcticdem_geoid_fn)

    # -----Reproject DEM to UTM
    dem_ds = dem_ds.rio.reproject('EPSG:' + epsg_utm)
    dem_ds = xr.where((dem_ds > 1e38) | (dem_ds <= -9999), np.nan, dem_ds)
    dem_ds = dem_ds.rio.write_crs('EPSG:' + epsg_utm)

    return dem_ds


def split_date_range(date_start, date_end, month_start, month_end):
    """
    Split date range into a list of annual date ranges to mitigate GEE user memory limit issues. 

    Parameters
    ----------
    date_start: str
        start date of search range ("YYYY-MM-DD")
    date_end: str
        end date of search range ("YYYY-MM-DD")
    month_start: int | float
        start month of search range, inclusive
    month_end: int | float
        end month of search range, inclusive
    
    Returns
    ----------
    date_ranges: list of str
        list of individual date ranges for image querying
    """
    date_start_dt = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_end_dt = datetime.datetime.strptime(date_end, '%Y-%m-%d')

    # get range of years
    year_start = date_start_dt.year
    year_end = date_end_dt.year
    years = np.arange(year_start, year_end+1)

    # construct date ranges
    date_ranges = []
    for i, year in enumerate(years):
        start_dt = datetime.datetime.strptime(f"{year}-{month_start}-01", '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(f"{year}-{month_end+1}-01", '%Y-%m-%d') - datetime.timedelta(days=1)
        # check that start and end dates are included, and that it does not extent beyond the specified date range
        if (i==0) & (start_dt < date_start_dt):
            start_dt = date_start_dt
        if (i==len(years)-1) & (end_dt < date_end_dt):
            end_dt = date_end_dt
        if (i==len(years)-1) & (end_dt > date_end_dt):
            end_dt = date_end_dt
        # convert back to strings
        start_str = str(start_dt)[0:10]
        end_str = str(end_dt)[0:10]
        # store in date ranges list
        date_ranges.append([start_str, end_str])

    return date_ranges


def get_image_ids(dataset, region, date_start, date_end, cloud_cover_max):
    """
    Query GEE based on search criteria and return list of image IDs. 

    Parameters
    ----------
    dataset: str
        which dataset to query ("Landsat", "Sentinel-2_TOA", or "Sentinel-2_SR")
    region: ee.Geometry.Polygon
        region for filtering image bounds
    date_start: str
        start date of search range ("YYYY-MM-DD")
    date_end: str
        end date of search range ("YYYY-MM-DD")
    cloud_cover_max: int | float
        maximum cloud cover for filtering images based on respective image cloud cover property
    
    Returns
    ----------
    im_ids: list of str
        list of image IDs satisfying search critera
    im_dts: list of datetime.datetime objects
        list of datetimes associated with each image
    """
    # Query GEE for the respective dataset
    if dataset=='Landsat':
        im_col_ee_8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                       .filterBounds(region)
                       .filterDate(date_start, date_end)
                       .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                       )
        im_col_ee_9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') # <---- NEW
                       .filterBounds(region)
                       .filterDate(date_start, date_end)
                       .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                       )
        im_col_ee = im_col_ee_8.merge(im_col_ee_9)
    elif dataset=='Sentinel-2_TOA':
        im_col_ee = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                     .filterBounds(region)
                     .filterDate(date_start, date_end)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                     )
    elif dataset=='Sentinel-2_SR':
        im_col_ee = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region)
                     .filterDate(date_start, date_end)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
                    )

    # Create arrays of image IDs and datetimes
    im_ids = np.array(im_col_ee.aggregate_array('system:id').getInfo())
    im_dts = np.array(im_col_ee.aggregate_array('system:time_start').getInfo())
    
    # Convert datetimes from UTC to calendar dates and times
    im_dts = np.array([datetime.datetime.fromtimestamp(dt/1000, UTC) for dt in im_dts])
    
    # Sort lists by datetime
    isort = np.argsort(im_dts)
    im_ids = im_ids[isort]
    im_dts = im_dts[isort]
    
    return im_ids, im_dts


def filter_invalid_image_ids(im_ids_list, im_dts_list):
    """
    For Sentinel-2, some images have invalid end points, i.e. results in an error when trying to query. 
    This function filters the invalid images from the input list of IDs. 

    Parameters
    ---------
    im_ids_list: list of str
        list of image IDs
    im_dts_list: list of datetime.datetime objects
        list of datetimes associated with each image
    
    Returns
    ----------
    valid_im_ids_list: list of str
        list of valid image IDs
    valid_im_dts_list: list of datetime.datetime objects
        list of datetimes associated with each valid image
    error_im_ids_list: list of str
        list of error image IDs
    error_im_dts_list: list of datetime.datetime objects
        list of datetimes associated with each error image
    """
    # initialize lists of valid and error images
    valid_im_ids_list, valid_im_dts_list = [], []
    error_im_ids_list, error_im_dts_list = [], []
    # iterate over images
    for im_id, im_dt in zip(im_ids_list, im_dts_list):
        # get masked image
        im_gd = gd.MaskedImage.from_id(im_id)
        try:
            _ = im_gd.properties # Error occurs here if image is not accesible!
            valid_im_ids_list.append(im_id)
            valid_im_dts_list.append(im_dt)
        except Exception as e:
            print('Error querying GEE for', im_id)
            error_im_ids_list.append(im_id)
            error_im_dts_list.append(im_dt)
            continue
    return valid_im_ids_list, valid_im_dts_list, error_im_ids_list, error_im_dts_list


def image_mosaic_ids(im_ids_list, im_dts_list):
    """
    Identify images captured on the same day, concatenate into a list of image combinations for later mosaicking.

    Parameters
    ----------
    im_ids_list: list of str
        list of image IDs
    im_dts_list: list of datetime.datetime objects
        list of datetimes associated with each image
    
    Returns
    ----------
    im_mosaic_ids: list of list of str
        list of image combinations for mosaicking
    im_mosaic_dts: list of list of datetime.datetime objects
        list of datetimes associated with image combination for mosaicking
    """
    # Identify unique days
    im_days_str = [f"{x.year}-{x.month}-{x.day}" for x in im_dts_list]
    unique_days = list(set(im_days_str))
    # initialize list of image IDs
    im_mosaic_ids = []
    im_mosaic_dts = []
    # iterate over days
    for day in unique_days:
        iday = np.argwhere([x==day for x in im_days_str]).ravel()
        im_ids_day = [im_ids_list[i] for i in iday]
        im_dts_day = [im_dts_list[i] for i in iday]
        im_mosaic_ids.append(im_ids_day)
        im_mosaic_dts.append(im_dts_day)
    return im_mosaic_ids, im_mosaic_dts   


def calculate_aoi_coverage(im_xr, aoi_gdf):
    """
    Calculate the percentage of the AOI covered by the image

    Parameters
    ----------
    im_xr: xarray.Dataset
        input image
    aoi_gdf: gpd.GeoDataFrame
        Area of Interest (AOI) used to clip the image and calculate coverage

    Returns
    -------
    percentage_covered: float
        Percentage of the AOI covered by the image
    """
    mask = rio.features.geometry_mask(aoi_gdf.geometry,
                                      transform=im_xr.rio.transform(),
                                      out_shape=(len(im_xr.y.data), len(im_xr.x.data)),
                                      all_touched=True,
                                      invert=True)
    # Clip the image with the AOI mask
    masked_data = im_xr.where(mask)
    # Count the non-NaN values within the clipped dataset
    count_non_nan = np.sum(~np.isnan(masked_data[list(im_xr.data_vars)[0]].data))
    # Calculate the total number of pixels in the AOI
    total_pixels = np.sum(xr.where(mask == 1, 1, 0).data)
    # Calculate the percentage of coverage
    percent_coverage = (count_non_nan / total_pixels) * 100

    return percent_coverage


def classify_image(im_xr, clf, feature_cols, aoi, dataset_dict, dataset, im_classified_fn, out_path, verbose=False):
    """
    Function to classify image collection using a pre-trained classifier

    Parameters
    ----------
    im_xr: xarray.Dataset
        Dataset containing image bands in data variables with x, y, and time coordinates.
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier
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
        whether to output verbage for each image (default=False)

    Returns
    ----------
    im_classified_xr: xarray.Dataset
        classified image
    """

    # -----Make output directory if it doesn't already exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print('Made directory for classified images: ' + out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Define image bands and capture date
    bands = [band for band in ds_dict['refl_bands'] if 'QA' not in band]
    im_date = np.datetime64(str(im_xr.time.data[0])[0:19], 'ns')

    # -----Crop image to the AOI and remove time dimension
    # Create dummy band for AOI masking comparison
    im_xr['aoi_mask'] = (['y', 'x'], np.ones(np.shape(im_xr[dataset_dict[dataset]['RGB_bands'][0]].data[0])))
    im_aoi = im_xr.rio.clip(aoi.geometry, im_xr.rio.crs).isel(time=0)

    # -----Check that the image has 70% real values in the AOI
    perc_real_values_aoi = (len(np.ravel(~np.isnan(im_aoi[dataset_dict[dataset]['RGB_bands'][0]].data)))
                            / len(np.ravel(~np.isnan(im_aoi['aoi_mask'].data))))
    if perc_real_values_aoi < 0.7:
        if verbose:
            print('Less than 70% coverage of the AOI, skipping image...')
        return 'N/A'

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
                                                      'Classes': '1 = Snow, 2 = Shadowed snow, 3 = Ice, 4 = Rock, 5 = Water',
                                                      '_FillValue': '-9999'
                                                      })
    # replace NaNs with -9999, convert data types to int
    im_classified_xr_int = im_classified_xr.copy(deep=True).astype(np.int8)
    im_classified_xr_int = xr.where(np.isnan(im_classified_xr.classified), -9999, im_classified_xr_int)
    # reproject to WGS84 horizontal coordinates for consistency before saving
    im_classified_xr_int = im_classified_xr_int.rio.write_crs(im_classified_xr.rio.crs)
    im_classified_xr_int = im_classified_xr_int.rio.reproject('EPSG:4326', nodata=-9999)

    # -----Save to file
    if '.nc' in im_classified_fn:
        im_classified_xr_int.to_netcdf(os.path.join(out_path, im_classified_fn))
    elif '.tif' in im_classified_fn:
        im_classified_xr_int.rio.to_raster(os.path.join(out_path, im_classified_fn))
    if verbose:
        print('Classified image saved to file: ', os.path.join(out_path, im_classified_fn))

    return im_classified_xr


def delineate_snowline(im_classified, site_name, aoi, dem, dataset_dict, dataset, im_date, snowline_fn,
                       out_path, figures_out_path, plot_results, im_xr=None, verbose=False):
    """
    Delineate the seasonal snowline in classified images. Snowlines will likely not be detected in images with nearly all or no snow.

    Parameters
    ----------
    im_classified: xarray.Dataset
        classified image
    site_name: str
        name of study site
    aoi:  geopandas.geodataframe.GeoDataFrame
        area of interest
        must be in the same coordinate reference system as the classified image
    dem: xarray.Dataset
        digital elevation model over the aoi
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
        input reflectance image
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
        print("Made directory for snowlines:", out_path)

    # -----Make directory for figures (if it does not already exist)
    if (not os.path.exists(figures_out_path)) & plot_results:
        os.mkdir(figures_out_path)
        print('Made directory for output figures: ', figures_out_path)

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
    dem_aoi = xr.where(np.abs(dem_aoi) < 3e38, dem_aoi, np.nan)
    dem_aoi_interp = dem_aoi.interp(x=im_classified.x.data, y=im_classified.y.data, method='linear')
    dem_aoi_interp = xr.where(np.abs(dem_aoi_interp) < 3e38, dem_aoi_interp, np.nan)
    # add elevation as a band to classified image for convenience
    im_classified['elevation'] = (('y', 'x'), dem_aoi_interp.elevation.data)

    # -----Determine snow covered elevations
    all_elev = np.ravel(dem_aoi_interp.elevation.data)
    all_elev = all_elev[~np.isnan(all_elev)]  # remove NaNs
    snow_est_elev = np.ravel(im_classified.where((im_classified.classified == 1) | (im_classified.classified==2))
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
        # create a binary mask for everything above the first instance of 10% snow-covered
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
    dem_clip = xr.where(dem_clip < 3e38, dem_clip, np.nan)
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
            print('Snow cover statistics saved to file: ' + os.path.join(out_path, snowline_fn))
    elif 'csv' in snowline_fn:
        snowline_df.to_csv(os.path.join(out_path, snowline_fn), index=False)
        if verbose:
            print('Snow cover staistics saved to file: ' + os.path.join(out_path, snowline_fn))
    else:
        print('Please specify snowline_fn with extension .pkl or .csv. Exiting...')
        return 'N/A'

    # -----Plot results
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
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
                         extent=(np.min(im_xr.x.data) / 1e3, np.max(im_xr.x.data) / 1e3,
                                 np.min(im_xr.y.data) / 1e3, np.max(im_xr.y.data) / 1e3))
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
        # determine figure title and file name
        title = f"{im_date.replace('-','').replace(':','')}_{site_name}_{dataset}_snow_cover"
        # add legends
        ax[0].legend(loc='lower right')
        ax[1].legend(loc='lower right')
        fig.suptitle(title)
        fig.tight_layout()
        # save figure
        fig_fn = os.path.join(figures_out_path, title + '.png')
        fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
        if verbose:
            print('Figure saved to file:', fig_fn)

    return snowline_df


def apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path, snowlines_path,
                                  aoi_utm, dem, epsg_utm, clf, feature_cols, figures_out_path,
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
        name of site
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
    figures_out_path: str
        path in directory where figures will be saved
    plot_results: bool
        whether to plot results and save figures
    verbose: bool
        whether to output details during processing steps

    Returns
    -------
    snowline_df: pandas.DataFrame
        resulting data table containing snow cover statistics and snowline geometry
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
        im_classified = classify_image(im_xr, clf, feature_cols, aoi_utm,
                                       dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
        if type(im_classified) == str:  # skip if error in classification
            return

    # Check if snowline already exists in file
    snowline_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snow_cover_stats.csv'
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


def query_gee_for_imagery(dataset_dict, dataset, aoi_utm, dem, date_start, date_end, month_start, month_end, 
                          mask_clouds=True, cloud_cover_max=70, aoi_coverage=70, run_pipeline=True, site_name='SITE', 
                          clf=None, feature_cols=None, im_download=False, im_out_path=None, im_classified_path=None, 
                          snowlines_path=None, figures_out_path=None, plot_results=True, verbose=True):
    """
    Query Google Earth Engine for Landsat 8 and 9 surface reflectance (SR), Sentinel-2 top of atmosphere (TOA) or SR imagery.
    Images captured on the same day will be mosaicked. For each image, run the classification and snowline detection workflow.

    Parameters
    __________
    dataset_dict: dict
        dictionary of parameters for each image product
    dataset: str
        name of dataset ('Landsat', 'Sentinel-2_SR', 'Sentinel-2_TOA', 'PlanetScope')
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest with CRS in local UTM zone
    dem: xarray.Dataset
        digital elevation model over the aoi
        must be in the same coordinate reference system as the classified image
    date_start: str
        start date for image search ('YYYY-MM-DD')
    date_end: str
        end date for image search ('YYYY-MM-DD')
    month_start: int
        starting month for calendar range filtering
    month_end: int
        ending month for calendar range filtering
    mask_clouds: bool
        whether to mask clouds using geedim masking tools
    cloud_cover_max: int or float
        maximum image cloud cover percentage (0-100)
    run_pipeline: bool
        whether to run the snow detection pipeline
    im_download: bool
        whether to download multispectral images locally by default. Folder for downloads (out_folder) must be specified. 
        If images over the AOI exceed the GEE user limit, they must be downloaded.  
    aoi_coverage: int or float
        minimum percent coverage of the AOI after filtering clouds (0-100)
    site_name: str
        name of study site
    clf: sklearn pre-trained model
        classifier applied to the input image, specific to image dataset
    feature_cols: list of str
        features (i.e., image bands and NDSI) to use for classifying
    im_out_path: str
        path where images will be saved if im_download = True
    im_classified_path: str
        path where classified image netCDF's will be saved
    snowlines_path: str
        path where snowline CSV's will be saved
    figures_out_path: str
        path where figures will be saved
    crop_to_aoi: bool
        whether to crop images to the aoi_utm geometry before classifying
    plot_results: bool
        whether to plot results and save figure
    verbose: bool
        whether to output details about each image during the snow detection pipeline

    Returns
    __________
    None
    """

    # -----Grab optimal UTM zone from AOI CRS
    epsg_utm = str(aoi_utm.crs.to_epsg())

    # -----Reformat AOI for image filtering
    # reproject CRS from AOI to WGS
    aoi_wgs = aoi_utm.to_crs('EPSG:4326')
    # prepare AOI for querying geedim (AOI bounding box)
    region = ee.Geometry.Polygon(list(zip(aoi_wgs.geometry[0].exterior.coords.xy[0],
                                          aoi_wgs.geometry[0].exterior.coords.xy[1])))     

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
        print(f'No image downloads necessary, {dataset} images over the AOI are within the GEE limit.')
    if (im_out_path is None) & im_download:
        print('Variable out_path must be specified to download images. Exiting...')
        return 'N/A'
    if (not im_download) & (not run_pipeline):
        print('NOTE: Images will not be downloaded (im_download=False) and the snow detection pipeline will not be run (run_pipeline=False).'
              'Function will proceed querying and filtering imagery, but no outputs will be saved.')

    # -----Apply image processing and filtering functions
    # Split date range into yearly date ranges to mitigate GEE user memory limit issues
    print('Splitting full date range into annual date ranges')
    date_ranges = split_date_range(date_start, date_end, month_start, month_end)

    # Iterate over date ranges
    print('Iterating over date ranges')
    for date_range in date_ranges:
        print(f"\n{date_range}")

        # Get image IDs
        print('Querying GEE for all images meeting search criteria...')
        im_ids, im_dts = get_image_ids(dataset, region, date_range[0], date_range[1], cloud_cover_max)
        print(f"{len(im_ids)} total images found")

        # For Sentinel-2 SR, filter error images from IDs list
        # NOTE: the Sentinel-2 SR collection has some empty or invalid images that lead to errors when trying to access.
        # This is a workaround to simply remove them from the collection. 
        if dataset=='Sentinel-2_SR':
            print('Filtering image IDs with invalid end points...')
            im_ids, im_dts, _, _ = filter_invalid_image_ids(im_ids, im_dts)
            print(f"{len(im_ids)} valid images remain")

        # Identify images captured the same hour for mosaicking
        im_mosaic_ids, im_mosaic_dts = image_mosaic_ids(im_ids, im_dts)
        print(f"{len(im_mosaic_ids)} total images will remain after mosaicking")
    
        # -----Create xarray.Datasets from list of image IDs
        # loop through image IDs
        for i in tqdm(range(0, len(im_mosaic_ids))):

            # subset image IDs and image datetimes
            im_ids, im_dts = im_mosaic_ids[i], im_mosaic_dts[i]

            # if images must be downloaded, use geedim
            if im_download:

                # make directory for outputs (out_path) if it doesn't exist
                if not os.path.exists(im_out_path):
                    os.mkdir(im_out_path)
                    print('Made directory for image downloads: ' + im_out_path)
                # define filename
                if len(im_dts) > 1:
                    im_fn = dataset + '_' + str(im_dts[0]).replace('-', '')[0:8] + '_MOSAIC.tif'
                else:
                    im_fn = dataset + '_' + str(im_dts[0]).replace('-', '')[0:8] + '.tif'
                # check file does not already exist in directory, download
                if not os.path.exists(os.path.join(im_out_path, im_fn)):
                    # create list of MaskedImages from IDs
                    im_gd_list = [gd.MaskedImage.from_id(im_id) for im_id in im_ids]
                    # combine into new MaskedCollection
                    im_collection = gd.MaskedCollection.from_list(im_gd_list)
                    # create image composite
                    im_composite = im_collection.composite(method=gd.CompositeMethod.q_mosaic,
                                                        mask=mask_clouds,
                                                        region=region)
                    # clip to exact region (otherwise, it's a bounding box region)
                    im_composite.ee_image = im_composite.ee_image.clip(region)
                    # download to file
                    im_composite.download(os.path.join(im_out_path, im_fn),
                                        region=region,
                                        scale=res,
                                        crs='EPSG:' + epsg_utm,
                                        dtype='int16',
                                        bands=im_composite.refl_bands)
                # load image from file
                im_da = rxr.open_rasterio(os.path.join(im_out_path, im_fn))
                # convert to xarray.DataSet
                im_xr = im_da.to_dataset('band')
                band_names = list(dataset_dict[dataset]['refl_bands'].keys())
                im_xr = im_xr.rename({i + 1: name for i, name in enumerate(band_names)})
                # account for image scalar and no data values
                im_xr = xr.where(im_xr != dataset_dict[dataset]['no_data_value'],
                                im_xr / dataset_dict[dataset]['image_scalar'], np.nan)
                im_xr = xr.where(im_xr > 0, im_xr, np.nan)
                # add time dimension
                im_dt = np.datetime64(datetime.datetime.fromtimestamp(im_da.attrs['system-time_start'] / 1000))
                im_xr = im_xr.expand_dims({'time': [im_dt]})
                # set CRS
                im_xr.rio.write_crs('EPSG:' + str(im_da.rio.crs.to_epsg()), inplace=True)

                # check that image covered >= aoi_coverage % of the AOI
                percentage_covered = calculate_aoi_coverage(im_xr, aoi_utm)
                if (percentage_covered >= aoi_coverage) & (run_pipeline):
                    # -----Run classification pipeline
                    apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                snowlines_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                                figures_out_path, plot_results, verbose)
                else:
                    print(f'Image covers < {aoi_coverage}% of the AOI, skipping...')
                    continue

            # if no image downloads necessary, use wxee
            elif run_pipeline:

                # if more than one ID, composite images
                if len(im_dts) > 1:
                    # create list of MaskedImages from IDs
                    ims_gd = [gd.MaskedImage.from_id(im_id, mask=mask_clouds, region=region) for im_id in im_ids]
                    # convert to list of ee.Images
                    ims_ee = [ee.Image(im_gd.ee_image).select(im_gd.refl_bands).clip(region) for im_gd in ims_gd]
                    # convert to xarray.Datasets
                    ims_xr = [im_ee.wx.to_xarray(scale=res, region=region, crs='EPSG:' + epsg_utm) for im_ee in ims_ee]
                    # composite images
                    ims_xr_composite = xr.merge(ims_xr, compat='override')
                    # account for image scalar
                    ims_xr_composite = xr.where(ims_xr_composite != dataset_dict[dataset]['no_data_value'],
                                                ims_xr_composite / dataset_dict[dataset]['image_scalar'], np.nan)
                    ims_xr_composite = xr.where(ims_xr_composite > 0, ims_xr_composite, np.nan)
                    # set CRS
                    ims_xr_composite.rio.write_crs('EPSG:' + epsg_utm, inplace=True)
                    im_xr = ims_xr_composite
                else:
                    # create MaskedImage from ID
                    im_gd = gd.MaskedImage.from_id(im_ids[0], mask=mask_clouds, region=region)
                    # convert to ee.Image
                    im_ee = ee.Image(im_gd.ee_image).select(im_gd.refl_bands).clip(region)
                    # convert to xarray.Datasets
                    im_xr = im_ee.wx.to_xarray(scale=res, region=region, crs='EPSG:' + epsg_utm)
                    # account for image scalar
                    im_xr = xr.where(im_xr != dataset_dict[dataset]['no_data_value'],
                                    im_xr / dataset_dict[dataset]['image_scalar'], np.nan)
                    im_xr = xr.where(im_xr > 0, im_xr, np.nan)
                    # set CRS
                    im_xr.rio.write_crs('EPSG:' + epsg_utm, inplace=True)

                # -----Check that image covers at least 70% of the AOI
                percentage_covered = calculate_aoi_coverage(im_xr, aoi_utm)
                if percentage_covered >= aoi_coverage:

                    # -----Run classification pipeline
                    apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                                snowlines_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                                figures_out_path, plot_results, verbose)
                else:
                    print(f'Image covers < {aoi_coverage}% of the AOI, skipping...')
                    continue

    return


def query_gee_for_image_thumbnail(dataset, dt, aoi_utm):
    """

    Parameters
    ----------
    dataset: str
        which dataset / image collection to query
    dt: numpy.datetime64
        image capture datetime
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest used for filtering the image collection, reprojected to the optimal UTM zone

    Returns
    -------
    image: PIL.image object
        resulting image thumbnail
    bounds: numpy.array
        bounds of the image, derived from the aoi (minx, miny, maxx, maxy)
    """

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


def plot_xr_rgb_image(im_xr, rgb_bands):
    """Plot RGB image of xarray.DataSet

    Parameters
    ----------
    im_xr: xarray.DataSet
        Dataset containing image bands in data variables with x, y, and time coordinates.
        Function assumed x and y coordinates are in units of meters.
    rgb_bands: List
        List of data variable names for RGB bands contained within the dataset, e.g. ['red', 'green', 'blue']

    Returns
    ----------
    fig: matplotlib.pyplot.figure
        figure handle for the resulting plot
    ax: matplotlib.pyplot.figure.Axes
        axis handle for the resulting plot
    """

    # -----Grab RGB bands from dataset
    if len(np.shape(im_xr[rgb_bands[0]].data)) > 2:  # check if a dimension must be cut from the band data
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
