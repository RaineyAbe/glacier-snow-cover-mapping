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


def query_gee_for_dem(aoi_utm, site_name, out_path=None):
    """
    Query GEE for the ArcticDEM Mosaic (where there is coverage) or the NASADEM,
    clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM, reprojected to the optimal UTM zone
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
    aoi = aoi_utm.to_crs("EPSG:4326")

    # -----Define function to transform ellipsoid to geoid heights
    def ellipsoid_to_geoid_heights(ds, output_path, out_fn):
        print('Transforming elevations from the ellipsoid to the geoid...')

        # Load EGM96 model from GEE Assets
        geoid_model = xr.open_dataset(os.path.join(os.getcwd(), '..', 'inputs-outputs', 'us_nga_egm96_15.tif'))

        # Resample geoid model to DEM coordinates
        geoid_model_resampled = geoid_model.interp(x=ds.x, y=ds.y, method='linear')
        geoid_height = geoid_model_resampled.band_data.data[0]

        # Subtract geoid heights from ds heights and update the dataset
        ds['elevation'] -= geoid_height

        # Re-save to file with updated elevations
        ds.rio.to_raster(os.path.join(output_path, out_fn), dtype='float32', zlib=True, compress='deflate')
        print('DEM re-saved with elevations referenced to the EGM96 geoid.')

        return ds

    # -----Define function for calculating % coverage of the AOI
    def calculate_percent_image_aoi_coverage(ee_image):
        # Create binary image of masked (0) and unmasked (1) pixels
        unmasked_pixels = ee_image.mask().reduce(ee.Reducer.allNonZero()).selfMask()
        # Calculate the area of unmasked pixels in the ROI
        pixel_area = ee.Image.pixelArea()
        aoi_area = aoi_ee.area()
        scale = ee_image.projection().nominalScale()
        unmasked_area = unmasked_pixels.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=scale,
            maxPixels=1e13
        ).get('all')
        # Calculate the percentage of the ROI covered by unmasked pixels
        percentage_unmasked = ee.Number(unmasked_area).divide(aoi_area).multiply(100)
        return ee_image.set('percent_AOI_coverage', percentage_unmasked).copyProperties(ee_image)

    # -----Define output image names, check if already exists in directory
    arcticdem_fn = site_name + '_ArcticDEM_clip.tif'
    arcticdem_geoid_fn = site_name + '_ArcticDEM_clip_geoid.tif'
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
        dem_ds = ellipsoid_to_geoid_heights(dem_ds, out_path, arcticdem_geoid_fn)
    elif os.path.exists(os.path.join(out_path, nasadem_fn)):
        print('Clipped NASADEM already exists in directory, loading...')
        dem_ds = xr.open_dataset(os.path.join(out_path, nasadem_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)
    else:  # if no DEM exists in directory, load from GEE

        # -----Reformat AOI for clipping DEM
        aoi_ee = ee.Geometry.Polygon(list(zip(aoi.geometry[0].exterior.coords.xy[0],
                                              aoi.geometry[0].exterior.coords.xy[1])))

        # -----Check for ArcticDEM coverage over AOI
        # load ArcticDEM_Mosaic_coverage.shp
        arcticdem_coverage = gpd.read_file(os.path.join(os.getcwd(), '..', 'inputs-outputs', 'ArcticDEM_Mosaic_coverage.shp'))
        # reproject to optimal UTM zone
        arcticdem_coverage_utm = arcticdem_coverage.to_crs(f'EPSG:{epsg_utm}')
        # check for intersection with AOI
        intersects = arcticdem_coverage_utm.geometry[0].intersects(aoi_utm.geometry[0])
        # check for actual coverage of ArcticDEM (some sites have nearly empty DEM coverage even within data boundaries)
        coverage = False
        if intersects:
            dem = ee.Image('UMN/PGC/ArcticDEM/V4/2m_mosaic').clip(aoi_ee).select('elevation')
            percent_coverage = calculate_percent_image_aoi_coverage(dem).get('percent_AOI_coverage').getInfo()
            if percent_coverage > 70:
                coverage = True
        # use ArcticDEM if intersects==True and coverage==True
        if intersects & coverage:
            print('ArcticDEM coverage over AOI')
            dem = ee.Image('UMN/PGC/ArcticDEM/V4/2m_mosaic').select('elevation').clip(aoi_ee)
            dem_fn = arcticdem_fn  # file name for saving
            scale = 10  # spatial resolution [m]
            elevation_source = 'ArcticDEM Mosaic (https://developers.google.com/earth-engine/datasets/catalog/UMN_PGC_ArcticDEM_V3_2m_mosaic)'
        else:
            print('No ArcticDEM coverage, using NASADEM')
            dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(aoi_ee)
            dem_fn = nasadem_fn  # file name for saving
            scale = 30  # spatial resolution [m]
            elevation_source = 'NASADEM (https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001)'
            # Check for NASADEM coverage
            percent_coverage = calculate_percent_image_aoi_coverage(dem, aoi_ee).get('percent_AOI_coverage').getInfo()
            if percent_coverage > 70:
                coverage = True

        # -----Check if either DEM had coverage over AOI
        if not coverage:
            print('Neither ArcticDEM nor NASADEM have > 70% coverage over the AOI. Please acquire a different DEM.')
            return

        # -----Download DEM and open as xarray.Dataset
        print('Downloading DEM to ', out_path)
        # create out_path if it doesn't exist
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # convert DEM to geedim MaskedImage
        dem_gd = gd.MaskedImage(dem, mask=False, region=aoi_ee)
        # download DEM
        dem_gd.download(os.path.join(out_path, dem_fn), region=aoi_ee, scale=scale, crs="EPSG:4326")
        # read DEM as xarray.Dataset
        dem_ds = xr.open_dataset(os.path.join(out_path, dem_fn))
        dem_ds = adjust_dem_data_vars(dem_ds)

        # -----If using ArcticDEM, transform elevations with respect to the geoid (rather than the ellipsoid)
        if 'ArcticDEM' in elevation_source:
            dem_ds = ellipsoid_to_geoid_heights(dem_ds, out_path, arcticdem_geoid_fn)

    # -----Reproject DEM to UTM
    dem_ds = dem_ds.rio.reproject(f'EPSG:{epsg_utm}')
    dem_ds = xr.where((dem_ds > 1e38) | (dem_ds <= -9999), np.nan, dem_ds)
    dem_ds = dem_ds.rio.write_crs(f'EPSG:{epsg_utm}')

    return dem_ds


def query_gee_for_imagery(aoi_utm, dataset, start_date, end_date, start_month, end_month, mask_clouds,
                          percent_aoi_coverage, im_download, out_path, run_pipeline, dataset_dict, site_name,
                          im_classified_path, snow_cover_stats_path, dem, clf, feature_cols, figures_out_path,
                          plot_results, verbose, delineate_snowline=False):
    """

    Parameters
    ----------
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        Area of interest for querying and clipping imagery.
    dataset: str
        Name of dataset: "Landsat", "Sentinel-2_SR", or "Sentinel-2_TOA"
    start_date: str
        Start date for image querying, format: "YYYY-MM-DD"
    end_date: str
        End date for image querying, format: "YYYY-MM-DD"
    start_month: int
        Start month for image querying, inclusive
    end_month: int
        End month for image querying, inclusive
    mask_clouds: bool
        Whether to mask clouds using the approach by Moussavi et al. (2020)
    percent_aoi_coverage: int/float
        Minimum percent coverage of the AOI after cloud masking for filtering images (e.g., 50% AOI coverage = 50).
    im_download: bool
        Whether to download images to out_path. If image sizes exceed the GEE user memory limit (10 MB), images will be
        downloaded regardless.
    out_path: str
        Path for output images. Only used if im_download = True or image sizes exceed the GEE user memory limit
    run_pipeline: bool
        Whether to run the classification pipeline for each image.
    -----Options if run_pipeline=True-----
    dataset_dict: dict
        Dictionary of dataset characteristics
    site_name: str
        Name of site, used for output file names
    im_classified_path: str
        Path in directory where classified images will be saved
    snow_cover_stats_path: str
        Path in directory where snow cover statistics will be saved
    dem: xarray.Dataset
        Digital elevation model (DEM) over the area of interest
    clf: sklearn pre-trained model
        classifier applied to the input image, specific to image dataset
    feature_cols: list of str
        features (i.e., image bands and NDSI) to use for classifying
    figures_out_path: str
        path where figures will be saved
    plot_results: bool
        whether to plot results and save figure
    verbose: bool
        whether to output details about each image
    delineate_snowline: bool
        whether to delineate snowline in each classified image

    Returns
    -------

    """
    # Load and reformat AOI as ee.Geometry.Polygon
    epsg_utm = aoi_utm.crs.to_epsg()
    aoi_wgs = aoi_utm.to_crs("EPSG:4326")
    aoi_ee = ee.Geometry.Polygon(list(zip(aoi_wgs.geometry[0].exterior.coords.xy[0],
                                          aoi_wgs.geometry[0].exterior.coords.xy[1])))
    # Calculate the total area of the AOI
    aoi_area = aoi_ee.area()

    # Define dataset-specific properties
    scale = dataset_dict[dataset]['resolution_m']
    analysis_bands = [x for x in feature_cols if x != 'NDSI']
    ndsi_bands = dataset_dict[dataset]['NDSI_bands']

    # Query for imagery
    if dataset == 'Landsat':
        # Get full collection for dates and bounds
        im_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                  .filterBounds(aoi_ee)
                  .filter(ee.Filter.date(start_date, end_date))
                  .filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
                  .select(analysis_bands)
                  )
        # Apply scale factors
        def apply_scale_factors(image):
            adjusted_bands = image.multiply(0.0000275).add(-0.2)
            return image.addBands(adjusted_bands, None, True).copyProperties(ee.Image(image))
        im_col = im_col.map(apply_scale_factors)
    elif (dataset == 'Sentinel-2_SR') or (dataset == 'Sentinel-2_TOA'):
        # Get full collection for dates and bounds
        if dataset == 'Sentinel-2_TOA':
            col_name = "COPERNICUS/S2_HARMONIZED"
        elif dataset == 'Sentinel-2_SR':
            col_name = "COPERNICUS/S2_SR_HARMONIZED"
        im_col = (ee.ImageCollection(col_name)
                  .filterBounds(aoi_ee)
                  .filter(ee.Filter.date(start_date, end_date))
                  .filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
                  .select(analysis_bands)
                  )
        # Apply scale factors
        def apply_scale_factors(image):
            adjusted_bands = image.multiply(0.0001)
            return image.addBands(adjusted_bands, None, True).copyProperties(ee.Image(image))
        im_col = im_col.map(apply_scale_factors)
    else:
        print(
            "'dataset' variable not recognized. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. "
            "Exiting...")
        return

    # Clip to AOI
    def clip_to_aoi(image):
        return image.clip(aoi_ee).copyProperties(ee.Image(image))

    im_col = im_col.map(clip_to_aoi)

    # Sort by capture date
    im_col = im_col.sort('system:time_start')

    # Add NDSI band
    def calculate_ndsi(image):
        ndsi = image.expression(
            '((GREEN - SWIR1) / (GREEN + SWIR1))',
            {'GREEN': image.select(ndsi_bands[0]),
             'SWIR1': image.select(ndsi_bands[1])}).rename('NDSI')
        return image.addBands(ndsi, None, True)
    im_col = im_col.map(calculate_ndsi)

    # Apply cloud masking approach from Moussavi et al. (2020): https://doi.org/10.3390/rs12010134
    if mask_clouds:
        def create_apply_cloud_mask(image):
            ndsi = image.select('NDSI')
            ndsi_mask = ndsi.lt(0.8)
            swir_mask = image.select(ndsi_bands[1]).gt(0.1)
            cloud_mask = ndsi_mask.And(swir_mask).rename('cloud_mask')
            # Add NDSI band
            image = image.addBands(ndsi, None, True)
            # Apply cloud mask to image
            image_masked = image.updateMask(cloud_mask.Not())
            return image_masked.copyProperties(ee.Image(image))
        im_col = im_col.map(create_apply_cloud_mask)

    # Mosaic images captured same day
    def mosaic_image_collection_by_date(collection):
        # Functions to mosaic images for each date
        def mosaic_by_date(date_str):
            date = ee.Date(date_str)
            # Filter images by the date
            day_images = collection.filter(ee.Filter.date(date, date.advance(1, 'day')))
            # Mosaic the images for this date
            return day_images.mosaic().set({'system:time_start': date.millis(), 'date': date_str})
        # Map over each unique date and create mosaics
        mosaics_collection = dates.map(lambda d: mosaic_by_date(d))
        return ee.ImageCollection(mosaics_collection)
    # Create a list of unique dates in the collection
    dates = im_col.aggregate_array('system:time_start').map(lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct()
    im_col_mosaicked = mosaic_image_collection_by_date(im_col)

    # Calculate % coverage of the AOI
    def calculate_percent_image_aoi_coverage(ee_image):
        # Create binary image of masked (0) and unmasked (1) pixels
        unmasked_pixels = ee_image.mask().reduce(ee.Reducer.allNonZero()).selfMask()
        # Calculate the area of unmasked pixels in the ROI
        pixel_area = ee.Image.pixelArea()
        aoi_area = aoi_ee.area()
        unmasked_area = unmasked_pixels.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi_ee,
            scale=scale,
            maxPixels=1e13
        ).get('all')
        # Calculate the percentage of the ROI covered by unmasked pixels
        percentage_unmasked = ee.Number(unmasked_area).divide(aoi_area).multiply(100)
        return ee_image.set('percent_AOI_coverage', percentage_unmasked).copyProperties(ee_image)
    im_col_mosaicked = im_col_mosaicked.map(calculate_percent_image_aoi_coverage)

    # Filter based on percent coverage of the ROI
    im_col_mosaicked_filtered = im_col_mosaicked.filter(ee.Filter.gte('percent_AOI_coverage', percent_aoi_coverage))

    # Determine whether images must be downloaded to file
    if not im_download:
        # Estimate image size
        image_size_bits = aoi_area.getInfo() / (scale ** 2) * len(analysis_bands) * 64
        image_size_mb = image_size_bits / (8 * 1024 * 1024)
        # If image size > 10 MB, images must be downloaded
        if image_size_mb > 10:
            im_download = True
            print('Image sizes exceed GEE user limit, images must be downloaded to file.')
        # else:
        #     print('Images are within GEE user limit, no need to download.')
    # Make sure out_path exists if downloading files
    if im_download:
        if not os.path.exists(out_path):
            os.mkdir(out_path)

    # Download images by date
    im_xr_list = []
    dates = im_col_mosaicked_filtered.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct().getInfo()
    for date in tqdm(dates):
        im = im_col_mosaicked_filtered.filter(ee.Filter.date(date, str(np.datetime64(date)
                                                                       + np.timedelta64(1, 'D')))).first()
        if im_download:
            im_gd = gd.MaskedImage(im, mask=False)
            # Determine file name
            im_fn = os.path.join(out_path, f"{str(date).replace('-','')}_{dataset}.tif")
            # Download to file
            download_bands = analysis_bands + ['NDSI']
            if not os.path.exists(im_fn):
                im_gd.download(im_fn, region=aoi_ee, scale=scale, bands=download_bands, crs='EPSG:4326')
            # Open from file
            im_da = rxr.open_rasterio(im_fn)
            im_xr = im_da.to_dataset('band')
            # rename bands
            im_xr = im_xr.rename({i + 1: name for i, name in enumerate(download_bands)})
            # set CRS
            im_xr.rio.write_crs('EPSG:4326', inplace=True)
            # Add time dimension
            im_xr = im_xr.expand_dims(dim={'time': [np.datetime64(date, 'ns')]})
        else:
            im_xr = im.wx.to_xarray(region=aoi_ee, scale=scale, crs='EPSG:4326')

        # Reproject to UTM for better distance calculations later
        im_xr = im_xr.rio.write_crs("EPSG:4326")
        im_xr = im_xr.rio.reproject(f'EPSG:{epsg_utm}', nodata=np.nan)

        if run_pipeline:
            # -----Run classification pipeline
            apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path,
                                          snow_cover_stats_path, aoi_utm, dem, epsg_utm, clf, feature_cols,
                                          figures_out_path, plot_results, verbose, delineate_snowline=delineate_snowline)
        else:
            im_xr_list += [im_xr]

    return im_xr_list


def query_gee_for_imagery_yearly(aoi_utm, dataset, start_date, end_date, start_month, end_month, mask_clouds,
                                 percent_aoi_coverage, im_download, out_path, run_pipeline=False, dataset_dict=None,
                                 site_name=None, im_classified_path=None, snow_cover_stats_path=None, dem=None,
                                 clf=None, feature_cols=None, figures_out_path=None, plot_results=None, verbose=False, delineate_snowline=False):
    # -----Define date ranges for querying imagery
    # Convert start_date and end_date to pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    # Get start and end years
    start_year = start_date.year
    end_year = end_date.year
    # Initialize list to store date ranges
    date_ranges = []
    # Iterate over years
    for year in range(start_year, end_year + 1):
        # Define start and end dates for the year
        range_start = pd.Timestamp(year=year, month=start_month, day=1)
        range_end = pd.Timestamp(year=year, month=end_month, day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
        # Make sure the range is within the provided start_date and end_date
        if year == start_year:
            range_start = max(range_start, start_date)
        if year == end_year:
            range_end = min(range_end, end_date)
        # Add range to list if valid
        if range_start <= range_end:
            date_ranges.append((range_start.strftime('%Y-%m-%d'), range_end.strftime('%Y-%m-%d')))

    # -----Return pipeline for each date range
    for date_range in date_ranges:
        if verbose:
            print(' ')
            print(date_range[0], date_range[1])
        im_list = []
        im_list += query_gee_for_imagery(aoi_utm, dataset, date_range[0], date_range[1], start_month, end_month,
                                         mask_clouds, percent_aoi_coverage, im_download, out_path, run_pipeline,
                                         dataset_dict, site_name, im_classified_path, snow_cover_stats_path, dem, clf,
                                         feature_cols, figures_out_path, plot_results, verbose)

    print('Done!')

    return im_list


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

    # -----Define image capture date
    im_date = np.datetime64(str(im_xr.time.data[0]))

    # -----Crop image to the AOI and remove time dimension
    # Create dummy band for AOI masking comparison
    im_xr['aoi_mask'] = (['y', 'x'], np.ones(np.shape(im_xr[dataset_dict[dataset]['RGB_bands'][0]].data[0])))
    im_aoi = im_xr.rio.clip(aoi.geometry, im_xr.rio.crs).isel(time=0)

    # -----Prepare image for classification
    # find indices of real numbers (no NaNs allowed in classification)
    ix = [np.where((np.isfinite(im_aoi[band].data) & ~np.isnan(im_aoi[band].data)), True, False)
          for band in feature_cols]
    ireal = np.full(np.shape(im_aoi[feature_cols[0]].data), True)
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
    # replace NaNs with -9999, convert data types to int
    im_classified_xr_int = im_classified_xr.fillna(-9999).astype(int)
    # reproject to WGS84 horizontal coordinates for consistency before saving
    im_classified_xr_int = im_classified_xr_int.rio.reproject('EPSG:4326')
    # add additional attributes to image before saving
    im_classified_xr_int = im_classified_xr_int.assign_attrs({'Description': 'Classified image',
                                                              'Classes': '1 = Snow, 2 = Shadowed snow, 3 = Ice, '
                                                                         '4 = Rock, 5 = Water',
                                                              'source': dataset,
                                                              'date': str(im_date),
                                                              '_FillValue': '-9999'
                                                              })
    # remove attributes that are no longer applicable
    for attr in ['add_offset', 'long_name', 'name', 'system-index']:
        if attr in im_classified_xr.attrs.keys():
            del im_classified_xr.attrs[attr]

    # -----Save to file
    if '.nc' in im_classified_fn:
        im_classified_xr_int.to_netcdf(os.path.join(out_path, im_classified_fn))
    elif '.tif' in im_classified_fn:
        im_classified_xr_int.rio.to_raster(os.path.join(out_path, im_classified_fn))
    if verbose:
        print('Classified image saved to file: ' + os.path.join(out_path, im_classified_fn))

    return im_classified_xr


def calculate_snow_cover_stats(dataset_dict, dataset, im_date, im_classified, dem, aoi, site_name='', delineate_snowline=False, 
                               scs_fn=None, out_path=None, figures_out_path=None, plot_results=False, verbose=True):
    """
    Calculate snow cover statistics, including the area of each image class, the snowline altitude, the transient AAR, 
    and optionally, delineate the snowline. 

    Parameters
    ----------
    dataset_dict: dict
        Dictionary of dataset characteristics
    dataset: str
        Name of dataset: "Landsat", "Sentinel-2_SR", "Sentinel-2_TOA", or "PlanetScope"
    im_date: str
        Date of image capture, format: "YYYY-MM-DD"
    im_classified: xarray.Dataset
        Classified image dataset
    dem: xarray.Dataset
        Digital elevation model (DEM) over the area of interest
    aoi: geopandas.geodataframe.GeoDataFrame
        Area of interest for querying and clipping imagery.
    site_name: str
        Name of site, used for output file names
    delineate_snowline: bool
        Whether to delineate the snowline from the classified image
    scs_fn: str
        File name for snow cover statistics output file
    out_path: str
        Path for output snow cover statistics. Only used if scs_fn is not None
    figures_out_path: str
        Path for output figures. Only used if plot_results is True
    plot_results: bool
        Whether to plot results and save figure
    verbose: bool
        Whether to output details about each image
    
    Returns
    -------
    scs_df: pandas.DataFrame
        DataFrame containing snow cover statistics
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
    im_dt = np.datetime64(im_date)
    im_classified = im_classified.isel(time=0)

    # -----Calculate area of each class
    dx = im_classified.x.data[1] - im_classified.x.data[0]
    snow_area = len(np.ravel(im_classified.classified.data[im_classified.classified.data <= 2])) * (dx ** 2)
    ice_area = len(np.ravel(im_classified.classified.data[im_classified.classified.data == 3])) * (dx ** 2)
    rock_area = len(np.ravel(im_classified.classified.data[im_classified.classified.data == 4])) * (dx ** 2)
    water_area = len(np.ravel(im_classified.classified.data[im_classified.classified.data == 5])) * (dx ** 2)
    glacier_area = ice_area + snow_area

    # -----Calculate transient accumulation area ratio (AAR)
    aar = snow_area / glacier_area    

    # -----Calculate the snow line altitude (SLA) from the AAR and the DEM
    dem_clip = dem.rio.clip(aoi.geometry, aoi.crs)
    dem_clip = xr.where(dem_clip < 3e38, dem_clip, np.nan)
    elevations = np.ravel(dem_clip.elevation.data)
    sla_from_aar = np.nanquantile(elevations, 1 - aar)

    # -----Compile results in dataframe
    scs_df = pd.DataFrame({'RGIId': [site_name],
                            'source': [dataset],
                            'datetime': [im_dt],
                            'HorizontalCRS': [f'EPSG:{im_classified.rio.crs.to_epsg()}'],
                            'VerticalCRS': ['EPSG:5773'],
                            'snow_area_m2': [snow_area],
                            'ice_area_m2': [ice_area],
                            'rock_area_m2': [rock_area],
                            'water_area_m2': [water_area],
                            'glacier_area_m2': [glacier_area],
                            'transient_AAR': [aar],
                            'SLA_m': [sla_from_aar],
                            })
    
    # -----Delineate snowline
    if delineate_snowline:
        snowlines_coords_x, snowlines_coords_y, snowline_elevs = delineate_snowline_from_image(im_classified, aoi, dem, ds_dict)
        scs_df['snowlines_coords_X'] = snowlines_coords_x
        scs_df['snowlines_coords_Y'] = snowlines_coords_y
        scs_df['snowline_elevs_m'] = [snowline_elevs]

    # -----Save snowline df to file
    # reduce memory storage of dataframe
    scs_df = reduce_memory_usage(scs_df, verbose=False)
    # save using user-specified file extension
    if 'pkl' in scs_fn:
        scs_df.to_pickle(os.path.join(out_path, scs_fn))
        if verbose:
            print('Snowline saved to file: ' + os.path.join(out_path, scs_fn))
    elif 'csv' in scs_df:
        scs_df.to_csv(os.path.join(out_path, scs_df), index=False)
        if verbose:
            print('Snowline saved to file: ' + os.path.join(out_path, scs_fn))
    else:
        print('Please specify snowline_fn with extension .pkl or .csv. Exiting...')
        return 'N/A'

    # -----Plot results
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
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
        if delineate_snowline & (len(snowlines_coords_x) > 0):
            ax[0].plot(np.divide(snowlines_coords_x, 1e3), np.divide(snowlines_coords_y, 1e3),
                       '.', color='#f768a1', markersize=2)
            ax[1].plot(np.divide(snowlines_coords_x, 1e3), np.divide(snowlines_coords_y, 1e3),
                       '.', color='#f768a1', markersize=2)
            ax[0].scatter(0, 0, color='#f768a1', s=30, label='Snowline estimate')
            ax[1].scatter(0, 0, color='#f768a1', s=30, label='Snowline estimate')
        # plot dummy points for legend
        ax[1].scatter(0, 0, color=colors[0], s=30, marker='s', label='Snow')
        ax[1].scatter(0, 0, color=colors[1], s=30, marker='s', label='Shadowed snow')
        ax[1].scatter(0, 0, color=colors[2], s=30, marker='s', label='Ice')
        ax[1].scatter(0, 0, color=colors[3], s=30, marker='s', label='Rock')
        ax[1].scatter(0, 0, color=colors[4], s=30, marker='s', label='Water')
        ax[0].set_ylabel('Northing [km]')
        ax[0].set_xlabel('Easting [km]')
        ax[1].set_xlabel('Easting [km]')
        # AOI
        if type(aoi.geometry[0].boundary) == MultiLineString:
            for ii, geom in enumerate(aoi.geometry[0].boundary.geoms):
                if ii > 0:
                    label = '_nolegend_'
                else:
                    label = 'AOI'
                ax[0].plot(np.divide(geom.coords.xy[0], 1e3),
                           np.divide(geom.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
                ax[1].plot(np.divide(geom.coords.xy[0], 1e3),
                           np.divide(geom.coords.xy[1], 1e3), '-k', linewidth=1, label=label)
        elif type(aoi.geometry[0].boundary) == LineString:
            ax[0].plot(np.divide(aoi.geometry[0].boundary.coords.xy[0], 1e3),
                       np.divide(aoi.geometry[0].boundary.coords.xy[1], 1e3), '-k', linewidth=1, label='AOI')
            ax[1].plot(np.divide(aoi.geometry[0].boundary.coords.xy[0], 1e3),
                       np.divide(aoi.geometry[0].boundary.coords.xy[1], 1e3), '-k', linewidth=1, label='AOI')
        # reset x and y limits
        ax[0].set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        # add legend
        handles, labels = ax[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncols=len(labels))
        # determine figure title and file name
        title = im_date + ' ' + site_name + ' ' + dataset + ' snow cover'
        fig.suptitle(title)
        fig.tight_layout()
        # save figure
        fig_fn = os.path.join(figures_out_path, title.replace(' ', '_').replace('-','') + '.png')
        fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
        if verbose:
            print('Figure saved to file:' + fig_fn)

    return scs_df


def delineate_snowline_from_image(im_classified, aoi, dem, ds_dict):
    """
    Delineate the seasonal snowline in classified images. Snowlines will likely not be detected in images with nearly
    all or no snow.

    Parameters
    ----------
    im_classified: xarray.Dataset
        classified image
    aoi:  geopandas.geodataframe.GeoDataFrame
        area of interest
        must be in the same coordinate reference system as the classified image
    dem: xarray.Dataset
        digital elevation model over the aoi
        must be in the same coordinate reference system as the classified image
    ds_dict: dict
        dictionary of dataset-specific parameters

    Returns
    ----------
    snowlines_coords_x: list
        x-coordinates of the snowline(s)
    snowlines_coords_y: list
        y-coordinates of the snowline(s)
    snowline_elevs: list
        surface elevation at each snowline coordinate
    """

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
    dem_aoi = xr.where(dem_aoi < 3e38, dem_aoi, np.nan)
    dem_aoi_interp = dem_aoi.interp(x=im_classified.x.data, y=im_classified.y.data, method='linear')
    dem_aoi_interp = xr.where(dem_aoi_interp < 3e38, dem_aoi_interp, np.nan)
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
        snowline_elevs = []

    # -----If AOI is ~covered in snow, set snowline elevation to the minimum elevation in the AOI
    if np.all(np.isnan(snowline_elevs)) and (np.nanmedian(hist_snow_est_elev_norm) > 0.5):
        snowline_elevs = np.nanmin(np.ravel(im_classified.elevation.data))

    # -----Reformat coords as list
    if type(snowline) == LineString:
        snowlines_coords_x = [list(snowline.coords.xy[0])]
        snowlines_coords_y = [list(snowline.coords.xy[1])]
    else:
        snowlines_coords_x = [[]]
        snowlines_coords_y = [[]]

    return snowlines_coords_x, snowlines_coords_y, snowline_elevs


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


def apply_classification_pipeline(im_xr, dataset_dict, dataset, site_name, im_classified_path, snow_cover_stats_path,
                                  aoi_utm, dem, epsg_utm, clf, feature_cols, figures_out_path,
                                  plot_results, verbose, delineate_snowline=False):
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
    snow_cover_stats_path: str
        path in directory where snow cover statistics CSV files will be saved
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
    im_date = str(im_xr.time.data[0].astype('datetime64[D]'))

    # Check if classified image already exists in file
    im_classified_fn = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_classified.nc'
    if not os.path.exists(os.path.join(im_classified_path, im_classified_fn)):
        # classify image
        im_classified = classify_image(im_xr, clf, feature_cols, aoi_utm,
                                       dataset_dict, dataset, im_classified_fn, im_classified_path, verbose)
        if type(im_classified) == str:  # skip if error in classification
            return

    # Check if snow cover stats already exists in file
    snow_cover_stats_fn = (im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset
                           + '_snow_cover_stats.csv')
    if os.path.exists(os.path.join(snow_cover_stats_path, snow_cover_stats_fn)):
        # No need to load snow cover stats if it already exists
        return
    else:
        # Load classified image
        im_classified = xr.open_dataset(os.path.join(im_classified_path, im_classified_fn))
        # remove no data values
        im_classified = xr.where(im_classified == -9999, np.nan, im_classified)
        im_classified = im_classified.rio.write_crs('EPSG:4326').rio.reproject(f'EPSG:{epsg_utm}')

        # Calculate snow cover stats
        snowline_df = calculate_snow_cover_stats(dataset_dict, dataset, im_date, im_classified, dem, aoi_utm, 
                                                 site_name, delineate_snowline, snow_cover_stats_fn, snow_cover_stats_path, 
                                                 figures_out_path, plot_results, verbose)
        plt.close()

    return snowline_df


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
    region_buffer_ee = ee.Geometry.Polygon(
        [[[aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.miny[0]],
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
        print(
            "'dataset' variable not recognized. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. Exiting...")
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

