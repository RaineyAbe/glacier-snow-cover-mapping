# Functions for image adjustment and snow classification in Landsat surface reflectance imagery
# Rainey Aberle
# 2022

import math
import rasterio as rio
from rasterio.mask import mask
import numpy as np
from pyproj import Proj, transform, Transformer
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import os
from shapely.geometry import Polygon, MultiPolygon, shape, Point, LineString
from scipy.interpolate import interp2d, griddata
from scipy.signal import medfilt
from scipy.stats import iqr
from skimage.measure import find_contours
from scipy.ndimage import binary_fill_holes
import glob
import ee
import geopandas as gpd
import pandas as pd
from scipy import stats
import geemap
from osgeo import gdal
import wxee as wx
import xarray as xr
import rioxarray as rxr
from symfit import parameters, variables, sin, cos, Fit
from sklearn.model_selection import train_test_split
import pickle


# --------------------------------------------------
def query_GEE_for_DEM(AOI):
    '''Query GEE for the ASTER Global DEM, clip to the AOI, and return as a numpy array.

    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM

    Returns
    ----------
    DEM_ds: xarray.Dataset
        elevations extracted within the AOI
    AOI_UTM: geopandas.geodataframe.GeoDataFrame
        AOI reprojected to the appropriate UTM coordinate reference system
    '''

    # -----Reformat AOI for clipping DEM
    # reproject AOI to WGS 84 for compatibility with DEM
    AOI_WGS = AOI.to_crs(4326)
    # reformat AOI_WGS bounding box as ee.Geometry for clipping DEM
    AOI_WGS_bb_ee = ee.Geometry.Polygon(
                            [[[AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]]]
                            ]).buffer(1000)

    # -----Query GEE for DEM, clip to AOI
    DEM = ee.Image("NASA/ASTER_GED/AG100_003").clip(AOI_WGS_bb_ee).select('elevation')

    # -----Grab optimal UTM zone, reproject AOI
    AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                        AOI_WGS.geometry[0].centroid.xy[1][0]]
    epsg_UTM = convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
    AOI_UTM = AOI.to_crs(str(epsg_UTM))

    # -----Convert DEM to xarray.Dataset
    DEM = DEM.set('system:time_start', 0) # set an arbitrary time
    DEM_ds = DEM.wx.to_xarray(scale=30, crs='EPSG:'+str(epsg_UTM))

    return DEM_ds, AOI_UTM
    
    
# --------------------------------------------------
def query_GEE_for_Landsat_SR(AOI, date_start, date_end, month_start, month_end, cloud_cover_max, ds_dict):

    # reproject AOI to WGS for image searching
    AOI_WGS = AOI.to_crs(4326)
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
    L = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
             .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))
             .filterDate(ee.Date(date_start), ee.Date(date_end))
             .filter(ee.Filter.calendarRange(month_start, month_end, 'month'))
             .filterBounds(AOI_WGS_bb_ee))
    # define band names
    L_band_names = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    #  clip images to AOI and select bands
    L_clip = L.map(clip_image).select(L_band_names)
    # convert image collection to xarray Dataset
    L_xr = L_clip.wx.to_xarray(scale=30, crs='EPSG:4326')
    # reproject to UTM CRS
    L_xr = L_xr.rio.reproject('EPSG:'+epsg_UTM)
    # replace no data values with NaN
    for band in L_band_names:
        L_xr[band] = L_xr[band].where(L_xr[band] != ds_dict['no_data_value'])
    # account for image scalar
    for band in L_band_names[0:-1]:
        L_xr[band] = L_xr[band] / ds_dict['SR_scalar']
    # Add NDSI band
    NDSI_bands = ds_dict['NDSI']
    L_xr['NDSI'] = ((L_xr[NDSI_bands[0]] - L_xr[NDSI_bands[1]]) / (L_xr[NDSI_bands[0]] + L_xr[NDSI_bands[1]]))
    
    return L_xr

# --------------------------------------------------
def query_GEE_for_Sentinel2_SR(dataset, site_name, AOI, date_start, date_end, month_start, month_end, cloud_cover_max, ds_dict, out_path, plot_results):
    '''
    Query GEE for Sentinel-2 surface reflectance (SR) imagery and apply a mask for clouds and shadows. Cloud masking method using s2cloudless adapted from: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    
    Parameters
    ----------
    dataset: str
        name of dataset
    site_name: str
        site name
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest
    date_start: str
        start date for image search
    date_end: str
        end date for image search
    month_start: str
        starting month for calendar range filtering
    month_end: str
        ending month for calendar range filtering
    cloud_cover max: float
        maximum cloud cover filter (0 - 100)
    ds_dict: dict
        dictionary of dataset-specific parameters
    out_path: str
        path in directory where output files will be saved
    plot_results: bool
        whether to plot image results
        
    Returns
    ----------
    S2_xr_fns: list
        list of Sentinel-2 image output file names
    '''
    # -----Reformat AOI for image filtering
    # reproject AOI to WGS
    AOI_WGS = AOI.to_crs(4326)
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
        
    # -----Query GEE for imagery
    print('Querying GEE for imagery...')
    S2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
             .filterDate(ee.Date(date_start), ee.Date(date_end))
             .filter(ee.Filter.calendarRange(month_start, month_end, 'month'))
             .filterBounds(AOI_WGS_bb_ee))
    #  clip images to AOI and select bands
    S2_band_names = [band for band in ds_dict['bands'] if 'QA' not in band]
    S2_clip = S2.map(clip_image).select(S2_band_names)
    if S2_clip.size().getInfo() < 1:
        print('No images found, exiting...')
        return

    # -----Apply cloud and shadow mask
    # define thresholds for cloud mask
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 1
    BUFFER = 50
    # Import and filter s2cloudless
    S2_cloudless = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(AOI_WGS_bb_ee)
        .filterDate(date_start, date_end))
    # clip to AOI
    S2_cloudless_clip = S2_cloudless.map(clip_image)
    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    S2_merge = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': S2_clip,
        'secondary': S2_cloudless_clip,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    
    def add_cloud_bands(img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        # Condition s2cloudless by the probability threshold value.
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, is_cloud]))
        
    def add_shadow_bands(img):
        # Identify water pixels from the clouds band.
        not_water = img.select('clouds').neq(6)
        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));
        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (img.select('probability').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
            .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
            .select('distance')
            .mask()
            .rename('cloud_transform'))
        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img):
        # Add cloud component bands.
        img_cloud = add_cloud_bands(img)
        # Add cloud shadow component bands.
        img_cloud_shadow = add_shadow_bands(img_cloud)
        # Combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
        # Remove small cloud-shadow patches and dilate remaining pixels by BUFFER input.
        # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
            .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
            .rename('cloudmask'))
        # Add the final cloud-shadow mask to the image.
        return img_cloud_shadow.addBands(is_cld_shdw)
        
    def apply_cld_shdw_mask(img):
        # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
        not_cld_shdw = img.select('cloudmask').Not()
        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld_shdw)

    # -----Add bands for clouds, shadows, mask, and apply the mask
    S2_merge_masked = (S2_merge.map(add_cloud_bands)
                               .map(add_shadow_bands)
                               .map(add_cld_shdw_mask)
                               .map(apply_cld_shdw_mask))

    # -----Filter image collection by coverage of the AOI
#    def getCover(image, AOI, scale):
#        # calculate the number of inputs
#        vtotPixels = ee.Number(image.unmask(1).reduceRegion({
#            reducer: ee.Reducer.count(),
#            scale: scale,
#            geometry: aoi,
#        }).values().get(0))
#        # calculate the actual amount of pixels inside the AOI
#        actPixels = ee.Number(image.reduceRegion({
#            reducer: ee.Reducer.count(),
#            scale: scale,
#            geometry: aoi,
#        }).values().get(0));
#        # calculate the percent coverage of the AOI
#        perc_AOI_cover = actPixels.divide(totPixels).multiply(100).round();
#        # add perc_cover as property
#        return image.set('perc_AOI_cover', perc_AOI_cover);
#
#    # apply percent coverage property
#    S2_merge_masked_AOIcover = S2_merge_masked.map(function(image){return getCover(image, aoi, 7000)})
#    # filter images with < 50% coverage of the AOI
#    S2_merge
    
    # -----Adjust and filter image collection
    print('Adjusting and filtering image collection by AOI coverage...')
    # convert ImageCollection to List
    S2_merge_masked_list = ee.ImageCollection(S2_merge_masked).toList(1e3)
    S2_ds_fns = []
    im_count = 0 # image counter
    # loop through images
    for i in range(0, S2_merge_masked_list.size().getInfo()):
        
        # -----Select image by index
        im = ee.Image(ee.List(S2_merge_masked_list).get(i))
        # get image time
        im_date = im.date().format(None, 'GMT').getInfo()
        print(im_date)
        
        # -----Check if file already exists in directory
        im_ds_fn = dataset + '_' + site_name + '_' + im_date.replace('-', '').replace(':','') + '_masked.nc'
        if len(glob.glob(out_path + im_ds_fn)) > 0:
            print('image already exists in directory, continuing...')
            S2_ds_fns = S2_ds_fns + [im_ds_fn]
            im_count+=1 # increase image counter
            continue
            
        # -----Convert image to xarray.Dataset
        im_xr = im.wx.to_xarray(scale=ds_dict['resolution_m'], crs='EPSG:4326')
        # reproject to UTM CRS
        im_xr_UTM = im_xr.rio.reproject('EPSG:'+epsg_UTM)
        # replace no data values with NaN and account for image scalar
        for band in S2_band_names:
            im_xr_UTM[band] = xr.where(im_xr_UTM[band] != ds_dict['no_data_value'],
                                       im_xr_UTM[band] / ds_dict['SR_scalar'], np.nan)
        # add NDSI band
        NDSI_bands = ds_dict['NDSI']
        im_xr_UTM['NDSI'] = ((im_xr_UTM[NDSI_bands[0]] - im_xr_UTM[NDSI_bands[1]]) / (im_xr_UTM[NDSI_bands[0]] + im_xr_UTM[NDSI_bands[1]]))
        
        # -----Skip image if it covers < 50% of the AOI
        # reproject AOI to image CRS is necessary
        AOI = AOI.to_crs(im_xr_UTM.rio.crs)
        # create bounding box of clip region
        AOI_clip_region = Polygon([[AOI.bounds.minx[0], AOI.bounds.miny[0]],
                                   [AOI.bounds.maxx[0], AOI.bounds.miny[0]],
                                   [AOI.bounds.maxx[0], AOI.bounds.maxy[0]],
                                   [AOI.bounds.minx[0], AOI.bounds.maxy[0]],
                                   [AOI.bounds.minx[0], AOI.bounds.miny[0]]])
        # clip image to AOI
        im_AOI = im_xr_UTM.rio.clip([AOI_clip_region], im_xr_UTM.rio.crs)
        # count number of pixels in image
        d = np.ravel(im_AOI.to_array().data[0])
        npx = len(d)
        # count number of non-NaN pixels in image
        npx_real = len(d[~np.isnan(d)])
        # percentage of non-NaN pixels in image
        p_real = npx_real / npx
        # skip if p_real < 0.50
        if p_real < 0.50:
            print('< 50% data coverage in the AOI, skipping...')
            print(' ')
            continue
        else:
            im_count+=1 # increase image counter
            # save to file
            im_xr_UTM.to_netcdf(out_path + im_ds_fn)
            print('Sentinel-2 masked SR image saved to file: ' + out_path + im_ds_fn)
            # concatenate image file name to list
            S2_ds_fns = S2_ds_fns + [im_ds_fn]
            # plot
            if plot_results:
                plt.imshow(np.dstack([im_xr_UTM[ds_dict['RGB_bands'][0]].data[0],
                                      im_xr_UTM[ds_dict['RGB_bands'][1]].data[0],
                                      im_xr_UTM[ds_dict['RGB_bands'][2]].data[0]]),
                           extent=(np.min(im_xr_UTM.x.data), np.max(im_xr_UTM.x.data), np.min(im_xr_UTM.y.data), np.max(im_xr_UTM.y.data)))
                plt.show()
            
        print(' ')

    # -----Check if any images were saved
    if im_count==0:
        print('No usable images found, exiting...')
        S2_ds_fns = 'N/A'

    return S2_ds_fns

# --------------------------------------------------
def query_GEE_for_MODIS_SR(AOI, date_start, date_end, month_start, month_end, cloud_cover_max, ds_dict):

    # reproject AOI to WGS for image searching
    AOI_WGS = AOI.to_crs(4326)
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
def plot_im_classified_histogram_contour(im, im_classified, DEM, AOI, contour):
    '''
    Plot the classified image with snow elevations histogram and plot an elevation contour corresponding to the estimated snow line elevation.

    Parameters
    ----------
    im: xarray.DataArray
        input image
    im_classified: xarray.DataArray
        single band, classified image
    DEM: xarray.DataSet
        digital elevation model over the image area
    DEM_rio: rasterio.DatasetReader
        digital elevation model opened using rasterio (to access the transform for masking)
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest
    contour: float
        elevation to be plotted as a contour on the figure

    Returns
    ----------
    fig: matplotlib.figure
        resulting figure handle
    ax: matplotlib.Axes
        axes handles on figure
    sl_points_AOI:
    '''

    # -----Determine snow-covered elevations
    # mask the DEM using the AOI
    mask = rio.features.geometry_mask(AOI.geometry,
                                      out_shape=(len(DEM.y), len(DEM.x)),
                                      transform=DEM_rio.transform,
                                      invert=True)
    mask = xr.DataArray(mask , dims=("y", "x"))
    # mask DEM values outside the AOI
    DEM_AOI = DEM.where(mask == True)
    # interpolate DEM to the image coordinates
    im_classified = im_classified.squeeze(drop=True) # drop uneccesary dimensions
    x, y = im_classified.indexes.values() # grab indices of image
    DEM_AOI_interp = DEM_AOI.interp(x=x, y=y, method="nearest") # interpolate DEM to image coordinates
    # determine snow covered elevations
    DEM_AOI_interp_snow = DEM_AOI_interp.where(im_classified<=2) # mask pixels not classified as snow
    snow_est_elev = DEM_AOI_interp_snow.elevation.data.flatten() # create array of snow-covered pixel elevations
    snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)] # remove NaN values

    # -----Determine bins to use in histogram
    elev_min = np.fix(np.nanmin(DEM_AOI_interp.elevation.data.flatten())/10)*10
    elev_max = np.round(np.nanmax(DEM_AOI_interp.elevation.data.flatten())/10)*10
    bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max-elev_min)/10 + 1))
    bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2

    # -----Calculate elevation histograms
    H_DEM = np.histogram(DEM_AOI_interp.elevation.data.flatten(), bins=bin_edges)[0]
    H_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
    H_snow_est_elev_norm = H_snow_est_elev / H_DEM

    # -----Plot
    fig, ax = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]})
    ax = ax.flatten()
#    plt.rcParams.update({'font.size': 14, 'font.sans-serif': 'Arial'})
    # define x and y limits
    xmin, xmax = np.min(im_classified.x.data)/1e3, np.max(im_classified.x.data)/1e3
    ymin, ymax = np.min(im_classified.y.data)/1e3, np.max(im_classified.y.data)/1e3
    # define colors for plotting
    color_snow = '#4eb3d3'
    color_ice = '#084081'
    color_rock = '#fdbb84'
    color_water = '#bdbdbd'
    color_contour = '#f768a1'
    # create colormap
    colors = [color_snow, color_snow, color_ice, color_rock, color_water]
    cmp = matplotlib.colors.ListedColormap(colors)
    # RGB image
    ax[0].imshow(np.dstack([im.data[2], im.data[1], im.data[0]]),
               extent=(xmin, xmax, ymin, ymax))
    ax[0].set_xlabel("Easting [km]")
    ax[0].set_ylabel("Northing [km]")
    # classified image
    ax[1].imshow(im_classified.data, cmap=cmp, vmin=1, vmax=5,
                 extent=(xmin, xmax, ymin, ymax))
    # plot dummy points for legend
    ax[1].scatter(0, 0, color=color_snow, s=50, label='snow')
    ax[1].scatter(0, 0, color=color_ice, s=50, label='ice')
    ax[1].scatter(0, 0, color=color_rock, s=50, label='rock')
    ax[1].scatter(0, 0, color=color_water, s=50, label='water')
    ax[1].set_xlabel('Easting [km]')
    # AOI
    if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
        for poly in AOI.geometry[0].geoms:
            ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
            ax[1].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
    else:
        ax[0].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
        ax[1].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
    # elevation contour - save only those inside the AOI
    if contour is not None:
        sl = plt.contour(DEM.x.data, DEM.y.data, DEM.elevation.data[0],[contour])
        sl_points_AOI = [] # initialize list of points
        for path in sl.collections[0].get_paths(): # loop through paths
            v = path.vertices
            for pt in v:
                pt_shapely = Point(pt[0], pt[1])
                if AOI.contains(pt_shapely)[0]:
                        sl_points_AOI.append([pt_shapely.xy[0][0], pt_shapely.xy[1][0]])
        ax[0].plot([pt[0]/1e3 for pt in sl_points_AOI], [pt[1]/1e3 for pt in sl_points_AOI], '.', color=color_contour, markersize=3, label='sl$_{estimated}$')
        ax[1].plot([pt[0]/1e3 for pt in sl_points_AOI], [pt[1]/1e3 for pt in sl_points_AOI], '.', color=color_contour, markersize=3, label='_nolegend_')
        ax[1].set_xlabel("Easting [km]")
    else:
        sl_points_AOI = None
    # reset x and y limits
    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_ylim(ymin, ymax)
    # image bands histogram
    h_b = ax[2].hist(im.data[0].flatten(), color='blue', histtype='step', linewidth=2, bins=100, label="blue")
    h_g = ax[2].hist(im.data[1].flatten(), color='green', histtype='step', linewidth=2, bins=100, label="green")
    h_r = ax[2].hist(im.data[2].flatten(), color='red', histtype='step', linewidth=2, bins=100, label="red")
    h_nir = ax[2].hist(im.data[3].flatten(), color='brown', histtype='step', linewidth=2, bins=100, label="NIR")
    ax[2].set_xlabel("Surface reflectance")
    ax[2].set_ylabel("Pixel counts")
    ax[2].legend(loc='best')
    ax[2].grid()
    # normalized snow elevations histogram
    ax[3].bar(bin_centers, H_snow_est_elev_norm, width=(bin_centers[1]-bin_centers[0]), color=color_snow, align='center')
    ax[3].set_xlabel("Elevation [m]")
    ax[3].set_ylabel("% snow-covered")
    ax[3].grid()
    ax[3].set_xlim(elev_min-10, elev_max+10)
    ax[3].set_ylim(0,1)
    # contour line
    if contour is not None:
        ax[3].plot((contour, contour), (0, 1), color=color_contour)
    fig.tight_layout()

    return fig, ax, sl_points_AOI


# --------------------------------------------------
def Landsat_mask_clouds(im_collection, AOI, ds_dict, plot_results):
    '''
    Adapted from: https://medium.com/analytics-vidhya/python-for-geosciences-raster-bit-masks-explained-step-by-step-8620ed27141e
    '''

    # -----Create dictionary of possible mask values
    mask_flags = {
            'dilated_cloud': 1<<1,
            'cirrus': 1<<2,
            'cloud': 1<<3,
#             'shadow': 1<<4,
#             'snow': 1<<5,
#             'clear': 1<<6,
    #         'water': 1<<7
        }

    # -----Define bands in imagery
    bands = [x for x in im_collection.data_vars]
    
    # -----Loop through times in image collection
    # loop through times
    im_count=0
    for i, t in enumerate(im_collection.time):
        
        # parse image date
        im_date = str(t.data)[0:10]
        print(im_date)
        
        # subset image from image collection using time
        im = im_collection.sel(time=t)
        im_array = im.to_array()
        # add time dimension
        im['time'] = t
            
        # initialize boolean matrix for final mask
        final_mask = np.full(np.shape(im['QA_PIXEL'].data), False)
        # loop through dictionary values
        for flag in mask_flags.items():

            # get the mask for this flag
            flag_mask = np.bitwise_and(im['QA_PIXEL'].data.astype(int), flag[1])

            # add it to the final flag
            final_mask = final_mask | flag_mask

        # apply mask to image bands (except 'QA_PIXEL')
        im_mask = im.copy()
        bands_noQA = [band for band in bands if band != 'QA_PIXEL']
        for band in bands_noQA:
            im_mask[band] = im_mask[band].where(final_mask ==0)
        im_mask = im_mask.rio.write_crs('EPSG:'+str(im.rio.crs.to_epsg()))

        # -----Skip image if it contains < 50% of the AOI
        # clip image to AOI
        AOI = AOI.to_crs(im.rio.crs)
        AOI_clip_region = Polygon([[AOI.bounds.minx[0], AOI.bounds.miny[0]],
                                   [AOI.bounds.maxx[0], AOI.bounds.miny[0]],
                                   [AOI.bounds.maxx[0], AOI.bounds.maxy[0]],
                                   [AOI.bounds.minx[0], AOI.bounds.maxy[0]],
                                   [AOI.bounds.minx[0], AOI.bounds.miny[0]]])
        im_AOI = im_mask.rio.clip([AOI_clip_region], im.rio.crs)
        d = np.ravel(im_AOI.to_array().data[0])
        # count number of pixels in image
        npx = len(d)
        # count number of non-NaN pixels in image
        npx_real = len(d[~np.isnan(d)])
        # percentage of non-NaN pixels in image
        p_real = npx_real / npx
        # skip if p_real < 0.75
        if p_real < 0.50:
            print('< 50% data coverage in the AOI, skipping...')
            continue
        else:
            
            # plot results
            if plot_results:
                fig, ax = plt.subplots(1, 2, figsize=(10,6))
                fig.suptitle(str(t.data)[0:10])
                # original image
                ax[0].imshow(np.dstack([im[ds_dict['RGB_bands'][0]].data,
                                        im[ds_dict['RGB_bands'][1]].data,
                                        im[ds_dict['RGB_bands'][2]].data]),
                             extent=(np.min(im.x.data)/1e3, np.max(im.x.data)/1e3,
                                     np.min(im.y.data)/1e3, np.max(im.y.data)/1e3))
                ax[0].set_xlabel('Easting [km]')
                ax[0].set_ylabel('Northing [km]')
                ax[0].set_title('Raw image')
                # masked image
                ax[1].imshow(np.dstack([im_mask[ds_dict['RGB_bands'][0]].data,
                                        im_mask[ds_dict['RGB_bands'][1]].data,
                                        im_mask[ds_dict['RGB_bands'][2]].data]),
                    extent=(np.min(im.x.data)/1e3, np.max(im.x.data)/1e3,
                                              np.min(im.y.data)/1e3, np.max(im.y.data)/1e3))
                ax[1].set_xlabel('Easting [km]')
                ax[1].set_title('Masked image')
                plt.show()
            
        # -----Concatenate image to xarray.Dataset
        if im_count==0:
            # create im_collection_masked xarray.Dataset for first image
            im_collection_masked = xr.Dataset(
                data_vars=im_mask.data_vars,
                coords=im_mask.coords,
                attrs=im_mask.attrs
                )
            # add time dimension
            im_collection_masked = im_collection_masked.expand_dims(dim={'time':[np.datetime64(im_mask.time.data)]})
            
        else:
            # create xarray.Dataset for image
            im_masked_xr = xr.Dataset(
                data_vars=im_mask.data_vars,
                coords=im_mask.coords,
                attrs=im_mask.attrs
                )
            # add time dimension
            im_masked_xr = im_masked_xr.expand_dims(dim={'time':[np.datetime64(im.time.data)]})
            # concatenate to classified image collection
            im_collection_masked = xr.concat([im_collection_masked, im_masked_xr], dim='time')
            
        im_count+=1 # increase image counter
            
    return im_collection_masked
    

# --------------------------------------------------
def classify_image_collection(im_collection, clf, feature_cols, crop_to_AOI, AOI, ds_dict, dataset, site_name, out_path, date_start, date_end, plot_results, figures_out_path):
    '''
    Function to classify image collection using a pre-trained classifier
    Parameters
    ----------
    im_collection: xarray.Dataset
        stack of Landsat images
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier
    crop_to_AOI: bool
        whether to mask everywhere outside the AOI before classifying
    AOI: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked if crop_to_AOI==True. AOI must be in the same CRS as the images.
    ds_dict: dict
    
    dataset: str
        the dataset name, e.g.: 'Landsat', 'MODIS', 'PlanetScope', 'Sentinel'
    date_start: str
    
    date_end: str
    
    out_path: str
        path to save classified images
    plot_results: bool
        whether to plot RGB and classified image
    figures_out_path: str
    Returns
    ----------
    im_collection_classified: xarray.Dataset
    
    im_collection_classified_fn: str
    
    fig:
    
    '''

    # -----Make directory for snow images (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print("Made directory for classified snow images:" + out_path)
        
    # -----check if classified snow image collection exists in directory already
    im_collection_classified_fn = dataset + '_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_masked_classified.nc'
    if os.path.exists(os.path.join(out_path, im_collection_classified_fn)):
        print('classified image collection already exists in file, loading...')
        im_collection_classified = xr.open_dataset(out_path + im_collection_classified_fn)
        fig = 'N/A'
        
    else:
        
        # -----Define image bands
        bands = [x for x in im_collection.data_vars]
        bands = [band for band in bands if (band != 'QA_PIXEL') and ('B' in band)]

        # -----Loop through image capture dates
        # loop through image capture dates
        im_count = 0
        for i, t in enumerate(im_collection.time):
        
            im_date = str(t.data)[0:10]
            print(im_date)
            

            # subset image collection to time
            im = im_collection.sel(time=t)
        
            # mask image pixels outside the AOI
            if crop_to_AOI:
                im_AOI = im.copy()
                # reproject AOI to im CRS if necessary
                AOI = AOI.to_crs("EPSG:"+str(im.rio.crs.to_epsg()))
                mask = np.zeros(np.shape(im.to_array().data[0]))
                if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
                    for poly in AOI.geometry[0].geoms:
                        d = {'geometry': [Polygon(poly.exterior)]}
                        gdf = gpd.GeoDataFrame(d, crs="EPSG:"+str(im.rio.crs.to_epsg()))
                        m = rio.features.geometry_mask(gdf.geometry,
                                                       im.to_array().data[0].shape,
                                                       im.rio.transform(),
                                                       all_touched=False,
                                                       invert=False)
                        mask[m==0] = 1
                elif AOI.geometry[0].geom_type=='Polygon':
                    d = {'geometry': [Polygon(AOI.geometry[0].exterior)]}
                    gdf = gpd.GeoDataFrame(d, crs="EPSG:"+str(im.rio.crs.to_epsg()))
                    m = rio.features.geometry_mask(gdf.geometry,
                                                   im.to_array().data[0].shape,
                                                   im.rio.transform(),
                                                   all_touched=False,
                                                   invert=False)
                    mask[m==0] = 1
                # apply mask to bands
                for band in bands:
                    im_AOI[band].data = np.where(mask==1, im_AOI[band].data, np.nan)
            else:
                im_AOI = im
            
            # find indices of real numbers (no NaNs allowed in classification)
            ix = [np.where(np.isnan(im_AOI[band].data), False, True) for band in bands]
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
                continue

            # reshape from flat array to original shape
            im_classified = np.zeros(im_AOI.to_array().data[0].shape)
            im_classified[:] = np.nan
            im_classified[I_real] = array_classified
            
            # -----Plot results
            if plot_results:
                fig, ax = plt.subplots(1, 2, figsize=(10,6))
                ax = ax.flatten()
                # define x and y limits
                xmin, xmax = np.min(im.x.data)/1e3, np.max(im.x.data)/1e3
                ymin, ymax = np.min(im.y.data)/1e3, np.max(im.y.data)/1e3
                # define colors for plotting
                color_snow = '#4eb3d3'
                color_ice = '#084081'
                color_rock = '#fdbb84'
                color_water = '#bdbdbd'
                color_contour = '#f768a1'
                # create colormap
                colors = [color_snow, color_snow, color_ice, color_rock, color_water]
                cmp = matplotlib.colors.ListedColormap(colors)
                # RGB image
                # ax[0].imshow(im[ds_dict['RGB_bands'][0]], cmap='Greys', extent=(xmin, xmax, ymin, ymax))
                ax[0].imshow(np.dstack([im[ds_dict['RGB_bands'][0]], # red
                                        im[ds_dict['RGB_bands'][1]], # green
                                        im[ds_dict['RGB_bands'][2]]]), # blue
                             extent=(xmin, xmax, ymin, ymax))
                ax[0].set_xlabel("Easting [km]")
                ax[0].set_ylabel("Northing [km]")
                ax[0].set_title('RGB image')
                # classified image
                ax[1].imshow(im_classified, cmap=cmp, vmin=1, vmax=5,
                             extent=(np.min(im_AOI.x.data)/1e3, np.max(im_AOI.x.data)/1e3,
                                     np.min(im_AOI.y.data)/1e3, np.max(im_AOI.y.data)/1e3))
                # plot dummy points for legend
                ax[1].scatter(0, 0, color=color_snow, s=50, label='snow')
                ax[1].scatter(0, 0, color=color_ice, s=50, label='ice')
                ax[1].scatter(0, 0, color=color_rock, s=50, label='rock')
                ax[1].scatter(0, 0, color=color_water, s=50, label='water')
                ax[1].set_title('Classified image')
                ax[1].set_xlabel('Easting [km]')
                ax[1].legend(loc='best')
                # AOI
                if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
                    for j, poly in enumerate(AOI.geometry[0].geoms):
                        # only include legend label for first geom
                        if j==0:
                            ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                        else:
                            ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                        ax[1].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                else:
                    ax[0].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                    ax[1].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                # reset x and y limits
                ax[0].set_xlim(xmin, xmax)
                ax[0].set_ylim(ymin, ymax)
                ax[1].set_xlim(xmin, xmax)
                ax[1].set_ylim(ymin, ymax)
                fig.suptitle(im_date)
                fig.tight_layout()
                plt.show()
    
                # save figure
                fig_date = im_date.replace('-','')
                fig_fn = figures_out_path + dataset + '_' + site_name + '_' + im_date + '_SCA.png'
                fig.savefig(fig_fn, dpi=300, facecolor='w')
                print('figure saved to file: ' + fig_fn)
                
            else:
            
                fig = 'N/A'

            # concatenate image to xarray.Dataset
            if im_count==0:
                
                # create im_collection_classified xarray.Dataset for first image
                im_collection_classified = xr.Dataset(
                    data_vars=dict(
                        classified=(["y", "x"], im_classified)
                    ),
                    coords=im.coords,
                    attrs=im.attrs)
                # add time dimension
                im_collection_classified.expand_dims(dim={"time": [im_collection.time.data[i]]})
                
            else:
                # create xarray.Dataset for image
                im_classified_xr = xr.Dataset(
                    data_vars=dict(
                        classified=(["y", "x"], im_classified)
                    ),
                    coords=im.coords,
                    attrs=im.attrs)
                # add time dimension
                im_classified_xr.expand_dims(dim={"time": [im_collection.time.data[i]]})
                # concatenate to classified image collection
                im_collection_classified = xr.concat([im_collection_classified, im_classified_xr], dim='time')
                
            im_count+=1
            
        # save classified image collection to netCDF file
        im_collection_classified.to_netcdf(out_path + im_collection_classified_fn, mode='w')
        print(dataset + ' classified image collection saved to file: ' + out_path + im_collection_classified_fn)
    
    return im_collection_classified, im_collection_classified_fn, fig
    
    
# --------------------------------------------------
def classify_image(im, clf, feature_cols, crop_to_AOI, AOI, ds_dict, dataset, site_name, out_path, plot_results, figures_out_path):
    '''
    Function to classify image collection using a pre-trained classifier

    Parameters
    ----------
    im: xarray.Dataset
        stack of images
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier
    crop_to_AOI: bool
        whether to mask everywhere outside the AOI before classifying
    AOI: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked if crop_to_AOI==True. AOI must be in the same CRS as the images.
    ds_dict: dict
    
    dataset: str
        the dataset name, e.g.: 'Landsat', 'PlanetScope', 'Sentinel-2'
    site_name: str
    
    out_path: str
        path to save classified images
    plot_results: bool
        whether to plot RGB and classified image
    figures_out_path: str

    Returns
    ----------
    im_classified_fn
    
    '''

    # -----Make directory for snow images (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print('Made directory for classified snow images:' + out_path)
        
    # -----Define image bands and capture date
    bands = [x for x in im.data_vars]
    bands = [band for band in bands if (band != 'QA_PIXEL')]
    im_date = str(im.time.data[0])[0:19]
    print(im_date)

    # -----Check if classified image already exists in file
    im_classified_fn = dataset + '_' + site_name + '_' + im_date.replace('-','').replace(':','') + '_classified.nc'
    if os.path.exists(out_path + im_classified_fn):
        
        print('classified image already exists in file, exiting...')
        
    else:
    
        # -----mask image pixels outside the AOI
        im_AOI = im.copy()
        im_AOI = im_AOI.isel(time=0) # remove time dimension
        if crop_to_AOI:
            # reproject AOI to im CRS if necessary
            mask = np.zeros((len(im.y.data), len(im.x.data)))
            if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
                for poly in AOI.geometry[0].geoms:
                    d = {'geometry': [Polygon(poly.exterior)]}
                    gdf = gpd.GeoDataFrame(d, crs="EPSG:"+str(im.rio.crs.to_epsg()))
                    m = rio.features.geometry_mask(gdf.geometry,
                                                   (len(im.y.data), len(im.x.data)),
                                                   im.rio.transform(),
                                                   all_touched=False,
                                                   invert=False)
                    mask[m==0] = 1
            elif AOI.geometry[0].geom_type=='Polygon':
                d = {'geometry': [Polygon(AOI.geometry[0].exterior)]}
                gdf = gpd.GeoDataFrame(d, crs="EPSG:"+str(im.rio.crs.to_epsg()))
                m = rio.features.geometry_mask(gdf.geometry,
                                               (len(im.y.data), len(im.x.data)),
                                               im.rio.transform(),
                                               all_touched=False,
                                               invert=False)
                mask[m==0] = 1
            # apply mask to bands
            for band in bands:
                im_AOI[band] = im_AOI[band].where(mask==1)
                
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
            
        # -----Plot results
        if plot_results:
            fig, ax = plt.subplots(1, 2, figsize=(10,6))
            ax = ax.flatten()
            # define x and y limits
            xmin, xmax = np.min(im.x.data)/1e3, np.max(im.x.data)/1e3
            ymin, ymax = np.min(im.y.data)/1e3, np.max(im.y.data)/1e3
            # define colors for plotting
            color_snow = '#4eb3d3'
            color_ice = '#084081'
            color_rock = '#fdbb84'
            color_water = '#bdbdbd'
            color_contour = '#f768a1'
            # create colormap
            colors = [color_snow, color_snow, color_ice, color_rock, color_water]
            cmp = matplotlib.colors.ListedColormap(colors)
            # RGB image
            # ax[0].imshow(im[ds_dict['RGB_bands'][0]], cmap='Greys', extent=(xmin, xmax, ymin, ymax))
            ax[0].imshow(np.dstack([im[ds_dict['RGB_bands'][0]].data[0], # red
                                    im[ds_dict['RGB_bands'][1]].data[0], # green
                                    im[ds_dict['RGB_bands'][2]].data[0]]), # blue
                         extent=(xmin, xmax, ymin, ymax))
            ax[0].set_xlabel("Easting [km]")
            ax[0].set_ylabel("Northing [km]")
            ax[0].set_title('RGB image')
            # classified image
            ax[1].imshow(im_classified, cmap=cmp, vmin=1, vmax=5,
                         extent=(np.min(im_AOI.x.data)/1e3, np.max(im_AOI.x.data)/1e3,
                                 np.min(im_AOI.y.data)/1e3, np.max(im_AOI.y.data)/1e3))
            # plot dummy points for legend
            ax[1].scatter(0, 0, color=color_snow, s=50, label='snow')
            ax[1].scatter(0, 0, color=color_ice, s=50, label='ice')
            ax[1].scatter(0, 0, color=color_rock, s=50, label='rock')
            ax[1].scatter(0, 0, color=color_water, s=50, label='water')
            ax[1].set_title('Classified image')
            ax[1].set_xlabel('Easting [km]')
            ax[1].legend(loc='best')
            # AOI
            if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
                for j, poly in enumerate(AOI.geometry[0].geoms):
                    # only include legend label for first geom
                    if j==0:
                        ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                    else:
                        ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                    ax[1].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
            else:
                ax[0].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                ax[1].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
            # reset x and y limits
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            ax[1].set_xlim(xmin, xmax)
            ax[1].set_ylim(ymin, ymax)
            fig.suptitle(im_date)
            fig.tight_layout()
            # save figure
            fig_date = im_date.replace('-','')
            fig_fn = (figures_out_path + dataset + '_' + site_name + '_' +
                      im_date.replace('-','').replace(':','') + '_SCA.png')
            fig.savefig(fig_fn, dpi=300, facecolor='w')
            print('figure saved to file: ' + fig_fn)

    if 'im_classified' in locals():
        # -----Save classified image to file
        im_classified_xr = xr.Dataset(data_vars = dict(classified=(['y', 'x'], im_classified)),
                                      coords = im_AOI.coords,
                                      attrs = im_AOI.attrs)
        # add time dimension
        im_classified_xr = im_classified_xr.expand_dims(dim={'time':im.time.data})
        im_classified_xr.to_netcdf(out_path + im_classified_fn)
        print('Classified image saved to file: ' + out_path + im_classified_fn)
    else:
        im_classified = 'N/A'

    return im_classified
    
    
# --------------------------------------------------
def delineate_im_collection_snowlines(im_collection, im_collection_classified, date_start, date_end, site_name, AOI, DEM, ds_dict, dataset, out_path, figures_out_path, plot_results):
    '''
    Parameters
    ----------
    im_collection:
    
    im_collection_classified:
    
    date_start:
    
    date_end:
    
    site_name:
    
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest
    DEM: xarray.DataSet
        digital elevation model used to extract elevations of the delineated snow line
        
    ds_dict:
    
    dataset:
    
    out_path:
    
    figures_out_path:
    
    plot_results:

    Returns
    ----------
    fig: matplotlib.figure
        resulting figure handle
    ax: matplotlib.axes
        resulting figure axes handles
    sl_est_split: list
        list of shapely LineStrings representing the delineated snowlines
    sl_est_elev: list
        list of floats representing the elevation at each snowline coordinate interpolated using the DEM
    '''

    # -----Make directory for snowlines (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print("Made directory for snowlines:" + out_path)
        
    # -----Check if classified snow image collection exists in directory already
    snowlines_fn = dataset + '_' + site_name + '_' + date_start.replace('-','') + '_' + date_end.replace('-','') + '_snowlines.pkl'
    if os.path.exists(os.path.join(out_path, snowlines_fn)):
    
        print('snowlines already exist in file, loading...')
        snowlines = pickle.load(open(out_path + snowlines_fn,'rb'))
        fig = 'N/A'
        
    else:
        
        # -----Initialize results data frame
        results_df = pd.DataFrame(columns=['study_site', 'datetime', 'snowlines_coords', 'snowlines_elevs', 'snowlines_elevs_median'])

        # -----Define image bands
        bands = [x for x in im_collection.data_vars]
        bands = [band for band in bands if band != 'QA_PIXEL']

        # -----Loop through image capture dates
        for i, t in enumerate(im_collection_classified.time):

            im_date = str(t.data)[0:10]
            print(im_date)

            # -----Subset image collections
            im = im_collection.sel(time=t)
            im_classified = im_collection_classified.sel(time=t)

            # -----Create no data mask
            no_data_mask = xr.where(np.isnan(im_classified), 1, 0).to_array().data[0]
            # convert to polygons
            no_data_polygons = []
            for s, value in rio.features.shapes(no_data_mask.astype(np.int16),
                                                mask=(no_data_mask > 0),
                                                transform=im.rio.transform()):
                no_data_polygons.append(shape(s))
            no_data_polygons = MultiPolygon(no_data_polygons)

            # -----Mask the DEM using the AOI
            # create AOI mask
            mask_AOI = rio.features.geometry_mask(AOI.geometry,
                                              out_shape=(len(DEM.y), len(DEM.x)),
                                              transform=DEM.rio.transform(),
                                              invert=True)
            # convert mask to xarray DataArray
            mask_AOI = xr.DataArray(mask_AOI , dims=("y", "x"))
            # mask DEM values outside the AOI
            DEM_AOI = DEM.copy(deep=True)
            DEM_AOI.elevation.data = np.where(mask_AOI==True, DEM_AOI.elevation.data, np.nan)

            # -----Interpolate DEM to the image coordinates
            DEM_AOI_interp = DEM_AOI.interp(x=im_classified.x.data,
                                            y=im_classified.y.data,
                                            method="nearest")

            # -----Determine snow covered elevations
            # create array of snow-covered pixel elevations
            snow_est_elev = np.ravel(np.where(im_classified.classified.data <= 2, DEM_AOI_interp.elevation.data, np.nan))
            # remove NaNs
            snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)]

            # -----Create elevation histograms
            # determine bins to use in histograms
            elev_min = np.fix(np.nanmin(DEM_AOI_interp.elevation.data.flatten())/10)*10
            elev_max = np.round(np.nanmax(DEM_AOI_interp.elevation.data.flatten())/10)*10
            bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max-elev_min)/10 + 1))
            bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2
            # calculate elevation histograms
            H_DEM = np.histogram(DEM_AOI_interp.elevation.data.flatten(), bins=bin_edges)[0]
            H_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
            H_snow_est_elev_norm = H_snow_est_elev / H_DEM

            # -----Make all pixels at elevations >75% snow coverage snow
            # determine elevation with > 75% snow coverage
            if len(np.where(H_snow_est_elev_norm > 0.75)[0]) > 1:
                elev_75_snow = bin_centers[np.where(H_snow_est_elev_norm > 0.75)[0][0]]
                # set all pixels above the elev_75_snow to snow (1)
                im_classified_adj = xr.where(DEM_AOI_interp.elevation > elev_75_snow, 1, im_classified) # set all values above elev_75_snow to snow (1)
                im_classified_adj = im_classified_adj.squeeze(drop=True) # drop unecessary dimensions
                H_snow_est_elev_norm[bin_centers >= elev_75_snow] = 1
            else:
                im_classified_adj = im_classified.squeeze(drop=True)

            # -----Delineate snow lines
            # create binary snow matrix
            im_binary = xr.where(im_classified_adj  > 2, 1, 0)
            # apply median filter to binary image with kernel_size of 1 pixel (~30 m)
            im_binary_filt = im_binary['classified'].data #medfilt(im_binary['classified'].data, kernel_size=1)
            # fill holes in binary image (0s within 1s = 1)
            im_binary_filt_no_holes = binary_fill_holes(im_binary_filt)
            # find contours at a constant value of 0.5 (between 0 and 1)
            contours = find_contours(im_binary_filt_no_holes, 0.5)
            # convert contour points to image coordinates
            contours_coords = []
            for contour in contours:
                ix = np.round(contour[:,1]).astype(int)
                iy = np.round(contour[:,0]).astype(int)
                coords = (im.isel(x=ix, y=iy).x.data, # image x coordinates
                          im.isel(x=ix, y=iy).y.data) # image y coordinates
                # zip points together
                xy = list(zip([x for x in coords[0]],
                              [y for y in coords[1]]))
                contours_coords = contours_coords + [xy]
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
                sl_est_elev = [DEM.sel(x=x, y=y, method='nearest').elevation.data[0]
                               for x, y in list(zip(xpts, ypts))]

            else:
                sl_est_split = None
                sl_est_elev = np.nan

            # -----Concatenate results to df
            # calculate median snow line elevation
            sl_est_elev_median = np.nanmedian(sl_est_elev)
            # compile results in df
            result_df = pd.DataFrame({'study_site': site_name,
                                      'datetime': im_date,
                                      'snowlines_coords': [sl_est],
                                      'snowlines_elevs': [sl_est_elev],
                                      'snowlines_elevs_median': sl_est_elev_median})
            # concatenate to results_df
            results_df = pd.concat([results_df, result_df])

            # -----Plot results
            if plot_results:
                contour = None
                fig, ax = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]})
                ax = ax.flatten()
                # define x and y limits
                xmin, xmax = np.min(im_classified.x.data)/1e3, np.max(im_classified.x.data)/1e3
                ymin, ymax = np.min(im_classified.y.data)/1e3, np.max(im_classified.y.data)/1e3
                # define colors for plotting
                color_snow = '#4eb3d3'
                color_ice = '#084081'
                color_rock = '#fdbb84'
                color_water = '#bdbdbd'
                color_contour = '#f768a1'
                # create colormap
                colors = [color_snow, color_snow, color_ice, color_rock, color_water]
                cmp = matplotlib.colors.ListedColormap(colors)
                # RGB image
                ax[0].imshow(np.dstack([im[ds_dict['RGB_bands'][0]].data,
                                        im[ds_dict['RGB_bands'][1]].data,
                                        im[ds_dict['RGB_bands'][2]].data]),
                             extent=(xmin, xmax, ymin, ymax))
                ax[0].set_xlabel("Easting [km]")
                ax[0].set_ylabel("Northing [km]")
                # classified image
                ax[1].imshow(im_classified['classified'].data, cmap=cmp, vmin=1, vmax=5,
                             extent=(xmin, xmax, ymin, ymax))
                # plot dummy points for legend
                ax[1].scatter(0, 0, color=color_snow, s=50, label='snow')
                ax[1].scatter(0, 0, color=color_ice, s=50, label='ice')
                ax[1].scatter(0, 0, color=color_rock, s=50, label='rock')
                ax[1].scatter(0, 0, color=color_water, s=50, label='water')
                ax[1].set_xlabel('Easting [km]')
                # AOI
                if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
                    for j, poly in enumerate(AOI.geometry[0].geoms):
                        if j==0:
                            ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                        else:
                            ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                        ax[1].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                else:
                    ax[0].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                    ax[1].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                # reset x and y limits
                ax[0].set_xlim(xmin, xmax)
                ax[0].set_ylim(ymin, ymax)
                ax[1].set_xlim(xmin, xmax)
                ax[1].set_ylim(ymin, ymax)
                # image bands histogram
                h_b = ax[2].hist(im[ds_dict['RGB_bands'][0]].data.flatten(), color='blue', histtype='step', linewidth=2, bins=100, label="blue")
                h_g = ax[2].hist(im[ds_dict['RGB_bands'][1]].data.flatten(), color='green', histtype='step', linewidth=2, bins=100, label="green")
                h_r = ax[2].hist(im[ds_dict['RGB_bands'][2]].data.flatten(), color='red', histtype='step', linewidth=2, bins=100, label="red")
                ax[2].set_xlabel("Surface reflectance")
                ax[2].set_ylabel("Pixel counts")
                ax[2].legend(loc='best')
                ax[2].grid()
                # normalized snow elevations histogram
                ax[3].bar(bin_centers, H_snow_est_elev_norm, width=(bin_centers[1]-bin_centers[0]), color=color_snow, align='center')
                ax[3].set_xlabel("Elevation [m]")
                ax[3].set_ylabel("% snow-covered")
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
                # add legends
                ax[0].legend(loc='best')
                ax[1].legend(loc='best')
                fig.suptitle(site_name + ': ' + im_date)
                fig.tight_layout()
                plt.show()
                # save figure
                fig_fn = figures_out_path + dataset + '_' + site_name + '_' + im_date + '_snowline.png'
                fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
                print('figure saved to file:' + fig_fn)

            print(' ')

        # -----Save results_df
        results_df = results_df.reset_index(drop=True)
        results_df.to_pickle(out_path + snowlines_fn)
        print('snowline data table saved to file:' + out_path + snowlines_fn)

        # -----Plot median snow line elevations
        if plot_results:
            fig2, ax2 = plt.subplots(figsize=(10,6))
            plt.rcParams.update({'font.size':12, 'font.sans-serif':'Arial'})
            # plot snowlines
            ax2.plot(results_df['datetime'].astype(np.datetime64),
                     results_df['snowlines_elevs_median'], '.b', markersize=10)
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
        
        print(' ')

    return results_df

# --------------------------------------------------
def delineate_im_snowline(im, im_classified, site_name, AOI, DEM, ds_dict, dataset, im_dt, out_path, figures_out_path, plot_results):
    '''
    Parameters
    ----------
    im_collection:
    
    im_collection_classified:
    
    date_start:
    
    date_end:
    
    site_name:
    
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest
    DEM: xarray.DataSet
        digital elevation model used to extract elevations of the delineated snow line
        
    ds_dict:
    
    dataset:
    
    out_path:
    
    figures_out_path:
    
    plot_results:

    Returns
    ----------
    fig: matplotlib.figure
        resulting figure handle
    ax: matplotlib.axes
        resulting figure axes handles
    sl_est_split: list
        list of shapely LineStrings representing the delineated snowlines
    sl_est_elev: list
        list of floats representing the elevation at each snowline coordinate interpolated using the DEM
    '''

    # -----Make directory for snowlines (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print("Made directory for snowlines:" + out_path)
                
    # -----Define image bands
    bands = [x for x in im.data_vars]
    bands = [band for band in bands if band != 'QA_PIXEL']

    # -----Remove time dimension
    im = im.isel(time=0)
    im_classified = im_classified.isel(time=0)

    # -----Create no data mask
    no_data_mask = xr.where(np.isnan(im_classified), 1, 0).to_array().data[0]
    # convert to polygons
    no_data_polygons = []
    for s, value in rio.features.shapes(no_data_mask.astype(np.int16),
                                        mask=(no_data_mask > 0),
                                        transform=im.rio.transform()):
        no_data_polygons.append(shape(s))
    no_data_polygons = MultiPolygon(no_data_polygons)

    # -----Mask the DEM using the AOI
    # create AOI mask
    mask_AOI = rio.features.geometry_mask(AOI.geometry,
                                      out_shape=(len(DEM.y), len(DEM.x)),
                                      transform=DEM.rio.transform(),
                                      invert=True)
    # convert mask to xarray DataArray
    mask_AOI = xr.DataArray(mask_AOI , dims=("y", "x"))
    # mask DEM values outside the AOI
    DEM_AOI = DEM.copy(deep=True)
    DEM_AOI['elevation'].data = np.where(mask_AOI==True, DEM_AOI['elevation'].data, np.nan)

    # -----Interpolate DEM to the image coordinates
    DEM_AOI_interp = DEM_AOI.interp(x=im_classified.x.data,
                                    y=im_classified.y.data,
                                    method="nearest")

    # -----Determine snow covered elevations
    # create array of snow-covered pixel elevations
    snow_est_elev = np.ravel(np.where(im_classified.classified.data <= 2, DEM_AOI_interp.elevation.data, np.nan))
    # remove NaNs
    snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)]

    # -----Create elevation histograms
    # determine bins to use in histograms
    elev_min = np.fix(np.nanmin(DEM_AOI_interp.elevation.data.flatten())/10)*10
    elev_max = np.round(np.nanmax(DEM_AOI_interp.elevation.data.flatten())/10)*10
    bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max-elev_min)/10 + 1))
    bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2
    # calculate elevation histograms
    H_DEM = np.histogram(DEM_AOI_interp.elevation.data.flatten(), bins=bin_edges)[0]
    H_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
    H_snow_est_elev_norm = H_snow_est_elev / H_DEM

    # -----Make all pixels at elevations >75% snow coverage snow
    # determine elevation with > 75% snow coverage
    if len(np.where(H_snow_est_elev_norm > 0.75)[0]) > 1:
        elev_75_snow = bin_centers[np.where(H_snow_est_elev_norm > 0.75)[0][0]]
        # set all pixels above the elev_75_snow to snow (1)
        im_classified_adj = xr.where(DEM_AOI_interp['elevation'].isel(band=0) > elev_75_snow, 1, im_classified) # set all values above elev_75_snow to snow (1)
        im_classified_adj = im_classified_adj.squeeze(drop=True) # drop unecessary dimensions
        H_snow_est_elev_norm[bin_centers >= elev_75_snow] = 1
    else:
        im_classified_adj = im_classified.squeeze(drop=True)

    # -----Delineate snow lines
    # create binary snow matrix
    im_binary = xr.where(im_classified_adj  > 2, 1, 0)
    # apply median filter to binary image with kernel_size of 1 pixel (~30 m)
    im_binary_filt = im_binary['classified'].data #medfilt(im_binary['classified'].data, kernel_size=1)
    # fill holes in binary image (0s within 1s = 1)
    im_binary_filt_no_holes = binary_fill_holes(im_binary_filt)
    # find contours at a constant value of 0.5 (between 0 and 1)
    contours = find_contours(im_binary_filt_no_holes, 0.5)
    # convert contour points to image coordinates
    contours_coords = []
    for contour in contours:
        ix = np.round(contour[:,1]).astype(int)
        iy = np.round(contour[:,0]).astype(int)
        coords = (im.isel(x=ix, y=iy).x.data, # image x coordinates
                  im.isel(x=ix, y=iy).y.data) # image y coordinates
        # zip points together
        xy = list(zip([x for x in coords[0]],
                      [y for y in coords[1]]))
        contours_coords = contours_coords + [xy]
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
        sl_est_elev = [DEM.sel(x=x, y=y, method='nearest').elevation.data[0]
                       for x, y in list(zip(xpts, ypts))]

    else:
        sl_est_split = None
        sl_est_elev = np.nan

    # -----Compile results in dataframe
    # calculate median snow line elevation
    sl_est_elev_median = np.nanmedian(sl_est_elev)
    # compile results in df
    result_df = pd.DataFrame({'study_site': site_name,
                              'datetime': im_dt,
                              'snowlines_coords': [sl_est],
                              'snowlines_elevs': [sl_est_elev],
                              'snowlines_elevs_median': sl_est_elev_median})

    # -----Plot results
    if plot_results:
        contour = None
        fig, ax = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'height_ratios': [3, 1]})
        ax = ax.flatten()
        # define x and y limits
        xmin, xmax = np.min(im_classified.x.data)/1e3, np.max(im_classified.x.data)/1e3
        ymin, ymax = np.min(im_classified.y.data)/1e3, np.max(im_classified.y.data)/1e3
        # define colors for plotting
        color_snow = '#4eb3d3'
        color_ice = '#084081'
        color_rock = '#fdbb84'
        color_water = '#bdbdbd'
        color_contour = '#f768a1'
        # create colormap
        colors = [color_snow, color_snow, color_ice, color_rock, color_water]
        cmp = matplotlib.colors.ListedColormap(colors)
        # RGB image
        ax[0].imshow(np.dstack([im[ds_dict['RGB_bands'][0]].data,
                                im[ds_dict['RGB_bands'][1]].data,
                                im[ds_dict['RGB_bands'][2]].data]),
                     extent=(xmin, xmax, ymin, ymax))
        ax[0].set_xlabel("Easting [km]")
        ax[0].set_ylabel("Northing [km]")
        # classified image
        ax[1].imshow(im_classified['classified'].data, cmap=cmp, vmin=1, vmax=5,
                     extent=(xmin, xmax, ymin, ymax))
        # plot dummy points for legend
        ax[1].scatter(0, 0, color=color_snow, s=50, label='snow')
        ax[1].scatter(0, 0, color=color_ice, s=50, label='ice')
        ax[1].scatter(0, 0, color=color_rock, s=50, label='rock')
        ax[1].scatter(0, 0, color=color_water, s=50, label='water')
        ax[1].set_xlabel('Easting [km]')
        # AOI
        if AOI.geometry[0].geom_type=='MultiPolygon': # loop through geoms if AOI = MultiPolygon
            for j, poly in enumerate(AOI.geometry[0].geoms):
                if j==0:
                    ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
                else:
                    ax[0].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
                ax[1].plot([x/1e3 for x in poly.exterior.coords.xy[0]], [y/1e3 for y in poly.exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
        else:
            ax[0].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='AOI')
            ax[1].plot([x/1e3 for x in AOI.geometry[0].exterior.coords.xy[0]], [y/1e3 for y in AOI.geometry[0].exterior.coords.xy[1]], '-k', linewidth=1, label='_nolegend_')
        # reset x and y limits
        ax[0].set_xlim(xmin, xmax)
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        # image bands histogram
        h_b = ax[2].hist(im[ds_dict['RGB_bands'][0]].data.flatten(), color='blue', histtype='step', linewidth=2, bins=100, label="blue")
        h_g = ax[2].hist(im[ds_dict['RGB_bands'][1]].data.flatten(), color='green', histtype='step', linewidth=2, bins=100, label="green")
        h_r = ax[2].hist(im[ds_dict['RGB_bands'][2]].data.flatten(), color='red', histtype='step', linewidth=2, bins=100, label="red")
        ax[2].set_xlabel("Surface reflectance")
        ax[2].set_ylabel("Pixel counts")
        ax[2].legend(loc='best')
        ax[2].grid()
        # normalized snow elevations histogram
        ax[3].bar(bin_centers, H_snow_est_elev_norm, width=(bin_centers[1]-bin_centers[0]), color=color_snow, align='center')
        ax[3].set_xlabel("Elevation [m]")
        ax[3].set_ylabel("% snow-covered")
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
        # add legends
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.suptitle(site_name + ': ' + im_dt)
        fig.tight_layout()
        plt.show()
        # save figure
        fig_fn = figures_out_path + dataset + '_' + site_name + '_' + im_dt + '_snowline.png'
        fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
        print('figure saved to file:' + fig_fn)

    return result_df


# --------------------------------------------------
def reduce_memory_usage(df, verbose=True):
# from Bex T (2021): https://towardsdatascience.com/6-pandas-mistakes-that-silently-tell-you-are-a-rookie-b566a252e60d
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
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lon, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


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
    

# --------------------------------------------------
#def delineate_snow_line(im_fn, im_path, im_classified_fn, im_classified_path, AOI, DEM, DEM_rio):
#    '''
#    Parameters
#    ----------
#    im_fn: str
#        file name of the input image
#    im_path: str
#        path in directory to the input image
#    im_classified_fn: str
#        file name of the classified image
#    im_classified_path: str
#        path in directory to the classified image
#    AOI: geopandas.GeoDataFrame
#        area of interest
#    DEM: xarray.DataSet
#        digital elevation model used to extract elevations of the delineated snow line
#
#    Returns
#    ----------
#    fig: matplotlib.figure
#        resulting figure handle
#    ax: matplotlib.axes
#        resulting figure axes handles
#    sl_est_split: list
#        list of shapely LineStrings representing the delineated snowlines
#    sl_est_elev: list
#        list of floats representing the elevation at each snowline coordinate interpolated using the DEM
#    '''
#
#    # -----Open images
#    # image
#    im = rxr.open_rasterio(os.path.join(im_path, im_fn)) # open image as xarray.DataArray
#    im = im.where(im!=-9999) # remove no data values
#    if np.nanmean(im) > 1e3:
#        im = im / 1e4 # account for surface reflectance scalar multiplier
#    date = im_fn[0:8] # grab image capture date from file name
#    # classified image
#    im_classified = rxr.open_rasterio(im_classified_path + im_classified_fn) # open image as xarray.DataArray
#    # create no data mask
#    no_data_mask = xr.where(im_classified==-9999, 1, 0).data[0]
#    # convert to polygons
#    no_data_polygons = []
#    for s, value in rio.features.shapes(no_data_mask.astype(np.int16),
#                                        mask=(no_data_mask >0),
#                                        transform=rio.open(im_path + im_fn).transform):
#        no_data_polygons.append(shape(s))
#    no_data_polygons = MultiPolygon(no_data_polygons)
#    # mask no data points in classified image
#    im_classified = im_classified.where(im_classified!=-9999) # now, remove no data values
#
#    # -----Mask the DEM using the AOI
#    # create AOI mask
#    mask_AOI = rio.features.geometry_mask(AOI.geometry,
#                                      out_shape=(len(DEM.y), len(DEM.x)),
#                                      transform=DEM_rio.transform,
#                                      invert=True)
#    # convert mask to xarray DataArray
#    mask_AOI = xr.DataArray(mask_AOI , dims=("y", "x"))
#    # mask DEM values outside the AOI
#    DEM_AOI = DEM.where(mask_AOI == True)
#
#    # -----Interpolate DEM to the image coordinates
#    im_classified = im_classified.squeeze(drop=True) # remove unecessary dimensions
#    x, y = im_classified.indexes.values() # grab indices of image
#    DEM_AOI_interp = DEM_AOI.interp(x=x, y=y, method="nearest") # interpolate DEM to image coordinates
#
#    # -----Determine snow covered elevations
#    # mask pixels not classified as snow
#    DEM_AOI_interp_snow = DEM_AOI_interp.where(im_classified<=2)
#    # create array of snow-covered pixel elevations
#    snow_est_elev = DEM_AOI_interp_snow.elevation.data.flatten()
#
#    # -----Create elevation histograms
#    # determine bins to use in histograms
#    elev_min = np.fix(np.nanmin(DEM_AOI_interp.elevation.data.flatten())/10)*10
#    elev_max = np.round(np.nanmax(DEM_AOI_interp.elevation.data.flatten())/10)*10
#    bin_edges = np.linspace(elev_min, elev_max, num=int((elev_max-elev_min)/10 + 1))
#    bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2
#    # calculate elevation histograms
#    H_DEM = np.histogram(DEM_AOI_interp.elevation.data.flatten(), bins=bin_edges)[0]
#    H_snow_est_elev = np.histogram(snow_est_elev, bins=bin_edges)[0]
#    H_snow_est_elev_norm = H_snow_est_elev / H_DEM
#
#    # -----Make all pixels at elevations >75% snow coverage snow
#    # determine elevation with > 75% snow coverage
#    if len(np.where(H_snow_est_elev_norm > 0.75)) > 1:
#        elev_75_snow = bin_centers[np.where(H_snow_est_elev_norm > 0.75)[0][0]]
#        # set all pixels above the elev_75_snow to snow (1)
#        im_classified_adj = xr.where(DEM_AOI_interp.elevation > elev_75_snow, 1, im_classified) # set all values above elev_75_snow to snow (1)
#        im_classified_adj = im_classified_adj.squeeze(drop=True) # drop unecessary dimensions
#    else:
#        im_classified_adj = im_classified.squeeze(drop=True)
#
#    # -----Determine snow line
#    # generate and filter binary snow matrix
#    # create binary snow matrix
#    im_binary = xr.where(im_classified_adj  > 2, 1, 0).data
#    # apply median filter to binary image with kernel_size of 33 pixels (~99 m)
#    im_binary_filt = medfilt(im_binary, kernel_size=33)
#    # fill holes in binary image (0s within 1s = 1)
#    im_binary_filt_no_holes = binary_fill_holes(im_binary_filt)
#    # find contours at a constant value of 0.5 (between 0 and 1)
#    contours = find_contours(im_binary_filt_no_holes, 0.5)
#    # convert contour points to image coordinates
#    contours_coords = []
#    for contour in contours:
#        ix = np.round(contour[:,1]).astype(int)
#        iy = np.round(contour[:,0]).astype(int)
#        coords = (im.isel(x=ix, y=iy).x.data, # image x coordinates
#                  im.isel(x=ix, y=iy).y.data) # image y coordinates
#        # zip points together
#        xy = list(zip([x for x in coords[0]],
#                      [y for y in coords[1]]))
#        contours_coords = contours_coords + [xy]
#    # create snow-free polygons
#    c_polys = []
#    for c in contours_coords:
#        c_points = [Point(x,y) for x,y in c]
#        c_poly = Polygon([[p.x, p.y] for p in c_points])
#        c_polys = c_polys + [c_poly]
#    # only save the largest polygon
#    if len(c_polys) > 1:
#        # calculate polygon areas
#        areas = np.array([poly.area for poly in c_polys])
#        # grab top 3 areas with their polygon indices
#        areas_max = sorted(zip(areas, np.arange(0,len(c_polys))), reverse=True)[:1]
#        # grab indices
#        ic_polys = [x[1] for x in areas_max]
#        # grab polygons at indices
#        c_polys = [c_polys[i] for i in ic_polys]
#    # extract coordinates in polygon
#    polys_coords = [list(zip(c.exterior.coords.xy[0], c.exterior.coords.xy[1]))  for c in c_polys]
#
#    # extract snow lines (sl) from contours
#    # filter contours using no data and AOI masks (i.e., along glacier outline or data gaps)
#    sl_est = [] # initialize list of snow lines
#    min_sl_length = 100 # minimum snow line length
#    for c in polys_coords:
#        # create array of points
#        c_points =  [Point(x,y) for x,y in c]
#        # loop through points
#        line_points = [] # initialize list of points to use in snow line
#        for point in c_points:
#            # calculate distance from the point to the no data polygons and the AOI boundary
#            distance_no_data = no_data_polygons.distance(point)
#            distance_AOI = AOI.boundary[0].distance(point)
#            # only include points 100 m from both
#            if (distance_no_data >= 100) and (distance_AOI >=100):
#                line_points = line_points + [point]
#        if line_points: # if list of line points is not empty
#            if len(line_points) > 1: # must have at least two points to create a LineString
#                line = LineString([(p.xy[0][0], p.xy[1][0]) for p in line_points])
#                if line.length > min_sl_length:
#                    sl_est = sl_est + [line]
#
#    # -----Check if any snow lines were found
#    if sl_est:
#        # split lines with points more than 100 m apart and filter by length
#        sl_est = sl_est[0]
#        max_dist = 100 # m
#        first_point = Point(sl_est.coords.xy[0][0], sl_est.coords.xy[1][0])
#        points = [Point(sl_est.coords.xy[0][i], sl_est.coords.xy[1][i])
#                  for i in np.arange(0,len(sl_est.coords.xy[0]))]
#        isplit = [0] # point indices where to split the line
#        for i, p in enumerate(points):
#            if i!=0:
#                dist = p.distance(points[i-1])
#                if dist > max_dist:
#                    isplit.append(i)
#        isplit.append(len(points)) # add ending point to complete the last line
#        sl_est_split = [] # initialize split lines
#        # loop through split indices
#        if len(isplit) > 1:
#            for i, p in enumerate(isplit[:-1]):
#                if isplit[i+1]-isplit[i] > 1: # must have at least two points to make a line
#                    line = LineString(points[isplit[i]:isplit[i+1]])
#                    if line.length > min_sl_length:
#                        sl_est_split = sl_est_split + [line]
#        else:
#            sl_est_split = [sl_est]
#
#        # interpolate elevations at snow line coordinates
#        # compile all line coordinates into arrays of x- and y-coordinates
#        xpts, ypts = [], []
#        for line in sl_est_split:
#            xpts = xpts + [x for x in line.coords.xy[0]]
#            ypts = ypts + [y for y in line.coords.xy[1]]
#        xpts, ypts = np.array(xpts).flatten(), np.array(ypts).flatten()
#        # interpolate elevation at snow line points
#        sl_est_elev = [DEM.sel(x=x, y=y, method='nearest').elevation.data[0]
#                       for x, y in list(zip(xpts, ypts))]
#
#    else:
#        sl_est_split = None
#        sl_est_elev = np.nan
#
#    # -----Plot results
#    contour = None
#    fig, ax, sl_points_AOI = plot_im_classified_histogram_contour(im, im_classified_adj, DEM, DEM_rio, AOI, contour)
#    # plot estimated snow line coordinates
#    if sl_est_split!=None:
#        for i, line  in enumerate(sl_est_split):
#            if i==0:
#                ax[0].plot([x/1e3 for x in line.coords.xy[0]],
#                           [y/1e3 for y in line.coords.xy[1]],
#                           '-', color='#f768a1', label='sl$_{estimated}$')
#            else:
#                ax[0].plot([x/1e3 for x in line.coords.xy[0]],
#                           [y/1e3 for y in line.coords.xy[1]],
#                           '-', color='#f768a1', label='_nolegend_')
#            ax[1].plot([x/1e3 for x in line.coords.xy[0]],
#                       [y/1e3 for y in line.coords.xy[1]],
#                       '-', color='#f768a1', label='_nolegend_')
#    # add legends
#    ax[0].legend(loc='best')
#    ax[1].legend(loc='best')
#    if contour is not None:
#        ax[3].set_title('Contour = ' + str(np.round(contour,1)) + ' m')
#    fig.suptitle(date)
#
#    return fig, ax, sl_est_split, sl_est_elev
    
    
# --------------------------------------------------
#def determine_snow_elevs(DEM, im, im_classified_fn, im_classified_path, im_dt, AOI, plot_output):
#    '''Determine elevations of snow-covered pixels in the classified image.
#    Parameters
#    ----------
#    DEM: xarray.Dataset
#        digital elevation model
#    im: xarray.DataArray
#        input image used to classify snow
#    im_classified_fn: str
#        classified image file name
#    im_classified_path: str
#        path in directory to classified image
#    im_dt: numpy.datetime64
#        datetime of the image capture
#    AOI: geopandas.GeoDataFrame
#        area of interest
#    plot_output: bool
#        whether to plot the output RGB and snow classified image with histograms for surface reflectances of each band and the elevations of snow-covered pixels
#
#    Returns
#    ----------
#    snow_elev: numpy array
#        elevations at each snow-covered pixel
#    '''
#
#    # -----Set up original image
#    # account for image scalar multiplier if necessary
#    im_scalar = 10000
#    if np.nanmean(im.data[0])>1e3:
#        im = im / im_scalar
#    # replace no data values with NaN
#    im = im.where(im!=-9999)
#    # drop uneccesary dimensions
#    im = im.squeeze(drop=True)
#    # extract bands info
#    b = im.data[0].astype(float)
#    g = im.data[1].astype(float)
#    r = im.data[2].astype(float)
#    nir = im.data[3].astype(float)
#
#    # -----Load classified image
#    im_classified = rxr.open_rasterio(os.path.join(im_classified_path, im_classified_fn))
#    # replace no data values with NaN
#    im_classified = im_classified.where(im_classified!=-9999)
#    # drop uneccesary dimensions
#    im_classified = im_classified.squeeze(drop=True)
#
#    # -----Interpolate DEM to image points
#    x, y = im_classified.indexes.values() # grab indices of image
#    DEM_interp = DEM.interp(x=x, y=y, method="nearest") # interpolate DEM to image coordinates
#    DEM_interp_masked = DEM_interp.where(im_classified<=2) # mask image where not classified as snow
#    snow_elev = DEM_interp_masked.elevation.data.flatten() # create array of snow elevations
#    snow_elev = np.sort(snow_elev[~np.isnan(snow_elev)]) # sort and remove NaNs
#
#    # minimum elevation of the image where data exist
#    im_elev_min = np.nanmin(DEM_interp.elevation.data.flatten())
#    im_elev_max = np.nanmax(DEM_interp.elevation.data.flatten())
#
#    # plot snow elevations histogram
#    if plot_output:
#        fig, ax, sl_points_AOI = plot_im_classified_histogram_contour(im, im_classified, DEM, AOI, None)
#        return im_elev_min, im_elev_max, snow_elev, fig
#
#    return im_elev_min, im_elev_max, snow_elev
    
    
# --------------------------------------------------
#def calculate_SCA(im, im_classified):
#    '''Function to calculated total snow-covered area (SCA) from using an input image and a snow binary mask of the same resolution and grid.
#    Parameters
#    ----------
#        im: rasterio object
#            input image
#        im_classified: numpy array
#            classified image array with the same shape as the input image bands. Classes: snow = 1, shadowed snow = 2, ice = 3, rock/debris = 4.
#    Returns
#    ----------
#        SCA: float
#            snow-covered area in classified image [m^2]'''
#
#    pA = im.res[0]*im.res[1] # pixel area [m^2]
#    snow_count = np.count_nonzero(im_classified <= 2) # number of snow and shadowed snow pixels
#    SCA = pA * snow_count # area of snow [m^2]
#
#    return SCA
    
    
# --------------------------------------------------
#def crop_images_to_AOI(im_path, im_fns, AOI):
#    '''
#    Crop images to AOI.
#
#    Parameters
#    ----------
#    im_path: str
#        path in directory to input images
#    im_fns: str array
#        file names of images to crop
#    AOI: geopandas.geodataframe.GeoDataFrame
#        cropping region - everything outside the AOI will be masked. Only the exterior bounds used for cropping (no holes). AOI must be in the same CRS as the images.
#
#    Returns
#    ----------
#    cropped_im_path: str
#        path in directory to cropped images
#    '''
#
#    # make folder for cropped images if it does not exist
#    cropped_im_path = os.path.join(im_path, "../cropped/")
#    if os.path.isdir(cropped_im_path)==0:
#        os.mkdir(cropped_im_path)
#        print(cropped_im_path+" directory made")
#
#    # loop through images
#    for im_fn in im_fns:
#
#        # open image
#        im = rio.open(os.path.join(im_path, im_fn))
#
#        # check if file exists in directory already
#        cropped_im_fn = os.path.join(cropped_im_path, im_fn[0:15] + "_crop.tif")
#        if os.path.exists(cropped_im_fn)==True:
#            print("cropped image already exists in directory, continuing...")
#        else:
#            # mask image pixels outside the AOI exterior
##            AOI_bb = [AOI.bounds]
#            out_image, out_transform = mask(im, AOI.buffer(100), crop=True)
#            out_meta = im.meta.copy()
#            out_meta.update({"driver": "GTiff",
#                         "height": out_image.shape[1],
#                         "width": out_image.shape[2],
#                         "transform": out_transform})
#            with rio.open(cropped_im_fn, "w", **out_meta) as dest:
#                dest.write(out_image)
#            print(cropped_im_fn + " saved")
#
#    return cropped_im_path
    
# --------------------------------------------------
#def into_range(x, range_min, range_max):
#    shiftedx = x - range_min
#    delta = range_max - range_min
#    return (((shiftedx % delta) + delta) % delta) + range_min


# --------------------------------------------------
#def sunpos(when, location, refraction):
#    '''
#    Determine the sun azimuth and elevation using the date and location.
#    Modified from: https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777
#    Parameters
#    ----------
#    when: str array
#        date of image capture ('YYYY', 'MM', 'DD', 'hh', 'mm', 'ss')
#    location = coordinate pair (floats)
#        approximate location of image capture (latitude, longitude)
#    refraction: bool
#        whether to account for refraction (bool)
#
#    Returns
#    ----------
#    azimuth: float
#        sun azimuth in degrees
#    elevation: float
#        sun elevation in degrees (float)
#    '''
#
#    # Extract the passed data
#    year, month, day, hour, minute, second = when
#    latitude, longitude = location
#    # Math typing shortcuts
#    rad, deg = math.radians, math.degrees
#    sin, cos, tan = math.sin, math.cos, math.tan
#    asin, atan2 = math.asin, math.atan2
#    # Convert latitude and longitude to radians
#    rlat = rad(latitude)
#    rlon = rad(longitude)
#    # Decimal hour of the day at Greenwich
#    greenwichtime = hour + minute / 60 + second / 3600
#    # Days from J2000, accurate from 1901 to 2099
#    daynum = (
#        367 * year
#        - 7 * (year + (month + 9) // 12) // 4
#        + 275 * month // 9
#        + day
#        - 730531.5
#        + greenwichtime / 24
#    )
#    # Mean longitude of the sun
#    mean_long = daynum * 0.01720279239 + 4.894967873
#    # Mean anomaly of the Sun
#    mean_anom = daynum * 0.01720197034 + 6.240040768
#    # Ecliptic longitude of the sun
#    eclip_long = (
#        mean_long
#        + 0.03342305518 * sin(mean_anom)
#        + 0.0003490658504 * sin(2 * mean_anom)
#    )
#    # Obliquity of the ecliptic
#    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
#    # Right ascension of the sun
#    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
#    # Declination of the sun
#    decl = asin(sin(obliquity) * sin(eclip_long))
#    # Local sidereal time
#    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
#    # Hour angle of the sun
#    hour_ang = sidereal - rasc
#    # Local elevation of the sun
#    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
#    # Local azimuth of the sun
#    azimuth = atan2(
#        -cos(decl) * cos(rlat) * sin(hour_ang),
#        sin(decl) - sin(rlat) * sin(elevation),
#    )
#    # Convert azimuth and elevation to degrees
#    azimuth = into_range(deg(azimuth), 0, 360)
#    elevation = into_range(deg(elevation), -180, 180)
#    # Refraction correction (optional)
#    if refraction:
#        targ = rad((elevation + (10.3 / (elevation + 5.11))))
#        elevation += (1.02 / tan(targ)) / 60
#
#    # Return azimuth and elevation in degrees
#    return (round(azimuth, 2), round(elevation, 2))

    
# --------------------------------------------------
#def apply_hillshade_correction(crs, polygon, im_fn, im_path, DEM, out_path, skip_clipped, plot_results):
#    '''
#    Adjust image using by generating a hillshade model and minimizing the standard deviation of each band within the defined SCA
#
#    Parameters
#    ----------
#    crs: float
#        Coordinate Reference System (EPSG code)
#    polygon:  shapely.geometry.polygon.Polygon
#            polygon, where the band standard deviation will be minimized
#    im: rasterio object
#        input image
#    im_name: str
#        file name name of the input image
#    im_path: str
#        path in directory to the input image
#    DEM_path: str
#        path in directory to the DEM used to generate the hillshade model
#    hs_path: str
#        path to save hillshade model
#    out_path: str
#        path to save corrected image file
#    skip_clipped: bool
#        whether to skip images where bands appear "clipped"
#    plot_results: bool
#        whether to plot results to a matplotlib.pyplot.figure
#
#    Returns
#    ----------
#    im_corrected_name: str
#        file name of the hillshade-corrected image saved to file
#    '''
#
#    print('HILLSHADE CORRECTION')
#
#    # -----Load image
#    im = rxr.open_rasterio(im_path + im_fn)
#    # replace no data values with NaN
#    im = im.where(im!=-9999)
#    # account for image scalar multiplier
#    if (np.nanmean(im.data[0])>1e3):
#        im_scalar = 1e4
#        im = im / im_scalar
#
#    # -----Read image bands
#    b = im.isel(band=0)
#    g = im.isel(band=1)
#    r = im.isel(band=2)
#    nir = im.isel(band=3)
#
#    # -----Return if image bands are likely clipped
#    if skip_clipped==True:
#        if (np.nanmax(b) < 0.8) or (np.nanmax(g) < 0.8) or (np.nanmax(r) < 0.8):
#            print('image bands appear clipped... skipping.')
#            im_corrected_name = 'N/A'
#            return im_corrected_name
#
#    # -----Filter image points outside the SCA
#    # create a mask using the polygon geometry
#    mask = rio.features.geometry_mask([polygon],
#                                       np.shape(b),
#                                       im.rio.transform,
#                                       all_touched=False,
#                                       invert=False)
#    b_polygon = b[mask==0]
#    g_polygon = g[mask==0]
#    r_polygon = r[mask==0]
#    nir_polygon = nir[mask==0]
#
#    # -----Return if image does not contain real values within the SCA
#    if len(~np.isnan(b))<1:
#        print('image does not contain real values within the SCA... skipping.')
#        im_corrected_name = 'N/A'
#        return im_corrected_name
#
#    # -----Extract image information for sun position calculation
#    # location: grab center image coordinate, convert to lat lon
#    xmid = ((im.x.data[-1] - im.x.data[0])/2 + im.x.data[0])
#    ymid = ((im.y.data[-1] - im.y.data[0])/2 + im.y.data[0])
#    transformer = Transformer.from_crs("epsg:"+str(crs), "epsg:4326")
#    location = transformer.transform(xmid, ymid)
#    # when: year, month, day, hour, minute, second
#    when = (float(im_fn[0:4]), float(im_fn[4:6]), float(im_fn[6:8]),
#            float(im_fn[9:11]), float(im_fn[11:13]), float(im_fn[13:15]))
#    # sun azimuth and elevation
#    azimuth, elevation = sunpos(when, location, refraction=1)
#
#    # -----Make directory for hillshade models (if it does not already exist in file)
#    if os.path.exists(out_path)==False:
#        os.mkdir(out_path)
#        print('made directory for hillshade correction outputs: ' + out_path)
#
#    # -----Create hillshade model (if it does not already exist in file)
#    hs_fn = str(azimuth) + '-az_' + str(elevation) + '-z_hillshade.tif'
#    if os.path.exists(out_path + hs_fn):
#        print('hillshade model already exists in directory, loading...')
#    else:
##                print('creating hillshade model...')
#        # construct the gdal_merge command
#        # modified from: https://github.com/clhenrick/gdal_hillshade_tutorial
#        # gdaldem hillshade -az aximuth -z elevation dem.tif hillshade.tif
#        cmd = 'gdaldem hillshade -az ' + str(azimuth) + ' -z ' + str(elevation)+' ' + str(DEM_path) + ' ' + hs_path + hs_fn
#        # run the command
#        p = subprocess.run(cmd, shell=True, capture_output=True)
#        print(p)
#
#    # -----load hillshade model from file
#    hs = rxr.open_rasterio(hs_path, hs_fn)
##            print('hillshade model loaded from file...')
#
#    # -----Resample hillshade to image coordinates
#    # resampled hillshade file name
#    hs_resamp_fn = str(azimuth) + '-az_' + str(elevation) + '-z_hillshade_resamp.tif'
#    band, x, y = im.indexes.values() # grab indices of image
#    DEM_AOI_interp = DEM_AOI.interp(x=x, y=y, method="nearest") # interpolate DEM
#    # save to file
#    with rio.open(out_path + hs_resamp_fn,'w',
#                  driver='GTiff',
#                  height=hs_resamp.shape[0],
#                  width=hs_resamp.shape[1],
#                  dtype=hs_resamp.dtype,
#                  count=1,
#                  crs=im.crs,
#                  transform=im.transform) as dst:
#        dst.write(hs_resamp, 1)
#    print('resampled hillshade model saved to file:' + out_path + hs_resamp_fn)
#
#    # -----load resampled hillshade model
#    hs_resamp = rxr.open_rasterio(hs_resamp_fn).squeeze()
#    print('resampled hillshade model loaded from file')
#    # -----filter hillshade model points outside the SCA
#    hs_polygon = hs_resamp.data[0][mask==0]
#
#    # -----normalize hillshade model
#    hs_norm = (hs_resamp - np.min(hs_resamp)) / (np.max(hs_resamp) - np.min(hs_resamp))
#    hs_polygon_norm = (hs_polygon - np.min(hs_polygon)) / (np.max(hs_polygon) - np.min(hs_polygon))
#
#    # -----loop through hillshade scalar multipliers
##            print('solving for optimal band scalars...')
#    # define scalars to test
#    hs_scalars = np.linspace(0,0.5,num=21)
#    # blue
#    b_polygon_mu = np.zeros(len(hs_scalars)) # mean
#    b_polygon_sigma =np.zeros(len(hs_scalars)) # std
#    # green
#    g_polygon_mu = np.zeros(len(hs_scalars)) # mean
#    g_polygon_sigma = np.zeros(len(hs_scalars)) # std
#    # red
#    r_polygon_mu = np.zeros(len(hs_scalars)) # mean
#    r_polygon_sigma = np.zeros(len(hs_scalars)) # std
#    # nir
#    nir_polygon_mu = np.zeros(len(hs_scalars)) # mean
#    nir_polygon_sigma = np.zeros(len(hs_scalars)) # std
#    i=0 # loop counter
#    for hs_scalar in hs_scalars:
#        # full image
#        b_adj = b - (hs_norm * hs_scalar)
#        g_adj = g - (hs_norm * hs_scalar)
#        r_adj = r - (hs_norm * hs_scalar)
#        nir_adj = nir - (hs_norm * hs_scalar)
#        # SCA
#        b_polygon_mu[i] = np.nanmean(b_polygon- (hs_polygon_norm * hs_scalar))
#        b_polygon_sigma[i] = np.nanstd(b_polygon- (hs_polygon_norm * hs_scalar))
#        g_polygon_mu[i] = np.nanmean(g_polygon- (hs_polygon_norm * hs_scalar))
#        g_polygon_sigma[i] = np.nanstd(g_polygon- (hs_polygon_norm * hs_scalar))
#        r_polygon_mu[i] = np.nanmean(r_polygon- (hs_polygon_norm * hs_scalar))
#        r_polygon_sigma[i] = np.nanstd(r_polygon- (hs_polygon_norm * hs_scalar))
#        nir_polygon_mu[i] = np.nanmean(nir_polygon- (hs_polygon_norm * hs_scalar))
#        nir_polygon_sigma[i] = np.nanstd(nir_polygon- (hs_polygon_norm * hs_scalar))
#        i+=1 # increase loop counter
#
#    # -----Determine optimal scalar for each image band
#    Ib = np.where(b_polygon_sigma==np.min(b_polygon_sigma))[0][0]
#    b_scalar = hs_scalars[Ib]
#    Ig = np.where(g_polygon_sigma==np.min(g_polygon_sigma))[0][0]
#    g_scalar = hs_scalars[Ig]
#    Ir = np.where(r_polygon_sigma==np.min(r_polygon_sigma))[0][0]
#    r_scalar = hs_scalars[Ir]
#    Inir = np.where(nir_polygon_sigma==np.min(nir_polygon_sigma))[0][0]
#    nir_scalar = hs_scalars[Inir]
#    print('Optimal scalars:  Blue   |   Green   |   Red   |   NIR')
#    print(b_scalar, g_scalar, r_scalar, nir_scalar)
#
#    # -----Apply optimal hillshade model correction
#    b_corrected = b - (hs_norm * hs_scalars[Ib])
#    g_corrected = g - (hs_norm * hs_scalars[Ig])
#    r_corrected = r - (hs_norm * hs_scalars[Ir])
#    nir_corrected = nir - (hs_norm * hs_scalars[Inir])
#
#    # -----Replace previously 0 values with 0 to signify no-data
#    b_corrected[b==0] = 0
#    g_corrected[g==0] = 0
#    r_corrected[r==0] = 0
#    nir_corrected[nir==0] = 0
#
#    # -----Plot original and corrected images and band histograms
#    if plot_results==True:
#        fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,12), gridspec_kw={'height_ratios': [3, 1]})
#        plt.rcParams.update({'font.size': 14, 'font.serif': 'Arial'})
#        # original image
#        ax1.imshow(np.dstack([r, g, b]),
#                   extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
#        ax1.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
#        ax1.set_xlabel('Northing [km]')
#        ax1.set_ylabel('Easting [km]')
#        ax1.set_title('Original image')
#        # corrected image
#        ax2.imshow(np.dstack([r_corrected, g_corrected, b_corrected]),
#                   extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
#        ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
#        ax2.set_xlabel('Northing [km]')
#        ax2.set_title('Corrected image')
#        # band histograms
#        ax3.hist(nir[nir>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
#        ax3.hist(b[b>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
#        ax3.hist(g[g>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
#        ax3.hist(r[r>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
#        ax3.set_xlabel('Surface reflectance')
#        ax3.set_ylabel('Pixel counts')
#        ax3.grid()
#        ax3.legend()
#        ax4.hist(nir_corrected[nir_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
#        ax4.hist(b_corrected[b_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
#        ax4.hist(g_corrected[g_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
#        ax4.hist(r_corrected[r_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
#        ax4.set_xlabel('Surface reflectance')
#        ax4.grid()
#        fig1.tight_layout()
#        plt.show()
#
#    # -----save hillshade-corrected image to file
#    # create output directory (if it does not already exist in file)
#    if os.path.exists(out_path)==False:
#        os.mkdir(out_path)
#        print('created output directory:',out_path)
#    # file name
#    im_corrected_name = im_name[0:-4]+'_hs-corrected.tif'
#    # metadata
#    out_meta = im.meta.copy()
#    out_meta.update({'driver':'GTiff',
#                     'width':b_corrected.shape[1],
#                     'height':b_corrected.shape[0],
#                     'count':4,
#                     'dtype':'float64',
#                     'crs':im.crs,
#                     'transform':im.transform})
#    # write to file
#    with rio.open(out_path+im_corrected_name, mode='w',**out_meta) as dst:
#        dst.write_band(1,b_corrected)
#        dst.write_band(2,g_corrected)
#        dst.write_band(3,r_corrected)
#        dst.write_band(4,nir_corrected)
#    print('corrected image saved to file: '+im_corrected_name)
#
#    return im_corrected_name

