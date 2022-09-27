# Functions for snow classification in Landsat images
# Rainey Aberle
# Department of Geosciences, Boise State University
# 2022

import rasterio as rio
import numpy as np
from pyproj import Proj, transform, Transformer
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon, MultiPolygon, shape
from scipy.interpolate import interp2d, griddata
from scipy.stats import iqr
import glob
import ee
import geopandas as gpd
import pandas as pd
from scipy import stats
import geemap
from shapely.geometry import Polygon
import matplotlib
from osgeo import gdal

# --------------------------------------------------

def query_GEE_for_Landsat(AOI, im_path, im_fns):
    '''Query GEE for Landsat Collection 2 Tier 1 Surface Reflectance images, clip to the AOI, and return as a numpy array.

    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM
    im_path: string
        full path to the directory holding the images to be classified
    im_fns: list of strings
        file names of images to be classified, located in im_path. Used to extract the desired coordinate reference system of the DEM

    Returns
    ----------
    L_np: numpy array
        elevations extracted within the AOI
    L_x: numpy array
        vector of x coordinates of the DEM
    L_y: numpy array
        vector of y coordinates of the DEM
    AOI_UTM: geopandas.geodataframe.GeoDataFrame
        AOI reprojected to the coordinate reference system of the images
    '''

    # -----Reformat AOI for clipping images
    # reproject AOI to WGS 84 for compatibility with images
    AOI_WGS = AOI.to_crs(4326)
    # reformat AOI_WGS bounding box as ee.Geometry for clipping DEM
    AOI_WGS_bb_ee = ee.Geometry.Polygon(
                            [[[AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]]]
                            ])

    # -----Query GEE for Landsat images, clip to AOI
    L_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max));

    # -----Grab UTM projection from images, reproject image collection and AOI
    if type(im_fns)==str:
        im = rio.open(im_path + im_fns)
    else:
        im = rio.open(im_path + im_fns[0])
#    L_col_UTM = L_col.map(function(im){ee.Image.reproject(im, str(im.crs))})
    AOI_UTM = AOI.to_crs(str(im.crs)[5:])

    # -----Convert DEM to numpy array, extract coordinates
    DEM_np = geemap.ee_to_numpy(DEM, ['elevation'], region=AOI_WGS_bb_ee, default_value=-9999).astype(float)
    DEM_np[DEM_np==-9999] = np.nan # set no data value to NaN
    DEM_x = np.linspace(AOI_UTM.geometry.bounds.minx[0], AOI_UTM.geometry.bounds.maxx[0], num=np.shape(DEM_np)[1])
    DEM_y = np.linspace(AOI_UTM.geometry.bounds.miny[0], AOI_UTM.geometry.bounds.maxy[0], num=np.shape(DEM_np)[0])

    # -----Plot to check for success
    fig, ax = plt.subplots(figsize=(8,8))
    plt.rcParams.update({'font.size':14, 'font.sans-serif':'Arial'})
    DEM_im = plt.imshow(DEM_np, cmap='Greens_r',
                        extent=(np.min(DEM_x), np.max(DEM_x), np.min(DEM_y), np.max(DEM_y)))
    AOI_UTM.plot(ax=ax, facecolor='none', edgecolor='black', label='AOI')
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    fig.colorbar(DEM_im, ax=ax, shrink=0.5, label='Elevation [m]')
    plt.show()

    return DEM_np, DEM_x, DEM_y, AOI_UTM
