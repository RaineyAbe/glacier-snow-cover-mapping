# Functions for image adjustment and snow classification in PlanetScope 4-band images
# Rainey Aberle
# 2022

import math
import rasterio as rio
from rasterio.mask import mask
import numpy as np
from pyproj import Proj, transform, Transformer
import matplotlib.pyplot as plt
import subprocess
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
def plot_im_RGB_histogram(im_path, im_fn):
    '''
    Plot PlanetScope 4-band RGB image with histograms for the B, G, R, and NIR bands.
    
    Parameters
    ----------
    im_path: str
        path in directory to image
        
    im_fn: str
        image file name
    
    Returns
    ----------
    fig: matplotlib.figure
        resulting figure handle
    
    '''
    
    # load image
    os.chdir(im_path)
    im = rio.open(im_fn)
    
    # load bands (blue, green, red, near infrared)
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    if np.nanmax(b) > 1e3:
        im_scalar = 10000
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
    # replace no data values with NaN
    b[b==0] = np.nan
    g[g==0] = np.nan
    r[r==0] = np.nan
    nir[nir==0] = np.nan
        
    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
    
    # plot RGB image and band histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6), gridspec_kw={'height_ratios': [1]})
    plt.rcParams.update({'font.size': 12, 'font.serif': 'Arial'})
    ax1.imshow(np.dstack([r, g, b]), aspect='auto',
               extent=(np.min(im_x)/1000, np.max(im_x/1000), np.min(im_y)/1000, np.max(im_y)/1000))
    ax1.set_xlabel('Easting [km]')
    ax1.set_ylabel('Northing [km]')
    ax2.hist(b.flatten(), color='blue', histtype='step', bins=100, label='blue')
    ax2.hist(g.flatten(), color='green', histtype='step', bins=100, label='green')
    ax2.hist(r.flatten(), color='red', histtype='step', bins=100, label='red')
    ax2.hist(nir.flatten(), color='brown', histtype='step', linewidth=2, bins=100, label='NIR')
    ax2.set_xlabel('Surface reflectance')
    ax2.set_ylabel('Pixel counts')
    ax2.grid()
    ax2.legend(loc='right')
    fig.suptitle(im_fn)
    fig.tight_layout()
    plt.show()
    
    return fig
    
# --------------------------------------------------
def mosaic_ims_by_date(im_path, im_fns, ext, out_path, AOI, plot_results):
    '''
    Mosaic PlanetScope 4-band images captured within the same hour using gdal_merge.py. Skips images which contain no real data in the AOI. Adapted from code developed by Jukes Liu.
    
    Parameters
    ----------
    im_path: str
    
    im_fns: list of str
    
    ext: str
    
    out_path: str
    
    AOI: geopandas.GeoDataFrame
    
    plot_results: bool
    
    Returns
    ----------
    
    '''
    # ----Grab all unique scenes (images captured within the same hour)
    unique_scenes = []
    for scene in im_fns:
        date = scene[0:11]
        unique_scenes.append(date)
    unique_scenes = list(set(unique_scenes))
    unique_scenes.sort() # sort chronologically
    
    # -----Loop through unique scenes
    for scene in unique_scenes:
        
        if '202108' in scene:
            # define the out path with correct extension
            if ext == 'DN_udm.tif':
                out_im_fn = os.path.join(scene + "_DN_mask.tif")
            elif ext == 'udm2.tif':
                out_im_fn = os.path.join(scene + "_mask.tif")
            else:
                out_im_fn = os.path.join(scene + ".tif")
            print(out_im_fn)
                
            # check if image mosaic already exists in directory
            if os.path.exists(out_path + out_im_fn)==True:
                print("image mosaic already exists... skipping.")
                print(" ")
                
                # plot output file
                if plot_results:
                    fig = plot_im_RGB_histogram(out_path, out_im_fn)
                
            else:
                
                file_paths = [] # files from the same hour to mosaic together
                for im_fn in im_fns: # check all files
                    if (scene in im_fn): # if they match the scene date
                        im = rio.open(im_path + im_fn) # open image
                        AOI_UTM = AOI.to_crs(str(im.crs)[5:]) # reproject AOI to image CRS
                        # mask the image using AOI geometry
                        b = im.read(1).astype(float) # blue band
                        mask = rio.features.geometry_mask(AOI_UTM.geometry,
                                                       b.shape,
                                                       im.transform,
                                                       all_touched=False,
                                                       invert=False)
                        # check if real data values exist within AOI
                        b_AOI = b[mask==0] # grab blue band values within AOI
                        # set no-data values to NaN
                        b_AOI[b_AOI==-9999] = np.nan
                        b_AOI[b_AOI==0] = np.nan
                        if (len(b_AOI[~np.isnan(b_AOI)]) > 0):
                            file_paths.append(im_path + im_fn) # add the path to the file
                            
                # check if any filepaths were added
                if len(file_paths) > 0:

                    # construct the gdal_merge command
                    cmd = 'gdal_merge.py -v '

                    # add input files to command
                    for file_path in file_paths:
                        cmd += file_path+' '

                    cmd += '-o ' + out_path + out_im_fn

                    # run the command
                    p = subprocess.run(cmd, shell=True, capture_output=True)
                    print(p)
                
                    # plot output file
                    if plot_results:
                        fig = plot_im_RGB_histogram(out_path, out_im_fn)
                else:
                    
                    print("No real data values within the AOI for images on this date... skipping.")
                    print(" ")

# --------------------------------------------------
def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min
    
# --------------------------------------------------
def sunpos(when, location, refraction):
    '''
    Determine the sun azimuth and elevation using the date and location.
    Modified from: https://levelup.gitconnected.com/python-sun-position-for-solar-energy-and-research-7a4ead801777
    Parameters
    ----------
    when: str array
        date of image capture ('YYYY', 'MM', 'DD', 'hh', 'mm', 'ss')
    location = coordinate pair (floats)
        approximate location of image capture (latitude, longitude)
    refraction: bool
        whether to account for refraction (bool)
    
    Returns
    ----------
    azimuth: float
        sun azimuth in degrees
    elevation: float
        sun elevation in degrees (float)
    '''
    
    # Extract the passed data
    year, month, day, hour, minute, second = when
    latitude, longitude = location

    # Math typing shortcuts
    rad, deg = math.radians, math.degrees
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2

    # Convert latitude and longitude to radians
    rlat = rad(latitude)
    rlon = rad(longitude)

    # Decimal hour of the day at Greenwich
    greenwichtime = hour + minute / 60 + second / 3600

    # Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )

    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873

    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768

    # Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )

    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum

    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))

    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))

    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon

    # Hour angle of the sun
    hour_ang = sidereal - rasc

    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))

    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation),
    )
    
    # Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)

    # Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60

    # Return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))

# --------------------------------------------------
def apply_hillshade_correction(crs, polygon, im, im_name, im_path, DEM_path, hs_path, out_path, skip_clipped, plot_results):
    '''
    Adjust image using by generating a hillshade model and minimizing the standard deviation of each band within the defined SCA
    
    Parameters
    ----------
    crs: float
        Coordinate Reference System (EPSG code)
    polygon:  shapely.geometry.polygon.Polygon
            polygon, where the band standard deviation will be minimized
    im: rasterio file
        input image
    im_name: str =
        file name name of the input image
    im_path: str
        path in directory to the input image
    DEM_path: str
        path in directory to the DEM used to generate the hillshade model
    hs_path: str =
        path to save hillshade model
    out_path: str
        path to save corrected image file
    skip_clipped: bool
        whether to skip images where bands appear "clipped"
    plot_results: bool
        whether to plot results to a matplotlib.pyplot.figure
    
    Returns
    ----------
    im_corrected_name: str
        file name of the hillshade-corrected image saved to file
    '''

    print('HILLSHADE CORRECTION')

    # -----Read image bands
    im_scalar = 10000
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    # divide by im_scalar if they have not been already
    if (np.nanmean(b)>1e3):
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
            
    # -----Return if image bands are likely clipped
    if skip_clipped==True:
        if (np.nanmax(b) < 0.8) or (np.nanmax(g) < 0.8) or (np.nanmax(r) < 0.8):
            print('image bands appear clipped... skipping.')
            im_corrected_name = 'N/A'
            return im_corrected_name
        
    # -----Define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
        
    # -----filter image points outside the SCA
    im_x_mesh, im_y_mesh = np.meshgrid(im_x, im_y)
    b_polygon = b[np.where((im_x_mesh >= polygon.bounds[0]) & (im_x_mesh <= polygon.bounds[2]) &
                      (im_y_mesh >= polygon.bounds[1]) & (im_y_mesh <= polygon.bounds[3]))]
    g_polygon = g[np.where((im_x_mesh >= polygon.bounds[0]) & (im_x_mesh <= polygon.bounds[2]) &
                      (im_y_mesh >= polygon.bounds[1]) & (im_y_mesh <= polygon.bounds[3]))]
    r_polygon = r[np.where((im_x_mesh >= polygon.bounds[0]) & (im_x_mesh <= polygon.bounds[2]) &
                      (im_y_mesh >= polygon.bounds[1]) & (im_y_mesh <= polygon.bounds[3]))]
    nir_polygon = nir[np.where((im_x_mesh >= polygon.bounds[0]) & (im_x_mesh <= polygon.bounds[2]) &
                           (im_y_mesh >= polygon.bounds[1]) & (im_y_mesh <= polygon.bounds[3]))]
                               
    # -----Return if image does not contain real values within the SCA
    if ((np.min(polygon.exterior.xy[0])>np.min(im_x))
        & (np.max(polygon.exterior.xy[0])<np.max(im_x))
        & (np.min(polygon.exterior.xy[1])>np.min(im_y))
        & (np.max(polygon.exterior.xy[1])<np.max(im_y))
        & (np.nanmean(b_polygon)>0))==False:
        
        print('image does not contain real values within the SCA... skipping.')
        im_corrected_name = 'N/A'
        return im_corrected_name
                
    # -----Extract image information for sun position calculation
    # location: grab center image coordinate, convert to lat lon
    xmid = ((im.bounds.right - im.bounds.left)/2 + im.bounds.left)
    ymid = ((im.bounds.top - im.bounds.bottom)/2 + im.bounds.bottom)
    transformer = Transformer.from_crs("epsg:"+str(crs), "epsg:4326")
    location = transformer.transform(xmid, ymid)
    # when: year, month, day, hour, minute, second
    when = (float(im_name[0:4]), float(im_name[4:6]), float(im_name[6:8]),
            float(im_name[9:11]), float(im_name[11:13]), float(im_name[13:15]))
    # sun azimuth and elevation
    azimuth, elevation = sunpos(when, location, refraction=1)

    # -----Make directory for hillshade models (if it does not already exist in file)
    if os.path.exists(hs_path)==False:
        os.mkdir(hs_path)
        print('made directory for hillshade model:'+hs_path)
            
    # -----Create hillshade model (if it does not already exist in file)
    hs_fn = hs_path+str(azimuth)+'-az_'+str(elevation)+'-z_hillshade.tif'
    if os.path.exists(hs_fn):
        print('hillshade model already exists in directory, loading...')
    else:
#                print('creating hillshade model...')
        # construct the gdal_merge command
        # modified from: https://github.com/clhenrick/gdal_hillshade_tutorial
        # gdaldem hillshade -az aximuth -z elevation dem.tif hillshade.tif
        cmd = 'gdaldem hillshade -az '+str(azimuth)+' -z '+str(elevation)+' '+str(DEM_path)+' '+hs_fn
        # run the command
        p = subprocess.run(cmd, shell=True, capture_output=True)
        print(p)

    # -----load hillshade model from file
    hs = rio.open(hs_fn)
#            print('hillshade model loaded from file...')
    # coordinates
    hs_x = np.linspace(hs.bounds.left, hs.bounds.right, num=np.shape(hs.read(1))[1])
    hs_y = np.linspace(hs.bounds.top, hs.bounds.bottom, num=np.shape(hs.read(1))[0])

    # -----Resample hillshade to image coordinates
    # resampled hillshade file name
    hs_resamp_fn = hs_path+str(azimuth)+'-az_'+str(elevation)+'-z_hillshade_resamp.tif'
    # create interpolation object
    f = interp2d(hs_x, hs_y, hs.read(1))
    hs_resamp = f(im_x, im_y)
    hs_resamp = np.flipud(hs_resamp)
    # save to file
    with rio.open(hs_resamp_fn,'w',
                  driver='GTiff',
                  height=hs_resamp.shape[0],
                  width=hs_resamp.shape[1],
                  dtype=hs_resamp.dtype,
                  count=1,
                  crs=im.crs,
                  transform=im.transform) as dst:
        dst.write(hs_resamp, 1)
    print('resampled hillshade model saved to file:',hs_resamp_fn)

    # -----load resampled hillshade model
    hs_resamp = rio.open(hs_resamp_fn).read(1)
    print('resampled hillshade model loaded from file')
    # -----filter hillshade model points outside the SCA
    hs_polygon = hs_resamp[np.where((im_x_mesh >= polygon.bounds[0]) & (im_x_mesh <= polygon.bounds[2]) & (im_y_mesh >= polygon.bounds[1]) & (im_y_mesh <= polygon.bounds[3]))]

    # -----normalize hillshade model
    hs_norm = (hs_resamp - np.min(hs_resamp)) / (np.max(hs_resamp) - np.min(hs_resamp))
    hs_polygon_norm = (hs_polygon - np.min(hs_polygon)) / (np.max(hs_polygon) - np.min(hs_polygon))

            # -----plot resampled, normalized hillshade model for sanity check
    #        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,8))
    #        hs_im = ax1.imshow(hs.read(1), extent=(np.min(hs_x)/1000, np.max(hs_x)/1000, np.min(hs_y)/1000, np.max(hs_y)/1000))
    #        hsnorm_im = ax2.imshow(hs_norm, extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    #        ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='white', linewidth=2, label='SCA')
    #        fig.colorbar(hs_im, ax=ax1, shrink=0.5)
    #        fig.colorbar(hsnorm_im, ax=ax2, shrink=0.5)
    #        plt.show()
            
    # -----loop through hillshade scalar multipliers
#            print('solving for optimal band scalars...')
    # define scalars to test
    hs_scalars = np.linspace(0,0.5,num=21)
    # blue
    b_polygon_mu = np.zeros(len(hs_scalars)) # mean
    b_polygon_sigma =np.zeros(len(hs_scalars)) # std
    # green
    g_polygon_mu = np.zeros(len(hs_scalars)) # mean
    g_polygon_sigma = np.zeros(len(hs_scalars)) # std
    # red
    r_polygon_mu = np.zeros(len(hs_scalars)) # mean
    r_polygon_sigma = np.zeros(len(hs_scalars)) # std
    # nir
    nir_polygon_mu = np.zeros(len(hs_scalars)) # mean
    nir_polygon_sigma = np.zeros(len(hs_scalars)) # std
    i=0 # loop counter
    for hs_scalar in hs_scalars:
        # full image
        b_adj = b - (hs_norm * hs_scalar)
        g_adj = g - (hs_norm * hs_scalar)
        r_adj = r - (hs_norm * hs_scalar)
        nir_adj = nir - (hs_norm * hs_scalar)
        # SCA
        b_polygon_mu[i] = np.nanmean(b_polygon- (hs_polygon_norm * hs_scalar))
        b_polygon_sigma[i] = np.nanstd(b_polygon- (hs_polygon_norm * hs_scalar))
        g_polygon_mu[i] = np.nanmean(g_polygon- (hs_polygon_norm * hs_scalar))
        g_polygon_sigma[i] = np.nanstd(g_polygon- (hs_polygon_norm * hs_scalar))
        r_polygon_mu[i] = np.nanmean(r_polygon- (hs_polygon_norm * hs_scalar))
        r_polygon_sigma[i] = np.nanstd(r_polygon- (hs_polygon_norm * hs_scalar))
        nir_polygon_mu[i] = np.nanmean(nir_polygon- (hs_polygon_norm * hs_scalar))
        nir_polygon_sigma[i] = np.nanstd(nir_polygon- (hs_polygon_norm * hs_scalar))
        i+=1 # increase loop counter

    # -----Determine optimal scalar for each image band
    Ib = np.where(b_polygon_sigma==np.min(b_polygon_sigma))[0][0]
    b_scalar = hs_scalars[Ib]
    Ig = np.where(g_polygon_sigma==np.min(g_polygon_sigma))[0][0]
    g_scalar = hs_scalars[Ig]
    Ir = np.where(r_polygon_sigma==np.min(r_polygon_sigma))[0][0]
    r_scalar = hs_scalars[Ir]
    Inir = np.where(nir_polygon_sigma==np.min(nir_polygon_sigma))[0][0]
    nir_scalar = hs_scalars[Inir]
    print('Optimal scalars:  Blue   |   Green   |   Red   |   NIR')
    print(b_scalar, g_scalar, r_scalar, nir_scalar)

    # -----Apply optimal hillshade model correction
    b_corrected = b - (hs_norm * hs_scalars[Ib])
    g_corrected = g - (hs_norm * hs_scalars[Ig])
    r_corrected = r - (hs_norm * hs_scalars[Ir])
    nir_corrected = nir - (hs_norm * hs_scalars[Inir])

    # -----Replace previously 0 values with 0 to signify no-data
    b_corrected[b==0] = 0
    g_corrected[g==0] = 0
    r_corrected[r==0] = 0
    nir_corrected[nir==0] = 0
        
    # -----Plot original and corrected images and band histograms
    if plot_results==True:
        fig1, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,12), gridspec_kw={'height_ratios': [3, 1]})
        plt.rcParams.update({'font.size': 14, 'font.serif': 'Arial'})
        # original image
        ax1.imshow(np.dstack([r, g, b]),
                   extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
        ax1.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
        ax1.set_xlabel('Northing [km]')
        ax1.set_ylabel('Easting [km]')
        ax1.set_title('Original image')
        # corrected image
        ax2.imshow(np.dstack([r_corrected, g_corrected, b_corrected]),
                   extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
        ax2.plot([x/1000 for x in SCA.exterior.xy[0]], [y/1000 for y in SCA.exterior.xy[1]], color='black', linewidth=2, label='SCA')
        ax2.set_xlabel('Northing [km]')
        ax2.set_title('Corrected image')
        # band histograms
        ax3.hist(nir[nir>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
        ax3.hist(b[b>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
        ax3.hist(g[g>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
        ax3.hist(r[r>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
        ax3.set_xlabel('Surface reflectance')
        ax3.set_ylabel('Pixel counts')
        ax3.grid()
        ax3.legend()
        ax4.hist(nir_corrected[nir_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
        ax4.hist(b_corrected[b_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
        ax4.hist(g_corrected[g_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
        ax4.hist(r_corrected[r_corrected>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
        ax4.set_xlabel('Surface reflectance')
        ax4.grid()
        fig1.tight_layout()
        plt.show()
    
    # -----save hillshade-corrected image to file
    # create output directory (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print('created output directory:',out_path)
    # file name
    im_corrected_name = im_name[0:-4]+'_hs-corrected.tif'
    # metadata
    out_meta = im.meta.copy()
    out_meta.update({'driver':'GTiff',
                     'width':b_corrected.shape[1],
                     'height':b_corrected.shape[0],
                     'count':4,
                     'dtype':'float64',
                     'crs':im.crs,
                     'transform':im.transform})
    # write to file
    with rio.open(out_path+im_corrected_name, mode='w',**out_meta) as dst:
        dst.write_band(1,b_corrected)
        dst.write_band(2,g_corrected)
        dst.write_band(3,r_corrected)
        dst.write_band(4,nir_corrected)
    print('corrected image saved to file: '+im_corrected_name)
                    
    return im_corrected_name

# --------------------------------------------------
def create_top_elev_AOI_poly(AOI, im_path, im_fns, DEM, DEM_x, DEM_y):
    '''
    Function to generate a polygon of the top 70th percentile elevations within the defined Area of Interest (AOI).
    
    Parameters
    ----------
    AOI:
        must be in same coordinate reference system (CRS) as the image
    
    im_fns:
    
    im_path:
    
    DEM:
    
    DEM_x:
    
    DEM_y:
    
    Returns
    ----------
    polygon:
    
    b: numpy.ndarray
        surface reflectance of the blue image band
    
    g: numpy.ndarray
        surface reflectance of the green image band
    
    r: numpy.ndarray
        surface reflectance of the red image band
    
    nir: numpy.ndarray
        surface reflectance of the near-infrared image band
    '''

    # -----Read one image that contains AOI to create polygon
    os.chdir(im_path)
    for i in range(0,len(im_fns)):
        # define image filename
        im_fn = im_fns[i]
        # open image
        im = rio.open(im_fn)
        # mask the image using AOI geometry
        mask = rio.features.geometry_mask(AOI.geometry,
                                       im.read(1).shape,
                                       im.transform,
                                       all_touched=False,
                                       invert=False)
        # check if any image values exist within AOI
        if (0 in mask.flatten()):
            break

    # -----Get image band shape and define image coordinates
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    if (np.nanmax(b) > 1e3):
        im_scalar = 10000 # scalar multiplier for image reflectance values
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
    
    # -----Create mask from AOI on image grid
    mask = rio.features.geometry_mask(AOI.geometry,
                                           b.shape,
                                           im.transform,
                                           all_touched=False,
                                           invert=False)
    # -----Regrid DEM to image coordinates
    DEM_x_mesh, DEM_y_mesh = np.meshgrid(DEM_x, DEM_y)
    im_x_mesh, im_y_mesh = np.meshgrid(im_x, im_y)
    DEM_regrid = griddata(list(zip(DEM_x_mesh.flatten(), DEM_y_mesh.flatten())),
        np.flipud(DEM).flatten(),
        (im_x_mesh, im_y_mesh))

    # -----Mask DEM outside the AOI and the bottom percentile of elevations
    DEM_regrid_AOImasked = np.where(mask==0, DEM_regrid, np.nan)
    DEM_P = np.nanmedian(DEM_regrid_AOImasked) + iqr(DEM_regrid_AOImasked, rng=(30,70), nan_policy='omit')/2
    DEM_regrid_AOImasked_Pmasked = np.where(DEM_regrid_AOImasked > DEM_P, DEM_regrid_AOImasked, np.nan)
    mask = np.where(~np.isnan(DEM_regrid_AOImasked_Pmasked), 1, 0)

    # -----Convert mask to polygon
    # adapted from: https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    all_polygons = []
    for s, value in rio.features.shapes(mask.astype(np.int16), mask=(mask >0), transform=im.transform):
        all_polygons.append(shape(s))
    all_polygons = MultiPolygon(all_polygons)
    # ouput results
    polygon = all_polygons
    
    return polygon, im_fn, im, r, g, b, im_x, im_y
    
    
# --------------------------------------------------
def adjust_image_radiometry(im, im_fn, im_path, polygon, out_path, skip_clipped, plot_results):
    '''
    Adjust PlanetScope image band radiometry using the band values in a defined snow-covered area (SCA) and the expected surface reflectance of snow.
    
    Parameters
    ----------
    im: rasterio file
        input image
    im_fn: str
        file name of the input image
    im_path: str
        path in directory to the input image
    polygon: shapely.geometry.polygon.Polygon
        polygon used to adjust band values
    out_path: str
        path in directory where adjusted image file will be saved
    skip_clipped: bool
        whether to skip images where bands appear "clipped"
    plot_results: bool
        whether to plot results to a matplotlib.pyplot.figure
    
    Returns
    ----------
    im_adj_name: str
        file name of the adjusted image saved to file
    '''
    
    print('RADIOMETRIC ADJUSTMENT')
    
    # -----Load image
    # define bands (blue, green, red, near infrared)
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    if np.nanmax(b) > 1e3:
        im_scalar = 10000 # scalar multiplier for image reflectance values
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
    # replace no-data values with NaN
    b[b==0] = np.nan
    g[g==0] = np.nan
    r[r==0] = np.nan
    nir[nir==0] = np.nan
    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(b)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(b)[0])
    
    # -----Check if out_path and adjusted image file exist
    # create output directory (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print('created output directory:',out_path)
    # check if adjusted image file exists
    im_adj_fn = im_fn[0:-4]+'_adj.tif' # adjusted image file name
    if os.path.exists(out_path + im_adj_fn)==True:
    
        print('adjusted image already exists... loading from file.')
        
        # load adjusted image from file
        im_adj = rio.open(out_path+im_adj_fn)
        # define bands (blue, green, red, near infrared)
        b_adj = im_adj.read(1).astype(float)
        g_adj = im_adj.read(2).astype(float)
        r_adj = im_adj.read(3).astype(float)
        nir_adj = im_adj.read(4).astype(float)
        if (np.nanmax(b_adj) > 1e3):
            im_scalar = 10000 # scalar multiplier for image reflectance values
            b_adj = b_adj / im_scalar
            g_adj = g_adj / im_scalar
            r_adj = r_adj / im_scalar
            nir_adj = nir_adj / im_scalar

    else:
            
        # -----Return if image bands are likely clipped
        if skip_clipped==True:
            if (np.nanmax(b) < 0.8) or (np.nanmax(g) < 0.8) or (np.nanmax(r) < 0.8):
                print('image bands appear clipped... skipping.')
                im_adj_fn = 'N/A'
                return im_adj_fn

        # -----Return if image does not contain polygon
        # mask the image using polygon geometry
        mask = rio.features.geometry_mask([polygon],
                                       b.shape,
                                       im.transform,
                                       all_touched=False,
                                       invert=False)
        # skip if image does not contain polygon
        if (0 not in mask.flatten()):
            print('image does not contain polygon... skipping.')
            im_adj_fn = 'N/A'
            return im_adj_fn
            
        # -----Return if no real values exist within the SCA
        if (np.nanmean(b==0)) or (np.isnan(np.nanmean(b))):
            print('image does not contain any real values within the polygon... skipping.')
            im_adj_name = 'N/A'
            return im_adj_name
            
        # -----Define desired SR values at the bright area and darkest point for each band
        # bright area
        bright_b_adj = 0.94
        bright_g_adj = 0.95
        bright_r_adj = 0.94
        bright_nir_adj = 0.78
        # dark point
        dark_adj = 0.0
        
        # -----Filter image points outside the polygon
        b_polygon = b[mask==0]
        g_polygon = g[mask==0]
        r_polygon = r[mask==0]
        nir_polygon = nir[mask==0]
                                  
        # -----Adjust SR using bright and dark points
        # band_adjusted = band*A - B
        # A = (bright_adjusted - dark_adjusted) / (bright - dark)
        # B = (dark*bright_adjusted - bright*dark_adjusted) / (bright - dark)
        # blue band
        bright_b = np.nanmedian(b_polygon) # SR at bright point
        dark_b = np.nanmin(b) # SR at darkest point
        A = (bright_b_adj - dark_adj) / (bright_b - dark_b)
        B = (dark_b*bright_b_adj - bright_b*dark_adj) / (bright_b - dark_b)
        b_adj = (b * A) - B
        b_adj = np.where(b==0, np.nan, b_adj) # replace no data values with nan
        # green band
        bright_g = np.nanmedian(g_polygon) # SR at bright point
        dark_g = np.nanmin(g) # SR at darkest point
        A = (bright_g_adj - dark_adj) / (bright_g - dark_g)
        B = (dark_g*bright_g_adj - bright_g*dark_adj) / (bright_g - dark_g)
        g_adj = (g * A) - B
        g_adj = np.where(g==0, np.nan, g_adj) # replace no data values with nan
        # red band
        bright_r = np.nanmedian(r_polygon) # SR at bright point
        dark_r = np.nanmin(r) # SR at darkest point
        A = (bright_r_adj - dark_adj) / (bright_r - dark_r)
        B = (dark_r*bright_r_adj - bright_r*dark_adj) / (bright_r - dark_r)
        r_adj = (r * A) - B
        r_adj = np.where(r==0, np.nan, r_adj) # replace no data values with nan
        # nir band
        bright_nir = np.nanmedian(nir_polygon) # SR at bright point
        dark_nir = np.nanmin(nir) # SR at darkest point
        A = (bright_nir_adj - dark_adj) / (bright_nir - dark_nir)
        B = (dark_nir*bright_nir_adj - bright_nir*dark_adj) / (bright_nir - dark_nir)
        nir_adj = (nir * A) - B
        nir_adj = np.where(nir==0, np.nan, nir_adj) # replace no data values with nan

        # -----Save adjusted raster image to file
        # copy metadata
        out_meta = im.meta.copy()
        out_meta.update({'driver':'GTiff',
                         'width':b_adj.shape[1],
                         'height':b_adj.shape[0],
                         'count':4,
                         'dtype':'uint16',
                         'crs':im.crs,
                         'transform':im.transform})
        # write to file
        with rio.open(out_path+im_adj_fn, mode='w',**out_meta) as dst:
            # write bands - multiply bands by im_scalar and convert datatype to uint64 to decrease file size
            dst.write_band(1, b_adj * im_scalar)
            dst.write_band(2, g_adj * im_scalar)
            dst.write_band(3, r_adj * im_scalar)
            dst.write_band(4, nir_adj * im_scalar)
        print('adjusted image saved to file: '+im_adj_fn)

    # -----Plot RGB images and band histograms for the original and adjusted image
    if plot_results:
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(16,12), gridspec_kw={'height_ratios': [3, 1]})
        plt.rcParams.update({'font.size': 12, 'font.serif': 'Arial'})
        # original image
        im_original = ax1.imshow(np.dstack([r, g, b]),
                    extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
        count=0
        for geom in polygon.geoms:
            xs, ys = geom.exterior.xy
            if count==0:
                ax1.plot([x/1000 for x in xs], [y/1000 for y in ys], color='orange', label='polygon(s)')
            else:
                ax1.plot([x/1000 for x in xs], [y/1000 for y in ys], color='orange', label='_nolegend_')
            count+=1
        ax1.legend()
        ax1.set_xlabel('Easting [km]')
        ax1.set_ylabel('Northing [km]')
        ax1.set_title('Original image')
        # adjusted image
        ax2.imshow(np.dstack([r_adj, g_adj, b_adj]),
            extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
        count=0
        for geom in polygon.geoms:
            xs, ys = geom.exterior.xy
            if count==0:
                ax2.plot([x/1000 for x in xs], [y/1000 for y in ys], color='orange', label='polygon(s)')
            else:
                ax2.plot([x/1000 for x in xs], [y/1000 for y in ys], color='orange', label='_nolegend_')
            count+=1
        ax2.set_xlabel('Easting [km]')
        ax2.set_title('Adjusted image')
        # band histograms
        ax3.hist(nir[nir>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
        ax3.hist(b[b>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
        ax3.hist(g[g>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
        ax3.hist(r[r>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
        ax3.set_xlabel('Surface reflectance')
        ax3.set_ylabel('Pixel counts')
        ax3.grid()
        ax3.legend()
        ax4.hist(nir_adj[nir_adj>0].flatten(), bins=100, histtype='step', linewidth=1, color='purple', label='NIR')
        ax4.hist(b_adj[b_adj>0].flatten(), bins=100, histtype='step', linewidth=1, color='blue', label='Blue')
        ax4.hist(g_adj[g_adj>0].flatten(), bins=100, histtype='step', linewidth=1, color='green', label='Green')
        ax4.hist(r_adj[r_adj>0].flatten(), bins=100, histtype='step', linewidth=1, color='red', label='Red')
        ax4.set_xlabel('Surface reflectance')
        ax4.grid()
        fig.tight_layout()
        plt.show()
            
    return im_adj_fn

# --------------------------------------------------
def query_GEE_for_DEM(AOI, im_path, im_fns):
    '''Query GEE for the ASTER Global DEM, clip to the AOI, and return as a numpy array.
    
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
    DEM_np: numpy array
        elevations extracted within the AOI
    DEM_x: numpy array
        vector of x coordinates of the DEM
    DEM_y: numpy array
        vector of y coordinates of the DEM
    AOI_UTM: geopandas.geodataframe.GeoDataFrame
        AOI reprojected to the coordinate reference system of the images
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
                            ])

    # -----Query GEE for DEM, clip to AOI
    # Use ArcticDEM if it fully covers the AOI. Otherwise, use ASTER GDEM.
#    if ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic").geometry().contains(AOI_WGS_bb_ee).getInfo()==True:
#        DEM = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic").clip(AOI_WGS_bb_ee)
#        print("DEM = ArcticDEM")
#    else:
#        DEM = ee.Image("NASA/ASTER_GED/AG100_003").clip(AOI_WGS_bb_ee)
#        print("DEM = ASTER GDEM")
    DEM = ee.Image("NASA/ASTER_GED/AG100_003").clip(AOI_WGS_bb_ee)

    # -----Grab UTM projection from images, reproject DEM and AOI
    if type(im_fns)==str:
        im = rio.open(im_path + im_fns)
    else:
        im = rio.open(im_path + im_fns[0])
    DEM_UTM = ee.Image.reproject(DEM, str(im.crs))
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
    
# --------------------------------------------------
def crop_images_to_AOI(im_path, im_fns, AOI):
    '''
    Crop images to AOI.
    
    Parameters
    ----------
    im_path: str
        path in directory to images
    im_fns: str array
        file names of images to crop
    AOI: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked. Only the exterior bounds used for cropping (no holes). AOI must be in the same CRS as the images.
    
    Returns
    ----------
    cropped_im_path: str
        path in directory to cropped images
    '''
    
    # make folder for cropped images if it does not exist
    cropped_im_path = im_path + "../cropped/"
    if os.path.isdir(cropped_im_path)==0:
        os.mkdir(cropped_im_path)
        print(cropped_im_path+" directory made")
    
    # loop through images
    for im_fn in im_fns:

        # open image
        im = rio.open(im_path + im_fn)

        # check if file exists in directory already
        cropped_im_fn = cropped_im_path + im_fn[0:15] + "_crop.tif"
        if os.path.exists(cropped_im_fn)==True:
            print("cropped image already exists in directory...skipping.")
        else:
            # mask image pixels outside the AOI exterior
#            AOI_bb = [AOI.bounds]
            out_image, out_transform = mask(im, AOI.buffer(100), crop=True)
            out_meta = im.meta.copy()
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
            with rio.open(cropped_im_fn, "w", **out_meta) as dest:
                dest.write(out_image)
            print(cropped_im_fn + " saved")
            
    return cropped_im_path

# --------------------------------------------------
def plot_im_classified_histograms(im, im_dt, im_x, im_y, im_classified, snow_elev, b, g, r, nir, DEM_x, DEM_y, DEM):

    # -----Grab 2nd percentile snow elevation
    P = np.median(snow_elev) - iqr(snow_elev, rng=(2, 98))/2
    
    # -----Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
    plt.rcParams.update({'font.size': 14, 'font.sans-serif': 'Arial'})
    # RGB image
    ax1.imshow(np.dstack([r, g, b]),
                extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    ax1.set_xlabel("Easting [km]")
    ax1.set_ylabel("Northing [km]")
    # define colors for plotting
    color_snow = '#4eb3d3'
    color_ice = '#084081'
    color_rock = '#fdbb84'
    color_water = '#bdbdbd'
    # define x and y limits
    xmin, xmax = np.min(im_x)/1000, np.max(im_x)/1000
    ymin, ymax = np.min(im_y)/1000, np.max(im_y)/1000
    # snow
    if any(im_classified.flatten()==1):
        ax2.imshow(np.where(im_classified == 1, 1, np.nan), cmap=matplotlib.colors.ListedColormap([color_snow, 'white']),
                    extent=(xmin, xmax, ymin, ymax))
        ax2.scatter(0, 0, color=color_snow, s=50, label='snow') # plot dummy point for legend
    if any(im_classified.flatten()==2):
        ax2.imshow(np.where(im_classified == 2, 4, np.nan), cmap=matplotlib.colors.ListedColormap([color_snow, 'white']),
                    extent=(xmin, xmax, ymin, ymax))
    # ice
    if any(im_classified.flatten()==3):
        ax2.imshow(np.where(im_classified == 3, 1, np.nan), cmap=matplotlib.colors.ListedColormap([color_ice, 'white']),
                    extent=(xmin, xmax, ymin, ymax))
        ax2.scatter(0, 0, color=color_ice, s=50, label='ice') # plot dummy point for legend
    # rock/debris
    if any(im_classified.flatten()==4):
        ax2.imshow(np.where(im_classified == 4, 1, np.nan), cmap=matplotlib.colors.ListedColormap([color_rock, 'white']),
                    extent=(xmin, xmax, ymin, ymax))
        ax2.scatter(0, 0, color=color_rock, s=50, label='rock') # plot dummy point for legend
    # water
    if any(im_classified.flatten()==5):
        ax2.imshow(np.where(im_classified == 5, 10, np.nan), cmap=matplotlib.colors.ListedColormap([color_water, 'white']),
                    extent=(xmin, xmax, ymin, ymax))
        ax2.scatter(0, 0, color=color_water, s=50, label='water') # plot dummy point for legend
    # snow elevation contour
    cs = ax2.contour(DEM_x/1000, DEM_y/1000, np.flipud(DEM.squeeze()), [P], colors=['black'])
    ax2.legend(loc='lower left')
    ax2.set_xlabel("Easting [km]")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    # image bands histogram
    h_b = ax3.hist(b[b!=0].flatten(), color='blue', histtype='step', linewidth=2, bins=100, label="blue")
    h_g = ax3.hist(g[g!=0].flatten(), color='green', histtype='step', linewidth=2, bins=100, label="green")
    h_r = ax3.hist(r[r!=0].flatten(), color='red', histtype='step', linewidth=2, bins=100, label="red")
    h_nir = ax3.hist(nir[nir!=0].flatten(), color='brown', histtype='step', linewidth=2, bins=100, label="NIR")
    ax3.set_xlabel("Surface reflectance")
    ax3.set_ylabel("Pixel counts")
    ax3.legend(loc='upper left')
    ax3.set_ylim(0,np.max([h_nir[0][1:], h_g[0][1:], h_r[0][1:], h_b[0][1:]])+5000)
    ax3.grid()
    # snow elevations histogram
    ax4.hist(snow_elev.flatten(), bins=100, color=color_snow)
    ax4.set_xlabel("Elevation [m]")
    ax4.grid()
    ymin, ymax = ax4.get_ylim()[0], ax4.get_ylim()[1]
    ax4.plot((P, P), (ymin, ymax), color='black', label='P$_{2}$')
    ax4.set_ylim(ymin, ymax)
    ax4.legend(loc='lower right')
    fig.tight_layout()
    fig.suptitle(im_dt)
    plt.show()
    
    # extract contour vertices
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    
    return fig

# --------------------------------------------------
def classify_image(im, im_fn, clf, feature_cols, out_path):
    '''
    Function to classify input image using a pre-trained classifier
    
    Parameters
    ----------
    im: rasterio object
        input image
    im_fn: str
        file name of input image
    clf: sklearn.classifier
        previously trained SciKit Learn Classifier
    feature_cols: array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red']
        features used by classifier ()
    out_path: str
        path to save classified images
    plot_output: bool
        whether to plot RGB and classified image
        
    Returns
    ----------
    im_x: numpy.array
        x coordinates of input image
    im_y: numpy.array
        y coordinates of image
    snow: numpy.array
        binary array of predicted snow presence in input image, where 0 = no snow and 1 = snow
    '''

    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(im.read(1))[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(im.read(1))[0])
        
    # -----Make directory for snow images (if it does not already exist in file)
    if os.path.exists(out_path)==False:
        os.mkdir(out_path)
        print("Made directory for classified snow images:" + out_path)
            
    # -----Check if classified snow image exists in directory already
    im_classified_fn = im_fn[0:-4] + "_classified.tif"
    if os.path.exists(out_path+im_classified_fn):
    
        print("Classified snow image already exists in directory, loading...")
        s = rio.open(out_path + im_classified_fn)
        im_classified = s.read(1).astype(float)
        
    else:
    
        # define bands
        b = im.read(1).astype(float)
        g = im.read(2).astype(float)
        r = im.read(3).astype(float)
        nir = im.read(4).astype(float)
        # check if bands must be divided by scalar
        if (np.nanmax(b) > 1000):
            im_scalar = 10000
            b = b / im_scalar
            g = g / im_scalar
            r = r / im_scalar
            nir = nir / im_scalar
        # replace no data values with NaN
        b[b==0] = np.nan
        g[g==0] = np.nan
        r[r==0] = np.nan
        nir[nir==0] = np.nan
        b[b==-9999] = np.nan
        g[g==-9999] = np.nan
        r[r==-9999] = np.nan
        nir[nir==-9999] = np.nan
        # calculate NDSI with red and NIR bands
        ndsi = (r - nir) / (r + nir)
        
        # Find indices of real numbers (no NaNs allowed in classification)
        I_real = np.where((~np.isnan(b)) & (~np.isnan(g)) & (~np.isnan(r)) & (~np.isnan(nir)) & (~np.isnan(ndsi)))
        
        # save in Pandas dataframe
        df = pd.DataFrame()
        df['blue'] = b[I_real].flatten()
        df['green'] = g[I_real].flatten()
        df['red'] = r[I_real].flatten()
        df['NIR'] = nir[I_real].flatten()
        df['NDSI'] = ndsi[I_real].flatten()

        # classify image
        array_classified = clf.predict(df[feature_cols])
        
        # reshape from flat array to original shape
        im_classified = np.zeros((np.shape(b)[0], np.shape(b)[1]))
        im_classified[:] = np.nan
        im_classified[I_real] = array_classified
        
        # replace nan values with -9999 in order to save file with datatype int16
        im_classified[np.isnan(im_classified)] = -9999
        
        # save to file
        with rio.open(out_path + im_classified_fn,'w',
                      driver='GTiff',
                      height=im.shape[0],
                      width=im.shape[1],
                      dtype='int16',
                      count=1,
                      crs=im.crs,
                      transform=im.transform) as dst:
            dst.write(im_classified, 1)
        print("Classified image saved to file:",im_classified_fn)
                
    return im_x, im_y, im_classified
    
# --------------------------------------------------
def calculate_SCA(im, im_classified):
    '''Function to calculated total snow-covered area (SCA) from using an input image and a snow binary mask of the same resolution and grid.
    INPUTS:
        im: rasterio object
            input image
        im_classified: numpy array
            classified image array with the same shape as the input image bands. Classes: snow = 1, shadowed snow = 2, ice = 3, rock/debris = 4.
    OUTPUTS:
        SCA: float
            snow-covered area in classified image [m^2]'''

    pA = im.res[0]*im.res[1] # pixel area [m^2]
    snow_count = np.count_nonzero(im_classified <= 2) # number of snow and shadowed snow pixels
    SCA = pA * snow_count # area of snow [m^2]

    return SCA

# --------------------------------------------------
def determine_snow_elevs(DEM, DEM_x, DEM_y, im, im_classified, im_dt, im_x, im_y, plot_output):
    '''Determine elevations of snow-covered pixels in the classified image.
    Parameters
    ----------
    DEM: numpy array
        digital elevation model
    DEM_x: numpy array
        vector of x coordinates of the DEM
    DEM_y: numpy array
        vector of y coordinates of the DEM
    im: rasterio object
        input image used to classify snow
    im_classified: numpy array
        classified image array with the same shape as the input image bands. Classes: snow = 1, shadowed snow = 2, ice = 3, rock/debris = 4.
    im_dt: numpy.datetime64
        datetime of the image capture
    im_x: numpy array
        vector of x coordinates of the input image
    im_y: numpy array
        vector of y coordinates of the input image
    plot_output: bool
        whether to plot the output RGB and snow classified image with histograms for surface reflectances of each band and the elevations of snow-covered pixels
        
    Returns
    ----------
    snow_elev: numpy array
        elevations at each snow-covered pixel
    '''
    
    # extract bands info
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    # check if bands must be divided by scalar
    if (np.nanmax(b) > 1000):
        im_scalar = 10000
        b = b / im_scalar
        g = g / im_scalar
        r = r / im_scalar
        nir = nir / im_scalar
    # replace no data values with NaN
    b[b==0] = np.nan
    g[g==0] = np.nan
    r[r==0] = np.nan
    nir[nir==0] = np.nan
    b[b==-9999] = np.nan
    g[g==-9999] = np.nan
    r[r==-9999] = np.nan
    nir[nir==-9999] = np.nan
    im_classified[im_classified==-9999] = np.nan
    # interpolate elevation from DEM at image points
    f = interp2d(DEM_x, DEM_y, DEM)
    im_elev = f(im_x, im_y)
    
    # minimum elevation of the image where data exist
    im_elev_real = np.where((b>0) & (~np.isnan(b)), im_elev, np.nan)
    im_elev_min = np.nanmin(im_elev_real)
    im_elev_max = np.nanmax(im_elev_real)
    
    # extract elevations where snow is present
    snow_elev = im_elev[im_classified<=2]
    
    # plot snow elevations histogram
    if plot_output:
        fig = plot_im_classified_histograms(im, im_dt, im_x, im_y, im_classified, snow_elev, b, g, r, nir, DEM_x, DEM_y, DEM)
        return im_elev_min, im_elev_max, snow_elev, fig
        
    return im_elev_min, im_elev_max, snow_elev
