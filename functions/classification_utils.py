# Functions related to classification of snow in PlanetScope 4-band images
# Rainey Aberle
# 2022

import rasterio as rio
from rasterio.mask import mask
import os
import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy import stats
import ee
import geemap
from shapely.geometry import Polygon

def query_GEE_for_DEM(AOI, im_path, im_names):
    '''Query GEE for the ASTER Global DEM, clip to the AOI, and return as a numpy array.
    
    Parameters
    ----------
    AOI: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM
    im_path: string
        full path to the directory holding the images to be classified
    im_names: list of strings
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
    # reformat AOI_UTM bounding box as ee.Geometry for clipping DEM
    AOI_WGS_bb_ee = ee.Geometry({"type": "Polygon","coordinates":
                            [[[AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.miny[0]],
                              [AOI_WGS.geometry.bounds.maxx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.maxy[0]],
                              [AOI_WGS.geometry.bounds.minx[0], AOI_WGS.geometry.bounds.miny[0]]]
                            ]})

    # -----Query GEE for ASTER Global DEM, clip to AOI
    DEM = ee.Image("NASA/ASTER_GED/AG100_003").clip(AOI_WGS_bb_ee)

    # -----Grab UTM projection from images, reproject DEM and AOI
    if type(im_names)==str:
        im = rio.open(im_path+im_names)
    else:
        im = rio.open(im_path+im_names[0])
    DEM_UTM = ee.Image.reproject(DEM, str(im.crs))
    AOI_UTM = AOI.to_crs(str(im.crs)[5:])

    # -----Convert DEM to numpy array, extract coordinates
    DEM_np = geemap.ee_to_numpy(DEM, ['elevation'])
    DEM_x = np.linspace(AOI_UTM.geometry.bounds.minx[0], AOI_UTM.geometry.bounds.maxx[0], num=np.shape(DEM_np)[1])
    DEM_y = np.linspace(AOI_UTM.geometry.bounds.miny[0], AOI_UTM.geometry.bounds.maxy[0], num=np.shape(DEM_np)[0])

    # -----Plot to check for success
    fig, ax = plt.subplots(figsize=(8,8))
    plt.rcParams.update({'font.size':14, 'font.sans-serif':'Arial'})
    DEM_im = plt.imshow(DEM_np, cmap='Greens_r',
                        extent=(np.min(DEM_x), np.max(DEM_x), np.min(DEM_y), np.max(DEM_y)))
    AOI_UTM.plot(ax=ax, facecolor='none', edgecolor='black', label='AOI')
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    fig.colorbar(DEM_im, ax=ax, shrink=0.5, label='Elevation [m]')
    plt.show()
    
    return DEM_np, DEM_x, DEM_y, AOI_UTM
    
def crop_images_to_AOI(im_path, im_names, AOI):
    '''
    Crop images to AOI.
    
    Parameters
    ----------
    im_path: str
        path in directory to images
    im_names: str array
        file names of images to crop (str array)
    AOI: geopandas.geodataframe.GeoDataFrame
        cropping region - everything outside the AOI will be masked. Only the exterior bounds used for cropping (no holes). AOI must be in the same CRS as the images.
    
    Returns
    ----------
    cropped_im_path: str
        path in directory to cropped images
    '''
    
    # make folder for cropped images if it does not exist
    cropped_im_path = im_path+'../cropped/'
    if os.path.isdir(cropped_im_path)==0:
        os.mkdir(cropped_im_path)
        print(cropped_im_path+' directory made')
    
    # loop through images
    for im_name in im_names:

        # open image
        im = rio.open(im_path+im_name)

        # check if file exists in directory already
        cropped_im_fn = cropped_im_path + im_name[0:15] + '_crop.tif'
        if os.path.exists(cropped_im_fn)==True:
            print('cropped image already exists in directory...skipping.')
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
            print(cropped_im_fn+' saved')
            
    return cropped_im_path

def plot_im_snow_histograms(im, im_dt, im_x, im_y, snow, snow_elev, b, g, r, nir):
    # grab 10th percentile snow elevation
    iqr = stats.iqr(snow_elev, rng=(10, 90))
    med = np.median(snow_elev)
    P10 = med - iqr/2
    
    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
    plt.rcParams.update({'font.size': 14, 'font.sans-serif': 'Arial'})
    # RGB image
    ax1.imshow(np.dstack([r, g, b]),
                extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    ax1.set_xlabel('Easting [km]')
    ax1.set_ylabel('Northing [km]')
    # snow
    ax2.imshow(np.where(snow==1, 1, np.nan), cmap='Blues', clim=(0,1.5),
                extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    ax2.imshow(np.where(snow==0, 0, np.nan), cmap='Oranges', clim=(-1,2),
    extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    ax2.set_xlabel('Easting [km]')
    # image bands histogram
    h_b = ax3.hist(b[b!=0].flatten(), color='blue', histtype='step', linewidth=2, bins=100, label='blue')
    h_g = ax3.hist(g[g!=0].flatten(), color='green', histtype='step', linewidth=2, bins=100, label='green')
    h_r = ax3.hist(r[r!=0].flatten(), color='red', histtype='step', linewidth=2, bins=100, label='red')
    h_nir = ax3.hist(nir[nir!=0].flatten(), color='brown', histtype='step', linewidth=2, bins=100, label='NIR')
    ax3.set_xlabel('Surface reflectance')
    ax3.set_ylabel('Pixel counts')
    ax3.legend(loc='upper left')
    ax3.set_ylim(0,np.max([h_nir[0][1:], h_g[0][1:], h_r[0][1:], h_b[0][1:]])+5000)
    ax3.grid()
    # snow elevations histogram
    ax4.hist(snow_elev.flatten(), bins=100)
    ax4.set_xlabel('Elevation [m]')
    ax4.grid()
    ymin, ymax = ax4.get_ylim()[0], ax4.get_ylim()[1]
    ax4.plot((P10, P10), (ymin, ymax), '--', color='black', label='P$_{10}$')
    ax4.set_ylim(ymin, ymax)
    ax4.legend(loc='upper left')
    fig.tight_layout()
    fig.suptitle(im_dt)
    plt.show()
    
    return fig

def classify_image(im, im_name, clf, feature_cols, out_path):
    '''
    Function to classify input image using a pre-trained classifier
    
    Parameters
    ----------
    im: rasterio object
        input image
    im_name: str
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
        print('made directory for classified snow images:' + out_path)
            
    # -----Check if classified snow image exists in directory already
    snow_fn = im_name[0:-4]+'_snow.tif'
    if os.path.exists(out_path+snow_fn):
    
        print('classified snow image already exists in directory, loading...')
        s = rio.open(out_path+snow_fn)
        snow = s.read(1).astype(float)
        
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

        # classify snow
        snow_array = clf.predict(df[feature_cols])
        
        # reshape from flat array to original shape
        snow = np.zeros((np.shape(b)[0], np.shape(b)[1]))
        snow[:] = np.nan
        snow[I_real] = snow_array
        
        # replace nan values with -9999 in order to save file with datatype int16
        snow[np.isnan(snow)] = -9999
        
        # save to file
        with rio.open(out_path+snow_fn,'w',
                      driver='GTiff',
                      height=im.shape[0],
                      width=im.shape[1],
                      dtype='int16',
                      count=1,
                      crs=im.crs,
                      transform=im.transform) as dst:
            dst.write(snow, 1)
        print('classified snow image saved to file:',snow_fn)
                
    return im_x, im_y, snow
    
def calculate_SCA(im, snow):
    '''Function to calculated total snow-covered area (SCA) from using an input image and a snow binary mask of the same resolution and grid.
    INPUTS:
        - im: input image ()
        - snow: binary snow mask created from image 
    OUTPUTS:
        - SCA: '''

    pA = im.res[0]*im.res[1] # pixel area [m^2]
    snow_count = np.count_nonzero(snow) # number of snow pixels
    SCA = pA * snow_count # area of snow [m^2]

    return SCA
    
def determine_snow_elevs(DEM, DEM_x, DEM_y, snow, im, im_dt, im_x, im_y, plot_output):
    '''Determine elevations of snow-covered pixels in the classified image.
    Parameters
    ----------
    DEM: numpy array
        digital elevation model
    DEM_x: numpy array
        vector of x coordinates of the DEM
    DEM_y: numpy array
        vector of y coordinates of the DEM
    snow: numpy array
        binary array where snow-covered pixels = 1, non-snow-covered pixels=0 (same shape as image bands)
    im: rasterio object
        input image used to classify snow
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

    # interpolate elevation from DEM at image points
    f = interp2d(DEM_x, DEM_y, DEM)
    im_elev = f(im_x, im_y)
    
    # minimum elevation of the image where data exist
    im_elev_real = np.where((b>0) & (~np.isnan(b)), im_elev, np.nan)
    im_elev_min = np.nanmin(im_elev_real)
    im_elev_max = np.nanmax(im_elev_real)
    
    # extract elevations where snow is present
    snow_elev = im_elev[snow==1]
    
    # plot snow elevations histogram
    if plot_output:
        fig = plot_im_snow_histograms(im, im_dt, im_x, im_y, snow, snow_elev, b, g, r, nir)
        return im_elev_min, im_elev_max, snow_elev, fig
        
    return im_elev_min, im_elev_max, snow_elev
    
    
    

