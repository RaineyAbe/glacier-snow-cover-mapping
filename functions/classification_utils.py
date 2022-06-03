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
        cropping region - everything outside the AOI will be masked
    
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
            # mask image pixels outside the AOI
            AOI_bb = [AOI.bounds]
            out_image, out_transform = mask(im, list(AOI.geometry), crop=True)
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
#        fig.colorbar(snow_plot, ax=ax2, shrink=0.5)
    fig.tight_layout()
    fig.suptitle(im_dt)
    plt.show()

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
        snow = s.read(1)
        
    else:
    
        # define bands
        b = im.read(1).astype(float)
        g = im.read(2).astype(float)
        r = im.read(3).astype(float)
        nir = im.read(4).astype(float)
        # check if bands must be divided by scalar
        if (np.nanmean(b) > 1000):
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
        
        # save to file
        with rio.open(out_path+snow_fn,'w',
                      driver='GTiff',
                      height=im.shape[0],
                      width=im.shape[1],
                      dtype=snow.dtype,
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
    
def determine_snow_elevs(DEM, snow, im, im_dt, im_x, im_y, plot_output):

    # extract DEM info
    DEM_x = np.linspace(DEM.bounds.left, DEM.bounds.right, num=np.shape(DEM)[1])
    DEM_y = np.flipud(np.linspace(DEM.bounds.top, DEM.bounds.bottom, num=np.shape(DEM)[0]))
    DEM_elev = DEM.read(1)
    
    # extract one band info
    b = im.read(1)
    g = im.read(2)
    r = im.read(3)
    nir = im.read(4)
    
    # interpolate elevation from DEM at image points
    f = interp2d(DEM_x, DEM_y, DEM_elev)
    im_elev = f(im_x, im_y)
    
    # minimum elevation of the image where data exist
    im_elev_real = np.where((b>0) & (~np.isnan(b)), im_elev, np.nan)
    im_elev_min = np.nanmin(im_elev_real)
    
    # extract elevations where snow is present
    snow_elev = im_elev[snow==1]
    
    # plot snow elevations histogram
    if plot_output:
        plot_im_snow_histograms(im, im_dt, im_x, im_y, snow, snow_elev, b, g, r, nir)
    
    return snow_elev
    
# -----function to convert Pandas gdf to ee.FeatureCollection
# from: https://bikeshbade.com.np/tutorials/Detail/?title=Geo-pandas%20data%20frame%20to%20GEE%20feature%20collection%20using%20Python&code=13
#def feature2ee(gdf):
#    g = [i for i in gdf.geometry]
#    features=[]
#    #for Point geo data type
#    if (gdf.geom_type[0] == 'Point'):
#        for i in range(len(g)):
#            g = [i for i in gdf.geometry]
#            x,y = g[i].coords.xy
#            cords = np.dstack((x,y)).tolist()
#            double_list = reduce(lambda x,y: x+y, cords)
#            single_list = reduce(lambda x,y: x+y, double_list)
#            g=ee.Geometry.Point(single_list)
#            feature = ee.Feature(g)
#            features.append(feature)
#        ee_object = ee.FeatureCollection(features)
#        return ee_object
        
def sample_dem_at_im_coords(im_x, im_y, DEM):

    # convert image coordinates to ee.FeatureCollection
    im_x_mesh, im_y_mesh = np.meshgrid(im_x, im_y)
    coords = zip(im_x_mesh, im_y_mesh)
    
    # convert DEM to numpy array
#    DEM_np = geemap.ee_to_numpy(DEM)
    
    # interpolate DEM at image coordinates
    
    
    

