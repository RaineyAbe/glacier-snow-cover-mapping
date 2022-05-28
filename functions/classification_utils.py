# Functions related to classification of snow in PlanetScope 4-band images
# Rainey Aberle
# 2022

import rasterio as rio
from rasterio.mask import mask
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def crop_images_to_AOI(im_path, im_names, AOI):
    '''Function to crop single image to AOI
    INPUTS:
        - im_path: path to images (str)
        - im_names: file names of images to crop (str array)
        - AOI: cropping region, where everything outside the AOI will be cropped ()
    OUTPUTS:
        - cropped_im_path: path to cropped images (str)'''
    
    # make folder for cropped images if it does not exist
    cropped_im_path = im_path+'../cropped/'
    if os.path.isdir(cropped_im_path)==0:
        os.mkdir(cropped_im_path)
        print(cropped_im_path+' directory made')
    
    # loop through images
    for im_name in im_names:

        # open image
        im = rio.open(im_path+im_name)

        # check if file exists in file already
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


def classify_image(im, clf, feature_cols, plot_output):
    '''Function to classify input image using a pre-trained classifier
    INPUTS:
        - im = input image (rasterio object)
        - clf = previously trained SciKit Learn Classifier (sklearn.classifier)
        - feature_cols = features used by classifier (array of pandas.DataFrame columns, e.g. ['blue', 'green', 'red'])
        - plot_output = logical input, where True = plot RGB image and classified image and False = no plots output
    OUTPUTS:
        - im_x = x coordinates of image (numpy.array)
        - im_y = y coordinates of image (numpy.array)
        - snow = binary array of predicted snow presence in input image, where 0 = no snow and 1 = snow (numpy.array)
        - fig = handle of figure if plot_output==True (matplotlib.pyplot.figure)'''

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
        
    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(r)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(r)[0])
    
    # Find indices of real numbers (no NaNs allowed in classification)
    I_real = np.where(~np.isnan(r))
    
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
    snow = np.zeros((np.shape(r)[0], np.shape(r)[1]))
    snow[:] = np.nan
    snow[I_real] = snow_array
    
    # plot
    if plot_output==True:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'height_ratios': [3, 1]})
        plt.rcParams.update({'font.size': 14, 'font.sans-serif': 'Arial'})
        # RGB image
        ax1.imshow(np.dstack([im.read(3), im.read(2), im.read(1)]),
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
        h_b = ax3.hist(b.flatten(), color='blue', histtype='step', linewidth=2, bins=100, label='blue')
        h_g = ax3.hist(g.flatten(), color='green', histtype='step', linewidth=2, bins=100, label='green')
        h_r = ax3.hist(r.flatten(), color='red', histtype='step', linewidth=2, bins=100, label='red')
        h_nir = ax3.hist(nir.flatten(), color='brown', histtype='step', linewidth=2, bins=100, label='NIR')
        ax3.set_xlabel('Surface reflectance')
        ax3.set_ylabel('Pixel counts')
        ax3.legend(loc='upper right')
        ax3.set_ylim(0,np.max([h_nir[0][1:], h_g[0][1:], h_r[0][1:], h_b[0][1:]])+5000)
        ax3.grid()
        # snow classification histogram
        ax4.hist(snow.flatten())
        ax4.set_xlabel('Snow classification')
        ax4.grid()
#        fig.colorbar(snow_plot, ax=ax2, shrink=0.5)
        fig.tight_layout()
        
        return im_x, im_y, snow, fig
        
    else:
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
    
def determine_min_snow_elevation():

