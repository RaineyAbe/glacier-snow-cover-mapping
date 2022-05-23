# Functions required for the calculate_snow_covered_area.ipynb notebook
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
    '''Function to crop single image to AOI'''
    
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
            print('cropped image already exists in directory')
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


def classify_image(im, clf):
    '''Function to classify input image usinga pre-trained classifier
    INPUTS:
        - im = input image (rasterio object)
        - clf = previously trained SciKit Learn Classifier (sklearn.classifier)
        - plot = logical input, where True = plot RGB image and classified image and False = no plots output
    OUTPUTS:
        - snow = binary array of predicted snow presence in input image, where 0 = no snow and 1 = snow (numpy.array)'''

    # define bands
    b = im.read(1).astype(float)
    g = im.read(2).astype(float)
    r = im.read(3).astype(float)
    nir = im.read(4).astype(float)
    ndsi = (r - nir) / (r + nir)
    # replace no data values with NaN
    b[b==0] = np.nan
    g[g==0] = np.nan
    r[r==0] = np.nan
    nir[r==0] = np.nan
    ndsi[r==0] = np.nan
    
    # define coordinates grid
    im_x = np.linspace(im.bounds.left, im.bounds.right, num=np.shape(r)[1])
    im_y = np.linspace(im.bounds.top, im.bounds.bottom, num=np.shape(r)[0])
    
    # Find indices of real numbers (no NaNs allowed in classification)
    Ir_real = np.where(~np.isnan(r))
    Inir_real = np.where(~np.isnan(nir))
    Indsi_real =np.where(~np.isnan(ndsi))
    
    # save in Pandas dataframe
    df = pd.DataFrame()
    df['red'] = r[Ir_real].flatten()
    df['NIR'] = nir[Ir_real].flatten()
    df['NDSI'] = ndsi[Ir_real].flatten()

    # classify snow
    snow_array = clf.predict(df[['red', 'NIR', 'NDSI']])
    
    # reshape from flat array to original shape
    snow = np.zeros((np.shape(r)[0], np.shape(r)[1]))
    snow[:] = np.nan
    snow[Ir_real] = snow_array
    
    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10), gridspec_kw={'height_ratios': [3, 1]})
    plt.rcParams.update({'font.size': 14, 'font.sans-serif': 'Arial'})
    # RGB image
    ax1.imshow(np.dstack([im.read(3), im.read(2), im.read(1)]),
                extent=(np.min(im_x)/1000, np.max(im_x)/1000, np.min(im_y)/1000, np.max(im_y)/1000))
    ax1.set_xlabel('Easting [km]')
    ax1.set_ylabel('Northing [km]')
    # snow
    snow_plot = ax2.imshow(np.where(snow==1, snow, np.nan), cmap='Blues', clim=(0,1),
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
    # snow classification histogram
    h_snow = ax4.hist(snow.flatten())
    ax4.set_xlabel('Snow classification')
    fig.colorbar(snow_plot, ax=ax2, shrink=0.5)
    fig.tight_layout()
    
    return im_x, im_y, snow, fig
    

def calculate_SCA(im, snow):

    '''Function to calculated total snow-covered area (SCA) from using an input image and a snow binary mask of the same resolution and grid'''

    pA = im.res[0]*im.res[1] # pixel area [m^2]
    snow_count = np.count_nonzero(snow) # number of snow pixels
    SCA = pA * snow_count # area of snow [m^2]

    return SCA
