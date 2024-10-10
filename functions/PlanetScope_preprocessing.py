"""
Functions for preprocessing PlanetScope 4-band surface reflectance imagery
Rainey Aberle
2023
"""

import os
import rioxarray as rxr
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import subprocess
import rasterio as rio
from scipy.interpolate import interp1d
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon
import xarray as xr


def planetscope_mask_image_pixels(im_path, im_fn, out_path, save_outputs=True, plot_results=True):
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
        im_mask.rio.to_raster(os.path.join(out_path, im_mask_fn), dtype='int32')

    # -----Plot results
    if plot_results:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(np.dstack([im.data[2], im.data[1], im.data[0]]))
        # set no data values to NaN, divide my im_scalar for plotting
        im_mask = im_mask.where(im_mask != -9999) / im_scalar
        ax[1].imshow(np.dstack([im_mask.data[2], im_mask.data[1], im_mask.data[0]]))
        plt.show()

    return


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
    dem_aoi = dem.rio.clip(aoi.geometry.values, aoi.crs)
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
                        file_paths.append(os.path.join(im_path_adj, im_fn))  # add the path to the file

            # check if any filepaths were added
            if len(file_paths) > 0:

                # construct the gdal_merge command
                cmd = 'gdal_merge.py -v -n -9999 -a_nodata -9999 '

                # add input files to command
                for file_path in file_paths:
                    cmd += file_path + ' '

                cmd += '-o ' + os.path.join(out_path_adj, out_im_fn)

                # run the command
                subprocess.run(cmd, shell=True, capture_output=True)