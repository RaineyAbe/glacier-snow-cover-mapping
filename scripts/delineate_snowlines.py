# Delineate snowlines
# Must already have classified images available in file

# -----Import packages
import glob
import os
import pandas as pd
import geopandas as gpd
import ee
import matplotlib.pyplot as plt
import sys
from shapely.geometry import LineString, MultiLineString, Point
import json
import xarray as xr
import rioxarray as rxr
import numpy as np
from tqdm.auto import tqdm
from scipy.ndimage import binary_fill_holes, binary_dilation
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import matplotlib
import geedim as gd
import requests
from PIL import Image
import io

# -----Define paths in directory
site_name = 'Gulkana'
# path to snow-cover-mapping/
base_path = '/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/'
# path to folder containing AOI files
AOI_path = '/Users/raineyaberle/Google Drive/My Drive/Research/PhD/snow_cover_mapping/snow_cover_mapping_application/study-sites/' + site_name + '/AOIs/'
# AOI file name
AOI_fn = glob.glob(AOI_path + site_name + '*USGS*outline*.shp')[0]
# path to classified images
im_classified_path = AOI_path + '../imagery/classified/'
# path for output snowlines
snowlines_path = AOI_path + '../imagery/snowlines/'
# path to PlanetScope image mosaics
# Note: set PS_im_path=None if not using PlanetScope
PS_im_path = AOI_path + '../imagery/PlanetScope/mosaics/'
# path for output figures
figures_out_path = AOI_path + '../figures/'

# -----Add path to functions
sys.path.insert(1, base_path+'functions/')
import pipeline_utils as f

# -----Load dataset dictionary
dataset_dict = json.load(open(base_path + 'inputs-outputs/datasets_characteristics.json'))

# -----Initialize Google Earth Engine account
print('Initializing Google Earth Engine account...')
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
except:
    ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

# -----Load AOI as gpd.GeoDataFrame
print('Loading AOI from file...')
AOI = gpd.read_file(AOI_fn)
# reproject the AOI to WGS to solve for the optimal UTM zone
AOI_WGS = AOI.to_crs('EPSG:4326')
AOI_WGS_centroid = [AOI_WGS.geometry[0].centroid.xy[0][0],
                    AOI_WGS.geometry[0].centroid.xy[1][0]]
# grab the optimal UTM zone EPSG code
epsg_UTM = f.convert_wgs_to_utm(AOI_WGS_centroid[0], AOI_WGS_centroid[1])
print('Optimal UTM CRS = EPSG:' + str(epsg_UTM))
# reproject AOI to the optimal UTM zone
AOI_UTM = AOI.to_crs('EPSG:' + epsg_UTM)

# Define necessary functions

# Define necessary functions

def delineate_snowline(im_classified, site_name, aoi, dataset_dict, dataset, im_date, snowline_fn,
                       out_path, figures_out_path, plot_results, verbose=False):
    # -----Make directory for snowlines (if it does not already exist)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        print('Made directory for snowlines: ' + out_path)

    # -----Make directory for figures (if it does not already exist)
    if (not os.path.exists(figures_out_path)) & plot_results:
        os.mkdir(figures_out_path)
        print('Made directory for output figures: ' + figures_out_path)

    # -----Subset dataset_dict to dataset
    ds_dict = dataset_dict[dataset]

    # -----Remove time dimension
    im_classified = im_classified.isel(time=0)

    # -----Create no data mask
    no_data_mask = xr.where(np.isnan(im_classified), 1, 0).classified.data
    # dilate by ~30 m
    iterations = int(np.round(30 / ds_dict['resolution_m']))  # number of pixels equal to 30 m
    dilated_mask = binary_dilation(no_data_mask, iterations=iterations)
    no_data_mask = np.logical_not(dilated_mask)
    # add no_data_mask variable classified image
    im_classified = im_classified.assign(no_data_mask=(["y", "x"], no_data_mask))

    # -----Determine snow covered elevations
    all_elev = np.ravel(im_classified.elevation.data)
    all_elev = all_elev[~np.isnan(all_elev)]  # remove NaNs
    snow_est_elev = np.ravel(im_classified.where((im_classified.classified <= 2))
                             .where(im_classified.classified != -9999).elevation.data)
    snow_est_elev = snow_est_elev[~np.isnan(snow_est_elev)]  # remove NaNs

    # -----Create elevation histograms
    # determine bins to use in histograms
    elev_min = np.fix(np.nanmin(np.ravel(im_classified.elevation.data)) / 10) * 10
    elev_max = np.round(np.nanmax(np.ravel(im_classified.elevation.data)) / 10) * 10
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
        # create a binary mask for everything above the first instance of the snow-covered percentage threshold
        sca_perc_threshold = 0.1
        if np.any(hist_snow_est_elev_norm > sca_perc_threshold):
            elev_thresh_snow = bin_centers[np.argmax(hist_snow_est_elev_norm > sca_perc_threshold)]
            elevation_threshold_mask = xr.where(im_classified.elevation > elev_thresh_snow, 1, 0)
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
                                'CRS': ['EPSG:' + str(aoi.crs.to_epsg())],
                                'snowline_elevs_m': [snowline_elevs],
                                'snowline_elevs_median_m': [median_snowline_elev],
                                'SCA_m2': [sca],
                                'AAR': [aar],
                                'dataset': [dataset],
                                'geometry': [snowline]
                                })

    # -----Save snowline df to file
    # reduce memory storage of dataframe
    snowline_df = f.reduce_memory_usage(snowline_df, verbose=False)
    # save using user-specified file extension
    if 'pkl' in snowline_fn:
        snowline_df.to_pickle(out_path + snowline_fn)
        if verbose:
            print('Snowline saved to file: ' + out_path + snowline_fn)
    elif 'csv' in snowline_fn:
        snowline_df.to_csv(out_path + snowline_fn, index=False)
        if verbose:
            print('Snowline saved to file: ' + out_path + snowline_fn)
    else:
        print('Please specify snowline_fn with extension .pkl or .csv. Exiting...')
        return 'N/A'

    # -----Plot results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    # define x and y limits
    xmin, xmax = aoi.geometry[0].buffer(100).bounds[0] / 1e3, aoi.geometry[0].buffer(100).bounds[2] / 1e3
    ymin, ymax = aoi.geometry[0].buffer(100).bounds[1] / 1e3, aoi.geometry[0].buffer(100).bounds[3] / 1e3
    # define colors for plotting
    colors = list(dataset_dict['classified_image']['class_colors'].values())
    cmp = matplotlib.colors.ListedColormap(colors)
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
    # normalized snow elevations histogram
    ax[2].bar(bin_centers, hist_snow_est_elev_norm, width=(bin_centers[1] - bin_centers[0]), color=colors[0],
              align='center')
    ax[2].plot([median_snowline_elev, median_snowline_elev], [0, 1], '-', color='#f768a1',
               linewidth=3, label='Median snowline elevation')
    ax[2].set_xlabel("Elevation [m]")
    ax[2].set_ylabel("Fraction snow-covered")
    ax[2].grid()
    ax[2].set_xlim(elev_min - 10, elev_max + 10)
    ax[2].set_ylim(0, 1)
    # determine figure title and file name
    title = im_date.replace('-', '').replace(':', '') + '_' + site_name + '_' + dataset + '_snow-cover'
    # add legends
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')
    ax[2].legend(loc='lower right')
    fig.suptitle(title)
    fig.tight_layout()

    return snowline_df, fig, ax, title


def query_gee_for_image_thumbnail(dataset, dt, aoi_utm):
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
            optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

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
    image = Image.open(image_bytes)

    return image, bounds

# -----Load classified image file names
im_classified_fns = sorted([os.path.basename(x) for x in glob.glob(im_classified_path +'*.nc')])

# -----Iterate over classified image file names
for im_classified_fn in tqdm(im_classified_fns[1:]):

    # load classified image
    im_classified = xr.open_dataset(im_classified_path + im_classified_fn)
    # remove no data values
    im_classified = xr.where(im_classified == -9999, np.nan, im_classified)

    # determine image date and dataset
    im_date = im_classified_fn[0:15]
    im_dt = np.datetime64(im_date[0:4] + '-' + im_date[4:6] + '-' + im_date[6:8])
    dataset = im_classified_fn.split(site_name + '_')[1].split('_classified')[0]
    print(str(im_dt) + ' ' + dataset)

    # check whether snowline exists in file
    snowline_fn = im_date + '_' + site_name + '_' + dataset + '_snowline.csv'
    if os.path.exists(snowlines_path + snowline_fn):
        print('Snowline already exists in file, continuing...')
        print(' ')
        continue # no need to delineate snowline if it already exists

    # delineate snowline and set up figure
    snowline_df, fig, ax, title = delineate_snowline(im_classified, site_name, AOI_UTM, dataset_dict, dataset, im_date, snowline_fn,
                                                     snowlines_path, figures_out_path, plot_results=True, verbose=True)
    print('Accumulation Area Ratio =  ' + str(snowline_df['AAR'][0]))

    # load image from file
    if dataset == 'PlanetScope':
        try:
            im_fn = glob.glob(PS_im_path + '*' + str(im_date)[0:8] + '*.tif')[0]
            im = rxr.open_rasterio(im_fn)
            im = xr.where(im != -9999, im / 1e4, np.nan)
            ax[0].imshow(np.dstack([im.data[2], im.data[1], im.data[0]]),
                        extent=(np.min(im.x.data) / 1e3, np.max(im.x.data) / 1e3,
                                np.min(im.y.data) / 1e3, np.max(im.y.data) / 1e3))
            # save figure
            fig_fn = figures_out_path + title + '.png'
            fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
            print('Figure saved to file: ' + fig_fn)
            plt.close()
            print(' ')
        except Exception as e:
            print(e)
            print(' ')
            plt.close()
            continue
    # otherwise, load image thumbnail from GEE
    else:
        try:
            # get PIL image object
            im, bounds = query_gee_for_image_thumbnail(dataset, im_dt, AOI_UTM)
            # plot RGB image
            ax[0].imshow(im, extent=(bounds[0] / 1e3, bounds[2] / 1e3, bounds[1] / 1e3, bounds[3] / 1e3))
            # save figure
            fig_fn = figures_out_path + title + '.png'
            fig.savefig(fig_fn, dpi=300, facecolor='white', edgecolor='none')
            print('Figure saved to file: ' + fig_fn)
            plt.close()
            print(' ')
        except Exception as e:
            print(e)
            print(' ')
            plt.close()
            continue
