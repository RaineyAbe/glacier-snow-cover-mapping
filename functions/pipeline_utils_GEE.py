"""
Functions for running the snow classification pipeline on the Google Earth Engine (GEE) side.
Rainey Aberle
2025
"""

import ee
import geedim as gd
import geemap
import geetools
import numpy as np
import os
import pandas as pd
import wxee as wx
import xarray as xr
from datetime import datetime

current_datetime = datetime.now()
current_datetime_str = str(current_datetime).replace(' ','').replace(':','').replace('-','').replace('.','')

def query_gee_for_dem(aoi):
    """
    Query GEE for digital elevation model (DEM) over study site. If the study site is within the ArcticDEM coverage,
    use the ArcticDEM V3 2m mosaic. Otherwise, use NASADEM.

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest (AOI) to query for DEM.
    
    Returns
    ----------
    dem : ee.Image
        Digital elevation model (DEM) image.
    """
    
    print('\nQuerying GEE for DEM')

    # Determine whether to use ArcticDEM or NASADEM
    # Check for ArcticDEM coverage
    arcticdem_coverage = ee.FeatureCollection('projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/ArcticDEM_Mosaic_coverage')
    intersects = arcticdem_coverage.geometry().intersects(aoi).getInfo()
    if intersects:        
        # make sure there's data (some areas are empty even though they're within the ArcticDEM coverage geometry)
        dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic").clip(aoi)
        dem_area = dem.mask().multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        dem_percent_coverage = ee.Number(dem_area).divide(ee.Number(aoi.area())).multiply(100).getInfo()
        if dem_percent_coverage >= 90:
            dem_name = "ArcticDEM Mosaic"
            dem_string = "UMN/PGC/ArcticDEM/V3/2m_mosaic"
            print('ArcticDEM coverage >= 90%')
        else:
            dem_name = "NASADEM"
            dem_string = "NASA/NASADEM_HGT/001"
            print('ArcticDEM coverage < 90%')
        
    # Otherwise, use NASADEM
    else:
        dem_name = "NASADEM"
        dem_string = "NASA/NASADEM_HGT/001"
        print('No ArcticDEM coverage')
    print(f"Using {dem_name}")

    # Get the DEM, clip to AOI
    dem = ee.Image(dem_string).select('elevation').clip(aoi)
    dem.set({'source': dem_name})

    return dem


def query_gee_for_imagery(dataset, aoi, date_start, date_end, month_start, month_end, fill_portion, mask_clouds):
    """
    Query GEE for imagery over study site. The function will return a collection of pre-processed, clipped images 
    that meet the search criteria. Images captured on the same day will be mosaicked together to increase spatial coverage.

    Parameters
    ----------
    dataset : str
        Name of the dataset to query for. Options are 'Landsat', 'Sentinel-2_SR', or 'Sentinel-2_TOA'.
    aoi : ee.Geometry
        Area of interest (AOI) to query for imagery.
    date_start : str
        Start date for the image search in the format 'YYYY-MM-DD'.
    date_end : str
        End date for the image search in the format 'YYYY-MM-DD'.
    month_start : int
        Start month for the image search (1-12).
    month_end : int
        End month for the image search (1-12).
    fill_portion : float
        Minimum percent coverage of the AOI required for an image to be included in the collection (0-100).
    mask_clouds : bool
        Whether to mask clouds in the imagery. If True, clouds will be masked using the dataset's cloud mask. 
        If False, no cloud masking will be applied.
    
    Returns
    ----------
    im_mosaics : ee.ImageCollection
        Image collection of pre-processed, clipped images that meet the search criteria. 
    """

    print(f'\nQuerying GEE for {dataset} image collection')

    # Define image collection
    if dataset=='Landsat':
        im_col_l8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(date_start, date_end, 
                                                                                    region=aoi, 
                                                                                    mask=mask_clouds).ee_collection
        im_col_l9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(date_start, date_end, 
                                                                                    region=aoi, 
                                                                                    mask=mask_clouds).ee_collection
        im_col = im_col_l8.merge(im_col_l9)
        image_scaler = 1/2.75e-05
        refl_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']
        ndsi_bands = ['SR_B3', 'SR_B6']
        resolution = 30
    elif 'Sentinel-2' in dataset:
        if dataset=='Sentinel-2_SR':
            im_col = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(date_start, date_end, 
                                                                                        region=aoi, 
                                                                                        mask=mask_clouds).ee_collection
        elif dataset=='Sentinel-2_TOA':
            im_col = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(date_start, date_end, 
                                                                                        region=aoi, 
                                                                                        mask=mask_clouds).ee_collection
        image_scaler = 1e4
        refl_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        rgb_bands = ['B4', 'B3', 'B2']
        ndsi_bands = ['B3', 'B11']
        resolution = 10

    # Filter collection by month range
    im_col = im_col.filter(ee.Filter.calendarRange(month_start, month_end, 'month'))

    # Clip to AOI
    def clip_to_aoi(im):
        return im.clip(aoi)
    im_col = im_col.map(clip_to_aoi)

    # Select needed bands
    im_col = im_col.select(refl_bands)

    # Divide by image scaler
    def divide_im_by_scaler(im):
        im_scaled = ee.Image(im.divide(image_scaler)).copyProperties(im, im.propertyNames()) 
        return im_scaled
    im_col = im_col.map(divide_im_by_scaler)

    # Calculate NDSI
    def calculate_ndsi(im):
        ndsi = im.normalizedDifference(ndsi_bands).rename('NDSI')
        im_with_ndsi = im.addBands(ndsi).copyProperties(im, im.propertyNames()) 
        return im_with_ndsi
    im_col = im_col.map(calculate_ndsi)

    # Mosaic images captured the same day to increase spatial coverage
    def make_daily_mosaics(collection):
        # modified from: https://gis.stackexchange.com/questions/280156/mosaicking-image-collection-by-date-day-in-google-earth-engine
        # Get the list of unique dates in the collection
        date_list = (collection.aggregate_array('system:time_start')
                    .map(lambda ts: ee.Date(ts).format('YYYY-MM-dd')))
        date_list = ee.List(date_list.distinct())

        def day_mosaics(date, new_list):
            date = ee.Date.parse('YYYY-MM-dd', date)
            new_list = ee.List(new_list)
            filtered = collection.filterDate(date, date.advance(1, 'day'))
            image = ee.Image(filtered.mosaic()).set('system:time_start', date.millis())
            return ee.List(ee.Algorithms.If(filtered.size(), new_list.add(image), new_list))

        return ee.ImageCollection(ee.List(date_list.iterate(day_mosaics, ee.List([]))))
    # call the function
    im_mosaics = make_daily_mosaics(im_col)

    # Filter by percent coverage of the AOI
    def calculate_percent_aoi_coverage(im):
        # Count total pixels in the AOI (use one band, first RGB band arbitrarily)
        pixel_count = im.select(rgb_bands[0]).unmask().reduceRegion(
            reducer = ee.Reducer.count(),
            geometry = aoi,
            scale = resolution,
            maxPixels = 1e9,
            bestEffort = True
        ).get(rgb_bands[0])

        # Count unmasked pixels in the AOI (use one band, first RGB band arbitrarily)
        unmasked_pixel_count = im.select(rgb_bands[0]).reduceRegion(
            reducer = ee.Reducer.count(),
            geometry = aoi,
            scale = resolution,
            maxPixels = 1e9,
            bestEffort = True
        ).get(rgb_bands[0])

        # Calculate percent coverage
        percent_coverage = ee.Number(pixel_count).divide(unmasked_pixel_count).multiply(100)

        return im.copyProperties(im, im.propertyNames()).set({'percent_AOI_coverage': percent_coverage})
    
    im_mosaics = im_mosaics.map(calculate_percent_aoi_coverage)
    im_mosaics = im_mosaics.filter(ee.Filter.gte('percent_AOI_coverage', fill_portion))

    return ee.ImageCollection(im_mosaics)


def classify_image_collection(collection, dataset):
    """
    Classify the image collection using a pre-trained classifier. The classifier is trained on a set of training data
    that is specific to the dataset. 

    Parameters
    ----------
    collection : ee.ImageCollection
        Image collection to classify.
    dataset : str
        Name of the dataset used for classification. Options are 'Landsat', 'Sentinel-2_SR', or 'Sentinel-2_TOA'.

    Returns
    ----------
    classified_collection : ee.ImageCollection
        Classified image collection.
    """

    print('\nClassifying image collection')

    # Retrain classifier
    if dataset=='Landsat':
        clf = ee.Classifier.smileKNN(3)
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Landsat_training_data")
        feature_cols = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'NDSI']
    elif dataset=='Sentinel-2_SR':
        clf = ee.Classifier.libsvm()
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Sentinel-2_SR_training_data")
        feature_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDSI']
    elif dataset=='Sentinel-2_TOA':
        clf = ee.Classifier.libsvm()
        training_data = ee.FeatureCollection("projects/ee-raineyaberle/assets/glacier-snow-cover-mapping/Sentinel-2_TOA_training_data")
        feature_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDSI']
    clf = clf.train(training_data, 'class', feature_cols)

    # Classify collection
    def classify(im):
        return im.classify(clf).copyProperties(im, im.propertyNames())
    classified_collection = collection.map(classify)

    return ee.ImageCollection(classified_collection)


def calculate_snow_cover_statistics(image_collection, dem, aoi, scale=30, 
                                    out_folder='snow_cover_exports', 
                                    file_name_prefix=f'snow_cover_stats_{current_datetime_str}'):
    """
    Calculate snow cover statistics for each image in the collection. The function will calculate the following
    statistics for each image: snow area, ice area, rock area, water area, glacier area, transient AAR, SLA,
    SLA upper bound, and SLA lower bound. 

    Parameters
    ----------
    image_collection : ee.ImageCollection
        Image collection to calculate statistics for.
    dem : ee.Image
        Digital elevation model (DEM) image to use for SLA calculations.
    aoi : ee.Geometry
        Area of interest (AOI) to calculate statistics for.
    scale : int
        Scale to use for calculations (default is 30m).
    out_folder: str
        Name of Google Drive Folder where statistics will be saved as CSV.
    file_name_prefix: str
        Prefix for output file name.
    
    Returns
    ----------
    statistics : ee.FeatureCollection
        Feature collection of snow cover statistics for each image in the collection.
    """

    print('\nCalculating snow cover statistics')

    def process_image(image):
        # Grab the image date
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')

        # Create masks for each class
        snow_mask = image.eq(1).Or(image.eq(2))
        ice_mask = image.eq(3)
        rock_mask = image.eq(4)
        water_mask = image.eq(5)

        # Calculate areas of each mask
        def calculate_class_area(mask):
            return mask.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9,
                bestEffort=True
            ).get('classification')
        snow_area = calculate_class_area(snow_mask)
        ice_area = calculate_class_area(ice_mask)
        rock_area = calculate_class_area(rock_mask)
        water_area = calculate_class_area(water_mask)

        # Calculate glacier area (snow + ice area)
        glacier_area = ee.Number(snow_area).add(ee.Number(ice_area))

        # Calculate transient AAR (snow area / glacier area)
        transient_aar = ee.Number(snow_area).divide(ee.Number(glacier_area))

        # Estimate snowline altitude (SLA) using the transient AAR and the DEM
        sla_percentile = (ee.Number(1).subtract(ee.Number(transient_aar)))
        sla = dem.reduceRegion(
            reducer=ee.Reducer.percentile(ee.List([ee.Number(sla_percentile).multiply(100).toInt()])),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')

        # Estimate upper and lower bounds for the SLA
        # upper bound: snow-free pixels above the SLA
        snow_free_mask = image.eq(3).Or(image.eq(4)).Or(image.eq(5))
        above_sla_mask = dem.gt(ee.Number(sla))
        upper_mask = snow_free_mask.And(above_sla_mask)
        upper_mask_area = upper_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_upper_percentile = (ee.Number(sla_percentile)
                                .add(ee.Number(upper_mask_area)
                                     .divide(ee.Number(aoi.area()))))
        sla_upper = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_upper_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        # lower bound: snow-covered pixels below the SLA
        below_sla_mask = dem.lt(ee.Number(sla))
        lower_mask = snow_mask.And(below_sla_mask)
        lower_mask_area = lower_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get('classification')
        sla_lower_percentile = (ee.Number(sla_percentile)
                                .subtract(ee.Number(lower_mask_area)
                                          .divide(ee.Number(aoi.area()))))
        sla_lower = dem.reduceRegion(
            reducer=ee.Reducer.percentile([ee.Number(sla_lower_percentile).multiply(100).toInt()]),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            bestEffort=True
            ).get('elevation')
        
        # Return feature
        feature = ee.Feature(None, {
            'date': date,
            'snow_area_m2': snow_area,
            'ice_area_m2': ice_area,
            'rock_area_m2': rock_area,
            'water_area_m2': water_area,
            'glacier_area_m2': glacier_area,
            'transient_AAR': transient_aar,
            'SLA_m': sla,
            'SLA_upper_bound_m': sla_upper,
            'SLA_lower_bound_m': sla_lower 
        })

        return feature
    
    # Process each image in the collection
    statistics = ee.FeatureCollection(image_collection.map(process_image))

    # Export to Google Drive folder
    task = ee.batch.Export.table.toDrive(
        collection=statistics, 
        description=file_name_prefix, 
        folder=out_folder, 
        fileNamePrefix=file_name_prefix, 
        fileFormat='CSV', 
        )
    task.start()
    print('Exporting snow cover statistics to Google Drive folder with description:', file_name_prefix)
    print('To monitor tasks, go to your GEE Task Manager: https://code.earthengine.google.com/tasks')

    return statistics
