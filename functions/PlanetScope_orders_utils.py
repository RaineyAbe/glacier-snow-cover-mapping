# orders.py
# Functions used in the planetAPI_image_download.ipynb notebook
# Modified from Planet Labs notebooks: https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks

from planet.api import filters
from planet import api
import os
from shapely import geometry as sgeom
import shapely
import pyproj
from functools import partial
import requests
import json
import time
import rasterio as rio
from rasterio import features as rfeatures
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


def build_QS_request(AOI_shape, max_cloud_cover, start_date, end_date,
                  item_type, asset_type):
    '''Compile input filters to create request for Planet API search'''
    
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": sgeom.mapping(AOI_shape)
    }

    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date + "T00:00:00.000Z",
        "lte": end_date + "T00:00:00.000Z"
      }
    }

    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": max_cloud_cover
      }
    }
    
#    asset_filter = []
#    for asset_type in asset_types:
#        af = {
#            "type": "AssetFilter",
#            "config": asset_type
#        }
#        asset_filter = asset_filter + [af]
#
    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    request = {
        "item_types": [item_type],
        "asset_types": [asset_type],
        "filter": combined_filter
    }
            
    return request

def build_request_itemIDs(AOI_box, clip_AOI, harmonize, im_ids, item_type, asset_type):
    '''Build Planet API request for image orders with image IDs'''
    
    # define the clip and harmonize tools
    clip_tool = {"clip": {"aoi": AOI_box}}
    harmonize_tool = {"harmonize": {"target_sensor": "Sentinel-2"}}
    
    # determine which product bundle to use
    if (item_type=="PSScene") & ("sr" in asset_type):
        product_bundle = "analytic_sr_udm2" # 4-band surface reflectance products with UDM2 mask
#    elif (item_type=="PSScene") & ("udm2" in asset_type):
#        product_bundle = "analytic_udm2" # 4-band top of atomsphere reflectance products with UDM2 mask
#    elif (item_type=="PSScene") & ("sr" in asset_type):
#        product_bundle = "analytic_sr" # 4-band surface reflectance products with any mask
    else:
        print("Error in product bundle selection, exiting...")
        return "N/A"
    
    # create request object depending on settings
    if (clip_AOI==True) & (harmonize==True):
        request = {
           "name":"simple order",
           "products":[
              {
                  "item_ids": im_ids,
                  "item_type": item_type,
                  "product_bundle": product_bundle
              }
           ],
            "tools": [clip_tool, harmonize_tool]
        }
    elif (clip_AOI==True) & (harmonize==False):
        request = {
           "name":"simple order",
           "products":[
              {
                  "item_ids": im_ids,
                  "item_type": item_type,
                  "product_bundle": product_bundle
#                  "asset_type": asset_types
              }
           ],
            "tools": [clip_tool]
        }
    elif (clip_AOI==False) & (harmonize==True):
        request = {
           "name":"simple order",
           "products":[
              {
                  "item_ids": im_ids,
                  "item_type": item_type,
                  "product_bundle": product_bundle
              }
           ],
            "tools": [harmonize_tool]
        }
    elif (clip_AOI==False) & (harmonize==False):
        request = {
           "name":"simple order",
           "products":[
              {
                  "item_ids": im_ids,
                  "item_type": item_type,
                  "product_bundle": product_bundle
              }
           ]
        }
    return request

def search_pl_api(request, limit):
    '''Search Planet API using request'''
    client = api.ClientV1(api_key=os.environ['PL_API_KEY'])
    result = client.quick_search(request)
    
    # note that this returns a generator
    return result.items_iter(limit=limit)
    
def place_order(orders_url, search_request, auth):
    # set content type to json
    headers = {'content-type': 'application/json'}
    
    response = requests.post(orders_url, data=json.dumps(search_request), auth=auth, headers=headers)
    print(response.json())
    order_id = response.json()['id']
    print(order_id)
    order_url = orders_url + '/' + order_id
    return order_url
    
def poll_for_success(order_url, auth, num_loops=1e10):
    count = 0
    while(count < num_loops):
        count += 1
        r = requests.get(order_url, auth=auth)
        response = r.json()
        state = response['state']
        print(state)
        end_states = ['success', 'failed', 'partial']
        if state in end_states:
            break
        time.sleep(10)
        
def download_results(results, out_folder, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    print('{} items to download'.format(len(results_urls)))
    
    count = 0 # count for downloaded files
    for url, name in tqdm(list(zip(results_urls, results_names))):
        path = Path(os.path.join(out_folder,name))
        if overwrite or not path.exists():
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping {}'.format(path, name))
        
        count+=1
    print('Done!')

def get_utm_projection_fcn(shape):
    # define projection
    # from shapely [docs](http://toblerity.org/shapely/manual.html#shapely.ops.transform)
    proj_fcn = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'), #wgs84
        _get_utm_projection(shape))
    return proj_fcn

def _get_utm_zone(shape):
    '''geom: geojson geometry'''
    centroid = shape.centroid
    lon = centroid.x
    lat = centroid.y
    
    if lat > 84 or lat < -80:
        raise Exception('UTM Zones only valid within [-80, 84] latitude')
    
    # this is adapted from
    # https://www.e-education.psu.edu/natureofgeoinfo/book/export/html/1696
    zone = int((lon + 180) / 6 + 1)
    
    hemisphere = 'north' if lat > 0 else 'south'
    
    return (zone, hemisphere)

def _get_utm_projection(shape):
    zone, hemisphere = _get_utm_zone(shape)
#    proj_str = "+proj=utm +zone={zone}, +{hemi} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(
#        zone=zone, hemi=hemisphere)
    return pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
    
def get_overlap_shapes_utm(items, AOI_shape, epsg):
    '''Determine overlap between item footprint and AOI in UTM.'''
   
    proj_fcn = partial(pyproj.transform)
    AOI_shape_utm = shapely.ops.transform(proj_fcn, AOI_shape)

    def calculate_overlap(item):
        footprint_shape = sgeom.shape(item['geometry'])
        footprint_shape_utm = shapely.ops.transform(proj_fcn, footprint_shape)
        return AOI_shape.intersection(footprint_shape_utm)

    for i in items:
        yield calculate_overlap(i)

def get_coverage_dimensions(aoi_shape_utm):
    '''Checks that aoi is big enough and calculates the dimensions for coverage grid.'''
    minx, miny, maxx, maxy = aoi_shape_utm.bounds
    width = maxx - minx
    height = maxy - miny
    
    min_cell_size = 9 # in meters, approx 3x ground sampling distance
    min_number_of_cells = 3
    max_number_of_cells = 3000
    
    
    min_dim = min_cell_size * min_number_of_cells
    if height < min_dim:
        raise Exception('AOI height too small, should be {}m.'.format(min_dim))

    if width < min_dim:
        raise Exception('AOI width too small, should be {}m.'.format(min_dim))
    
    def _dim(length):
        return min(int(length/min_cell_size), max_number_of_cells)

    return [_dim(l) for l in (height, width)]

def get_overlap_shapes_utm(items, aoi_shape):
    '''Determine overlap between item footprint and AOI in UTM.'''
    
    proj_fcn = get_utm_projection_fcn(aoi_shape)
    aoi_shape_utm = shapely.ops.transform(proj_fcn, aoi_shape)

    def _calculate_overlap(item):
        footprint_shape = sgeom.shape(item['geometry'])
        footprint_shape_utm = shapely.ops.transform(proj_fcn, footprint_shape)
        return aoi_shape_utm.intersection(footprint_shape_utm)

    for i in items:
        yield _calculate_overlap(i)

#def filter_by_coverage(overlaps, dimensions, bounds):
    

def calculate_coverage(overlaps, dimensions, bounds):
    
    # get dimensions of coverage raster
    mminx, mminy, mmaxx, mmaxy = bounds

    y_count, x_count = dimensions
    
    # determine pixel width and height for transform
    width = (mmaxx - mminx) / x_count
    height = (mminy - mmaxy) / y_count # should be negative

    # Affine(a, b, c, d, e, f) where:
    # a = width of a pixel
    # b = row rotation (typically zero)
    # c = x-coordinate of the upper-left corner of the upper-left pixel
    # d = column rotation (typically zero)
    # e = height of a pixel (typically negative)
    # f = y-coordinate of the of the upper-left corner of the upper-left pixel
    # ref: http://www.perrygeo.com/python-affine-transforms.html
    transform = rio.Affine(width, 0, mminx, 0, height, mmaxy)
    
    coverage = np.zeros(dimensions, dtype=np.uint16)
    for overlap in overlaps:
        if not overlap.is_empty:
            # rasterize overlap vector, transforming to coverage raster
            # pixels inside overlap have a value of 1, others have a value of 0
            overlap_raster = rfeatures.rasterize(
                    [sgeom.mapping(overlap)],
                    fill=0,
                    default_value=1,
                    out_shape=dimensions,
                    transform=transform)
            
            # add overlap raster to coverage raster
            coverage += overlap_raster
    return coverage
    
def plot_coverage(coverage):
    fig, ax = plt.subplots()
    cax = ax.imshow(coverage, interpolation='nearest', cmap=cm.viridis)
    ax.set_title('Coverage\n(median: {})'.format(int(np.median(coverage))))
    ax.axis('off')
    
    ticks_min = coverage.min()
    ticks_max = coverage.max()
    cbar = fig.colorbar(cax,ticks=[ticks_min, ticks_max])
