# PlanetScope_orders_utils.py
# Functions used in the planetAPI_image_download.ipynb notebook
# Modified from Planet Labs notebooks: https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks

from planet import OrdersClient, Session
from shapely import geometry as sgeom
from planet import order_request, reporting
import glob
import requests
import os
import json
from requests.auth import HTTPBasicAuth

def build_quick_search_request(aoi_box_shape, max_cloud_cover, start_date, end_date,
                               item_type, asset_type, auth):
    """
    Compile input filters to create request for Planet API Quick Search

    Parameters
    ----------
    aoi_box_shape
    max_cloud_cover
    start_date
    end_date
    item_type
    asset_type
    auth

    Returns
    -------
    im_ids: list[str]
        list of image IDs resulting from Quick Search
    """

    # -----Build and combine search filters
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": sgeom.mapping(aoi_box_shape)
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
    combined_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    # -----Build request
    request = {
        "item_types": [item_type],
        "asset_types": [asset_type],
        "filter": combined_filter
    }

    # -----Fire off the POST request
    qs_result = \
        requests.post(
            'https://api.planet.com/data/v1/quick-search',
            auth=auth,
            json=request)

    # -----Print number of resulting image IDs
    im_ids = [feature['id'] for feature in qs_result.json()['features']]
    im_ids = sorted(im_ids)
    print(len(im_ids), 'images found')

    return im_ids


def filter_image_ids(im_ids, start_month, end_month, out_path):
    """
    Filter list of image IDs by month range and check if files already exist in out_path

    Parameters
    ----------
    im_ids: list[str]
        list of image IDs
    start_month: int
        start month in the date range (including start_month)
    end_month: int
        end month in the date range (including end_month)
    out_path

    Returns
    -------
    im_ids_filtered: list[str]
        list of image IDs after filtering
    """
    im_ids_filtered = []
    for im_id in im_ids:
        # don't include if image capture month is outside month range
        im_month = int(im_id[4:6])
        if (im_month < start_month) or (im_month > end_month):
            continue
        # don't include if file already exists in directory
        if len(glob.glob(os.path.join(out_path, im_id + '*.tif'))) == 0:
            im_ids_filtered.append(im_id)
    im_ids_filtered = sorted(im_ids_filtered)

    # print number of resulting images and IDs
    print('Number of new images to download = ' + str(len(im_ids_filtered)))
    print('Image IDs:')
    print(im_ids_filtered)

    return im_ids_filtered

def build_request_with_item_ids(base_path, request_name, aoi_box, clip_to_aoi, harmonize, item_ids, item_type, asset_type):
    """
    Build Planet API request for image orders using image IDs

    Parameters
    ----------
    base_path: str
        path in directory to snow-cover-mapping
    request_name: str
        name of the request for monitoring order status
    aoi_box: geojson
        bounding box of the AOI used for Quick Search and (optionally) image cropping
    clip_to_aoi: bool
        whether to clip images to the AOI bounding box before downloading
    harmonize: bool
        whether to harmonize image bands with Sentinel-2 surface reflectance imagery
    item_ids: list[str]
        list of item (image) IDs to download
    item_type: str
        item type to download
    asset_type: str
        asset type to download

    Returns
    -------
    request
    """

    # determine "bundle" using item_type and asset_type
    # see this page for more info: https://developers.planet.com/apis/orders/product-bundles-reference/
    # load Planet bundles dictionary
    bundles_dict_fn = os.path.join(base_path, 'inputs-outputs', 'Planet_bundles.json')
    bundles_dict = json.load(open(bundles_dict_fn))
    # grab all bundles
    bundles = list(bundles_dict['bundles'].keys())
    # iterate over bundles
    bundle_type = None
    for bundle in bundles:
        items_in_bundle = list(bundles_dict['bundles'][bundle]['assets'].keys())
        if item_type in items_in_bundle:
            assets_in_bundle = bundles_dict['bundles'][bundle]['assets'][item_type]
            if asset_type in assets_in_bundle:
                bundle_type = bundle
    if bundle_type is None:
        print(f'No bundle type found associated with specified asset and item types. \nPlease check Planet guides (https://developers.planet.com/apis/orders/product-bundles-reference/) to confirm item and asset types are part of a specific bundle before rerunning.')

    # define the tools
    clip_tool = order_request.clip_tool(aoi_box)
    harmonize_tool = order_request.harmonize_tool("Sentinel-2")

    # compile tools
    tools = []
    if clip_to_aoi:
        tools.append(clip_tool)
    if harmonize:
        tools.append(harmonize_tool)
    # define products to download
    products = [order_request.product(item_ids, bundle_type, item_type)]

    # build request
    request = order_request.build_request(request_name, products=products, tools=tools)

    return request


def place_order_request(auth, request, out_path):
    """
    Compile request, place order, and download files to out_path once ready.

    Parameters
    ----------
    auth
    request
    out_path: str
        output directory for downloads

    Returns
    -------
    None
    """

    # remember: "async def" to create the async coroutine
    async def create_poll_and_download():
        async with Session(auth=auth) as sess:
            cl = OrdersClient(sess)

            # Use "reporting" to manage polling for order status
            with reporting.StateBar(state='creating') as bar:
                # create order via Orders client
                order = await cl.create_order(request)
                bar.update(state='created', order_id=order['id'])

                # poll...poll...poll...
                # setting max_attempts=0 means there is no limit on the number of attempts
                await cl.wait(order['id'], callback=bar.update_state, max_attempts=0)

            # if we get here that means the order completed. Yay! Download the files.
            await cl.download_order(order['id'], directory=out_path)
