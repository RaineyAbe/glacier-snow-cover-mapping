# orders.py
# Functions used in the planetAPI_image_download.ipynb notebook
# Modified from Planet Labs notebooks: https://github.com/planetlabs/notebooks/tree/master/jupyter-notebooks

from planet import OrdersClient, Session
from shapely import geometry as sgeom
from planet import order_request, reporting
from pathlib import Path


def build_quick_search_request(aoi_box_shape, max_cloud_cover, start_date, end_date,
                               item_type, asset_type):
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

    Returns
    -------
    request
    """

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

    request = {
        "item_types": [item_type],
        "asset_types": [asset_type],
        "filter": combined_filter
    }

    return request


def build_request_with_item_ids(request_name, aoi_box, clip_to_aoi, harmonize, item_ids, item_type, asset_type):
    """
    Build Planet API request for image orders using image IDs

    Parameters
    ----------
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
    products = [order_request.product(item_ids, asset_type, item_type)]

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
