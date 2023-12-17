# glacier-snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Department of Geosciences, Boise State University

## Description
Workflow for detecting glacier snow-covered area, accumulation area ratios, and seasonal snowlines in Sentinel-2, Landsat 8/9, and PlanetScope 4-band imagery.

__Basic image processing workflow:__

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow_no_filtering.png" alt="Image processing workflow" width="600"/>

## Requirements

1. __Google Earth Engine account__ to access Landsat and Sentinel-2 imagery. [Sign up here](https://earthengine.google.com/new_signup/).
2. (Optional) __Planet account__ ([sign up here](https://www.planet.com/signup/)) with access to PlanetScope imagery through the NASA Commercial SmallSat Data Acquisition program ([apply here](https://www.planet.com/markets/nasa/)). It may take time for your account to be approved for free PlanetScope images access.
3. __Shapefile__ containing a polygon of your area of interest (AOI). This is used for querying imagery from Google Earth Engine and/or Planet Labs, Inc. and cropping images before classifying.
4. (Optional) __Digital elevation model (DEM)__. If you do not specify a DEM, the code will automatically use the ArcticDEM Mosaic where there is coverage and the NASADEM otherwise, accessed through the GEE data repository.

## Installation
Please see the [Installation Instructions](https://github.com/RaineyAbe/snow-cover-mapping/blob/main/docs/installation_instructions.md).

## Basic Workflow
Please see the [Basic Workflow Instructions](https://github.com/RaineyAbe/snow-cover-mapping/blob/main/docs/basic_workflow.md).

## Citation

Aberle, R., Enderlin, E., O'Neel, S., Marshall., H.P., Florentine, C., Sass, L., and Dickson, A. (_in prep_) Automated snow cover detection on mountain glaciers using space-borne imagery.

DOI via Zenodo for this GitHub repository coming soon, upon acceptance of manuscript.

## Correspondence
Rainey Aberle (raineyaberle@u.boisestate.edu)

## Example results
Below are an example time series of snow-covered area (SCA), accumulation area ratio (AAR), and median snowline elevations at South Cascade Glacier, Washington state for 2013-2022.

__Classified images and seasonal snowlines__

![](figures/SouthCascadeGlacier_results_gif/output.gif)

__Snow cover metrics time series and weekly median trends__

Weekly median trends are excluding PlanetScope to mitigate noise.

<img src='figures/timeseries_SouthCascade_Glacier.png' width='700'>

## Funding and Acknowledgements
We would like to thank members of the [CryoGARS Glaciology](https://github.com/CryoGARS-Glaciology) lab at Boise State University and the [USGS Benchmark Glacier program](https://www.usgs.gov/programs/climate-research-and-development-program/science/usgs-benchmark-glacier-project) staff for their support and input. This work was funded by BAA-CRREL award W913E520C0017 and NASA EPSCoR award 80NSSC20M0222 and utilized data from [Planet Labs, Inc.](https://www.planet.com/) which was made available through the [NASA Commercial Smallsat Data Acquisition (CSDA) Program](https://www.earthdata.nasa.gov/esds/csda).

Several open packages made the integration of this Python-based workflow with Google Earth Engine possible. Thank you to the developers of [geemap](https://geemap.org/), [geedim](https://geedim.readthedocs.io/en/latest/index.html), and [wxee](https://wxee.readthedocs.io/en/latest/index.html).
