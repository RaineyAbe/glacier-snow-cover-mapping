# snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Department of Geosciences, Boise State University

Contact: raineyaberle@u.boisestate.edu

## Description
Workflow for detecting glacier snow-covered area, seasonal snowlines, and equilibrium line altitudes in PlanetScope 4-band, Landsat 8/9, and Sentinel-2 imagery.

_Basic image processing workflow:_

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow_no_filtering.png" alt="Image processing workflow" width="600"/>

## Requirements

1. __Google Earth Engine account__ to access Landsat and Sentinel-2 imagery. [Sign up here](https://earthengine.google.com/new_signup/).
2. (Optional) __Planet account__ ([sign up here](https://www.planet.com/signup/)) with access to PlanetScope imagery through the NASA Commercial SmallSat Data Acquisition program ([apply here](https://www.planet.com/markets/nasa/)). It may take time for your account to be approved for free PlanetScope images access. 
3. __Shapefile__ containing a polygon of your area of interest (AOI). This will be used for querying imagery from Google Earth Engine and cropping images before classifying. 
4. (Optional) __Digital elevation model (DEM)__. If you do not specify a DEM, the code will automatically use the ArcticDEM Mosaic where there is coverage and the NASADEM otherwise. 

## Installation
Please see the [Installation Instructions](https://github.com/RaineyAbe/snow-cover-mapping/blob/main/docs/installation_instructions.md). 

## Basic Workflow
Please see the [Basic Workflow Instructions](https://github.com/RaineyAbe/snow-cover-mapping/blob/main/docs/basic_workflow.md).

## Citation

Coming soon...

## Funding and Acknowledgements
We would like to thank members of the [CryoGARS Glaciology](https://github.com/CryoGARS-Glaciology) lab at Boise State University and the [USGS Benchmark Glacier program](https://www.usgs.gov/programs/climate-research-and-development-program/science/usgs-benchmark-glacier-project) staff for their support and input. This work was funded by BAA-CRREL award W913E520C0017 and NASA EPSCoR award 80NSSC20M0222 and utilized data from [Planet Labs, Inc.](https://www.planet.com/) which was made available through the [NASA Commercial Smallsat Data Acquisition (CSDA) Program](https://www.earthdata.nasa.gov/esds/csda). 