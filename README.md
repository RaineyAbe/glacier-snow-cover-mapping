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
Please see the [installation instructions](https://github.com/RaineyAbe/snow-cover-mapping/docs/installation_instructions.md). 

## Basic Workflow
### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

### 2. Run the snow detection workflow
The workflow can be run using Jupyter Notebooks, located in the `notebooks` directory, or Python scripts, located in the `scripts` directory. 

When using the Python scripts to run the workflow, there are two options:
1. `snow_classification_pipeline.py`: Edit the set-up portion of the script and run the script from the command line. 
2. `snow_classification_pipeline_pass_arguments.py`: Run the workflow without having to edit the script each time by passing arguments to the script. See the `pass_args_example.sh` file for an example. This can be useful particularly when batch running at many sites using high-performance computing clusters, etc. 

Using Jupyter Notebook to run the workflow:
- Open a new command line window, navigate (`cd`) to `snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser. 
- Open the `snow_classification_pipeline.ipynb` notebook.
- Modify the set-up cell where indicated to specify paths in directory, file names for your AOI shapefile and (optionally) DEM, and options for image querying and downloading.  

Notes:
- The first time you open a Jupyter Notebook, you will likely have to specify the kernel as `snow-cover-mapping`.
- I recommend using the Python scripts particularly for larger sites and date ranges. Anecdotally, the ipykernel has frozen or shut down when trying to access and process relatively large amounts of imagery.

### 3. (Optional) Filter median snowline altitude time series 

To mitigate the impact of poor image quality or classification, we have developed a preliminary method for filtering the snowline time series using the monthly distribution of snowline altitudes. 

To apply the filtering method to your time series, use the `filter_snowline_timeseries_automatic` notebook or Python script. 

_Example filtered timeseries at South Cascade Glacier:_

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/filtered_snowline_timeseries_SouthCascade.png" alt="South Cascade Glacier filtered time series" width="800"/>

## Citation

Coming soon...

## Funding and Acknowledgements
We would like to thank members of the [CryoGARS Glaciology](https://github.com/CryoGARS-Glaciology) lab at Boise State University and the [USGS Benchmark Glacier program](https://www.usgs.gov/programs/climate-research-and-development-program/science/usgs-benchmark-glacier-project) staff for their support and input. This work was funded by BAA-CRREL award W913E520C0017 and NASA EPSCoR award 80NSSC20M0222 and utilized data from [Planet Labs, Inc.](https://www.planet.com/) which was made available through the [NASA Commercial Smallsat Data Acquisition (CSDA) Program](https://www.earthdata.nasa.gov/esds/csda). 