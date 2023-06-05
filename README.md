# snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Department of Geosciences, Boise State University

Contact: raineyaberle@u.boisestate.edu

## Description
Workflow for detecting glacier snow-covered area, seasonal snowlines, and equilibrium line altitudes in PlanetScope 4-band, Landsat 8/9, and Sentinel-2 imagery.

_Image processing workflow:_

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow.png" alt="Image processing workflow" width="600"/>

## Requirements

1. Google Earth Engine account to access Landsat and Sentinel-2 imagery. [Sign up here](https://earthengine.google.com/new_signup/). 

2. (Optional) Planet account ([sign up here](https://www.planet.com/signup/)) with access to PlanetScope imagery through the NASA Commercial SmallSat Data Acquisition program ([apply here](https://www.planet.com/markets/nasa/)). It may take time for your account to be approved for free PlanetScope images access. 

## Installation
Please see the [installation instructions](https://github.com/RaineyAbe/snow-cover-mapping/docs/installation_instructions.md). 

## Basic Workflow
### 1. Download PlanetScope imagery (optional)
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

### 2. Run the snow detection workflow
The workflow can be run using Jupyter Notebooks -- located in the `notebooks` directory, or Python scripts -- located in the `scripts` directory. To run a Notebook, open a new Terminal window, navigate (`cd`) to `snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser. To run a script, modify the first section in the script where indicated, then run from the Terminal.

Notes:
- The first time you open a notebook, you will likely have to specify the kernel as `snow-cover-mapping`.
- I recommend using the Python scripts particularly for larger sites and date ranges. Anecdotally, the ipykernel has frozen or shut down when trying to access and process relatively large amounts of imagery.

Run the `snow_classification_pipeline` notebook or script.

### 3. Filter median snowline elevations time series using the `filter_snowline_timeseries_automatic` notebook or script to mitigate the impact of poor image quality or classification.

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/filtered_snowline_timeseries_SouthCascade.png" alt="South Cascade Glacier filtered time series" width="800"/>

## Citation

_Coming soon..._

## Funding and Acknowledgements
We would like to thank members of the [CryoGARS Glaciology](https://github.com/CryoGARS-Glaciology) lab at Boise State University and the [USGS Benchmark Glacier program](https://www.usgs.gov/programs/climate-research-and-development-program/science/usgs-benchmark-glacier-project) staff for their support and input. This work was funded by BAA-CRREL award W913E520C0017 and NASA EPSCoR award 80NSSC20M0222 and utilized data from [Planet Labs, Inc.](https://www.planet.com/) which was made available through the [NASA Commercial Smallsat Data Acquisition (CSDA) Program](https://www.earthdata.nasa.gov/esds/csda). 