# Basic workflow instructions for `glacier-snow-cover-mapping`

## Workflow selection

There are multiple ways to run the classification workflow, depending on user needs and computing/time constraints. The two main decisions that must be made to run the pipeline are outlined below. 

__1. Whether to export classified images__

Google Earth Engine (GEE) is a powerful tool, but limited in the amount of data you can export as a user. Thus, there are two version of the pipeline: one where images are downloaded then classified using SciKit-Learn classifiers, and one where all processing is done on the GEE server and only the summary statistics are exported. Downloading classified images requires more local computing power, so this choice will depend on resources and desired outputs. 

To run the pipeline locally and export classified images, use the _"snow_classification_pipeline_SKL*"_ files.

To run the pipeline on the GEE server and export only the snow cover statistics to your Google Drive, use the _"snow_classification_pipeline_GEE*"_ files. 

__2. Whether to run using the Jupyter notebook or the Python script__

There are pros and cons to running the pipeline using the notebook or the script. 

- Notebook (located in the `notebooks` folder): Better for processing a single site. Inputs and outputs can be more easily visualized. Anecdotally, the ipykernel has frozen or shut down when trying to access and process larger image collections. Thus, we recommend using the Python script particularly for larger sites and date ranges. 

- Script (located in the `scripts` folder): Better for batch processing multiple sites. Arguments are passed and run through the command line. Note: if many sites are run at once, you may exceed your maximum GEE synchronous processes rate. We have found running < 5 sites at a time helps to avoid this issue.  

## Workflow steps

### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with `notebooks/Planet_images_order_download.ipynb`.

__Disclaimers:__
- Occasionally, Planet orders will fail before the download is complete. Previously submitted orders can be downloaded to file using `notebooks/Planet_re-download_past_orders.ipynb`.

- Planet Labs updates their API and data repositories fairly regularly. The functions and processes for querying and ordering PlanetScope and other images are not guaranteed to work after January 2024. If the notebooks are unsuccessful, we recommend referring to the [Planet SDK documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/) or downloading imagery instead through Planet Explorer.

- Because PlanetScope images must be downloaded locally, they can only be processed using the SciKit-Learn classifiers. Thus, the user must use the _"snow_classification_pipeline_SKL"_ files. 

### 2. Run the snow detection workflow

#### Notebook version using JupyterLab:
- Open a new command line window, navigate (`cd`) to `glacier-snow-cover-mapping/notebooks`, activate the Conda/Mamba environment (if using), and run the command `jupyter lab` to open the Jupyter interface in a web browser.
- Open `snow_classification_pipeline_XXX.ipynb` (replace "XXX" with "GEE" or "SKL")
- Modify the cells where indicated to specify paths in directory, (optionally) file names for your AOI shapefile and DEM, and options for image querying and downloading.  

See the notebook for more detailed instructions.

#### Script version:
- Open a new command window, navigate (`cd`) to `glacier-snow-cover-mapping/scripts` and activate the Conda/Mamba environment (if using). 
- Run the pipeline with the following command: `python glacier_snow_cover_mapping_XXX.py **args` (replace "XXX" with "GEE" or "SKL")
- See the script file for descriptions of the required and optional input arguments, or run `help(glacier_snow_cover_mapping_XXX.py)`
