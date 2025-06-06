# Basic workflow instructions for `glacier-snow-cover-mapping`

## Workflow selection

There are multiple ways to run the classification workflow, depending on user needs and computing/time constraints. The two decisions the user should make are _1. Whether to export classified images_ and _2. Whether to run using the Jupyter Notebook or the Python script_ by passing arguments on the command line. More info on each decision is provided below. 

__1. Whether to export classified images__

Google Earth Engine (GEE) is a powerful tool, but limited in the amount of data you can export as a user. Thus, there are two version of the pipeline. In this repository, images are downloaded then classified using SciKit-Learn classifiers and all computations are done locally to enable access to the classified images. In the partner repository (`gscm_gee`), all processing is done on the GEE server and only the summary statistics are exported to your Google Drive. Downloading and classifying images requires more local computing power, so this choice will depend on resources and desired outputs. 

__2. Whether to run using the Jupyter notebook or the Python script__

There are pros and cons to running the pipeline using the notebook or the script. 

- Notebook (located in the `notebooks` folder): Better for processing a single site. Inputs and outputs can be more easily visualized. Anecdotally, the ipykernel has frozen or shut down when trying to access and process larger image collections. Thus, we recommend using the Python script particularly for larger sites and date ranges. 

- Script (located in the `scripts` folder): Better for batch processing multiple sites. Arguments are passed and run through the command line. Note: if many sites are run at once, you may exceed your maximum GEE synchronous processes rate. We have found running < 5 sites at a time helps to avoid this issue.  

## Workflow steps

### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with `notebooks/Planet_images_order_download.ipynb`.

__Disclaimers:__
- Occasionally, communication with Planet orders will disconnect before the download is complete. Previously submitted orders can be downloaded using `notebooks/Planet_re-download_past_orders.ipynb`.

- Planet Labs updates their API and data repositories fairly regularly. The functions and processes for querying and ordering PlanetScope and other images are not guaranteed to work after January 2024. If the notebooks are unsuccessful, we recommend referring to the [Planet SDK documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/) or downloading imagery instead through Planet Explorer.

### 2. Run the snow detection workflow

#### Notebook version using JupyterLab:
- Open a new command line window, navigate (`cd`) to `glacier-snow-cover-mapping/notebooks`, activate the Conda/Mamba environment (if using), and run the command `jupyter lab` to open the Jupyter interface in a web browser.
- Open `snow_classification_pipeline.ipynb`
- Modify the cells where indicated to specify paths in directory, (optionally) file names for your AOI shapefile and DEM, and options for image querying and downloading.  

See the notebook for more detailed instructions.

#### Script version:
- Open a new command window, navigate (`cd`) to `glacier-snow-cover-mapping/scripts` and activate the Conda/Mamba environment (if using). 
- Run the pipeline with the following command: `python glacier_snow_cover_mapping.py **args` 
- See the script file for descriptions of the required and optional input arguments, or run `help(snow_classification_pipeline.py)`

#### Boise State University Borah and/or other HPC users
To run the pipeline by submitting a job to BSU's Borah cluster, see the example script: `glacier-snow-cover-mapping/scripts/slurm_example_SITE-ID.bash`. Here, the pipeline is run for a single site using the Docker image. 