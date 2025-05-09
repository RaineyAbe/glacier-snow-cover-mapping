# Basic workflow instructions for `glacier-snow-cover-mapping`

### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with `Planet_images_order_download.ipynb` in the `notebooks/` folder.

__Disclaimers:__
- Occasionally, Planet orders will fail before the download is complete. Previously submitted orders can be downloaded to file using  `Planet_re-download_past_orders.ipynb` in the `notebooks/` folder.

- Planet Labs, Inc. updates their API and data repositories fairly regularly. The functions and processes for querying and ordering PlanetScope and other images are not guaranteed to work after January 2024. If the notebooks are unsuccessful, we recommend referring to the [Planet SDK documentation](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/) or downloading imagery instead through Planet Explorer.

### 2. Run the snow detection workflow
The workflow can be run using a Jupyter Notebook, located in the `notebooks/` directory, or a Python script, located in the `scripts/` directory.

To run the workflow by passing arguments in the command winder to the Python script, see `scripts/snow_classification_pipeline_args_example.sh`.

To run the workflow using the Jupyter Notebook:
- Open a new command line window, navigate (`cd`) to `glacier-snow-cover-mapping/notebooks`, activate the Conda/Mamba environment (if using), and run the command `jupyter lab` to open the Jupyter interface in a web browser.
- Open `snow_classification_pipeline.ipynb`.
- Modify the set-up cell where indicated to specify paths in directory, file names for your AOI shapefile and (optionally) DEM, and options for image querying and downloading.  

See the notebook/script for more detailed instructions.

Notes:
- The first time you open a Jupyter Notebook, you will likely have to specify the kernel as `glacier-snow-cover-mapping`.
- We recommend using the Python script particularly for larger sites and date ranges. Anecdotally, the ipykernel has frozen or shut down when trying to access and process relatively large amounts of imagery.
