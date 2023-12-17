# Basic workflow instructions for `glacier-snow-cover-mapping`

### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

### 2. Run the snow detection workflow
The workflow can be run using a Jupyter Notebook, located in the `notebooks` directory, or a Python script, located in the `scripts` directory. 

To run the workflow by passing arguments to a Python script in the command window, see the `pass_args_example.sh` for an example. 

To run the workflow using the Jupyter Notebook:
- Open a new command line window, navigate (`cd`) to `glacier-snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser. 
- Open the `snow_classification_pipeline.ipynb` notebook.
- Modify the set-up cell where indicated to specify paths in directory, file names for your AOI shapefile and (optionally) DEM, and options for image querying and downloading.  

Notes:
- The first time you open a Jupyter Notebook, you will likely have to specify the kernel as `glacier-snow-cover-mapping`.
- I recommend using the Python script particularly for larger sites and date ranges. Anecdotally, the ipykernel has frozen or shut down when trying to access and process relatively large amounts of imagery.