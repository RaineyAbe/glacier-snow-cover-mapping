# Basic workflow instructions for `snow-cover-mapping`

### 1. (Optional) Download PlanetScope imagery
Download imagery either through [Planet Explorer](planet.com/explorer) or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

### 2. Run the snow detection workflow
The workflow can be run using a Jupyter Notebook, located in the `notebooks` directory, or a Python script, located in the `scripts` directory. 

To run the workflow by passing arguments to a Python script in the command window, see the `pass_args_example.sh` for an example. 

To run the workflow using the Jupyter Notebook:
- Open a new command line window, navigate (`cd`) to `snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser. 
- Open the `snow_classification_pipeline.ipynb` notebook.
- Modify the set-up cell where indicated to specify paths in directory, file names for your AOI shapefile and (optionally) DEM, and options for image querying and downloading.  

Notes:
- The first time you open a Jupyter Notebook, you will likely have to specify the kernel as `snow-cover-mapping`.
- I recommend using the Python script particularly for larger sites and date ranges. Anecdotally, the ipykernel has frozen or shut down when trying to access and process relatively large amounts of imagery.

### 3. (Optional) Filter median snowline altitude time series 

To mitigate the impact of poor image quality or classification, we have developed a preliminary method for filtering the snowline time series using the monthly distribution of snowline altitudes. 

To apply the filtering method to your time series, use the `filter_snowline_timeseries_automatic` notebook or Python script. 

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/filtered_snowline_timeseries_SouthCascade.png" alt="South Cascade Glacier filtered time series" width="800"/>
__Figure: Example filtered timeseries at South Cascade Glacier, Washington state.__
