# Basic workflow instructions for `snow-cover-mapping`

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
