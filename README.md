# snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Boise State University

Contact: raineyaberle@u.boisestate.edu

## Description
Notebooks & short workflow for detecting snow-covered area, seasonal snowlines, and equilibrium line altitudes in PlanetScope 4-band, Landsat and Sentinel-2 imagery.

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow_nostep4.png" alt="Image processing workflow" width="600"/>

## Installation
#### 1. Clone repository
To clone the `snow-cover-mapping` repository into your local directory, execute the following command from a terminal in your desired directory:

`git clone https://github.com/RaineyAbe/snow-cover-mapping.git`

#### 2. Create Conda environment from .yml file
To ensure all required packages for the notebooks are installed, I recommend creating a conda environment using the `environment.yml` file provided. To create the conda environment using the .yml file for this repository, execute the following command:

`conda env create -f environment.yml`

[Here is a helpful resource](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for working with Conda environments.

#### 3. Activate Conda environment
To activate the Conda environment, execute the following command:

`conda activate snow-cover-mapping`

#### 4. Add Conda environment as an ipykernel

Now, run the following command in a terminal so that you can use the `planet-snow` environment in Jupyter Notebook:

`python -m ipykernel install --user --name=snow-cover-mapping`

## Run the snow classification and snowline detection pipeline

The workflow can be run using Jupyter Notebooks -- located in the `notebooks` directory or Python scripts -- located in the `scripts` directory. 

#### 1. Download PlanetScope imagery
Download imagery either through Planet Explorer or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

#### 2. Run the snow detection workflow
Run the `snow_classification_pipeline.ipynb` notebook. This requires a [free Google Earth Engine account](https://signup.earthengine.google.com/#!/) to access imagery. 

#### 3. Filter median snowline elevations and identify the annual ELAs for all years of observation using the `snowline_filter_fit.ipynb` notebook. 

#### Optional: some example code for plotting results is provided in the `make_figures.ipynb` notebook. 

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/median_snowline_elevs.png" alt="Image processing workflow" width="600"/>

## Recommended directory structure
The notebooks are set up so that inputs and outputs can be found easily and programmatically. Thus, I recommend that you structure your `base_path` and `study-sites` folders similar to the following so that you don't need to modify as much code at the beginning of each notebook. The results will be automatically saved in the folders as shown below, defined by the `out_path` variable in each notebook. 

    .
    ├── ...
    ├── base_path                       # Folder containing the snow-cover-mapping repository 
    │   ├── snow-cover-mapping          # snow-cover-mapping repository, root directory
    │   ├── study-sites                 # Folder where results and imagery for individual sites will be saved 
    │   │   ├── site_name               # Folder containing all data for one study site   
    │   │   │   ├── imagery             # Where images and workflow inputs and outputs are/will be saved
    │   │   │   │   ├── PlanetScope     
    │   │   │   │   │   ├── raw_images  # Folder containing raw PlanetScope image downloads
    │   │   │   │   │   ├── masked      # Where masked PlanetScope images will be saved
    │   │   │   │   │   └── mosaics     # Where mosaicked, masked PlanetScope images will be saved
    │   │   │   │   ├── classified      # Where all classified images will be saved
    │   │   │   │   └── snowlines       # Where all snowlines and ELAs will be saved
    │   │   │   ├── AOI                 # Folder containing outline of the Area of Interest (AOI), shapefile
    │   │   │   └── DEM                 # Folder containing digital elevation model of the AOI (optional)  
    │   │   └── ...              
    │   └── ...
    └── ...

