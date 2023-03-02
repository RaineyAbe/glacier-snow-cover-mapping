# snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Boise State University

Contact: raineyaberle@u.boisestate.edu

## Description
Notebooks & short workflow for detecting snow-covered area, seasonal snowlines, and equilibrium line altitudes in PlanetScope 4-band, Landsat and Sentinel-2 imagery.

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow.png" alt="Image processing workflow" width="600"/>

## Requirements

1. Planet account ([sign up here](https://www.planet.com/signup/)) with access to PlanetScope imagery through the NASA Commercial SmallSat Data Acquisition program ([apply here](https://www.planet.com/markets/nasa/)). It may take time to have your account approved for free PlanetScope image downloads. 

2. Google Earth Engine account to access Landsat and Sentinel-2 imagery. [Sign up here](https://earthengine.google.com/new_signup/). 

## Installation
#### Optional: Fork repository for personal use
To save a copy of the code for personal use, fork the `snow-cover-mapping` code repository to your personal GitHub account. See [this page](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for instructions on how to fork a repository. 

#### 1. Clone repository
To clone the `snow-cover-mapping` repository into your local directory, open a new Terminal window and change directory (`cd`) to where you want it to be stored. Then, execute the following command:

`git clone https://github.com/RaineyAbe/snow-cover-mapping.git`

If you forked the code repository to your personal account, replace `RaineyAbe` with `YourUserName` in the command above. 

#### 2. Create Conda environment from .yml file
To ensure all required packages for the notebooks/scripts are installed, I recommend creating a conda environment using the `environment.yml` file provided. To create the conda environment using the .yml file for this repository, execute the following command:

`conda env create -f environment.yml`

[Here is a helpful resource](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for working with Conda environments.

#### 3. Activate Conda environment
To activate the Conda environment, execute the following command:

`conda activate snow-cover-mapping`

#### 4. Add Conda environment as an ipykernel

Now, run the following command in a terminal so that you can use the `planet-snow` environment in Jupyter Notebook/Lab:

`python -m ipykernel install --user --name=snow-cover-mapping`

## Run the snow classification and snowline detection pipeline

The workflow can be run using Jupyter Notebooks -- located in the `notebooks` directory or Python scripts -- located in the `scripts` directory. To run a Notebook, open a new Terminal, navigate (`cd`) to `snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser.  

#### 1. Download PlanetScope imagery
Download imagery either through Planet Explorer or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

#### 2. Run the snow detection workflow
Run the `snow_classification_pipeline.ipynb` notebook. This requires a [free Google Earth Engine account](https://signup.earthengine.google.com/#!/) to access imagery. 

#### 3. Filter median snowline elevations and identify the annual ELAs for all years of observation using the `snowline_filter_fit.ipynb` notebook. 

#### Optional: some example code for plotting results is provided in the `make_figures.ipynb` notebook. 

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/median_snowline_elevs_SouthCascade.png" alt="Example filtered snowline time series at South Cascade Glacier, WA" width="600"/>

## Recommended directory structure
The notebooks are set up so that inputs and outputs can be found easily and programmatically. Thus, I recommend that you structure your directory as outlined below.  

_Initial set-up:_ Before running any notebooks

    .
    ├── ...
    ├── base_path                       # Folder containing the snow-cover-mapping code repository and study-sites folder
    │   ├── snow-cover-mapping          # snow-cover-mapping code repository, root directory
    │   ├── study-sites                 # Folder where results and imagery for individual sites will be saved 
    │   │   ├── site_name               # Folder containing all data for one study site. Replace site_name with the name of your site.  
    │   │   │   ├── imagery             # Folder where images and workflow inputs and outputs are/will be saved
    │   │   │   │   └── PlanetScope     # Folder where all PlanetScope raw and pre-processed images will be saved 
    │   │   │   │       └── raw_images  # Folder containing raw PlanetScope image downloads
    │   │   │   ├── AOIs                # Folder containing outline of the Area of Interest (AOI), shapefile
    │   │   │   └── DEMs                # Folder containing digital elevation model of the AOI (optional)  
    │   │   └── ...              
    │   └── ...
    └── ...

_After running the snow classification workflow:_ Includes directories that are automatically created. 

    .
    ├── ...
    ├── base_path                       # Folder containing the snow-cover-mapping code repository and study-sites folder
    │   ├── snow-cover-mapping          # snow-cover-mapping code repository, root directory
    │   ├── study-sites                 # Folder where results and imagery for individual sites will be saved 
    │   │   ├── site_name               # Folder containing all data for one study site   
    │   │   │   ├── imagery             # Where images and workflow inputs and outputs are/will be saved
    │   │   │   │   ├── PlanetScope     # Folder where all PlanetScope raw and pre-processed images will be saved
    │   │   │   │   │   ├── raw_images  # Folder containing raw PlanetScope image downloads
    │   │   │   │   │   ├── masked      # Where masked PlanetScope images will be saved
    │   │   │   │   │   └── mosaics     # Where masked, mosaicked PlanetScope images will be saved
    │   │   │   │   ├── classified      # Where all classified images will be saved
    │   │   │   │   └── snowlines       # Where all snowlines and ELAs will be saved
    │   │   │   ├── AOIs                # Folder containing outline of the Area of Interest (AOI), shapefile
    │   │   │   └── DEMs                # Folder containing digital elevation model of the AOI (optional)  
    │   │   └── ...              
    │   └── ...
    └── ...
