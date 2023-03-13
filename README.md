# snow-cover-mapping

Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores

Department of Geosciences, Boise State University

Contact: raineyaberle@u.boisestate.edu

## Description
Workflow for detecting glacier snow-covered area, seasonal snowlines, and equilibrium line altitudes in PlanetScope 4-band, Landsat 8/9, and Sentinel-2 imagery.

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/methods_workflow.png" alt="Image processing workflow" width="600"/>

## Requirements

1. Google Earth Engine account to access Landsat and Sentinel-2 imagery. [Sign up here](https://earthengine.google.com/new_signup/). 

2. (Optional) Planet account ([sign up here](https://www.planet.com/signup/)) with access to PlanetScope imagery through the NASA Commercial SmallSat Data Acquisition program ([apply here](https://www.planet.com/markets/nasa/)). It may take time for your account to be approved for free PlanetScope images access. 

## Installation
#### Optional: Fork repository for personal use
To save a copy of the code for personal use, fork the `snow-cover-mapping` code repository to your personal GitHub account. See [this page](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for instructions on how to fork a GitHub repository. 

#### 1. Clone code repository
To clone the `snow-cover-mapping` repository into your local directory, open a new Terminal window and change directory (`cd`) to where you want it to be stored (which is referred to as the `base_path` in the code). Then, execute the following command:

`git clone https://github.com/RaineyAbe/snow-cover-mapping.git`

If you forked the code repository to your personal Git account, replace `RaineyAbe` with `YourUserName` in the command above. 

#### 2. Download Miniconda or Anaconda
For packaging and managing all of the required Python packages, I recommend downloading either [Miniconda or Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html). This will enable you to install the environment directly using the .yml file below. See [this helpful guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) for getting started with Conda if you are unfamiliar. 

#### 3. Create Conda environment from .yml file
To ensure all required packages for the notebooks/scripts are installed, I recommend creating a conda environment using the `environment.yml` file provided by executing the following command:

`conda env create -f environment.yml`

[Here is a helpful resource](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for working with Conda environments.

#### 4. Activate Conda environment
To activate the Conda environment, execute the following command:

`conda activate snow-cover-mapping`

#### 5. Add Conda environment as an ipykernel

Now, run the following command so that you can use the `snow-cover-mapping` environment in Jupyter Notebook/Lab:

`python -m ipykernel install --user --name=snow-cover-mapping`

## Run the snow classification and snowline detection pipeline

The workflow can be run using Jupyter Notebooks -- located in the `notebooks` directory, or Python scripts -- located in the `scripts` directory. To run a Notebook, open a new Terminal window, navigate (`cd`) to `snow-cover-mapping/notebooks`, activate the Conda environment (if using), and run the command `jupyter lab` or `jupyter notebook` to open the Jupyter interface in a web browser. 

Note: The first time you open a notebook, you will likely have to specify the kernel as `snow-cover-mapping`.  

#### 1. Download PlanetScope imagery 
Download imagery either through Planet Explorer or using the Python API with the `download_PlanetScope_images.ipynb` notebook.

#### 2. Run the snow detection workflow
Run the `snow_classification_pipeline.ipynb` notebook. This requires a [free Google Earth Engine account](https://signup.earthengine.google.com/#!/) to access imagery. 

#### 3. Filter median snowline elevations time series using the `snowline_filter_fit.ipynb` notebook to mitigate the impact of poor image quality or classification. 

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/median_snowline_elevs.png" alt="Image processing workflow" width="600"/>

## Recommended directory structure
The notebooks are set up so that inputs and outputs can be found easily and programmatically. Thus, I recommend that you structure your directory as outlined below. Otherwise, you can modify the file paths and names in the "Set-up" section of each notebook. 

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
    │   │   │   ├── DEMs                # Folder containing digital elevation model of the AOI (optional)  
    │   │   │   └── figures             # Where figures/images will be saved
    │   │   └── ...              
    │   └── ...
    └── ...
