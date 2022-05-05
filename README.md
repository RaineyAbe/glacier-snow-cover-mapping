# planet-snow
#### Rainey Aberle (raineyaberle@u.boisestate.edu), Boise State University
#### Last updated: January 2022

### Description
Preliminary notebooks & short workflow for detecting snow-covered area in PlanetScope 4-band imagery.

1. `planetAPI_image_download.ipynb`: bulk download PlanetScope 4-band images using the Planet API
2. `stitch_by_date.ipynb`: stitch all images captured on the same date
3. `develop_mndsi_threshold.ipynb`: preliminary threshold developed for a modified NDSI (MNDSI) - the normalized difference of PlanetScope NIR and red bands - using manually digitized snow line picks (from PlanetScope RGB imagery) on Wolverine Glacier, AK for August 2021
4. `calculate_snow_covered_area.ipynb`: apply MNDSI threshold to images to create timeseries of snow-covered area in area of interest (AOI)

### Installation
#### 1. Clone repository
To clone the `planet-snow` repository into your local directory, execute the following command from a terminal in your desired directory:

`git clone https://github.com/RaineyAbe/planet-snow.git`

#### 2. Create Conda environment from .yml file
To ensure all required packages for the notebooks are installed, I recommend creating a conda environment using the `environment.yml` file provided. To create the conda environment using the .yml file for this repository, execute the following command:

`conda env create -f environment.yml`

[Here is a helpful resource](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for working with Conda environments.

#### 3. Activate Conda environment
To activate the Conda environment, execute the following command:

`conda activate planet-snow`

#### 4. Add Conda environment as an ipykernel

Now, run the following command in a terminal so that you can use the `planet-snow` environment in Jupyter Notebook:

`python -m ipykernel install --user --name=planet-snow`

#### 5. Open a Jupyter Notebook
To open a jupyter notebook, navigate (`cd`) to the `planet-snow` directory on your machine if you have not already and run the following command: `jupyter notebook notebook.ipynb`, replacing `notebook.ipynb` with the name of the notebook you would like to open. The notebook should open in a browser.
