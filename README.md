# snow-cover-mapping
#### Rainey Aberle
#### Boise State University
#### Contact: raineyaberle@u.boisestate.edu

### Description
Preliminary notebooks & short workflow for detecting snow-covered area and seasonal snowlines in PlanetScope 4-band, Landsat, Sentinel-2, and MODIS imagery.

### Installation
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

#### 5. Open a Jupyter Notebook
To open a jupyter notebook, navigate (`cd`) to the `planet-snow` directory on your machine if you have not already and run the following command: `jupyter notebook notebook.ipynb`, replacing `notebook.ipynb` with the name of the notebook you would like to open. The notebook should open in a browser.
