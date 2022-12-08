# snow-cover-mapping
#### Rainey Aberle, Ellyn Enderlin, HP Marshall, Shad O'Neel, & Alejandro Flores
#### Boise State University
#### Contact: raineyaberle@u.boisestate.edu

### Description
Notebooks & short workflow for detecting snow-covered area and seasonal snowlines in PlanetScope 4-band, Landsat and Sentinel-2 imagery.
<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/median_snowline_elevs.png" alt="Image processing workflow" width="200"/>

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

#### 5. Run the snow classification and snowline detection pipelines

The workflow can be run using Jupyter Notebooks -- located in the `notebooks` directory or Python scripts -- located in the `scripts` directory. 

__PlanetScope__
- Download imagery either through Planet Explorer or using the Python API with the `download_PlanetScope_images.ipynb` notebook.
- Run the snow classification pipeline: `snow_classification_pipeline_PlanetScope.ipynb`

__Landsat 8/9 & Sentinel-2__
- Run the `snow_classification_pipeline*.ipynb` notebooks. These require a free Google Earth Engine account to access and download imagery. 

#### 6. Fit linear trendlines to annual snowline elevation timeseries using the `snowline_linear_regression.ipynb` notebook. 

#### Optional: some example code for plotting results is provided in the `make_figures.ipynb` notebook. 

<img src="https://github.com/RaineyAbe/snow-cover-mapping/blob/main/figures/median_snowline_elevs.png" alt="Image processing workflow" width="200"/>

