# Installation for `glacier-snow-cover-mapping`

## (Optional) Fork repository for personal use
To save a copy of the code for personal use, fork the `glacier-snow-cover-mapping` code repository to your personal GitHub account. See [this page](https://docs.github.com/en/get-started/quickstart/fork-a-repo) for instructions on how to fork a GitHub repository.

## 1. Clone code repository
To clone the `glacier-snow-cover-mapping` repository into your local directory, open a new Terminal window and change directory (`cd`) to where you want it to be stored (which is referred to as the `base_path` in the code). Then, execute the following command:

`git clone https://github.com/RaineyAbe/glacier-snow-cover-mapping.git`

If you forked the code repository to your personal Git account, replace `RaineyAbe` with `YourUserName` in the command above.

## 2. Install Conda or Mamba
For managing the required Python packages, we recommend downloading either [Miniconda, Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html). This will allow you to install the environment directly using the `.yml` file. See the online user guides for [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) if you are unfamiliar. If using Mamba, replace all instances of `conda` below with `mamba`.

## 3. Install the environment using the .yml file
To ensure all required packages for the notebooks/scripts are installed, we recommend creating a conda/mamba environment using the `environment.yml` file provided by executing the following command:

`conda env create -f environment.yml`

## 4. Activate environment
To activate the environment, execute the following command:

`conda activate glacier-snow-cover-mapping`

## 5. Add environment as an ipykernel

Run the following command so that you can use the `glacier-snow-cover-mapping` environment in Jupyter Lab:

`python -m ipykernel install --user --name=glacier-snow-cover-mapping`

## Recommended directory structure
The notebooks are set up so that inputs and outputs can be found easily and programmatically. Thus, we recommend that you structure your directory as outlined below. Otherwise, you can modify the file paths and names in the first section of each notebook/script.

_Initial set-up:_ Before running any notebooks

    .
    ├── ...
    ├── base_path                       # Folder containing the glacier-snow-cover-mapping code repository and study-sites folder
    │   ├── glacier-snow-cover-mapping  # glacier-snow-cover-mapping code repository, root directory
    │   ├── study-sites                 # Folder where results and imagery for individual sites will be saved
    │   │   ├── site_name               # Folder containing all data for one study site. Replace site_name with the name of your site.
    │   │   │   ├── imagery             # Folder where images and workflow inputs and outputs are/will be saved
    │   │   │   │   └── PlanetScope     # (Optional) Folder where all PlanetScope raw and pre-processed images will be saved
    │   │   │   │       └── raw_images  # Folder containing raw PlanetScope image downloads
    │   │   │   ├── AOIs                # Folder containing outline of the Area of Interest (AOI), shapefile
    │   │   │   └── DEMs                # Folder containing digital elevation model of the AOI (optional)
    │   │   └── ...
    │   └── ...
    └── ...

_After running the snow classification workflow:_ Includes directories that are automatically created.

    .
    ├── ...
    ├── base_path                       # Folder containing the glacier-snow-cover-mapping code repository and study-sites folder
    │   ├── glacier-snow-cover-mapping  # glacier-snow-cover-mapping code repository, root directory
    │   ├── study-sites                 # Folder where results and imagery for individual sites will be saved
    │   │   ├── site_name               # Folder containing all data for one study site
    │   │   │   ├── imagery             # Where images and workflow inputs and outputs are/will be saved
    │   │   │   │   ├── PlanetScope     # Folder where all PlanetScope raw and pre-processed images will be saved
    │   │   │   │   │   ├── raw_images  # Folder containing raw PlanetScope image downloads
    │   │   │   │   │   ├── masked      # Where masked PlanetScope images will be saved
    │   │   │   │   │   └── mosaics     # Where masked, mosaicked PlanetScope images will be saved
    │   │   │   │   ├── Landsat         # Where Landsat surface reflectance image downloads will be saved if specified or images are larger than GEE limit
    │   │   │   │   ├── Sentinel-2_SR   # Where Sentinel-2 surface reflectance image downloads will be saved if specified or images are larger than GEE limit
    │   │   │   │   ├── Sentinel-2_TOA  # Where Sentinel-2 top of atmosphere reflectance image downloads will be saved if specified or images are larger than GEE limit
    │   │   │   │   ├── classified      # Where all classified images will be saved
    │   │   │   │   └── snowlines       # Where all snowline and snow cover statistics CSVs will be saved
    │   │   │   ├── AOIs                # Folder containing outline of the Area of Interest (AOI), shapefile
    │   │   │   ├── DEMs                # Folder containing digital elevation model of the AOI
    │   │   │   └── figures             # Where figures will be saved
    │   │   └── ...
    │   └── ...
    └── ...
