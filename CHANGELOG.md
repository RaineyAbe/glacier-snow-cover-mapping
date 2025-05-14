# Change Log

Here we write upgrading notes for the `glacier-snow-cover-mapping` package.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] - 2025-05-08

### Added
- Fully client-side pipeline with no image exports. See the `notebooks/snow_classification_pipeline_GEE.ipynb` and `scripts/snow_classification_pipeline_GEE.py` files. All classification and processing is conducted via Google Earth Engine. Snow cover statistics are saved to Google Drive. The associated utility functions are located in `functions/pipeline_utils_GEE.py`.

- Dockerfile and container instructions added to `docs/installation.md`.

### Changed
- Renamed server-side pipeline and utilities with the "_SKL" prefix to signify the use of SciKit-Learn classifiers.  

- `docs/basic_workflow.md`: Modified to include the GEE client-side pipeline.

- Updated the ArcticDEM Mosaic queried from GEE from V3 to V4.

- Snow cover statistics: modified the saved attributes, including the addition of "ice_area_m2", "rock_area_m2", and "water_area_m2". 

- In the SciKit-Learn (SKL)-based pipeline, delineating snowlines using edge detection of the classified images is now optional with the `delineate_snowlines` argument. 
