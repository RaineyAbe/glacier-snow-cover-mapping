
# Change Log 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - 2025-05-07
 
Here we write upgrading notes for the `glacier-snow-cover-mapping` package. 
 
### Added
- Fully client-side pipeline with no image exports. See the `notebooks/snow_classification_pipeline_GEE.ipynb` and `scripts/snow_classification_pipeline_GEE.py` files. All classification and processing is conducted via Google Earth Engine. Snow cover statistics are saved to Google Drive. The associated utility functions are located in `functions/pipeline_utils_GEE.py`. 
 
### Changed
- Renamed server-side pipeline and utilities with the "_SKL" prefix to signify the use of SciKit-Learn classifiers.  

- `docs/basic_workflow.md`: Modified to include the fully client-side pipeline. 