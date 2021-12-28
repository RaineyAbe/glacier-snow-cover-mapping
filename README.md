# planet-snow GitHub repository
### Rainey Aberle (raineyaberle@u.boisestate.edu)
### Fall 2021

Preliminary notebooks & short workflow for detecting snow-covered area in PlanetScope 4-band imagery.

- ```planetAPI_image_download.ipynb```: bulk download PlanetScope 4-band images using the Planet API
- ```stitch_by_date.ipynb```: stitch all images captured on the same date
- ```develop_mndsi_threshold.ipynb```: preliminary threshold developed for a modified NDSI (MNDSI) - the normalized difference of PlanetScope NIR and red bands - using a manually digitized snow line picks (from PlanetScope RGB imagery) on Wolverine Glacier, AK for August 2021
- ```compute_mndsi.ipynb```: apply MNDSI to images and calculate snow-covered area in area of interest (AOI)
