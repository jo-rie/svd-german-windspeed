# SVD-reduction of high-dimensional German spatio-temporal wind speed data and clusters of similarity

This repository contains accompanying code to the paper "SVD-reduction of high-dimensional German spatio-temporal wind speed data and clusters of similarity" by Oliver Grothe and Jonas Rieger (Karlsruhe Institute of Technology).
More on the data set, its interpretation and the results can be found therein. 

## Downloading the data

The data can be downloaded and preprocessed through the zsh script `download_data.zsh`, e.g., by
`source download_data.zsh`.
Preprocessing comprises cropping to a bounding box of the German borders and aggregating the data per month into a single file. 
The temporal files are deleted in the end of the script. 
The script uses [cdo](https://code.mpimet.mpg.de/projects/cdo/) by Uwe Schulzweida.

**Note that other files with ending .nc4 in the current folder could be deleted when executing the script!**

## Performing SVD

The script `perform_svd_and_clustering.py` computes the singular value decomposition, loadings, local approximation, and clustering. 
The results are stored in `results/`.
Various helper functions are in the script `utils.py`.

## Analysing the results

In the script `create_plots.py`, the plots of the paper are generated based on the results of the svd and stored in the folder `plots/`.
