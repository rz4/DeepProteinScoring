# Directory /src/
Updated: 04/23/18

This directory contains three subdirectories which contain the scripts used for
data preparation, data visualization, and network training. Descriptions for each
sub-directory can be found below.

## Sub-Directories

- [/data_processing/](src/data_processing) : This sub-directory contains scripts
used for generating pairwise distance feature maps and torsion angle feature maps
from decoy structures. Scripts used to extract PDB files from Rosetta silent files,
and scripts used to calculate GDT-TS can also be found here.

- [/data_visualization/](src/data_visualization) : This sub-dirsectory contains
scripts used to visualize the GDT-TS composition of a given dataset and
visualize generated feature maps.

- [/network_training/](src/network_training) : This sub-directory contains scripts
used to train binary classifier networks and cascading classifier networks on
the protein decoy datasets.
