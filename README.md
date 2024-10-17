# COMPSCI715
Agent for Automated Navigation in VR Environments

Code relating to the training and testing of each model can be found in their respective folders (CNNRNN, GAIL, ViT, c-rnn-gan and transfer-learning).

Utils contain shared code for retrieving the dataset, formatting data, creating pytorch dataset objects and visualisation tools.

Datasets contain the various train/val/test splits used throughout development. Each split is described by a txt of session names. A CSV of all session names is also included to help make new splits.

## Import packages
Ensure you run the script from the root directory of the project to allow Python to resolve the package structure correctly. For example:
python CNNRNN/practice1.py

Note that some models (such as ViT) require additional repositories to work. These repos are submodules so they should automatically be cloned along with this repo. If this does not happen, please ensure you clone these repos manually and place them in the same location as the submodules.
