# COMPSCI715
Agent for Automated Navigation in VR Environments

Code relating to the training and testing of each model can be found in their respective folders (CNNRNN, GAIL, ViT, c-rnn-gan and transfer-learning).

Utils contain shared code for retrieving the dataset, formatting data, creating pytorch dataset objects and visualisation tools.

Datasets contain the various train/val/test splits used throughout development. Each split is described by a txt of session names. A CSV of all session names is also included to help make new splits.

## Import packages
Kindly download packages from the ```requirements.txt``` file

Ensure you run the script from the root directory of the project to allow Python to resolve the package structure correctly. For example:

```
python CNNRNN/CNNRNN_GRU.py
```

All csvs, pytorch models, and tensorboard files will be generated and located inside each model's corresponding directory

### CNN-RNN
For training, please set the hyperparameters in the file and run the script:

```
python CNNRNN/CNNRNN_GRU.py
```

After training, use the same hyperparameters to retrieve the model and evaluate using the script:

```
python CNNRNN/EvalCNNRNN_GRU.py
```

### GAIL
For training, please set the hyperparameters in the file and run any script:

```
python GAIL/practiceGAIL*.py
```

After training, use the same hyperparameters to retrieve the model and evaluate using any script:

```
python GAIL/evalPracGAIL*.py
```

### Video Transformer
Note that some models (such as ViT) require additional repositories to work. These repos are submodules so they should automatically be cloned along with this repo. If this does not happen, please ensure you clone these repos manually and place them in the same location as the submodules.
