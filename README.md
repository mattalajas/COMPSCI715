# COMPSCI715
Agent for Automated Navigation in VR Environments

Code relating to the training and testing of each model can be found in their respective folders (CNNRNN, GAIL, ViT, c-rnn-gan and transfer-learning).

Utils contain shared code for retrieving the dataset, formatting data, creating pytorch dataset objects and visualisation tools. Most notably, ```visualisation.ipynb``` contains our generated visualisations and results

Datasets contain the various train/val/test splits used throughout development. Each split is described by a txt of session names. A CSV of all session names is also included to help make new splits.

## Import packages
Kindly download packages from the ```requirements.txt``` file

```
pip install -r requirements.txt
```

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

After training, use the same hyperparameters to retrieve the model, evaluate, and create the result CSVs using the script:

```
python CNNRNN/EvalCNNRNN_GRU.py
```

### GAIL
For training, please set the hyperparameters in the file and run any script:

```
python GAIL/practiceGAIL*.py
```

After training, use the same hyperparameters to retrieve the model, evaluate, and create the result CSVs using any script:

```
python GAIL/evalPracGAIL*.py
```

### Video Transformer
Note that the ViT-related code requires an additional repository to work. This repo is a submodule so it should automatically be cloned along with this repo. If this does not happen, please ensure you clone the vit-pytorch (https://github.com/kraw084/vit_pytorch) repo manually into the ViT folder.

ViT and ViViT models can be trained by running the following script:
```
python ViT/train_vit.py
```
By default, this script is set up to train a ViViT model with the best hyperparameters we found. To change the hyperparameters or dataset used, follow the comments in this script and adjust the variables defined in the ```if __name__ == "__main__"``` section of the code. Running this script will create a folder at save_dir (specified in the script), which will store the model checkpoints and tensor boards. 

To load a model checkpoint and calculate the validation and test set MSE run:
```
python ViT/test_vit.py
```

To create a CSV of ViT (unormalised) model predictions on the test set run the following script:
```
python ViT/create_csv.py
```
This will create a csv in the ViT/csvs folder. To change the model being used or the name of csv, edit the script accordingly.

### Pretrained Models
To run the pretrained models, please utilise any script in the ```transfer-learning``` directory and change its hyperparameters. For example:

```
python transfer-learning/ResNet/ResNet50-MultipleGame.py
python transfer-learning/MobileNet/mobilenetv4ConvSmall-MultipleGame.py
```
This finally saves the model which can be used to create the CSV files.

To create a CSV of Pretrained Models predictions on the test set run the following script:

```
python transfer-learning/ResNet/ResNet50_MultipleGame_Generate_CSVs.py
python transfer-learning/MobileNet/mobilenetv4ConvSmall-MultipleGame_Generate_CSVs.py
```


