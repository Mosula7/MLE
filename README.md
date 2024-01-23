# MLE project for IRIS Dataset
The project consists of three main parts: 
* Data processing
* Model training
* Inference

In the repository you will also find some other files:
* .gitignore - I'm not uploading model files, inference/prediction csvs, pycache and notebooks used for testing. (I did incude a data folder as specifed in the homework)
* requirements.txt - all the dependencies needed
* settings.json - almsot all the values that would need to be hard coded, for example: test set ratio, random state, learning rate, names of inferance datasets, etc.

## Data
This folder includes train test split data. I'm using test dataset for inferance and model evaluation after training, because the dataset is already extremly small. You can add extra inference X, y data in this folder, update names of datasets in the settings.json file ("X" and "y" in the inference part) and the inference would be ran on that dataset

## Data Processing
To run the script you can simply run "python data_process/split_iris.py" in the terminal while being in the MLE directory and it will create a data folder and upload X_train, X_test, y_train, y_test data to it. IMPORTANT: this script needs to be ran first, before training the model, if you don't copy the data, or you delete the data folder. There is no docker container for this part because it wasn't mentioned in the homework.

## Training
This script can be ran locally by "python training/train.py" or you can also build a docker image with the following command: TODO. 

This script gets the data from the data folder and trains a pytorch model, then it creates a models folder and saves the model in this folder, you can specify the model name in settings. You will need the model name for the inference part, so it is important. Then it logs f1 and accuracy score on the test set. The model has two hidden layers, and you can choose how many units are going to be in them by specifing them in n_units_1 and n_units_2 in the settings.json, you can also adjust the learning rate and number of epochs here. 


## Inference
This script can be ran locally by "python inference/run.py" or you can also build a docker image with the following command: TODO. 

This script gets the inference data specified in settings, logs models f1 and accuracy score on this data and then saves the predictions in the results folder



