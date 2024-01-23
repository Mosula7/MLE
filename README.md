# MLE project for IRIS Dataset

In the directory includes the following files:
* **IMPORTANT** create .env file. In this file you should have `CONF_PATH=settings.json`
  
* .gitignore - I'm not uploading model files, inference/prediction csvs, pycache and notebooks used for testing. (I did incude a data folder as specifed in the homework)
* requirements.txt - all the dependencies needed
* settings.json - almsot all the values that would need to be hard coded, for example: test set ratio, random state, learning rate, names of inferance datasets, etc.
  
The project consists of three main parts: 
* Data processing
* Model training
* Inference

## Data Processing
To run the script in the terminal locally, if you want to get different data folders, than what's already uploaded
```
python data_process/split_iris.py
``` 

being in the MLE directory and it will create a data folder split IRIS data and upload X_train, X_test, y_train, y_test data to it. You can modify random dtate and test size from the settings. IMPORTANT: this script needs to be ran first, before training the model, if you don't copy the data, or you delete the data folder. There is no docker container for this part because it wasn't mentioned in the homework.

## Training
You can run this script in two ways:
* To run the script in the terminal locally
```
python training/train.py
```
* You can also build a docker image with the following command and run the script on a container
```
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
* You can move trained model from container into a local machine by
```
docker cp <container_id>:/app/models/<model_name>.pt ./models
```
**NOTE**: model extension should be .pt

This script gets the data from the data folder and trains a pytorch model, then it creates a models folder and saves the model in this folder, you can specify the model name in settings. You will need the model name for the inference part, so it is important. Then it logs f1 and accuracy score on the test set. The model has two hidden layers, and you can choose how many units are going to be in them by specifing them in n_units_1 and n_units_2 in the settings.json, you can also adjust the learning rate and number of epochs here. 


## Inference
You can run this script in two ways:
* To run the script in the terminal locally
```
python inference/run.py
```
* You can also build a docker image with the following command and run the script on a container
```
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pt --build-arg settings_name=settings.json -t inference_image .
```
**NOTE**: the extention of the model needs to be .pt and it needs to be specified

This script gets the inference data specified in settings, logs models f1 and accuracy score on this data and then saves the predictions in the results folder as a csv. Note: the predictions aren't probabilites they are classes (already "Argmaxed")


## Data
This folder includes train test split data. I'm using test dataset for inferance and model evaluation after training, because the dataset is already extremly small. You can add extra inference X, y data in this folder, update names of datasets in the settings.json file ("X" and "y" in the inference part) and the inference would be ran on that dataset



