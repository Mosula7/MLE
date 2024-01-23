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
This folder includes train test split data. I'm using test dataset for inferance and model evaluation after training, because the dataset is already extremly small. You can add extra inference X, y data in this folder, update names of datasets in the settings.json file and the inference would be ran on that dataset
