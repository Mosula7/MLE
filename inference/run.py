import os
import json
from dotenv import load_dotenv
load_dotenv() # env variable wasn't being loaded without this function

import pandas as pd 

import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from sklearn.metrics import f1_score, accuracy_score

def get_data(inference_x, inference_y):
    X = pd.read_csv(os.path.join('data', inference_x))
    X = torch.tensor(X.values).float()

    y = pd.read_csv(os.path.join('data', inference_y))
    y = torch.tensor(y.values).flatten()
    
    return X, y

def main():
    logging.info('loading env file')
    CONF_FILE = os.getenv('CONF_PATH')
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)

    model_name = conf['general']['model_name']
    
    logging.info('loading model')
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'models'))

    try:
        model = torch.load(os.path.join(MODEL_DIR, f'{model_name}.pt'))
    except:
        raise FileExistsError('try training the model first')

    inference_x = conf['inference']['X']
    inference_y = conf['inference']['y']

    X, y = get_data(inference_x, inference_y)
    y_pred = torch.argmax(model(X), axis=1)

    logging.info(f'accuracy score: {accuracy_score(y, y_pred)}')
    logging.info(f'f1 score: {f1_score(y, y_pred, average="weighted")}')




if __name__ == '__main__':

    main()