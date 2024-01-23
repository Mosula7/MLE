import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv() # env variable wasn't being loaded without this function

import pandas as pd 

import torch
import torch.nn as nn

import logging
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(n_units_1, n_units_2, learning_rate):
    model = nn.Sequential(
        nn.Linear(4, n_units_1),
        nn.ReLU(),
        nn.Linear(n_units_1, n_units_2),
        nn.ReLU(),
        nn.Linear(n_units_2, 3)
    )

    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return lossfun, optimizer, model


def get_data(train=True):
    if train:
        x_name = 'X_train.csv'
        y_name = 'y_train.csv'
    else:
        x_name = 'X_test.csv'
        y_name = 'y_test.csv'

    try:
        X = pd.read_csv(os.path.join('data', x_name))
        X = torch.tensor(X.values).float()

        y = pd.read_csv(os.path.join('data', y_name))
        y = torch.tensor(y.values).flatten()
    except:
        raise FileNotFoundError('try creating data files first')

    
    return X, y


def main():
    logging.info('loading env file')
    CONF_FILE = os.getenv('CONF_PATH')
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'models'))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    logging.info('getting data')
    X, y = get_data()
    
    n_units_1 = conf['training']['n_units_1']
    n_units_2 = conf['training']['n_units_2']
    learning_rate = conf['training']['learning_rate']
    epochs = conf['training']['epochs']

    model_name = conf['general']['model_name']

    logging.info('creating model')
    lossfun, optimizer, model = create_model(n_units_1, n_units_2, learning_rate)
	
    logging.info('training model')
    for _ in range(epochs):
		# forward pass
        yHat = model(X)

        # compute loss
        loss = lossfun(yHat, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    logging.info('saving model')

    #model_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    torch.save(model, os.path.join(MODEL_DIR, f'{model_name}.pt'))

    X, y = get_data(train=False)
    y_pred = torch.argmax(model(X), axis=1)
    logging.info(f'accuracy score (test set): {accuracy_score(y, y_pred)}')
    logging.info(f'f1 score (test set): {f1_score(y, y_pred, average="weighted")}')

    

    logging.info('process finished successfully')
            
if __name__ == '__main__':
    main()