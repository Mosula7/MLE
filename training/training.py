import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv() # env variable wasn't being loaded without this function

import pandas as pd 

import torch
import torch.nn as nn

import logging
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


def get_data():
    X = pd.read_csv(os.path.join('data', 'X_train.csv'))
    X = torch.tensor(X.values).float()

    y = pd.read_csv(os.path.join('data', 'y_train.csv'))
    y = torch.tensor(y.values).flatten()
    
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
    

    logging.info('process finished successfully')
            
if __name__ == '__main__':
    main()