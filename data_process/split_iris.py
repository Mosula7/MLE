import os
import json 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from dotenv import load_dotenv
load_dotenv() # env variable wasn't being loaded without this function

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def main():
    logging.info('loading env file')
    CONF_FILE = os.getenv('CONF_PATH')
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)
    
    logging.info('getting setting')
    test_size = conf['data_process']['test_size']
    random_state = conf['data_process']['random_state']
    
    logging.info('importing iris data')
    target_name = 'target'
    iris = load_iris()

    # Convert the iris dataset to a pandas dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target variable to the dataframe
    df[target_name] = iris.target

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'data'))

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    logging.info('splitting iris data')
    
    train, test = train_test_split(df, stratify=df[target_name], test_size=test_size, random_state=random_state)
    logging.info(f'split data with shape {df.shape} into train test sets with shapes: {train.shape}, {test.shape}')

    X_train = train[train.columns.drop(target_name)]
    X_test = test[test.columns.drop(target_name)]

    y_train = train[target_name]
    y_test = test[target_name]
    
    logging.info('saving iris data')
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
    
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)

    logging.info('process finished successfully')


if __name__ == '__main__':
    main()