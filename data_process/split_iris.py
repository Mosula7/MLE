import os
import sys
import json 
import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def main():
    CONF_FILE = os.getenv('CONF_PATH')

    with open(CONF_FILE, "r") as file:
        conf = json.load(file)

    test_size = conf['data_process']['test_size']
    random_state = conf['data_process']['random_state']

    target_name = 'target'
    iris = load_iris()

    # Convert the iris dataset to a pandas dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target variable to the dataframe
    df[target_name] = iris.target

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(ROOT_DIR))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


    train, test = train_test_split(df, stratify=df[target_name], test_size=test_size, random_state=random_state)

    X_train = train[train.columns.drop(target_name)]
    X_test = test[test.columns.drop(target_name)]

    y_train = train[target_name]
    y_test = test[target_name]
    
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
    
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)


if __name__ == '__main__':
    main()