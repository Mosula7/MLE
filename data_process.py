import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


if not os.path.exists('data'):
    os.mkdir('data')


def split_data(df: pd.DataFrame, target: str, test_size: float, 
               val_size: float=None, random_state:int = 0):
    """
    returns (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    if not val_size:
        val_size = test_size / (1 - test_size)

    train_val, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val[target], random_state=random_state)

    X_train = train[train.columns.drop(target)]
    X_val = val[val.columns.drop(target)]
    X_test = test[test.columns.drop(target)]

    y_train = train[target]
    y_val = val[target]
    y_test = test[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    test_val_size = 0.1
    random_state = 42

    target_name = 'target'
    iris = load_iris()

    # Convert the iris dataset to a pandas dataframe
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add the target variable to the dataframe
    df[target_name] = iris.target

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target = target_name, test_size=test_val_size, 
                                                                random_state=random_state)

    X_train.to_csv('data/X_train.csv', index=False)
    X_val.to_csv('data/X_val.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)

    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


if __name__ == '__main__':
    main()
