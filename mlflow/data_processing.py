import sys
import argparse
import mlflow
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')

def load_dataset():
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    Y = pd.Series(wine.target)
    logger.info(f'Cancer data downloaded')
    return X, Y

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", default=0.2, type=float)
    test_size = parser.parse_args().test_size
    
    logger.info(f'Обработка данных с размером тестовой выборки {test_size}')
        
    X, Y = load_dataset()

    # рассчет метрик по датасету
    mlflow.log_metric('full_data_size', X.shape[0])
    mlflow.log_metric('features_count', X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    mlflow.log_metric('train_size', X_train.shape[0])
    mlflow.log_metric('test_size', X_test.shape[0])
    
    # выгрузка датасета
    train = X_train.assign(target=y_train)
    mlflow.log_text(train.to_csv(index=False),'datasets/train.csv')
    dataset_source_link = mlflow.get_artifact_uri('datasets/train.csv')
    dataset = mlflow.data.from_pandas(train, name='train', targets="target", source=dataset_source_link)
    mlflow.log_input(dataset)

    test = X_test.assign(target=y_test)
    mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')
    dataset_source_link = mlflow.get_artifact_uri('datasets/test.csv')
    dataset = mlflow.data.from_pandas(train, name='test', targets="target", source=dataset_source_link)
    mlflow.log_input(dataset)
    
    logger.info('Data preprocessing finished')