import gc
import time

import numpy as np
from hyperopt import tpe
from sklearn.model_selection import train_test_split

from som.som_common import train_som, test_som
from utils.config import nbaiot_1K_data_path
from utils.datasets import get_train_test_data, get_test_data
from utils.preprocessing import scale_data, normalize_data

'''
Data preprocessing:
1. Chi thu voi benign va mirai
    Gafgyt co the dung nhu du lieu testing
2. Chi thu voi benign va gafgyt
Flow:
1. Load data
    - Combine data
    - Get train, test data
2. Scale data
3. Normalize data
4. Train SOM
5. Optimize SOM
6. Classify
- Using default algorithm
- Using KNN algorithm


'''


def som_test(som, winmap, outliers_percentage, scaler, X_test, y_test, using_algo=False, algo='KNN'):
    # scal data
    print("Shape: ", X_test.shape, y_test.shape)
    print("----------------------Test is starting----------------------")
    print("Scale data")
    X_test = scaler.transform(X_test)
    print("Shape: ", X_test.shape, y_test.shape)
    # normalize data
    print("Normalize data")
    _, X_test = normalize_data(None, X_test)
    print("Shape: ", X_test.shape, y_test.shape)
    print("Testing")
    test_som(som=som, winmap=winmap, outliers_percentage=outliers_percentage, X_test=X_test,
             y_test=y_test, using_algo=using_algo, algo=algo)
    print("----------------------Test Done----------------------")


def load_data_test(choose_folder, test_index, gafgyt=True, benign_included=True):
    X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=gafgyt,
                                   benign_included=benign_included)
    return X_test, y_test


def process_train_partial(X_train, y_train, algo='tpe'):
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("--------------Training and testing in the same device----------------")
    print(X_train.shape, y_train.shape)
    start_time = time.time()
    # print("====================================Train scaler==================================")
    print("------------Scale data-----------------")
    scaler, X_train_scaled, _ = scale_data(X_train=X_train, X_test=None)
    print(X_train_scaled.shape, y_train.shape)
    print("X_train_scaled")
    print(X_train_scaled[:10])

    # print("=================================Normalization====================================")
    print("---------Normalize data--------------")
    # normalize data from scaled data
    X_train_normalized, _ = normalize_data(X_train=X_train_scaled, X_test=None)
    print(X_train_normalized.shape, y_train.shape)

    # train SOM
    print("--------------------Train SOM on normalized data--------------")
    som, winmap, outliers_percentage = train_som(X_train=X_train_normalized, y_train=y_train, algo=algo)
    end_time = time.time()
    print('Total train time: {}'.format(end_time - start_time))
    return som, winmap, outliers_percentage, scaler


def process_test_partial(X_test, y_test, som, winmap, outliers_percentage, scaler, encoder=None, pca=None):
    som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
             y_test=y_test, encoder=encoder, pca=pca)


if __name__ == '__main__':
    pass
    # test gafgyt
    # main(choose_folder=nbaiot_1K_data_path, choose_index=1,method='ae', gafgyt=True)
    # main(choose_folder=nbaiot_data_path, choose_index=1)
