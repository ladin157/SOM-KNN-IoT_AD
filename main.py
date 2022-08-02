# import sys
import sys
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils.config import data_path, nbaiot_1K_data_path, nbaiot_10K_data_path, nbaiot_5K_data_path, \
    nbaiot_20K_data_path, nbaiot_data_path, dn_nbaiot
from utils.datasets import get_data, get_data_d3

from pyscripts.main_som import process_train_partial, som_test, load_data_test

if __name__ == '__main__':
    choose_folder = nbaiot_data_path
    print(choose_folder)
    train_index = 1
    # Load data on device 1
    data_benign, target_benign, data_gafgyt, target_gafgyt, data_mirai, target_mirai = get_data(
        choose_folder=choose_folder, choose_index=train_index)