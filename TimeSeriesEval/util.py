import numpy as np
import json
import pandas as pd

def train_test_Data(folder):
    with open(folder + 'train.json', "r") as train_file:
        inputdata = json.load(train_file)
    train_data = np.array(
        [[pd.Series(tuple(data['pupilListSmoothed'][:280])), str(data['type'])] for data in inputdata])
    train_data = pd.DataFrame(data=train_data, columns=["time_Series", "label"])

    with open(folder + 'test.json', "r") as test_file:
        inputdata = json.load(test_file)
    test_data = np.array([[pd.Series(tuple(data['pupilListSmoothed'][:280])), str(data['type'])] for data in inputdata])
    test_data = pd.DataFrame(data=test_data, columns=["time_Series", "label"])

    return pd.DataFrame(train_data['time_Series']), train_data['label'], pd.DataFrame(test_data['time_Series']), \
           test_data['label']


def train_test_DataTrainTest(folder):
    with open(folder + 'train.json', "r") as train_file:
        inputdata = json.load(train_file)
    train_data = np.array([[tuple(data['pupilListSmoothed'][:280]), str(data['type'])] for data in inputdata])
    train_data = pd.DataFrame(data=train_data, columns=["time_Series", "label"])

    with open(folder + 'test.json', "r") as test_file:
        inputdata = json.load(test_file)
    test_data = np.array([[tuple(data['pupilListSmoothed'][:280]), str(data['type'])] for data in inputdata])
    test_data = pd.DataFrame(data=test_data, columns=["time_Series", "label"])

    return train_data, test_data