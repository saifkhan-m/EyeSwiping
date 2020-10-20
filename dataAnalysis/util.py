from dataAnalysis.analysisData import AnalysisData
import pandas as pd
import numpy as np
#an = AnalysisData()
def get_train_test_DS():
    train = pd.read_json('../FinalEvaluation/data/global3/raw/train.json')
    train_small0 = train.loc[(train['type'] == 0)].head(495)
    train_small1 = train.loc[(train['type'] == 1)]
    train = pd.concat([train_small0, train_small1], sort=False)

    test = pd.read_json('../FinalEvaluation/data/global3/raw/test.json')
    test_small0 = test.loc[(test['type'] == 0)].head(50)
    test_small1 = test.loc[(test['type'] == 1)].head(50)
    test = pd.concat([test_small0, test_small1], sort=False)
    return (train['pupilListSmoothed'],train[ 'type']),(test['pupilListSmoothed'],test[ 'type'])

def peakvalley_slice(sequence):
    y_val = np.array(sequence)
    lowestValley = np.where(y_val == min(y_val))[0][0]
    highestPeak = np.where(y_val == max(y_val[lowestValley:]))[0][0]
    y_val = y_val[lowestValley: highestPeak + 1]
    return y_val

#peak_N_valley_DS()
