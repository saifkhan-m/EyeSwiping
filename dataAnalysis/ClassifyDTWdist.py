from dtaidistance import dtw
import numpy as np
import pickle
from dataAnalysis.analysisData import AnalysisData
an= AnalysisData()
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

def check_similarity(seriesDTW, seriesType):
    distances=[]
    for i in seriesType:
        distances.append(distanceOfSeries(seriesDTW, i))
    return distances

def distanceOfSeries(s1, s2):
    return dtw.distance(s1, s2)

def getSeries(type):
    indi_data = an.evaluateIndividual('../FinalEvaluation/data/individual3/' + str(1) + '/augmented_train/train.json')

    series = []
    typeval = [data['pupilListSmoothed'][:291] for data in indi_data if type == data['type']]
    peaks_valleys = an.detectPeakandValley(np.array(
        [tuple(data['pupilListSmoothed'][:291]) for data in indi_data if type == data['type']]), range(291))
    for i in range(len(typeval)):
        y_val = typeval[i]
        y_val = np.array(y_val)
        lowestValley = np.where(y_val == min(y_val))[0][0]
        highestPeak = np.where(y_val == max(y_val[lowestValley:]))[0][0]
        peaks_valleys = np.concatenate(([highestPeak - lowestValley], [0]))
        x_value = np.array(range(lowestValley, highestPeak + 1))
        y_val = y_val[lowestValley: highestPeak + 1]
        series.append(y_val)
    return series

def getDTWseries(file):
    with open(file, 'rb') as f:
        s2=pickle.load( f)
    return s2

def get_testData(type):
    indi_data = an.evaluateIndividual('../FinalEvaluation/data/individual3/' + str(1) + '/augmented_train/test.json')

    series = []
    typeval = [data['pupilListSmoothed'][:291] for data in indi_data if type == data['type']]
    peaks_valleys = an.detectPeakandValley(np.array(
        [tuple(data['pupilListSmoothed'][:291]) for data in indi_data if type == data['type']]), range(291))
    for i in range(len(typeval)):
        y_val = typeval[i]
        y_val = np.array(y_val)
        lowestValley = np.where(y_val == min(y_val))[0][0]
        highestPeak = np.where(y_val == max(y_val[lowestValley:]))[0][0]
        peaks_valleys = np.concatenate(([highestPeak - lowestValley], [0]))
        x_value = np.array(range(lowestValley, highestPeak + 1))
        y_val = y_val[lowestValley: highestPeak + 1]
        series.append(y_val)
    return series
def smoothSeries(s):

    series = DataFrame (s,columns=['Column_Name'])
    # Tail-rolling average transform
    rolling = series.rolling(window=3)
    rolling_mean = rolling.mean()
    return rolling_mean
def classify():
    seriesType= getSeries(0)
    seriesDTW= getDTWseries('DTW_obj/DTW_Train0_.obj')
    d_bw_0_DTW= check_similarity(seriesDTW, seriesType)
    #print(d_bw_0_DTW)

    seriesType = getSeries(1)
    seriesDTW = getDTWseries('DTW_obj/DTW_Train1_.obj')
    d_bw_1_DTW = check_similarity(seriesDTW, seriesType)
    #print(d_bw_1_DTW)

    seriesType = getSeries(2)
    seriesDTW = getDTWseries('DTW_obj/DTW_Train2_.obj')
    d_bw_2_DTW = check_similarity(seriesDTW, seriesType)
    #print(d_bw_2_DTW)

    seriesType = getSeries(2)
    seriesDTW = getDTWseries('DTW_obj/DTW_Train1_.obj')
    diff_bw_12_DTW = check_similarity(seriesDTW, seriesType)
    #print(diff_bw_12_DTW)

    ser0=getDTWseries('DTW_obj/DTWFull_Train0_.obj')
    ser1 = getDTWseries('DTW_obj/DTWFull_Train2_.obj')
    ser2 = getDTWseries('DTW_obj/DTWFull_Train2_.obj')

    sim_bw_12 = distanceOfSeries(ser1, ser2)
    sim_bw_10 = distanceOfSeries(ser1, ser0)
    print(sim_bw_12)
    print(sim_bw_10)

    #plt.plot(sim_bw_12, color='red', label='Negative Vs Neutral')
    #plt.plot(ser1, color='blue', label='Neutral')
    #plt.plot(sim_bw_10, color='green', label='Positive Vs Neutral')
    #plt.figure()
    #plt.plot(smoothSeries(d_bw_2_DTW), color='red', label='Negative')
    #plt.plot(smoothSeries(d_bw_1_DTW), color='blue', label='Neutral')
    #plt.plot(smoothSeries(d_bw_0_DTW), color='green', label='Positive')
    #plt.legend()
    plt.figure(figsize=(10, 6))
    plt.plot(d_bw_2_DTW, color='red', label='Negative')
    plt.plot(d_bw_1_DTW, color='blue', label='Neutral')
    plt.plot(d_bw_0_DTW, color='green', label='Positive')
    plt.legend(loc='best')
    plt.show()

classify()