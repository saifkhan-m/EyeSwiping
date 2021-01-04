from dataAnalysis.analysisData import AnalysisData
import matplotlib.pyplot as plt
from dataAnalysis.DBA import performDBA
from matplotlib import pyplot
import numpy as np
import pickle
from dtaidistance import dtw

an= AnalysisData()
def calculateDBA(type):
    indi_data= an.evaluateIndividual('../FinalEvaluation/data/individual3/' + str(1) + '/augmented.json')
    series=[]

    for i in indi_data:
        if i['type']==type:
            series.append(i['pupilListSmoothed'][:290])
    series = np.array(series)
    count=0
    for s in series:
        plt.plot(range(0,len(s)), s)
        count+=1
    plt.draw()

    #calculating average series with DBA
    average_series = performDBA(series,10)

    with open('DTWFull_Train' + str(type) + '_.obj', 'wb') as f:
        pickle.dump(average_series, f)

    #plotting the average series
    plt.figure()
    plt.plot(range(0,len(average_series)), average_series)
    plt.show()

def DBALIB(type):
    indi_data = an.evaluateIndividual('../FinalEvaluation/data/individual3/' + str(1) + '/augmented.json')
    series = []
    print(indi_data)
    for i in indi_data:
        if i['type'] == type:
            series.append(i['pupilListSmoothed'][:290])
    series = np.array(series)
    count = 0
    for s in series:
        print(count, end=' ')
        plt.plot(range(0, len(s)), s)
        count += 1
    plt.draw()

    with open('DTWFull_Train'+str(type)+'_.obj', 'rb') as f:
        s2=pickle.load( f)
    plt.figure()
    plt.plot(range(0, len(s2)), s2)
    plt.show()

def calculateDBAPeakAndValley(type):
    indi_data = an.evaluateIndividual('../FinalEvaluation/data/individual3/' + str(1) + '/augmented_train/train.json')

    series = []
    typeval =[data['pupilListSmoothed'][:291]for data in indi_data if type == data['type']]
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
        plt.plot(x_value, y_val)
    average_series = performDBA(series)

    with open('DTW_Train' + str(type) + '_.obj', 'wb') as f:
        pickle.dump(average_series, f)

    plt.figure()
    plt.plot(range(0, len(average_series)), average_series)
    plt.show()
def DBALIBPeakValley(type):
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
        #plt.plot(x_value, y_val)

    with open('../dataAnalysis/DTW_obj/DTW_Train'+str(type)+'_.obj', 'rb') as f:
        s2=pickle.load( f)
    plt.figure()
    plt.plot(range(0, len(s2)), s2)
    plt.show()
#calculateDBA(2)
#calculateDBAPeakAndValley(2)

DBALIBPeakValley(2)

plt.show()
