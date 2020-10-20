import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import signal
import peakutils
from peakutils.plot import plot as pplot

class AnalysisData:

    def __init__(self):
        pass

    def readJSON(self,filenameWithPath):
        with open(filenameWithPath, "r") as json_file:
            data = json.load(json_file)
        return data

    def mean(self,inputdata,category):
        if category not in {data['type'] for data in inputdata}:
            print('category not found')
            return []

        dataArray = np.array([tuple(data['pupilListSmoothed'][:291]) for data in inputdata if category == data['type']])
        dataArray = np.round(dataArray, 1)
        return np.mean(dataArray, axis=0)

    def plotTheData(self,val1,val2,val3):
        plt.figure(figsize=(3, 3))
        plt.plot(val1, color='red', label='Negative')
        plt.plot(val2, color='blue', label='Neutral')
        plt.plot(val3, color='green', label='Positive')
        plt.legend()

        # for ploting the valeys and peaks
        # input takes x ,y signal and ndarray of peaks and values
        # return nothong
    def plotPeakValley(self, x, y, peaks=None, valleys=None):
        colors=['blue','red', 'green']
        labels=['Negative', 'Neutral', 'Positive']

        peaks_valleys = np.concatenate((peaks, valleys))
        pplot(x, y, peaks_valleys)

    def plotPeakValleyGen(self, mean_all,peaksandvalleys, x_value, cut=False):
        colors = ['red', 'blue', 'green']
        labels = ['Negative', 'Neutral', 'Positive']

        for i in range(len(mean_all)):
            y_val=mean_all[i]
            if cut:
                lowestValley = np.where(y_val == min(y_val))[0][0]
                highestPeak = np.where(y_val == max(y_val[lowestValley:]))[0][0]
                peaks_valleys=np.concatenate(([highestPeak-lowestValley] , [0]))
                x_value=np.array(range(lowestValley, highestPeak+1))
                y_val=y_val[lowestValley: highestPeak + 1]
            else:
                peaks_valleys = np.concatenate((peaksandvalleys[i][0], peaksandvalleys[i][1]))
            plt.plot( x_value,y_val, color=colors[i], label=labels[i])

            marker_x = x_value.iloc[peaks_valleys] if hasattr(x_value, "iloc") else x_value[peaks_valleys]
            marker_y = y_val.iloc[peaks_valleys] if hasattr(y_val, "iloc") else y_val[peaks_valleys]
            plt.plot(marker_x, marker_y, "r+", ms=5, mew=2)
            plt.legend()

    def meanAll(self,inputdata):
        types= {data['type'] for data in inputdata}
        result=[]
        for type in types:
            dataArray = np.array(
                [tuple(data['pupilListSmoothed'][:291]) for data in inputdata if type == data['type']])
            dataArray = np.round(dataArray, 3)
            result.append(np.mean(dataArray, axis=0))
        return result
    def category(self, inputdata):
        return {data['type'] for data in inputdata}

    def median(self,inputdata,category):
        if category not in {data['type'] for data in inputdata}:
            print('category not found')
            return []
        dataArray = np.array([tuple(data['pupilListSmoothed'][:291]) for data in inputdata if category == data['type']])
        dataArray = np.round(dataArray, 1)
        return np.median(dataArray, axis=0)

    def medianAll(self,inputdata):
        types= {data['type'] for data in inputdata}
        result=[]
        for type in types:
            dataArray = np.array(
                [tuple(data['pupilListSmoothed'][:291]) for data in inputdata if type == data['type']])
            dataArray = np.round(dataArray, 1)
            result.append(np.median(dataArray, axis=0))
        return result

    def evaluateIndividual(self,filenameWithPath):
        individualData = self.readJSON(filenameWithPath)
        #print('jhgdsfjbsjf',individualData)
        return individualData

        #mean0 = self.mean(individualData, 0)
        #mean1 = self.mean(individualData, 1)
        #mean2 = self.mean(individualData, 2)
        #median0 = self.median(individualData, 0)
        #median1 = self.median(individualData, 1)
        #median2 = self.median(individualData, 2)

        #self.plotTheData(median0,median1,median2)
 #       self.plotTheData(*self.meanAll(individualData)
#        counter= 0

    #takes a list of multiple (y signal and x signal)
    # return a 2d list containing peaks and valleys of all
    #signals provided in the input
    def detectPeakandValley(self,y_val, x_val=None):
        peaks_vallleys=[]
        for item in y_val:
            peak_indexes = peakutils.indexes(item, thres=-0.5, min_dist=50)
            valley_indexes = signal.argrelextrema(item, np.less)[0]
            peaks_vallleys.append([peak_indexes, valley_indexes])
        return peaks_vallleys



    def evaluateGlobal(self,filenameWithPath):
        return self.readJSON(filenameWithPath)

data= AnalysisData()
for i in range(1,11):
    #data.evaluateIndividual('../FinalEvaluation/data/individual3/'+str(i)+'/augmented.json')
    pass
plt.show()
#data.evaluateIndividual('../FinalEvaluation/data/individual3/1/augmented.json')
#data.evaluateIndividual('../FinalEvaluation/data/individual2/1/raw.json')
#data.evaluateIndividual('../FinalEvaluation/data/individual2/1/augmented/train.json')