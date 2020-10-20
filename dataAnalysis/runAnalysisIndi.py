from dataAnalysis.analysisData import AnalysisData
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

an= AnalysisData()
indi_data=an.evaluateIndividual('../FinalEvaluation/data/individual3/1/augmented.json')
#data.evaluateIndividual('../FinalEvaluation/data/individual2/1/raw.json')
#data.evaluateIndividual('../FinalEvaluation/data/individual2/1/augmented/train.json')


globalData= an.evaluateGlobal('../FinalEvaluation/data/global3/augmented/test.json')

#plotting mean of all
#analysis.plotTheData(analysis.meanAll(globalData))

#finding peaks and valleys
#https://peakutils.readthedocs.io/en/latest/tutorial_a.html##
#https://openwritings.net/pg/python/python-find-peaks-and-valleys-chart-using-scipysignalargrelextrema
#https://blog.ytotech.com/2015/11/01/findpeaks-in-python/


def plotMean(cut=False):

    for i in range(1,11):
        indi_data= an.evaluateIndividual('../FinalEvaluation/data/individual3/'+str(i)+'/augmented.json')
        mean_all = an.meanAll(indi_data)
        peaks_valleys = an.detectPeakandValley(mean_all, range(291))
        pyplot.figure(figsize=(3, 3))
        an.plotPeakValleyGen(mean_all, peaks_valleys, np.array(range(291)),cut)
        #pyplot.figure(figsize=(3, 3))
        #an.plotPeakValleyGen(mean_all, peaks_valleys, np.array(range(291)),True)
        #plt.show()

def plotMedian(cut=False):

    for i in range(1,11):
        indi_data= an.evaluateIndividual('../FinalEvaluation/data/individual3/'+str(i)+'/augmented.json')
        median_all = an.medianAll(indi_data)
        peaks_valleys = an.detectPeakandValley(median_all, range(291))
        pyplot.figure(figsize=(3, 3))
        an.plotPeakValleyGen(median_all, peaks_valleys, np.array(range(291)),cut)

plotMean(True)



plt.show()