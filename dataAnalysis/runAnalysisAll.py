from dataAnalysis.analysisData import AnalysisData
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np

#plotting mean of all
#analysis.plotTheData(analysis.meanAll(globalData))

#finding peaks and valleys
#https://peakutils.readthedocs.io/en/latest/tutorial_a.html##
#https://openwritings.net/pg/python/python-find-peaks-and-valleys-chart-using-scipysignalargrelextrema
#https://blog.ytotech.com/2015/11/01/findpeaks-in-python/

an= AnalysisData()

def plotMedian(cut=False):
    globalData= an.evaluateGlobal('../FinalEvaluation/data/global3/augmented/test.json')
    median_all=an.medianAll(globalData)
    peaks_valleys = an.detectPeakandValley(median_all, range(291))
    pyplot.figure(figsize=(10, 6))
    an.plotPeakValleyGen(median_all, peaks_valleys, np.array(range(291)),cut)

def plotMean(cut=False):
    globalData= an.evaluateGlobal('../FinalEvaluation/data/global3/augmented/test.json')
    mean_all=an.meanAll(globalData)
    peaks_valleys = an.detectPeakandValley(mean_all, range(291))
    pyplot.figure(figsize=(10, 6))
    an.plotPeakValleyGen(mean_all, peaks_valleys, np.array(range(291)),cut)

plotMean(True)
#plotMean()


#plotMedian()
#plotMedian(True)
plt.show()