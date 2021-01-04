
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sktime.transformers.series_as_features.reduce import Tabularizer
from TimeSeriesEval.util import train_test_Data
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf

average_method ='macro'
def ar_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    lags = np.minimum(len(x) - 1, maxlag) // 2
    model = AutoReg(endog=x, trend="n", lags=lags)
    return model.fit().params.ravel()

def acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags, fft=True).ravel()

def powerspectrum(x):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[:ps.shape[0] // 2].ravel()

def getMetrics(folderWithTestTrainData):
    X_train, y_train, X_test, y_test = train_test_Data(folderWithTestTrainData)

    ##############

    steps = [
        ('segment', RandomIntervalSegmenter(n_intervals=1, min_length=5)),
        ('transform', FeatureUnion([
            ('ar', RowTransformer(FunctionTransformer(func=ar_coefs, validate=False))),
            ('acf', RowTransformer(FunctionTransformer(func=acf_coefs, validate=False))),
            ('ps', RowTransformer(FunctionTransformer(func=powerspectrum, validate=False)))
        ])),
        ('tabularise', Tabularizer()),
        ('clf', DecisionTreeClassifier())
    ]
    rise_tree = Pipeline(steps)

    rise = TimeSeriesForestClassifier(estimator=rise_tree, n_estimators=10)
    rise.fit(X_train, y_train)
    y_pred = rise.predict(X_test)

    return [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average=average_method),
            precision_score(y_test, y_pred, average=average_method),
            recall_score(y_test, y_pred, average=average_method)]

def getName():
    return 'RISEClassification'

#folderWithTestTrainData='../FinalEvaluation/data/global3/augmented_train/'
#print(getMetrics(folderWithTestTrainData))
