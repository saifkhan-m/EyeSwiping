import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.transformers.series_as_features.rocket import Rocket

from TimeSeriesEval.util import train_test_Data

average_method ='macro'
def getMetrics(folderWithTestTrainData):
    X_train, y_train, X_test, y_test = train_test_Data(folderWithTestTrainData)

    ##############

    rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                   normalize=True)
    classifier.fit(X_train_transform, y_train)
    X_test_transform = rocket.transform(X_test)
    y_pred = classifier.predict(X_test_transform)

    return [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average=average_method),
            precision_score(y_test, y_pred, average=average_method),
            recall_score(y_test, y_pred, average=average_method)]

def getName():
    return 'RocketClassification'

#folderWithTestTrainData='../FinalEvaluation/data/global3/augmented_train/'
#print(getMetrics(folderWithTestTrainData))
