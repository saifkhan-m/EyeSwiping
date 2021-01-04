from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sktime.classification.frequency_based import RandomIntervalSpectralForest

from TimeSeriesEval.util import train_test_Data

average_method ='macro'
def getMetrics(folderWithTestTrainData):
    X_train, y_train, X_test, y_test = train_test_Data(folderWithTestTrainData)

    ##############

    knn = RandomIntervalSpectralForest()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average=average_method),
            precision_score(y_test, y_pred, average=average_method),
            recall_score(y_test, y_pred, average=average_method)]

def getName():
    return 'KNeighborsClassification'

folderWithTestTrainData='../FinalEvaluation/data/global3/augmented/'
print(getMetrics(folderWithTestTrainData))
