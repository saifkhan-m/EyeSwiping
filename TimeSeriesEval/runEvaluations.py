from TimeSeriesEval import TimeSeriesForestClassification, RocketClassification, RISEClassification, \
    KNeighborsClassification, WeaselClassification#, BOSSClassification, ShapeletTransformClassification

print('Starting Evaluations!!!')
folderWithTestTrainDataaugmented='../FinalEvaluation/data/global3/augmented/'
folderWithTestTrainDataaugmentedtrain ='../FinalEvaluation/data/global3/augmented_train/'
#classifiers =[TimeSeriesForestClassification, RocketClassification, RISEClassification, KNeighborsClassification,
#              BOSSClassification, ShapeletTransformClassification, WeaselClassification]

classifiers =[TimeSeriesForestClassification, RocketClassification, RISEClassification, KNeighborsClassification, WeaselClassification]

print('Classification Metrics of the algorithms on augmented data are given as follows')
for i in classifiers:
    m = list(map(str, i.getMetrics(folderWithTestTrainDataaugmented)))
    print(i.getName() + ' : (Accuracy - '+m[0]+', F1 score - '+m[1]+', Precision - '+m[2]+', Recall - '+m[3])

print('Finished Evaluations!!!')

print('Classification Metrics of the algorithms on augmented train data are given as follows')
for i in classifiers:
    m = list(map(str, i.getMetrics(folderWithTestTrainDataaugmentedtrain)))
    print(i.getName() + ' : (Accuracy - '+m[0]+', F1 score - '+m[1]+', Precision - '+m[2]+', Recall - '+m[3])

print('Finished Evaluations!!!')