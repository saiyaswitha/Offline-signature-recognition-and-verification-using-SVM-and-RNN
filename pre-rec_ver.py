print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.svm import SVC 
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
from scipy import interp


def train():
    bankdata = pd.read_csv('data/Features/Training/trainingbin_.csv')
    X = bankdata.drop('class_label', axis=1)  
    y = bankdata['class_label']
    y = label_binarize(y, classes=[0,1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109])
    n_classes = y.shape[1]
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    

# shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)

# Learn to predict each class against the other
    scaler = QuantileTransformer(output_distribution='uniform')
    X_train= scaler.fit_transform(X_train)
    #y_train= scaler.fit_transform(y_train)
    X_test= scaler.fit_transform(X_test)
    #y_test= scaler.fit_transform(y_test)	
    #from sklearn.ensemble import RandomForestClassifier
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,C=512.0, gamma=0.0078125,random_state=random_state))	
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    #y_score = clf.predict(X_test)
# Compute ROC curve and ROC area for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
   
    average_precision["micro"] = average_precision_score(y_test[:, i], y_score[:, i])
    
    
##############################################################################
# Plot of a ROC curve for a specific class
    plt.figure()
    
    plt.plot(precision["micro"], recall["micro"], color='darkorange',
              label='ROC curve (area = %0.2f)' % average_precision["micro"])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR Curve _Verification')
    plt.legend(loc="lower right")
    plt.show()


##############################################################################
# Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
    all_precision = np.unique(np.concatenate([precision[i] for i in range(n_classes)]))
    #all_fpr1 = np.concatenate([fpr[i] for i in range(n_classes)])
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    #all_tpr1 = np.concatenate([tpr[i] for i in range(n_classes)])
# Then interpolate all ROC curves at this points
    mean_recall = np.zeros_like(all_precision)
    for i in range(n_classes):
        mean_recall += interp(all_precision, precision[i], recall[i])

# Finally average it and compute AUC
    mean_recall /= n_classes

    precision["macro"] = all_precision
    recall["macro"] = mean_recall
    average_precision["macro"] = average_precision_score(y_test[:, i], y_score[:, i])
    print((sum(precision["macro"])/len(precision["macro"])),sum(recall["macro"])/len(recall["macro"]),average_precision["macro"])
    plt.figure()
    
    plt.plot(precision["macro"], recall["macro"], color='darkorange',
              label='ROC curve (area = %0.2f)' % average_precision["macro"])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR Curve _Verification')
    plt.legend(loc="lower right")
    plt.show()
# Plot all ROC curves
    plt.figure()
    plt.plot(precision["micro"], recall["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(precision["macro"], recall["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(average_precision["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(precision[i], recall[i], color=color, 
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, average_precision[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('PR_VER')
    plt.legend(loc="lower right")
    plt.show()

    
if __name__ == "__main__":
    #main()
    train()
    

