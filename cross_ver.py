
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
#matplotlib inline

def train():
    bankdata = pd.read_csv('data/Features/Training/trainingbin_.csv')
    X = bankdata.drop('class_label', axis=1)  
    y = bankdata['class_label']
    from sklearn.model_selection import train_test_split	
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
	
    scaler = QuantileTransformer(output_distribution='uniform') 
    X= scaler.fit_transform(X)
    #y_train= scaler.fit_transform(y_train)
    #X_test= scaler.fit_transform(X_test)
    #y_test= scaler.fit_transform(y_test)	
    #from sklearn.ensemble import RandomForestClassifier
    #kf = KFold(1320, n_splits=10, shuffle=False)
    #for iteration, data in enumerate(kf, start=1):
        #print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))
    clf = svm.SVC(kernel='linear',  probability=True,C=512.0, gamma=0.0078125)
    pipe= Pipeline([('scaler', QuantileTransformer(output_distribution='uniform')), ('clf', clf)])
    y_pred = cross_val_predict(pipe, X, y, cv=24 ,method='predict')	
    #print("Accuracy:", scores.mean())
    '''clf.fit(X_train,y_train)	
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics	
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))	'''
    #y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics	
    print(confusion_matrix(y,y_pred))
    
    cnf_matrix = confusion_matrix(y,y_pred)	
    #print(classification_report(y_test,y_pred))
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)


# Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
# Specificity or true negative rate
    TNR = TN/(TN+FP) 
# Precision or positive predictive value
    PPV = TP/(TP+FP)
# Negative predictive value
    NPV = TN/(TN+FN)
# Fall out or false positive rate
    FPR = FP/(FP+TN)
# False negative rate
    FNR = FN/(TP+FN)
# False discovery rate
    FDR = FP/(TP+FP)

# Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
	
    print("FPR:",sum(FPR)/110)
    print("FNR:",sum(FNR)/110)
    print("ACC:",100*(sum(ACC))/110)	
    
    
    
if __name__ == "__main__":
    #main()
    train()
  

