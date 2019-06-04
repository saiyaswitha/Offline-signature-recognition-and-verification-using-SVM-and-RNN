import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer  
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
#matplotlib inline

def train():
    bankdata = pd.read_csv('trainingbin_.csv')
    X = bankdata.drop('class_label', axis=1)  
    y = bankdata['class_label']
    from sklearn.model_selection import train_test_split	
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.feature_selection import SequentialFeatureSelector as sfs
    from sklearn.svm import SVC
    scaler = QuantileTransformer(output_distribution='uniform')
    X_train= scaler.fit_transform(X_train)
    #y_train= scaler.fit_transform(y_train)
    X_test= scaler.fit_transform(X_test)
    #y_test= scaler.fit_transform(y_test)	
    #from sklearn.ensemble import RandomForestClassifier
    clf = svm.SVC(kernel='linear',  C=8192) 	
    #clf = SVC(kernel='linear')	
    #clf = RandomForestClassifier(n_estimators=100)
    sfs1 = sfs(clf,k_features=10,forward=True, floating=False,verbose=2,scoring='accuracy')
    sfs1 = sfs1.fit_transform(X_train, y_train)
    X_train_rfe = sfs1.fit_transform(X_train)
    X_test_rfe = sfs1.fit_transform(X_test)
    #clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=11)
    clf.fit(X_train_rfe, y_train)
    y_train_pred = clf.predict(X_train_rfe)
    from sklearn.metrics import accuracy_score as acc
    print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))   
    y_test_pred = clf.predict(X_test_rfe)
    print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))
    #svclassifier = SVC(kernel='rbf', gamma='auto', degree=3)  
    
    #y_pred = test.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics
    from sklearn.metrics import accuracy_score as acc	
    print(confusion_matrix(y_test,y_test_pred))  
    cnf_matrix = confusion_matrix(y_test,y_test_pred)	
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
	
    print("FNR:",sum(FNR)/55)
    print("FPR:",sum(FPR)/55)
    print("ACC:",100*(sum(ACC)/55))
    
if __name__ == "__main__":
    #main()
    train()
    

