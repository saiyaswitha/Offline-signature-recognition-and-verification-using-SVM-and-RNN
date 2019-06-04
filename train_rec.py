import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
#matplotlib inline

def train():
    bankdata = pd.read_csv('data/trainingbin_.csv')
    X = bankdata.drop('class_label', axis=1)  
    y = bankdata['class_label']
    from sklearn.model_selection import train_test_split	
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    scaler = QuantileTransformer(output_distribution='uniform')
    X_train= scaler.fit_transform(X_train)
    #y_train= scaler.fit_transform(y_train)
    X_test= scaler.fit_transform(X_test)
    #y_test= scaler.fit_transform(y_test)	
    #from sklearn.ensemble import RandomForestClassifier
    clf = svm.SVC(kernel='linear',  C=512.0, gamma=0.0078125)          
    clf.fit(X_train,y_train)	
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics	
    print(confusion_matrix(y_test,y_pred))

    cnf_matrix = confusion_matrix(y_test,y_pred)	
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
	
    print("FPR:",(FPR))
    print("FNR:",(FNR))
    print("ACC:",(ACC))	

    
if __name__ == "__main__":
    #main()
    train()
    

