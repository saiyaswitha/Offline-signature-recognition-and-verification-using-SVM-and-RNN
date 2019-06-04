
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#matplotlib inline

def train():
    bankdata = pd.read_csv('data/trainingbin_.csv')
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
    clf = svm.SVC(kernel='linear', C = 8192.0 )
    scores = cross_val_score(clf, X, y, cv=24, scoring='accuracy')	
    print("Accuracy:", scores.mean())
    '''clf.fit(X_train,y_train)	
    y_pred = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics	
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))	'''
    
if __name__ == "__main__":
    #main()
    train()
  

