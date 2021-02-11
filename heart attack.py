#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('D:/Breast Cancer.csv')
X=dataset.iloc[:,2:32].values
y=dataset.iloc[:,1].values


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

#spiliting the dataset into the trining set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)



#training the random forest Classifier model
from sklearn.ensemble import RandomForestClassifier
regressor=RandomForestClassifier(n_estimators=300,random_state=0)
regressor.fit(X_train,y_train)

#predict a new result
y_pred=regressor.predict(X_test)

print(f'{regressor.score(X_test,y_test):0.2%}')
