import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("c:/Users/Pranav/Documents/IRIS.csv")
print(data.head())
print(data.info())
print(data.shape)
print(data.ndim)
print(data.size)
print(data.describe())
print(data.isnull().sum())
print(data['species'].value_counts())
sns.histplot(data,x='sepal_length',hue='species',palette='magma' ).set(title='Sepal Length Histogram')
plt.show()
sns.histplot(data,x='sepal_width',hue='species',palette='magma' ).set(title='Sepal Width Histogram')
plt.show()
sns.histplot(data,x='petal_length',hue='species',palette='magma' ).set(title='Petal Length Histogram')
plt.show()
sns.histplot(data,x='petal_width',hue='species', palette='magma').set(title='Petal Width Histogram')
plt.show()
fig = px.box(data_frame=data, y='sepal_length',title='Outlier Detection for Sepal Length')
fig.show()
fig = px.box(data_frame=data, y='sepal_width',title='Outlier Detection for Sepal Width')
fig.show()
fig = px.box(data_frame=data, y='petal_length',title='Outlier Detection for Petal Length')
fig.show()
fig = px.box(data_frame=data, y='petal_width',title='Outlier Detection for Petal Width')
fig.show()
Q1=data['sepal_width'].quantile(0.25)
Q3=data['sepal_width'].quantile(0.75)
IQR=Q3-Q1
print("Q1: ",Q1)
print("Q3: ",Q3)
print("IQR: ",IQR)
Lower_Bound=Q1-1.5*IQR
Upper_Bound=Q3+1.5*IQR
print("Lower_Bound: ",Lower_Bound)
print("Upper_Bound: ",Upper_Bound)
l1=[]
for i in data['sepal_width']:
    if i<Lower_Bound:
        l1.append(i)
l2=[]
for i in data['sepal_width']:
    if i>Upper_Bound:
        l2.append(i)
outliers_list=l1+l2
outliers_list.sort()
data.hist()
plt.show()
x=data.drop("species",axis=1)
y=data["species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
#logistical
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)
print(classification_report(y_test,y_pred_lr))
print(accuracy_score(y_test,y_pred_lr))
print(confusion_matrix(y_test,y_pred_lr))
#k neighbour
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
print(classification_report(y_test,y_pred_knn))
print(accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))
#random forest
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
RandomForestClassifier
RandomForestClassifier()
y_pred_rf=rf.predict(x_test)
print(classification_report(y_test,y_pred_rf))
print(accuracy_score(y_test,y_pred_rf))
print(confusion_matrix(y_test,y_pred_rf))
#decision tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)
print(classification_report(y_test,y_pred_dt))
print(accuracy_score(y_test,y_pred_dt))
print(confusion_matrix(y_test,y_pred_dt))
