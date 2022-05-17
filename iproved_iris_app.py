from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
from random import Random, random
import numpy as np
import pandas as pd
from soupsieve import select
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('iris-species.csv')
print(df)

df['Species']=df['Species'].map({'Iris-setosa':0,'Iris-virginica':1,'Iris-versicolor':2})
features_df=df.iloc[:,1:-1]
target_df=df['Species']

X_train,X_test,y_train,y_test=train_test_split(features_df,target_df,random_state=42,test_size=0.3)

svc_model=SVC(kernel='linear')
svc_model.fit(X_train,y_train)

rf_clf=LogisticRegression(n_jobs=-1)
rf_clf.fit(X_train,y_train)

rf_forest=RandomForestClassifier(n_estimators=100)
rf_forest.fit(X_train,y_train)

@st.cache()
def prediction(model,sepal_length,sepal_width,petal_lenght,petal_width):
    pred=model.predict([[sepal_length,sepal_width,petal_lenght,petal_width]])
    pred=pred[0]
    if pred==0:
        return 'Iris-setosa'
    elif pred==1:
        return 'Iris-virginica'
    elif pred==2:
        return 'Iris-versicolor'
    
st.sidebar.title('Iris species prediction app')
s_length=st.sidebar.slider('Sepal length',float(df['SepalLengthCm'].min()),float(df['SepalLengthCm'].max()))
s_width=st.sidebar.slider('Sepal Width',float(df['SepalWidthCm'].min()),float(df['SepalWidthCm'].max()))
p_length=st.sidebar.slider('petal length',float(df['PetalLengthCm'].min()),float(df['PetalLengthCm'].max()))
p_width=st.sidebar.slider('petal Width',float(df['PetalWidthCm'].min()),float(df['PetalWidthCm'].max()))

classifier=st.sidebar.selectbox('Classifier',('Support vector machine','Logitic Regression','Random forest classifier'))

if st.sidebar.button('Predict'):
    if classifier=='Support vector machine':
        species=prediction(svc_model,s_length,s_width,p_length,p_width)
        score=svc_model.score(X_train,y_train)

    elif classifier=='Logistic Regression':
        species=prediction(rf_clf,s_length,s_width,p_length,p_width)
        score=rf_clf.score(X_train,y_train)

    else:
        species=prediction(rf_forest,s_length,s_width,p_length,p_width)
        score=rf_forest.score(X_train,y_train)
    st.write(f"species predicted {species}")
    st.write(f"score of the model is {score}")
    st.balloons()

        