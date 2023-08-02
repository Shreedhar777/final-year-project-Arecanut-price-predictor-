 #Importing Libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import *
from plotly import tools


#Importing Data
df=pd.read_excel('data.xls',parse_dates=True, squeeze=True)
test=pd.read_excel('predictionempty.xls',parse_dates=True, squeeze=True)

print(df.head())

print(test.head())

#Converting to date time
df['Date'] = pd.to_datetime(df['Date']).dt.date

print(df.head())

#Dropping TimeStamp
df=df.groupby(['Date'])['Avg_Price'].sum().reset_index()

print(df)

#Extracting More info
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Week'] = pd.to_datetime(df['Date']).dt.week
df['Day'] = pd.to_datetime(df['Date']).dt.day
df['WeekDay'] = pd.to_datetime(df['Date']).dt.dayofweek


test['Date'] = pd.to_datetime(test['Date']).dt.date
test['Year'] = pd.to_datetime(test['Date']).dt.year
test['Week'] = pd.to_datetime(test['Date']).dt.week
test['Day'] = pd.to_datetime(test['Date']).dt.day
test['WeekDay'] = pd.to_datetime(test['Date']).dt.dayofweek


print(df.head())

df["Date"] = pd.to_datetime(df["Date"])
test["Date"] = pd.to_datetime(test["Date"])


# import seaborn as sns
# sns.set(rc={'figure.figsize':(16.7,10)})
# sns.boxplot(x=df['Avg_Price'])

#Import ML Algorithms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score



from sklearn.model_selection import train_test_split
predictors=df.drop(['Avg_Price','Date'],axis=1)
target=df['Avg_Price']
x_train,x_cv,y_train,y_cv=train_test_split(predictors,target,test_size=0.2,random_state=7)


#Comparing Algorithms
def scores(i):
    lin = i()
    lin.fit(x_train, y_train)
    y_pred=lin.predict(x_cv)
    lin_r= r2_score(y_cv, y_pred)
    s.append(lin_r)
    
#Checking the scores by using our function
algos=[LinearRegression,KNeighborsRegressor,RandomForestRegressor,Lasso,ElasticNet,DecisionTreeRegressor]
s=[]
for i in algos:
    scores(i)
    
    
#Checking the score
models = pd.DataFrame({
    'Method': ['LinearRegression', 'KNeighborsRegressor', 
              'RandomForestRegressor', 'Lasso','DecisionTreeRegressor'],
    'Score': [s[0],s[1],s[2],s[3],s[4]]})
models.sort_values(by='Score', ascending=False)

print(models)



#Hypertuned Model
model = RandomForestRegressor(oob_score = True,n_jobs =3,random_state =7,
                              max_features = "auto", min_samples_leaf =4)

model.fit(x_train,y_train)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=4, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=3,
                          oob_score=True, random_state=7, verbose=0,
                          warm_start=False)
pred=model.predict(x_cv)

print(r2_score(pred,y_cv))

print(df.head())

test1=test.drop(['sales', 'Date'],axis=1)

pred2=model.predict(test1)

test['Sales']=pred2.round(0)


print(test.head())

result=test[['Date','Sales']]
print(result.head())

result.to_csv('finalresult.csv')