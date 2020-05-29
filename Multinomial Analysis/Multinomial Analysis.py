"""
Created on Thu Apr 16 14:46:08 2020
@author: DESHMUKH
MULTINOMIAL REGRESSION
"""
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================================================================================================
# Business Problem :- Prepare a prediction model for predict the type of program a student is in, based on other attributes.
# ==========================================================================================================================

data = pd.read_csv("mdata.csv",index_col = 0)
pd.set_option('display.max_column',10)
data.isnull().sum()
data.shape
data.head(10)
data = data.drop(["id"],axis = 1)
data.columns

############################### - Exploratary Data Analysis - ################################

# Summary
data.describe()

# Boxplot & Scatter plot of independent variable distribution for each category of choice
sns.boxplot(x="prog",y="read",data=data)
sns.stripplot(x="prog",y="read",jitter=True,data=data)

sns.boxplot(x="prog",y="write",data=data)
sns.stripplot(x="prog",y="write",jitter=True,data=data)

sns.boxplot(x="prog",y="math",data=data)
sns.stripplot(x="prog",y="math",jitter=True,data=data)

sns.boxplot(x="prog",y="science",data=data)
sns.stripplot(x="prog",y="science",jitter=True,data=data)

# Boxplot                       
sns.boxplot(data.read)
sns.boxplot(data.write)
sns.boxplot(data.math)
sns.boxplot(data.science)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(data,hue="prog") # With showing the category of each car choice in the scatter plot

# Correlation
data.corr()

# Heatmap
sns.heatmap(data.corr(),annot = True,cmap="CMRmap")

# Kdeplot
sns.kdeplot(data.corr(), cmap="Reds", shade='prog',n_levels=30)

# Determining Properties of Y
# Counts of Y obsevation
data.prog.value_counts()
sns.countplot(data.prog, data = data, palette="winter")

############################## - Converting Dummy variable - ###############################

# Checking data
data.head()
data.female.value_counts()
data.ses.value_counts()
data.schtyp.value_counts()
data.honors.value_counts()

# Dummy variable
dummy = pd.get_dummies(data[['female','ses','schtyp','honors']],drop_first = True)
dummy.head()

# Rename column
dummy = dummy.rename({'female_male' : 'male'}, axis=1)

data.head(2)
# Droping nomial columns from dataframe
data = data.drop(['female','ses','schtyp','honors'],axis = 1)

# Concating dummy variable with dataframe
data = pd.concat([data,dummy],axis = 1)

################################### - Splitting Data - ####################################

# Splitting data into X and Y
X = data.iloc[:,1:]
y = data.iloc[:,0:1]

# Splitting data into Train and Test
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3 , random_state = 111)

######################## - Multinomial regression model Building - ########################

# Model Fitting
# ‘Multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(X_train,y_train) #fit(X,Y) or fit(input, output) index 1 =second column, everything starting from here , index o is first column.      

# Train Prediction
train_predict = model.predict(X_train)

# Test Prediction
test_predict = model.predict(X_test)

# Train accuracy 
accuracy_score(y_train,train_predict)

# Test accuracy 
accuracy_score(y_test,test_predict)
                        
                                 # ---------------------------- #


