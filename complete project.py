#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[5]:


#Combine test and train into one file
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)


# In[6]:


data.head()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


#Distribution of the variable Item_Type


# In[10]:


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())
    
##The output gives us following observations:

##Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. 
##Also, some of ‘Regular’ are mentioned as ‘regular’.

##Item_Type: Not all categories have substantial numbers. 
##It looks like combining them can give better results.


# In[11]:


data['Item_Fat_Content'].value_counts().plot(kind="bar",color=['r','b','y','green','black'])


# In[12]:


data['Item_Type'].value_counts().plot(kind="bar", color=['r','green','black','b','yellow'])


# In[13]:


data['Outlet_Location_Type'].value_counts().plot(kind="bar")


# In[14]:


data['Outlet_Size'].value_counts().plot(kind="bar")


# In[15]:


data['Outlet_Type'].value_counts().plot(kind="bar")


# In[16]:


#Modify categories of Item_Fat_Content

#Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())


# In[17]:


#numeric value
data['Item_Weight'].plot.density()


# In[18]:


data['Item_MRP'].plot.density()


# In[19]:


data['Item_Visibility'].plot.density()


# In[20]:


data['Outlet_Establishment_Year'].plot.density()


# In[21]:


data['Item_Outlet_Sales'].plot.density()


# In[22]:


#Data Cleaning
#Check missing values:
data.apply(lambda x: sum(x.isnull()))


# In[23]:


data.Item_Outlet_Sales = data.Item_Outlet_Sales.fillna(data.Item_Outlet_Sales.mean())
data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
data['Outlet_Size'].value_counts()


# In[33]:


data.Outlet_Size = data.Outlet_Size.fillna('Medium')
data.apply(lambda x: sum(x.isnull()))


# In[32]:


data.groupby('Outlet_Identifier').Outlet_Size.value_counts(dropna=False)
## see that only OUT010, OUT017, OUT045 HAS NA values


# In[26]:


data.loc[data.Outlet_Identifier.isin(['OUT010','OUT017','OUT045']), 'Outlet_Size'] = 'Small'
data.Outlet_Size.value_counts()


# In[27]:


### Feature Engineering ###

data.min()


# In[28]:


## Notice that Item_Visibility has a minimum value of 0. It seems absurd that an item has 0 
## visibility. Therefore, we will modify that column.
## Here we Group by Item_Identifier, calculate mean for each group(excluding zero values), then we proceed
## to replace the zero values in each group with the group's mean.

## we have to replace 0's by na because, mean() doesnt support exclude '0' parameter 
##but it includes exclude nan parameter which is true by default

data.loc[data.Item_Visibility == 0, 'Item_Visibility'] = np.nan

#aggregate by Item_Identifier
IV_mean = data.groupby('Item_Identifier').Item_Visibility.mean()
IV_mean


# In[51]:


data.Item_Visibility.fillna(0, inplace=True)

#replace 0 values
for index, row in data.iterrows():
    if(row.Item_Visibility == 0):
        data.loc[index, 'Item_Visibility'] = IV_mean[row.Item_Identifier]
        #print(combined.loc[index, 'Item_Visibility'])
        
data.Item_Visibility.describe()
## see that min value is not zero anymore


# In[34]:


#Determine the years of operation of a store
#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[35]:


data['MRP_Factor'] = pd.cut(data.Item_MRP, [0,70,130,201,400], labels=['Low', 'Medium', 'High', 'Very High'])


# In[36]:


#Item type combine:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                            'NC':'Non-Consumable',
                                                            'DR':'Drinks'})
data['Item_Type_Combined'].value_counts() 


# In[37]:


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet', 'MRP_Factor']
le = LabelEncoder()
for i in var_mod:
   data[i] = le.fit_transform(data[i])


# In[38]:


#One Hot Coding: dummy varriables

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                             'Item_Type_Combined','Outlet', 'MRP_Factor'])


# In[39]:


data.dtypes
#Here we can see that all variables are now float and each category has a new variable.


# In[40]:


data[['Item_Fat_Content_0','Item_Fat_Content_1']].head(10)


# In[41]:


### Exporting Data ###
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year',],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
#train.to_csv("train_modified.csv",index=False)
#test.to_csv("test_modified.csv",index=False)


# In[42]:


train.head()


# In[43]:


test.head()


# In[44]:


## lets draw some plots to see that the regression assumptions are not voilated
## QQ plot

import pylab 
import scipy.stats as stats

quantile = data.Item_Outlet_Sales

stats.probplot(quantile, dist="uniform", plot=pylab)
pylab.show()

## the line is almost linear except for the end points 


# In[45]:


### Model Building ###

#Define target and ID columns:

##Since I’ll be making many models, instead of repeating the codes again and again, 
##I would like to define a generic function which takes the algorithm and data as input and makes the model
##performs cross-validation and generates submission

# we want to predict target
target = 'Item_Outlet_Sales'

#below are just identifiers which we dont want to fit
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import metrics
from sklearn.model_selection import cross_validate, cross_val_score
import matplotlib.pyplot as plt

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename, resid=False, transform=False):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    residuals = dtrain_predictions - dtrain[target]
    if(transform == True):
        train_mod = train.copy(deep = True)
        train_mod[target] = train_mod[target].apply(np.log)
        dtrain_predictions = np.exp(dtrain_predictions)
        #print(dtrain_predictions)

    
    #residuals vs fitted plot
    if(resid == True):
        plt.scatter(dtrain_predictions, residuals)
        plt.xlabel('fitted values')
        plt.ylabel('residuals')
        plt.show()
    
    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


# In[46]:


### Linear Regression Model

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
pred1 = np.nan
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv', resid=True)


coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients', figsize=(10,6))

#if you notice the coefficients, they are very large in magnitude which signifies overfitting. 
#To cater to this, we will use a ridge regression model.

## residual vs fitted plot and model coefficients plot is given below


# In[47]:


from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='hist', title='Feature Importances',color="red")


# In[52]:


from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='hist', title='Feature Importances')


# In[55]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  


# In[ ]:





# In[ ]:




