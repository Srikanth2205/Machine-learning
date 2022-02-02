#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt
import seaborn as sns
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import skew
from sklearn import preprocessing
from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


# In[2]:


pd.pandas.set_option('display.max_columns',None)
Input_data=pd.read_csv('train.csv')

Input_data_copy=Input_data
## print shape of dataset with rows and columns
print(Input_data.shape)


# In[3]:



print(Input_data.head())


# In[ ]:





# In[4]:


print(Input_data.info())


# In[ ]:





# In[5]:


# Find columns with missing values and their percent missing
Input_data.isnull().sum()                                                     
miss_val = Input_data.isnull().sum().sort_values(ascending=False)
miss_val = pd.DataFrame(data=Input_data.isnull().sum().sort_values(ascending=False), columns=['MissvalCount'])

# Add a new column to the dataframe and fill it with the percentage of missing values
miss_val['Percent'] = miss_val.MissvalCount.apply(lambda x : '{:.2f}'.format(float(x)/Input_data.shape[0] * 100)) 
miss_val = miss_val[miss_val.MissvalCount > 0]
print(miss_val)


# In[6]:


# drop columns with high missing values
Input_data = Input_data.drop(['Fence', 'MiscFeature', 'PoolQC','FireplaceQu','Alley'], axis=1)


# In[7]:


Input_data.dropna(inplace=True)


# In[8]:


print(Input_data.shape)


# In[9]:


sns.distplot(Input_data.SalePrice)
plt.show(block=True)

# In[10]:


# Transform the target variable 
sns.distplot(np.log(Input_data.SalePrice))

plt.show(block=True)
# In[11]:


Input_data['LogOfPrice'] = np.log(Input_data.SalePrice)
Input_data.drop(["SalePrice"], axis=1, inplace=True)


# In[12]:


# Review the skewness of each feature
Input_data.skew().sort_values(ascending=False)


# In[13]:


# set the target and predictors
y = Input_data.LogOfPrice  # target

# use only those input features with numeric data type 
data_temp = Input_data.select_dtypes(include=["int64","float64"]) 
X = data_temp.drop(["LogOfPrice"],axis=1)  # predictors


# In[ ]:





# In[ ]:





# In[14]:


# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 3)


# In[15]:


lr = LinearRegression()
# fit optimal linear regression line on training data, this performs gradient descent under the hood
lr.fit(X_train, y_train)


# In[16]:


# given our model and our fit, predict y_values using X_test set
yr_hat = lr.predict(X_test)
print(yr_hat)


# In[17]:


# evaluate the algorithm with a test set 
lr_score = lr.score(X_test, y_test)  # train test 
print("Accuracy: ", lr_score)


# In[18]:


# cross validation to find 'validate' score across multiple samples, automatically does Kfold stratifying
lr_cv = cross_val_score(lr, X_test, y_test, cv = 5, scoring= 'r2')
print("Cross-validation results: ", lr_cv)
print("R2: ", lr_cv.mean())


#


# In[21]:


ridge = Ridge(alpha = 1)  # sets alpha to a default value as baseline  
ridge.fit(X_train, y_train)

ridge_cv = cross_val_score(ridge, X_test, y_test, cv = 5, scoring = 'r2')
print ("Cross-validation results: ", ridge_cv)
print ("R2: ", ridge_cv.mean())


# In[22]:


from sklearn.model_selection import GridSearchCV
param_grid = [
{'alpha': [2,3,5,6,7,8] },
]
Ridge_gred_reg = Ridge()
grid_search = GridSearchCV(Ridge_gred_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(X_train, y_train)
alpha_value=grid_search.best_params_['alpha']
print(alpha_value)


# In[23]:


ridge_gred = Ridge(alpha=alpha_value)  # sets alpha to a default value as baseline  
ridge_gred.fit(X_train, y_train)

ridge_gred_cv = cross_val_score(ridge_gred, X_test, y_test, cv = 5, scoring = 'r2')
print ("Cross-validation results: ", ridge_gred_cv)
print ("R2: ", ridge_gred_cv.mean())



regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)

ElasticNet_cv = cross_val_score(regr, X_test, y_test, cv = 5, scoring = 'r2')
print ("Cross-validation results: ", ElasticNet_cv)
print ("R2: ", ElasticNet_cv.mean())


# In[27]:


from sklearn.model_selection import GridSearchCV
param_grid = [
{'alpha': [0.001, 0.0025, 0.005,0.0075], 'l1_ratio': [0.01,0.04,0.06,0.08,0.1]},
]
ElasticNet_gred_reg = ElasticNet()
grid_search = GridSearchCV(ElasticNet_gred_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(X_train, y_train)
alpha_value=grid_search.best_params_['alpha']
l1_ratio_val=grid_search.best_params_['l1_ratio']
print(alpha_value,l1_ratio_val)


# In[28]:


ElasticNet_gred = ElasticNet(alpha_value,l1_ratio_val,random_state=0)
ElasticNet_gred.fit(X_train, y_train)

ElasticNet_cv_gred = cross_val_score(ElasticNet_gred, X_test, y_test, cv = 5, scoring = 'r2')
print ("Cross-validation results: ", ElasticNet_cv_gred)
print ("R2: ", ElasticNet_cv_gred.mean())





# In[31]:


print(Input_data_copy.head())


# In[32]:


Input_data_copy = Input_data_copy.apply(pd.to_numeric, errors='coerce')
Input_data_copy.fillna(0, inplace=True)
print(Input_data_copy.head())
Input_data_copy.shape
Dependent_var=Input_data_copy.SalePrice
Input_data=Input_data_copy.drop(["SalePrice"], axis=1, inplace=True)


# In[33]:


from sklearn.preprocessing import StandardScaler


# In[34]:


scaler=StandardScaler()
scaler.fit(Input_data_copy)


# In[35]:


scaled_data=scaler.transform(Input_data_copy)


# In[36]:


print(scaled_data.shape)


# In[37]:


from sklearn.decomposition import PCA
pca=PCA(n_components=65)
pca.fit(scaled_data)


# In[38]:


x_pca=pca.transform(scaled_data)


# In[39]:


print(x_pca.shape)


# In[40]:


Input_data_copy.keys()


# In[41]:

Independent_var=x_pca


# In[42]:


In_train,In_test,Out_train,Out_test=train_test_split(Independent_var,Dependent_var,test_size=0.30)

print("shape of original dataset :", Input_data_copy.shape)
print("shape of input - training set", In_train.shape)
print("shape of output - training set", Out_train.shape)
print("shape of input - testing set", In_test.shape)
print("shape of output - testing set", Out_test.shape)


# In[43]:



#neigh = KNeighborsClassifier(n_neighbors=3)
Linear_PCA=LinearRegression()
Linear_PCA.fit(In_train, Out_train)


# In[44]:


Linear_cv = cross_val_score(ridge, In_test, Out_test, cv = 5, scoring = 'r2')
print ("Cross-validation results: ", Linear_cv)
print ("R2: ", Linear_cv.mean())


# In[45]:


from matplotlib.pyplot import figure

fig = plt.figure(figsize=(18, 9), dpi=80)
ax = fig.add_subplot(111)
# x axis values
x = ["Linear regression","Ridge Regression","Ridge Regression-GredSearch","Elasticnet Regression","Elasticnet Regression-GredSearch","PCA using Linear regression"]
# corresponding y axis values
y = [lr_cv.mean()*100,ridge_cv.mean()*100,ridge_gred_cv.mean()*100,ElasticNet_cv.mean()*100,ElasticNet_cv_gred.mean()*100,Linear_cv.mean()*100]
  

#plt.barh(x, y)

#for index, value in enumerate(y):
 #   plt.text(value, index, str(value))
plt.plot(range(len(x)), y, 'go-') # Plotting data
plt.xticks(range(len(x)), x) # Redefining x-axis labels

for i, v in enumerate(y):
    ax.text(i, v+3, "%f" %v, ha="center")
plt.ylim(50, 107)
# naming the x axis
plt.xlabel('Regression model ')
# naming the y axis
plt.ylabel(' Accuracy %')
  
# giving a title to my graph
plt.title('Accuracy comparison of various Regression alogorithms')

plt.show(block=True)

from matplotlib import pyplot
# get importance
importance = lr.coef_

feature_importances = zip(importance, X.columns)
sorted_feature_importances = sorted(feature_importances, reverse = True)
#print(sorted_feature_importances)

top_15_predictors = sorted_feature_importances[0:15]
values = [value for value, predictors in top_15_predictors]
predictors = [predictors for value, predictors in top_15_predictors]
print(predictors)

# Plot the feature importances of the linear regressor
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(predictors)), values,color="r", align="center")
plt.xticks(range(len(predictors)), predictors, rotation=90)
plt.show(block=True)





# In[ ]:





# In[ ]:





# In[ ]:




