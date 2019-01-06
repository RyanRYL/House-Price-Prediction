
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import Imputer, LabelEncoder
np.set_printoptions(threshold='nan')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


pd.__version__


# In[3]:


train_df = pd.read_csv("train.csv")
print(train_df.shape)
train_df.head(5)


# In[4]:


test_df = pd.read_csv("test.csv")
print(test_df.shape)


# In[5]:


train_df['SalePrice'].describe()


# In[6]:


Missing = pd.concat([train_df.isnull().sum(), test_df.isnull().sum()], axis=1, keys=['train', 'test'])
Missing[Missing.sum(axis=1) > 0]


# In[7]:


train_df.describe()


# In[8]:


na_count = train_df.isnull().sum().sort_values(ascending=False)   
print(na_count)


# In[9]:


# test_df = test_df.drop("Id", axis=1)


# In[10]:


train_df = train_df.drop("Id", axis=1)


# In[11]:


na_rate = na_count/train_df.shape[0]  


# In[12]:


na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])


# In[13]:


print(na_data[na_data['ratio'] > 0])


# In[14]:


del_miss = na_data[na_data['ratio'] > 0.40].index
train = train_df.drop(del_miss, axis=1)
test = test_df.drop(del_miss, axis=1)  


# In[15]:


print(train.isnull().sum().sort_values(ascending=False)[:13])
print("\n")
print(test.isnull().sum().sort_values(ascending=False)[:13])


# In[16]:


factor = [col for col in train.columns if train[col].dtypes == 'object']
print(factor)
print (len(factor))


# In[17]:


numeric = [col for col in train.columns if train[col].dtypes != 'object']
print(numeric)
print (len(numeric))


# In[18]:


fillna = Imputer(strategy='most_frequent')


# In[19]:


train[numeric] = fillna.fit_transform(train[numeric])
test[numeric[:-1]] = fillna.fit_transform(test[numeric[:-1]])


# In[20]:


train[factor] = train[factor].fillna('None')
test[factor] = test[factor].fillna('None')


# In[21]:


print(train.shape)
print(test.shape)


# In[22]:


number = LabelEncoder()
for f in factor:
    train[f] = number.fit_transform(train[f].astype("str"))
    test[f] = number.fit_transform(test[f].astype("str"))
    
factor.append("SalePrice")


# In[23]:


corrmat = train[factor].corr('pearson')
f,ax = plt.subplots(figsize=(12,9))
ax.set_xticklabels(corrmat,rotation='horizontal')
sns.heatmap(corrmat, square=False, center=1)  
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.xlabel('Heat map for objective attrbiutes')
plt.show()


# In[24]:


c = train[factor].corr(method='pearson')['SalePrice'].sort_values(ascending=False)
print(c)


# In[25]:


corrmat = train[numeric].corr('pearson')
f,ax = plt.subplots(figsize=(24,18))
ax.set_xticklabels(corrmat,rotation='horizontal')
sns.heatmap(np.fabs(corrmat), square=False, center=1)  
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.xlabel('Heat Map for numeric attributes')
plt.show()


# In[26]:


c = train[numeric].corr(method='pearson')['SalePrice'].sort_values(ascending=False)
print(c)


# In[27]:


f1 = []
f1.append('KitchenQual')
f1.append('BsmtQual')
f1.append('ExterQual')
c = train[f1].corr(method='pearson')
f,ax = plt.subplots(figsize=(24,18))
ax.set_xticklabels(c,rotation='horizontal')
sns.heatmap(np.fabs(c), square=False, center=1)  
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.show()


# In[28]:


f2 = []
f2.append('OverallQual')
f2.append('GrLivArea')
f2.append('GarageCars')
f2.append('GarageArea')
f2.append('TotalBsmtSF')
f2.append('1stFlrSF')
f2.append('FullBath')
f2.append('TotRmsAbvGrd')
f2.append('YearBuilt')
f2.append('YearRemodAdd')
c = train[f2].corr(method='pearson')
f,ax = plt.subplots(figsize=(24,18))
ax.set_xticklabels(c,rotation='horizontal')
sns.heatmap(np.fabs(c), square=False, center=1)  
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.show()


# In[29]:


f2.remove('GrLivArea')
f2.remove('1stFlrSF')
f2.remove("GarageCars")
f2.remove("TotRmsAbvGrd")
for f in f2:
    f1.append(f)
print(f1)


# In[30]:


c = train[f1].corr(method='pearson')
f,ax = plt.subplots(figsize=(24,18))
ax.set_xticklabels(c,rotation='horizontal')
sns.heatmap(np.fabs(c), square=False, center=1)  
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.show()


# In[31]:


selected_features = f1
selected_features.append('SalePrice')
print(selected_features)


# In[32]:


r_train = train.copy(deep=True)
for c1 in range(0, train.shape[1]):
    if train.columns[c1] not in selected_features:
        r_train.drop(train.columns[c1], axis = 1, inplace = True)
print(r_train.shape)

r_test = test.copy(deep=True)
for c2 in range(0, test.shape[1]):
    if test.columns[c2] not in selected_features:
        r_test.drop(test.columns[c2], axis = 1, inplace = True)
print(r_test.shape)


# In[33]:


y = r_train.SalePrice
x = r_train.drop('SalePrice', axis=1)
one_hot = pd.get_dummies(x)
one_hot = (one_hot - one_hot.mean()) / (one_hot.max() - one_hot.min())


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(one_hot, y, train_size=0.7)



# In[49]:


from sklearn.metrics import r2_score
m = RandomForestRegressor(n_estimators = 75, min_samples_leaf=15)
m.fit(x_train, y_train)
score = r2_score(y_test,m.predict(x_test))
print(score)


# In[36]:


from sklearn.metrics import r2_score
# print(m.feature_importances_.astype(str))
score = r2_score(y_test,m.predict(x_test))
print(score)


# In[37]:


print("Random forest Features Importance")
headers = ["name", "score"]
values = sorted(zip(x_train.columns, m.feature_importances_), key=lambda x: x[1]*-1)
print(tabulate(values, headers, tablefmt="plain"))


# In[105]:


from sklearn.ensemble import GradientBoostingRegressor

m3 = GradientBoostingRegressor(loss = "ls", n_estimators = 150, learning_rate=0.01, max_depth=5, verbose=1)
m3.fit(x_train, y_train)


# In[106]:


score = r2_score(y_test,m3.predict(x_test))
print(score)


# In[79]:


print("Gardient Boosting Features Importance")
headers = ["name", "score"]
values = sorted(zip(x_train.columns, m3.feature_importances_), key=lambda x: x[1]*-1)
print(tabulate(values, headers, tablefmt="plain"))


# In[54]:


from sklearn.neural_network  import MLPRegressor

m2 = MLPRegressor(hidden_layer_sizes=(128, 11), learning_rate_init=0.01, verbose=True, max_iter=1000)
m2.fit(x_train, y_train)


# In[42]:


score = r2_score(y_test,m2.predict(x_test))
print(score)


# In[43]:


sub = pd.DataFrame()
sub['Id'] = test_df['Id']
predict = m.predict(r_test)
sub['SalePrice'] = predict
sub.to_csv('01.csv', index=False)


# In[92]:


predict = m3.predict(r_test)
sub['SalePrice'] = predict
sub.to_csv('gardient_boosting.csv', index=False)


# In[55]:


predict = m2.predict(r_test)
sub['SalePrice'] = predict
sub.to_csv('multi_layer_perceptron.csv', index=False)

