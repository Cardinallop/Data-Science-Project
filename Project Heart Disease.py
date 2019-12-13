#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/Cardinallop/ML-project-2/master/"
HEART_PATH = os.path.join("datasets", "HEART")
HEART_URL = DOWNLOAD_ROOT + "heart.csv"


#this code below automatically downloads the publicly available data from my github


def fetch_data(heart_url=HEART_URL, heart_path=HEART_PATH):
    if not os.path.isdir(heart_path):
        os.makedirs(heart_path)
    tgz_path = os.path.join(heart_path, "heart.csv")
    urllib.request.urlretrieve(heart_url, tgz_path)


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.tree import export_graphviz
import sklearn.metrics as met
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()


# In this project I will try to find a proper model to predict the probability of a heart disease given input factors and I will try to reach the model's maximum predicting capacity by fine-tuning it with a portion of a data. The dataset is available at: https://archive.ics.uci.edu/ml/datasets/Heart+Disease¶

# In[2]:


fetch_data()


# In[3]:


import pandas as pd

def load_data(heart_path=HEART_PATH):
    csv_path = os.path.join(heart_path, "heart.csv")
    return pd.read_csv(csv_path)


# In[4]:


data = load_data()
data.head()


# As you can see, some factors are categorical and some are linear.
# 
# Before I go any further I will define the input values and describe what they are:
# 
# 1- age is just an age of a patient
# 
# 2- sex is a gender, and I have to turn it into Ordinally encoded values, such as 1 for maale and 0 for female
# 
# 3- cp is a chest-pain. Its encoded as 4 types of it: 1-typical angina, 2- nontypical angina, 3- nonanginal pain, 4- asymptotic. Here I have to encode them into OneHot encoded values
# 
# 4- trestbps is a resting blood pressure
# 
# 5- chol is a level of cholesterol
# 
# 6- fbs is a fasting blood sugar, that is 1 if its higher than 120 or 0 otherwise. Again we have to change its code.
# 
# 7- restecg is a resting electrocardiographic results encoded as 0-normal, 1-having abnormality, 2-having a hypertrophy
# 
# 8- thalach is a maximum heart rate achieved
# 
# 9- exang is a exercise induced angina (1 = yes; 0 = no). This also needs to be encoded as ordinal encoder
# 
# 10- oldpeak is a ST depression induced by exercise relative to rest
# 
# 11- slope is a the slope of the peak exercise ST segment
# 
# 12- ca is a number of major vessels (0-3) colored by flourosopy
# 
# 13- thal is a heart defect 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[5]:


data.info()


# ### Notice that this is a clean, perfect data without any missing values

# In[6]:


data.describe()


# you can see from the above graph that a cholesterol has a big effect on heart disease with a mean of 246, much bigger number compared to other values.

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data['age'].hist(bins=50, figsize=(5,5))
plt.xlabel('age') 
plt.ylabel('frequency')    
plt.show()


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data['chol'].hist(bins=50, figsize=(5,5))
plt.xlabel('cholesterol level') 
plt.ylabel('rate') 
plt.show()


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data['thalach'].hist(bins=50, figsize=(5,5))
plt.xlabel('maximum heart rate') 
plt.ylabel('frequency') 
plt.show()


# In[124]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data['trestbps'].hist(bins=50, figsize=(5,5))
plt.xlabel('resting blood pressure') 
plt.ylabel('frequency') 
plt.show()


# ### To make analysis on outliers (reminder)

# In[125]:


sns.relplot(x='age', y='chol', hue='sex', data=data)
plt.show()


# In[26]:


sns.relplot(x='age', y='thalach', hue='sex', data=data)
plt.show()


# In[27]:


sns.relplot(x='age', y='trestbps', hue='sex', data=data)
plt.show()


# In[28]:


lm = smf.ols('target ~ age + chol + sex + cp + trestbps + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal', data).fit()
lm.summary()


# In[29]:


lm.resid.hist(bins=20)
plt.show()


# In[30]:


data_new = pd.get_dummies(data, columns=['sex','cp', 'fbs','restecg','exang','slope','ca','thal'], drop_first=True)
data_new.head()


# In[31]:


data_new.columns


# In[38]:


lm2 = smf.ols('target ~ age+ trestbps + chol + thalach + oldpeak + target + sex_1 +            cp_1 + cp_2 + cp_3 + fbs_1 + restecg_1 + restecg_2 + exang_1 +                slope_1 + slope_2 + ca_1 + ca_2 + ca_3 + ca_4 + thal_1 +                    thal_2 + thal_3', data=data_new).fit()

lm2.summary()


# In[39]:


lm2.resid.hist(bins=20)
plt.show()


# In[95]:


X = data_new.drop('target', 1)
X.head()


# ## Decision Tree regressor:

# In[96]:


reg = tree.DecisionTreeRegressor(max_depth = 5)
reg = reg.fit(X, data_new['target'])


# In[97]:


tree.plot_tree(reg)
plt.show()


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X, data_new['target'], test_size = 0.2)


# In[99]:


reg2 = tree.DecisionTreeRegressor(max_depth = 3)
reg3 = reg2.fit(X_train, y_train)
reg3


# In[100]:


predictions = reg3.predict(X_test)
predictions


# In[101]:


((predictions - y_test)**2).mean()


# In[102]:


(data['chol']==0).sum()


# In[104]:


sns.regplot(y='target', x='chol', data = data, fit_reg = True)
plt.show()


# In[106]:


logit_model = smf.logit('target ~ chol', data = data).fit()
logit_model.summary()


# In[107]:


logit_model.params


# In[108]:


X = np.linspace(0, 600, 320)


# In[115]:


p = logit_model.params
reg44 = p['Intercept'] + X*p['chol']


# In[116]:


y = np.exp(reg44)/(1 + np.exp(reg44))


# In[122]:


sns.relplot(x='chol', y='target', data=data)
plt.plot(X,y)
plt.show()


# In[123]:


logit_model.pred_table()


# ### In order to work further, I have to split our data and compose and automatic transformer, so whenever I get a new data, I filter the new data with this transformer instead of sorting it all over again¶

# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)


# In[44]:


len(train_set)


# In[45]:


len(test_set)


# In[46]:


corr_matrix = train_set.corr()


# In[47]:


corr_matrix["target"].sort_values(ascending=False)


# #### Correlation Matrix shows how the target value is related to other factors, meaning which factors affects to heart disease by what amoun or what percentage

# In[48]:


from pandas.plotting import scatter_matrix

attributes = ["cp", "thalach", "slope", "sex", "exang", "oldpeak", "ca", "age"]
scatter_matrix(train_set[attributes], figsize=(12, 8))
plt.show()


# #### In order to work further, I have to split our data and compose and automatic transformer, so whenever I get a new data, I filter the new data with this transformer instead of sorting it all over again

# In[49]:


dataTrain = train_set.drop("target", axis=1)
dataLabels = train_set["target"].copy()


# In[50]:


dataTrain.info()


# In[51]:


dataTrain_num = dataTrain.drop("sex", axis=1)
dataTrain_num = dataTrain_num.drop("cp", axis=1)
dataTrain_num = dataTrain_num.drop("fbs", axis=1)
dataTrain_num = dataTrain_num.drop("restecg", axis=1)
dataTrain_num = dataTrain_num.drop("exang", axis=1)
dataTrain_num = dataTrain_num.drop("slope", axis=1)
dataTrain_num = dataTrain_num.drop("thal", axis=1)


# In[52]:


dataTrain_num.head()


# In[53]:


dataTrain_num.shape


# In[54]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])


# In[55]:


dataTrain.shape


# In[56]:


from sklearn.compose import ColumnTransformer

num_attribs = list(dataTrain_num)
cat_attribs = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

train_prepared = full_pipeline.fit_transform(dataTrain)


# In[57]:


train_prepared.shape


# ### Now let's compare and contrast 3 different models which seem applicable to the given data in order to predict our information. At the end I will pick the best model to work with the given data
# 
# ### 1-Model: DecisionTree Classifier¶
# 

# In[58]:


from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(train_prepared, dataLabels)


# ### Testing and Validation:

# In[59]:


some_dat = dataTrain.iloc[:5]
some_labs = dataLabels.iloc[:5]
some_dat_prepared = full_pipeline.transform(some_dat)
print("Predictions:", tree_clf.predict(some_dat_prepared))
print("Labels:", list(some_labs))


# ### Cross Validation Score:

# In[60]:


from sklearn.model_selection import cross_val_score
cvsscores = cross_val_score(tree_clf, train_prepared, dataLabels,
                         scoring="accuracy", cv=10)


# In[61]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(cvsscores)


# ### Confusion Matrix:¶

# In[62]:


from sklearn.model_selection import cross_val_predict

train_predo = cross_val_predict(tree_clf, train_prepared, dataLabels, cv=10)


# In[63]:


from sklearn.metrics import confusion_matrix
confusion_matrix(dataLabels, train_predo)


# ### Precision and Recall:

# In[64]:


from sklearn.metrics import precision_score, recall_score
precision_score(dataLabels, train_predo)


# In[65]:


recall_score(dataLabels, train_predo)


# ### F-score:

# In[66]:


from sklearn.metrics import f1_score
f1_score(dataLabels, train_predo)


# ### ROC Curve:

# In[67]:


y_scores = cross_val_predict(tree_clf, train_prepared, dataLabels, cv=3)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(dataLabels, y_scores)


# In[68]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])                                   
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                            

plt.figure(figsize=(8, 6))                         
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") 
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  
plt.plot([4.837e-3], [0.4368], "ro")               
                      
plt.show()


# In[69]:


from sklearn.metrics import roc_auc_score

roc_auc_score(dataLabels, y_scores)


# ## 2-Model: RandomForestClassifier:

# In[70]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(train_prepared, dataLabels)


# ### Test and Validation: 

# In[71]:


some_datq = dataTrain.iloc[:5]
some_labsq = dataLabels.iloc[:5]
some_dat_preparedq = full_pipeline.transform(some_datq)
print("Predictions:", forest_clf.predict(some_dat_preparedq))

print("Labels:", list(some_labsq))


# ### Cross Validation Score: 

# In[72]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, train_prepared, dataLabels,
                                scoring="accuracy", cv=10)

display_scores(forest_scores)


# ### Confusion Matrix: 

# In[73]:


train_predo1 = cross_val_predict(forest_clf, train_prepared, dataLabels, cv=10)


# In[74]:


from sklearn.metrics import confusion_matrix
confusion_matrix(dataLabels, train_predo1)


# ### Precision and Recall: 

# In[75]:


from sklearn.metrics import precision_score, recall_score
precision_score(dataLabels, train_predo1)


# ### ROC Curve:

# In[76]:


y_scores1 = cross_val_predict(forest_clf, train_prepared, dataLabels, cv=3)

fpr1, tpr1, thresholds1 = roc_curve(dataLabels, y_scores1)


# In[77]:


plt.figure(figsize=(8, 6))                         
plot_roc_curve(fpr1, tpr1)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") 
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  
plt.plot([4.837e-3], [0.4368], "ro")               
                      
plt.show()


# In[78]:


roc_auc_score(dataLabels, y_scores1)


# ## 3-Model: KNeighbors Classifier:

# In[79]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(train_prepared, dataLabels)


# ### Test and Validation:

# In[80]:


some_dat6 = dataTrain.iloc[:10]
some_labs6 = dataLabels.iloc[:10]
some_dat_prepared6 = full_pipeline.transform(some_dat6)
print("Predictions:", neigh.predict(some_dat_prepared6))

print("Labels:", list(some_labs6))


# ### Cross Validation Score:

# In[81]:


KN_scores = cross_val_score(neigh, train_prepared, dataLabels,
                                scoring="accuracy", cv=10)

display_scores(KN_scores)


# ### Confusion Matrix: 

# In[82]:


train_predo2 = cross_val_predict(neigh, train_prepared, dataLabels, cv=10)


# In[83]:


from sklearn.metrics import confusion_matrix
confusion_matrix(dataLabels, train_predo2)


# ### Precision Score: 

# In[84]:


from sklearn.metrics import precision_score, recall_score
precision_score(dataLabels, train_predo2)


# ## Here I stop, because if you compare the above 3 models, among them, the best model is Random Forest Classifier with over 80% precision and I don't think I can get any better result from other models. However I may get much improved results if I apply Neural Networks, unfortunately, I yet have to practice how to work with Neural Networks. I hope that I will have the time and opportunity to use Neural Networks. As for this results, we can use Random Forest Classifier (note: not a regressor) to predict how highly a person can get a heart disease. But before we apply RFC to our test data, let's tune our model with more better parameters

# ## Fine_Tuning the Model:

# In[85]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]},
    {'bootstrap': [False],'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]},
  ]

forest_clf = RandomForestClassifier()

grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='accuracy',
                           return_train_score=True)

grid_search.fit(train_prepared, dataLabels)


# In[86]:


grid_search.best_params_


# In[87]:


grid_search.best_estimator_


# In[88]:


cvres = grid_search.cv_results_


# In[89]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[90]:


cat_encoder = full_pipeline.named_transformers_["cat"]
cat_ordinal_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_ordinal_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[91]:


final_model = grid_search.best_estimator_


# In[92]:


from sklearn.metrics import make_scorer, accuracy_score

data_train2 = load_data()

X_all = data_train2.drop(['target'], axis=1)
y_all = data_train2['target']

X_all_transformed = full_pipeline.transform(X_all)

num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all_transformed, y_all, test_size=num_test, random_state=23)

final_model.fit(X_train, y_train)


# In[93]:


predictions = final_model.predict(X_test)
print(accuracy_score(y_test, predictions))


# ### As you can see above model reached 80% precision compared to the previous 80% of RandomForestClassifier and compared to 70% of DecisionTree Classifier with much more fine-tuned parameters

# In[ ]:




