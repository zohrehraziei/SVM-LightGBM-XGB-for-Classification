#!/usr/bin/env python
# coding: utf-8

# ### Import necessary libraries

# In[1]:


# Data representation and computation
import pandas as pd  
import numpy as np  
pd.options.display.float_format = '{:20,.4f}'.format

# plotting
import matplotlib.pyplot as plt  
import seaborn as sns

# Data splitting and pipeline for training models
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Used ML models
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

# Miscellaneous
import warnings
from prettytable import PrettyTable

# Declaration
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('precision', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1)


# ### Load Data

# In[2]:


import json
with open(r"transactions.txt","r+") as f:
    llst_transaction_records = f.readlines()
llst_transaction_records = [json.loads(item) for item in llst_transaction_records]
ldf_training_dataset = pd.DataFrame(llst_transaction_records)

ldf_training_dataset['transactionDateTime'] = pd.to_datetime(ldf_training_dataset['transactionDateTime'])


# In[3]:


ldf_training_dataset.head()


# ## Question 1: Load

# ##### Statistical summary

# In[4]:


print("Number of records: "+str(len(ldf_training_dataset)))


# In[5]:


ldf_training_dataset.describe()


# In[6]:


ldf_training_dataset.info()


# ##### Checking for null or NA values

# In[8]:


ldf_training_dataset.isnull().sum()


# ##### Checking for values stored as blank ('')

# In[9]:


ldf_training_dataset.eq('').sum()


# ## Question 2: Plot

# ##### Plotting bar-chart of "isFraud`" 

# In[75]:


ldf_temp = ldf_training_dataset.isFraud.value_counts().to_frame("_count")
ldf_temp.plot(kind="bar")


# ##### Plotting histogram of "transactionAmount" 

# In[76]:


ldf_training_dataset['transactionAmount'].plot.hist()


# ## Question 3: Data Wrangling - Duplicate Transactions

# #### To identify Multi-swipe transactions, please see the explanation in Word documents)
# In[77]:


ldf_training_dataset['IsDuplicate'] = (ldf_training_dataset.sort_values(['transactionDateTime'])
                       .groupby(['accountNumber', 'transactionAmount', 'merchantName'], sort=False)['transactionDateTime']
                       .diff()
                       .dt.total_seconds()
                       .lt(300))


# In[78]:


ldf_training_dataset.IsDuplicate.value_counts().to_frame("Count")


# #### To identify Reversed transactions, using the Transaction Type column

# In[79]:


ldf_training_dataset.transactionType.value_counts().to_frame("Count")


# ## Estimating the total number and dollar amount of Normal & Duplicate Transactions.

# In[80]:


Model_Accuracy = PrettyTable()
Model_Accuracy.field_names = ["","No. of Transactions", "Doller Amount"]
Model_Accuracy.align[""] = "r"
ldf_training_dataset_normal = ldf_training_dataset[(ldf_training_dataset['transactionType']!="REVERSAL") &
                                           (ldf_training_dataset['IsDuplicate'] == False)]

Model_Accuracy.add_row(["Normal (Excluding Duplicates)",len(ldf_training_dataset_normal), 
                        round(sum(ldf_training_dataset_normal['transactionAmount']))])
print("\nA - Normal (Excluding Duplicates)")
print(Model_Accuracy)


# In[81]:



Model_Accuracy = PrettyTable()
Model_Accuracy.field_names = ["","No. of Transactions", "Doller Amount"]
Model_Accuracy.align[""] = "r"
ldf_training_dataset_rev = ldf_training_dataset[ldf_training_dataset["transactionType"] == "REVERSAL"]
ldf_training_dataset_multi_swipe = ldf_training_dataset[(ldf_training_dataset['transactionType']!="REVERSAL") &
                                           (ldf_training_dataset['IsDuplicate'] == True)]

Model_Accuracy.add_row(["Reversal",len(ldf_training_dataset_rev), 
                        round(sum(ldf_training_dataset_rev['transactionAmount']))])
Model_Accuracy.add_row(["Multi-Swipe",len(ldf_training_dataset_multi_swipe), 
                        round(sum(ldf_training_dataset_multi_swipe['transactionAmount']))])
Model_Accuracy.add_row(["","----","----"])
Model_Accuracy.add_row(["Reversal + Multi-Swipe",int(len(ldf_training_dataset_rev)+
                                                            len(ldf_training_dataset_multi_swipe)),
                        round(sum(ldf_training_dataset_rev['transactionAmount']))+
                       round(sum(ldf_training_dataset_multi_swipe['transactionAmount']))])

print("\nB - Duplicates(Reversal + Multi-Swipe)")
print(Model_Accuracy)


# In[82]:


# =====================================================================================================

Model_Accuracy = PrettyTable()
Model_Accuracy.field_names = ["","No. of Transactions", "Doller Amount"]
Model_Accuracy.align[""] = "r"

Model_Accuracy.add_row(["A + B",len(ldf_training_dataset_normal)+
                                    (len(ldf_training_dataset_rev)+len(ldf_training_dataset_multi_swipe)),
                        (round(sum(ldf_training_dataset_rev['transactionAmount']))+
                       round(sum(ldf_training_dataset_multi_swipe['transactionAmount'])))+
                       round(sum(ldf_training_dataset_normal['transactionAmount']))])
print("\nC = A + B (Normal & Duplicate both)")
print(Model_Accuracy)


# ##Question 4: Model

# ### Sampling to make the dataset balanced

# In[83]:


ldf_training_dataset.isFraud.value_counts().to_frame("Count")


# Since the number of isFraud=True cases is very low as compared to the isFraud=False cases,
# we will take out a balanced sample by keeping all the records of isFraud=True cases and taking only 18,000 records of isFraud=False cases.
# In[84]:


ldf_training_dataset_2 = ldf_training_dataset.groupby('isFraud', group_keys=False).apply(lambda x:x.sample(min(len(x), 18000)))
ldf_training_dataset_2 = ldf_training_dataset_2.reset_index(drop=True)
print("Number of records after under-sampling: "+str(len(ldf_training_dataset_2.isFraud)))


# In[85]:


ldf_training_dataset_2.isFraud.value_counts().to_frame("Count")


# ##### Removing the duplicate transactions (Reversal and Multi-swipe)

# In[86]:


ldf_training_dataset_3 = ldf_training_dataset_2[(ldf_training_dataset_2['transactionType']!="REVERSAL") &
                                           (ldf_training_dataset_2['IsDuplicate'] == False)]


# ## Predictive model to determine whether a given transaction will be fraudulent or not

# #### Select input variables

# In[87]:


input_col = ['creditLimit', 'transactionAmount', 'merchantCountryCode',
            'merchantCategoryCode','cardPresent', 'posEntryMode', 'posConditionCode', 
            'acqCountry', 'currentBalance', 'isFraud']
ldf_training_dataset_clf = ldf_training_dataset_3[input_col]


# #### Convert categorical columns to dummy variables 

# In[88]:


feature_dummy = pd.get_dummies(ldf_training_dataset_clf["creditLimit"], prefix='FRAUD_CLASSIFIER_creditLimit')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("creditLimit", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["merchantCountryCode"], prefix='FRAUD_CLASSIFIER_merchantCountryCode')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("merchantCountryCode", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["merchantCategoryCode"], prefix='FRAUD_CLASSIFIER_merchantCategoryCode')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("merchantCategoryCode", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["cardPresent"], prefix='FRAUD_CLASSIFIER_cardPresent')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("cardPresent", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["posEntryMode"], prefix='FRAUD_CLASSIFIER_posEntryMode')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("posEntryMode", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["posConditionCode"], prefix='FRAUD_CLASSIFIER_posConditionCode')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("posConditionCode", axis=1, inplace=True)

feature_dummy = pd.get_dummies(ldf_training_dataset_clf["acqCountry"], prefix='FRAUD_CLASSIFIER_acqCountry')
ldf_training_dataset_clf = pd.concat([ldf_training_dataset_clf, feature_dummy], axis = 1)
ldf_training_dataset_clf.drop("acqCountry", axis=1, inplace=True)


# #### Train Test & Validation Split

# In[97]:


y = ldf_training_dataset_clf.isFraud
X = ldf_training_dataset_clf.drop("isFraud", axis=1)

my_tags = y


# In[98]:


# splitting X and y into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123, shuffle=True)


# In[99]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Algorithm 1: Support Vector Machines

# ##### Define a model

# In[89]:


clf_svm = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=1e-3,
        random_state=42,
        max_iter=5,
        tol=None)


# ##### Fit model on training data

# In[102]:


svm_model = clf_svm.fit(X_train, y_train)


# ##### Predicting on test data

# In[103]:


svm_y_pred = svm_model.predict(X_test)


# ##### Model evaluation metrics: Accuracy, Precision, Recall, F1 Score

# In[95]:


confusion_matrix(y_test,svm_y_pred)


# In[109]:


print('Accuracy %s' % accuracy_score(svm_y_pred, y_test))
print('Precision:', precision_score(svm_y_pred, y_test, average='weighted'))
print('Recall:', recall_score(svm_y_pred, y_test,average='weighted'))
print('F1 score:', f1_score(svm_y_pred, y_test,average='weighted'))


# ### Algorithm 2: Light GBM (Light Gradient Boosting Machine Classifier)

# ##### Define a model

# In[112]:


clf_lgbm = lgb.LGBMClassifier()


# ##### Fit model on training data

# In[ ]:


lgbm_model = clf_lgbm.fit(X_train, y_train)


# ##### Predicting on test data

# In[ ]:


lgbm_y_pred = lgbm_model.predict(X_test)


# ##### Model evaluation metrics: Accuracy, Precision, Recall, F1 Score

# In[113]:


confusion_matrix(y_test,lgbm_y_pred)


# In[114]:


print('Accuracy %s' % accuracy_score(lgbm_y_pred, y_test))
print('Precision:', precision_score(lgbm_y_pred, y_test, average='weighted'))
print('Recall:', recall_score(lgbm_y_pred, y_test,average='weighted'))
print('F1 score:', f1_score(lgbm_y_pred, y_test,average='weighted'))


# ### Algorithm 3: XGB (Extreme Gradient Boost Classifier)

# ##### Define a model

# In[116]:


clf_xgb = XGBClassifier()


# ##### Fit model on training data

# In[117]:


xgb_model = clf_xgb.fit(X_train, y_train)


# ##### Predicting on test data

# In[118]:


xgb_y_pred = xgb_model.predict(X_test)


# ##### Model evaluation metrics: Accuracy, Precision, Recall, F1 Score

# In[119]:


confusion_matrix(y_test,xgb_y_pred)


# In[120]:


print('Accuracy %s' % accuracy_score(xgb_y_pred, y_test))
print('Precision:', precision_score(xgb_y_pred, y_test, average='weighted'))
print('Recall:', recall_score(xgb_y_pred, y_test,average='weighted'))
print('F1 score:', f1_score(xgb_y_pred, y_test,average='weighted'))


# <br>

# ## Using an estimate of performance using an appropriate sample

# ### Model Performance Comparison

# In[121]:


models = {'SVM': [accuracy_score(svm_y_pred, y_test),
                      precision_score(svm_y_pred, y_test, average='weighted'),
                      recall_score(svm_y_pred, y_test, average='weighted'),
                      f1_score(svm_y_pred, y_test,average='weighted')],
          'LGBM': [accuracy_score(lgbm_y_pred, y_test),
                      precision_score(lgbm_y_pred, y_test, average='weighted'),
                      recall_score(lgbm_y_pred, y_test,average='weighted'),
                      f1_score(lgbm_y_pred, y_test,average='weighted')],
          'XGB': [accuracy_score(xgb_y_pred, y_test),
                      precision_score(xgb_y_pred, y_test, average='weighted'),
                      recall_score(xgb_y_pred, y_test,average='weighted'),
                      f1_score(xgb_y_pred, y_test,average='weighted')]
         }

df_models = pd.DataFrame(models, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

df_models


# In[125]:


plt.rcParams["figure.figsize"] = (15,8)

df_models.plot.bar(edgecolor = "white")
plt.legend().set_visible(True)
plt.title("Comparing Classification Performance of Multiple Algorithms")
plt.xticks(rotation = 0)
plt.show()




# In[115]:


feat_importances = pd.Series(lgbm_model.feature_importances_, index=X_test.columns)
feat_importances.nsmallest(20).plot(kind='barh')
plt.show()


# We can observe that the top 5 features for determining Fraud Transactions are:
# * posConditionCode=01
# * merchantCategoryCode="furniture"
# * creditLimit=250
# * creditLimit=1000
# * acqCountry= " "

# In[ ]:




