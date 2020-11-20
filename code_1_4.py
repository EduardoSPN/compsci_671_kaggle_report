# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:41:32 2020

@author: eschi
"""

import csv 
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import statsmodels.api as sm
from datetime import datetime

from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.neural_network import MLPClassifier
from sklearn.utils import class_weight

from sklearn.naive_bayes import GaussianNB

np.random.seed(123)


##############################################################################
# ============================================================================
# # LOAD DATA
# ============================================================================
##############################################################################

# Read-in data w/o outliers from R
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##############################################################################
# ============================================================================
# # DATA CLEANING AND FEATURE EVALUATION
# ============================================================================
##############################################################################

# We drop 'is_business_travel_ready' because it has no variability (all zeros)
train = train.drop("is_business_travel_ready", axis=1)
test =  test.drop("is_business_travel_ready", axis=1)

# y = train["price"]
# x = train.iloc[:,:-1]
# x_dum = pd.get_dummies(train.drop(["last_review", "host_since"], axis=1))


# We don't include 'is_business_travel_ready' because it only has true values 
cat_cols = ["neighbourhood", "room_type", "host_is_superhost",
            "bed_type", "instant_bookable", "cancellation_policy",
            "require_guest_profile_picture", "require_guest_phone_verification"]


date_cols = ["last_review", "host_since"]

# here we include the date_vars converted into integers
num_cols = ["minimum_nights", "number_of_reviews", "reviews_per_month", 
            "calculated_host_listings_count", "availability_365", "bathrooms",
            "bedrooms", "beds", "cleaning_fee", "guests_included", "extra_people",
            "maximum_nights", "last_review_num", "host_since_num"]


# =============================================================================
# Clean Train - Encoding ' LabelEconder' and Scale of categorical vars
# =============================================================================

train_clean = copy.deepcopy(train)

#### Deal with dates and convert to feature ######
# Convert date vars to datetime
train_clean[date_cols] = train_clean[date_cols].apply(pd.to_datetime)

# Convert to feature
train_clean["last_review_num"] = train_clean["last_review"].apply(lambda x: int(x.strftime("%Y%m%d%H%M%S")))
train_clean["host_since_num"] = train_clean["host_since"].apply(lambda x: int(x.strftime("%Y%m%d%H%M%S")))

# caca = pd.Series(train_clean_clean["host_since_num"])
# caca.sort_values().plot(use_index=False)

# Only keep the numeric dates
train_clean=train_clean.drop(date_cols, axis=1)

# Put 'price' at the end just for convenience
cols_train_clean = train_clean.columns.tolist()
cols_train_clean[-1], cols_train_clean[-3] = cols_train_clean[-3], cols_train_clean[-1]
cols_train_clean 

train_clean = train_clean[cols_train_clean]

#### Encode ######
label_encoder = LabelEncoder()
scaler = preprocessing.MinMaxScaler()

# Encode categorical vars
for j in cat_cols:
  train_clean.loc[:,j] = label_encoder.fit_transform(train_clean.loc[:,j]).astype('float64')

#### Rescale ######
# Rescale numerical vars 
scaler.fit(train_clean.loc[:, num_cols])
train_clean_minmax = pd.DataFrame(scaler.transform(train_clean.loc[:, num_cols]), columns= num_cols)


# We want eveything in train_clean_clean
train_clean.loc[:, num_cols] = train_clean_minmax


#### Store Data ######
# writing to csv file  
# train_clean_only_encode.to_csv(r'train_clean.csv', index = False, header=True)


# =============================================================================
# Clean test - Encoding ' LabelEconder' and Scale of categorical vars
# =============================================================================

test_clean = copy.deepcopy(test)

#### Deal with dates and convert to feature ######
# Convert date vars to datetime
test_clean[date_cols] = test_clean[date_cols].apply(pd.to_datetime)

# Convert to feature
test_clean["last_review_num"] = test_clean["last_review"].apply(lambda x: int(x.strftime("%Y%m%d%H%M%S")))
test_clean["host_since_num"] = test_clean["host_since"].apply(lambda x: int(x.strftime("%Y%m%d%H%M%S")))

# caca = pd.Series(test_clean_clean["host_since_num"])
# caca.sort_values().plot(use_index=False)

# Only keep the numeric dates
test_clean=test_clean.drop(date_cols, axis=1)

# # Put 'price' at the end just for convenience
# cols_test_clean = test_clean.columns.tolist()
# cols_test_clean[-1], cols_test_clean[-3] = cols_test_clean[-3], cols_test_clean[-1]
# cols_test_clean 

# test_clean = test_clean[cols_test_clean]

#### Only Encode ######
label_encoder = LabelEncoder()
scaler = preprocessing.MinMaxScaler()

# Encode categorical vars
for j in cat_cols:
  test_clean.loc[:,j] = label_encoder.fit_transform(test_clean.loc[:,j]).astype('float64')

#### Rescale######
# Rescale numerical vars 
scaler.fit(test_clean.loc[:, num_cols])
test_clean_minmax = pd.DataFrame(scaler.transform(test_clean.loc[:, num_cols]), columns= num_cols)


# We want eveything in test_clean_clean
test_clean.loc[:, num_cols] = test_clean_minmax


#### Store Data ######
# writing to csv file  
# test_clean_only_encode.to_csv(r'test_clean.csv', index = False, header=True)


# =============================================================================
# Correlation Matrix
# =============================================================================


# Get data ready for algo (We drop the old date_cols and keep the new integer ones)
d= train_clean.drop(["id"], axis=1)
x = d.drop("price", axis=1)
y = d.loc[:,"price"]


corr = d.corr()
sns.heatmap(corr , yticklabels = True)

# =============================================================================
# Create dummies for categorical - train
# =============================================================================

train_cat = train.loc[:,cat_cols]

# We need to use the data before the encoding we used for the corr_matrix
train_dums = pd.get_dummies(train_cat)

# Everything in one dataframe - Clean data for numerical and the dummies for categorical
train_ready = copy.deepcopy(train_clean)

# We remove the initial categorical vars to include the dummies
train_ready = train_ready.drop(cat_cols, axis=1)

# We concatenate the numerical and dummies
train_ready  = pd.concat([train_ready, train_dums], axis=1)


# =============================================================================
# Create dummies for categorical - test
# =============================================================================

test_cat = test.loc[:,cat_cols]

# We need to use the data before the encoding we used for the corr_matrix
test_dums = pd.get_dummies(test_cat)

# Everything in one dataframe - Clean data for numerical and the dummies for categorical
test_ready = copy.deepcopy(test_clean)

# We remove the initial categorical vars to include the dummies
test_ready = test_ready.drop(cat_cols, axis=1)

# We concatenate the numerical and dummies
test_ready  = pd.concat([test_ready, test_dums], axis=1)


##############################################################################
# =============================================================================
# FEATURE SELECTION
# =============================================================================
##############################################################################

# First we want only the vars that are also in the test set
#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]


# =============================================================================
# L1-based feature selection
# =============================================================================

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)

x_new = model.transform(x)
x_new.shape


# =============================================================================
# Tree-based feature selection
# ============================================================================

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(x, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)

x_new_2 = model.transform(x)
x_new_2.shape               




# =============================================================================
# # DATA FOR MODELS
# =============================================================================
#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')
#==========================================================================



##############################################################################
# =============================================================================
# RANDOMIZED DECISION TREES
# =============================================================================
##############################################################################

# Before feature Selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = ExtraTreesClassifier(n_estimators=100, random_state=0, criterion= 'entropy', bootstrap= True)
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

# # With L1 Feature SElection
# x = x_new
# y = d.loc[:,"price"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
# clf.fit(x_train, y_train)

# print(clf.score(x_test,y_test))


# # With Tree-based feature selection
# x = x_new_2
# y = d.loc[:,"price"]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
# clf.fit(x_train, y_train)

# # Score - mean accuracy on the given test data and labels
# # Note - In multi-label classification, this is the subset accuracy 
# # which is a harsh metric since you require for each sample that each label set be correctly predicted.
# # print(rf.score(x_train,y_train))
# print(clf.score(x_test,y_test))

## Better without feature selection

# =============================================================================
# Predictions on the actual test set
# =============================================================================

# Not all the new dummies from the train set are in the test set. 
# We need to keep in the train set only the cols that are also in the test set.


# Extra Trees - Before feature Selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(x, y)

#prediction on actual test
tree_pred = clf.predict(x_t)

tree_pred = pd.DataFrame(data=tree_pred,columns=["price"])

tree_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
tree_pred.to_csv(r'tree_pred.csv', index = True, header=True)


##############################################################################
# ============================================================================
# ADABOOST WITH TREES
# ============================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Tree
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
# clf.fit(x, y)

# Adaboost based on Tree
adab_tree = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)

# evaluate the model before fitting
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(adab_tree, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# # report performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Fit to Test
adab_tree.fit(x_train, y_train)

adab_tree.score(x_test, y_test)


# =============================================================================
# Predictions on the actual test set
# =============================================================================

# Not all the new dummies from the train set are in the test set. 
# We need to keep in the train set only the cols that are also in the test set.

# Extra Trees - Before feature Selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Tree
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)

# Adaboost based on Tree
adab_tree = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)


#prediction on actual test
adab_tree_pred = adab_tree.predict(x_t)

adab_tree_pred = pd.DataFrame(data=adab_tree_pred, columns=["price"])

adab_tree_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
adab_tree_pred.to_csv(r'adab_tree_pred.csv', index = True, header=True)


##############################################################################
# ============================================================================
# RANDOM FOREST
# ============================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')


# Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap= True)

rf.fit(x_train,y_train)

print(rf.score(x_test,y_test))


# =============================================================================
# Predictions on the actual test set
# =============================================================================

# Not all the new dummies from the train set are in the test set. 
# We need to keep in the train set only the cols that are also in the test set.


# Extra Trees - Before feature Selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap= True)

rf.fit(x_train,y_train)


#prediction on actual test
rf_pred = rf.predict(x_t)

rf_pred = pd.DataFrame(data=rf_pred,columns=["price"])

rf_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
rf_pred.to_csv(r'rf_pred.csv', index = True, header=True)




##############################################################################
# ============================================================================
# ADABOOST WITH RANDOM FOREST
# ============================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap= True)

# Adaboost of the Random Forest
adab = AdaBoostClassifier(base_estimator=rf,n_estimators=rf.n_estimators)

# evaluate the model before fitting
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(adab, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# # report performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Fit to Test
adab.fit(x_train, y_train)

adab.score(x_test, y_test)


# =============================================================================
# Predictions on the actual test set
# ============================================================================


#prediction on actual test
adab_rf_pred = adab.predict(x_t)

adab_rf_pred = pd.DataFrame(data=adab_rf_pred,columns=["price"])

adab_rf_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
adab_rf_pred.to_csv(r'adab_rf_pred.csv', index = True, header=True)


##############################################################################
# ============================================================================
# RANDOM FOREST IDENTIFYING RELEVANT FEATURES AND RE-CLASSIFYING
# ============================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')


# Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap= True)

rf.fit(x_train,y_train)

print(rf.score(x_test,y_test))

# Create a list of feature names
feat_labels = list(x.columns)

# Print the name and gini importance of each feature
feat_vals = []
for feature in zip(feat_labels, rf.feature_importances_):
    print(feature)
    feat_vals.append(feature[1])

np.min(feat_vals)
np.max(feat_vals)


# =============================================================================
# Lets chose ITERATIVELY the importance threshold to select relevant features
# =============================================================================

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

# Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

rf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, bootstrap=True)


acc = []

for i in np.arange(0.003, 0.0081, 0.0001):
    sfm = SelectFromModel(rf, threshold=i)
    sfm.fit(x_train, y_train)
    x_important_train = sfm.transform(x_train)
    x_important_test = sfm.transform(x_test)
    rf_important.fit(x_important_train, y_train)
    s = rf_important.score(x_important_test, y_test)
    acc.append(s)
    
acc = np.array(acc)
acc_max = np.max(acc)

indx = np.where(acc == acc_max)[0][0]

thres_relev = np.arange(0.003, 0.0081, 0.0001)[indx]

# =============================================================================
# Use  most relevant features in Random Forest - based on the threshold we found
# =============================================================================

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap= True)

# Random Forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.05
sfm = SelectFromModel(rf, threshold=thres_relev )

#Relevant range --> 0.003 - 0.08
# Train the selector
sfm.fit(x_train, y_train)

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
print(len(sfm.get_support(indices=True)))

# =============================================================================
# Create A Data Subset With Only The Most Important Features
# =============================================================================

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)

#Actual test set
x_t_important = sfm.transform(x_t)

# =============================================================================
# Train A New Random Forest Classifier Using Only Most Important Features
# =============================================================================

#### Form previous random forest (full model) ####

# # Train the Full Featured Classifier 
# rf.fit(x_train, y_train)

# # View The Accuracy Of Our Full Feature Model
# rf.score(x_test, y_test)

#### New one (limited features) #####

# Create a new random forest classifier for the most important features
rf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, bootstrap=True)

# Train the new classifier on the new dataset containing the most important features
rf_important.fit(x_important_train, y_train)

# View The Accuracy Of Our Limited Feature Model
rf_important.score(x_important_test, y_test)


# =============================================================================
# Predictions on the actual test set
# ============================================================================

# We need the same number of features in the test set

#prediction on actual test
rf_important_pred = rf_important.predict(x_t_important)

rf_important_pred = pd.DataFrame(data=rf_important_pred,columns=["price"])

rf_important_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
rf_important_pred.to_csv(r'rf_important_pred.csv', index = True, header=True)


##############################################################################
# ============================================================================
# ADABOOST WITH RANDOM FOREST - BASED ON 'IMPORTANT FEATURES'
# ============================================================================
##############################################################################

# #train data
# cols = train_ready.columns.isin(test_ready.columns)

# x = train_ready.loc[:,cols].set_index('id')
# y = train_ready.loc[:,"price"]

# #test data
# x_t = test_ready.set_index('id')


# Random Forest
rf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
rf_important.fit(x_important_train, y_train)

# Adaboost of the Random Forest
adab_rf_important = AdaBoostClassifier(base_estimator=rf_important,n_estimators=rf_important.n_estimators)

# evaluate the model before fitting
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(adab, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# # report performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Fit to Test
adab_rf_important.fit(x_important_train, y_train)

adab_rf_important.score(x_important_test, y_test)


# =============================================================================
# Predictions on the actual test set
# ============================================================================


#prediction on actual test
adab_rf_important_pred = adab_rf_important.predict(x_t_important)

adab_rf_important_pred = pd.DataFrame(data=adab_rf_important_pred,columns=["price"])

adab_rf_important_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
adab_rf_important_pred.to_csv(r'adab_rf_important_pred.csv', index = True, header=True)



#############################################################################
# ===========================================================================
# NEURAL NETS
# ===========================================================================
#############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')


# Using Cross Validation
# kf = KFold(n_splits=10)
# nn_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, activation = 'softmax')
# nn_model.fit(x,y)


# cv_results = cross_validate(nn_model, x, y, cv=10, 
#                             return_train_score=False, 
#                             scoring=nn_model.score) 
# print("Fit scores: {}".format(cv_results['test_score']))


# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

snn = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, activation = 'logistic')
snn.fit(x_train, y_train)
# snn_pred = snn.predict(x_test)
snn.score(x_test, y_test)

# More layers
# dnn_classifier = MLPClassifier(hidden_layer_sizes = [100]*5, activation = "logistic")
# dnn_classifier.fit(x_train, y_train)
# # dnn_predictions = dnn_classifier.predict(x_test)
# dnn_classifier.score(x_test, y_test)


# Digresion --> This allows us to use adaboost on the MLPClassifer (neural nets)
#############################################################################
class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):

        # normalize sample_weights if not already
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled


    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


#############################################################################

adabooster = AdaBoostClassifier(base_estimator=customMLPClassifer())

# Adaboost of the Random Forest
adab = AdaBoostClassifier(base_estimator=rf,n_estimators=rf.n_estimators)

adabooster.fit(x_train, y_train)

# Fit to Test
adab.fit(x_train, y_train)

adabooster.score(x_test, y_test)

# =============================================================================
# Predictions on the actual test set
# ============================================================================

#prediction on actual test
dnn_pred = dnn_classifier.predict(x_t)

dnn_pred = pd.DataFrame(data=dnn_pred,columns=["price"])

dnn_pred["id"] = x_t.index

# =============================================================================
# Store Predictions
# =============================================================================
# writing to csv file  
dnn_pred.to_csv(r'dnn_pred.csv', index = True, header=True)


#############################################################################
# ===========================================================================
# LOGISTIC REGRESSION
# ===========================================================================
#############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logit = LogisticRegressionCV(cv=10, random_state=0).fit(x_train, y_train)
logit.score(x_test, y_test)


# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

logit = LogisticRegressionCV(cv=10, random_state=0).fit(x_train, y_train)
logit.score(x_test, y_test)




logit = LogisticRegressionCV(cv=10, random_state=0).fit(x_important_train, y_train)
logit.score(x_important_test, y_test)

# =============================================================================
# Predictions on the actual test set
# ============================================================================

# Dont Bother

# =============================================================================
# Store Predictions
# =============================================================================


##############################################################################
# ===========================================================================
# NAIVE BAYES
# ===========================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')


# For training evaluation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
mnb = MultinomialNB().fit(x_train, y_train)

mnb.score(x_test, y_test)


##############################################################################
# ===========================================================================
# Linear SVM
# ===========================================================================
##############################################################################

#train data
cols = train_ready.columns.isin(test_ready.columns)

x = train_ready.loc[:,cols].set_index('id')
y = train_ready.loc[:,"price"]

#test data
x_t = test_ready.set_index('id')

# Training Accuracy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

svc=SVC() # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)

prediction = svc.predict(x_test)

cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)



