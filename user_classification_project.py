# -*- coding: utf-8 -*-

# Machine Learning for User Classification

import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Trying out with different algorithms.

"""# Objective

The ability to predict potential paying customers is critical not only for 365 Data Science but for any online platform. Such predictions can:

Enhance targeted advertising campaigns.
Support outreach initiatives with exclusive offers.
Optimize marketing budgets by focusing on users most likely to convert to paid customers.
This project aims to increase company revenue through data-driven strategies.

# Data Preprocessing
"""

raw_data = pd.read_csv('data/ml_datasource.csv')
raw_data.head()

data = raw_data.copy()

sns.reset_orig()
sns.set(font_scale=1.5)
fig, axes = plt.subplots(3, 2, figsize=(20,20))

# Plotting distribution plots for each of the columns in the dataset

sns.kdeplot(data=data['days_on_platform'], ax=axes[0,0])
sns.kdeplot(data=data['minutes_watched'], ax=axes[0,1])
sns.kdeplot(data=data['courses_started'], ax=axes[1,0])
sns.kdeplot(data=data['practice_exams_started'], ax=axes[1,1])
sns.kdeplot(data=data['practice_exams_passed'], ax=axes[2,0])
sns.kdeplot(data=data['minutes_spent_on_exams'], ax=axes[2,1]);

plt.show()

#removing outliers

data_no_outliers = data[(data['minutes_watched'] <= 1000)
                            & (data['courses_started']<=10)
                            & (data['practice_exams_started']<=10)
                            & (data['minutes_spent_on_exams']<=40)]

sns.reset_orig()
sns.set(font_scale=1.5)
fig, axes = plt.subplots(3, 2, figsize=(20,20))


sns.kdeplot(data=data_no_outliers['days_on_platform'], ax=axes[0,0])
sns.kdeplot(data=data_no_outliers['minutes_watched'], ax=axes[0,1])
sns.kdeplot(data=data_no_outliers['courses_started'], ax=axes[1,0])
sns.kdeplot(data=data_no_outliers['practice_exams_started'], ax=axes[1,1])
sns.kdeplot(data=data_no_outliers['practice_exams_passed'], ax=axes[2,0])
sns.kdeplot(data=data_no_outliers['minutes_spent_on_exams'], ax=axes[2,1]);

plt.show()

"""### Multicollinearity Check"""

data_no_outliers.columns.to_numpy()

variables = data_no_outliers[['days_on_platform',
                              'minutes_watched',
                              'courses_started',
                              'practice_exams_started',
                              'practice_exams_passed',
                              'minutes_spent_on_exams']]

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.to_numpy(), i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif

# Dropping 'practice_exams' to prevent multicollinearity
data_no_mult = data_no_outliers.drop('practice_exams_started', axis = 1)
data_no_mult.head()

#again

variables = data_no_outliers[['days_on_platform',
                              'minutes_watched',
                              'courses_started',
                              'practice_exams_passed',
                              'minutes_spent_on_exams']]

vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.to_numpy(), i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif

"""### Handling NaN Values"""

data_no_mult.isnull().sum()

data_no_mult.loc[ data_no_mult['student_country'].isna()]

data_no_nulls = data_no_mult.fillna('NAM', axis = 1)

data_no_nulls.loc[ data_no_nulls['student_country'] == 'NAM', 'student_country']

data_no_nulls.isnull().sum()

inputs = data_no_nulls.drop(['purchased'],axis=1)
target = data_no_nulls['purchased']

#train test split

x_train, x_test, y_train, y_test = train_test_split(inputs,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=365,
                                                    stratify = target)

x_train.head()

"""### Feature Engineering"""

enc = OrdinalEncoder(handle_unknown = 'use_encoded_value',
                     unknown_value = 170);

x_train['student_country_enc'] = enc.fit_transform(x_train['student_country'].to_numpy().reshape(-1, 1));
x_test['student_country_enc'] = enc.transform(x_test['student_country'].to_numpy().reshape(-1, 1));

x_train = x_train.drop('student_country', axis = 1)
x_test = x_test.drop('student_country', axis = 1)

x_train.head()

#sklearn/statsmodel acts up sometimes when casting data from pandas to arrays so I'm doing it here explicitly

x_train_array = np.asarray(x_train, dtype = 'float')
y_train_array = np.asarray(y_train, dtype = 'int')

x_test_array = np.asarray(x_test, dtype = 'float')
y_test_array = np.asarray(y_test, dtype = 'int')

"""# Logistic Regression Model"""

log_reg = sm.Logit(y_train_array, x_train_array)
log_reg_results = log_reg.fit()
log_reg_results.summary()

y_test_pred_log_reg = [round(log_reg_results.predict(x_test_array)[i], 0)
                       for i in range(len(y_test_array))]

sns.reset_orig()

ConfusionMatrixDisplay.from_predictions(
    y_test_array, y_test_pred_log_reg,
    cmap = 'magma'
);

plt.show()

"""# KNN"""

#Brute force parameter tuning

parameters_knn = {'n_neighbors':range(1, 51),
                  'weights':['uniform', 'distance']}

grid_search_knn = GridSearchCV(estimator = KNeighborsClassifier(),
                               param_grid = parameters_knn,
                               scoring = 'accuracy')

grid_search_knn.fit(x_train_array, y_train_array)

grid_search_knn.best_params_, grid_search_knn.best_score_

knn_clf = grid_search_knn.best_estimator_
knn_clf

y_test_pred_knn = knn_clf.predict(x_test_array)
sns.reset_orig()

ConfusionMatrixDisplay.from_predictions(
    y_test_array, y_test_pred_knn,
    labels = knn_clf.classes_,
    cmap = 'magma'
);

plt.show()

# Eval
print(classification_report(y_test_array,
                            y_test_pred_knn,
                            target_names = ['0', '1']))

"""# Creating a Support Vector Machines Model"""

scaling = MinMaxScaler(feature_range=(-1,1))
x_train_array_svc = scaling.fit_transform(x_train_array)
x_test_array_svc = scaling.transform(x_test_array)

parameters_svc = {'kernel':['linear', 'poly', 'rbf'],
                  'C':range(1, 11),
                  'gamma': ['scale', 'auto']}

grid_search_svc = GridSearchCV(estimator = SVC(),
                               param_grid = parameters_svc,
                               scoring = 'accuracy')

#Brute Force approach again, takes time

grid_search_svc.fit(x_train_array_svc, y_train_array)

grid_search_svc.best_estimator_

svc_clf = grid_search_svc.best_estimator_

y_test_pred_svc = svc_clf.predict(x_test_array_svc)
sns.reset_orig()

ConfusionMatrixDisplay.from_predictions(
    y_test_array, y_test_pred_svc,
    labels = svc_clf.classes_,
    cmap = 'magma'
);

plt.show()

print(classification_report(y_test_array,
                            y_test_pred_svc,
                            target_names = ['0', '1']))

"""# Creating a Decision Trees Model"""

#For hyperparam tunig

parameters_dt = {'ccp_alpha':[0,
                              0.001,
                              0.002,
                              0.003,
                              0.004,
                              0.005]}

grid_search_dt = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 365),
                              param_grid = parameters_dt,
                              scoring = 'accuracy')

grid_search_dt.fit(x_train_array, y_train_array)

grid_search_dt.best_estimator_

dt_clf = grid_search_dt.best_estimator_

plt.figure(figsize=(15,10))
plot_tree(dt_clf,
          filled=True,
          feature_names = ['Days on platform',
                           'Minutes watched',
                           'Courses started',
                           'Practice exams passed',
                           'Time spent on exams',
                           'Student country encoded'],
          class_names = ['Will not purchase',
                         'Will purchase'])

plt.show()

y_test_pred_dt = dt_clf.predict(x_test_array)

sns.reset_orig()

ConfusionMatrixDisplay.from_predictions(
    y_test_array, y_test_pred_dt,
    labels = dt_clf.classes_,
    cmap = 'magma'
);

plt.show()

print(classification_report(y_test_array, y_test_pred_dt))

"""# Creating a Random Forests Model"""

# might have to change penalty depending if overfitting
rf_clf = RandomForestClassifier(ccp_alpha = 0.0001, random_state = 365)

rf_clf.fit(x_train_array, y_train_array)

y_test_pred_rf = rf_clf.predict(x_test_array)

sns.reset_orig()

ConfusionMatrixDisplay.from_predictions(
    y_test_array, y_test_pred_rf,
    labels = rf_clf.classes_,
    cmap = 'magma'
);

plt.show()

print(classification_report(y_test_array, y_test_pred_rf))

