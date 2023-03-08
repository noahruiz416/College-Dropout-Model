#standard data loading and manipulation libraires
import pandas as pd
import numpy as np
import math

#models, to create pipelines for
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

#pipeline and data processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from category_encoders.quantile_encoder import QuantileEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.quantile_encoder import SummaryEncoder
from sklearn.model_selection import train_test_split


#metrics stuff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#visuals
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def metric_scoring(classifier, x_test_data, y_test_data, model_name):
  y_true = y_test_data
  y_pred = classifier.predict(x_test_data)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)

  metric_data = {
    'Model Name': model_name,
    'Precision' : round(precision,4),
    'Recall' : round(recall,4),
    'Accuracy': round(accuracy,4),
    'F1 Score': round(f1,4)
  }
  return metric_data




df = pd.read_csv('Desktop/STP_494_Project/cleaned_data.csv')
df = df.drop('Unnamed: 0', axis = 1)


#creating proper encodings, prepping for transformations + additional data prep
df['Marital status'] = df['Marital status'].astype(str)
df['Application mode'] = df['Application mode'].astype(str)
df['Course'] = df['Course'].astype(str)
df['Nacionality'] = df['Nacionality'].astype(str)
df["Mother's qualification"] = df["Mother's qualification"].astype(str)
df["Father's qualification"] = df["Father's qualification"].astype(str)
df["Mother's occupation"] = df["Mother's occupation"].astype(str)
df["Father's occupation"] = df["Father's occupation"].astype(str)



X = df.drop('Target', axis = 1)
y = df['Target']


#trying a simple dummy encoder
X = pd.get_dummies(X, drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

#binarizing labels and splitting up the train data into numeric and categorical
y_train = np.array(preprocessing.LabelBinarizer().fit_transform(y_train))
y_test = np.array(preprocessing.LabelBinarizer().fit_transform(y_test))

#fitting three baseline models
logit =  LogisticRegression(max_iter = 1000, penalty = 'l1', solver = 'liblinear').fit(X_train, y_train.ravel())

#geting metrics for the baseline models


metric_scoring(logit, X_test, y_test, 'Logit Baseline Model Dummy OHE')

ConfusionMatrixDisplay.from_estimator(logit, X_test, y_test)
plt.title("Confusion Matrix Logit Model")


#getting Coefficients for the baseline logit model
logit_coefficeints = np.transpose(np.round(logit.coef_, 2)).reshape(102,1)
column_names = np.array(X_train.columns).reshape((102,1))

data = {"Column Name": column_names.ravel(),
        "Estimated Logit Coefficient": logit_coefficeints.ravel()}

est_logit_coefficients = pd.DataFrame(data)

print(est_logit_coefficients.to_markdown())

#getting feature importance
feature_names = column_names
feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
feature_importance["importance"] = pow(math.e, logit_coefficeints)

feature_importance_top_10 = feature_importance.sort_values(by = ["importance"], ascending=False).head(10)
feature_importance_top_20 = feature_importance.sort_values(by = ["importance"], ascending=False).head(20)

ax = feature_importance_top_10.plot.barh(x='feature', y='importance')
plt.show()

print(feature_importance.to_markdown())



#trying a simple dummy encoder, with just a few indep vars (does suprisingly really well)

df.info()

X = pd.get_dummies(df[['Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'Debtor', 'Displaced', 'Unemployment rate', 'GDP', 'Inflation rate' , 'Daytime/evening attendance', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']], drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

#binarizing labels and splitting up the train data into numeric and categorical
y_train = np.array(preprocessing.LabelBinarizer().fit_transform(y_train))
y_test = np.array(preprocessing.LabelBinarizer().fit_transform(y_test))

#fitting three baseline models
logit =  LogisticRegression(max_iter = 1000, penalty = 'l1', solver = 'liblinear').fit(X_train, y_train.ravel())

#geting metrics for the baseline models


metric_scoring(logit, X_test, y_test, 'Logit Baseline Model Dummy OHE')


ConfusionMatrixDisplay.from_estimator(logit, X_test, y_test)
plt.title("Confusion Matrix Logit Model")

#getting Coefficients for the baseline logit model
logit_coefficeints = np.transpose(np.round(logit.coef_, 2)).reshape(13,1)
column_names = np.array(X_train.columns).reshape((13,1))

data = {"Column Name": column_names.ravel(),
        "Estimated Logit Coefficient": logit_coefficeints.ravel()}

est_logit_coefficients = pd.DataFrame(data)

print(est_logit_coefficients.to_markdown())

#getting feature importance
feature_names = column_names
feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
feature_importance["importance"] = pow(math.e, logit_coefficeints)

feature_importance_top_10 = feature_importance.sort_values(by = ["importance"], ascending=False).head(14)
feature_importance_top_20 = feature_importance.sort_values(by = ["importance"], ascending=False).head(20)

ax = feature_importance_top_10.plot.barh(x='feature', y='importance')
plt.show()

print(feature_importance.to_markdown())
