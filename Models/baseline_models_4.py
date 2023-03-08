#standard data loading and manipulation libraires
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#trying catboost
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

#metrics stuff
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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

#first we will test our parsimonious catboost model
X = df[['Tuition fees up to date', 'Scholarship holder','Age at enrollment', 'International', 'Debtor', 'Displaced' , 'Daytime/evening attendance', 'Binned 1st Semester', 'Binned 2nd Semester']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)


y_train = np.array(preprocessing.LabelBinarizer().fit_transform(y_train))
y_test = np.array(preprocessing.LabelBinarizer().fit_transform(y_test))

categorical_features_indices = np.where(X.dtypes != np.float)[0]


Cat = CatBoostClassifier(cat_features = categorical_features_indices)

Cat.fit(X_train, y_train)

metric_scoring(Cat, X_test, y_test, 'Catboost Classifier Baseline Parsimonious')
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(Cat, X_test, y_test)
plt.title("Confusion Matrix Cat Model")

predictions = Cat.predict_proba(X_test)

df_test = pd.DataFrame(X_test)
df_test['Target'] = y_test
df_test['prob_0'] = predictions[:,0]
df_test['prob_1'] = predictions[:,1]

df_test.head(5)


#now we will test the parsimonious logistic model
X = pd.get_dummies(df[['Tuition fees up to date', 'Scholarship holder','Age at enrollment', 'International', 'Debtor', 'Displaced' , 'Daytime/evening attendance', 'Binned 1st Semester', 'Binned 2nd Semester']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 100)

#binarizing labels and splitting up the train data into numeric and categorical
y_train = np.array(preprocessing.LabelBinarizer().fit_transform(y_train))
y_test = np.array(preprocessing.LabelBinarizer().fit_transform(y_test))

#fitting three baseline models
logit =  LogisticRegression(max_iter = 1000, penalty = 'l1', solver = 'liblinear').fit(X_train, y_train.ravel())

#geting metrics for the baseline models
metric_scoring(logit, X_test, y_test, 'Logit Baseline Model Dummy OHE')


#now we will use 10 fold cross validation on both models to see which one has better "general performance"
from catboost import Pool, cv

Y = np.array(preprocessing.LabelBinarizer().fit_transform(y))


cv_dataset = Pool(data= X,
                  label=Y,
                  cat_features=categorical_features_indices)

params = {"iterations": 5000,
          "depth": 2,
          "loss_function": "Logloss",
          "verbose": False}

scores = cv(cv_dataset,
            params,
            fold_count=5,
            return_models = True)

scores
