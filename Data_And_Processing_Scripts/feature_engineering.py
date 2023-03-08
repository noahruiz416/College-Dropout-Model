#this script will contain the first iteration of data cleaning / engineering.
    #from here if further iteration is needed it will be added

import pandas as pd
import numpy as np
import math

df = pd.read_csv("Desktop/STP_494_Project/dataset.csv")


#converting data to proper type (encodings -> categorical)
df['Marital status'] = df['Marital status'].astype(str)
df['Application mode'] = df['Application mode'].astype(str)
df['Course'] = df['Course'].astype(str)
df['Nacionality'] = df['Nacionality'].astype(str)
df["Mother's qualification"] = df['Marital status'].astype(str)
df["Father's qualification"] = df['Marital status'].astype(str)
df["Mother's occupation"] = df['Marital status'].astype(str)
df["Father's occupation"] = df['Marital status'].astype(str)
df['Application order'] = df['Application order'].astype(str)

#since their were issues with cardinality in the independent variables, we will do some binning, based on a set of criteria

#in addition we will drop any records that contain Enrolled in the target column
new_data = df.query('Target != "Enrolled"')

new_data.info()


GPA_SCORES_1 = []
for row in new_data['Curricular units 1st sem (grade)']:
    if row <= 20 and row >= 18:
        GPA_SCORES_1.append("Very High GPA")
    if row < 18 and row >= 16:
        GPA_SCORES_1.append("High GPA")
    if row < 16 and row >= 14:
        GPA_SCORES_1.append("Above Average GPA")
    if row < 14 and row >= 12:
        GPA_SCORES_1.append("Average GPA")
    if row < 12 and row >= 9:
        GPA_SCORES_1.append("Low GPA")
    if row < 9:
        GPA_SCORES_1.append("Very Low GPA")

GPA_SCORES_2 = []
for row in new_data['Curricular units 2nd sem (grade)']:
    if row <= 20 and row >= 18:
        GPA_SCORES_2.append("Very High GPA")
    if row < 18 and row >= 16:
        GPA_SCORES_2.append("High GPA")
    if row < 16 and row >= 14:
        GPA_SCORES_2.append("Above Average GPA")
    if row < 14 and row >= 12:
        GPA_SCORES_2.append("Average GPA")
    if row < 12 and row >= 9:
        GPA_SCORES_2.append("Low GPA")
    if row < 9:
        GPA_SCORES_2.append("Very Low GPA")

#trimmed_data_dropped['TRANSFORMED_GPA'] = GPA_SCORES
new_data['Binned 1st Semester'] = GPA_SCORES_1
new_data['Binned 2nd Semester'] = GPA_SCORES_2

new_data

#saving this dataframe for modeling (only run if needed)
new_data.to_csv("Desktop/STP_494_Project/cleaned_data.csv")
