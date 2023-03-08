import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dataprep.eda import create_report

#importing data
df = pd.read_csv("Desktop/STP_494_Project/dataset.csv")

df.info()


#exploring the dataset, with a simple report, we will build analysis off this
create_report(df)


#getting an idea of the most prevalent backgrounds, but first we must properly specify datatypes
df["Father's occupation"] = df["Father's occupation"].astype(dtype = str)
df["Mother's occupation"] = df["Mother's occupation"].astype(dtype = str)
df["Mother's qualification"] = df["Mother's qualification"].astype(dtype = str)
df["Father's qualification"] = df["Father's qualification"].astype(dtype = str)



#getting the top 5 occupations and qualifications for mother and father
    # Results on Occupation
    #in terms of occupation type both mother and father seem to hold similar styles of jobs
    #we also find that for fathers they tend to gravitate towards jobs with a heavy emphasis on labor,
    #mothers on the other hand tend to gravitate towards admin roles
df["Father's occupation"].value_counts(ascending = False).head(5)
df["Mother's occupation"].value_counts(ascending = False).head(5)

    # Results on Qualification
    # for qualifiactions we find that on average fathers tend to have less atainment in terms of education
    # mothers tend to have a higher level of education, which we find quite interesting
df["Mother's qualification"].value_counts(ascending = False).head(5)
df["Father's qualification"].value_counts(ascending = False).head(5)

#next we will analyze student backgrounds
    #percentage of studnets do not hold scholarships (66.9%)
    #percentage of students who do hold scholarships (33$)

#as excpected the distribution of enrollment ages has the top 5 within (18 - 22, in that order)
df['Scholarship holder'].value_counts(ascending = False).head(5)
df['Age at enrollment'].value_counts(ascending = False).head(5)

#lets condiiton on the target value to see how, scholarships effect dropout rates,
    #given a studnet has a scholarship, 835 graduated, 130 stayed enrolled and 134 students dropped out, quite good
df[['Scholarship holder', 'Target']].groupby('Target').sum()

#once we increase that gap though, those in their mid 20's begin to appear, still though the vast majority of students enrolled start before they are 30
df['Age at enrollment'].value_counts(ascending = False).head(10)

#exploring students past qualifications
df['Previous qualification'].value_counts(ascending = False).head(5)
