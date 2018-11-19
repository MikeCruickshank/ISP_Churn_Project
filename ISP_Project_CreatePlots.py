# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:22:17 2018

@author: Mike
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


showSummary = False
showCounts = False
showPlots = True

saveFigure = False

df=pd.read_csv('ispcustomerchurn.csv')

df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)
df = df[df["TotalCharges"].notnull()]
df = df.reset_index()[df.columns]
df["TotalCharges"] = df["TotalCharges"].astype(float)

df["SeniorCitizen"] = df["SeniorCitizen"].replace({1:"Yes",0:"No"})

def tenure_lab(df) :
    
    if df["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (df["tenure"] > 12) & (df["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (df["tenure"] > 24) & (df["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (df["tenure"] > 48) & (df["tenure"] <= 60) :
        return "Tenure_48-60"
    elif df["tenure"] > 60 :
        return "Tenure_gt_60"
df["Tenure_group"] = df.apply(lambda df:tenure_lab(df),axis=1)
      

cat_cols = ['gender','SeniorCitizen','Partner','Dependents', 'Tenure_group',
                        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
                        'PaperlessBilling','PaymentMethod','Churn']
                        

                        
id_col     = ['customerID']
target_col = ["Churn"]

num_cols   = [x for x in df.columns if x not in cat_cols + target_col + id_col]

df_churn     = df[df["Churn"] == "Yes"]
df_not_churn = df[df["Churn"] == "No"]


        


if (showSummary):


    print ("Rows     : " ,df.shape[0])
    print ("Columns  : " ,df.shape[1])
    print ("\nFeatures : \n" ,df.columns.tolist())
    print ("\nMissing values :  ", df.isnull().sum().values.sum())


    print("\nHead of full data frame:")
    print(df.head())
    print(df.dtypes)



if (showCounts == 1):
    print("")
    print("Counts of categorical values:\n")
    print(df['gender'].value_counts())
    print(df['Partner'].value_counts())
    print(df['Dependents'].value_counts())
    print(df['PhoneService'].value_counts())
    print(df['MultipleLines'].value_counts())
    print(df['InternetService'].value_counts())
    print(df['OnlineSecurity'].value_counts())
    print(df['DeviceProtection'].value_counts())
    print(df['TechSupport'].value_counts())
    print(df['StreamingTV'].value_counts())
    print(df['Contract'].value_counts())
    print(df['PaperlessBilling'].value_counts())
    print(df['PaymentMethod'].value_counts())
    print(df['Churn'].value_counts())

if (showPlots == 1):
    print("Box Plots: \n")
    
    # monthly charges box plot
    df.boxplot(column=['MonthlyCharges'],by=['Churn'],rot = 0,figsize=(5,6))
    plt.ylabel('Monthly Charges [$]')
    plt.xlabel('Churn')
    plt.title('Monthly Charges')
    if (saveFigure == 1):
        plt.savefig('MonthlyCharges.png')       
    plt.show()

    # total charges box plot
    df.boxplot(column=['TotalCharges'],by=['Churn'],rot = 0,figsize=(5,6))
    plt.ylabel('Total Charges [$]')
    plt.xlabel('Churn')
    plt.title('Total Charges')
    if (saveFigure == 1):
        plt.savefig('TotalCharges.png')
    plt.show()


    # tenure box plot
    df.boxplot(column=['tenure'],by=['Churn'],rot = 0,figsize=(5,6))
    plt.ylabel('Tenure [months]')
    plt.xlabel('Churn')
    plt.title('Tenure Box Plot')
    if (saveFigure == 1):
        plt.savefig('Tenure.png')
    plt.show()

    # internet service bar plot
    InternetService_count = df['InternetService'].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(InternetService_count.index, InternetService_count.values, alpha=0.9)
    plt.title('Distribution of InternetService')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Service', fontsize=12)
    if (saveFigure == 1):
        plt.savefig('InternetService.png')
    plt.show()

    # churn bar plot
    Churn_count = df['Churn'].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(Churn_count.index, Churn_count.values, alpha=0.9)
    plt.title('Distribution of Churn')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Churn', fontsize=12)
    if (saveFigure == 1):
        plt.savefig('ChurnDistribution.png')
    plt.show()
    
    # churn pie plot
    lab = df["Churn"].value_counts().keys().tolist()
    val = df["Churn"].value_counts().values.tolist()
    
    plt.pie(val, explode=None, labels=lab, autopct='%1.1f%%', startangle=90)
    plt.title('Customer Churn')
    plt.axis('equal')
    if (saveFigure == 1):
        plt.savefig('ChurnPie.png')
    plt.show()

    

        
        
