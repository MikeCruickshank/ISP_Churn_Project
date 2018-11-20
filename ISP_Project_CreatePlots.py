# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:22:17 2018

@author: Mike
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#Functions --------------------------------------------------------------------
def MakeBarChart(df, column):
    df_churn     = df[df["Churn"] == "Yes"]
    df_no_churn = df[df["Churn"] == "No"]
    fig, ax = plt.subplots()
    
    
    lab0 = df_no_churn[column].value_counts().keys().tolist()
    val0 = df_no_churn[column].value_counts().values.tolist()
    lab1 = df_churn[column].value_counts().keys().tolist()
    val1 = df_churn[column].value_counts().values.tolist()
     
    srt0 = np.argsort(lab0)
    srt1 = np.argsort(lab1)
    n_groups = np.size(srt0)
    
    lab0 = [ lab0[i] for i in srt0]
    val0 = [ val0[i] for i in srt0]
    
    lab1 = [ lab1[i] for i in srt1]
    val1 = [ val1[i] for i in srt1]
    
    m0 = np.arange(2)
    m0[0] = np.max(val0)
    m0[1] = np.max(val1)
    m = np.max(m0)
    
    index = np.arange(n_groups)
    bar_width = 0.30
    opacity = 0.8
   
    rects1 = ax.bar(index, val0, bar_width, 
                    alpha=opacity, color = 'b', label = 'Churn: No')
    rects2 = ax.bar(index + bar_width, val1, bar_width, 
                    alpha=opacity, color = 'g', label = 'Churn: Yes')
       
    plt.ylabel('Number of Customers')
    plt.title(column)
    if (n_groups > 3):
        plt.xticks(index + bar_width, lab0, rotation=60.)
    else:
        plt.xticks(index + bar_width, lab0, rotation=0.)
    plt.axis([-0.5, index[n_groups-1]+1.0, 0, 1.5*m])
    plt.legend(loc='best')
    saveStr = column + '_BarChart.png'
    print(saveStr)
    plt.tight_layout()
    if (saveFigure == 1):
        plt.savefig(saveStr)
    plt.show()   
 
    
def MakeBoxPlot(df, column):
    df.boxplot(column=column,by=['Churn'],rot = 0,figsize=(5,6))
    plt.ylabel(column)
    plt.xlabel('Churn')
    plt.title(column)
    saveStr = column + '_BoxPlot.png'
    print(saveStr)
    plt.tight_layout()
    if (saveFigure == 1):
        plt.savefig(saveStr)      
    plt.show()

def MakeChurnPlots(df):
    
    # churn bar plot
    Churn_count = df['Churn'].value_counts()
    sns.set(style="darkgrid")
    sns.barplot(Churn_count.index, Churn_count.values, alpha=0.9)
    plt.title('Distribution of Churn')
    plt.ylabel('Number of Customers', fontsize=12)
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
    
    
def label_tenure(df) :  
    if df["tenure"] <= 12 :
        return "Tenure: 0-1 Year"
    elif (df["tenure"] > 12) & (df["tenure"] <= 24 ):
        return "Tenure: 1-2 Years"
    elif (df["tenure"] > 24) & (df["tenure"] <= 36) :
        return "Tenure: 2-3 Years"
    elif (df["tenure"] > 36) & (df["tenure"] <= 48) :
        return "Tenure: 3-4 Years"
    elif (df["tenure"] > 48) & (df["tenure"] <= 60) :
        return "Tenure: 4-5 Years"
    elif df["tenure"] > 60 :
        return "Tenure: 5+ Years"

        
def ShowSummary(df):
    print ("Rows     : " ,df.shape[0])
    print ("Columns  : " ,df.shape[1])
    print ("\nFeatures : \n" ,df.columns.tolist())
    print ("\nMissing values :  ", df.isnull().sum().values.sum())


    print("\nHead of full data frame:")
    print(df.head())
    print(df.dtypes)
    

#----------------------------------------------------------------------------- 

saveFigure = True

df=pd.read_csv('ispcustomerchurn.csv')

#==============================================================================
# ShowSummary(df)
#==============================================================================

df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)

print("Objects with Empty Total Charges:\n")
df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = 0
print(df.loc[df['TotalCharges'] == 0])
       
df = df[df["TotalCharges"].notnull()]
df = df.reset_index()[df.columns]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1:"Yes",0:"No"})
df["TenureByGroup"] = df.apply(lambda df:label_tenure(df),axis=1)
      

           
num_cols = ['tenure','MonthlyCharges','TotalCharges']
                        
id_col     = ['customerID']
target_col = ["Churn"]

cat_cols  = [x for x in df.columns if x not in num_cols + target_col + id_col]


df_churn     = df[df["Churn"] == "Yes"]
df_no_churn = df[df["Churn"] == "No"]
   
    
for i in cat_cols :
    MakeBarChart(df,i)
for i in num_cols :
    MakeBoxPlot(df,i)
 
MakeChurnPlots(df)





    

   
        
