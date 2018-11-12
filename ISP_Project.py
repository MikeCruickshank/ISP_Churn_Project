# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import os
from sklearn import tree

def ExploreDatset(df, showSummary, showCounts, showPlots):

    if (showSummary):
        print("")
        print("Head of full data frame:")
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
        plt.figure()
        df.boxplot(column=['MonthlyCharges'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Monthly Charges [$]')
        plt.xlabel('Churn')
        plt.title('Monthly Charges')
        
        plt.figure()
        df.boxplot(column=['TotalCharges'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Total Charges [$]')
        plt.xlabel('Churn')
        plt.title('Total Charges')
        
        plt.figure()
        df.boxplot(column=['tenure'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Tenure [months]')
        plt.xlabel('Churn')
        plt.title('Tenure')

        plt.figure()
        InternetService_count = df['InternetService'].value_counts()
        sns.set(style="darkgrid")
        sns.barplot(InternetService_count.index, InternetService_count.values, alpha=0.9)
        plt.title('Frequency Distribution of InternetService')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Service', fontsize=12)
        plt.show()

#==============================================================================

def ReduceDataFrame(df, column_names):
    replace_map = {'gender': {'Female': 0, 'Male': 1},
               'Partner': {'Yes': 1, 'No': 0},
               'Dependents': {'Yes': 1, 'No': 0},
               'PhoneService': {'Yes': 1, 'No': 0},
               'MultipleLines': {'Yes': 1, 'No': 0,'No phone service': 2},
               'InternetService': {'DSL': 1, 'No': 0,'Fiber optic': 2},
               'OnlineSecurity': {'Yes': 1, 'No': 0,'No internet service': 2},
               'OnlineBackup': {'Yes': 1, 'No': 0,'No internet service': 2},
               'DeviceProtection': {'Yes': 1, 'No': 0,'No internet service': 2},
               'TechSupport': {'Yes': 1, 'No': 0,'No internet service': 2},
               'StreamingTV': {'Yes': 1, 'No': 0,'No internet service': 2},
               'StreamingMovies': {'Yes': 1, 'No': 0,'No internet service': 2},
               'Contract': {'One year': 1, 'Month-to-month': 0,'Two year': 2},
               'PaperlessBilling': {'Yes': 1, 'No': 0},
               'PaymentMethod': {'Mailed check': 0, 'Electronic check': 1,'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
               'Churn': {'Yes': 1, 'No': 0},
            }

    df_replace = df.copy()
    df_replace.replace(replace_map, inplace=True)
    
    for c_it in column_names:
        df_replace[c_it] =df_replace[c_it].astype('category')
   
    df_replace['Churn'] =df_replace['Churn'].astype('int64')

    df_replace['MonthlyCharges'] =(df_replace['MonthlyCharges'])/(np.max(df_replace['MonthlyCharges'].values))
    df_replace['TotalCharges'] =(df_replace['TotalCharges'])/(np.max(df_replace['TotalCharges'].values))


    return df_replace
  
#==============================================================================
def  OneHotDataFrame(df, column_names):

    df_onehot = df.copy()
    for c_it in column_names:
        df_onehot = pd.get_dummies(df_onehot, columns=[c_it], prefix = [c_it])
   
    df_onehot['MonthlyCharges'] =(df_onehot['MonthlyCharges'])/(np.max(df_onehot['MonthlyCharges'].values))
    df_onehot['TotalCharges'] =(df_onehot['TotalCharges'])/(np.max(df_onehot['TotalCharges'].values))

    
    return df_onehot  
#==============================================================================


def TrainNeuralNetwork(df_onehot):
    start_time = time.time()
    tf.reset_default_graph()
    
    ## Set Parameters
    MODEL_NAME = 'NeuralNet_SaveVersion'
    SAVE_PATH = './neuralnetwork_save'
    
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    n_epochs = 40000
    display_step = 5
    learning_rate = 0.005
   
    n_attributes = 45
    n_total_attributes = 48
    X= df_onehot.iloc[:,1:n_attributes+1].values
    Y= df_onehot.iloc[:,n_attributes+2:n_total_attributes+1].values

    print("\nTarget Attributes:")
    print("  ", df_onehot.columns[n_attributes+2:n_total_attributes+1])

    n_inputs = np.size(Y,0)
    print("\nNumber of Objects: ", n_inputs)
    
    X_train, X_test, Y_train, Y_test = SplitData(X,Y,train_size = 0.8)

    
    # Model input and output
    x = tf.placeholder(tf.float32, [None, n_attributes], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    
    net = tf.nn.dropout(tf.layers.dense(x, 10, activation=tf.nn.relu),1.0)
    prediction = tf.layers.dense(net, 1, activation=tf.nn.sigmoid, name='prediction')
    
    
    cost = tf.reduce_mean((-y*tf.log(prediction) - (1-y)*tf.log(1-prediction)), name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#     init_op = tf.initialize_all_variables()
#     checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
    
    ## Run Training Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
    
        for epoch in range(n_epochs):
            _, c = sess.run([optimizer, cost], 
                            feed_dict={x: X_train, y: Y_train})
            if epoch % display_step == 0:
                print("Epoch:", '%d' % (epoch), ", cost=", \
                    "{:.9f}".format(c))
        path = saver.save(sess, SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
        print("saved at {}".format(path))
        print("Optimization Finished!")      
        
    
        ## Determine Peformance of Network
        Y_pred =  np.rint(sess.run(prediction, feed_dict={x: X_test,  y: Y_test})) 
    
        Y_error = Y_pred - Y_test
        error_rate = 100*(np.sum(np.abs(Y_error[:,0]))/np.size(Y_test,0))
        print("")
        print("Error Rate: ", error_rate, "%")
        print("Time to Train: ", np.rint((time.time() - start_time)/60), " minutes")
        
        Y_pred_all =  np.rint(sess.run(prediction, feed_dict={x: X,  y: Y}))
    
        Y_error_all = Y_pred_all - Y
        error_rate_all = 100*(np.sum(np.abs(Y_error_all[:,0]))/np.size(Y,0))
        print("")
        print("Error Rate All: ", error_rate_all, "%")
        
        
        print("Number of predicted yes's: ", np.sum(Y_pred_all[:,0]))
        print("Number of predicted no's: ",  np.sum(1-Y_pred_all[:,0]), "\n")
        print("Number of yes's: ", np.sum(Y[:,0]))
        print("Number of no's: ",  np.sum(1-Y[:,0]), "\n")
                
               
        
def AnalyzeError(Y_train, Y_test, Y_pred_train, Y_pred_test):
    error_test = Y_pred_test - Y_test
    error_test_rate = np.sum(np.abs(error_test))/np.size(Y_test,0)
    
    print("\nTest Error Rate: ", 100*error_test_rate, "%")
        
    error_train = Y_pred_train - Y_train
    error_train_rate = np.sum(np.abs(error_train))/np.size(Y_train,0)
    
    print("Training Error Rate: ", 100*error_train_rate, "%")
    
    
def SplitData(X,Y,train_size):
    train_size = train_size
    train_cnt = int((X.shape[0] * train_size))
    np.random.shuffle(X)
    np.random.shuffle(Y)
    X_train = X[0:train_cnt][:]
    Y_train = Y[0:train_cnt][:]
    X_test = X[train_cnt:][:]
    Y_test = Y[train_cnt:][:]
    
    n_training_inputs = np.size(Y_train,0)
    print("Number of Training Objects: ", n_training_inputs)
    Y_train = np.reshape(Y_train, (n_training_inputs,1))
    
    n_testing_inputs = np.size(Y_test,0)
    print("Number of Testing Objects: ", n_testing_inputs)
    Y_test = np.reshape(Y_test,(n_testing_inputs,1))
    
    return X_train, X_test, Y_train, Y_test
    

def TrainDecisionTree(X,Y, max_depth, min_samples_split): 
    
    X_train, X_test, Y_train, Y_test = SplitData(X,Y,train_size = 0.8)
    
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=1)
    clf = clf.fit(X_train, Y_train)
    Y_pred = np.reshape(clf.predict(X_test),(np.size(Y_test,0),1));
    Y_pred_train = np.reshape(clf.predict(X_train),(np.size(Y_train,0),1));
    
    
    AnalyzeError(Y_train = Y_train, Y_test = Y_test, Y_pred_train = Y_pred_train, Y_pred_test = Y_pred)

def AccessData(df,xArray,yIdx):
    X= df.iloc[:,xArray].values
    Y= df.iloc[:,yIdx].values
    return X,Y
#==============================================================================
column_names = ['customerID','gender','SeniorCitizen','Partner','Dependents','tenure',
        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','	Contract',
        'PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'];

column_names_objects = ['gender','Partner','Dependents',
                        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
                        'PaperlessBilling','PaymentMethod','Churn'];

df=pd.read_csv('ispcustomerchurn.csv')

if(df.isnull().values.sum() != 0):
    print("Dataset contains null values")


totalChargesBool = (df['TotalCharges'].values == ' ')
df_drop = df.drop(df.index[totalChargesBool])
df_drop.to_csv('ispcustomerchurn_modified.csv',index=False)
df_drop=pd.read_csv('ispcustomerchurn_modified.csv')


ExploreDatset(df = df_drop, showSummary = 0, showCounts = 0, showPlots = 0)

df_replace = ReduceDataFrame(df_drop, column_names = column_names_objects)
df_onehot = OneHotDataFrame(df_drop, column_names = column_names_objects)

#==============================================================================
# TrainNeuralNetwork(df_onehot)
#==============================================================================


X,Y = AccessData(df = df_replace, xArray = np.arange(1,20), yIdx = 20)
print("With Replace Data: ")
TrainDecisionTree(X = X, Y = Y, max_depth = 8, min_samples_split = 6)

X,Y = AccessData(df = df_onehot, xArray = np.arange(1,47), yIdx = 47)
print("\nWith One-hot Data: ")
TrainDecisionTree(X = X, Y = Y, max_depth = 8, min_samples_split = 10)





