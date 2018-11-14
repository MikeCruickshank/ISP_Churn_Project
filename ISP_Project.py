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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn import tree
from sklearn import metrics

from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz


from sklearn.cluster import KMeans


def Kmeans(X,Y,train_size = 0.8):
    
    X_train, X_test, Y_train, Y_test = SplitData(X,Y,train_size = train_size)

        
    n_clusters = 4
    clf = KMeans(n_clusters = n_clusters, random_state=42, verbose=0)
    clf.fit(X_train)
    y_labels_train = clf.labels_
    y_labels_test = clf.predict(X_test)
    
    y_labels_train = np.reshape(y_labels_train,(np.size(y_labels_train),1))
    y_labels_test = np.reshape(y_labels_test,(np.size(y_labels_test),1))


   
    np.append(X_train,y_labels_train,axis=1)
    np.append(X_test,y_labels_test,axis=1)

    
    return X_train, X_test, Y_train, Y_test

def RandomForest(X_train, X_test, Y_train, Y_test, max_depth = 8):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 5000, random_state = 42, max_depth = max_depth)


    rf.fit(X_train, Y_train);
        
    Y_pred_train = rf.predict(X_train)
    Y_pred_test =rf.predict(X_test)
    
    print("\n\nTraining Set:")
    measure_performance(Y_test=Y_train, Y_pred = np.rint(Y_pred_train), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)

    print("\n\nTest Set:")
    measure_performance(Y_test=Y_test, Y_pred = np.rint(Y_pred_test), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
 
def LogisticRegression(X_train, X_test, Y_train, Y_test, penalty = 'l2'):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, penalty = penalty)

    clf.fit(X_train, Y_train);
        
    Y_pred_train =clf.predict(X_train)
    Y_pred_test =clf.predict(X_test)
    
    print("\n\nTraining Set:")
    measure_performance(Y_test=Y_train, Y_pred = np.rint(Y_pred_train), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)

    print("\n\nTest Set:")
    measure_performance(Y_test=Y_test, Y_pred = np.rint(Y_pred_test), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
 


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
        df.boxplot(column=['MonthlyCharges'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Monthly Charges [$]')
        plt.xlabel('Churn')
        plt.title('Monthly Charges')
        plt.show()
        plt.savefig('MonthlyCharges.png')

        
        df.boxplot(column=['TotalCharges'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Total Charges [$]')
        plt.xlabel('Churn')
        plt.title('Total Charges')
        plt.show()
        plt.savefig('TotalCharges.png')

        
        df.boxplot(column=['tenure'],by=['Churn'],rot = 0,figsize=(5,6))
        plt.ylabel('Tenure [months]')
        plt.xlabel('Churn')
        plt.title('Tenure Box Plot')
        plt.show()
        plt.savefig('Tenure.png')

        InternetService_count = df['InternetService'].value_counts()
        sns.set(style="darkgrid")
        sns.barplot(InternetService_count.index, InternetService_count.values, alpha=0.9)
        plt.title('Distribution of InternetService')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Service', fontsize=12)
        plt.show()
        plt.savefig('InternetService.png')
        
        Churn_count = df['Churn'].value_counts()
        sns.set(style="darkgrid")
        sns.barplot(Churn_count.index, Churn_count.values, alpha=0.9)
        plt.title('Distribution of Churn')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Churn', fontsize=12)
        plt.show()
        plt.savefig('ChurnDistribution.png')
        


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
#==============================================================================
# 
#     df_replace['MonthlyCharges'] =(df_replace['MonthlyCharges'])/(np.max(df_replace['MonthlyCharges'].values))
#     df_replace['TotalCharges'] =(df_replace['TotalCharges'])/(np.max(df_replace['TotalCharges'].values))
#==============================================================================

    return df_replace

def ToCategories(df, column_names):
    df_cat = df.copy()
    for c_it in column_names:
        df_cat[c_it] =df_cat[c_it].astype('category')
    
    return df_cat
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
    
    n_epochs = 30000
    display_step = 1000
    learning_rate = 0.0001
   
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
    
    net = tf.nn.dropout(tf.layers.dense(x, 45, activation=tf.nn.relu),1.0)
    net = tf.nn.dropout(tf.layers.dense(net, 6, activation=tf.nn.relu),1.0)

    prediction = tf.layers.dense(net, 1, activation=tf.nn.sigmoid, name='prediction')
    
    
    cost = tf.reduce_mean((-2*y*tf.log(prediction) - (1-y)*tf.log(1-prediction)), name='cost')
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
        Y_pred_train =  np.rint(sess.run(prediction, feed_dict={x: X_train,  y: Y_train}))

    
        Y_error_all = Y_pred_all - Y
        error_rate_all = 100*(np.sum(np.abs(Y_error_all[:,0]))/np.size(Y,0))
        print("")
        print("Error Rate All: ", error_rate_all, "%")
        
        
        print("Number of predicted yes's: ", np.sum(Y_pred_all[:,0]))
        print("Number of predicted no's: ",  np.sum(1-Y_pred_all[:,0]), "\n")
        print("Number of yes's: ", np.sum(Y[:,0]))
        print("Number of no's: ",  np.sum(1-Y[:,0]), "\n")
                
        print("\n\n\n\nTrain Set\n")
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test,Y_pred)),"\n")

        print ("Classification report")
        print (metrics.classification_report(Y_train,Y_pred_train, target_names = ['Churn: No','Churn: Yes']),"\n")
        
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_train,Y_pred_train),"\n")
        
        print("\n\n\n\nTest Set\n")
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test,Y_pred)),"\n")

        print ("Classification report")
        print (metrics.classification_report(Y_test,Y_pred, target_names = ['Churn: No','Churn: Yes']),"\n")
        
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_test,Y_pred),"\n")
        
       
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
    

def TrainDecisionTree(X_train, X_test, Y_train, Y_test, max_depth, min_samples_split,feature_names, target_names, filename = 'tree.dot'): 
    
    
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=1)
    clf = clf.fit(X_train, Y_train)
    Y_pred = np.reshape(clf.predict(X_test),(np.size(Y_test,0),1));
    Y_pred_train = np.reshape(clf.predict(X_train),(np.size(Y_train,0),1));
    
    
    AnalyzeError(Y_train = Y_train, Y_test = Y_test, Y_pred_train = Y_pred_train, Y_pred_test = Y_pred)

    tree.export_graphviz(clf,out_file=filename,feature_names = feature_names,
                class_names = target_names)
    
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, special_characters=True,
                    rounded = True, proportion = True, label = 'none')

    print("Train Set: \n")
    measure_performance(Y_test = Y_train,Y_pred = Y_pred_train, target_labels = ['Churn: No','Churn: Yes'])
    print("Test Set: \n")
    measure_performance(Y_test = Y_test, Y_pred = Y_pred , target_labels = ['Churn: No','Churn: Yes'])

    return clf
    
def DeleteRowsFromDF(df,n_delete = 1000, targetIdx = 47):
    df_copy = df.copy()
    import random
    deleteCount = 0
    randArray = random.sample(range(1, df.shape[0] - 1), n_delete)
    for i in np.arange(0,n_delete):
        if (randArray[i] <  (df.shape[0] - deleteCount)):
            if (df_copy.iloc[randArray[i]][targetIdx] == 0):
                df_copy.drop([randArray[i]],inplace=True)
                deleteCount = deleteCount + 1
    
    print("Delete Count:", deleteCount)
    print("Number of 0's:", np.sum(1-df_copy.iloc[:,targetIdx]))
    print("Number of 1's:", np.sum(df_copy.iloc[:,targetIdx]))

    return df_copy
            
  
def AccessData(df,xArray,yIdx):
    X= df.iloc[:,xArray].values
    Y= df.iloc[:,yIdx].values
    print("\nTarget Attribute:")
    print("  ", df.columns[yIdx])

    return X,Y
    

def measure_performance(Y_test,Y_pred, target_labels, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):

    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test,Y_pred)),"\n")

    if show_classification_report:
        print ("Classification report")
        print (metrics.classification_report(Y_test,Y_pred, target_names = target_labels),"\n")
        
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_test,Y_pred),"\n")
 
        
def TrainKerasNeuralNetwork(X_train, X_test, Y_train, Y_test,epochs = 1000, batch_size = 32):
    print("\nNN Number of Attributes: ", np.size(X_train,1))
    # fix random seed for reproducibility
    np.random.seed(7)
    
    # create model
    model = Sequential()
    model.add(Dense(np.size(X_train,1), input_dim=np.size(X_train,1), activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=_loss_tensor, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=epochs, batch_size= batch_size)

    Y_pred_train = model.predict(X_train, batch_size=None, steps=None)
    Y_pred_test = model.predict(X_test, batch_size=None, steps=None)
    
    print("\n\nTraining Set:")
    measure_performance(Y_test=Y_train, Y_pred = np.rint(Y_pred_train), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)

    print("\n\nTest Set:")
    measure_performance(Y_test=Y_test, Y_pred = np.rint(Y_pred_test), target_labels =  ['Churn: No','Churn: Yes'], show_accuracy=True, show_classification_report=True, show_confusion_matrix=True)
 

def _loss_tensor(y_true, y_pred):
    from keras import backend as K
    _EPSILON = K.epsilon()
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(1*y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

    
    
# Main    
#==============================================================================
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

df_cat = ToCategories(df_drop, column_names = column_names_objects)
df_replace = ReduceDataFrame(df_drop, column_names = column_names_objects)
df_onehot = OneHotDataFrame(df_drop, column_names = column_names_objects)

column_names = list(df_replace.head(0))
n_columns = np.size(column_names)
#==============================================================================


# Decision Tree
#==============================================================================
X,Y = AccessData(df = df_replace, xArray = np.arange(1,18), yIdx = 20)
X_train, X_test, Y_train, Y_test = SplitData(X,Y,train_size = 0.8)

print("\nWith Replace Data: ")
clf = TrainDecisionTree(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test, max_depth = 14,  min_samples_split = 10,feature_names = column_names[1:n_columns-1], target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')
#==============================================================================
# print("\n\nAdding KMeans:")
# X_train, X_test, Y_train, Y_test = Kmeans(X,Y,train_size = 0.8)
# clf = TrainDecisionTree(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test, max_depth = 5,  min_samples_split = 10,feature_names = column_names[1:n_columns-1], target_names = ['Churn: No','Churn: Yes'],filename = 'treeWithKmeans.dot')
#==============================================================================


# Random Forest
#==============================================================================
print("\nBefore KMeans: ")
RandomForest(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)
print("\n\nAdding KMeans:")
X_train, X_test, Y_train, Y_test = Kmeans(X,Y,train_size = 0.6)
RandomForest(X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test = Y_test)


#==============================================================================
# Neural Network w/ Keras
#==============================================================================
# df_onehot = DeleteRows(df_onehot,n_delete = 3500, targetLabel = ['Churn_Yes'])
# print(np.shape(df_onehot))
# X,Y = AccessData(df = df_onehot, xArray = np.concatenate(np.arange(1,20),np.arange(30,42)), yIdx = 47)
# 
# print("\nWith One-hot Data: ")
# X_train, X_test, Y_train, Y_test = SplitData(X,Y,train_size = 0.7)
# TrainKerasNeuralNetwork(X_train, X_test, Y_train, Y_test,epochs = 100, batch_size = 32)
# 
#==============================================================================
#==============================================================================
# print("\n\nAdding KMeans:")
# X_train, X_test, Y_train, Y_test = Kmeans(X,Y,train_size = 0.7)
# TrainKerasNeuralNetwork(X_train, X_test, Y_train, Y_test,epochs = 2000, batch_size = 64)
# 
#==============================================================================

# Logistic Regression
LogisticRegression(X_train, X_test, Y_train, Y_test, penalty = 'l1')
