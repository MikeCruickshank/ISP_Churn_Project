# -*- coding: utf-8 -*-
"""

"""
# =============================================================================
# import tensorflow as tf 
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# =============================================================================
# import seaborn as sns
# =============================================================================
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as smf
from sklearn.metrics import precision_score,recall_score

from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz




def Kmeans(X,Y,train_size = 0.8):
    from sklearn.cluster import KMeans
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


def LogisticRegressionModel(X_train, X_test, Y_train, Y_test, cols, penalty = 'l2'):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, penalty = penalty)
    
    Y_train = np.ravel(Y_train)
    clf.fit(X_train, Y_train);
      
    Y_pred_train =clf.predict(X_train)
    Y_pred_test =clf.predict(X_test)
    
    print (clf)

    measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train)
    
def TrainSVM(X_train, X_test, Y_train, Y_test, cols):
    from sklearn import svm
    
    Y_train  = np.ravel(Y_train)
    clf = svm.SVC(gamma=0.001, C=1.0, kernel='linear')
    clf.fit(X_train, Y_train)  
    
    Y_pred_train =clf.predict(X_train)
    Y_pred_test =clf.predict(X_test)
    print("\nSVM:")
    measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train)
    
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
    from sklearn import tree
    Y_train = np.ravel(Y_train)
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=1)
    clf = clf.fit(X_train, Y_train)

    tree.export_graphviz(clf,out_file=filename,max_depth = 3, feature_names = feature_names,
                filled = True, class_names = target_names,label = 'all', rounded= True,
                proportion = True, rotate = True)


    Y_pred_train =clf.predict(X_train)
    Y_pred_test =clf.predict(X_test)
    
    print (clf)

    measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train)
    
    return clf
    
def RandomForest(X_train, X_test, Y_train, Y_test, max_depth = 8, min_samples_split = 5, n_estimators = 1000):
    from sklearn.ensemble import RandomForestClassifier
    Y_train = np.ravel(Y_train)
    rf = RandomForestClassifier(n_estimators =  n_estimators, random_state = 42, max_depth = max_depth, min_samples_split = min_samples_split)


    rf.fit(X_train, Y_train);
        

    Y_pred_train =rf.predict(X_train)
    Y_pred_test =rf.predict(X_test)
    print (rf)
    measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train)
               

    
def measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train):

    print("\n\nTraining Set:")
    conf_matrix = confusion_matrix(Y_train,Y_pred_train)
    print(conf_matrix) 
    print ("Accuracy Score : ",metrics.accuracy_score(Y_train,Y_pred_train))
    print ("\n Classification report : \n",metrics.classification_report(Y_train,Y_pred_train))
     
    print("\n\nTest Set:")
    conf_matrix = confusion_matrix(Y_test,Y_pred_test)
    print(conf_matrix)
    print ("Accuracy Score : ",metrics.accuracy_score(Y_test,Y_pred_test))
    print ("\n Classification report : \n",metrics.classification_report(Y_test,Y_pred_test))

    
def TrainKerasNeuralNetwork(X_train, X_test, Y_train, Y_test,epochs = 1000, batch_size = 32):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    
    print("\nNN Number of Attributes: ", np.size(X_train,1))
    # fix random seed for reproducibility
    np.random.seed(7)
    
    Y_train = np.ravel(Y_train)

    # create model
    model = Sequential()
    model.add(Dense(np.size(X_train,1), input_dim=np.size(X_train,1), activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.0))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=_loss_tensor, optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=epochs, batch_size= batch_size)

    Y_pred_train = model.predict_classes(X_train)
    Y_pred_test = model.predict_classes(X_test)
    
    measure_performance(Y_test,Y_pred_test, Y_train, Y_pred_train)

def _loss_tensor(y_true, y_pred):
    from keras import backend as K
    _EPSILON = K.epsilon()
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(1*y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(out, axis=-1)

    
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
    
# Main    
#==============================================================================
df=pd.read_csv('ispcustomerchurn.csv')
df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)

df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = 0
       
df = df[df["TotalCharges"].notnull()]
df = df.reset_index()[df.columns]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1:"Yes",0:"No"})
# =============================================================================
# df["TenureByGroup"] = df.apply(lambda df:label_tenure(df),axis=1)
# =============================================================================
      
      
num_cols = ['tenure','MonthlyCharges','TotalCharges']
                        
id_col     = ['customerID']
target_col = ["Churn"]

cat_cols  = [x for x in df.columns if x not in num_cols + target_col + id_col]


        
binary_cols = [x for x in df.columns if df[x].nunique() == 2 and x not in target_col]
multiple_cols = [x for x in cat_cols if df[x].nunique() > 2 and x not in target_col]

noInternet_cols = ['OnlineSecurity_No internet service','DeviceProtection_No internet service','TechSupport_No internet service',
                     'StreamingTV_No internet service','StreamingMovies_No internet service','OnlineBackup_No internet service']
 
#Label encoding Binary columns
le = LabelEncoder()
for i in binary_cols:
    df[i] = le.fit_transform(df[i])
for i in target_col:
    df[i] = le.fit_transform(df[i])
    
#Duplicating columns for multi value columns
df = pd.get_dummies(data = df ,columns = multiple_cols)

#Eliminating redundant columns
if 1:
    cols    = [i for i in df.columns if i not in noInternet_cols]
    df = df[cols]


#df_overyear = df.copy()
df_overyear_temp = df[df["tenure"] > 12]
df_overyear = df_overyear_temp.reset_index() 
cols    = [i for i in df_overyear.columns if i not in 'index']
df_overyear = df_overyear[cols]

#==============================================================================
# df['MonCharXTenure'] = df.MonthlyCharges*df.tenure
# print(df.head())
#==============================================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)                 

df_copy = df.copy()
df = df.drop(labels=num_cols, axis=1)
df = df.merge(scaled,left_index=True,right_index=True,how = "left")

scaler2 = StandardScaler()
scaled2 = scaler2.fit_transform(df_overyear[num_cols])
scaled2 = pd.DataFrame(scaled2,columns=num_cols)
                 

df_overyear_copy = df_overyear.copy()
df_overyear = df_overyear.drop(labels=num_cols, axis=1)
df_overyear = df_overyear.merge(scaled2,left_index=True,right_index=True,how = "left")

#correlation
correlation = df.corr()
#tick labels
matrix_cols = df.columns.tolist()
correlation_byChurn = correlation.Churn
#convert to array
corr_array  = np.array(correlation)

myList = correlation.columns


srt0 = np.argsort(correlation_byChurn)

correlation_byChurnSorted = [ correlation_byChurn[i] for i in srt0]
myListSorted = [myList[i] for i in srt0]

correlation_byChurnSorted = correlation_byChurnSorted[0:np.size(correlation_byChurnSorted)-1]
myListSorted = myListSorted[0:np.size(myListSorted)-1]
# print(correlation.Churn)

#==============================================================================
# =============================================================================
# fig, ax = plt.subplots()
# index = np.arange(np.size(correlation_byChurnSorted))
# bar_width = 0.30
# opacity = 0.8
#     
# rects1 = ax.bar(index, correlation_byChurnSorted, bar_width, 
#                  alpha=opacity, color = 'b')
# plt.ylabel('Correlation to Churn')
# plt.yticks(rotation = 90.)
# plt.xticks(index + bar_width, myListSorted, rotation=90.)
# saveStr = 'Correlation_BarChart.png'
# plt.axis([-0.5, np.size(correlation_byChurnSorted), -1.0, 1.0])
# fig = plt.gcf()
# fig.set_size_inches(11,8)
# plt.tight_layout()
# plt.savefig(saveStr)
# plt.show()   
# =============================================================================
#==============================================================================

### Models
df = df_overyear

count_class_0, count_class_1 = df.Churn.value_counts()


train,test = train_test_split(df,test_size = .25 ,random_state = 111)


cols    = [i for i in df.columns if i not in id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]  



# Logistic Regression
if 0:
    LogisticRegressionModel(train_X, test_X, train_Y, test_Y, cols, penalty = 'l2')

# Decision Tree 
if 0:
    clf = TrainDecisionTree(X_train = train_X, X_test = test_X, Y_train = train_Y,
                             Y_test = test_Y, max_depth = 1,  min_samples_split = 20,
                            feature_names = cols, target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')

# Random Forest 
if 0:
    RandomForest(train_X, test_X, train_Y, test_Y,
                 max_depth = 8, min_samples_split = 20,  n_estimators = 1000)

# SVM
if 0: 
    TrainSVM(train_X, test_X, train_Y, test_Y, cols)

# Neural Network (Keras)
if 0:
    TrainKerasNeuralNetwork(train_X, test_X, train_Y, test_Y, epochs = 35, batch_size = 1024)

# Undersampling/Oversampling
if 0:
    train,test = train_test_split(df,test_size = .25 ,random_state = 111)

    cols    = [i for i in df.columns if i not in id_col + target_col]

    test_X  = test[cols]
    test_Y  = test[target_col]  


    count_no_churn, count_churn = df.Churn.value_counts()
    df_churn     = df[df["Churn"] == 1]
    df_no_churn = df[df["Churn"] == 0]
    
    df_no_churn_reduced = df_no_churn.sample(count_churn)
    df_reduced = pd.concat([df_no_churn_reduced, df_churn], axis=0)
    
    print('Random under-sampling:')
    print(df_reduced.Churn.value_counts())
    
    train,test = train_test_split(df_reduced,test_size = .25 ,random_state = 111)
    
    train_X = train[cols]
    train_Y = train[target_col]

    
    #LogisticRegressionModel(train_X, test_X, train_Y, test_Y, cols, penalty = 'l2')
    #clf = TrainDecisionTree(train_X, test_X, train_Y, test_Y, max_depth = 5,  min_samples_split = 10,
    #                        feature_names = cols, target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')
    #RandomForest(train_X, test_X, train_Y, test_Y, max_depth = 5, min_samples_split = 5,  n_estimators = 1000)
    TrainSVM(train_X, test_X, train_Y, test_Y, cols)
    
    #Oversampling
    df_churn_over = df_churn.sample(count_no_churn, replace=True)
    df_over = pd.concat([df_no_churn, df_churn_over], axis=0)
    
    print('Random over-sampling:')
    print(df_over.Churn.value_counts())
    
    train,test = train_test_split(df_over,test_size = .25 ,random_state = 111)
    
    train_X = train[cols]
    train_Y = train[target_col]
    
    #TrainSVM(train_X, test_X, train_Y, test_Y, cols)
    #LogisticRegressionModel(train_X, test_X, train_Y, test_Y, cols, penalty = 'l2')
    #clf = TrainDecisionTree(train_X, test_X, train_Y, test_Y, max_depth = 5,  min_samples_split = 10,
    #                        feature_names = cols, target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')
    #RandomForest(train_X, test_X, train_Y, test_Y, 
    #                 max_depth = 6, min_samples_split = 5,  n_estimators = 1000)
    #clf = TrainDecisionTree(X_train = train_X, X_test = test_X, Y_train = train_Y,
    #                      Y_test = test_Y, max_depth = 5,  min_samples_split = 2,
    #                     feature_names = cols, target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')


# =============================================================================
# PCA Visualization
if 0:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    
    X = df[[i for i in df.columns if i not in id_col + target_col]]
    Y = df[target_col]
    
    principal_components = pca.fit_transform(X)
    
    pca_data = pd.DataFrame(principal_components,columns = ["PC1","PC2"])
    pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")
    
    pca_data["Churn"] = pca_data["Churn"].replace({1:"Churn",0:"Not Churn"})
    plt.scatter(x = pca_data[pca_data["Churn"] == 'Churn']["PC1"],
                y = pca_data[pca_data["Churn"] == 'Churn']["PC2"],
                s = 20, c = 'red', marker='o', alpha = 0.5)

    plt.scatter(x = pca_data[pca_data["Churn"] == 'Not Churn']["PC1"],
                y = pca_data[pca_data["Churn"] == 'Not Churn']["PC2"],
                s = 20, c = 'blue', marker='o', alpha = 0.1)
    plt.legend(['Churn: Yes', 'Churn: No'])
    saveStr = 'PCA_Scatter.png'
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.tight_layout()
    plt.savefig(saveStr)
    
# PCA Logistic Regression
if 0:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 4)
    
    X = df[[i for i in df.columns if i not in id_col + target_col]]
    Y = df[target_col]
    
    principal_components = pca.fit_transform(X)
    
    pca_data = pd.DataFrame(principal_components)
    pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")
    
    train,test = train_test_split(pca_data,test_size = .25 ,random_state = 111)
    cols    = [i for i in pca_data.columns if i not in id_col + target_col]
    train_X = train[cols]
    train_Y = train[target_col]
    test_X  = test[cols]
    test_Y  = test[target_col]  
    LogisticRegressionModel(train_X, test_X, train_Y, test_Y, cols, penalty = 'l2')
    
# Recursive Feature Elimination
if 0:
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42, max_depth = 6, min_samples_split = 10)

    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=10, min_samples_leaf=1)
    
    train,test = train_test_split(df,test_size = .25 ,random_state = 111)
    X = df[[i for i in df.columns if i not in id_col + target_col]]
    Y = df[target_col]
    col_names = [i for i in df.columns if i not in id_col + target_col]
    #logit = LogisticRegression()


    rfe = RFE(rf,1)
    rfe = rfe.fit(X,Y.values.ravel())
    
    rfe.support_
    rfe.ranking_
        
    #identified columns Recursive Feature Elimination
    idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,
                           "columns" : [i for i in df.columns if i not in id_col + target_col],
                           "ranking" : rfe.ranking_,
                          })
    cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()

    print("Features sorted by their rank:")
    print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), col_names)))

    #separating train and test data
    train_rf_X = X[cols]
    train_rf_Y = Y
    test_rf_X  = test[cols]
    test_rf_Y  = test[target_col]
    
    print('\nRecursive Feature Elimination:')
    #LogisticRegressionModel(train_rf_X,test_rf_X, train_rf_Y, test_rf_Y, cols, penalty = 'l2')
    
    
    #clf = TrainDecisionTree(train_rf_X,test_rf_X, train_rf_Y, test_rf_Y,
                            max_depth = 5,  min_samples_split = 10,
                            feature_names = cols, target_names = ['Churn: No','Churn: Yes'], filename = 'tree.dot')
    RandomForest(train_rf_X,test_rf_X, train_rf_Y, test_rf_Y, max_depth = 7, min_samples_split = 5,  n_estimators = 1000)

