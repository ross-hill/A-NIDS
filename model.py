#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    This file contains a very general model that should work for any data
    The data that was used in the study was altered in data.py
    
    Use of this code applies under an Apache 2.0 licence.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import Autoencoder

'''
    Constants can be set at the beginning of each time the program is run. 
    Comment out/remove the input parts if you want to assign them in the code
'''

rand_seed = input("Enter random seed: ")
hidden_nodes = int(input("Enter hidden nodes: "))
n_blocks = int(input("Enter n_blocks:"))
n_features = int(input("Enter n_features:"))
n_epochs = int(input("Enter n_epochs:"))
RAND_SEED =  rand_seed if isinstance(rand_seed, int) else 0
data_location = input("Enter the location of the Data:")
kdd_data_location = input("Enter the location of the KDD Data:")
fdata = pd.read_csv(data_location)

# If you would not like the iForest to remove the anomalies from the training data 
# after fitting the model, change remove_anomalies to False (lines 117-120)
remove_anomalies = True

''' 
    Begin modelling section
    In this section, fdata_fit should already be created.
    If the fdata loaded in previously is the data for modelling, fdata_fit will be set 
    automatically
'''

if fdata_fit is None:
    fdata_fit = fdata

yfake = np.zeros(fdata_fit.shape[0])
Xtrain, Xtest, ytrain, ytest = train_test_split(fdata_fit, yfake, test_size = 0.3, random_state = RAND_SEED)

print("Xtrain, Xtest shapes: %s %s" % (Xtrain.shape, Xtest.shape))
'''
    ISOLATION FOREST
    SK-learn package: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    Original Publication: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
'''
iForest = ensemble.IsolationForest(bootstrap = True,random_state = RAND_SEED)
print("Fitting iForest")
iForest.fit(Xtrain)
print("Predicting iForest for Xtrain")
Xtrainpred = iForest.predict(Xtrain)

if remove_anomalies:
    print("Outliers removed from Xtrain")
    Xtrain_anom = Xtrain[Xtrainpred == -1]
    Xtrain = Xtrain[Xtrainpred == 1]

print("Predicting iForest for Xtest")
iFpred = iForest.predict(Xtest)

'''
    RANDOM FOREST
    Sk-learn package: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    Website: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    Original publication: https://link.springer.com/article/10.1023/A:1010933404324
    Unsupervised Random Forest: https://labs.genetics.ucla.edu/horvath/RFclustering/RFclustering/RFclusterTutorialTheory.PDF
'''

# Synthetic_data creates a synthetic dataset by sampling from within the columns and combining

def synthetic_data(df, frac = 1.0, replace = True):
    columns = list(df)
    new_data = pd.DataFrame()
    for col in columns:
        sam = np.array(df[col].sample(frac = frac, replace = True, random_state = RAND_SEED))
        new_data[col] = sam
    return new_data

'''
    Note that the fraction of synthetic data can be altered. In this study, the fraction
    was chosen to be 7 times the ratio of anomalies to normal data classified by the 
    iForest.
    Changing n_frac will change this multiple
'''

nanom_prop = np.sum(Xtrainpred == -1) / np.sum(Xtrainpred == 1)
n_frac = 7

sXtrain = synthetic_data(Xtrain, frac = nanom_prop*n_frac)
rfTrain = Xtrain.append(sXtrain)
y_rf = list(np.zeros(len(Xtrain))) + list(np.ones(len(sXtrain)))

clf_rf = ensemble.RandomForestClassifier(n_jobs = -1, random_state = RAND_SEED, n_estimators=250, oob_score = True)
print("Fitting Random Forest on synthetic + real data")
clf_rf.fit(rfTrain, y_rf)
print("Predicting random forest on Xtest")
rf_pred = clf_rf.predict(Xtest)

test_rf_pred = clf_rf.predict(Xtrain)

print("Number of anomalies from Random Forest: %s / %s" % (sum(rf_pred), len(rf_pred)))

# Measure the importances

importances = pd.DataFrame({"FeatureName": list(Xtrain), "Importance": clf_rf.feature_importances_})

'''
    AUTOENCODER NEURAL NETWORK
    Keras Neural Networks package: https://keras.io/
    Original motivation: http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
'''

# Data is normalised

Xdata = [Xtrain, Xtest]

for (tfdata, j) in zip([Xtrain, Xtest], [0,1]):
    for i in range(len(list(fdata_fit))):
        minmax = preprocessing.MinMaxScaler()
        dt = np.array(tfdata.iloc[:,i]).reshape(-1, 1) 
        minmax.fit(dt)
        tfdata.iloc[:,i] = minmax.transform(dt).reshape(-1)
    Xdata[j] = tfdata
    
Xtrain_norm, Xtest_norm = Xdata

# In order to ensure categorical estimates result in 0,1 not 0.78 etc, we define
# a list that has the name of numerical variables, there other variables are assumed
# to be categorical

kdd_numerical_vars = [
        'duration','src_bytes','dst_bytes', 'hot', 'num_failed_logins', 'num_compromised',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
        'num_outbound_cmds','count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]

fdata_numerical_vars = [
        "dst_port", "duration","recv_bytes", "count", "sent_bytes", "sent_pkts", 
        "src_port", "transmittedkbits"
        ]

''' 
 The autoencoder used is a custom class EnsemblePartialAutoencoder
'''

ae = Autoencoder.EnsemblePartialAutoencoders(n_features = 200, n_blocks = 5, hidden_nodes = hidden_nodes, middle_nodes = 5)
ae.create_network(X_shape = Xtrain_norm.shape)
ae.train_network(Xtrain_norm)
ae.predict(Xtest_norm, fdata_numerical_vars)
pred = ae.predictions

pred_means = np.zeros(pred.shape[0])

for i in range(pred.shape[0]):
    emean = np.mean(pred[i])
    pred_means[i] = emean

''' Store all predictions in a dataframe for the test data'''
pred_all = pd.DataFrame({
        "if" : iFpred,
        "rf" : rf_pred,
        "aenn" : pred_means
        })
  
'''
 Calculate Cutoff from rForest and iForest
'''
i1 = iFpred == -1; 
i2 = rf_pred == 1;
i3 = iFpred == 1; 
i4 = rf_pred == 0;    
naa = np.logical_not(i1) & np.logical_not(i2)
c1 = pred_means[naa]
c1_sd = np.std(c1)
u90 = np.mean(c1) + (1.645 * c1_sd)
cutoff = u90

'''
 Evaluate final classification and plot histogram of errors with cutoff
'''

ix1 = pred_means > cutoff
anomaly_bool = (i1 & i2 & ix1) | (i1 & i2) | (i1 & ix1) | (i2 & ix1)
pred_a = pred_means[anomaly_bool]
pred_n = pred_means[np.logical_not(anomaly_bool)]
plt.figure(figsize = (8,6))
plt.hist((pred_a), histtype='step', label = "Anormaly", bins = 100, normed=True)
plt.hist((pred_n), histtype='step', label = "Normal", bins = 100, normed=True)
plt.plot([cutoff,cutoff], [0, 5], "red")
plt.legend()
plt.xlabel("Average Euclidean distance")
plt.ylabel("Frequency")


''' 
    BEGIN KDD SECTION 
    This section includes code getting the data and column names for the KDD data.
    These files can be obtained from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
    The names.csv file is a csv version of kddcup.names
    The kdata.csv file is a csv version of kddcup.data.gz
'''

def get_kdd_names(url):
    names_file = open(url)
    names_lines = names_file.readlines()
    names_file.close()
    names = []
    for i in range(1, len(names_lines)):
        name = names_lines[i]
        name = name[:name.find(":")]
        names.append(name)
    names.append("record_type")
    return names

kdata_location = kdd_data_location + "kdata.csv"
knames_location = kdd_data_location + "names.csv"
kdd_data = pd.read_csv(kdata_location, header=None)
kdd_names = get_kdd_names(knames_location)
kdd_data.columns = kdd_names

''' 
    Data Filtering involves removing y-label from data and making dummy variables
'''

kdd_y = kdd_data["record_type"]
kdd_data = kdd_data.drop(["record_type"], axis = 1)
kdd_X = pd.get_dummies(kdd_data, drop_first=True)

# Split into training and test data

ktrain, ktest, kytrain, kytest = train_test_split(kdd_X, kdd_y, test_size = 0.3, random_state = RAND_SEED)

# Model random forest on the kdd data

krf = ensemble.RandomForestClassifier(n_jobs = -1, random_state = RAND_SEED, n_estimators=250, oob_score = True)
krf.fit(ktrain, kytrain)

# Measure variable importances from random forest

kimp = pd.DataFrame({"FeatureName": list(ktrain), "Importance": krf.feature_importances_})


''' END KDD SECTION '''
