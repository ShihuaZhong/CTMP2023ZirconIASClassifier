import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, PowerTransf
''' Python code for predicting the source rocks of detrital zircons using trace elements.

This code can be used to reproduce the results shown in Zhong et al.(2023) published in Contributions to Mineralogy and
Petrology titled "A machine learning method for distinguishing detrital zircon provenance". It contains trained 
algorithms for Support Vector Machine (SVM), Random Forest (RF), and Multilayer Perceptron (MLP). It can also be used 
to predict the source rocks of users' zircons. Please see the article for more details.
'''

""" Data processing
"""
# IMPORTANT: replace the file path in the line below with your file path
# The whole dataset was randomly divided into a training and a test set at a ratio of 8:2 but the procedure was not shown here
# x_train.txt (trace element data) and y_train.txt (label) files here correspond to the training set; x_val.txt (trace element data) and y_val.txt (label) files here correspond to the test set; they were saved after random division to reproduce the results
# x_val.txt and y_val.txt files can also be replaced by users' data to predict the source rocks of their zircons
# All four files can be found in the same folder to this python code
x_train = np.loadtxt('D:/XXXX/x_train.txt')
y_train = np.loadtxt('D:/XXXX/y_train.txt')
x_val1 = np.loadtxt('D:/XXXX/x_val.txt')
y_val = np.loadtxt('D:/XXXX/y_val.txt')

# Applying a natural logarithmic scale for zircon trace element data
x_train = np.log10(x_train)

# Under-sampling (“TomekLinks”) to get a balanced database
xru_train, yru_train = TomekLinks().fit_resample(x_train, y_train)

# Standardization
scaleru = StandardScaler().fit(xru_train)  # method='box-cox'
xru_train = scaleru.transform(xru_train)


""" Machine learning training
"""
# SVM method training and the selected hyperparameter
svm = SVC(kernel='rbf',
          C=2,
          gamma=1,
          cache_size=1000,
          class_weight='balanced',
          probability=True)
svm.fit(xru_train, yru_train)
svm.score(xru_train, yru_train)

# RF method training and the selected hyperparameter
rf = RandomForestClassifier(n_estimators=400,
                            min_samples_split=4,
                            max_depth=20,
                            min_samples_leaf=1,
                            max_features=3,
                            random_state=42,
                            class_weight='balanced',
                            oob_score=True)
rf.fit(xru_train, yru_train)
rf.score(xru_train, yru_train)

# MLP method training and the selected hyperparameter
mlp = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                     random_state=42,
                     max_iter=500,
                     solver='adam',
                     activation='tanh')
mlp.fit(xru_train, yru_train)
mlp.score(xru_train, yru_train)


"""Prediction result
"""
x_val = scaleru.transform(x_val1)
yru_val_pred_svm = svm.predict(x_val)
yru_val_pred_rf = rf.predict(x_val)
yru_val_pred_mlp = mlp.predict(x_val)
print(classification_report(y_val, yru_val_pred_svm))
print(classification_report(y_val, yru_val_pred_rf))
print(classification_report(y_val, yru_val_pred_mlp))
