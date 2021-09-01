import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = pd.read_csv('Satellite_digital_data.csv')

data.head() #print output

print(data.shape) #print output

data.describe() #print output

#printing the numbers of each columns (features), it counts number of distinct elements in specified axis 
for cols in data.columns:
    print(cols,':',data[cols].nunique())

# delete columns which don't have info for this classification (because it is a simple model)
data.drop(['objid', 'rerun', 'specobjid', 'plate'], axis=1, inplace=True)

data.corr() #print output

# delete columns which don't have info for this classification (they are correlated with the telescope features and the time when the pictures were taken)
data.drop(['g', 'r', 'i', 'mjd'], axis=1, inplace=True)

# Step 1. 'run' column identifies specific scan (there are 23 different scans), value_counts --> left number is a unique value, right number is how many are there
print(data['run'].value_counts()) #output

# Step 2. 'camcol' column identifies scanline with 'run' (there are 6 camcol values)
print(data['camcol'].value_counts()) #print output

#simplify from 6 digit number to 0 and 1 
#get_dummies --> used to convert categorical variable into dummy/indicator variables 
dummy_df = pd.get_dummies(data['camcol'])
new_data = pd.concat([data, dummy_df], axis=1)
new_data.drop('camcol', axis=1, inplace=True)
new_data.head() #print output

# Analyse class column
print(data['class'].value_counts(normalize=True)*100) #print output

# Create subset of data, QSO are eliminated (their percentage is small: 8.50%) so that the binary classification can be developed
data1 = new_data[new_data['class']!='QSO']
# Analyse the mean values of the features of'class' 
data1.groupby('class').mean() #print output

# create train and data sets
from sklearn.model_selection import train_test_split
X = data1.drop('class', axis=1)
y = data1['class']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=1551)
X_train.head() #print output

# preprocessing --> the package thar provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators
# MinMaxScaler --> transforms features by scaling each feature to a given range
# fit --> compute the minimum and maximum to be used for later scaling
# transform --> scale features according to feature_range
# column_stack --> used to stack 1D arrays as columns into a 2D array
from sklearn.preprocessing import MinMaxScaler
train_scale = MinMaxScaler().fit(X_train[['ra', 'dec', 'u', 'z', 'run', 'field', 'redshift', 'fiberid']])
train_trans_data = train_scale.transform(X_train[['ra', 'dec', 'u', 'z', 'run', 'field', 'redshift', 'fiberid']])
X_train = np.column_stack([train_trans_data,X_train[[1,2,3,4,5,6]]])

# scale the test set as well
test_trans_data = train_scale.transform(X_test[['ra', 'dec', 'u', 'z', 'run', 'field', 'redshift', 'fiberid']])
X_test = np.column_stack([test_trans_data,X_test[[1,2,3,4,5,6]]])

# LabelEncoder --> fit and transform are used to normilize labels
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# shape --> the elements of the shape tuple give the lengths of the corresponding array dimensions.
X = np.column_stack(([1]*X_train.shape[0], X_train)) # add column of 1s as bias 
m,n = X.shape # get rows and columns
y = y_train
theta = [1]*n # initiate array of coefficents
iterations = 1800
alpha = 0.5 # alpha value to update theta
cost = []
t_temp = []


# Sigmoid function
def sigmoid(val):
    y = 1/(1 + np.exp(-val)) # math.exp() calculates e to the power of -(theta * X)
    return y


# matmul --> matrix product of two arrays/matrices
for i in range(0,iterations):
    val = np.matmul(X,theta) # calculate linear regression 
    y_prob = sigmoid(val) # calculate probability 
    
    # append --> adds an element to the end of the list
    J = -1/m * sum((y*np.log(y_prob)) + (1- y) * np.log(1-y_prob))
    cost.append(J) # append cost function values to list
    
    # Update theta values as a gradient of the cost function
    for t in range(0,n):
        temp = round(theta[t] - alpha/m * sum((y_prob - y)*X[:,t]),4)
        t_temp.append(temp)
    
    theta = []
    theta = t_temp
    t_temp = []

print('The final Theta values are: ',theta)

plt.figure(figsize=(10,8))
plt.title('Cost Function Slope')
plt.plot(cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Error Values')
plt.show()



### TESTING ###

# Prepare the data
X_test1 = np.column_stack(([1]*X_test.shape[0],X_test)) # add bias column to X_test

# Predict values 
y_vals = np.matmul(X_test1,theta) # Multiply theta with X_test
y_prob = sigmoid(y_vals) # calculate probabilities
y_pred = [1 if y>0.5 else 0 for y in y_prob] # Convert probabilities to classes with 0.5 decision boundary

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
accuracy = accuracy_score(y_test, y_pred,normalize=True)
conf_mat = confusion_matrix(y_test,y_pred)
print("The accuracy of the model is :", round(accuracy,3)*100,"%")
print("Confusion Matrix:\n",conf_mat)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
print('Precision = ',round(precision,4),'\nRecall = ', round(recall,4), '\nF-Score = ',round(fscore,4))

# ROC = Receiver operating characteristic
# ROC AUC = Area Under the ROC Curve from prediction scores.
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y_test,  y_prob)  #fpr = false positive rate, tpr = true posititve rate
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label="data, auc="+str(round(auc,4)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Model from Scratch")
plt.legend(loc=4)
plt.show()

####################################
###########SIMPLIFIED###############
####################################

from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss='log', max_iter=100) # create a classifier using logistic regression using SGD
model = classifier.fit(X_train, y_train) # fit the model on the training data
y_pred = model.predict(X_test) # predict values on test set

# Determine model accuracy and goodness of fit
accuracy = accuracy_score(np.array(y_test),np.array(y_pred),normalize=True)
conf_mat = confusion_matrix(y_test,y_pred)
print("The accuracy of the model is :", round(accuracy,2)*100,"%")
print("Confusion Matrix:\n", conf_mat)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
print('Precision = ',round(precision,4),'\nRecall = ', round(recall,4), '\nF-Score = ',round(fscore,4))

from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = classifier.predict_proba(X_test)[::,1] # get probabilities of the model prediction
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(10,8))
plt.plot(fpr,tpr,label="data, auc="+str(round(auc,4)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for scikit-learn Model")
plt.legend(loc=4)
plt.show()













