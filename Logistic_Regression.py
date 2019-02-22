
# coding: utf-8

# # Logistic Regression

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score


# # Import and select the data

# In[7]:




data = pd.read_csv('wdbc.data.txt', header=None)
X = data.iloc[:, 2:]
Y = data[1]


# # Split the data

# In[8]:




train_set_X, test_set_X = train_test_split(X, test_size=0.2, random_state=12)
train_set_Y, test_set_Y = train_test_split(Y, test_size=0.2, random_state=12)


# # Normalize the data

# In[9]:




scaler = preprocessing.MinMaxScaler()
scaler.fit(train_set_X)
train_set_X = pd.DataFrame(scaler.transform(train_set_X))

scaler.fit(test_set_X)
test_set_X = pd.DataFrame(scaler.transform(test_set_X))


# # Because the purpose of the model is to detect cancer, our goal is to select a model with high recall and high accuracy, but recall has the highest priority.
# # We will run cross validation with both L1 and L2 penalties, to test our hyperparameters

# In[3]:




test_C = [0.01, .01, 1, 10, 100]


# # Model with L1 penalty

# In[4]:




for c in test_C:
    clf = linear_model.LogisticRegression(C=c,multi_class='ovr',solver ='liblinear',penalty='l1')
    clf = clf.fit(train_set_X, train_set_Y)

# Print confusion matrix and recall 
    print ("When hyperparameter equals %0.2f, the confusion matrix is as following:" % (c))
    print (confusion_matrix(test_set_Y, clf.predict(test_set_X.iloc[:, :])))
    print ("Recall for malignant cancer: %0.4f"
           % (recall_score(test_set_Y, clf.predict(test_set_X.iloc[:, :]), pos_label='M')))
    print ("Recall for benign cancer: %0.4f"
           % (recall_score(test_set_Y, clf.predict(test_set_X.iloc[:, :]), pos_label='B')))
    
    scores = cross_val_score(clf, train_set_X, train_set_Y, cv=10)
# The mean score and the 95% confidence interval of the score estimate are hence given by:
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print ("\n")


# # Model with L2 penalty

# In[53]:




for c in test_C:
    clf = linear_model.LogisticRegression(C=c,multi_class='multinomial',solver ='newton-cg')
    clf = clf.fit(train_set_X, train_set_Y)

# Print confusion matrix and recall 
    print ("When hyperparameter equals %0.2f, the confusion matrix is as following:" % (c))
    print (confusion_matrix(test_set_Y, clf.predict(test_set_X.iloc[:, :])))
    print ("Recall for malignant cancer: %0.4f"
           % (recall_score(test_set_Y, clf.predict(test_set_X.iloc[:, :]), pos_label='M')))
    print ("Recall for benign cancer: %0.4f"
           % (recall_score(test_set_Y, clf.predict(test_set_X.iloc[:, :]), pos_label='B')))
    
    scores = cross_val_score(clf, train_set_X, train_set_Y, cv=10)
# The mean score and the 95% confidence interval of the score estimate are hence given by:
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print ("\n")


# # Under L1 penalty, the best model is:

# In[ ]:



'''
When hyperparameter equals 1.00, the confusion matrix is as following:
[[58  8]
 [ 1 47]]
Recall for malignant cancer: 0.9792
Recall for benign cancer: 0.8788
Accuracy: 0.97 (+/- 0.04)
'''

# Under L2 penalty, the best model is:
'''
When hyperparameter equals 5.00, the confusion matrix is as following:
[[62  4]
 [ 1 47]]
Recall for malignant cancer: 0.9792
Recall for benign cancer: 0.9394
Accuracy: 0.98 (+/- 0.04)
'''



# # Comparing two models they both of the same recall for malignat cancer, the second one has higher recall for benign cancer. So the model under L2 penalty is the final model we choose. 
