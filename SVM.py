
# coding: utf-8

# In[74]:


##############################################################################
#Libraries
##############################################################################

from sklearn import tree
from sklearn.cross_validation import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors, datasets
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import KFold
from scikitplot.metrics import plot_lift_curve


# In[5]:


##############################################################################
###LOAD DATA
##############################################################################

breast_data = pd.read_table("wdbc.data", header = None, sep = ",")


# In[23]:


##############################################################################
### Create Training and Testing sets of Data
##############################################################################
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_data , test_data = split_train_test(breast_data, .2)


# In[24]:


##############################################################################
### Separate prediction value from the rest of the data
##############################################################################
train_y = train_data[1]
train_x = train_data[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]

test_y = test_data[1]
test_x = test_data[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]]


# In[37]:



##############################################################################
### Normalize Data
##############################################################################

minmax = MinMaxScaler()
minmax.fit(train_x)
train_xn = minmax.transform(train_x)
train_xn = pd.DataFrame(train_xn)
minmax.fit(test_x)
test_xn = minmax.transform(test_x)

train_xn.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', 'aa', 'ab', 'ac', 'ad']




# In[38]:


##############################################################################
### Hyper Parameters for SVM
##############################################################################

p_grid = {"C": [1, 10, 100, 1000],
          "gamma": [.0001 ,.001 , .01, .1]}
kernels = ['rbf','linear','poly']


NUM_TRIALS = 10
nested_scores = []

for kernel_type in kernels:
    svm = GridSearchCV(estimator=SVC(kernel=kernel_type), param_grid=p_grid, cv= 10)
    svm.fit(train_xn, train_y)
    nested_score = cross_val_score(svm, X=train_xn, y=train_y, cv= 10)
    print(kernel_type, nested_score.mean())
    print(svm.best_estimator_)



# # It appears that the best model is linear with an accuracy of 97.3%. The hyperparameter of C is at 1 so that may be overfitting. Lets go with Poly because it has a similar accuracy of 97.1% and the C is 100!

# ### How does it preform on the actual data?

# In[48]:


clf = SVC(C = 100, kernel = "poly", gamma = .1, probability = True)
clf = clf.fit(train_xn, train_y)


##############################################################################
### Confusion Matrix
##############################################################################


pre = clf.predict(test_xn)
cnf_matrix = confusion_matrix(test_y, pre)
cnf_matrix


# ### Fairly nice looking confusion matrix! I would like to avoid false negatives because I do not want to tell someone they dont have cancer but they really do, but 97% accuracy is still good!

# In[70]:


##############################################################################
### Precision Accuracy F Score
##############################################################################

a = precision_score(test_y, pre,pos_label = "M") * 100
b = recall_score(test_y, pre,pos_label = "M") * 100
c = f1_score(test_y, pre,pos_label = "M") * 100

print("Precision =", round(a,2),"% Recall =", round(b,2), "% F1 Score =", round(c,2),"%")


# ### A higher precision would be nice but we have nearly 100% recall. I am more concerned about telling someone they dont have cancer when they really do. The 97.37% recall is fairly good but the 1 time we were wrong is still costly.

# In[61]:


probas = clf.predict_proba(test_xn)
fpr, tpr, thresholds = roc_curve(test_y, probas[:,1], pos_label = "M")


# In[62]:


##############################################################################
### making ROC Curve
##############################################################################

def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth=2, label = label)
    plt.plot([0,1], [0,1], "k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
plot_roc_curve(fpr, tpr)
plt.show()


# ### Our high recall rate attributes to a fairly nice ROC curve. 

# In[82]:


##############################################################################
### SVM Lift Curve
##############################################################################

y_probas = clf.predict_proba(test_xn)
plot_lift_curve(test_y, y_probas)
plt.show()

