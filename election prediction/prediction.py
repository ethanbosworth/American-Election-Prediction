# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:50:07 2020

@author: Ethan Bosworth

a script for predicting American elections using Decision trees
"""

#%% import modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

#%% import data
#import the csv data
data = pd.read_csv("data.csv")

#take the election winner and convert winning party to numeric
y = pd.DataFrame(data["Election"])
y[y["Election"].str.contains("P")] = 1
y[y["Election"] != 1] = 0
y = y.astype("int")

#take the data for the answers to the 12 questions
X = data.drop(["Election","Year"],axis = 1)
#replace y with 1 and n with -1
X = X.replace(["y","n"],[1,-1])

#%% Decision Tree
#split data to take 3 P party victories and 2 O party victories

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

#from scikit learn documentation for viewing tree structure

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
        
del clf
#%% extension of dataset
#extend the dataframe by taking each entry and adding an unknonwn 0 in each possible
# positition and adding to the original dataframe

X_ext = X.copy() # create a copy of X and y to be extended
y_ext = y.copy()
for i in X.columns: # iterate over each column
    temp = X.copy()
    temp[i] = 0 # make a copy of X and set the i-th column to 0
    X_ext = pd.concat([X_ext,temp]) # add to the extended dataframe
    y_ext = pd.concat([y_ext,y])
y_ext = y_ext.reset_index()
y_ext = y_ext.drop("index",axis = 1)
X_ext.index = y_ext.index
#%% Decision Tree extended
#split data to take 50 random samples

X_train, X_test, y_train, y_test = train_test_split( X_ext, y_ext, test_size=0.127, random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(acc)

#from scikit learn documentation for viewing tree structure

n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
threshold = clf.tree_.threshold


# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has "
      "the following tree structure:"
      % n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
    else:
        print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
              "node %s."
              % (node_depth[i] * "\t",
                 i,
                 children_left[i],
                 feature[i],
                 threshold[i],
                 children_right[i],
                 ))
        

#This decision tree is much larger and more accurate than previously
del clf

#%% Pruning
#do not split the cells with less than m examples


m = [2,3,4] # task asks for m to be either 2,3 or 4



#create an error dataframe to add all errors to
error_full = pd.DataFrame(m)
error_full.index = m
error_full.columns = ["drop"]
error_full = error_full.drop("drop",axis = 1)

for j in range(50): #iterate over N different random states

    error_train = [] #set some empty variables to store error
    error_test = []
    accuracy = []
    
    for i in m: #iterate over m to make the decision tree
        X_train, X_test, y_train, y_test = train_test_split( X_ext, y_ext, test_size=0.127, random_state=j)
        
        clf = DecisionTreeClassifier(min_samples_split = i)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test,y_pred))
        #add the mse to the variables
        error_test.append(mean_squared_error(y_test,y_pred))
        error_train.append(mean_squared_error(y_train,clf.predict(X_train)))
    
    #convert the error lists to a dataframe and set the labels
    error = pd.concat([pd.DataFrame(error_test),pd.DataFrame(error_train)],axis = 1)
    error.index = m
    error.columns = ["test_error","train_error"]
    error_full = pd.concat([error_full,error]) #add the error to the error full

#remove missing values and group by the value for m by mean
error_full = error_full.dropna()
error_full = error_full.reset_index()
error_full = error_full.groupby(by = "index").mean()

#plot mse against m for the average error
a = sns.lineplot(data = error_full)
a.set_ylabel("mse")
plt.show()

#m = 3 gives best results
#%% predicting an election
#Prediction of the american 2016 election

X_2016 = pd.DataFrame(X.iloc[0]).T # take the first row of X to get the dataframe form
X_2016.iloc[0] = [1,1,-1,1,-1,-1,-1,0,1,1,-1,-1] # input the answers to questions

#create a classifier with best m
clf = DecisionTreeClassifier(min_samples_split = 3)
clf.fit(X_ext,y_ext)#fit the whole data
y_pred = clf.predict(X_2016) # predict
    
print(y_pred[0]) # prediction gives 0 which means an opposition win which is correct


#addition of the 2020 election

X_2020 = pd.DataFrame(X.iloc[0]).T # take the first row of X to get the dataframe form
X_2020.iloc[0] = [-1,-1,-1,-1,1,1,-1,1,1,1,-1,-1] # input the answers to questions

#create a classifier with best m
clf = DecisionTreeClassifier(min_samples_split = 3)
clf.fit(X_ext,y_ext)#fit the whole data
y_pred = clf.predict(X_2020) # predict
    
print(y_pred[0]) # prediction gives 1 which means a presidential party win which is incorrect
#this might be because it was an unusual election or I might have put some answers to questions wrong

