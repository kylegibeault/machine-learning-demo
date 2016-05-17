from sklearn import tree
# imports the decision tree module from the scikit-learn project
features = [[140, 1],[130, 1],[150, 0],[170, 0]]
# Stores training samples of data
labels = [0,0,1,1]
# stores the classification value of each of the data into an array
clf = tree.DecisionTreeClassifier()
# instantiating a decision tree classifier from the tree module into the variable clf
clf = clf.fit(features, labels)
# Importing the training data to "fit" the model
print clf.predict([[160,0]])
# Predicting how the model will evaluate a new data input