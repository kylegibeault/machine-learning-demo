
# imports the decision tree module from the scikit-learn project
from sklearn import tree

# Stores training samples of data
features = [[140, 1],[130, 1],[150, 0],[170, 0]]

# stores the classification value of each of the data into an array
labels = [0,0,1,1]

# instantiating a decision tree classifier from the tree module into the variable clf
clf = tree.DecisionTreeClassifier()

# Importing the training data to "fit" the model
clf = clf.fit(features, labels)

# Predicting how the model will evaluate a new data input
print clf.predict([[160,0]])

#Visualizing the decision tree created by the model

from sklearn.externals.six import StringIO  
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, 
		out_file=dot_data,
		filled=True,rounded=True,
		impurity=False
		) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_svg("decisiontree.svg") 