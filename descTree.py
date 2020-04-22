from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pydot
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)
#print(classification_report(y_test, predictions))


features = list(df.columns[0:8])


def plot_decision_tree(clf, feature_name, target_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


#plot_decision_tree(dtree, X_train.columns, df.columns[1])
