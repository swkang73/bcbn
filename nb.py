import sys
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import LeaveOneOut
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import pandas as pd 

#print("HELLO WORLD.")

# Read in data
data = pd.read_csv('p_data.csv')

#Convert the features and outcome variables to binary (1 or 0)
data.diagnosis = data.diagnosis.eq('M').mul(1)

# Divide the data into 80% training and 20% test.
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 2:].to_numpy(), data.iloc[:, 1].to_numpy(), test_size=0.2, random_state=1115)

#  Build a Naive Bayes model using the training data.
clf = BernoulliNB()
clf.fit(X_train, y_train)

# Use the Naive Bayes model to predict the classes of the test set.
y_predict = clf.predict(X_test)
#print(y_predict)

# Set up a confusion matrix comparing the predicted to actual values. 
print(confusion_matrix(y_test, y_predict))

# Examine which features are most related to the outcome of interest.
features = data.columns.values
print(mutual_info_classif(data.iloc[:, 2:].to_numpy(), data.iloc[:, 1].to_numpy(), discrete_features='auto', n_neighbors=3, copy=True, random_state=None))

