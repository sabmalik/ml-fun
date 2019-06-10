# load up the libraries we will need
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# load the dataset from the csv file
dataset = pd.read_csv('house-prices.csv')

# print the first few lines of the data
print(dataset.head())


# shows you the dimensionality of the data (rows, columns)
print(dataset.shape)

# spits out some interesting details about the data
print(dataset.describe())

# we always call the independent variables as X
# these are our features
X = dataset[['Bedrooms', 'Bathrooms', 'Land size sqm']]

# the dependent variable is called y
y = dataset['Above1.5m']

# lets split our data into training and test sets
# the size is mentioned in a number between 0 to 1
# here we have reserved 0.2 (20%) for our test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# instantiate the model (using the default parameters)
regressor = LogisticRegression(solver='lbfgs')

# fit the model with data
regressor.fit(X_train, y_train)

# lets make some predictions
y_pred = regressor.predict(X_test)

# show me the confusion matrix, takes the format
# (TrueNegative, FalsePositive, FalseNegative, TruePositive)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred).ravel()
print(confusion_matrix)

# print out some metrics
# how often is the prediction correct?
print('Accuracy [ (TP + TN)/Total ] :', metrics.accuracy_score(y_test, y_pred))

# whats the ratio of true positives to the all the positives returned
print('Precision [ TP/(TP + FP) ] :', metrics.precision_score(y_test, y_pred))

# true positives ratio to how many should have been true postives
print('Recall [ TP/(TP + FN) ]:', metrics.recall_score(y_test, y_pred))

# find and show me area under the curve - AUC - ROC
y_pred_proba = regressor.predict_proba(X_test)[::,1]
false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(false_positive_rate, true_positive_rate, label='auc=' + str(auc))
plt.legend(loc=4)
plt.show()