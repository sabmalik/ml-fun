# load up the libraries we will need
import pandas as pd  
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

# load the dataset from the csv file
dataset = pd.read_csv('house-prices.csv')

# print the first few lines of the data
print(dataset.head())


# shows you the dimensionality of the data (rows, columns)
print(dataset.shape)

# spits out some intersting details about the data
print(dataset.describe())

# we always call the independent variables as X
# these are our features
X = dataset[['Bedrooms', 'Bathrooms', 'Land size sqm']]

# the dependent variable is called y
y = dataset['Above500k']

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

# print the accuracy report
print(classification_report(y_test,y_pred))

# import the metrics class
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names = [0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")

plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()