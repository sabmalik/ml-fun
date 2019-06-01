import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load the dataset from the csv file
dataset = pd.read_csv('house-prices.csv')

# shows you the dimensionality of the data (rows, columns)
print(dataset.shape)

# spits out some intersting details about the data
print(dataset.describe())

# we always call the independent variables as X
X = dataset[['Bedrooms', 'Bathrooms', 'Land size sqm']]

# the dependent variable is called y
y = dataset['Price']

# lets split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# use regression on our training data
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

# get and show us the coefficients/weights
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)

# do some predictions on the test dataset
y_pred = regressor.predict(X_test)

# show us the differences in actual and predicted prices
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

# plot them for us please
df1.plot(kind='bar',figsize=(8,4))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# show some fancy metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))