import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load the dataset from the csv file
dataset = pd.read_csv('house-prices.csv')

# shows you the dimentionality of the data (rows, columns)
print(dataset.shape)

# spits out some intersting details about the data
print(dataset.describe())

# we always call the independent variables as X
X = dataset[['Bed Rooms', 'Bath Rooms', 'Land size sqm']]

# the dependent variable is called y
y = dataset['Price']

# show the distribution of prices
plt.figure(figsize=(8,4))
plt.tight_layout()
sns.distplot(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1.plot(kind='bar',figsize=(8,4))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()