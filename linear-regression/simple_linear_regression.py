# Load numpy, the charting library and scikit-learn the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the house area in an array and reshape it to an array of arrays
# that ML can understand. This is on the x-axis and we usually
# represent it with a capital X. As this is the training data, I
# named it X_train for convenience
X_train = np.array([326, 346, 371, 400, 407, 448, 452,
                    462, 466, 498, 500]).reshape((-1, 1))

# Load the house prices in an array and reshape it to an array of arrays
# that ML can understand. This is on the y-axis and we usually
# represent it with a small y. As this is the training data, I
# named it y_train for convenience
y_train = np.array([446000, 451500, 463500, 530000, 526000, 496000,
                    560000, 586000, 350000, 505000, 510000]).reshape((-1, 1))

# We want to provide the house area (x-axis) and want the model to predict
# the house price for the three values in the array.
X_pred = np.array([475, 700, 750]).reshape((-1, 1))

x_simple = np.array([326, 500]).reshape((-1, 1))
y_simple = np.array([446000, 510000]).reshape((-1, 1))

# create the object to use later
linear_regressor = LinearRegression()

# perform linear regression on the data
linear_regressor.fit(X_train, y_train)

# pass the original house areas again to
# see what the predictor says about them now
y_train_pred = linear_regressor.predict(X_train)

# make a scatter chart with the original values
plt.scatter(X_train, y_train)

# pass the original house area and the new prices
# for them. This is basically the trend line from Excel.
# plt.plot(X_train, y_train_pred, color='red')

plt.plot(x_simple, y_simple, color='red')

# show the chart
plt.show()

# uncomment for viewing in repl.it
# plt.savefig('plot.png')

# pass the values that we want to get a prediction for
y_pred = linear_regressor.predict(X_pred)


# print the predictions
print(y_pred)