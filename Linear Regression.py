import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# importing the Dataset
housing_data = pd.read_csv("C:\\Users\\hp\\PycharmProjects\\pythonProject4\\DA_housing.csv")
X = housing_data.iloc[:, :-1].values
y = housing_data.iloc[:, 1].values

# Visualizing the Dataset
sns.barplot(x='housing_median_age', y='median_house_value', data=housing_data)

# Splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# Predict the Test Set results
y_pred = lr.predict(X_test)
print(y_pred)

# 8 Visualizing the Test Set results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title("Salary VS Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
print(len(X_train))
print(len(y_train))
#7 Visualizing the train set Results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title("Salary ~ Experience (Train set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#9 Finding/Calculating the Residuals
from sklearn import metrics
print("MAE:\t", metrics.mean_absolute_error(y_test, y_pred))
print("MSE:\t", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:\t", np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))

