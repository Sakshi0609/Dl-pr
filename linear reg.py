
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/content/boston.csv')



plt.subplots(figsize=(12,8))
sns.heatmap(data.corr(),annot=True,cmap="RdGy")

selected_columns = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'LSTAT', 'MEDV']

# Plot pairplot
sns.pairplot(data[selected_columns], diag_kind='kde', corner=True)
plt.show()

X=data[['CRIM','ZN','INDUS','CHAS','NOX',	'RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y=data['MEDV']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)


lr=LinearRegression()

lr.fit(X_train,y_train)
prediction=lr.predict(X_test)
plt.scatter(y_test,prediction)
plt.xlabel("y test")
plt.ylabel("predictions")


print("MAE: ",(metrics.mean_absolute_error(y_test,prediction)))
print("MSE: ",(metrics.mean_squared_error(y_test,prediction)))
print("RMSE: ",(np.sqrt(metrics.mean_squared_error(y_test,prediction))))


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model with the correct input shape
model = Sequential()
# input_shape should match the number of features in X_train/X_test (13 in this case)
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, verbose=1, epochs=18, validation_split=0.05)
y_pred = model.predict(X_test)
mse_nn, mae_nn = model.evaluate(X_test, y_test)

print("MAE using NN: ", mae_nn)
print("MSE using NN: ", mse_nn)
print("RMSE using NN: ", np.sqrt(mse_nn))

import matplotlib.pyplot as plt # Make sure to import matplotlib
plt.scatter(y_test, y_pred)
plt.xlabel("Y test")
plt.ylabel("Y Predicted")
plt.show() # Add plt.show() to display the plot