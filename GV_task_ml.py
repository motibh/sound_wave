import pandas as pd
import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def plot_ml(y_pred, label_test, ml=[]):
    plt.figure()
    relative_err = np.abs(y_pred - label_test) / np.abs(label_test)
    plt.plot(relative_err, 'or')
    mse = mean_squared_error(label_test, y_pred)
    plt.title(f'relative error- {ml}, median= {np.round(np.median(relative_err),4)},  mse = {np.round(mse,4)}')
    plt.show()

xls = pd.read_csv(os.getcwd()+f'\\gv_data1.csv')
feat=['area_osc', 'Peak.us','FT.1', 'FT.3', 'FT.5' ,'FT.7', 'FT.9']
X = xls[feat]
# normalization of features
scaler = StandardScaler()
X[feat] = scaler.fit_transform(X[feat])
y = xls['Tag.Mpa']
# split train & test
x_train, x_test, label_train, label_test = train_test_split(X, y, test_size=0.2)
# first- K Neighbors Regressor
neigh = KNeighborsRegressor(n_neighbors=6,weights='distance')
neigh.fit(x_train, label_train)
y_pred=neigh.predict(x_test)
plot_ml(y_pred, label_test, ml='KNeighborsRegressor')
# second-  random forest
rfr = RandomForestRegressor(n_estimators=1000,max_features='auto')
rfr.fit(x_train, label_train)
y_pred_rf=rfr.predict(x_test)
plot_ml(y_pred_rf, label_test, ml='RandomForest')


# third -LinearRegression
reg = LinearRegression().fit(x_train, label_train)
y_pred_reg=reg.predict(x_test)
plot_ml(y_pred_reg, label_test, ml='LinearRegression')

#plt.plot(label_test[:20],'*b')
time.sleep(5.5)  #  # Pause 5.5 seconds

