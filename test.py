import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

import numpy as np
from tools.utils import read_dataset

X, y = read_dataset(1)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# 参数范围
n_estimators_range = range(10, 120, 10)
criterion = ['squared_error', 'friedman_mse']

# 字典存储评估指标
mse_values_rf = {crit: [] for crit in criterion}
mae_values_rf = {crit: [] for crit in criterion}
r2_values_rf = {crit: [] for crit in criterion}
train_durations_rf = {crit: [] for crit in criterion}
for n in n_estimators_range:
    crit = 'friedman_mse'
    if n > 100:
        break

    start_time = time.time()
    print(f'Start Training..............')
    print(f'Criterion: {crit}')
    print(f'Number of Estimators: {n}')

    rf = RandomForestRegressor(n_estimators=n, random_state=42, criterion=crit)
    rf.fit(X_train_rf, y_train_rf)

    training_duration = time.time() - start_time
    train_durations_rf[crit].append(training_duration)

    y_pred_rf = rf.predict(X_test_rf)

    mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
    mse_values_rf[crit].append(mse_rf)

    mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
    mae_values_rf[crit].append(mae_rf)

    r2_rf = r2_score(y_test_rf, y_pred_rf)
    r2_values_rf[crit].append(r2_rf)

    print(f'Training duration: {training_duration:.4f} seconds')
    print(f'Mean Squared Error: {mse_rf}')
    print(f'Mean Absolute Error: {mae_rf}')
    print(f'R2 score: {r2_rf}')
