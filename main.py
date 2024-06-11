import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

from tools.utils import read_dataset, test_knn, draw, test_rf, test_res, test_nn


if __name__ == '__main__':
    input_interval = [1, 2, 5, 10, 20]
    algorithms = ['NN']
    matrices = ['MSE', 'MAE', 'R2']
    parameters = ['selu', 'linear', 'sigmoid']
    result = {}
    for algorithm in algorithms:
        result[algorithm] = {}
        if algorithm != 'RES' and algorithm != 'NN':
            for matrix in matrices:
                result[algorithm][matrix] = {}
                for interval in input_interval:
                    result[algorithm][matrix][interval] = []
        else:
            for parameter in parameters:
                result[algorithm][parameter] = {}
                for matrix in matrices:
                    result[algorithm][parameter][matrix] = {}
                    for interval in input_interval:
                        result[algorithm][parameter][matrix][interval] = []
    print(result)
    for n in input_interval:
        X, y = read_dataset(n)
        knn_mse, knn_mae, knn_r2 = test_knn(X, y, n)
        result['KNN']['MSE'][n] = knn_mse
        result['KNN']['MAE'][n] = knn_mae
        result['KNN']['R2'][n] = knn_r2
        rf_mse, rf_mae, rf_r2 = test_rf(X, y, n)
        result['RF']['MSE'][n] = rf_mse
        result['RF']['MAE'][n] = rf_mae
        result['RF']['R2'][n] = rf_r2
        for activate in parameters:
            res_mse, res_mae, res_r2 = test_res(X, y, n, activate)
            result['RES'][activate]['MSE'][n] = res_mse
            result['RES'][activate]["MAE"][n] = res_mae
            result['RES'][activate]["R2"][n] = res_r2
        for activate in parameters:
            res_mse, res_mae, res_r2 = test_nn(X, y, n, activate)
            result['NN'][activate]['MSE'][n] = res_mse
            result['NN'][activate]["MAE"][n] = res_mae
            result['NN'][activate]["R2"][n] = res_r2


    # print(result)
    for algorithm in algorithms:
        for matrix in matrices:
            draw(result, algorithm, matrix)
