import pandas as pd
from keras import Input
from keras.src.layers import MultiHeadAttention, LayerNormalization, Add, Activation
from keras.src.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import MeanSquaredError
from tools.utils import read_dataset

def build_nn_model(input_dim, output_dim, first, third):
    inputs = Input(shape=(input_dim,))
    x = Dense(first, activation='linear')(inputs)

    # x = simple_resnet_block(x, 256)
    x = Dense(128, activation='linear')(x)
    x = Dense(third, activation='linear')(x)

    outputs = Dense(output_dim, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model


def test_nn(X, y, first, third):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_nn_model(input_dim, output_dim, first, third)

    model.fit(X_train, y_train, epochs=200, batch_size=1024, verbose=0)

    y_pred = model.predict(X_test)

    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inverse, y_pred_inverse)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    r2 = r2_score(y_test_inverse, y_pred_inverse)

    return mse, mae, r2


if __name__ == "__main__":
    MSE = {}
    MAE = {}
    R2 = {}
    first_list = [32, 64, 128, 256, 512]
    third_list = [32, 64, 128, 256, 512]
    for first in first_list:
        MSE[first] = {}
        MAE[first] = {}
        R2[first] = {}
        for third in third_list:
            MSE[first][third] = 0
            MAE[first][third] = 0
            R2[first][third] = 0

    X, y = read_dataset(1)
    for first in first_list:
        for third in third_list:
            mse, mae, r2 = test_nn(X, y, first, third)
            MSE[first][third] = mse
            MAE[first][third] = mae
            R2[first][third] = r2
    with open('result.txt', 'w') as f:
        f.write('MSE:\n' + str(MSE) + '\n')
        f.write('MAE:\n' + str(MAE) + '\n')
        f.write('R2:\n' + str(R2))