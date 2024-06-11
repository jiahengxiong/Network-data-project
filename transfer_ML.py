import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time
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
from tensorflow.keras.models import save_model
import joblib

dataset_path = 'dataset/Single_EDFA_Dataset.csv'


def read_dataset(columns):
    df = pd.read_csv(dataset_path)
    inat_columns = [col for col in df.columns if col.startswith('inat')]
    outnat_columns = [col for col in df.columns if col.startswith('outat')]
    input_columns = []
    output_columns = []
    for channel in columns:
        for col in inat_columns:
            if channel in col:
                input_columns.append(col)
    for channel in columns:
        for col in outnat_columns:
            if channel in col:
                output_columns.append(col)
    print(input_columns)
    print(output_columns)

    input_matrix = df[input_columns].copy()
    output_matrix = df[output_columns].copy()

    return input_matrix, output_matrix


def simple_resnet_block(input_tensor, units, use_projection=False):
    x = Dense(units, activation='relu')(input_tensor)
    x = Dense(units)(x)

    if use_projection:
        input_tensor = Dense(units)(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def build_res_model(input_dim, output_dim, activate):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='linear')(inputs)

    # x = simple_resnet_block(x, 256)
    x = simple_resnet_block(x, 128, use_projection=True)
    # x = Dense(128, activation='linear')(x)

    outputs = Dense(output_dim, activation=activate)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def pre_train(columns):
    X, y = read_dataset(columns)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mse_list = []
    mae_list = []
    r2_list = []

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    model = build_res_model(input_dim, output_dim, 'linear')

    model.fit(X, y, epochs=200, batch_size=1024, verbose=2)

    print(model.summary())

    return model


def transfer(model, remain_columns):
    X, y = read_dataset(remain_columns)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    model.layers[-1].trainable = True
    model.compile(optimizer='adam', loss='mean_squared_error')
    mse_list = []
    mae_list = []
    r2_list = []
    for i in range(1, 21):
        model.fit(X_train, y_train, epochs=10, batch_size=1024)
        y_pred = model.predict(X_test)

        y_pred_inverse = scaler_y.inverse_transform(y_pred)
        y_test_inverse = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
        r2 = r2_score(y_test_inverse, y_pred_inverse)

        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)

    return mse_list, mae_list, r2_list


def draw(result):
    epochs = np.arange(10, len(result['Front_to_Back']['mse']) * 10 + 1, 10)

    # 绘制 MSE 图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result['Front_to_Back']['mse'], label='Front_to_Back MSE', marker='o')
    plt.plot(epochs, result['Back_to_Front']['mse'], label='Back_to_Front MSE', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/trans/MSE.png')
    plt.show()

    # 绘制 MAE 图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result['Front_to_Back']['mae'], label='Front_to_Back MAE', marker='o')
    plt.plot(epochs, result['Back_to_Front']['mae'], label='Back_to_Front MAE', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('MAE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/trans/MAE.png')
    plt.show()

    # 绘制 R² 图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, result['Front_to_Back']['r2'], label='Front_to_Back R²', marker='o')
    plt.plot(epochs, result['Back_to_Front']['r2'], label='Back_to_Front R²', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.title('R² over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result/trans/R2.png')
    plt.show()


if __name__ == '__main__':
    result = {'Front_to_Back': {'mse': [], 'mae': [], 'r2': []}, 'Back_to_Front': {'mse': [], 'mae': [], 'r2': []}}
    columns = list(range(152782, 154710))
    columns = [str(column) for column in columns]
    remain_columns = list(range(154711, 156684))
    remain_columns = [str(column) for column in remain_columns]

    model = pre_train(columns=columns)
    model.save('models/pretrained_model_1.h5')
    print("finish the pre_train")
    mse_list, mae_list, r2_list = transfer(model, remain_columns)
    result['Front_to_Back']['mse'] = mse_list
    result['Front_to_Back']['mae'] = mae_list
    result['Front_to_Back']['r2'] = r2_list

    new_model = pre_train(columns=remain_columns)
    new_model.save('models/pretrained_model_2.h5')
    mse_list, mae_list, r2_list = transfer(new_model, columns)
    result['Back_to_Front']['mse'] = mse_list
    result['Back_to_Front']['mae'] = mae_list
    result['Back_to_Front']['r2'] = r2_list

    draw(result)
