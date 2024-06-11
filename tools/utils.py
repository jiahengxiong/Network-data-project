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

dataset_path = 'dataset/Single_EDFA_Dataset.csv'


def read_dataset(n):
    df = pd.read_csv(dataset_path)
    inat_columns = [col for col in df.columns if col.startswith('inat')]
    outnat_columns = [col for col in df.columns if col.startswith('outat')]
    inat_columns = inat_columns[::n]

    input_matrix = df[inat_columns].copy()
    output_matrix = df[outnat_columns].copy()
    """print(input_matrix)
    print(output_matrix)"""

    return input_matrix, output_matrix


def test_knn(X, y, interval):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_neighbors = range(1, 21)
    mse_values = []
    r2_values = []
    mae_values = []
    train_durations = []

    for n in n_neighbors:
        start_time = time.time()
        knn = KNeighborsRegressor(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # print(f'Training KNN model......')

        training_duration = time.time() - start_time
        train_durations.append(training_duration)

        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

        score = r2_score(y_test, y_pred)
        r2_values.append(score)

        mae = mean_absolute_error(y_test, y_pred)
        mae_values.append(mae)

        print(f'KNN Training duration with N={n} : {training_duration:.4f} seconds with interval {interval}')
        print(f'KNN Mean Squared Error with N={n} : {mse} with interval {interval}')
        print(f'KNN Mean Absolute Error with N={n} : {mae} with interval {interval}')
        print(f'KNN R2 score with N={n} : {score} with interval {interval}')

    return mse_values, mae_values, r2_values


def test_rf(X, y, interval):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mse_list = []
    mae_list = []
    r2_list = []
    n_estimators_list = range(10, 210, 10)

    for n_estimators in n_estimators_list:
        start_time = time.time()
        rf = RandomForestRegressor(random_state=42, n_estimators=n_estimators, max_features='sqrt', max_depth=10,
                                   min_samples_split=2, min_samples_leaf=1)
        multi_output_rf = MultiOutputRegressor(rf)

        multi_output_rf.fit(X_train, y_train)

        y_pred = multi_output_rf.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        training_duration = time.time() - start_time

        print(f'Random forest Training duration with n_estimators={n_estimators} : {training_duration:.4f} seconds '
              f'with interval {interval}')
        print(f'Random forest Mean Squared Error with n_estimators={n_estimators} : {mse} with interval {interval}')
        print(f'Random forest Mean Absolute Error with n_estimators={n_estimators} : {mae} with interval {interval}')
        print(f'Random forest R2 score with n_estimators={n_estimators} : {r2} with interval {interval}')

    return mse_list, mae_list, r2_list


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


def test_res(X, y, interval, activate):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mse_list = []
    mae_list = []
    r2_list = []

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    epochs_list = range(10, 210, 10)

    for epochs in epochs_list:
        model = build_res_model(input_dim, output_dim, activate)

        model.fit(X_train, y_train, epochs=epochs, batch_size=1024, verbose=0)

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


def build_nn_model(input_dim, output_dim, activate):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='linear')(inputs)

    # x = simple_resnet_block(x, 256)
    x = Dense(128, activation='linear')(x)
    x = Dense(128, activation='linear')(x)

    outputs = Dense(output_dim, activation=activate)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def test_nn(X, y, interval, activate):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mse_list = []
    mae_list = []
    r2_list = []

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    epochs_list = range(10, 210, 10)

    for epochs in epochs_list:
        model = build_nn_model(input_dim, output_dim, activate)

        model.fit(X_train, y_train, epochs=epochs, batch_size=1024, verbose=0)

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


def draw(data, algorithm, matrix):
    """if algorithm not in data or matrix not in data[algorithm]:
        print(f"No data available for algorithm '{algorithm}' and matrix '{matrix}'.")
        return"""

    plt.figure(figsize=(10, 6))

    if algorithm == 'KNN':
        matrix_data = data[algorithm][matrix]
        for key in matrix_data:
            plt.plot(range(1, len(matrix_data[key]) + 1), matrix_data[key], label=f'{key} interval of input',
                     marker='o',
                     linestyle='-')
        plt.title(f'{algorithm} - {matrix} Over Different K Values')
        plt.xlabel('K')
        plt.ylabel(matrix)
        plt.xticks(range(1, 21))
        plt.legend()
        plt.grid(True)
        plt.savefig(f'result/fig/knn/{algorithm}-{matrix}.png')
        plt.show()
    if algorithm == 'RF':
        matrix_data = data[algorithm][matrix]
        for key in matrix_data:
            plt.plot(range(10, 10 * len(matrix_data[key]) + 10, 10), matrix_data[key], label=f'{key} interval of input',
                     marker='o',
                     linestyle='-')
        plt.title(f'{algorithm} - {matrix} Over Different estimators Values')
        plt.xlabel('estimators')
        plt.ylabel(matrix)
        plt.xticks(range(10, 210, 10))
        plt.legend()
        plt.grid(True)
        plt.savefig(f'result/fig/rf/{algorithm}-{matrix}.png')
        plt.show()
    if algorithm == 'RES' or algorithm == 'NN':
        res_data = data[algorithm]
        for activate in res_data:
            matrix_data = res_data[activate][matrix]
            for key in matrix_data:
                if activate == 'linear':
                    plt.plot(range(10, 10 * len(matrix_data[key]) + 10, 10), matrix_data[key],
                             label=f'{key} interval input-{activate} activation',
                             marker='o',
                             linestyle='-')
                else:
                    plt.plot(range(10, 10 * len(matrix_data[key]) + 10, 10), matrix_data[key],
                             label=f'{key} interval input-{activate} activation',
                             marker='o',
                             linestyle='--')
        plt.title(f'{algorithm} - {matrix} Over Different epoch Values')
        plt.xlabel('epoch')
        plt.ylabel(matrix)
        plt.xticks(range(10, 210, 10))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.savefig(f'result/fig/{algorithm}/{algorithm}-{matrix}.png')
        plt.show()


""" plt.figure(figsize=(10, 6))
plt.plot(n_neighbors, mse_values, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (N)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'MSE vs. Number of Neighbors in KNN with interval {interval}')
plt.grid(True)
plt.savefig(f'result/fig/knn/MSE vs. Number of Neighbors in KNN with interval with interval {interval}.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_neighbors, mae_values, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (N)')
plt.ylabel('Mean Absolute Eorre (MAE)')
plt.title(f'MAE vs. Number of Neighbors in KNN with interval {interval}')
plt.grid(True)
plt.savefig(f'result/fig/knn/MAE vs. Number of Neighbors in KNN with interval {interval}.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_neighbors, r2_values, marker='o', linestyle='-')
plt.xlabel('Number of Neighbors (N)')
plt.ylabel('R2 score (R2)')
plt.title(f'R2 vs. Number of Neighbors in KNN with interval {interval}')
plt.grid(True)
plt.savefig(f'result/fig/knn/R2 vs. Number of Neighbors in KNN with interval {interval}.png')
plt.show()"""
