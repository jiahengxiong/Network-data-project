import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
from tools.utils import read_dataset
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    model = load_model('models/res_net/res_net_1_linear.h5')
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    X, y = read_dataset(1)

    print(X.shape, y.shape)
    y_name = list(y.columns)
    X_name = list(X.columns)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train_sample = shap.sample(X, 100)

    # 使用 KernelExplainer 进行解释
    explainer_shap = shap.KernelExplainer(model.predict, X_train_sample)

    # 控制计算的精度和速度
    shap_values = explainer_shap.shap_values(X, nsamples=100)

    shap.summary_plot(shap_values, scaler_X.inverse_transform(X), feature_names=X_name, show=False)
    plt.title("Summary Plot")
    fig = plt.gcf()
    ax = fig.gca()
    ax.get_legend().remove()
    plt.savefig(f'result/XAI/Summary_Plot.png')
    plt.show()

    for i in range(len(y_name)):
        name = y_name[i]
        shap.summary_plot(shap_values[i], scaler_X.inverse_transform(X), feature_names=X_name, show=False, max_display=10)
        plt.title(f'Summary Plot for {name}')
        plt.savefig(f'result/XAI/Summary_Plot_for_{name}.png')
        plt.show()

    shap.dependence_plot('0', shap_values[0], scaler_X.inverse_transform(X), feature_names=X_name, interaction_index='1')
    plt.savefig(f'result/XAI/dependence_plot.png')
    plt.show()
