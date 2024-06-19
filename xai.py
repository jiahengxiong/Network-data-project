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
    y_names = list(y.columns)
    X_names = list(X.columns)
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 不进行标准化处理，确保数据的原始值用于解释
    # X = scaler_X.fit_transform(X)
    # y = scaler_y.fit_transform(y)

    X_train_sample = shap.sample(X, 200)
    print(X_names)

    # 使用 KernelExplainer 进行解释
    explainer_shap = shap.KernelExplainer(model.predict, X_train_sample)

    # 控制计算的精度和速度
    shap_values = explainer_shap.shap_values(X)

    # 打印shap_values的类型和形状，以便调试
    print(f"shap_values type: {type(shap_values)}")
    if isinstance(shap_values, np.ndarray):
        print(f"shap_values shape: {shap_values.shape}")

        # 确保shap_values的形状是 (100, 84, 84)
        if shap_values.shape[1] == X_train_sample.shape[1] and shap_values.shape[2] == X_train_sample.shape[1]:
            """shap.summary_plot(shap_values, X_train_sample, show=False)
            plt.title("Summary Plot")
            fig = plt.gcf()
            ax = fig.gca()
            # ax.get_legend().remove()
            fig.savefig(f'result/XAI/Summary_Plot.png')"""
            for i in range(shap_values.shape[2]):
                shap.summary_plot(shap_values[:, :, i], X, feature_names=X_names, show=False)
                plt.title(f"Summary Plot for Output Feature {i}")
                plt.savefig(f'result/XAI/Summary_Plot_Output_Feature_{i}.png')
                # plt.show()

            # 选取一个输出特征绘制依赖图
            shap.dependence_plot('inat156589', shap_values[81], X, feature_names=X_names,
                                 interaction_index='inat156683')
            plt.savefig('result/XAI/dependence_plot.png')
            # plt.show()
        else:
            print("Error: shap_values shape does not match expected dimensions.")
    else:
        print("shap_values is not in the expected format.")
