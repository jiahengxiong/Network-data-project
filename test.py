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

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

if __name__ == '__main__':
    model = load_model('models/nn/nn_1_linear.h5')
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    # 打印模型结构
    model.summary()

    # 保存模型结构图
    plot_model(model, to_file='models/nn/nn_structure.png', show_shapes=True, show_layer_names=True)
    print("Model structure saved to 'models/nn/nn_1_structure.png'")


