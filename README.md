# Network Data Project: EDFA Profile Regression

## Overview
This project, completed by Hao Zhou and Jiaheng Xiong, involves using deep neural network regression to predict the output power profile based on the input power profile. The tasks are divided into three main projects focused on Multiple Input Multiple Output (MIMO) regression.

## Project Structure
### 1. Preprocessing
- **Raw Data Visualization and Analysis**: Analyze the raw data to identify suitable features and machine learning algorithms.
- **Data Preprocessing**: Generate features from raw data.

### 2. Model Development
- **ML Model Optimization and Training**: 
  - Tune hyperparameters using cross-validation.
  - Use 2 or 3 ML algorithms (e.g., Neural Networks, Random Forest).
  - Evaluate performance using metrics like MSE, MAE, Accuracy, Precision, Recall, F1-score, and training duration.
- **Transfer Learning**: 
  - Pretrain the model on the first half of the dataset, fine-tune on the second half, and vice versa.
  - Freeze all layers except the last one during fine-tuning.

### 3. Performance Testing
- **Scenario Testing**: Evaluate the impact of including/excluding different features, feature normalization, and training set size on performance.
- **Federated Learning**: Compare global models with local models to share knowledge.

## Files
- **[main.py](main.py)**: Runs the primary functions of the project.
- **[transfer_ML.py](transfer_ML.py)**: Contains the transfer learning implementation.
- **[xai.py](xai.py)**: Explains model predictions using SHAP.
- **[models](models)**: Contains saved models.
- **[result](result)**: Stores results and visualizations.
- **[dataset](dataset)**: Includes datasets used for training and testing.

## Results
- The figures in [knn](result/fig/knn) shows the MAE, MSR and R2 score of KNN in different k values of different interval of input data.
![KNN-MAE.png](result%2Ffig%2Fknn%2FKNN-MAE.png)
![KNN-MSE.png](result%2Ffig%2Fknn%2FKNN-MSE.png)
![KNN-R2.png](result%2Ffig%2Fknn%2FKNN-R2.png)
- The figures in [nn](result/fig/nn) folder show the MAE, MSE and R2 of RESNET and normal NN.
![NN-MAE.png](result%2Ffig%2Fnn%2FNN-MAE.png)
![NN-MSE.png](result%2Ffig%2Fnn%2FNN-MSE.png)
![NN-R2.png](result%2Ffig%2Fnn%2FNN-R2.png)
![RES-MAE.png](result%2Ffig%2Fnn%2FRES-MAE.png)
![RES-R2.png](result%2Ffig%2Fnn%2FRES-R2.png)
![RES-MSE.png](result%2Ffig%2Fnn%2FRES-MSE.png)
- The figures in [rf](result/fig/rf) folder show the MAE, MSE and R2 score of Random Forest.
![RF-MAE.png](result%2Ffig%2Frf%2FRF-MAE.png)
![RF-MSE.png](result%2Ffig%2Frf%2FRF-MSE.png)
![RF-R2.png](result%2Ffig%2Frf%2FRF-R2.png)
- The figures in [trans](result/trans) folder show the MAE, MSE and R2 score of 2 transfer learning.
![MAE.png](result%2Ftrans%2FMAE.png)
![MSE.png](result%2Ftrans%2FMSE.png)
![R2.png](result%2Ftrans%2FR2.png)
- In the [Project_Network_Analysis.ipynb](Project_Network_Analysis.ipynb) file contain the result of:
1. Data Visualization
2. KNN Results (MAE,MSE,R2 and Training Time comparison between Normalize and Non Normalize Dataset)
3. Random Forest Regressor hyperparameter comparison (MAE,MSE,R2 and Training Time)
4. XAI Result (XGB Regressor)
