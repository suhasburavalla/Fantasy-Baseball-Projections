import pandas as pd
import sys
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
np.set_printoptions(threshold=sys.maxsize)

# HYPERPARAMETER DEFINITION

LR = .00001
EPOCHS = 10000

def hitters_csv_new ():

    H_data = pd.read_csv("H_data.csv")
    # print(H_data.head())
    # print(H_data.shape)
    return H_data

def pitchers_csv_new ():

    P_data = pd.read_csv("P_data.csv")
    # print(P_data.head())
    # print(P_data.shape)
    return P_data

def hitters_rf (hitters_all) :
    
    hitters_all = hitters_all.fillna(0.0000000001)
    
    print("\n Random Forest Analysis: Hitters\n")

    X = hitters_all.iloc[:, 3:55].values
    y = hitters_all.iloc[:, 55:60].values
    col_names = hitters_all.columns[3:55].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, shuffle=True)

    regressor = DecisionTreeRegressor(random_state=0, max_features='sqrt')
    regressor.fit(X_train, y_train)
    print("10-fold CV score, RF model for Hitters:")
    print(cross_val_score(regressor, X_train, y_train, cv=10))
    
    # Get feature importances
    importances = regressor.feature_importances_

 
    feature_dict = {i: col_names[i] for i in range(len(col_names))}
    # features = ["Feature " + str(i) for i in range(X.shape[1])]
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    # sorted_features = []
  
    # for i in indices:
    #     sorted_features.append(feature_dict[i])

    # Plot the feature importances
    # plt.figure(figsize=(10,5))
    # plt.title("Feature Importance - Hitters")
    # plt.bar(sorted_features, importances[indices])
    # plt.xticks(rotation=90)
    # print(feature_dict)
    # plt.show()
    
    # Get the top 10 important features
    top_k = 10
    sorted_features = [feature_dict[i] for i in indices[:top_k]]

    # Get the importances for the top k features
    importances_top_k = importances[indices][:top_k]

    # Plot the feature importances
    plt.figure(figsize=(10,5))
    plt.title("Feature Importance - Hitters")
    plt.bar(sorted_features, importances_top_k)
    plt.xticks(rotation=90)
    plt.show()
    
    

def pitchers_rf(pitchers_all):

    pitchers_all = pitchers_all.fillna(0.0000000001)

    print("\n Random Forest Analysis: Pitchers\n")
    X = pitchers_all.iloc[:, 3:67].values
    y = pitchers_all.iloc[:, 67:72].values
    col_names = pitchers_all.columns[3:67].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, shuffle=True)

    regressor = DecisionTreeRegressor(random_state=0, max_features='sqrt')
    regressor.fit(X_train, y_train)
    print("10-fold CV score, RF model for Pitchers:")
    print(cross_val_score(regressor, X_train, y_train, cv=10))
    
    # Get feature importances
    importances = regressor.feature_importances_

 
    feature_dict = {i: col_names[i] for i in range(len(col_names))}
    # features = ["Feature " + str(i) for i in range(X.shape[1])]
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    # sorted_features = []
  
    # for i in indices:
    #     sorted_features.append(feature_dict[i])

    # Plot the feature importances
    # plt.figure(figsize=(10,5))
    # plt.title("Feature Importance - Pitchers")
    # plt.bar(sorted_features, importances[indices])
    # plt.xticks(rotation=90)
    # print(feature_dict)
    # plt.show()
    
    # Get the top 10 important features
    top_k = 10
    sorted_features = [feature_dict[i] for i in indices[:top_k]]

    # Get the importances for the top k features
    importances_top_k = importances[indices][:top_k]

    # Plot the feature importances
    plt.figure(figsize=(10,5))
    plt.title("Feature Importance - Pitchers")
    plt.bar(sorted_features, importances_top_k)
    plt.xticks(rotation=90)
    plt.show()
    
def MLP_hitters(H_data):

    H_data = H_data.iloc[:, 2:]
    dummies = pd.get_dummies(H_data.Team_x, prefix='Team')
    H_data = H_data.join(dummies)
    H_data = H_data.drop(["Team_x"], axis=1)

    # print(H_data.head(5))
    # print(H_data.isnull().any().any())

    X_train, X_test = train_test_split(H_data, test_size=0.20)

    # train targets
    HR_y_train = np.array(X_train["HR_y"])
    R_y_train = np.array(X_train["R_y"])
    RBI_y_train = np.array(X_train["RBI_y"])
    SB_y_train = np.array(X_train["SB_y"])
    AVG_y_train = np.array(X_train["AVG_y"])

    # test targets
    HR_y_test = np.array(X_test["HR_y"])
    R_y_test = np.array(X_test["R_y"])
    RBI_y_test = np.array(X_test["RBI_y"])
    SB_y_test = np.array(X_test["SB_y"])
    AVG_y_test = np.array(X_test["AVG_y"])

    X_train = X_train.drop(["HR_y", "R_y", "RBI_y", "SB_y", "AVG_y"], axis=1)
    X_test = X_test.drop(["HR_y", "R_y", "RBI_y", "SB_y", "AVG_y"], axis=1)

    input_layer = Input(shape=(len(X_train.columns)))
    dense_layer_1 = Dense(units=128, activation="relu")(input_layer)
    dense_layer_2 = Dense(units=64, activation="relu")(dense_layer_1)

    HR_y_output = Dense(units=1, activation="linear", name="HR_y_output")(dense_layer_2)
    R_y_output = Dense(units=1, activation="linear", name="R_y_output")(dense_layer_2)
    RBI_y_output = Dense(units=1, activation="linear", name="RBI_y_output")(dense_layer_2)
    SB_y_output = Dense(units=1, activation="linear", name="SB_y_output")(dense_layer_2)
    AVG_y_output = Dense(units=1, activation="linear", name="AVG_y_output")(dense_layer_2)

    model = Model(inputs=input_layer, outputs=[HR_y_output, R_y_output, RBI_y_output, SB_y_output, AVG_y_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(optimizer=optimizer,
                  loss={'HR_y_output': 'mse', 'R_y_output': 'mse', 'RBI_y_output': 'mse', 'SB_y_output': 'mse', 'AVG_y_output': 'mse'},
                  metrics={
                      'HR_y_output': tf.keras.metrics.MeanSquaredError(),
                      'R_y_output': tf.keras.metrics.MeanSquaredError(),
                      'RBI_y_output': tf.keras.metrics.MeanSquaredError(),
                      'SB_y_output': tf.keras.metrics.MeanSquaredError(),
                      'AVG_y_output': tf.keras.metrics.MeanSquaredError(),
                  })
    history = model.fit(X_train, (HR_y_train, R_y_train, RBI_y_train, SB_y_train, AVG_y_train), epochs=EPOCHS, batch_size=512, validation_data=(X_test, (HR_y_test, R_y_test, RBI_y_test, SB_y_test, AVG_y_test)))

    return 0

def MLP_pitchers(P_data):

    P_data = P_data.iloc[:, 2:]
    dummies = pd.get_dummies(P_data.Team_x, prefix='Team')
    P_data = P_data.join(dummies)
    P_data = P_data.drop(["Team_x"], axis=1)

    # print(P_data.head(5))
    # print(P_data.isnull().any().any())

    X_train, X_test = train_test_split(P_data, test_size=0.20)

    # train targets
    W_y_train = np.array(X_train["W_y"])
    SV_y_train = np.array(X_train["SV_y"])
    ERA_y_train = np.array(X_train["ERA_y"])
    SO_y_train = np.array(X_train["SO_y"])
    WHIP_y_train = np.array(X_train["WHIP_y"])

    # test targets
    W_y_test = np.array(X_test["W_y"])
    SV_y_test = np.array(X_test["SV_y"])
    ERA_y_test = np.array(X_test["ERA_y"])
    SO_y_test = np.array(X_test["SO_y"])
    WHIP_y_test = np.array(X_test["WHIP_y"])

    X_train = X_train.drop(["W_y", "SV_y", "ERA_y", "SO_y", "WHIP_y"], axis=1)
    X_test = X_test.drop(["W_y", "SV_y", "ERA_y", "SO_y", "WHIP_y"], axis=1)

    input_layer = Input(shape=(len(X_train.columns)))
    dense_layer_1 = Dense(units=128, activation="relu")(input_layer)
    dense_layer_2 = Dense(units=64, activation="relu")(dense_layer_1)

    W_y_output = Dense(units=1, activation="linear", name="W_y_output")(dense_layer_2)
    SV_y_output = Dense(units=1, activation="linear", name="SV_y_output")(dense_layer_2)
    ERA_y_output = Dense(units=1, activation="linear", name="ERA_y_output")(dense_layer_2)
    SO_y_output = Dense(units=1, activation="linear", name="SO_y_output")(dense_layer_2)
    WHIP_y_output = Dense(units=1, activation="linear", name="WHIP_y_output")(dense_layer_2)

    model = Model(inputs=input_layer, outputs=[W_y_output, SV_y_output, ERA_y_output, SO_y_output, WHIP_y_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    model.compile(optimizer=optimizer,
                  loss={'W_y_output': 'mse', 'SV_y_output': 'mse', 'ERA_y_output': 'mse', 'SO_y_output': 'mse', 'WHIP_y_output': 'mse'},
                  metrics={
                      'W_y_output': tf.keras.metrics.MeanSquaredError(),
                      'SV_y_output': tf.keras.metrics.MeanSquaredError(),
                      'ERA_y_output': tf.keras.metrics.MeanSquaredError(),
                      'SO_y_output': tf.keras.metrics.MeanSquaredError(),
                      'WHIP_y_output': tf.keras.metrics.MeanSquaredError(),
                  })
    history = model.fit(X_train, (W_y_train, SV_y_train, ERA_y_train, SO_y_train, WHIP_y_train), epochs=EPOCHS, batch_size=512, validation_data=(X_test, (W_y_test, SV_y_test, ERA_y_test, SO_y_test, WHIP_y_test)))

    return 0

def get_model_results(H_data, P_data):

    # Naive model: previous year's performance is predicted performance
    # We need to outperform these metrics

    naive_MSE_HR = mean_squared_error(H_data['HR_x'], H_data['HR_y'])
    naive_MSE_R = mean_squared_error(H_data['R_x'], H_data['R_y'])
    naive_MSE_RBI = mean_squared_error(H_data['RBI_x'], H_data['RBI_y'])
    naive_MSE_SB = mean_squared_error(H_data['SB_x'], H_data['SB_y'])
    naive_MSE_AVG = mean_squared_error(H_data['AVG_x'], H_data['AVG_y'])

    naive_MSE_W = mean_squared_error(P_data['W_x'], P_data['W_y'])
    naive_MSE_SV = mean_squared_error(P_data['SV_x'], P_data['SV_y'])
    naive_MSE_ERA = mean_squared_error(P_data['ERA_x'], P_data['ERA_y'])
    naive_MSE_SO = mean_squared_error(P_data['SO_x'], P_data['SO_y'])
    naive_MSE_WHIP = mean_squared_error(P_data['WHIP_x'], P_data['WHIP_y'])

    print("MSE for Naive model - HR: ", naive_MSE_HR)
    print("MSE for Naive model - R: ", naive_MSE_R)
    print("MSE for Naive model - RBI: ", naive_MSE_RBI)
    print("MSE for Naive model - SB: ", naive_MSE_SB)
    print("MSE for Naive model - AVG: ", naive_MSE_AVG)

    print("MSE for Naive model - W: ", naive_MSE_W)
    print("MSE for Naive model - SV: ", naive_MSE_SV)
    print("MSE for Naive model - ERA: ", naive_MSE_ERA)
    print("MSE for Naive model - SO: ", naive_MSE_SO)
    print("MSE for Naive model - WHIP: ", naive_MSE_WHIP)

    return 0
    
if __name__ == "__main__" :

    # H_data = hitters_data_read_new()
    # P_data = pitchers_data_read_new()
    # H_data = hitters_preprocessing_new(H_data)
    # P_data = pitchers_preprocessing_new(P_data)
    
    H_data = hitters_csv_new()
    P_data = pitchers_csv_new()
    # hitters_rf(H_data)
    # pitchers_rf(P_data)
    # MLP_hitters(H_data)
    # MLP_pitchers(P_data)
    get_model_results(H_data, P_data)
    
    
    