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

    print(H_data.head(5))
    print(H_data.isnull().any().any())

    X_train, X_test = train_test_split(H_data, test_size=0.25)

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
    dense_layer_2 = Dense(units=128, activation="relu")(dense_layer_1)
    dense_layer_3 = Dense(units=64, activation="relu")(dense_layer_2)

    y1_output = Dense(units=1, activation="linear", name="y1_output")(dense_layer_2)
    y2_output = Dense(units=1, activation="linear", name="y2_output")(dense_layer_3)

    model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss={'y1_output': 'mse', 'y2_output': 'mse'},
                  metrics={
                      'y1_output': tf.keras.metrics.MeanSquaredError(),
                      'y2_output': tf.keras.metrics.MeanSquaredError(),
                  })
    history = model.fit(X_train, (HR_y_train, R_y_train), epochs=200, batch_size=1024, validation_data=(X_test, (HR_y_test, R_y_test)))

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
    MLP_hitters(H_data)
    
    
    