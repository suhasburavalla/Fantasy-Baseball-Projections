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
    

    
    
if __name__ == "__main__" :

    # H_data = hitters_data_read_new()
    # P_data = pitchers_data_read_new()
    # H_data = hitters_preprocessing_new(H_data)
    # P_data = pitchers_preprocessing_new(P_data)
    
    H_data = hitters_csv_new()
    P_data = pitchers_csv_new()
    # hitters_rf(H_data)
    pitchers_rf(P_data)
    
    
    