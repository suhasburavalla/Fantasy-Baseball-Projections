import pandas as pd
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

def hitters_data_read():
# reading in position player data from Fangraphs custom exports

    hitters_15 = pd.read_csv("2015.csv")
    hitters_16 = pd.read_csv("2016.csv")
    hitters_17 = pd.read_csv("2017.csv")
    hitters_18 = pd.read_csv("2018.csv")
    hitters_19 = pd.read_csv("2019.csv")
    hitters_21 = pd.read_csv("2021.csv")
    hitters_22 = pd.read_csv("2022.csv")

    # inner join for only hitters with 200 PA in consecutive seasons

    hitters_15_16 = hitters_15.merge(hitters_16, on='playerid', how='inner')
    hitters_16_17 = hitters_16.merge(hitters_17, on='playerid', how='inner')
    hitters_17_18 = hitters_17.merge(hitters_18, on='playerid', how='inner')
    hitters_18_19 = hitters_18.merge(hitters_19, on='playerid', how='inner')
    hitters_21_22 = hitters_21.merge(hitters_22, on='playerid', how='inner')

    # combine into one hitters_all

    hitters_all = pd.concat([hitters_15_16, hitters_16_17, hitters_17_18, hitters_18_19, hitters_21_22])
    # print(hitters_all.head(5))
    # print(hitters_all.describe())
    return hitters_all

# reading in pitcher (SP and RP) data from Fangraphs custom exports
def pitchers_data_read():
    pitchers_15 = pd.read_csv("P2015.csv")
    pitchers_16 = pd.read_csv("P2016.csv")
    pitchers_17 = pd.read_csv("P2017.csv")
    pitchers_18 = pd.read_csv("P2018.csv")
    pitchers_19 = pd.read_csv("P2019.csv")
    pitchers_21 = pd.read_csv("P2021.csv")
    pitchers_22 = pd.read_csv("P2022.csv")

    # inner join for only pitchers with 50 IP in consecutive seasons

    pitchers_15_16 = pitchers_15.merge(pitchers_16, on='playerid', how='inner')
    pitchers_16_17 = pitchers_16.merge(pitchers_17, on='playerid', how='inner')
    pitchers_17_18 = pitchers_17.merge(pitchers_18, on='playerid', how='inner')
    pitchers_18_19 = pitchers_18.merge(pitchers_19, on='playerid', how='inner')
    pitchers_21_22 = pitchers_21.merge(pitchers_22, on='playerid', how='inner')

    # combine into one hitters_all

    pitchers_all = pd.concat([pitchers_15_16, pitchers_16_17, pitchers_17_18, pitchers_18_19, pitchers_21_22])
    print(pitchers_all.head(5))
    print(pitchers_all.describe())
    return pitchers_all

# print(hitters_all.columns)
# print(pitchers_all.columns)
def hitters_preprocessing(hitters_all):
    
    # drop unnecessary columns
    hitters_all = hitters_all.drop(hitters_all.loc[:, 'Name_y':'3B_y'].columns,axis = 1)
    hitters_all = hitters_all.drop(hitters_all.loc[:, 'BB_y':'GDP_y'].columns,axis = 1)
    hitters_all = hitters_all.drop(hitters_all.loc[:, 'CS_y':'CS_y'].columns,axis = 1)
    hitters_all = hitters_all.drop(hitters_all.loc[:, 'BB%_y':'xwOBA_y'].columns,axis = 1)

    # converting percentages to floats
    hitters_all['BB%_x'] = hitters_all['BB%_x'].str.rstrip("%").astype(float)/100
    hitters_all['K%_x'] = hitters_all['K%_x'].str.rstrip("%").astype(float)/100
    hitters_all['LD%_x'] = hitters_all['LD%_x'].str.rstrip("%").astype(float)/100
    hitters_all['GB%_x'] = hitters_all['GB%_x'].str.rstrip("%").astype(float)/100
    hitters_all['FB%_x'] = hitters_all['FB%_x'].str.rstrip("%").astype(float)/100
    hitters_all['HR/FB_x'] = hitters_all['HR/FB_x'].str.rstrip("%").astype(float)/100
    hitters_all['Swing%_x'] = hitters_all['Swing%_x'].str.rstrip("%").astype(float)/100
    hitters_all['O-Swing%_x'] = hitters_all['O-Swing%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Z-Swing%_x'] = hitters_all['Z-Swing%_x'].str.rstrip("%").astype(float)/100
    hitters_all['O-Contact%_x'] = hitters_all['O-Contact%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Z-Contact%_x'] = hitters_all['Z-Contact%_x'].str.rstrip("%").astype(float)/100
    hitters_all['SwStr%_x'] = hitters_all['SwStr%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Pull%_x'] = hitters_all['Pull%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Cent%_x'] = hitters_all['Cent%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Oppo%_x'] = hitters_all['Oppo%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Soft%_x'] = hitters_all['Soft%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Med%_x'] = hitters_all['Med%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Hard%_x'] = hitters_all['Hard%_x'].str.rstrip("%").astype(float)/100
    hitters_all['Barrel%_x'] = hitters_all['Barrel%_x'].str.rstrip("%").astype(float)/100
    hitters_all['HardHit%_x'] = hitters_all['HardHit%_x'].str.rstrip("%").astype(float)/100
    hitters_all['CSW%_x'] = hitters_all['CSW%_x'].str.rstrip("%").astype(float)/100

    print(hitters_all.info())
    
    return hitters_all

# drop unnecessary columns (pitcher data)
def pitchers_preprocessing(pitchers_all) :
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'Name_y':'Age_y'].columns,axis = 1)
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'L_y':'L_y'].columns,axis = 1)
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'G_y':'GS_y'].columns,axis = 1)
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'BS_y':'BK_y'].columns,axis = 1)
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'GB_y':'AVG_y'].columns,axis = 1)
    pitchers_all = pitchers_all.drop(pitchers_all.loc[:, 'BABIP_y':'xERA_y'].columns,axis = 1)

    # converting percentages to floats (pitcher data)

    pitchers_all['LOB%_x'] = pitchers_all['LOB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['LD%_x'] = pitchers_all['LD%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['GB%_x'] = pitchers_all['GB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['FB%_x'] = pitchers_all['FB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['IFFB%_x'] = pitchers_all['IFFB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['HR/FB_x'] = pitchers_all['HR/FB_x'].str.rstrip("%").astype(float)/100
    pitchers_all['FB%.1_x'] = pitchers_all['FB%.1_x'].str.rstrip("%").astype(float)/100
    pitchers_all['SL%_x'] = pitchers_all['SL%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['CT%_x'] = pitchers_all['CT%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['CB%_x'] = pitchers_all['CB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['CH%_x'] = pitchers_all['CH%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['SF%_x'] = pitchers_all['SF%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['KN%_x'] = pitchers_all['KN%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['O-Swing%_x'] = pitchers_all['O-Swing%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Z-Swing%_x'] = pitchers_all['Z-Swing%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Swing%_x'] = pitchers_all['Swing%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['O-Contact%_x'] = pitchers_all['O-Contact%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Z-Contact%_x'] = pitchers_all['Z-Contact%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Contact%_x'] = pitchers_all['Contact%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Zone%_x'] = pitchers_all['Zone%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['F-Strike%_x'] = pitchers_all['F-Strike%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['SwStr%_x'] = pitchers_all['SwStr%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['K%_x'] = pitchers_all['K%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['BB%_x'] = pitchers_all['BB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['K-BB%_x'] = pitchers_all['K-BB%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Pull%_x'] = pitchers_all['Pull%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Cent%_x'] = pitchers_all['Cent%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Oppo%_x'] = pitchers_all['Oppo%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Soft%_x'] = pitchers_all['Soft%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Med%_x'] = pitchers_all['Med%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Hard%_x'] = pitchers_all['Hard%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['Barrel%_x'] = pitchers_all['Barrel%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['HardHit%_x'] = pitchers_all['HardHit%_x'].str.rstrip("%").astype(float)/100
    pitchers_all['CSW%_x'] = pitchers_all['CSW%_x'].str.rstrip("%").astype(float)/100

    print(pitchers_all.info())


    #convert all nan values to mean for pitchers_all
    pitchers_all = pitchers_all.fillna(0.0000000001)

    print(pitchers_all.info())
    
    return pitchers_all
# Visualizations

# IN PROGRESS (for first stand-up meeting)

sns.set_style('whitegrid')

# is Barrel% more predictive of HR_y than HR_x?
# sns.lmplot(x ='Barrel%_x', y ='HR_y', data = hitters_all)
# sns.lmplot(x ='HR_x', y ='HR_y', data = hitters_all)
# plt.show()

# Seaborn heatmaps are broken, will try something else
# sns.heatmap(data=hitters_all['HR_x', 'SLG_x', 'ISO_x', 'EV_x', 'LA_x', 'maxEV_x', 'Barrel%_x', 'xSLG_x', 'HR_y'])
# plt.show()

# PCR




def pcr_hitters(hitters_all, Pipeline, LinearRegression, PCA, mean_squared_error, np):
    X = hitters_all.iloc[:, 2:56].values
    print(X.shape)

    print("PCR Analysis: Hitters\n")

    pca = PCA(n_components=4)
    reg = LinearRegression()
    Pipeline = Pipeline(steps=[('pca', pca), ('reg', reg)])

    for i in range(0,5): #iterate through hitter targets in order: HR, R, RBI, SB, AVG

        y = hitters_all.iloc[:, 58+i].values

        Pipeline.fit(X, y)

        #predict labels
        y_pred = Pipeline.predict(X)

        #metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = Pipeline.score(X, y)

        print(f'MSE: {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'R-Squared: {r2:.2f}')

def pcr_pitchers(pitchers_all, Pipeline, LinearRegression, PCA, mean_squared_error, np) : 
    X = pitchers_all.iloc[:, 3:76].values

    print("PCR Analysis: Pitchers\n")
    pca = PCA(n_components=4)
    reg = LinearRegression()
    Pipeline = Pipeline(steps=[('pca', pca), ('reg', reg)])
    for i in range(0,5): #iterate through pitcher targets in order: W, SV, SO, ERA, WHIP
        y = pitchers_all.iloc[:, 78+i].values
        
        Pipeline.fit(X, y)
        
        y_pred = Pipeline.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = Pipeline.score(X, y)

        print(f'MSE: {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'R-Squared: {r2:.2f}')
        #plot heatmap of PCA components
        
        

def hitters_visualization(hitters_all, plt, sns) : 
    # plot seaborn heatmap for all targets of hitters_all
    
    corr_matrix = hitters_all.corr()
    
    
    print(corr_matrix["AVG_y"].sort_values(ascending=False))
    
    # plt.matshow(hitters_all.corr())
    # plt.show()
    f = plt.figure(figsize=(20, 10))
    plt.matshow(hitters_all.corr(), fignum=f.number)
    plt.xticks(range(hitters_all.select_dtypes(['number']).shape[1]), hitters_all.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(hitters_all.select_dtypes(['number']).shape[1]), hitters_all.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

    # sns.heatmap(corr_matrix, annot=True)
    # plt.show()

def pitchers_visualization(pitchers_all, plt, sns) : 
    # plot seaborn heatmap for all targets of hitters_all
    
    corr_matrix = pitchers_all.corr()
    print(corr_matrix["WHIP_y"].sort_values(ascending=False))
    
    # plt.matshow(hitters_all.corr())
    # plt.show()
    f = plt.figure(figsize=(20,10))
    plt.matshow(pitchers_all.corr(), fignum=f.number)
    plt.xticks(range(pitchers_all.select_dtypes(['number']).shape[1]), pitchers_all.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(pitchers_all.select_dtypes(['number']).shape[1]), pitchers_all.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

    # sns.heatmap(corr_matrix, annot=True)
    # plt.show()


# # export updated .csv files at the end of script
# hitters_all.to_csv('hitters_all.csv')
# pitchers_all.to_csv('pitchers_all.csv')

def pcr_hitters_normalized(hitters_all, LinearRegression, PCA, np):

    print("PCR Analysis with Normalization: Hitters\n")

    X = hitters_all.iloc[:, 2:56].values

    pca = PCA()
    X_scaled = pca.fit_transform(scale(X))

    for i in range(0,5): #iterate through hitter targets in order: HR, R, RBI, SB, AVG

        y = hitters_all.iloc[:, 58+i].values
        cv = RepeatedKFold(n_splits=20, n_repeats=5, random_state=1)
        reg = LinearRegression()
        mse = []
        score = -1 * model_selection.cross_val_score(reg, np.ones((len(X_scaled), 1)), y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)

        for i in np.arange(1, 6):
            score = -1 * model_selection.cross_val_score(reg, X_scaled[:, :i], y, cv=cv, scoring='neg_mean_squared_error').mean()
            mse.append(score)

        # Plot cross-validation results
        plt.plot(mse)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('MSE')
        plt.show()

def pcr_pitchers_normalized(pitchers_all, LinearRegression, PCA, np):

    print("PCR Analysis with Normalization: Pitchers\n")

    X = pitchers_all.iloc[:, 3:76].values

    pca = PCA()
    X_scaled = pca.fit_transform(scale(X))

    for i in range(0,5): #iterate through hitter targets in order: HR, R, RBI, SB, AVG

        y = pitchers_all.iloc[:, 78+i].values
        cv = RepeatedKFold(n_splits=20, n_repeats=5, random_state=1)
        reg = LinearRegression()
        mse = []
        score = -1 * model_selection.cross_val_score(reg, np.ones((len(X_scaled), 1)), y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)

        for i in np.arange(1, 6):
            score = -1 * model_selection.cross_val_score(reg, X_scaled[:, :i], y, cv=cv, scoring='neg_mean_squared_error').mean()
            mse.append(score)

        # Plot cross-validation results
        plt.plot(mse)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('MSE')
        plt.show()

def rf_hitters(hitters_all):

    X = hitters_all.iloc[:, 2:56].values
    y = hitters_all.iloc[:, 58:62].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, shuffle=True)

    regressor = DecisionTreeRegressor(random_state=0, max_features='sqrt')
    print(cross_val_score(regressor, X_train, y_train, cv=10))

def rf_pitchers(pitchers_all):

    X = pitchers_all.iloc[:, 3:76].values
    y = pitchers_all.iloc[:, 78:82].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, shuffle=True)

    regressor = DecisionTreeRegressor(random_state=0, max_features='sqrt')
    print(cross_val_score(regressor, X_train, y_train, cv=10))

if __name__ == "__main__" :
    hitters_all = hitters_data_read()
    pitchers_all = pitchers_data_read()
    hitters_all = hitters_preprocessing(hitters_all)
    pitchers_all = pitchers_preprocessing(pitchers_all)
    pcr_hitters(hitters_all, Pipeline, LinearRegression, PCA, mean_squared_error, np)
    pcr_pitchers(pitchers_all, Pipeline, LinearRegression, PCA, mean_squared_error, np)
    # pcr_hitters_normalized(hitters_all, LinearRegression, PCA, np)
    # pcr_pitchers_normalized(pitchers_all, LinearRegression, PCA, np)
    rf_hitters(hitters_all)
    rf_pitchers(pitchers_all)
    # hitters_visualization(hitters_all, plt, sns)
    # pitchers_visualization(pitchers_all, plt, sns)