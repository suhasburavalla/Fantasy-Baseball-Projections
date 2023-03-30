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

def hitters_data_read_new():

    # read in custom Fangraphs exports

    H2008 = pd.read_csv("2008 to 2022 Data/H2008.csv")
    H2009 = pd.read_csv("2008 to 2022 Data/H2009.csv")
    H2010 = pd.read_csv("2008 to 2022 Data/H2010.csv")
    H2011 = pd.read_csv("2008 to 2022 Data/H2011.csv")
    H2012 = pd.read_csv("2008 to 2022 Data/H2012.csv")
    H2013 = pd.read_csv("2008 to 2022 Data/H2013.csv")
    H2014 = pd.read_csv("2008 to 2022 Data/H2014.csv")
    H2015 = pd.read_csv("2008 to 2022 Data/H2015.csv")
    H2016 = pd.read_csv("2008 to 2022 Data/H2016.csv")
    H2017 = pd.read_csv("2008 to 2022 Data/H2017.csv")
    H2018 = pd.read_csv("2008 to 2022 Data/H2018.csv")
    H2019 = pd.read_csv("2008 to 2022 Data/H2019.csv")
    H2020 = pd.read_csv("2008 to 2022 Data/H2020.csv")
    H2021 = pd.read_csv("2008 to 2022 Data/H2021.csv")
    H2022 = pd.read_csv("2008 to 2022 Data/H2022.csv")

    H1 = H2008.merge(H2009, on='playerid', how='inner')
    H2 = H2009.merge(H2010, on='playerid', how='inner')
    H3 = H2010.merge(H2011, on='playerid', how='inner')
    H4 = H2011.merge(H2012, on='playerid', how='inner')
    H5 = H2012.merge(H2013, on='playerid', how='inner')
    H6 = H2013.merge(H2014, on='playerid', how='inner')
    H7 = H2014.merge(H2015, on='playerid', how='inner')
    H8 = H2015.merge(H2016, on='playerid', how='inner')
    H9 = H2016.merge(H2017, on='playerid', how='inner')
    H10 = H2017.merge(H2018, on='playerid', how='inner')
    H11 = H2018.merge(H2019, on='playerid', how='inner')
    H12 = H2019.merge(H2020, on='playerid', how='inner')
    H13 = H2020.merge(H2021, on='playerid', how='inner')
    H14 = H2021.merge(H2022, on='playerid', how='inner')

    H_data = pd.concat([H1,H2,H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13,H14])

    print(H_data.head(5))
    print(H_data.describe())

    H_data.to_csv("H_data.csv")

    return H_data

def pitchers_data_read_new():

    # read in custom Fangraphs exports

    P2008 = pd.read_csv("2008 to 2022 Data/P2008.csv")
    P2009 = pd.read_csv("2008 to 2022 Data/P2009.csv")
    P2010 = pd.read_csv("2008 to 2022 Data/P2010.csv")
    P2011 = pd.read_csv("2008 to 2022 Data/P2011.csv")
    P2012 = pd.read_csv("2008 to 2022 Data/P2012.csv")
    P2013 = pd.read_csv("2008 to 2022 Data/P2013.csv")
    P2014 = pd.read_csv("2008 to 2022 Data/P2014.csv")
    P2015 = pd.read_csv("2008 to 2022 Data/P2015.csv")
    P2016 = pd.read_csv("2008 to 2022 Data/P2016.csv")
    P2017 = pd.read_csv("2008 to 2022 Data/P2017.csv")
    P2018 = pd.read_csv("2008 to 2022 Data/P2018.csv")
    P2019 = pd.read_csv("2008 to 2022 Data/P2019.csv")
    P2020 = pd.read_csv("2008 to 2022 Data/P2020.csv")
    P2021 = pd.read_csv("2008 to 2022 Data/P2021.csv")
    P2022 = pd.read_csv("2008 to 2022 Data/P2022.csv")

    P1 = P2008.merge(P2009, on='playerid', how='inner')
    P2 = P2009.merge(P2010, on='playerid', how='inner')
    P3 = P2010.merge(P2011, on='playerid', how='inner')
    P4 = P2011.merge(P2012, on='playerid', how='inner')
    P5 = P2012.merge(P2013, on='playerid', how='inner')
    P6 = P2013.merge(P2014, on='playerid', how='inner')
    P7 = P2014.merge(P2015, on='playerid', how='inner')
    P8 = P2015.merge(P2016, on='playerid', how='inner')
    P9 = P2016.merge(P2017, on='playerid', how='inner')
    P10 = P2017.merge(P2018, on='playerid', how='inner')
    P11 = P2018.merge(P2019, on='playerid', how='inner')
    P12 = P2019.merge(P2020, on='playerid', how='inner')
    P13 = P2020.merge(P2021, on='playerid', how='inner')
    P14 = P2021.merge(P2022, on='playerid', how='inner')

    P_data = pd.concat([P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14])

    print(P_data.head(5))
    print(P_data.describe())

    P_data.to_csv("P_data.csv")

    return P_data

def hitters_preprocessing_new(H_data):

    H_data = H_data.drop(H_data.loc[:, 'playerid':'PA_y'].columns,axis = 1)
    H_data = H_data.drop(H_data.loc[:, 'BB%_y':'BABIP_y'].columns, axis=1)
    H_data = H_data.drop(H_data.loc[:, 'OBP_y':'xSLG_y'].columns, axis=1)

    H_data['BB%_x'] = H_data['BB%_x'].str.rstrip("%").astype(float) / 100
    H_data['K%_x'] = H_data['K%_x'].str.rstrip("%").astype(float) / 100
    H_data['LD%_x'] = H_data['LD%_x'].str.rstrip("%").astype(float) / 100
    H_data['GB%_x'] = H_data['GB%_x'].str.rstrip("%").astype(float) / 100
    H_data['FB%_x'] = H_data['FB%_x'].str.rstrip("%").astype(float) / 100
    H_data['HR/FB_x'] = H_data['HR/FB_x'].str.rstrip("%").astype(float) / 100
    H_data['Swing%_x'] = H_data['Swing%_x'].str.rstrip("%").astype(float) / 100
    H_data['O-Swing%_x'] = H_data['O-Swing%_x'].str.rstrip("%").astype(float) / 100
    H_data['Z-Swing%_x'] = H_data['Z-Swing%_x'].str.rstrip("%").astype(float) / 100
    H_data['O-Contact%_x'] = H_data['O-Contact%_x'].str.rstrip("%").astype(float) / 100
    H_data['Z-Contact%_x'] = H_data['Z-Contact%_x'].str.rstrip("%").astype(float) / 100
    H_data['SwStr%_x'] = H_data['SwStr%_x'].str.rstrip("%").astype(float) / 100
    H_data['Barrel%_x'] = H_data['Barrel%_x'].str.rstrip("%").astype(float) / 100
    H_data['HardHit%_x'] = H_data['HardHit%_x'].str.rstrip("%").astype(float) / 100
    H_data['CSW%_x'] = H_data['CSW%_x'].str.rstrip("%").astype(float) / 100
    H_data['IFFB%_x'] = H_data['IFFB%_x'].str.rstrip("%").astype(float) / 100
    H_data['IFH%_x'] = H_data['IFH%_x'].str.rstrip("%").astype(float) / 100
    H_data['Contact%_x'] = H_data['Contact%_x'].str.rstrip("%").astype(float) / 100
    H_data['Zone%_x'] = H_data['Zone%_x'].str.rstrip("%").astype(float) / 100

    H_data.to_csv("H_data.csv")

    return H_data

def pitchers_preprocessing_new(P_data):

    P_data = P_data.drop(P_data.loc[:, 'playerid':'Team_y'].columns, axis=1)
    P_data = P_data.drop(P_data.loc[:, 'L_y':'L_y'].columns, axis=1)
    P_data = P_data.drop(P_data.loc[:, 'G_y':'HR/FB_y'].columns, axis=1)
    P_data = P_data.drop(P_data.loc[:, 'xERA_y':'WP_y'].columns, axis=1)
    P_data = P_data.drop(P_data.loc[:, 'GB_y':'AVG_y'].columns, axis=1)
    P_data = P_data.drop(P_data.loc[:, 'BABIP.1_y':'CSW%_y'].columns, axis=1)

    P_data['LOB%_x'] = P_data['LOB%_x'].str.rstrip("%").astype(float) / 100
    P_data['LD%_x'] = P_data['LD%_x'].str.rstrip("%").astype(float) / 100
    P_data['GB%_x'] = P_data['GB%_x'].str.rstrip("%").astype(float) / 100
    P_data['FB%_x'] = P_data['FB%_x'].str.rstrip("%").astype(float) / 100
    P_data['IFFB%_x'] = P_data['IFFB%_x'].str.rstrip("%").astype(float) / 100
    P_data['HR/FB_x'] = P_data['HR/FB_x'].str.rstrip("%").astype(float) / 100
    P_data['FB%.1_x'] = P_data['FB%.1_x'].str.rstrip("%").astype(float) / 100
    P_data['SL%_x'] = P_data['SL%_x'].str.rstrip("%").astype(float) / 100
    P_data['CT%_x'] = P_data['CT%_x'].str.rstrip("%").astype(float) / 100
    P_data['CB%_x'] = P_data['CB%_x'].str.rstrip("%").astype(float) / 100
    P_data['CH%_x'] = P_data['CH%_x'].str.rstrip("%").astype(float) / 100
    P_data['SF%_x'] = P_data['SF%_x'].str.rstrip("%").astype(float) / 100
    P_data['O-Swing%_x'] = P_data['O-Swing%_x'].str.rstrip("%").astype(float) / 100
    P_data['Z-Swing%_x'] = P_data['Z-Swing%_x'].str.rstrip("%").astype(float) / 100
    P_data['Swing%_x'] = P_data['Swing%_x'].str.rstrip("%").astype(float) / 100
    P_data['O-Contact%_x'] = P_data['O-Contact%_x'].str.rstrip("%").astype(float) / 100
    P_data['Z-Contact%_x'] = P_data['Z-Contact%_x'].str.rstrip("%").astype(float) / 100
    P_data['Contact%_x'] = P_data['Contact%_x'].str.rstrip("%").astype(float) / 100
    P_data['Zone%_x'] = P_data['Zone%_x'].str.rstrip("%").astype(float) / 100
    P_data['SwStr%_x'] = P_data['SwStr%_x'].str.rstrip("%").astype(float) / 100
    P_data['K%_x'] = P_data['K%_x'].str.rstrip("%").astype(float) / 100
    P_data['BB%_x'] = P_data['BB%_x'].str.rstrip("%").astype(float) / 100
    P_data['Barrel%_x'] = P_data['Barrel%_x'].str.rstrip("%").astype(float) / 100
    P_data['HardHit%_x'] = P_data['HardHit%_x'].str.rstrip("%").astype(float) / 100
    P_data['CSW%_x'] = P_data['CSW%_x'].str.rstrip("%").astype(float) / 100
    P_data['LOB%.1_x'] = P_data['LOB%.1_x'].str.rstrip("%").astype(float) / 100
    P_data['GB%.1_x'] = P_data['GB%.1_x'].str.rstrip("%").astype(float) / 100
    P_data['HR/FB.1_x'] = P_data['HR/FB.1_x'].str.rstrip("%").astype(float) / 100

    P_data.to_csv("P_data.csv")

    return P_data

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

    for i in range(0,5): #iterate through hitter targets in order: HR, R, RBI, SB, AVG

        X_scaled = pca.fit_transform(scale(X))

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

    for i in range(0,5): #iterate through hitter targets in order: HR, R, RBI, SB, AVG

        X_scaled = pca.fit_transform(scale(X))

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
    
    print("\n Random Forest Analysis: Hitters\n")

    X = hitters_all.iloc[:, 2:56].values
    y = hitters_all.iloc[:, 58:62].values
    col_names = hitters_all.columns[2:56].tolist()

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

def rf_pitchers(pitchers_all):

    print("\n Random Forest Analysis: Pitchers\n")
    X = pitchers_all.iloc[:, 3:76].values
    y = pitchers_all.iloc[:, 78:82].values
    col_names = pitchers_all.columns[3:76].tolist()

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
    # hitters_all = hitters_data_read()
    # pitchers_all = pitchers_data_read()
    # hitters_all = hitters_preprocessing(hitters_all)
    # pitchers_all = pitchers_preprocessing(pitchers_all)
    # pcr_hitters(hitters_all, Pipeline, LinearRegression, PCA, mean_squared_error, np)
    # pcr_pitchers(pitchers_all, Pipeline, LinearRegression, PCA, mean_squared_error, np)
    # pcr_hitters_normalized(hitters_all, LinearRegression, PCA, np)
    # pcr_pitchers_normalized(pitchers_all, LinearRegression, PCA, np)
    # rf_hitters(hitters_all)
    # rf_pitchers(pitchers_all)
    # hitters_visualization(hitters_all, plt, sns)
    # pitchers_visualization(pitchers_all, plt, sns)
    H_data = hitters_data_read_new()
    P_data = pitchers_data_read_new()
    H_data = hitters_preprocessing_new(H_data)
    P_data = pitchers_preprocessing_new(P_data)