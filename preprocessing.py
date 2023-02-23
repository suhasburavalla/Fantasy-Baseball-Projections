import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

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

# combine into one df

hitters_all = pd.concat([hitters_15_16, hitters_16_17, hitters_17_18, hitters_18_19, hitters_21_22])
print(hitters_all.head(5))
print(hitters_all.describe())

# reading in pitcher (SP and RP) data from Fangraphs custom exports

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

# combine into one df

pitchers_all = pd.concat([pitchers_15_16, pitchers_16_17, pitchers_17_18, pitchers_18_19, pitchers_21_22])
print(pitchers_all.head(5))
print(pitchers_all.describe())

print(hitters_all.columns)
print(pitchers_all.columns)

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

# TO-DO: need to do the same for pitcher data

# Visualizations

sns.set_style('whitegrid')

# is Barrel% more predictive of HR_y than HR_x?
sns.lmplot(x ='Barrel%_x', y ='HR_y', data = hitters_all)
sns.lmplot(x ='HR_x', y ='HR_y', data = hitters_all)
plt.show()

# Seaborn heatmaps are broken, will try something else
# sns.heatmap(data=hitters_all['HR_x', 'SLG_x', 'ISO_x', 'EV_x', 'LA_x', 'maxEV_x', 'Barrel%_x', 'xSLG_x', 'HR_y'])
# plt.show()

# export updated .csv files at the end of script
hitters_all.to_csv('hitters_all.csv')
pitchers_all.to_csv('pitchers_all.csv')
