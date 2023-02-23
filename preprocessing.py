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

# hitters_all.to_csv('hitters_all.csv')
# pitchers_all.to_csv('pitchers_all.csv')

hitters_all['Barrel%_x'] = hitters_all['Barrel%_x'].str.rstrip("%").astype(float)/100

# Visualizations

sns.set_style('whitegrid')

sns.lmplot(x ='Barrel%_x', y ='HR_y', data = hitters_all)
sns.lmplot(x ='HR_x', y ='HR_y', data = hitters_all)
plt.show()

# sns.heatmap(data=hitters_all['HR_x', 'SLG_x', 'ISO_x', 'EV_x', 'LA_x', 'maxEV_x', 'Barrel%_x', 'xSLG_x', 'HR_y'])
# plt.show()
