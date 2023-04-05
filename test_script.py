import pandas as pd 

df = pd.read_csv("P_data.csv")
#print nan values
X = pd.DataFrame(df.iloc[:, 67:72])
print(X.columns)




