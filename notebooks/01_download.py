import kagglehub

path = kagglehub.competition_download('titanic')
print("Path to competition files:", path)

import pandas as pd

df = pd.read_csv(path + '/train.csv')
print(df.head())
print(df.shape)