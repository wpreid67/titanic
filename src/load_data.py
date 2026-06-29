import pandas as pd
import kagglehub

def load_titanic():
    path = kagglehub.competition_download('titanic')
    df = pd.read_csv(path + '/train.csv')
    return df