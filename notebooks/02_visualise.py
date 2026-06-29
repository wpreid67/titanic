import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.competition_download('titanic')
df = pd.read_csv(path + '/train.csv')

# 1. How many survived vs died
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Survived vs Died')
plt.xticks([0, 1], ['Died', 'Survived'], rotation=0)
plt.ylabel('Count')
plt.show()