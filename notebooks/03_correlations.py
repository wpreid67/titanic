import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.competition_download('titanic')
df = pd.read_csv(path + '/train.csv')

# 1. Survival by Sex
df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['lightblue', 'pink'])
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

# 2. Survival by Passenger Class
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color=['gold', 'silver', 'brown'])
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks([0,1,2], ['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.show()

# 3. Age distribution - survivors vs died
df[df['Survived']==1]['Age'].plot(kind='hist', alpha=0.5, label='Survived', color='green')
df[df['Survived']==0]['Age'].plot(kind='hist', alpha=0.5, label='Died', color='red')
plt.title('Age Distribution: Survived vs Died')
plt.xlabel('Age')
plt.legend()
plt.show()

# 4. Correlation numbers
print(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()['Survived'])

# 5. Survival by Embarked port
df.groupby('Embarked')['Survived'].mean().plot(kind='bar', color=['lightblue', 'lightgreen', 'salmon'])
plt.title('Survival Rate by Port of Embarkation')
plt.ylabel('Survival Rate')
plt.xticks([0,1,2], ['Cherbourg', 'Queenstown', 'Southampton'], rotation=0)
plt.show()

# 6. Survival by whether they had a cabin number or not
df['Has_Cabin'] = df['Cabin'].notna()
df.groupby('Has_Cabin')['Survived'].mean().plot(kind='bar', color=['salmon', 'lightgreen'])
plt.title('Survival Rate: Had Cabin Number vs Not')
plt.ylabel('Survival Rate')
plt.xticks([0,1], ['No Cabin', 'Had Cabin'], rotation=0)
plt.show()

# 7. Survival by Pclass + Embarked combined
pivot = df.groupby(['Pclass', 'Embarked'])['Survived'].mean().unstack()
pivot.plot(kind='bar', color=['lightblue', 'lightgreen', 'salmon'])
plt.title('Survival Rate by Passenger Class and Embarked Port')
plt.ylabel('Survival Rate')
plt.xticks([0,1,2], ['1st Class', '2nd Class', '3rd Class'], rotation=0)
plt.legend(['Cherbourg', 'Queenstown', 'Southampton'])
plt.show()