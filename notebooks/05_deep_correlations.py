import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.competition_download('titanic')
df = pd.read_csv(path + '/train.csv')

# ---- 1. Extract title from name ----
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
    'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
    'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
    'Jonkheer': 'Rare', 'Dona': 'Rare'
})

# ---- 2. Family size ----
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ---- 3. Has cabin ----
df['HasCabin'] = df['Cabin'].notna().astype(int)

# ---- 4. Sex as number ----
df['IsFemale'] = (df['Sex'] == 'female').astype(int)

# ---- 5. Age - fill missing with median ----
df['Age'] = df['Age'].fillna(df['Age'].median())

# ---- SCORECARD - correlation with Survived ----
features = ['IsFemale', 'Pclass', 'Fare', 'Age', 
            'FamilySize', 'IsAlone', 'HasCabin', 'SibSp', 'Parch']

print("=== CORRELATION SCORECARD ===")
print("(closer to 1.0 or -1.0 = stronger predictor)")
print()
correlations = df[features + ['Survived']].corr()['Survived'].drop('Survived')
correlations = correlations.abs().sort_values(ascending=False)
for feature, score in correlations.items():
    bar = '█' * int(score * 20)
    print(f"{feature:<15} {score:.3f}  {bar}")

# ---- 6. Age deep dive ----
print()
print("=== AGE ANALYSIS ===")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], 
                         labels=['Child','Teen','YoungAdult','Adult','Senior'])
print(df.groupby('AgeGroup')['Survived'].mean().round(2))

# ---- 7. Title analysis ----
print()
print("=== SURVIVAL BY TITLE ===")
print(df.groupby('Title')['Survived'].mean().round(2))

# ---- 8. Family size analysis ----
print()
print("=== SURVIVAL BY FAMILY SIZE ===")
print(df.groupby('FamilySize')['Survived'].mean().round(2))

# ---- 9. Visualise top findings ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Age groups
df.groupby('AgeGroup')['Survived'].mean().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Survival by Age Group')
axes[0].set_ylabel('Survival Rate')
axes[0].tick_params(axis='x', rotation=45)

# Title
df.groupby('Title')['Survived'].mean().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Survival by Title')
axes[1].tick_params(axis='x', rotation=45)

# Family size
df.groupby('FamilySize')['Survived'].mean().plot(kind='bar', ax=axes[2], color='green')
axes[2].set_title('Survival by Family Size')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#-------10. redefining the family unit

df['FamilyCategory'] = pd.cut(df['FamilySize'], 
    bins=[0,1,4,20], 
    labels=['Alone', 'SmallFamily', 'LargeFamily'])

print("=== SURVIVAL BY FAMILY CATEGORY ===")
print(df.groupby('FamilyCategory')['Survived'].mean().round(2))
print()

# ---- 11. Rare titles broken out ----
df['TitleFull'] = df['Name'].str.extract(r',\s*([^\.]+)\.')

print("=== ALL TITLES - SURVIVAL RATE AND COUNT ===")
title_stats = df.groupby('TitleFull')['Survived'].agg(['mean', 'count']).round(2)
title_stats.columns = ['SurvivalRate', 'Count']
title_stats = title_stats.sort_values('SurvivalRate', ascending=False)
print(title_stats.to_string())