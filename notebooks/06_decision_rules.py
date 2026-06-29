import pandas as pd
import kagglehub

path = kagglehub.competition_download('titanic')
df = pd.read_csv(path + '/train.csv')

# ---- Setup ----
df['TitleFull'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['HasCabin'] = df['Cabin'].notna()
df['Age'] = df['Age'].fillna(df['Age'].median())
df['AgeGroup'] = df['Age'].apply(lambda x: 'Child' if x < 12 else 'Adult')

# Start with no prediction
df['Prediction'] = None
df['Rule'] = None

# ---- RULE 1: Title guarantees survival ----
survived_titles = ['Lady', 'the Countess', 'Mlle', 'Sir', 'Ms', 'Mme']
mask = df['TitleFull'].isin(survived_titles) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 1
df.loc[mask, 'Rule'] = 'Rule 1 - Title guarantees survival'

# ---- RULE 2: Title guarantees death ----
died_titles = ['Capt', 'Rev']
mask = df['TitleFull'].isin(died_titles) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 0
df.loc[mask, 'Rule'] = 'Rule 2 - Title guarantees death'

# ---- RULE 3: Large family + 3rd class = died ----
mask = (df['FamilySize'] >= 5) & (df['Pclass'] == 3) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 0
df.loc[mask, 'Rule'] = 'Rule 3 - Large family 3rd class died'

# ---- RULE 4: Female in small family ----
mask = (df['Sex'] == 'female') & (df['FamilySize'].between(2,4)) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 1
df.loc[mask, 'Rule'] = 'Rule 4 - Female in small family'

# ---- RULE 5: Female alone ----
mask = (df['Sex'] == 'female') & (df['FamilySize'] == 1) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 1
df.loc[mask, 'Rule'] = 'Rule 5 - Female alone'

# ---- RULE 6: Male child ----
mask = (df['Sex'] == 'male') & (df['AgeGroup'] == 'Child') & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 1
df.loc[mask, 'Rule'] = 'Rule 6 - Male child'

# ---- RULE 7: Male not in small family ----
mask = (df['Sex'] == 'male') & (~df['FamilySize'].between(2,4)) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 0
df.loc[mask, 'Rule'] = 'Rule 7 - Male not in small family'

# ---- RULE 8: Male no cabin ----
mask = (df['Sex'] == 'male') & (~df['HasCabin']) & df['Prediction'].isna()
df.loc[mask, 'Prediction'] = 0
df.loc[mask, 'Rule'] = 'Rule 8 - Male no cabin'

# ---- RESULTS ----
print("=== PREDICTIONS BY RULE ===")
print(df.groupby('Rule')['Rule'].count().to_string())
print()

print("=== ACCURACY BY RULE ===")
for rule in sorted(df['Rule'].dropna().unique()):
    subset = df[df['Rule'] == rule]
    accuracy = (subset['Prediction'] == subset['Survived']).mean()
    print(f"{rule:<45} accuracy: {accuracy:.2f}  ({len(subset)} passengers)")

print()
print("=== UNTAGGED PASSENGERS ===")
untagged = df[df['Prediction'].isna()]
print("Count:", len(untagged))
print()
print(untagged[['Name', 'Sex', 'Age', 'Pclass', 'FamilySize', 'HasCabin', 'Survived']].to_string())

print()
print("=== OVERALL ACCURACY ===")
tagged = df[df['Prediction'].notna()]
accuracy = (tagged['Prediction'] == tagged['Survived']).mean()
print(f"Accuracy on tagged passengers: {accuracy:.2f} ({len(tagged)} of {len(df)} passengers)")

# ---- INVESTIGATE HARD GROUP ----
print("=== 1ST CLASS MALES IN SMALL FAMILIES ===")
hard = df[(df['Sex'] == 'male') & 
          (df['Pclass'] == 1) & 
          (df['FamilySize'].between(2,4)) & 
          (df['HasCabin'] == True)]

print("Survived:", hard['Survived'].sum())
print("Died:", (hard['Survived'] == 0).sum())
print("Survival rate:", hard['Survived'].mean().round(2))
print()
print(hard[['Name', 'Age', 'FamilySize', 'Fare', 'Survived']].sort_values('Survived', ascending=False).to_string())

# ---- CHECK MISSED CHILDREN ----
print("=== CHECKING MISSED CHILDREN ===")
missed = df[df['Name'].str.contains('Dodge|Allison|Carter, Master')]
print(missed[['Name', 'Age', 'TitleFull', 'FamilySize', 'Pclass', 'Rule', 'Prediction', 'Survived']].to_string())

