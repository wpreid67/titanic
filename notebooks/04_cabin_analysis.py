import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

path = kagglehub.competition_download('titanic')
df = pd.read_csv(path + '/train.csv')

# ---- 1. How many have cabin numbers vs not ----
print("=== CABIN OVERVIEW ===")
print("Total passengers:", len(df))
print("Has cabin:", df['Cabin'].notna().sum())
print("No cabin:", df['Cabin'].isna().sum())
print()

# ---- 2. Class breakdown of no cabin ----
print("=== CLASS BREAKDOWN - NO CABIN ===")
print(df[df['Cabin'].isna()]['Pclass'].value_counts())
print()

# ---- 3. Survival rate by cabin vs no cabin ----
print("=== SURVIVAL RATE ===")
print("With cabin:", df[df['Cabin'].notna()]['Survived'].mean().round(2))
print("No cabin:", df[df['Cabin'].isna()]['Survived'].mean().round(2))
print()

# ---- 4. The mystery 1st class with no cabin ----
print("=== 1ST CLASS WITH NO CABIN ===")
first_no_cabin = df[(df['Pclass'] == 1) & (df['Cabin'].isna())]
print("Count:", len(first_no_cabin))
print(first_no_cabin[['Name', 'Sex', 'Age', 'Fare', 'Survived']].to_string())
print()

# ---- 5. Extract deck from cabin letter ----
df['Deck'] = df['Cabin'].str[0]
print("=== SURVIVAL RATE BY DECK ===")
print(df.groupby('Deck')['Survived'].mean().round(2))
print()

# ---- 6. Visualise survival by deck ----
df.groupby('Deck')['Survived'].mean().plot(kind='bar', color='steelblue')
plt.title('Survival Rate by Deck')
plt.ylabel('Survival Rate')
plt.xlabel('Deck')
plt.xticks(rotation=0)
plt.show()

# ---- 7. Overall survival rate comparison ----
print("=== OVERALL SURVIVAL RATES ===")
print("Overall survival rate:", df['Survived'].mean().round(2))
print("Survival rate WITH cabin:", df[df['Cabin'].notna()]['Survived'].mean().round(2))
print("Survival rate WITHOUT cabin:", df[df['Cabin'].isna()]['Survived'].mean().round(2))
print()
print("Passengers with cabin data:", df['Cabin'].notna().sum())
print("Passengers without cabin data:", df['Cabin'].isna().sum())