import pandas as pd
import kagglehub

path = kagglehub.competition_download('titanic')
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

def audit(df, name):
    print(f"\n=== {name} DATA QUALITY ===")
    print(f"Total rows: {len(df)}")
    print()
    print("Missing values:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col:<15} {missing[col]:>4} missing  ({missing_pct[col]}%)")
    
    print()
    print("Suspicious values:")
    
    # Age
    if 'Age' in df.columns:
        print(f"  Age range: {df['Age'].min()} to {df['Age'].max()}")
        print(f"  Age < 1:   {(df['Age'] < 1).sum()} passengers")
    
    # Fare
    if 'Fare' in df.columns:
        print(f"  Fare range: {df['Fare'].min()} to {df['Fare'].max()}")
        print(f"  Fare = 0:   {(df['Fare'] == 0).sum()} passengers")
    
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print(f"  FamilySize range: {df['FamilySize'].min()} to {df['FamilySize'].max()}")

audit(train, 'TRAINING')
audit(test, 'COMPETITION')

# ---- SPECIFIC CONCERNS ----
print("\n=== PASSENGERS WITH FARE = 0 ===")
zero_fare = train[train['Fare'] == 0]
print(zero_fare[['Name', 'Pclass', 'Sex', 'Age', 'Cabin', 'Survived']].to_string())

print("\n=== COMPETITION DATA - MISSING CRITICAL FIELDS ===")
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
critical = test[test['Age'].isna() | test['Fare'].isna()]
print(f"Missing Age or Fare: {len(critical)} passengers")
print(critical[['PassengerId', 'Name', 'Sex', 'Pclass', 'Age', 'Fare']].to_string())

# ---- FARE = 0 ANALYSIS ----
print("=== FARE = 0 PASSENGERS ===")
zero_fare = train[train['Fare'] == 0]
print("Count:", len(zero_fare))
print("Survival rate:", zero_fare['Survived'].mean().round(2))
print()
print("Class breakdown:")
print(zero_fare['Pclass'].value_counts())
print()
print("Sex breakdown:")
print(zero_fare['Sex'].value_counts())
print()

# Compare to paid passengers same class
print("=== COMPARISON: FARE=0 vs PAID, BY CLASS ===")
for pclass in [1, 2, 3]:
    paid = train[(train['Fare'] > 0) & (train['Pclass'] == pclass)]['Survived'].mean()
    free = train[(train['Fare'] == 0) & (train['Pclass'] == pclass)]
    if len(free) > 0:
        print(f"Class {pclass} - Paid survival: {paid:.2f}  Free survival: {free['Survived'].mean():.2f}  (n={len(free)})")
