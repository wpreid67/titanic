import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import kagglehub

# ---- LOAD DATA ----
path = kagglehub.competition_download('titanic')
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ---- FEATURE ENGINEERING (same logic we discovered) ----
def prepare_features(df):
    df = df.copy()
    
    # Title
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'the Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    })
    
    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyCategory'] = pd.cut(df['FamilySize'],
        bins=[0,1,4,20],
        labels=['Alone', 'Small', 'Large'])
    
    # Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('Unknown')
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Sex as number
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    
    return df

train = prepare_features(train)
test = prepare_features(test)

# ---- ENCODE TEXT COLUMNS ----
le = LabelEncoder()
for col in ['Title', 'FamilyCategory', 'Deck', 'Embarked']:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ---- SELECT FEATURES ----
features = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize', 
            'IsAlone', 'HasCabin', 'Title', 'Embarked', 
            'FamilyCategory', 'Deck']

X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# ---- TRAIN RANDOM FOREST ----
rf = RandomForestClassifier(
    n_estimators=100,    # 100 trees
    max_depth=6,         # limit tree depth to avoid overfitting
    random_state=42      # reproducible results
)

# ---- CROSS VALIDATION ----
print("\n=== CROSS VALIDATION (5 fold) ===")
scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Scores per fold: {scores.round(3)}")
print(f"Average accuracy: {scores.mean().round(3)}")
print(f"Standard deviation: {scores.std().round(3)}")

# ---- TRAIN ON FULL TRAINING SET ----
rf.fit(X_train, y_train)

# ---- FEATURE IMPORTANCE ----
print("\n=== FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"{row['Feature']:<20} {row['Importance']:.3f}  {bar}")

# ---- GENERATE PREDICTIONS ----
test['Survived'] = rf.predict(X_test)

# ---- SAVE SUBMISSION FILE ----
submission = test[['PassengerId', 'Survived']]
submission.to_csv('submission_rf.csv', index=False)
print("\n=== SUBMISSION FILE ===")
print("Saved to submission_rf.csv")
print(f"Predicted survived: {submission['Survived'].sum()}")
print(f"Predicted died: {(submission['Survived']==0).sum()}")