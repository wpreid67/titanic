print("Script starting...")
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import kagglehub

print("Loading data...")
path = kagglehub.competition_download('titanic')
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

def prepare_features(df):
    df = df.copy()
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace({
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'the Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    })
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FamilyCategory'] = pd.cut(df['FamilySize'],
        bins=[0,1,4,20], labels=[0,1,2]).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareIsZero'] = (df['Fare'] == 0).astype(int)
    df['Embarked'] = df['Embarked'].fillna('S')
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    df['IsChild'] = (df['Age'] < 12).astype(int)
    return df

print("Preparing features...")
train = prepare_features(train)
test = prepare_features(test)

le = LabelEncoder()
for col in ['Title', 'Deck', 'Embarked']:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

features = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize',
            'IsAlone', 'HasCabin', 'Title', 'Embarked',
            'FamilyCategory', 'Deck', 'IsChild', 'FareIsZero']

X_train = train[features]
y_train = train['Survived']
X_test = test[features]

print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

print("\n=== XGBOOST CROSS VALIDATION (5 fold) ===")
scores = cross_val_score(xgb, X_train, y_train, cv=5)
print(f"Scores per fold: {scores.round(3)}")
print(f"Average accuracy: {scores.mean().round(3)}")
print(f"Standard deviation: {scores.std().round(3)}")

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"\nRandom Forest average: {rf_scores.mean().round(3)}")
print(f"XGBoost average:       {scores.mean().round(3)}")
print(f"Improvement:           {(scores.mean() - rf_scores.mean()).round(3)}")

xgb.fit(X_train, y_train)

print("\n=== FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"{row['Feature']:<20} {row['Importance']:.3f}  {bar}")

test['Survived'] = xgb.predict(X_test)
submission = test[['PassengerId', 'Survived']]
submission.to_csv('submission_xgb.csv', index=False)
print("\n=== SUBMISSION FILE ===")
print(f"Predicted survived: {submission['Survived'].sum()}")
print(f"Predicted died:     {(submission['Survived']==0).sum()}")
print("Saved to submission_xgb.csv")