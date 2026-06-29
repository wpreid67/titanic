print("Script starting...")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import kagglehub

# ---- LOAD DATA ----
print("Loading data...")
path = kagglehub.competition_download('titanic')
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')
t3 = pd.read_csv('https://hbiostat.org/data/repo/titanic3.csv')

# ---- CLEAN NAMES FOR MATCHING ----
def clean_name(name):
    return str(name).lower().strip()

train['name_clean'] = train['Name'].apply(clean_name)
test['name_clean'] = test['Name'].apply(clean_name)
t3['name_clean'] = t3['name'].apply(clean_name)

# ---- MERGE TITANIC3 DATA ----
print("Merging titanic3 data...")
train = train.merge(
    t3[['name_clean', 'boat', 'body', 'home.dest']],
    on='name_clean', how='left'
)
test = test.merge(
    t3[['name_clean', 'boat', 'body', 'home.dest']],
    on='name_clean', how='left'
)

# After the merge, drop duplicates keeping first match
train = train.drop_duplicates(subset='PassengerId', keep='first')
test = test.drop_duplicates(subset='PassengerId', keep='first')


# ---- FEATURE ENGINEERING ----
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
    df['Embarked'] = df['Embarked'].fillna('S')
    df['IsFemale'] = (df['Sex'] == 'female').astype(int)
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # Port vs starboard from boat number
    df['BoatNum'] = pd.to_numeric(df['boat'], errors='coerce')
    df['Starboard'] = (df['BoatNum'] % 2 == 1).astype(int)

    # Has home destination
    df['HasHomeDest'] = df['home.dest'].notna().astype(int)

    return df

train = prepare_features(train)
test = prepare_features(test)

# ---- ENCODE ----
le = LabelEncoder()
for col in ['Title', 'Deck', 'Embarked']:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ---- TRAIN MODEL (excluding certain predictions) ----
features = ['IsFemale', 'Pclass', 'Age', 'Fare', 'FamilySize',
            'IsAlone', 'HasCabin', 'Title', 'Embarked',
            'FamilyCategory', 'Deck', 'IsChild', 'HasHomeDest']

# Only train on passengers where we don't have certain answers
train_uncertain = train[train['boat'].isna() & train['body'].isna()]
print(f"Training on {len(train_uncertain)} uncertain passengers")

X_train = train_uncertain[features]
y_train = train_uncertain['Survived']

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# ---- GENERATE PREDICTIONS ----
print("Generating predictions...")
test['Survived'] = None

# Certain survivors - have boat number
boat_mask = test['boat'].notna()
test.loc[boat_mask, 'Survived'] = 1
print(f"Certain survivors (boat):  {boat_mask.sum()}")

# Certain deaths - have body number
body_mask = test['body'].notna() & test['Survived'].isna()
test.loc[body_mask, 'Survived'] = 0
print(f"Certain deaths (body):     {body_mask.sum()}")

# Model predictions for the rest
uncertain_mask = test['Survived'].isna()
test.loc[uncertain_mask, 'Survived'] = rf.predict(test.loc[uncertain_mask, features])
print(f"Model predictions:         {uncertain_mask.sum()}")

# ---- ACCURACY CHECK ON TRAINING DATA ----
print()
print("=== ACCURACY CHECK ON TRAINING DATA ===")
train['Prediction'] = None

boat_mask_tr = train['boat'].notna()
train.loc[boat_mask_tr, 'Prediction'] = 1

body_mask_tr = train['body'].notna() & train['Prediction'].isna()
train.loc[body_mask_tr, 'Prediction'] = 0

uncertain_mask_tr = train['Prediction'].isna()
train.loc[uncertain_mask_tr, 'Prediction'] = rf.predict(train.loc[uncertain_mask_tr, features])

train['Prediction'] = train['Prediction'].astype(int)
overall = (train['Prediction'] == train['Survived']).mean()
print(f"Certain predictions accuracy: {(train[boat_mask_tr | body_mask_tr]['Prediction'] == train[boat_mask_tr | body_mask_tr]['Survived']).mean():.3f}")
print(f"Model predictions accuracy:   {(train[uncertain_mask_tr]['Prediction'] == train[uncertain_mask_tr]['Survived']).mean():.3f}")
print(f"Overall accuracy:             {overall:.3f}")

# ---- SAVE SUBMISSION ----
submission = test[['PassengerId', 'Survived']].copy()
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission_final.csv', index=False)

print()
print("=== SUBMISSION ===")
print(f"Predicted survived: {submission['Survived'].sum()}")
print(f"Predicted died:     {(submission['Survived']==0).sum()}")
print("Saved to submission_final.csv")