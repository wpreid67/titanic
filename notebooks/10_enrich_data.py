print("Script starting...")
import pandas as pd
import kagglehub

# ---- LOAD KAGGLE DATA ----
print("Loading Kaggle data...")
path = kagglehub.competition_download('titanic')
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

# ---- LOAD TITANIC3 ----
print("Loading titanic3...")
t3 = pd.read_csv('https://hbiostat.org/data/repo/titanic3.csv')
print(f"Titanic3 shape: {t3.shape}")
print(f"Columns: {list(t3.columns)}")
print()

# ---- CLEAN NAMES FOR MATCHING ----
def clean_name(name):
    return str(name).lower().strip()

train['name_clean'] = train['Name'].apply(clean_name)
test['name_clean'] = test['Name'].apply(clean_name)
t3['name_clean'] = t3['name'].apply(clean_name)

# ---- MATCH COMPETITION DATA TO TITANIC3 ----
print("=== MATCHING COMPETITION DATA TO TITANIC3 ===")
test_merged = test.merge(
    t3[['name_clean', 'boat', 'body', 'home.dest', 'age']],
    on='name_clean',
    how='left'
)

print(f"Competition passengers matched: {test_merged['boat'].notna().sum() + test_merged['body'].notna().sum()}")
print(f"Have boat number (survived):    {test_merged['boat'].notna().sum()}")
print(f"Have body number (died):        {test_merged['body'].notna().sum()}")
print(f"No match found:                 {test_merged['boat'].isna().sum()}")
print()

# ---- SHOW CERTAIN SURVIVORS ----
print("=== COMPETITION PASSENGERS WITH BOAT NUMBER (SURVIVED) ===")
survivors = test_merged[test_merged['boat'].notna()]
print(survivors[['PassengerId', 'Name', 'Sex', 'Pclass', 'boat']].to_string())
print()

# ---- SHOW CERTAIN DEATHS ----
print("=== COMPETITION PASSENGERS WITH BODY NUMBER (DIED) ===")
deaths = test_merged[test_merged['body'].notna()]
print(deaths[['PassengerId', 'Name', 'Sex', 'Pclass', 'body']].to_string())
print()

# ---- BOAT SIDE ANALYSIS ----
print("=== PORT VS STARBOARD ANALYSIS ===")
survivors['boat_num'] = pd.to_numeric(survivors['boat'], errors='coerce')
survivors['side'] = survivors['boat_num'].apply(
    lambda x: 'Port (Lightoller)' if x % 2 == 0 
    else 'Starboard (Murdoch)' if pd.notna(x) 
    else 'Collapsible'
)
print(survivors['side'].value_counts())
print()

# ---- MATCH TRAINING DATA TOO ----
print("=== MATCHING TRAINING DATA TO TITANIC3 ===")
train_merged = train.merge(
    t3[['name_clean', 'boat', 'body', 'home.dest']],
    on='name_clean',
    how='left'
)
print(f"Training passengers matched:    {train_merged['boat'].notna().sum() + train_merged['body'].notna().sum()}")
print(f"Have boat number:               {train_merged['boat'].notna().sum()}")
print(f"Have body number:               {train_merged['body'].notna().sum()}")

# ---- VERIFY BOAT = SURVIVED ----
print()
print("=== VERIFICATION: BOAT NUMBER ACCURACY ON TRAINING DATA ===")
has_boat = train_merged[train_merged['boat'].notna()]
print(f"Passengers with boat number:    {len(has_boat)}")
print(f"Of those, actually survived:    {has_boat['Survived'].sum()}")
print(f"Accuracy of boat = survived:    {has_boat['Survived'].mean():.3f}")

has_body = train_merged[train_merged['body'].notna()]
print(f"\nPassengers with body number:    {len(has_body)}")
print(f"Of those, actually died:        {(has_body['Survived']==0).sum()}")
print(f"Accuracy of body = died:        {(has_body['Survived']==0).mean():.3f}")

# ---- SAVE ENRICHED DATA ----
test_merged.to_csv('test_enriched.csv', index=False)
train_merged.to_csv('train_enriched.csv', index=False)
print()
print("Saved enriched datasets to test_enriched.csv and train_enriched.csv")