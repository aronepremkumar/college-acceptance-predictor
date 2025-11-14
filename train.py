# train.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load and clean data
df = pd.read_csv('data/stanford_admissions.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['act_composite'] = df['act_composite'].fillna(0)

# Convert categorical columns to string
categorical = ['gender_m', 'in_state', 'race_asian', 'race_white', 'race_hispanic',
               'race_black', 'race_other', 'first_gen', 'legacy', 'athlete']
for col in categorical:
    df[col] = df[col].astype(str)

numerical = ['age', 'sat_math', 'sat_ebrw', 'act_composite', 'gpa', 'extracurriculars_count']
features = categorical + numerical

# Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['admit'])
y_full_train = df_full_train['admit'].values
y_test = df_test['admit'].values

# Vectorize
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_full_train[features].to_dict(orient='records'))
X_test = dv.transform(df_test[features].to_dict(orient='records'))

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=(len(y_full_train) - sum(y_full_train)) / sum(y_full_train),
    eval_metric='logloss'
)
model.fit(X_train, y_full_train)

# Evaluate
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {auc:.4f}")

# Save
output_file = 'model_xgboost.bin'
with open(output_file, 'wb') as f:
    pickle.dump((dv, model), f)

print(f"Model saved to {output_file}")