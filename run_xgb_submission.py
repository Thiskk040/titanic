import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Clean strings
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

def clean_strings(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

train_df = clean_strings(train_df)
test_df = clean_strings(test_df)

# Store IDs
test_passenger_ids = test_df['PassengerId']

# Preprocess
def preprocess_data(df, is_train=True):
    df = df.copy()
    cols = ['Age', 'Fare', 'SibSp', 'Parch']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    cols_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    sex_map = {'male': 0, 'female': 1}
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map(sex_map)
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].map(embarked_map)
    
    return df

train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df, is_train=False)

# Double check for NaNs (XGBoost handles them but clean is better)
train_processed = train_processed.fillna(train_processed.mean()).fillna(0)
test_processed = test_processed.fillna(test_processed.mean()).fillna(0)

if 'Survived' in train_processed.columns:
    X_train = train_processed.drop('Survived', axis=1)
    y_train = train_processed['Survived']
else:
    raise ValueError("Survived column missing")

X_test = test_processed

# Train XGB (New)
print("Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Trained.")

# Submission
submission_xgb = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': xgb_pred
})
submission_xgb.to_csv('submission_xgb.csv', index=False)
print("Saved submission_xgb.csv")
