import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

print("XGBoost available:", xgb_available)

def preprocess_basic(df):
    df = df.copy()
    # Drop columns that are hard to use without feature engineering
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
    
    # Map Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Fill Embarked with mode and map
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Fill Fare with median
    if df['Fare'].isnull().sum() > 0:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        
    return df

# Load train data
train_original = pd.read_csv('train.csv', skipinitialspace=True)
# Strip trailing/leading spaces from column names (CSV has padded headers)
train_original.columns = train_original.columns.str.strip()
# Strip string values in object columns (CSV has padded cell values)
for col in train_original.select_dtypes(include='object').columns:
    train_original[col] = train_original[col].str.strip()
train_basic = preprocess_basic(train_original)

# Split features and target
y = train_basic['Survived']
X = train_basic.drop('Survived', axis=1)

# Method 1: Impute Age with Median
X_median = X.copy()
X_median['Age'] = X_median['Age'].fillna(X_median['Age'].median())

# Method 2: Impute Age with ML Model (XGBRegressor if available, else RandomForestRegressor)
X_model_imputed = X.copy()
# Features to train the age imputer (exclude Survived and Age)
age_features = [col for col in X.columns if col != 'Age']

# Separate data into missing and not missing age
age_not_null = X_model_imputed[X_model_imputed['Age'].notnull()]
age_null = X_model_imputed[X_model_imputed['Age'].isnull()]

if len(age_null) > 0:
    if xgb_available:
        age_model = XGBRegressor(n_estimators=100, random_state=42)
        print("Using XGBRegressor for Age imputation...")
    else:
        age_model = RandomForestRegressor(n_estimators=100, random_state=42)
        print("Using RandomForestRegressor for Age imputation...")
        
    # Train age model
    age_model.fit(age_not_null[age_features], age_not_null['Age'])
    
    # Predict ages
    predicted_ages = age_model.predict(age_null[age_features])
    X_model_imputed.loc[X_model_imputed['Age'].isnull(), 'Age'] = predicted_ages

# Let's evaluate final classification model 
# We'll use XGBClassifier if available, else GradientBoostingClassifier
if xgb_available:
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf_name = "XGBClassifier"
else:
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf_name = "GradientBoostingClassifier"

# Evaluate Method 1
scores_median = cross_val_score(clf, X_median, y, cv=5, scoring='accuracy')
print(f"\nFinal Model ({clf_name}) + Age Initialized with Median:")
print(f"Accuracy: {scores_median.mean():.4f} (+/- {scores_median.std():.4f})")

# Evaluate Method 2
scores_model = cross_val_score(clf, X_model_imputed, y, cv=5, scoring='accuracy')
print(f"\nFinal Model ({clf_name}) + Age Initialized with ML Model Prediction:")
print(f"Accuracy: {scores_model.mean():.4f} (+/- {scores_model.std():.4f})")

if scores_model.mean() > scores_median.mean():
    print("\nConclusion: Yes, using a model to predict missing Age yielded better accuracy!")
else:
    print("\nConclusion: In this specific simple setup, it didn't improve accuracy (or stayed about the same). Feature engineering (like extracting 'Title' from 'Name') along with model imputation usually gives the best results.")

# ─── Export clean datasets ────────────────────────────────────────────────────
# Method 1: Age imputed with Median
df_export_median = X_median.copy()
df_export_median['Survived'] = y.values
df_export_median.to_csv('train_clean_median.csv', index=False)
print("\n[Export] train_clean_median.csv saved.")

# Method 2: Age imputed with ML Model
df_export_model = X_model_imputed.copy()
df_export_model['Survived'] = y.values
df_export_model.to_csv('train_clean_model.csv', index=False)
print("[Export] train_clean_model.csv saved.")
