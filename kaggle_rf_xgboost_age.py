import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

# 1. โหลดข้อมูล
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_passenger_ids = test_df['PassengerId']

# รวมข้อมูลเพื่อ Clean ทรงเดียวกัน
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['Survived'] = np.nan

df = pd.concat([train_df, test_df], ignore_index=True)

# 2. ทำความสะอาดข้อมูลเบื้องต้น
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

if df['Fare'].isnull().sum() > 0:
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# สกัดคำนำหน้าชื่อ (Title) เพื่อช่วยให้เดาอายุได้แม่นขึ้น (เช่น Master มักจะอายุน้อย)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 1, "Countess": 4, "Ms": 1, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 2,"Capt": 4,"Sir": 4 }
df['Title'] = df['Title'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)

# ลบคอลัมน์ที่ไม่จำเป็น
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# 3. เติมค่า Age ที่หายไป ด้วย XGBoost
df_age_not_null = df[df['Age'].notnull()]
df_age_null = df[df['Age'].isnull()]

age_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']

if len(df_age_null) > 0:
    print("Training XGBRegressor for Age imputation...")
    xgb_age_model = XGBRegressor(n_estimators=100, random_state=42)
    # เทรนโมเดลทายอายุ
    xgb_age_model.fit(df_age_not_null[age_features], df_age_not_null['Age'])
    
    # ทายอายุคนที่หายไป
    predicted_ages = xgb_age_model.predict(df_age_null[age_features])
    # ใส่ค่ากลับเข้าไปใน DataFrame
    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    print("Age imputation completed. Missing ages filled.")

# แยกข้อมูลกลับเป็น train และ test
train_clean = df[df['is_train'] == 1].drop(columns=['is_train'])
test_clean = df[df['is_train'] == 0].drop(columns=['is_train', 'Survived'])

X_train = train_clean.drop('Survived', axis=1)
y_train = train_clean['Survived']

X_test = test_clean

# 4. เทรนโมเดล Random Forest เพื่อทำนาย Survived
print("Training Random Forest Classifier on cleaned data...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. ทำนายผลกับชุด Test
print("Predicting on test dataset...")
predictions = rf_model.predict(X_test)

# 6. สร้างไฟล์ CSV สำหรับส่ง Kaggle
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': predictions.astype(int)
})

csv_filename = 'submission_rf_xgboost_age.csv'
submission.to_csv(csv_filename, index=False)
print(f"Success! Saved Kaggle submission to: {csv_filename}")
