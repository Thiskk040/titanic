import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

df.columns = df.columns.str.strip()

df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].replace('', np.nan)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_cleaned = df.drop(columns=cols_to_drop)

print(df_cleaned.info())
print(df_cleaned.head())

df_cleaned.to_csv('titanic_clean.csv', index=False)