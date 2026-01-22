import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# การวิเคราะห์ข้อมูล Titanic และการสร้างโมเดลทำนายการรอดชีวิต\n",
    "\n",
    "สมุดงาน (Notebook) นี้จัดทำขึ้นเพื่อวิเคราะห์ชุดข้อมูล Titanic โดยมีวัตถุประสงค์เพื่อทำนายการรอดชีวิตของผู้โดยสาร โดยใช้โมเดล Decision Tree และ Random Forest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. การนำเข้าไลบรารีและโหลดข้อมูล (Import Libraries and Load Data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "\n",
    "# ตั้งค่าการแสดงผลกราฟ\n",
    "# %matplotlib inline\n",
    "sns.set(style=\"whitegrid\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# โหลดชุดข้อมูล\n",
    "df = pd.read_csv('Titanic-Dataset.csv')\n",
    "\n",
    "# แสดงข้อมูล 5 แถวแรก\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. การสำรวจข้อมูลเบื้องต้น (Exploratory Data Analysis - EDA)\n",
    "\n",
    "เราจะตรวจสอบโครงสร้างของข้อมูลและค่าที่หายไป (Missing Values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ตรวจสอบข้อมูลเบื้องต้น\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ตรวจสอบค่าที่หายไป\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. การเตรียมข้อมูล (Data Preprocessing)\n",
    "\n",
    "### 3.1 การจัดการค่าที่หายไป (Handling Missing Values)\n",
    "- **Age**: แทนค่าที่หายไปด้วยค่ามัธยฐาน (Median)\n",
    "- **Embarked**: แทนค่าที่หายไปด้วยฐานนิยม (Mode) หรือค่าที่ปรากฏบ่อยที่สุด\n",
    "- **Cabin**: มีค่าหายไปจำนวนมาก อาจจะตัดทิ้งหรือไม่นำมาใช้ในโมเดลนี้"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# แทนค่า Age ที่หายไปด้วย Median\n",
    "df['Age'].fillna(df['Age'].median(), inplace=True)\n",
    "\n",
    "# แทนค่า Embarked ที่หายไปด้วย Mode (ตัวแรก)\n",
    "df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)\n",
    "\n",
    "# ลบคอลัมน์ Cabin เนื่องจากข้อมูลหายไปเยอะเกินไป\n",
    "df.drop(columns=['Cabin'], inplace=True)\n",
    "\n",
    "# ตรวจสอบค่าที่หายไปอีกครั้ง\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 การแปลงข้อมูลและการคัดเลือกฟีเจอร์ (Feature Encoding & Selection)\n",
    "- แปลง **Sex** และ **Embarked** ให้เป็นตัวเลข\n",
    "- ตัดคอลัมน์ที่ไม่จำเป็นสำหรับการทำนายออก เช่น PassengerId, Name, Ticket"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# แปลงข้อมูล Categorical เป็นตัวเลข\n",
    "df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})\n",
    "df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n",
    "\n",
    "# เลือก Features ที่จะใช้\n",
    "features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']\n",
    "X = df[features]\n",
    "y = df['Survived']\n",
    "\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. การแบ่งข้อมูล (Data Splitting)\n",
    "แบ่งข้อมูลเป็นชุดสอน (Training Set) 80% และชุดทดสอบ (Testing Set) 20%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "print(f\"Training set shape: {X_train.shape}\")\n",
    "print(f\"Testing set shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. การสร้างและเทรนโมเดล (Model Training)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.1 Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# สร้างโมเดล Decision Tree\n",
    "dt_model = DecisionTreeClassifier(random_state=42)\n",
    "dt_model.fit(X_train, y_train)\n",
    "\n",
    "# ทำนายผล\n",
    "dt_pred = dt_model.predict(X_test)\n",
    "\n",
    "# ประเมินผล\n",
    "print(\"Decision Tree Accuracy:\", accuracy_score(y_test, dt_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, dt_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.2 Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# สร้างโมเดล Random Forest\n",
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "\n",
    "# ทำนายผล\n",
    "rf_pred = rf_model.predict(X_test)\n",
    "\n",
    "# ประเมินผล\n",
    "print(\"Random Forest Accuracy:\", accuracy_score(y_test, rf_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, rf_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. สรุปผล (Conclusion)\n",
    "\n",
    "จากการเปรียบเทียบผลลัพธ์ระหว่าง Decision Tree และ Random Forest จะเห็นได้ว่า..."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('c:/Users/user/Desktop/Titanic/titanic_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print("Created titanic_analysis.ipynb successfully.")
