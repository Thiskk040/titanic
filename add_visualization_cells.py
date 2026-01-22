import json
import os

path = r'c:\Users\user\Desktop\Titanic\titanic_analysis.ipynb'

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
insert_index = -1

# Find the index of the cell containing df.isnull().sum() to insert after it
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "df.isnull().sum()" in source:
            insert_index = i + 1
            break

if insert_index == -1:
    print("Could not find target cell.")
    # Fallback: insert before "Data Preprocessing"
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if "Data Preprocessing" in source or "การเตรียมข้อมูล" in source:
                insert_index = i
                break

if insert_index == -1:
    insert_index = len(cells) # Append to end if not found

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.1 การวิเคราะห์ตัวแปรเดี่ยว (Univariate Analysis)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# กราฟแสดงจำนวนผู้รอดชีวิตและผู้เสียชีวิต\n",
            "plt.figure(figsize=(6, 4))\n",
            "sns.countplot(x='Survived', data=df, palette='pastel')\n",
            "plt.title('Survival Count')\n",
            "plt.xlabel('Survived (0=No, 1=Yes)')\n",
            "plt.ylabel('Count')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# กราฟแสดงการกระจายตัวของอายุ\n",
            "plt.figure(figsize=(8, 5))\n",
            "sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='blue')\n",
            "plt.title('Age Distribution')\n",
            "plt.xlabel('Age')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.2 การวิเคราะห์ความสัมพันธ์ (Bivariate Analysis)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# กราฟแสดงการรอดชีวิตแยกตามเพศ\n",
            "plt.figure(figsize=(6, 4))\n",
            "sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')\n",
            "plt.title('Survival by Sex')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# กราฟแสดงการรอดชีวิตแยกตามชั้นที่นั่ง (Pclass)\n",
            "plt.figure(figsize=(6, 4))\n",
            "sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set2')\n",
            "plt.title('Survival by Pclass')\n",
            "plt.show()"
        ]
    },
     {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# กราฟ Heatmap แสดงความสัมพันธ์ระหว่างตัวแปร (Correlation Matrix)\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')\n",
            "plt.title('Correlation Heatmap')\n",
            "plt.show()"
        ]
    }
]

# Insert the new cells
for idx, cell in enumerate(new_cells):
    cells.insert(insert_index + idx, cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Visualization cells added successfully.")
