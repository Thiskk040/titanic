import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Create the Robust Cleaning Code Cell
clean_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# [CRITICAL FIX] Handle remaining NaNs before Gradient Boosting\n",
        "# Gradient Boosting in sklearn does not realize NaNs natively as easily as XGBoost\n",
        "print(\"Checking for NaNs before GB training:\")\n",
        "print(X_train.isnull().sum())\n",
        "\n",
        "# Fill with mean or 0\n",
        "X_train = X_train.fillna(X_train.mean()).fillna(0)\n",
        "X_test = X_test.fillna(X_test.mean()).fillna(0)\n",
        "\n",
        "print(\"NaNs remaining:\", X_train.isnull().sum().sum())"
    ]
}

# Find where to insert (before GB code)
gb_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'GradientBoostingClassifier' in ''.join(cell['source']):
        gb_index = i
        break

if gb_index != -1:
    # Check if we already added it to avoid duplicates
    if 'CRITICAL FIX' not in ''.join(nb['cells'][gb_index-1].get('source', [])):
        nb['cells'].insert(gb_index, clean_code_cell)
        print("Inserted NaN cleaning cell before Gradient Boosting.")
    else:
        print("Cleaning cell already exists.")
else:
    print("Could not find Gradient Boosting cell.")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
