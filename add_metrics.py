import json

nb_path = 'generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add Imports
# Finding the first code cell which usually contains imports
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "import pandas" in source or "from xgboost" in source:
            new_imports = "from sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, classification_report\n"
            if "train_test_split" not in source:
                cell['source'].insert(0, new_imports)
            break

# 2. Insert Data Splitting Cell
# We look for the cell that ends preprocessing, usually printing "Features:" or "X_train"
insert_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Features:" in source and "X_train" in source:
            insert_index = i + 1
            break

split_code = [
    "# Split data for validation\n",
    "X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)\n",
    "print(f\"Training set shape: {X_tr.shape}\")\n",
    "print(f\"Validation set shape: {X_val.shape}\")"
]

if insert_index != -1:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_code
    }
    # Check if already inserted to avoid dupes
    if "X_tr, X_val" not in "".join(nb['cells'][insert_index]['source']):
         nb['cells'].insert(insert_index, new_cell)


# Helper to update model cells
def update_model_cell(model_name_comment, model_var, title):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if f"{model_var} =" in source and "fit" in source and "accuracy_score" not in source:
                # We found the training cell
                # We prepend the validation logic
                validation_logic = [
                    f"# Validation for {title}\n",
                    f"{model_var}.fit(X_tr, y_tr)\n",
                    f"val_pred = {model_var}.predict(X_val)\n",
                    f"print(f'{title} Accuracy: {{accuracy_score(y_val, val_pred):.4f}}')\n",
                    f"print(f'{title} Report:\\n{{classification_report(y_val, val_pred)}}')\n",
                    "# Re-train on full data for submission\n"
                ]
                cell['source'] = validation_logic + cell['source']

# 3. Update Decision Tree
update_model_cell("Decision Tree", "dt_model", "Decision Tree")

# 4. Update Random Forest
update_model_cell("Random Forest", "rf_model", "Random Forest")

# 5. Update Gradient Boosting
update_model_cell("Gradient Boosting", "gb_model", "Gradient Boosting")

# 6. Update XGBoost
# XGBoost might be named xgb_model
update_model_cell("XGBoost", "xgb_model", "XGBoost")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("Notebook updated successfully.")
