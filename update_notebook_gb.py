import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Imports
first_code_cell = next(c for c in nb['cells'] if c['cell_type'] == 'code')
import_found = False
for line in first_code_cell['source']:
    if 'GradientBoostingClassifier' in line:
        import_found = True
        break

if not import_found:
    new_source = []
    for line in first_code_cell['source']:
        if 'from sklearn.ensemble import RandomForestClassifier' in line:
            new_source.append(line.replace('RandomForestClassifier', 'RandomForestClassifier, GradientBoostingClassifier'))
        else:
            new_source.append(line)
    first_code_cell['source'] = new_source

# 2. Find position to insert GB cells (after RF code cell)
rf_code_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'rf_model =' in ''.join(cell['source']):
        rf_code_index = i
        break

if rf_code_index != -1:
    # 3. Create GB Markdown Cell
    gb_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.3 Gradient Boosting\n",
            "\n",
            "เทคนิคที่สร้างต้นไม้แบบ Sequential เพื่อแก้ไขข้อผิดพลาดของต้นไม้ก่อนหน้า"
        ]
    }

    # 4. Create GB Code Cell
    gb_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# สร้างและเทรนโมเดล Gradient Boosting\n",
            "gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)\n",
            "gb_model.fit(X_train, y_train)\n",
            "\n",
            "# ทำนายผล\n",
            "gb_pred = gb_model.predict(X_test)\n",
            "print(\"Gradient Boosting Trained.\")"
        ]
    }

    # Insert cells
    # Check if already inserted to avoid dupes (naive check)
    # We look at the cell after RF code. If it's a markdown starting with "## 3", we are good to insert.
    # Or just check content.
    inserted = False
    if rf_code_index + 1 < len(nb['cells']):
        if 'Gradient Boosting' in ''.join(nb['cells'][rf_code_index+1].get('source', [])):
            inserted = True
    
    if not inserted:
        nb['cells'].insert(rf_code_index + 1, gb_code_cell)
        nb['cells'].insert(rf_code_index + 1, gb_md_cell)

# 5. Update Submission Cell
submission_cell = next(c for c in reversed(nb['cells']) if c['cell_type'] == 'code')
if 'submission_gb.csv' not in ''.join(submission_cell['source']):
    submission_cell['source'].append('\n')
    submission_cell['source'].append("# Submission สำหรับ Gradient Boosting\n")
    submission_cell['source'].append("submission_gb = pd.DataFrame({\n")
    submission_cell['source'].append("    'PassengerId': test_passenger_ids,\n")
    submission_cell['source'].append("    'Survived': gb_pred\n")
    submission_cell['source'].append("})\n")
    submission_cell['source'].append("submission_gb.to_csv('submission_gb.csv', index=False)\n")
    submission_cell['source'].append("print(\"Saved submission_gb.csv\")")

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Updated generate_submission.ipynb with Gradient Boosting.")
