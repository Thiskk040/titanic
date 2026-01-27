import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Imports
first_code_cell = next(c for c in nb['cells'] if c['cell_type'] == 'code')
import_found = False
for line in first_code_cell['source']:
    if 'xgboost' in line:
        import_found = True
        break

if not import_found:
    first_code_cell['source'].insert(0, "from xgboost import XGBClassifier\n")

# 2. Find position to insert XGB cells (after GB code cell)
gb_code_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'gb_model =' in ''.join(cell['source']):
        gb_code_index = i
        break

if gb_code_index != -1:
    # 3. Create XGB Markdown Cell
    xgb_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2.4 XGBoost (Extreme Gradient Boosting)\n",
            "\n",
            "โมเดลยอดนิยมที่ปรับปรุงประสิทธิภาพมาจาก Gradient Boosting"
        ]
    }

    # 4. Create XGB Code Cell
    xgb_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# สร้างและเทรนโมเดล XGBoost\n",
            "xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')\n",
            "xgb_model.fit(X_train, y_train)\n",
            "\n",
            "# ทำนายผล\n",
            "xgb_pred = xgb_model.predict(X_test)\n",
            "print(\"XGBoost Trained.\")"
        ]
    }

    # Insert cells if not already there
    inserted = False
    if gb_code_index + 1 < len(nb['cells']):
        if 'XGBoost' in ''.join(nb['cells'][gb_code_index+1].get('source', [])):
            inserted = True
    
    if not inserted:
        nb['cells'].insert(gb_code_index + 1, xgb_code_cell)
        nb['cells'].insert(gb_code_index + 1, xgb_md_cell)

# 5. Update Submission Cell
submission_cell = next(c for c in reversed(nb['cells']) if c['cell_type'] == 'code')
if 'submission_xgb.csv' not in ''.join(submission_cell['source']):
    submission_cell['source'].append('\n')
    submission_cell['source'].append("# Submission สำหรับ XGBoost\n")
    submission_cell['source'].append("submission_xgb = pd.DataFrame({\n")
    submission_cell['source'].append("    'PassengerId': test_passenger_ids,\n")
    submission_cell['source'].append("    'Survived': xgb_pred\n")
    submission_cell['source'].append("})\n")
    submission_cell['source'].append("submission_xgb.to_csv('submission_xgb.csv', index=False)\n")
    submission_cell['source'].append("print(\"Saved submission_xgb.csv\")")

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Updated generate_submission.ipynb with XGBoost.")
