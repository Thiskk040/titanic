import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the first code cell
first_code_cell = next(c for c in nb['cells'] if c['cell_type'] == 'code')

# Update the import
new_source = []
updated = False
for line in first_code_cell['source']:
    if 'from sklearn.ensemble import RandomForestClassifier' in line and 'GradientBoostingClassifier' not in line:
        new_source.append(line.replace('RandomForestClassifier', 'RandomForestClassifier, GradientBoostingClassifier'))
        updated = True
    else:
        new_source.append(line)

if updated:
    first_code_cell['source'] = new_source
    print("Fixed import: Added GradientBoostingClassifier.")
else:
    print("Import already correct or not found.")

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
