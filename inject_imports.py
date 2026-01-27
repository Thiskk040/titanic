import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Iterate through cells to find GB and XGB usage
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell['source'])
        
        # Fix GB
        if 'GradientBoostingClassifier' in source_str and 'import' not in source_str:
            print("Injecting import into GB cell...")
            cell['source'].insert(0, "from sklearn.ensemble import GradientBoostingClassifier\n")
            
        # Fix XGB
        if 'XGBClassifier' in source_str and 'import' not in source_str:
             print("Injecting import into XGB cell...")
             cell['source'].insert(0, "from xgboost import XGBClassifier\n")

# Save back
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Injected redundant imports.")
