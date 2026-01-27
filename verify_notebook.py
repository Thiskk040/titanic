import json

nb_path = 'c:/Users/user/Desktop/Titanic/generate_submission.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

import_found = False
cleaning_found = False
gb_found = False

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code':
        if 'GradientBoostingClassifier' in source and 'import' in source:
            print(f"Cell {i}: Import found.")
            import_found = True
        
        if 'fillna' in source and 'X_train' in source:
             print(f"Cell {i}: Cleaning code found.")
             cleaning_found = True
             
        if 'GradientBoostingClassifier' in source and 'fit' in source:
             print(f"Cell {i}: GB Training found.")
             gb_found = True
             # Clear the error output if present
             cell['outputs'] = []

if import_found and cleaning_found and gb_found:
    print("All components present.")
else:
    print(f"Missing components: Import={import_found}, Cleaning={cleaning_found}, GB={gb_found}")

# Save the cleared outputs version
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)
