import json
import os

path = r'c:\Users\user\Desktop\Titanic\titanic_analysis.ipynb'

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first cell is where we want to add text
cell = nb['cells'][0]
current_source = cell['source']

# Modify the last line of existing source to have a newline if needed
if current_source and not current_source[-1].endswith('\n'):
    current_source[-1] += "\n"

new_lines = [
    "\n",
    "### ความหมายของแต่ละคอลัมน์ (Data Dictionary)\n",
    "- **PassengerId**: รหัสผู้โดยสาร\n",
    "- **Survived**: การรอดชีวิต (0 = ไม่รอด, 1 = รอด)\n",
    "- **Pclass**: ชั้นที่นั่ง (1 = ชั้น 1, 2 = ชั้น 2, 3 = ชั้น 3)\n",
    "- **Name**: ชื่อผู้โดยสาร\n",
    "- **Sex**: เพศ\n",
    "- **Age**: อายุ\n",
    "- **SibSp**: จำนวนพี่น้อง/คู่สมรสที่เดินทางมาด้วย\n",
    "- **Parch**: จำนวนบิดามารดา/ลูกที่เดินทางมาด้วย\n",
    "- **Ticket**: หมายเลขตั๋ว\n",
    "- **Fare**: ค่าโดยสาร\n",
    "- **Cabin**: หมายเลขห้องพัก\n",
    "- **Embarked**: ท่าเรือที่ขึ้น (C = Cherbourg, Q = Queenstown, S = Southampton)"
]

current_source.extend(new_lines)

# Write back with indent=1 to match existing format
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook updated successfully.")
