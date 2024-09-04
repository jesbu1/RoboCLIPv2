import json
import os

import numpy as np

id_task_dir_path = "../id_task.json"
id_subset_dir_path = "./task_subset.json"

with open(id_task_dir_path, "r") as f:
    id_task = json.load(f)

subset_1 = []
subset_2 = []
subset_3 = []
subset_4 = []
subset_5 = []

for i in range(len(id_task)):
    key = list(id_task.keys())[i]
    if i % 5 == 0:
        subset_1.append(key)    
    elif i % 5 == 1:
        subset_2.append(key)
    elif i % 5 == 2:
        subset_3.append(key)
    elif i % 5 == 3:
        subset_4.append(key)
    elif i % 5 == 4:
        subset_5.append(key)

task_subset = {
    "subset_1": subset_1,
    "subset_2": subset_2,
    "subset_3": subset_3,
    "subset_4": subset_4,
    "subset_5": subset_5
}

with open(id_subset_dir_path, "w") as f:
    json.dump(task_subset, f, indent=4)








