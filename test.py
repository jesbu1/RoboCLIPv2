import json

id2task_path = "metaworld_generation/task_id.json"
with open(id2task_path, "r") as f:
    id2task = json.load(f)
task2id = {v: k for k, v in id2task.items()}
file_name = "id_task.json"
with open(file_name, "w") as f:
    json.dump(task2id, f, indent=4)