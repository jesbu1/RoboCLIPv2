from meta_world_name_ann import task_ann
import os
import shutil
from tqdm import tqdm

total_ann = []
new_path = "/scr/jzhang96/metaworld_generation_true_video/"
count = 0
folder = 0
for task_name in tqdm(task_ann):
    folder_name = "videos_" + task_name
    prefix = "/scr/jzhang96/metaworld_generation_wo_done/" + folder_name
    all_files = os.listdir(prefix)
    file_names = []
    new_file_names = []
    target_folder_path = os.path.join(new_path, folder_name)
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path, exist_ok=True)
    for file in all_files:
        if file.endswith("True.mp4"):
            file_real_path = os.path.join(prefix, file)
            file_names.append(file_real_path)
            new_file_names.append(os.path.join(target_folder_path, file))
    if len(file_names) >= 30:
        folder += 1
        count += len(file_names)
        for i in range(len(file_names)):
            # import pdb; pdb.set_trace()
            a = shutil.copy2(file_names[i], target_folder_path+"/")
            # import pdb; pdb.set_trace()
print("total folder: ", folder, "total video: ", count)
            
    