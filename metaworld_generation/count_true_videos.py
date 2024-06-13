import os

path = "/scr/jzhang96/metaworld_generation_wo_done"
all_folders = os.listdir(path)
total = 0
folder_count = 0
total_folders = 0
for folder in all_folders:
    if folder.startswith("videos_"):
        file_names = os.listdir(os.path.join(path, folder))
        num = 0
        for file_name in file_names:

            if file_name.endswith(".mp4"):
                name_wo = file_name.split(".m")[0]
                if name_wo.split("_")[-1] == "True":
                    num += 1
        if num > 25:
            total += num
            folder_count += 1
        total_folders += 1
        print("foler: ", folder, "num: ", num)
print("total: ", total)
print("folder_count: ", folder_count, "total_folders: ", total_folders)

                # prefix = "videos_"
                # new_file_name = file_name.replace(prefix, "")
                # os.rename(os.path.join(path, folder, file_name), os.path.join(path, folder, new_file_name))
                # print(new_file_name)
# prefix = "videos_"