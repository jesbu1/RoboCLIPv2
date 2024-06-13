from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import os
for i, key in enumerate(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()):
    folder_name = "videos_" + key
    prefix = "/scr/jzhang96/metaworld_generation_wo_done/" + folder_name
    all_files = os.listdir(prefix)
    num = 0
    for file in all_files:
        if file.endswith("True.mp4"):
            num += 1
    if num < 50:
        print("task id",i,"folder: ", folder_name, "num: ", num)

    