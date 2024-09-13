import h5py
import numpy as np
import imageio
from tqdm import tqdm

task_ann = {
    "assembly-v2": "assembling",
    "basketball-v2": "playing basketball",
    "bin-picking-v2": "Picking bin",
    "box-close-v2":  "closing box",
    "button-press-topdown-v2": "pressing button",
    "button-press-topdown-wall-v2": "pressing button",
    "button-press-v2": "pressing button",
    "button-press-wall-v2": "pressing button",
    "coffee-button-v2": "roasting coffee",
    "coffee-pull-v2": "pulling cup",
    "coffee-push-v2": "pushing cup",
    "dial-turn-v2": "turning dial.",
    "disassemble-v2": "disassembling",
    "door-close-v2": "closing door",
    "door-lock-v2": "locking door",
    "door-open-v2": "opening door",
    "door-unlock-v2": "unlocking door",
    "hand-insert-v2": "inserting bin",
    "drawer-close-v2": "closing drawer",
    "drawer-open-v2": "opening drawer",
    "faucet-open-v2": "opening faucet",
    "faucet-close-v2": "closing faucet",
    "hammer-v2": "hammering nail",
    "handle-press-side-v2": "pressing handle",
    "handle-press-v2": "pressing handle",
    "handle-pull-side-v2": "pulling handle",
    "handle-pull-v2": "pulling handle",
    "lever-pull-v2": "pulling lever.",
    "peg-insert-side-v2": "inserting peg",
    "pick-place-wall-v2": "picking bin",
    "pick-out-of-hole-v2": "picking bin",
    "reach-v2": "reaching red",
    "push-back-v2": "pushing bin back.",
    "push-v2": "pushing bin",
    "pick-place-v2": "picking bin",
    "plate-slide-v2": "sliding plate",
    "plate-slide-side-v2": "sliding plate",
    "plate-slide-back-v2": "sliding plate",
    "plate-slide-back-side-v2": "sliding plate",
    "peg-unplug-side-v2": "unpluging peg",
    "soccer-v2": "kicking soccer ball",
    "stick-push-v2": "pushing stick",
    "stick-pull-v2": "pulling stick",
    "push-wall-v2": "pushing bin",
    "reach-wall-v2": "reaching red",
    "shelf-place-v2": "placing bin to shelf",
    "sweep-into-v2": "sweeping bin",
    "sweep-v2": "sweeping bin",
    "window-open-v2": "opening window",
    "window-close-v2": "closing window",
}

traj_path = "/scr/jzhang96/metaworld_new_generate_with_state/metaworld_traj_15.h5"
video_path = "/scr/jzhang96/metaworld_new_generate_with_state/metaworld_video_15.h5"

h5_video = h5py.File(video_path, 'r')
h5_traj = h5py.File(traj_path, 'r')



# generate new traj file and video file, only sample first 10 traj for each task



traj_path = "/scr/jzhang96/metaworld_new_generate_with_state/metaworld_dataset.h5"
video_path = "/scr/jzhang96/metaworld_new_generate_with_state/metaworld_video_15.h5"

# h5_video_10 = h5py.File(video_path, 'w')
# h5_traj_10 = h5py.File(traj_path, 'w')
h5_dataset = h5py.File(traj_path, 'w')

state_list = []
action_list = []
done_list = []
string_list = []
img_list = []

for task in tqdm(h5_traj.keys()):
    text_string = task_ann[task]
    string = np.array(text_string, dtype='S')


    traj = h5_traj[task]
    for key in traj.keys():
        states = np.array(traj[key]["state"][:])
        actions = np.array(traj[key]["action"][:])
        dones = np.array(traj[key]["done"][:])
        imgs = np.array(h5_video[task][key]["video"][:])
        # sample the center 224 x 224
        center1 = 480 // 2 - 224 // 2
        center2 = 640 // 2 - 224 // 2
        imgs = imgs[:-1, center1:center1+224, center2:center2+224, :]

        num = states.shape[0]
        state_list.append(states)
        action_list.append(actions)
        done_list.append(dones)
        for i in range(num):
            string_list.append(string)
        img_list.append(imgs)

state_list = np.concatenate(state_list, axis=0)
print(state_list.shape, "state")
action_list = np.concatenate(action_list, axis=0)
print(action_list.shape, "action")
done_list = np.concatenate(done_list, axis=0)
print(done_list.shape, "done")
string_list = np.stack(string_list, axis=0)
print(string_list.shape, "string")
img_list = np.concatenate(img_list, axis=0)
print(img_list.shape, "img")


h5_dataset.create_dataset("state", data=state_list)
h5_dataset.create_dataset("action", data=action_list)
h5_dataset.create_dataset("done", data=done_list)
h5_dataset.create_dataset("string", data=string_list)
h5_dataset.create_dataset("img", data=img_list, dtype='uint8', compression='gzip', compression_opts=9) #compress the image data
                           

h5_dataset.close()








    # video_group = h5_video[task]
#     traj_group_10 = h5_traj_10.create_group(task)
#     video_group_10 = h5_video_10.create_group(task)

#     for i in range(15):
#         single_traj_group = traj_group_10.create_group(str(i))
#         single_video_group = video_group_10.create_group(str(i))

#         single_traj_group.create_dataset("state", data=task_group[list(task_group.keys())[i]]["state"][:])
#         single_traj_group.create_dataset("next_state", data=task_group[list(task_group.keys())[i]]["next_state"][:])
#         single_traj_group.create_dataset("action", data=task_group[list(task_group.keys())[i]]["action"][:])
#         single_traj_group.create_dataset("reward", data=task_group[list(task_group.keys())[i]]["reward"][:])
#         single_traj_group.create_dataset("done", data=task_group[list(task_group.keys())[i]]["done"][:])

# img 

#         single_video_group.create_dataset("video", data=video_group[list(video_group.keys())[i]]["video"][:])

# h5_video_10.close()
# h5_traj_10.close()


    
    

