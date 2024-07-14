from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
import imageio
import torch as th
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoProcessor 
# import cos similarity
from torch.nn.functional import cosine_similarity
import torch
from s3dg import S3D
model = S3D('s3d_dict.npy', 512)
model.load_state_dict(th.load('s3d_howto100m.pth'))
# Evaluation mode
model.eval().cuda()


def adjust_frames_xclip(frames, target_frame_count = 32):
    """
    Ensures same numbers of frames(32). 
    """
    frame_count = frames.shape[0]
    #print(f"frames number{frame_count}")
    frames = th.from_numpy(frames)
    if frame_count < target_frame_count:
        blank_frames = th.zeros(
            (target_frame_count - frame_count, frames.shape[1], frames.shape[2], frames.shape[3]),
            dtype=frames.dtype)
        adjusted_frames = th.cat((frames, blank_frames), dim=0)

    elif frame_count > target_frame_count:
        indices = th.linspace(0, frame_count - 1, target_frame_count, dtype=th.long)
        adjusted_frames = th.index_select(frames, 0, indices)

    else:
        adjusted_frames = frames

    return adjusted_frames

total_sim = 0
total_random_sim = 0
count = 0
with th.no_grad():
    for i in range (len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)):
        # random int 0-49 0 and 49 are included
        # key_id = np.random.randint(0, len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
        # key = list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys())[key_id]


        key = list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys())[i]
        
        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[key](seed=0)
        env.reset()
        # select a random action


        frame_buffer = []
        for _ in range(32):
            action = env.action_space.sample()
            obs, _, _, _ = env.step(action)
            imgss = env.render(offscreen=True)[:,:,:3]
            frame_buffer.append(imgss)
        collect_frames = np.array(frame_buffer)
        collect_frames = adjust_frames_xclip(collect_frames)
        collect_frames = collect_frames[:,240-112:240+112,320-112:320+112,:]


        target_gif_path = "/scr/jzhang96/metaworld_generate_gifs/{}/output_gif_1.gif".format(str(i))
        # load gif
        gif = imageio.get_reader(target_gif_path)
        # get all frames
        frames = [frame[:,:,:3] for frame in gif]
        frames = np.array(frames)
        gt_frames = adjust_frames_xclip(frames)
        gt_frames = gt_frames[:,240-112:240+112,320-112:320+112,:]


        random_gif_path = "/scr/jzhang96/metaworld_generate_gifs/{}/output_gif_2.gif".format(str(i))
        # load gif
        gif = imageio.get_reader(random_gif_path)
        # get all frames
        frames = [frame[:,:,:3] for frame in gif]
        frames = np.array(frames)
        random_frames = adjust_frames_xclip(frames)
        random_frames = random_frames[:,240-112:240+112,320-112:320+112,:]
        gt_frames = gt_frames.permute(3,0,1,2).float().unsqueeze(0).cuda()
        random_frames = random_frames.permute(3,0,1,2).float().unsqueeze(0).cuda()
        collect_frames = collect_frames.permute(3,0,1,2).float().unsqueeze(0).cuda()

        gt_frames = gt_frames/255.0
        random_frames = random_frames/255.0
        collect_frames = collect_frames/255.0
        frames = th.cat((gt_frames, random_frames, collect_frames), dim=0)
        import pdb; pdb.set_trace()
        embeddings = model(frames)["video_embedding"]
        gt_embedding = embeddings[0:1]
        random_embedding = embeddings[1:2]
        collect_embedding = embeddings[2:3]
        gt_embedding = torch.nn.functional.normalize(gt_embedding, p=2, dim=1)
        random_embedding = torch.nn.functional.normalize(random_embedding, p=2, dim=1)
        collect_embedding = torch.nn.functional.normalize(collect_embedding, p=2, dim=1)
        # sim = cosine_similarity(gt_embedding, random_embedding)
        # random_sim = cosine_similarity(gt_embedding, collect_embedding)
        # compute dot product
        sim = torch.mm(gt_embedding, random_embedding.t())[0][0]
        random_sim = torch.mm(gt_embedding, collect_embedding.t())[0][0]

        if sim > random_sim:
            count += 1
        print(key, sim, random_sim, i)
        total_sim += sim
        total_random_sim += random_sim
    print("Average similarity: ", total_sim/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN), total_random_sim/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
    print("Count: ", count, count/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
        # import pdb; pdb.set_trace()
