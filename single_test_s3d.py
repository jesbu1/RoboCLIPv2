import numpy as np
from s3dg import S3D
import torch as th
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
import imageio
import os
import PIL



def preprocess_metaworld(frames, shorten=True):
    center = 240, 320
    h, w = (250, 250)
    x = int(center[1] - w/2)
    y = int(center[0] - h/2)
    frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
    a = frames
    frames = frames[None, :,:,:,:]
    frames = frames.transpose(0, 4, 1, 2, 3)
    if shorten:
        frames = frames[:, :,::4,:,:]
    # frames = frames/255
    return frames



def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    
    Read images from an animated GIF file.  Returns a list of numpy 
    arrays, or, if asNumpy is false, a list if PIL images.
    
    """
    
    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")
    
    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")
    
    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))
    
    # Load file using PIL
    pilIm = PIL.Image.open(filename)    
    pilIm.seek(0)
    
    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass
    
    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:            
            images.append( PIL.Image.fromarray(im) )
    
    # Done
    return images




net = S3D('s3d_dict.npy', 512)
net.load_state_dict(th.load('s3d_howto100m.pth'))
# Evaluation mode
net.eval().cuda()


total_sim = 0
total_random_sim = 0
count = 0
max_diff = None
min_diff = None



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--norm_before', action='store_true')
parser.add_argument('--norm_after', action='store_true')

args = parser.parse_args()

with th.no_grad():

    for i in range (len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)):
        # random int 0-49 0 and 49 are included
        # key_id = np.random.randint(0, len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
        # key = list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys())[key_id]


        key = list(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys())[i]
        if key == "door-open-v2-goal-hidden":
            
            env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[key](seed=0)
            env.reset()
        #     # select a random action


            frame_buffer = []
            for _ in range(32):
                action = env.action_space.sample()
                obs, _, _, _ = env.step(action)
                imgss = env.render(offscreen=True)[:,:,:3]
                frame_buffer.append(imgss)
            # collect_frames = np.array(frame_buffer)

            collect_frames = preprocess_metaworld(frame_buffer, shorten=False)
            collect_frames = th.from_numpy(collect_frames).float().cuda()
            # collect_frames = collect_frames[:,240-112:240+112,320-112:320+112,:]

            gt_path_1 = "/scr/jzhang96/metaworld_generate_gifs/{}/output_gif_1.gif".format(str(i))
            gt_path_2 = "gifs/human_opening_door.gif"

            gt_gif_1 = imageio.get_reader(gt_path_1)
            gt_gif_2 = imageio.get_reader(gt_path_2)

            gt_frames_1 = [frame[:,:,:3] for frame in gt_gif_1]
            gt_frames_2 = [frame[:,:,:3] for frame in gt_gif_2]
            # more_frames = 32 - len(gt_frames_2)
            # for i in range(more_frames):
            #     gt_frames_2.append(gt_frames_2[-1])
            # frames_gif = readGif(gt_path_2)
            # import pdb; pdb.set_trace()

            gt_frames_1 = preprocess_metaworld(gt_frames_1, shorten=False)
            gt_frames_2 = preprocess_metaworld(gt_frames_2, shorten=False)
            frames_length_1 = gt_frames_1.shape[2]
            frames_length_2 = gt_frames_2.shape[2]
            
            # lienar select 32 frames from the gif
            if frames_length_1 > 32:
                index = np.linspace(0, frames_length_1-1, 32, dtype=np.int)
                gt_frames_1 = gt_frames_1[:,:,index,:,:]
            if frames_length_2 > 32:
                index = np.linspace(0, frames_length_2-1, 32, dtype=np.int)
                gt_frames_2 = gt_frames_2[:,:,index,:,:]
            gt_frames_2 = np.resize(gt_frames_2, (1, 3, 26, 136, 136))
                
            
            gt_frames_1 = th.from_numpy(gt_frames_1).float().cuda()
            gt_frames_2 = th.from_numpy(gt_frames_2).float().cuda()
            
            total_frames = th.cat((gt_frames_1, collect_frames), dim=0)
            import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            

            if args.norm_before:
                total_frames = total_frames/255.0
                gt_frames_2 = gt_frames_2/255.0
            video_output = net(total_frames)['video_embedding']
            # import pdb; pdb.set_trace()
            GT_embedding_2 = net(gt_frames_2)['video_embedding']

            GT_embedding_1 = video_output[0:1]
            # GT_embedding_2 = video_output[1:2]
            collect_embedding = video_output[1:2]


            if args.norm_after:
                GT_embedding_1 = th.nn.functional.normalize(GT_embedding_1, p=2, dim=1)
                GT_embedding_2 = th.nn.functional.normalize(GT_embedding_2, p=2, dim=1)
                collect_embedding = th.nn.functional.normalize(collect_embedding, p=2, dim=1)


            sim_1_2 = th.matmul(GT_embedding_1, GT_embedding_2.t()).detach().cpu().numpy()[0][0]
            sim_1_collect = th.matmul(GT_embedding_1, collect_embedding.t()).detach().cpu().numpy()[0][0]
            sim_diff = sim_1_2 - sim_1_collect
            if max_diff is None or sim_diff > max_diff:
                max_diff = sim_diff
            if min_diff is None or sim_diff < min_diff:
                min_diff = sim_diff


            if sim_1_2 > sim_1_collect:
                count += 1
            print(key, sim_1_2, sim_1_collect, i)
            total_sim += sim_1_2
            total_random_sim += sim_1_collect
            print("Average similarity: ", "positive sim", total_sim/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN), "negative sim",total_random_sim/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
            print("Count: ", count, count/len(ALL_V2_ENVIRONMENTS_GOAL_HIDDEN))
            print("Max diff: ", max_diff)
            print("Min diff: ", min_diff)

