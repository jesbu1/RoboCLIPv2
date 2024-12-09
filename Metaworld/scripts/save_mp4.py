import os
import functools

import cv2
import numpy as np

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise
import imageio
from tqdm import tqdm
import h5py
resolution = (640, 480)
camera = 'corner2' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False # if True, flips output image 180 degrees


config = [
    # env, action noise pct, cycles, quit on success
    ('assembly-v2', np.zeros(4), 3, True),
    ('basketball-v2', np.zeros(4), 3, True),
    ('bin-picking-v2', np.zeros(4), 3, True),
    ('box-close-v2', np.zeros(4), 3, True),
    ('button-press-topdown-v2', np.zeros(4), 3, True),
    ('button-press-topdown-wall-v2', np.zeros(4), 3, True),
    ('button-press-v2', np.zeros(4), 3, True),
    ('button-press-wall-v2', np.zeros(4), 3, True),
    ('coffee-button-v2', np.zeros(4), 3, True),
    ('coffee-pull-v2', np.zeros(4), 3, True),
    ('coffee-push-v2', np.zeros(4), 3, True),
    ('dial-turn-v2', np.zeros(4), 3, True),
    ('disassemble-v2', np.zeros(4), 3, True),
    ('door-close-v2', np.zeros(4), 3, True),
    ('door-lock-v2', np.zeros(4), 3, True),
    ('door-open-v2', np.zeros(4), 3, True),
    ('door-unlock-v2', np.zeros(4), 3, True),
    ('hand-insert-v2', np.zeros(4), 3, True),
    ('drawer-close-v2', np.zeros(4), 3, True),
    ('drawer-open-v2', np.zeros(4), 3, True),
    ('faucet-open-v2', np.zeros(4), 3, True),
    ('faucet-close-v2', np.zeros(4), 3, True),
    ('hammer-v2', np.zeros(4), 3, True),
    ('handle-press-side-v2', np.zeros(4), 3, True),
    ('handle-press-v2', np.zeros(4), 3, True),
    ('handle-pull-side-v2', np.zeros(4), 3, True),
    ('handle-pull-v2', np.zeros(4), 3, True),
    ('lever-pull-v2', np.zeros(4), 3, True),
    ('peg-insert-side-v2', np.zeros(4), 3, True),
    ('pick-place-wall-v2', np.zeros(4), 3, True),
    ('pick-out-of-hole-v2', np.zeros(4), 3, True),
    ('reach-v2', np.zeros(4), 3, True),
    ('push-back-v2', np.zeros(4), 3, True),
    ('push-v2', np.zeros(4), 3, True),
    ('pick-place-v2', np.zeros(4), 3, True),
    ('plate-slide-v2', np.zeros(4), 3, True),
    ('plate-slide-side-v2', np.zeros(4), 3, True),
    ('plate-slide-back-v2', np.zeros(4), 3, True),
    ('plate-slide-back-side-v2', np.zeros(4), 3, True),
    ('peg-unplug-side-v2', np.zeros(4), 3, True),
    ('soccer-v2', np.zeros(4), 3, True),
    ('stick-push-v2', np.zeros(4), 3, True),
    ('stick-pull-v2', np.zeros(4), 3, True),
    ('push-wall-v2', np.zeros(4), 3, True),
    ('reach-wall-v2', np.zeros(4), 3, True),
    ('shelf-place-v2', np.zeros(4), 3, True),
    ('sweep-into-v2', np.zeros(4), 3, True),
    ('sweep-v2', np.zeros(4), 3, True),
    ('window-open-v2', np.zeros(4), 3, True),
    ('window-close-v2', np.zeros(4), 3, True),
]

def trajectory_generator(env, policy, act_noise_pct, res=(640, 480), camera='corner2'):
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()

    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, info = env.step(a)
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]




def writer_for_gif(tag, fps, res):
    if not os.path.exists('../../metaworld_movies_corner'):
        os.mkdir('../../metaworld_movies_corner')
    return cv2.VideoWriter(
        f'../../metaworld_movies_corner/{tag}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        res
    )




def main():
    collect_num = 24
    config_range = (0,50)
    
    base_path = '/scr/jzhang96/metaworld_25_for_vlc/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)



    for config_idx in tqdm(range (config_range[0], config_range[1])):

        env_name, noise, cycles, quit_on_success = config[config_idx]
        tag = env_name + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)
        policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]
        # env = ALL_ENVS[env_name]()
        # env._partially_observable = False
        # env._freeze_rand_vec = False
        # env._set_task_called = True
        success_num = 0

        for i in range (collect_num + 10):


            env = ALL_ENVS[env_name]()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(i)
            env.reset()

            env.reset_model()
            o = env.reset()
            rollout_success = False
            img = env.sim.render(*resolution, mode='offscreen', camera_name=camera).astype(np.uint8)
            img = img[img.shape[0]//2 - 112: img.shape[0]//2 + 112, img.shape[1]//2 - 112: img.shape[1]//2 + 112, :]
            # rgb 2 bgr
            img = img[:,:,::-1]
            imgs = [img ]
            action_space_ptp = env.action_space.high - env.action_space.low
            


            for step in range(env.max_path_length):
                
                a = policy.get_action(o)
                a = np.random.normal(a, noise * action_space_ptp)

                o, r, done, info = env.step(a)
                # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']\
                img = env.sim.render(*resolution, mode='offscreen', camera_name=camera).astype(np.uint8)
                img = img[:,:,::-1]
                img = img[img.shape[0]//2 - 112: img.shape[0]//2 + 112, img.shape[1]//2 - 112: img.shape[1]//2 + 112, :]
                imgs.append(img)


                if info['success']:
                    rollout_success = True
                    success_num += 1
                    done = 1

                    break
                else:
                    done = 0

                # yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
            if rollout_success:
                folder_name = os.path.join(base_path, env_name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # save sith name success_num.gif
                # gif_file_name = os.path.join(folder_name, "{}.gif".format(str(success_num)))
                # imageio.mimsave(gif_file_name, imgs, fps=30)
                # print(env_name, i, success_num)

                # save as mp4 file don't call writer_for_gif function
                writer = cv2.VideoWriter(
                    f'{folder_name}/{success_num}.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    30,
                    (224, 224)
                )
                for img in imgs:
                    writer.write(img)
                writer.release()
                




            if success_num > collect_num:
                break
                # gif_file_name = os.path.join(folder_name, "output_gif_{}.gif".format(str(success_num)))
                # imageio.mimsave(gif_file_name, imgs, fps=30)

        



    # action_space_ptp = env.action_space.high - env.action_space.low

    # env.reset()
    # env.reset_model()
    # o = env.reset()

    # for _ in range(env.max_path_length):
    #     a = policy.get_action(o)
    #     a = np.random.normal(a, act_noise_pct * action_space_ptp)

    #     o, r, done, info = env.step(a)
    #     # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
    #     yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]


    #     import pdb ; pdb.set_trace()



    # for env, noise, cycles, quit_on_success in config:
    #     # import pdb ; pdb.set_trace()
    #     cycles = 1
    #     tag = env + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)

    #     policy = functools.reduce(lambda a,b : a if a[0] == env else b, test_cases_latest_nonoise)[1]
    #     env = ALL_ENVS[env]()
    #     env._partially_observable = False
    #     env._freeze_rand_vec = False
    #     env._set_task_called = True

    #     # writer = writer_for(tag, env.metadata['video.frames_per_second'], resolution)
    #     writer = writer_for_mp4(tag, env.metadata['video.frames_per_second'], resolution)

    #     for _ in range(cycles):
    #         for r, done, info, img in trajectory_generator(env, policy, noise, resolution, camera):
    #             if flip: img = cv2.rotate(img, cv2.ROTATE_180)
    #             writer.write(img)
    #             if quit_on_success and info['success']:
    #                 break

    #     writer.release()
    #     num += 1
    #     print(f'Finished {num}/{len(config)}')
    #     break


if __name__ == '__main__':
    main()
