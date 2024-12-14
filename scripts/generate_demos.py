import os
import functools

import cv2
import numpy as np

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise
import imageio
from tqdm import tqdm
import h5py

from envs.metaworld_envs.metaworld import create_wrapped_env


resolution = (640, 480)
camera = 'corner2' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False # if True, flips output image 180 degrees

environment_to_instruction = {
    'assembly-v2': 'assembling',
    'basketball-v2': 'playing basketball',
    'bin-picking-v2': 'picking bin',
    'box-close-v2': 'closing box',
    'button-press-topdown-v2': 'pressing button',
    'button-press-topdown-wall-v2': 'pressing button',
    'button-press-v2': 'pressing button',
    'button-press-wall-v2': 'pressing button',
    'coffee-button-v2': 'pressing button',
    'coffee-pull-v2': 'pulling cup',
    'coffee-push-v2': 'pushing cup',
    'dial-turn-v2': 'turning dial.',
    'disassemble-v2': 'disassembling',
    'door-close-v2': 'closing door',
    'door-lock-v2': 'locking door',
    'door-open-v2': 'opening door',
    'door-unlock-v2': 'unlocking door',
    'hand-insert-v2': 'inserting bin',
    'drawer-close-v2': 'closing drawer',
    'drawer-open-v2': 'opening drawer',
    'faucet-open-v2': 'opening faucet',
    'faucet-close-v2': 'closing faucet',
    'hammer-v2': 'hammering nail',
    'handle-press-side-v2': 'pressing handle',
    'handle-press-v2': 'pressing handle',
    'handle-pull-side-v2': 'pulling handle',
    'handle-pull-v2': 'pulling handle',
    'lever-pull-v2': 'pulling lever.',
    'peg-insert-side-v2': 'inserting peg',
    'pick-place-wall-v2': 'placing bin to shelf',
    'pick-out-of-hole-v2': 'picking bin',
    'reach-v2': 'reaching red',
    'push-back-v2': 'pushing bin back.',
    'push-v2': 'pushing bin',
    'pick-place-v2': 'placing bin to shelf',
    'plate-slide-v2': 'sliding plate',
    'plate-slide-side-v2': 'sliding plate',
    'plate-slide-back-v2': 'sliding plate',
    'plate-slide-back-side-v2': 'sliding plate',
    'peg-unplug-side-v2': 'unpluging peg',
    'soccer-v2': 'kicking soccer ball',
    'stick-push-v2': 'pushing stick',
    'stick-pull-v2': 'pulling stick',
    'push-wall-v2': 'pushing bin',
    'reach-wall-v2': 'reaching red',
    'shelf-place-v2': 'placing bin to shelf',
    'sweep-into-v2': 'sweeping bin',
    'sweep-v2': 'sweeping bin',
    'window-open-v2': 'opening window',
    'window-close-v2': 'closing window'
}

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
    collect_num = 50
    config_range = (0,len(config))
    
    base_path = './data/h5_buffers/orig/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    h5_traj = h5py.File(os.path.join(base_path, 'metaworld_traj_50_demos.h5'), 'w')
    # h5_video = h5py.File(os.path.join(base_path, 'metaworld_window_video.h5'), 'w')

    state_list = []
    next_state_list = []
    action_list = []
    reward_list = []
    done_list = []
    string_list = []
    video_list = []

    demo_number = 0

    for config_idx in tqdm(range (config_range[0], config_range[1])):

        env_name, noise, cycles, quit_on_success = config[config_idx]
        print('Generating data for', env_name, 'with noise', noise)
        tag = env_name + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)
        policy = functools.reduce(lambda a,b : a if a[0] == env_name else b, test_cases_latest_nonoise)[1]
        # env = ALL_ENVS[env_name]()
        # env._partially_observable = False
        # env._freeze_rand_vec = False
        # env._set_task_called = True

        success_num = 0
        # env_name += '-goal-hidden'



        for i in range (collect_num + 10):


            # env = ALL_ENVS[env_name]()
            env = create_wrapped_env(env_name, language_features=None, success_bonus=False, use_simulator_reward=True, goal_observable=True)()
            # breakpoint()
            env = env.base_env
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(i)
            env.reset()

            env.reset_model()
            o = env.reset()
            rollout_success = False
            imgs = [env.sim.render(*resolution, mode='offscreen', camera_name=camera).astype(np.uint8) ]
            action_space_ptp = env.action_space.high - env.action_space.low
            
            temp_state_list = []
            temp_next_state_list = []
            temp_action_list = []
            temp_reward_list = []
            temp_done_list = []
            temp_string_list = []
            temp_video_list = []

            # if env_name not in h5_traj:
                # h5_traj.create_group(env_name)
                # h5_video.create_group(env_name)

            for step in range(env.max_path_length):
                temp_state_list.append(o)
                a = policy.get_action(o)
                a = np.random.normal(a, noise * action_space_ptp)
                # clip action to be within the action space
                a = np.clip(a, env.action_space.low, env.action_space.high)

                o, r, done, info = env.step(a)
                # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
                imgs.append(env.sim.render(*resolution, mode='offscreen', camera_name=camera))
                temp_next_state_list.append(o)
                temp_action_list.append(a)
                temp_string_list.append(environment_to_instruction[env_name])

                if info['success']:
                    print("success")
                    rollout_success = True
                    success_num += 1
                    done = 1
                    temp_reward_list.append(r)
                    temp_done_list.append(done)
                    break
                else:
                    done = 0
                    temp_reward_list.append(r)
                    temp_done_list.append(done)
                # yield r, done, info, env.sim.render(*res, mode='offscreen', camera_name=camera)[:,:,::-1]
            if rollout_success:
                folder_name = os.path.join(base_path, str(config_idx))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                print(env_name, i, success_num)
                # h5_traj[env_name].create_group(str(i))

                # h5_video[env_name].create_group(str(i))
                # h5_video[env_name][str(i)]['video'] = np.array(imgs)

                # corrupts quickly, so let's save here instead

                h5_traj.create_group(str(demo_number))
                h5_traj[str(demo_number)]['state'] = np.array(temp_state_list)
                h5_traj[str(demo_number)]['next_state'] = np.array(temp_next_state_list)
                h5_traj[str(demo_number)]['action'] = np.array(temp_action_list)
                h5_traj[str(demo_number)]['reward'] = np.array(temp_reward_list)
                h5_traj[str(demo_number)]['done'] = np.array(temp_done_list)
                h5_traj[str(demo_number)]['string'] = np.array(temp_string_list).astype('S')
                h5_traj[str(demo_number)]['img'] = np.array(imgs)
                demo_number += 1

                # video_list.extend(imgs)                
                # state_list.extend(temp_state_list)
                # next_state_list.extend(temp_next_state_list)
                # action_list.extend(temp_action_list)
                # reward_list.extend(temp_reward_list)
                # done_list.extend(temp_done_list)
                # string_list.extend(temp_string_list)


            if success_num > collect_num:
                break
                # gif_file_name = os.path.join(folder_name, "output_gif_{}.gif".format(str(success_num)))
                # imageio.mimsave(gif_file_name, imgs, fps=30)

    # Create groups for state/action/reward/next_state/done
    # h5_traj.create_group('state')
    # h5_traj.create_group('next_state')
    # h5_traj.create_group('action')
    # h5_traj.create_group('reward')
    # h5_traj.create_group('done')
    # h5_traj.create_group('string')

    # h5_traj['state'] = np.array(state_list)
    # h5_traj['next_state'] = np.array(next_state_list)
    # h5_traj['action'] = np.array(action_list)
    # h5_traj['reward'] = np.array(reward_list)
    # h5_traj['done'] = np.array(done_list)
    # h5_traj['string'] = np.array(string_list).astype('S')
    # h5_traj['img'] = np.array(video_list)
    # Now let's flatten the h5 file to be a long list of state-action-reward-next_state-done tuples
    h5_traj.close()
    # h5_video.close()

    # Now let's create a new h5 file that contains keys of only state, action, reward, next_state, done
    # h5_traj = h5py.File(os.path.join(base_path, 'metaworld_window_traj.h5'), 'r')
    # h5_video = h5py.File(os.path.join(base_path, 'metaworld_window_video.h5'), 'r')

    # h5_traj_flat = h5py.File(os.path.join(base_path, 'metaworld_window_traj_flat.h5'), 'w')
    # h5_video_flat = h5py.File(os.path.join(base_path, 'metaworld_window_video_flat.h5'), 'w')

    


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