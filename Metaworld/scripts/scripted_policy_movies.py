import os
import functools

import cv2
import numpy as np

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise


# resolution = (1920, 1080)
resolution = (640, 480)

# camera = 'behindGripper' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
camera = 'corner2' # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False # if True, flips output image 180 degrees
# flip=True # if True, flips output image 180 degrees

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
    ('peg-insert-side-v2', np.zeros(4), 3, True),
    ('peg-unplug-side-v2', np.zeros(4), 3, True),
    ('soccer-v2', np.zeros(4), 3, True),
    ('stick-push-v2', np.zeros(4), 3, True),
    ('stick-pull-v2', np.zeros(4), 3, True),
    ('push-wall-v2', np.zeros(4), 3, True),
    ('push-v2', np.zeros(4), 3, True),
    ('reach-wall-v2', np.zeros(4), 3, True),
    ('reach-v2', np.zeros(4), 3, True),
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

def writer_for(tag, fps, res):
    if not os.path.exists('../../metaworld_movies_corner'):
        os.mkdir('../../metaworld_movies_corner')
    return cv2.VideoWriter(
        f'../../metaworld_movies_corner/{tag}.avi',
        cv2.VideoWriter_fourcc('M','J','P','G'),
        fps,
        res
    )


def writer_for_mp4(tag, fps, res):
    if not os.path.exists('../../metaworld_movies_corner'):
        os.mkdir('../../metaworld_movies_corner')
    return cv2.VideoWriter(
        f'../../metaworld_movies_corner/{tag}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        res
    )




def main():
    num = 0
    for env, noise, cycles, quit_on_success in config:
        # import pdb ; pdb.set_trace()
        cycles = 1
        tag = env + '-noise-' + np.array2string(noise, precision=2, separator=',', suppress_small=True)

        policy = functools.reduce(lambda a,b : a if a[0] == env else b, test_cases_latest_nonoise)[1]
        env = ALL_ENVS[env]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True

        # writer = writer_for(tag, env.metadata['video.frames_per_second'], resolution)
        writer = writer_for_mp4(tag, env.metadata['video.frames_per_second'], resolution)

        for _ in range(cycles):
            for r, done, info, img in trajectory_generator(env, policy, noise, resolution, camera):
                if flip: img = cv2.rotate(img, cv2.ROTATE_180)
                writer.write(img)
                if quit_on_success and info['success']:
                    break

        writer.release()
        num += 1
        print(f'Finished {num}/{len(config)}')
        break


if __name__ == '__main__':
    main()