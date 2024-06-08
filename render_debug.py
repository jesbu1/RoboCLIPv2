import gym
import os
import numpy as np
import PIL.Image
# from IPython import display

os.environ['MUJOCO_GL'] = 'egl'

# Create the environment
env = gym.make('CartPole-v1')

# Initialize the environment
env.reset()

for _ in range(1000):
    img = env.render(mode='rgb_array')  # Render the environment
    # save img to png
    img = PIL.Image.fromarray(img)
    img.save('cartpole1.png')

    action = env.action_space.sample()  # Take a random action
    env.step(action)  # Step the environment
    import pdb ; pdb.set_trace()

env.close()  