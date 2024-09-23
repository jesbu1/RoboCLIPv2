import os
import io
import json
import wandb
import random
import joblib
import imageio
import argparse

import torch as th
import numpy as np
from gym import Env
from gym.spaces import Box
import torch.nn.functional as F
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from gym.wrappers.time_limit import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from eval_utils import eval_policys
from encoders import xclip_encoder


id_task = json.load(open("../id_task.json", "r"))

class SingleLayerMLP(th.nn.Module):
    '''
    A linear transformation layer. The output will norm to 1, if normalize is set to True.
    '''
    def __init__(self, input_dim, output_dim, normalize=True):
        super(SingleLayerMLP, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.normalize = normalize

    def forward(self, x):
        x = self.linear(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

def parse_entropy_term(value):
    try:
        return float(value)
    except ValueError:
        return value


def normalize_embeddings(embeddings, return_tensor=True):
    '''
    Normalize the embeddings to have unit norm.
    If return_tensor is set to True, return th.tensor, otherwise return np.ndarray.
    '''
    if isinstance(embeddings, np.ndarray):
        embeddings = th.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--encoder', type=str, default='xclip')
    parser.add_argument('--text_string', type=str, default='opening door')
    parser.add_argument('--env_id', type=str, default='window-open-v2-goal-hidden')
    parser.add_argument('--total_time_steps', type=int, default=1000000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--video_freq', type=int, default=10000)
    parser.add_argument('--succ_end', action="store_true")
    parser.add_argument('--pca', action="store_true")
    parser.add_argument('--model_base_path', type=str, default=None)
    parser.add_argument('--transform_model_path', type=str, default=None)
    parser.add_argument('--random_reset', action="store_true")
    parser.add_argument('--target_gif_path', type=str, default="/scr/jzhang96/metaworld_generate_gifs/")
    parser.add_argument('--frame_num', type=int, default=32)
    parser.add_argument('--time_reward', type=float, default=1.0)
    parser.add_argument('--succ_bonus', type=float, default=0.0)
    parser.add_argument('--xclip_model', type=str, default='microsoft/xclip-base-patch16-zero-shot')
    parser.add_argument('--frame_length', type=int, default=32)
    parser.add_argument("--exp_name_end", type=str, default=None)
    parser.add_argument("--sparse_only", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--obs_env", action="store_true")
    parser.add_argument("--ep_length", type=int, default=128)

    args = parser.parse_args()
    return args

class MetaworldSparse(Env):
    '''
    Training metaworld environments

    '''
    def __init__(self, args):
        super(MetaworldSparse,self)
        self.args = args
        if args.encoder == 'xclip':
            self.encoder = xclip_encoder.XCLIPEncoder()
        else:
            raise ValueError("Please provide a valid encoder")
        if args.obs_env:
            self.env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.env_id]
        else:
            self.env_class = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.rank = args.seed
        self.baseEnv = self.env_class(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=args.ep_length)
        self.env.action_space.seed(self.rank)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if args.pca:
            pca_text_path = os.path.join(args.model_base_path, 'pca_model_text.pkl') 
            pca_video_path = os.path.join(args.model_base_path, 'pca_model_video.pkl') 
            pca_text_model = joblib.load(pca_text_path)
            pca_video_model = joblib.load(pca_video_path)
            self.pca_text_model = pca_text_model
            self.pca_video_model = pca_video_model
        self.past_observations = []
        self.past_dense_reward = []
        self.frame_length = args.frame_length

        # if only use sparse reward, we don't need to load the VLM and compute the similarity reward
        if not args.sparse_only:
            with th.no_grad():
                if not args.baseline: # baseline is RoboCLIPv1, only use the similarity from VLM output, not using VLM
                    # load transform layer model
                    if args.pca:
                        pca_dim = pca_video_model.components_.shape[0]
                        self.transform_model = SingleLayerMLP(pca_dim, pca_dim, normalize=True)
                    else:
                        self.transform_model = SingleLayerMLP(512, 512, normalize=True)
                    transform_model_path = os.path.join(args.model_base_path, args.transform_model_path)
                    dict = th.load(transform_model_path)
                    if 'model_state_dict' in dict.keys():
                        self.transform_model.load_state_dict(dict["model_state_dict"])
                    else:
                        self.transform_model.load_state_dict(dict)
                    self.transform_model = self.transform_model.eval().cuda()
                self.target_embedding = None

                if args.text_string: 
                    self.target_embedding = self.encoder.encode_text(args.text_string)
                    if args.pca:
                        self.target_embedding = th.from_numpy(self.pca_text_model.transform(self.target_embedding.cpu())).cuda()

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self):
        frame = self.env.render()
        return frame

    def compute_similarity_reward(self, video_embedding):
        video_embedding = normalize_embeddings(video_embedding, return_tensor=True).float() 
        similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
        reward = similarity_matrix.detach().cpu().numpy()[0][0]
        if self.args.time_reward != 1.0:
            reward = reward * self.args.time_reward
        return reward


    def step(self, action):
        obs, dense_reward, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.past_dense_reward.append(dense_reward)
        
        if self.args.succ_end:
            if info['success']:
                done = True

        if done:
            if self.args.sparse_only:
                if info['success']:
                    reward = 1.0 * self.args.time_reward
                else:
                    reward = 0.0

            else:
                with th.no_grad():
                    video_embedding = self.encoder.encode_video(self.past_observations)
                    if self.args.pca:
                        video_embedding = th.from_numpy(self.pca_video_model.transform(video_embedding.cpu())).float().cuda()
                    if not self.args.baseline:
                        video_embedding = self.transform_model(video_embedding)
                    reward = self.compute_similarity_reward(video_embedding)
                    info['roboclip_reward'] = reward
                    info['dense_return'] = sum(self.past_dense_reward)
                    info['dense_reward'] = dense_reward
                    info['ep_length'] = len(self.past_dense_reward)
                    if self.args.succ_bonus > 0:
                        if info['success']:
                            reward += self.args.succ_bonus

                    info["total_reward"] = reward
                return obs, reward, done, info
        info['roboclip_reward'] = 0.0
        info['dense_return'] = 0.0
        info['ep_length'] = 0.0
        info['dense_reward'] = dense_reward
        return obs, 0, done, info

    def reset(self):
        self.past_observations = []
        self.past_dense_reward = []

        if self.args.random_reset:
            self.rank = random.randint(0, 400)
            self.baseEnv = self.env_class(seed=self.rank)
            self.env = TimeLimit(self.baseEnv, max_episode_steps=self.args.ep_length)
            self.env.action_space.seed(self.rank)

        return self.env.reset()


class MetaworldDense(Env):
    def __init__(self, args):
        super(MetaworldDense, self)
        if args.obs_env:
            self.env_class = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.env_id]
        else:
            self.env_class = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[args.env_id]
        self.args = args
        self.rank = args.seed
        self.baseEnv = self.env_class(seed=self.rank)
        self.env = TimeLimit(self.baseEnv, max_episode_steps=args.ep_length)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.past_observations = []

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)      
    
    def render(self, camera_name="topview"):
        frame = self.env.render()
        return frame

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = 0.0 # original reward is always 0
        if self.args.succ_end:
            if info['success']:
                done = True
        if info["success"]:
            reward += 1

        return obs, reward, done, info
        
    def reset(self):
        self.counter = 0

        if self.args.random_reset:
            self.rank = random.randint(400, 500)
            self.baseEnv = self.env_class(seed=self.rank)
            self.env = TimeLimit(self.baseEnv, max_episode_steps=self.args.ep_length)
        return self.env.reset()


def make_env(args, eval = False):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if not eval:
                env = MetaworldSparse(args)
        else:
            env = MetaworldDense(args)
        env = Monitor(env, os.path.join(log_dir, str(args.seed)))
        return env
    return _init



class CustomWandbCallback(WandbCallback):
    # def _on_rollout_end(self):
    #     # Log episode metrics with environment steps as x-axis
    #     wandb.log({
    #         'episode_reward': sum(self.locals['rewards']),  # Cumulative reward for the episode
    #         'episode_length': len(self.locals['rewards'])   # Length of the episode
    #     }, step=self.model.num_timesteps)
            
    def _on_step(self):
        # Log training metrics
        # print done
        #if done and done is True, log the info

        done_array = self.locals["dones"]
        infos = self.locals["infos"]
        
        for i, done in enumerate(done_array):
            if done:

                succ = infos[i].get('success', 0)
                roboclip_reward = infos[i].get('roboclip_reward', 0)
                total_reward = infos[i].get('total_reward', 0)
                dense_return = infos[i].get('dense_return', 0)
                dense_reward = infos[i].get('dense_reward', 0)
                ep_length = infos[i].get('ep_length', 0)
                print("episode logged", self.num_timesteps)
                wandb.log({"episode_info/episode_success": succ,
                            "episode_info/roboclip_reward": roboclip_reward,
                            "episode_info/RoboCLIP_bonus_reward": total_reward,
                            "episode_info/dense_return": dense_return,
                            "episode_info/dense_reward": dense_reward,
                            "episode_info/ep_length": ep_length}, step = self.num_timesteps)
                




        return True







class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq

    def _on_step(self) -> bool:
        result = super(CustomEvalCallback, self)._on_step()

        if self.video_freq > 0 and self.n_calls % self.video_freq == 0:
            video_buffer = self.record_video()
            # wandb.log({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            wandb.log({f"eval/evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, step = self.num_timesteps)
            print("video logged")



    def record_video(self):
        frames = []
        obs = self.eval_env.reset()

        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode='rgb_array')
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, _, info = self.eval_env.step(action)

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format='mp4', fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer



def main():
    global args
    global log_dir
    args = get_args()

    # set seed
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    


    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    if args.pca:
        experiment_name = "ep" + str(args.ep_length) + "_PCA_" + "xclip_textTRANS_" + args.env_id
    else:
        experiment_name = "ep" + str(args.ep_length) + "_NOPCA_" +"xclip_textTRANS_" + args.env_id

    if args.succ_end:
        experiment_name = experiment_name + "_SuccEnd"

    experiment_name = experiment_name + args.exp_name_end
    run_group = experiment_name + "NEWDEBUG"
    experiment_name = experiment_name + "_" + str(args.seed) + "NEW"

    run = wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        group=run_group,
        config=args,
        name=experiment_name,
        monitor_gym=True,
        sync_tensorboard=False,
    )


    column1 = ["text_string"]
    table1 = wandb.Table(columns=column1)
    table1.add_data([args.text_string])  

    column2 = ["env_id"]
    table2 = wandb.Table(columns=column2)
    table2.add_data([args.env_id])  
    wandb.log({"text_string": table1, "env_id": table2})


    log_dir = f"/scr/jzhang96/logs/baseline_logs/{experiment_name}"
    # log_dir = f"/home/jzhang96/logs/baseline_logs/{experiment_name}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.n_envs > 1:
        envs = SubprocVecEnv([make_env(args, eval = False) for i in range(args.n_envs)])
    else:
        envs = DummyVecEnv([make_env(args, eval = False)])

    if args.n_envs > 1:
        eval_env = SubprocVecEnv([make_env(args, eval = True) for i in range(args.n_envs)])
    else:
        eval_env = DummyVecEnv([make_env(args, eval = True)]) 

    model = SAC("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, gradient_steps = args.n_envs,
                ent_coef="auto", buffer_size=args.total_time_steps, learning_starts=1000, seed=args.seed)

    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir, 
                                    log_path=log_dir, eval_freq=args.eval_freq, video_freq=args.video_freq,
                                    deterministic=True, render=False, n_eval_episodes = 25)
     
    customwandbcallback = CustomWandbCallback()
    callback = CallbackList([eval_callback, customwandbcallback])
    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/{experiment_name}")

    # Evaluate the agent
    # load the best model

    model = SAC.load(f"{log_dir}/best_model")
    success_rate = eval_policys(args, MetaworldDense, model)
    wandb.log({"eval_SR/evaluate_succ": success_rate}, step = 0)


if __name__ == '__main__':
    main()