from  metaworld_envs_xclip_wandb_fix_reset import MetaworldSparse, MetaworldDense, make_env, CustomEvalCallback
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import joblib
import os
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from stable_baselines3.common.monitor import Monitor


'''
Procedure:
Video --> Video Latent --> Normalized Video Latent --> PCA --> Linear Layer --> Normalized Output
'''

def normalize_embeddings(embeddings, return_tensor=True):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    if return_tensor:
        return normalized_embeddings
    else:
        return normalized_embeddings.detach().numpy()




class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, normalize=False):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)  # First fully connected layer
        self.normalize = normalize

    def forward(self, x):
        x = self.fc(x)  # Output layer
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize the output
        return x


def get_model(dim = 512, normalize = False):
    model = MLP(dim, dim, normalize)
    return model


class MetaworldSparsePCA(MetaworldSparse):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=False, load_model_path=None, pca_model_path=None):
        super().__init__(env_id, text_string=text_string, time=time, video_path=video_path, rank=rank, human=human)
        self.linear_layer = get_model(normalize=True)
        self.load_model_path = load_model_path
        self.load_model()
        pca_text_path = os.path.join(pca_model_path, 'pca_text_model.pkl')
        pca_video_path = os.path.join(pca_model_path, 'pca_video_model.pkl')
        pca_text_model = joblib.load(pca_text_path)
        pca_video_model = joblib.load(pca_video_path)
        self.pca_text_model = pca_text_model
        self.pca_video_model = pca_video_model

        self.target_embedding = normalize_embeddings(self.target_embedding, return_tensor=False)
        self.target_embedding = self.pca_text_model.transform(self.target_embedding)
        self.target_embedding = normalize_embeddings(self.target_embedding, return_tensor=True)
        self.target_embedding = torch.tensor(self.target_embedding).float()

    def load_model(self):
        if self.load_model_path is not None:
            self.linear_layer.load_state_dict(torch.load(self.load_model_path))
            self.linear_layer.eval()

        else:
            print('No model loaded')
            # assert please load a model
            assert self.load_model_path is None

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            with torch.no_grad():
                frames = self.preprocess_metaworld_xclip(self.past_observations)            
                # video = th.from_numpy(frames)
                video_embedding = self.net.get_video_features(frames)
                video_embedding = normalize_embeddings(video_embedding, return_tensor=False)
                video_embedding = self.pca_video_model.transform(video_embedding)

                # convert video embedding to tensor float 32
                video_embedding = torch.tensor(video_embedding).float()
                
                video_embedding = self.linear_layer(video_embedding)
                video_embedding = normalize_embeddings(video_embedding, return_tensor=True)

                # video_embedding = video_output['video_embedding']
                similarity_matrix = torch.matmul(self.target_embedding, video_embedding.t()) * 100

                reward = similarity_matrix.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info


#__init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=False, load_model_path=None, pca_model_path=None



def make_env(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "sparse_learnt":
            env = MetaworldSparsePCA(env_id=env_id, 
                                  text_string=args.text_string, 
                                  time=True, 
                                  rank=rank, 
                                  load_model_path = args.mlp_model_path, 
                                  pca_model_path=args.pca_model_path,
                                  )
            # env = MetaworldSparse(env_id=env_id, video_path="./gifs/human_opening_door.gif", time=True, rank=rank, human=True)

        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text_string', type=str, default='Closing door')
    parser.add_argument('--dir_add', type=str, default='')
    parser.add_argument('--env_id', type=str, default='door-close-v2-goal-hidden')
    parser.add_argument('--env_type', type=str, default='sparse_learnt')
    parser.add_argument('--total_time_steps', type=int, default=200000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_freq', type=int, default=512)
    parser.add_argument('--video_freq', type=int, default=2048)
    parser.add_argument('--pca_model_path', type=str, default='/scr/jzhang96/pca_models')
    parser.add_argument('--mlp_model_path', type=str, default='/scr/jzhang96/mlp_models')

    args = parser.parse_args()
    return args






def main():
    global args
    global log_dir
    args = get_args()

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    


    WANDB_ENTITY_NAME = "clvr"
    WANDB_PROJECT_NAME = "roboclip-v2"
    # wandb_eval_task_name = "_".join([str(i) for i in eval_tasks])
    # experiment_name = args.experiment_name + "_" + wandb_eval_task_name + "_" + str(args.seed)
    # if args.mse:
    #     experiment_name = experiment_name + "_mse_" + str(args.mse_weight)
    experiment_name = "xclip-wandb_fix_random_reset_PCA_x100_" + args.env_id + "_" + str(args.seed)

    if args.wandb:
        run = wandb.init(
            entity=WANDB_ENTITY_NAME,
            project=WANDB_PROJECT_NAME,
            group="x-clip-roboclipv1-train" + args.env_id,
            config=args,
            name=experiment_name,
            monitor_gym=True,
            sync_tensorboard=True,
        )

    

    column1 = ["text_string"]
    table1 = wandb.Table(columns=column1)
    table1.add_data([args.text_string])  

    column2 = ["env_id"]
    table2 = wandb.Table(columns=column2)
    table2.add_data([args.env_id])  
    wandb.log({"text_string": table1, "env_id": table2})




    log_dir = f"/scr/jzhang96/logs/{experiment_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)
    # Use deterministic actions for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                              log_path=log_dir, eval_freq=200,
    #                              deterministic=True, render=False)



    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir,
                                    log_path=log_dir, eval_freq=args.eval_freq, video_freq=args.video_freq,
                                    deterministic=True, render=False)



                                 
    wandb_callback = WandbCallback(verbose = 1)
    callback = CallbackList([eval_callback, wandb_callback])



    model.learn(total_timesteps=int(args.total_time_steps), callback=callback)
    model.save(f"{log_dir}/trained")







if __name__ == '__main__':
    main()





# args = get_args()
# env = MetaworldSparsePCA(env_id=args.env_id, text_string=args.text_string, time=True, rank=0, human=False)