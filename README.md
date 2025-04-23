# Pytorch + StableBaselines3 Implementation of RoboCLIP
This repository contains the implementation for the NeurIPS 2023 paper, [RoboCLIP: One Demonstration is Enough to Learn Robot Policies](https://arxiv.org/abs/2310.07899).

## Setting up the env

We recommend using conda for installation and provide a `.yml` file for installation. 

Might need to `rm -rf Metaworld` and then replace with `git clone git@github.com:sumedh7/Metaworld.git`

```sh
# git clone https://github.com/sumedh7/RoboCLIP.git --recursive
# cd RoboCLIP


conda env create -f lerobot_rl.yml
conda activate lerobot_rl

pip install -r requirements.txt

# Get mjrl
git clone https://github.com/aravindr93/mjrl.git
pip install -e mjrl

# Use metaworld fork
rm -rf Metaworld
git clone https://github.com/sumedh7/Metaworld.git
pip install -e Metaworld

pip install -e kitchen_alt
pip install -e kitchen_alt/kitchen/envs
#wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth # not needed if not running RoboCLIP metaworld experiments
#wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy # not needed if not running RoboCLIP metaworld experiments
git submodule init
git submodule update --recursive
rm -rf reward_model/LIV
git clone https://github.com/penn-pal-lab/LIV reward_model/LIV
pip install -e reward_model/LIV 
pip install -e reward_model/LIV/liv/models/clip

pip install stable_baselines3==1.8.0 --no-deps
pip install stable-baselines3[extra]==1.8.0 --no-deps


# Reinstall pytorch>=2.0. https://pytorch.org/
# Install torch with pip with your cuda version!
pip install --upgrade torch torchvision  # for example

pip install -e .
```


## How To use it ?

To run experiments on the Metaworld environment suite with the sparse learnt reward, we need to first define what the demonstration to be used is. For textual input, uncomment line 222 and comment 223 and add the string prompt you would like to use in the `text_string` param. Similarly, if you would like to use human demonstration, uncomment line 223 and pass the path of the gif of the demonstration you would like to use. Similarly, for a metaworld video demo, set `human=False` and set the `video_path`. 

We provide the gifs used in our experiments within the `gifs/`.
Then run: 
```sh
python metaworld_envs.py --env-type sparse_learnt --env-id drawer-open-v2-goal-hidden --dir-add <add experiment identifier>
```

To run the Kitchen experiments, similarly specify the gif path on line 345 and then run the following line with `--env-id` as `Kettle`, `Hinge` or `Slide`. 

```sh
python kitchen_env_wrappers.py --env-type sparse_learnt --env-id Kettle --dir-add <add experiment identifier>
```

These runs should produce default tensorboard experiments which save the best eval policy obtained by training on the RoboCLIP reward to disk. The plots in the paper are visualized by finetuning these policies for a handful of episodes. To replicate the Metaworld finetuning,  run:

```sh
python metaworld_envs.py --env-type dense_original --env-id drawer-open-v2-goal-hidden --pretrained <path_to_best_policy> --dir-add <add_experiment_identifier>  
```
## FAQ for Debugging
Please use the older version of Metaworld, i.e., pre Farama Foundation. Also rendering can be an issue sometimes, so setting the right renderer is necessary. We found `egl` to be useful. 
```sh
export MUJOCO_GL=egl
```



Generate demos:
```
python scripts/generate_demos.py
```

Label rewards:
```
python scripts/label_rewards.py --trajs_to_label data/h5_buffers/orig/metaworld_window_traj.h5 --encoder_type xclip --out data/h5_buffers/updated_trajs/metaworld_window_traj_xclip.h5 --sparse_only
```