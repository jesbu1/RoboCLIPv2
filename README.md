# Pytorch + StableBaselines3 Implementation of RoboCLIP
This repository contains the implementation for the NeurIPS 2023 paper, [RoboCLIP: One Demonstration is Enough to Learn Robot Policies](https://arxiv.org/abs/2310.07899).

## RUNNING SINGULARITY
1. `cd RoboCLIPv2 && gdown 1EzNLSeziabyA1cOHSvQt4mna1GeWbMek`
2. Make a `tmp` folder in `RoboCLIPv2`.
3. Just fill in `PYTHON_SCRIPT` and `ACCT`, and possibly change anything else as needed. You can maybe run multiple jobs srun to save resources and time by just using `&` in your `PYTHON_SCRIPT`. Note: you need to have the single quotes around `'PYTHON_SCRIPT'`. We added `k40` as it's the cheapest GPU but change this if needed.
```
srun \ 
    --time=2-00:00:00 \ 
    --ntasks=1 \ 
    --partition=gpu  \
    --cpus-per-task=16 \
    --mem=64G \
    --account=ACCT \
    --gres=gpu:k40:1 \
    bash docker/run_singularity_slurm_script.sh 'PYTHON_SCRIPT'
```


## Setting up the env

We recommend using conda for installation and provide a `.yml` file for installation. 

TODO: S3D: YOU GOTTA MV S3DG.PY FROM S3D_HOWTO100M.



```sh
git clone https://github.com/sumedh7/RoboCLIP.git --recursive
cd RoboCLIP
conda env create -f environment_roboclip.yml
conda activate roboclip
pip install -e mjrl
pip install -e Metaworld
pip install -e kitchen_alt
pip install -e kitchen_alt/kitchen/envs
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
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
