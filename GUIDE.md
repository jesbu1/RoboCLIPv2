# Robometer No-Action-Chunk Training Guide

Complete pipeline from clone to online training. Run each step manually
and verify it finishes before moving on.

## Step 1: Clone

```bash
git clone git@github.com:Jack20030123/rewind_no-action-chunk.git
cd rewind_no-action-chunk
```

## Step 2: Create robometer server conda env

```bash
conda create -p conda_envs/robometer python=3.10 -y
conda activate conda_envs/robometer
pip install robometer fastapi uvicorn torch
```

## Step 3: Create training conda env

```bash
conda create -p conda_envs/rewind_nochunk python=3.10 -y
conda activate conda_envs/rewind_nochunk
pip install torch torchvision stable-baselines3 sb3-contrib hydra-core \
    omegaconf wandb h5py transformers gym metaworld mujoco \
    opencv-python scipy numpy requests
pip install -e Metaworld
```

## Step 4: Prepare raw data

The two raw h5 files come from the rewind repo
(`data_generation/metaworld_generation.py` and
`data_preprocessing/generate_dino_embeddings.py`).
If you already have them, just symlink:

```bash
mkdir -p datasets logs
ln -s /path/to/your/rewind/datasets/metaworld_generation.h5 datasets/
ln -s /path/to/your/rewind/datasets/metaworld_embeddings_train.h5 datasets/
```

## Step 5: Start 2 robometer servers (for labeling + offline)

```bash
sbatch scripts/robometer_server_baseline.sbatch
sbatch scripts/robometer_server_gamma099.sbatch
```

Wait until both are running:
```bash
squeue -u $USER  # wait for 2 robo_srv jobs to be in R state
```

## Step 6: Label datasets

```bash
sbatch scripts/robometer_label_baseline.sbatch
sbatch scripts/robometer_label_gamma099.sbatch
```

Wait and verify:
```bash
squeue -u $USER  # wait until robo_lbl jobs disappear
ls -lh datasets/metaworld_labeled_robometer.h5
ls -lh datasets/metaworld_labeled_robometer_diff_gamma099.h5
```

## Step 7: Offline training

```bash
sbatch scripts/robometer_offline_baseline.sbatch
sbatch scripts/robometer_offline_gamma099.sbatch
```

Wait and verify:
```bash
squeue -u $USER  # wait until robo_off jobs disappear
ls logs/offline_robometer_baseline_seed0/last_offline
ls logs/offline_robometer_diff_gamma099_seed0/last_offline
```

## Step 8: Online training

Reuses the 2 servers from Step 5, starts 6 additional servers
(ports 8002-8007), and submits online training for both variants.

```bash
bash scripts/run_robometer_online.sh
squeue -u $USER  # monitor
```

The 8 environments are:

| Task ID | Environment |
|---------|-------------|
| 0 | window-close-v2 |
| 1 | reach-wall-v2 |
| 2 | faucet-close-v2 |
| 3 | coffee-button-v2 |
| 4 | button-press-wall-v2 |
| 5 | door-lock-v2 |
| 6 | handle-press-side-v2 |
| 7 | sweep-into-v2 |

## Notes

- **SLURM account**: All sbatch scripts use `--account=biyik_1165`.
  Change this if your account is different.
- **Server timeout**: Servers have a 48h time limit. Cancel and restart
  them if they expire before training finishes.
- **Logs**: All logs go to `logs/`. Check `.err` files for errors.
- **wandb**: Training logs to wandb project `rewind-policy-training`.
  Make sure you're logged in (`wandb login`).
