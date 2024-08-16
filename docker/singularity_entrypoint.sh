#!/bin/bash

# Ensure we're using the container's conda
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_SHLVL

__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate roboclip

# Set PYTHONPATH to include the writable directory
export PYTHONPATH=/opt/conda/envs/roboclip/local:${PYTHONPATH}

# Redirect writes to a writable location
# Update paths to /opt instead of /root
export MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco210
export MUJOCO_PY_MJKEY_PATH=/opt/mujoco/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=/opt/mujoco/mujoco210
export MUJOCO_PY_BUILD_DIR=/tmp/mujoco_build

# Unset LD_PRELOAD to avoid errors
unset LD_PRELOAD

# Run the command passed to the container
exec "$@"