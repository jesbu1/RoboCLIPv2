#!/bin/bash

_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-base}"

__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

# Restore our "indended" default env
conda activate "${_CONDA_DEFAULT_ENV}"
# This just logs the output to stderr for debugging. 
>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

conda activate roboclip
pip install -e mjrl
pip install -e Metaworld
pip install -e kitchen_alt
pip install -e kitchen_alt/kitchen/envs
exec "$@"