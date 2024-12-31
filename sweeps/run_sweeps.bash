#!/bin/bash

# start sweep
## wandb sweep sweep_config.yaml

# Number of agents to run
NUM_AGENTS=2
SWEEP_ID=$1

# Run agents in parallel
for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent $SWEEP_ID > sweeps/log/1_agent_$i.log 2>&1 &
done

# Wait for all agents to finish (optional)
wait
