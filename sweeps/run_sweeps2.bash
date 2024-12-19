#!/bin/bash

# start sweep
## wandb sweep sweep_config.yaml

# Number of agents to run
NUM_AGENTS=4
SWEEP_ID=abraranwar/RoboCLIPv2-test_scripts/2ll1d5fi # Replace with your actual sweep ID

# Run agents in parallel
for i in $(seq 1 $NUM_AGENTS); do
  nohup wandb agent $SWEEP_ID > sweeps/log/2_agent_$i.log 2>&1 &
done

# Wait for all agents to finish (optional)
wait
