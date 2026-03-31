#!/usr/bin/env bash
# Submit the full single-server Robometer pipeline:
# 1. start one shared server
# 2. after server starts:
#    - baseline offline (3 seeds)
#    - diff gamma=0.99 offline (3 seeds)
#    - diff gamma=0.999 labels
# 3. after baseline offline succeeds -> baseline online (3 seeds)
# 4. after gamma=0.99 offline succeeds -> gamma=0.99 online (3 seeds)
# 5. after gamma=0.999 labels succeed -> gamma=0.999 offline (3 seeds)
# 6. after gamma=0.999 offline succeeds -> gamma=0.999 online (3 seeds)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs

SERVER_JOB=$(sbatch --parsable scripts/robometer_server_doorlock_pipeline.sbatch)

BL_OFF_JOB=$(sbatch --parsable --dependency=after:${SERVER_JOB} \
  scripts/robometer_offline_baseline_bonus0_3seeds.sbatch)

G099_OFF_JOB=$(sbatch --parsable --dependency=after:${SERVER_JOB} \
  scripts/robometer_offline_diff_gamma099_bonus0_scaled_3seeds.sbatch)

G0999_LABEL_JOB=$(sbatch --parsable --dependency=after:${SERVER_JOB} \
  scripts/robometer_label_gamma0999_pipeline.sbatch)

BL_ON_JOB=$(sbatch --parsable --dependency=afterok:${BL_OFF_JOB} \
  scripts/robometer_online_baseline_base_reward_doorlock_3seeds.sbatch)

G099_ON_JOB=$(sbatch --parsable --dependency=afterok:${G099_OFF_JOB} \
  scripts/robometer_online_diff_gamma099_scaled_base_reward_doorlock_3seeds.sbatch)

G0999_OFF_JOB=$(sbatch --parsable --dependency=afterok:${G0999_LABEL_JOB} \
  scripts/robometer_offline_diff_gamma0999_bonus0_scaled_3seeds.sbatch)

G0999_ON_JOB=$(sbatch --parsable --dependency=afterok:${G0999_OFF_JOB} \
  scripts/robometer_online_diff_gamma0999_scaled_base_reward_doorlock_3seeds.sbatch)

echo "Submitted Robometer door-lock pipeline:"
echo "  Server job:            $SERVER_JOB"
echo "  Baseline offline:      $BL_OFF_JOB"
echo "  Gamma099 offline:      $G099_OFF_JOB"
echo "  Gamma0999 labels:      $G0999_LABEL_JOB"
echo "  Baseline online:       $BL_ON_JOB"
echo "  Gamma099 online:       $G099_ON_JOB"
echo "  Gamma0999 offline:     $G0999_OFF_JOB"
echo "  Gamma0999 online:      $G0999_ON_JOB"
echo
echo "Dependencies:"
echo "  server -> {baseline offline, gamma099 offline, gamma0999 labels}"
echo "  baseline offline -> baseline online"
echo "  gamma099 offline -> gamma099 online"
echo "  gamma0999 labels -> gamma0999 offline -> gamma0999 online"
echo
echo "Shared server info file:"
echo "  logs/robometer_server_doorlock_pipeline_info.txt"
echo
echo "Monitor with: squeue -u \$USER"
