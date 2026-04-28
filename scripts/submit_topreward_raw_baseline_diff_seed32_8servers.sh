#!/usr/bin/env bash
# Submit the raw no-chat TOPReward baseline+diff gamma1 seed32 pipeline.
# Defaults to attention auto mode, which tries FA2 first when the GPU supports it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED="${SEED:-32}"
export TOPREWARD_ATTN_IMPLEMENTATION="${TOPREWARD_ATTN_IMPLEMENTATION:-auto}"

# Keep seed32 server info/log/lock group ids separate from seed0 so both chains
# can be queued or run at the same time without touching each other's files.
export GROUP_IDS_RAW="${GROUP_IDS_RAW:-521 522 523 524 621 622 623 624}"

exec "${SCRIPT_DIR}/submit_topreward_raw_baseline_diff_seed0_8servers.sh" "$@"
