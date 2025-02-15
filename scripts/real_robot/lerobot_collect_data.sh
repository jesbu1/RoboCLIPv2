PATH_TO_LEROBOT="/home/abrar/koch_arms/lerobot"

cd $PATH_TO_LEROBOT

string_task=$1

# set repo id to usc_koch/string_task
# but convert all spaces to underscores
repo_id=$(echo $string_task | tr ' ' '_')
repo_id="usc_koch_rewind/$repo_id"

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch_bimanual.yaml \
  --fps 30 \
  --repo-id $repo_id \
  --tags rewind \
  --warmup-time-s 10 \
  --episode-time-s 15 \
  --reset-time-s 15 \
  --num-episodes 5 \
  --single-task "$string_task" \
  --push-to-hub 0