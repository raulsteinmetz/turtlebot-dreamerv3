agent="${1:-ddpg}"
stage="${2:-1}"
lidar="${3:-10}"

echo "Agent: $agent - Stage: $stage - Lidar=$lidar"
sleep 5


echo "...Training..."
python train.py --agent $agent --stage $stage --lidar $lidar
sleep 5

echo "...Saving models to best_models..."
python save_to_best.py --agent  $agent --stage $stage --lidar $lidar
sleep 5

echo "...Testing..."
python test.py --agent $agent --stage $stage --lidar $lidar
sleep 5
