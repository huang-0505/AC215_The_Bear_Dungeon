#!/bin/bash
# start conda environment

# conda activate venv

# Run gcp-pull.py
python gcp-pull.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
  echo "gcp-pull.py failed or no new CSVs downloaded"
  exit 1
fi

echo "-----------------Model training started-----------------"
# Run autotrain with the specified config and redirect output and error to logs
autotrain --config some_ml.yaml 

echo "-----------------Model training completed-----------------"

# Run gcp-push.py
python gcp-push.py

# Check if the previous command was successful
if [ $? -ne 0 ]; then
  echo "gcp-push.py failed"
  exit 1
fi

echo "All tasks completed successfully"
