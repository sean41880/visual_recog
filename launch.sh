#!/bin/bash
# Launch Slurm workers for training tasks
# Usage: bash launch.sh

launch-slurm-workers \
    training_tasks/ \
    --account MST108318 \
    --nodes 2 \
    --gpu-per-task 8 \
    --max-consecutive-fails 3 \
    --log-system-metrics \
    --sb--time 12:00:00 \
    --yes