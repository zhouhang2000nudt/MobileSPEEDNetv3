#! bin/bash

start_time=$(date +%s)

export HF_ENDPOINT=https://hf-mirror.com


python3 train.py


end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"
