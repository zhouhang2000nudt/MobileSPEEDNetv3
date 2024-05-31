#! bin/bash

start_time=$(date +%s)   #记录开始时间

export HF_ENDPOINT=https://hf-mirror.com

# strid为10 不再往外分配
python3 train.py

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"

/usr/bin/shutdown