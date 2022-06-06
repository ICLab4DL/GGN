#!/bin/bash
echo "------------ start --------"

# config path:
data_path=/li_zhengdao/dataset
proj_path=/li_zhengdao/github/GGN

k=$1
if [ ! -n "$k" ];then
    k="train"
fi

echo $k
pid=$(ps -ef | grep eeg_main | grep -v grep | awk '{print $2}')
if [ -n "$pid" ]; then
    echo "running testing: $pid" 
    kill -9 $pid
    echo "killed!"
fi

if [ $k = "kill" ]; then
    echo "only kill seizure process"
    exit 1
fi

# TUH:
echo "start running testing!"

training_tag=just_for_test

nohup python -u $proj_path/eeg_main.py \
--testing \
--arg_file=$proj_path/args/ggn_best.json \
--best_model_save_path=$proj_path/best_models/ggn_best.pth \
> $proj_path/logs/$training_tag.log 2>&1 &

