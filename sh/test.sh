#!/bin/bash

# 预测
#for file in `ls /work/RsCarData/images/test/`
#do
#    echo $file
#    python /workspace/testTrackingSort.py --model_name DSFNet --gpus 0 --load_model /work/model_75.pth --test_large_size False --datasetname RsCarData  --save_track_results True  --data_dir  /work/RsCarData/images/test/$file/img/
#done

# 记录开始时间
start_time=$(date +%s)

# 控制同时运行的进程数
MAX_JOBS=4

for file in `ls ./autodl-tmp/train1`
do
    echo $file
    python testTrackingDeepSort.py --model_name DSFNet --gpus 0 --load_model ./weights/model_40.pth --test_large_size False --datasetname RsCarData  --save_track_results True  --data_dir  ./autodl-tmp/train1/$file &

    # 检查当前正在运行的进程数，如果达到 MAX_JOBS 就等待
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        wait -n  # 等待任意一个后台任务完成
    done
done

# 记录结束时间
end_time=$(date +%s)

# 计算执行时间（秒）
execution_time=$((end_time - start_time))

# 输出总执行时间
echo "Total execution time: $execution_time seconds"


#cd /output
#zip -r results.zip ./results