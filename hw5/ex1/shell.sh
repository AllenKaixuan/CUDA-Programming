#!/bin/bash

# 定义更有层次的矩阵大小
declare -a sizes=("256 256 256" "512 512 512" "1024 1024 1024" "2048 2048 2048" "4096 4096 4096")

# 清空或创建result.log文件
echo "" > result.log

# 循环执行不同的矩阵大小
for size in "${sizes[@]}"
do
    echo "Running test for matrix size: $size" | tee -a result.log
    ./ex1 $size | tee -a result.log
    echo "" | tee -a result.log
done
