#!/bin/bash


params1=(256 128 64)
params2=(512 256 128)
params3=(1024 512 256)


> output.log


for p1 in "${params1[@]}"; do
  for p2 in "${params2[@]}"; do
    for p3 in "${params3[@]}"; do
      ./a.out $p1 $p2 $p3 >> output.log
    done
  done
done