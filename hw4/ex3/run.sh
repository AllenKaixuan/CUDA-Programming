#!/bin/bash

> output.log

for ((i=100; i<=10000; i+=100)); do
    if [ $i -eq 100 ]; then
        echo "N = $i"
    fi    
    ./plot 1024 $i
    
done






