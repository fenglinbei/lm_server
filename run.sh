#!/bin/bash

while true; do
    python api/server.py
    echo "Python script stopped with exit code $?.  Respawning..." >&2
    sleep 10
done
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
sleep 99999999
