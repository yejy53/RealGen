#!/bin/bash

# 这是一个推荐的实践，脚本会在任何命令失败时立即退出
set -e

# 这里放入你的 torchrun 命令
torchrun --nnodes=1 --nproc_per_node=4 \
    zimage_t2i.py