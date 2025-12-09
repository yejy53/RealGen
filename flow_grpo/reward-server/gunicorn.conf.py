# import os
# NUM_DEVICES = 8
# USED_DEVICES = set()

# def pre_fork(server, worker):
#     # runs on server
#     global USED_DEVICES
#     worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
#     USED_DEVICES.add(worker.device_id)

# def post_fork(server, worker):
#     # runs on worker
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)

# def child_exit(server, worker):
#     # runs on server
#     global USED_DEVICES
#     USED_DEVICES.remove(worker.device_id)

# workers = NUM_DEVICES
# worker_class = "sync"
# timeout = 120




import os

# Gunicorn Configuration

# 定义端口，你可以根据需要取消注释
# port=18087
# bind = f"0.0.0.0:{port}"

# 1. 将 worker 数量设置为 1
workers = 1

def post_fork(server, worker):
    """
    在 worker 进程启动后运行。
    直接将 CUDA_VISIBLE_DEVICES 设置为 '6'，代表第七张卡。
    """
    print(f"Worker {worker.pid} is being assigned to CUDA device 6.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 其他配置
worker_class = "sync"
timeout = 120