import os

workers = 1

def post_fork(server, worker):
    print(f"Worker {worker.pid} is being assigned to CUDA device 6.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 其他配置
worker_class = "sync"
timeout = 120