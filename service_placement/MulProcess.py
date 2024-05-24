from multiprocessing import Pool
import multiprocessing
import os, torch, random, numpy
import time 
def process_task(task_data):
    # 这里处理你的任务
    print("start process task: ", time.time())
    return f"处理结果: {task_data}"

def main():
    # 创建进程池，池中有4个进程
    multiprocessing.set_start_method('spawn')
    print("Gen start time : ", time.time())
    with Pool(4) as pool:
        for i in range(5):  # 假设循环5次
            # 定义一些任务，每次循环任务不同
            tasks = [x + i*5 for x in range(1, 6)]
            # map 方法将任务分配给进程池中的进程
            results = pool.map(process_task, tasks)
            print(results)

if __name__ == '__main__':
    main()
