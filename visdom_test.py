import threading
import subprocess
import atexit

# 定义启动 Visdom 的函数
def start_visdom_server():
    visdom_process = subprocess.Popen(["python", "-m", "visdom.server", "-p", "7580", "-env_path", "./logs"])

    # 使用 atexit 在程序退出时关闭 Visdom 服务器
    def stop_visdom_server():
        visdom_process.terminate()
        print("Visdom 服务器已关闭")

    atexit.register(stop_visdom_server)

# 创建线程来启动 Visdom 服务器
visdom_thread = threading.Thread(target=start_visdom_server)
visdom_thread.start()

# 模拟一些其他任务
import time
time.sleep(10)  # 模拟运行任务
print("程序结束，Visdom 服务器将被关闭")
