import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import joblib
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# 定义文件监控的类
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, folder_to_monitor, generate_plot_func, wait_time=5):
        self.folder_to_monitor = folder_to_monitor
        self.generate_plot_func = generate_plot_func
        self.wait_time = wait_time  # 文件大小不变的等待时间
    
    def on_created(self, event):
        if event.is_directory:
            return

        # 只处理 .dat 文件
        if event.src_path.endswith(".dat"):
            file_path = event.src_path
            png_file_path = file_path.replace('.dat', '.png')
            
            # 检查是否已经有对应的 .png 文件
            if not os.path.exists(png_file_path):
                print(f"新文件检测到: {file_path}")
                # 等待文件完成写入后再处理
                self.process_dat_file(file_path)

    def wait_for_file_complete(self, file_path):
        """等待文件写入完成，返回 True 表示文件已完成写入"""
        previous_size = -1
        while True:
            current_size = os.path.getsize(file_path)
            if current_size == previous_size:
                # 文件大小不再变化，认为写入完成
                return True
            previous_size = current_size
            time.sleep(self.wait_time)

    def process_dat_file(self, file_path):
        # 等待文件完成写入
        if not self.wait_for_file_complete(file_path):
            print(f"文件写入未完成，跳过处理: {file_path}")
            return

        try:
            res = joblib.load(file_path, mmap_mode='r')
            print("数据加载成功")
        except Exception as e:
            print(f"Joblib 加载失败: {e}")
            try:
                with open(file_path, 'rb') as f:
                    res = pickle.load(f)
                print("Pickle 加载成功")
            except Exception as e:
                print(f"Pickle 加载失败: {e}")
                return

        # 生成损失图
        self.generate_plot_func(res, file_path.replace('.dat', '.png'))

def generate_combined_loss_plot(res: Dict[str, Any], save_path: str):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate epochs and steps
    epochs = len(res['info']) / (200 / len(res['info'][0]['idx']))
    steps_per_epoch = 200 / len(res['info'][0]['idx'])

    # Training loss plot
    x_train = np.arange(len(res['train_losses'])) / steps_per_epoch
    ax.plot(x_train, res['train_losses'], label='Training Loss', alpha=0.7)

    # Main model loss plot
    x_main = np.arange(len(res['main_losses']))
    ax.plot(x_main, res['main_losses'], label='Main Model Loss', alpha=0.7)

    # Set labels and title
    ax.set_title('Training and Main Model Losses vs. Epochs', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    # Add text with additional information
    info_text = f"Total Epochs: {epochs:.1f}\n" \
                f"Steps per Epoch: {steps_per_epoch:.0f}\n" \
                f"Batch Size: {len(res['info'][0]['idx'])}\n" \
                f"Alpha: {res['alpha']}"
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss plot saved to: {save_path}")

# 扫描文件夹中的现有 .dat 文件并为其生成 .png
def process_existing_dat_files(folder_path, wait_time=5):
    print(f"扫描现有的 .dat 文件: {folder_path}")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".dat"):
            dat_file_path = os.path.join(folder_path, file_name)
            png_file_path = dat_file_path.replace('.dat', '.png')

            # 如果还没有对应的 .png 文件，生成损失图
            if not os.path.exists(png_file_path):
                print(f"处理现有文件: {dat_file_path}")
                handler = FileChangeHandler(folder_path, generate_combined_loss_plot, wait_time)
                handler.process_dat_file(dat_file_path)

# 监控文件夹
def monitor_folder(folder_path, wait_time=5):
    event_handler = FileChangeHandler(folder_path, generate_combined_loss_plot, wait_time)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    print(f"正在监控文件夹: {folder_path}")
    try:
        while True:
            time.sleep(1)  # 每秒检查一次
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# 使用示例，指定要监控的文件夹路径
if __name__ == "__main__":
    folder_to_monitor = "/Volumes/Mac_Ext/link_cache/codes/sgd-influence/experiment/Sec71/mnist_dnn"

    # 处理文件夹中已经存在的 .dat 文件
    process_existing_dat_files(folder_to_monitor)

    # 开始监控新文件
    monitor_folder(folder_to_monitor)
