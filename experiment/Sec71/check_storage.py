import torch
import sys
import numpy as np
import argparse
import warnings

# no future warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_tensor_size(tensor):
    """计算一个 torch.Tensor 的大小（以字节为单位)"""
    return tensor.numel() * tensor.element_size()

def get_size(obj, seen=None):
    """递归计算对象的大小，考虑 torch.Tensor 和 numpy.ndarray"""
    size = 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        size += get_tensor_size(obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        for key, value in obj.items():
            size += get_size(key, seen)
            size += get_size(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += get_size(item, seen)
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += get_size(item, seen)
    return size

# def analyze_netlist(netlist):
#     """分析 NetList 对象的内部结构"""
#     print("Analyzing NetList object:")
#     for attr in dir(netlist):
#         if attr.startswith('_'):
#             continue  # 跳过私有属性
#         try:
#             value = getattr(netlist, attr)
#             size = get_size(value)
#             print(f"  Attribute: {attr}, Size: {size / (1024 ** 2):.2f} MB")
#         except Exception as e:
#             print(f"  Attribute: {attr}, Error accessing: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Analyze a PyTorch checkpoint file')
    parser.add_argument('path', type=str, help='Path to the checkpoint file')
    args = parser.parse_args()

    # 加载文件
    checkpoint = torch.load(args.path, map_location='cpu')  # 确保在CPU上加载

    # 初始化总大小
    total_file_size = 0

    # 计算每个键的大小并累积总大小
    print("\nDetailed analysis of all keys:")
    for key, value in checkpoint.items():
        size = get_size(value)
        total_file_size += size
        print(f"Key: {key}, Type: {type(value)}, Size: {size / (1024 ** 3):.2f} GB")

    # 输出总大小
    print(f"\nTotal size calculated from checkpoint: {total_file_size / (1024 ** 3):.2f} GB")

    # # 如果 'models' 是 NetList 对象，分析它的内部结构
    # if hasattr(checkpoint['models'], 'models'):
    #     models_obj = checkpoint['models']
    #     analyze_netlist(models_obj)

if __name__ == "__main__":
    main()
