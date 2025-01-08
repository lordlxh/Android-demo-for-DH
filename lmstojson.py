import numpy as np
import json
import os

def process_landmark_file(lms_path):
    """
    处理每个landmark文件，计算并返回坐标
    """
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    
    lms = np.array(lms_list, dtype=np.int32)
    xmin = int(lms[1][0])  # 转换为Python的float类型
    ymin = int(lms[52][1]) # 转换为Python的float类型
    xmax = int(lms[31][0]) # 转换为Python的float类型
    width = xmax - xmin
    ymax = ymin + width
    
    return [xmin, xmax, ymin, ymax]

def save_landmarks_to_json(dataset_dir, json_file):
    """
    遍历landmarks文件夹，读取每个lms文件，计算坐标并保存到json文件中.
    """
    landmarks_dir = os.path.join(dataset_dir, "landmarks")
    
    if not os.path.isdir(landmarks_dir):
        raise ValueError(f"Landmarks directory does not exist: {landmarks_dir}")
    
    landmarks_dict = {}
    
    # 获取文件列表并按顺序排序
    lms_files = sorted(os.listdir(landmarks_dir), key=lambda x: int(os.path.splitext(x)[0]))
    
    # 遍历landmarks文件夹
    for lms_file in lms_files:
        if lms_file.endswith(".lms"):
            filename_without_ext = os.path.splitext(lms_file)[0]
            lms_path = os.path.join(landmarks_dir, lms_file)
            coords = process_landmark_file(lms_path)
            landmarks_dict[filename_without_ext] = coords
    
    # 将数据保存到json文件
    with open(json_file, "w") as f:
        json.dump(landmarks_dict, f, separators=(',', ':'))

    print(f"Landmarks have been saved to {json_file}")

# 示例用法
dataset_dir = "/home/nfs/10109/shuziren/data/head_video/"
json_file = "landmarks.json"
save_landmarks_to_json(dataset_dir, json_file)



