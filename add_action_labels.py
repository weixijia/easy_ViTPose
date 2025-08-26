#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def time_str_to_seconds(time_str):
    """Convert time string (MM.SS.FF) to seconds"""
    minutes, seconds, frame_part = map(float, time_str.split('.'))
    return minutes * 60 + seconds + frame_part / 100

def get_frame_timestamps(video_path):
    """获取视频中每一帧的时间戳"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = []
    frame_count = 0
    
    print("正在读取视频帧时间戳...")
    while True:
        ret = cap.grab()  # 只获取帧，不解码（更快）
        if not ret:
            break
        
        # 获取当前帧的时间戳（毫秒）
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 转换为秒
        timestamps.append(timestamp)
        frame_count += 1
        
        if frame_count % 100 == 0:  # 每100帧显示进度
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    cap.release()
    return timestamps, fps

def find_nearest_action(timestamp, actions):
    """找到最接近当前时间戳的动作标签"""
    for start_time, end_time, action_name in actions:
        start_sec = time_str_to_seconds(start_time)
        end_sec = time_str_to_seconds(end_time)
        if start_sec <= timestamp <= end_sec:
            return action_name
    return "no_action"

def main():
    # 读取视频文件获取每帧时间戳
    video_path = "WorkOut.mov"
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件 {video_path} 不存在")
    
    timestamps, fps = get_frame_timestamps(video_path)
    print(f"\n视频信息：")
    print(f"- FPS: {fps}")
    print(f"- 总帧数: {len(timestamps)}")
    print(f"- 时长: {timestamps[-1]:.2f} 秒")

    # 定义动作时间段
    actions = [
        ("0.09.21", "0.41.91", "Action 1"),
        ("1.07.21", "2.07.17", "Action 2"),
        ("2.29.24", "3.05.19", "Action 3"),
        ("3.26.27", "4.28.03", "Action 4"),
        ("5.09.14", "06.01.16", "Action 5"),
        ("06.19.23", "06.58.22", "Action 6"),
        ("07.16.03", "07.58.01", "Action 7"),
        ("08.18.27", "09.07.28", "Action 8"),
        ("09.31.28", "10.13.12", "Action 9"),
        ("10.36.25", "11.42.03", "Action 10"),
        ("12.00.13", "12.41.19", "Action 11")
    ]

    # 读取原始CSV文件
    csv_path = 'reference_skeletons.csv'
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV文件 {csv_path} 不存在")
    
    df = pd.read_csv(csv_path)
    print(f"\nCSV文件信息：")
    print(f"- 总行数: {len(df)}")
    print(f"- 帧号范围: {df['frame_number'].min()} 到 {df['frame_number'].max()}")
    
    # 添加时间戳和标签列
    df['timestamp'] = df['frame_number'].apply(lambda x: timestamps[x] if x < len(timestamps) else np.nan)
    df['action_label'] = df['timestamp'].apply(lambda x: find_nearest_action(x, actions) if pd.notnull(x) else "no_action")
    
    # 打印每个动作的时间范围和对应帧数
    print("\n动作标签统计：")
    for start_time, end_time, action_name in actions:
        start_sec = time_str_to_seconds(start_time)
        end_sec = time_str_to_seconds(end_time)
        action_frames = df[df['action_label'] == action_name]
        if not action_frames.empty:
            print(f"- {action_name}:")
            print(f"  时间范围: {start_time} 到 {end_time}")
            print(f"  帧数: {len(action_frames)}")
            print(f"  实际时间范围: {action_frames['timestamp'].min():.2f}s 到 {action_frames['timestamp'].max():.2f}s")
    
    # 统计每个标签的帧数
    print("\n标签统计：")
    label_counts = df['action_label'].value_counts()
    for label, count in label_counts.items():
        print(f"- {label}: {count} 帧")
    
    # 保存新的CSV文件
    output_path = 'reference_skeletons_with_labels.csv'
    df.to_csv(output_path, index=False)
    print(f"\n>>> 标签已添加并保存到 {output_path}")

if __name__ == "__main__":
    main() 