#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.post_processing.one_euro_filter import OneEuroFilter


def is_timestamp_folder(folder_name):
    """
    Check if the folder name is a pure numeric timestamp.
    """
    return folder_name.isdigit()


def process_subject_folder(subject_folder, model, args):
    """
    Process a single subject folder containing camera data.
    
    Args:
        subject_folder: Path to the subject folder
        model: VitInference model
        args: Command line arguments
    """
    subject_id = subject_folder.name
    print(f"\n>>> Processing subject: {subject_id}")
    
    # Check if camera folder exists
    camera_folder = subject_folder / "camera"
    if not camera_folder.exists() or not camera_folder.is_dir():
        print(f"Warning: No camera folder found in {subject_folder}")
        return
    
    # Check if timestamp file exists
    timestamp_file = subject_folder / "timestamp.txt"
    if not timestamp_file.exists():
        print(f"Warning: No timestamp.txt found in {subject_folder}")
        return
    
    # Load timestamps
    timestamps = {}
    try:
        with open(timestamp_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    frame_num = int(parts[0])
                    timestamp = float(parts[1])
                    timestamps[frame_num] = timestamp
    except Exception as e:
        print(f"Error reading timestamp file: {e}")
        return
    
    if not timestamps:
        print(f"Warning: No valid timestamps found in {timestamp_file}")
        return
    
    # Calculate relative times (starting from 0)
    min_timestamp = min(timestamps.values())
    relative_times = {frame: ts - min_timestamp for frame, ts in timestamps.items()}
    
    # Create output directories
    vis_dir = subject_folder / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Prepare CSV output
    output_csv = subject_folder / f"skeletons_{subject_id}.csv"
    
    # Get list of .npy files
    npy_files = list(camera_folder.glob('*.npy'))
    if not npy_files:
        print(f"Warning: No .npy files found in {camera_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files in {camera_folder}")
    
    # Sort .npy files by frame number
    npy_files.sort(key=lambda x: int(x.stem))
    
    # Get keypoint names from the model
    from easy_ViTPose.vit_utils.visualization import joints_dict
    dataset = model.dataset
    keypoint_names = joints_dict()[dataset]['keypoints']
    num_keypoints = len(keypoint_names)
    
    # Initialize smoothing filters for each person if smoothing is enabled
    smoothing_filters = {}
    
    # Prepare CSV header
    header = ['image_name', 'person_id', 'timestamp', 'relative_time']
    for kp_name in keypoint_names:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_score'])
    
    # Process .npy files and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for npy_path in tqdm.tqdm(npy_files, desc=f"Processing {subject_id}"):
            try:
                # Extract frame number from filename
                frame_num = int(npy_path.stem)
                
                # Get timestamp and relative time
                timestamp = timestamps.get(frame_num, 0)
                relative_time = relative_times.get(frame_num, 0)
                
                # Load image
                img = np.load(npy_path)
                
                # Run inference
                # VitInference returns a dictionary where keys are person IDs and values are keypoint arrays
                keypoints_dict = model.inference(img)
                
                # If no keypoints detected or empty
                if not keypoints_dict or len(keypoints_dict) == 0:
                    # Write row with frame number, default person ID and NaN values for all keypoints
                    row = [npy_path.stem, 'none', timestamp, relative_time] + ['NaN'] * (num_keypoints * 3)
                    writer.writerow(row)
                    continue
                
                # If single_pose is True, we only take the pose with the highest average confidence
                if len(keypoints_dict) > 1 and args.single_pose:
                    try:
                        # Calculate average confidence for each pose
                        avg_confidences = {}
                        for person_id, kp in keypoints_dict.items():
                            if isinstance(kp, np.ndarray) and kp.ndim >= 2:
                                avg_confidences[person_id] = np.mean(kp[:, 2])
                            else:
                                avg_confidences[person_id] = 0
                        
                        # Get the person ID with highest confidence
                        best_person_id = max(avg_confidences, key=avg_confidences.get)
                        # Keep only the best pose
                        keypoints_dict = {best_person_id: keypoints_dict[best_person_id]}
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Warning: Error processing keypoints for {npy_path.stem}: {e}")
                        # Just keep the first person as fallback
                        if len(keypoints_dict) > 0:
                            first_id = list(keypoints_dict.keys())[0]
                            keypoints_dict = {first_id: keypoints_dict[first_id]}
                
                # For each detected person, write a row to CSV
                for person_id, kp in keypoints_dict.items():
                    try:
                        # Check if kp is a valid array
                        if not isinstance(kp, np.ndarray) or kp.ndim < 2:
                            print(f"Warning: Invalid keypoint format for {npy_path.stem} person {person_id}. Writing default values.")
                            # Write row with frame number, person ID and NaN values for all keypoints
                            row = [npy_path.stem, f"person_{person_id}", timestamp, relative_time] + ['NaN'] * (num_keypoints * 3)
                            writer.writerow(row)
                            continue
                        
                        # Apply smoothing filter if enabled
                        if args.use_smoothing:
                            # Create a filter for this person if it doesn't exist yet
                            person_key = f"subject_{subject_id}_person_{person_id}"
                            if person_key not in smoothing_filters:
                                # Initialize with first keypoints
                                smoothing_filters[person_key] = OneEuroFilter(
                                    x0=kp[:, :2],  # Only smooth x,y coordinates, not confidence
                                    min_cutoff=args.min_cutoff,
                                    beta=args.beta,
                                    d_cutoff=30.0  # Assuming 30 FPS
                                )
                            
                            # Apply the filter to smooth the keypoints
                            # Only smooth coordinates (first two columns), keep confidence scores as is
                            smoothed_xy = smoothing_filters[person_key](kp[:, :2])
                            
                            # Create a copy of keypoints and update with smoothed values
                            smoothed_kp = kp.copy()
                            smoothed_kp[:, :2] = smoothed_xy
                            kp = smoothed_kp
                        
                        # Flatten keypoints array to [x1, y1, score1, x2, y2, score2, ...]
                        flat_kp = kp.reshape(-1).tolist()
                        
                        # Create row with frame number, person ID, timestamp, relative time, and keypoints
                        row = [npy_path.stem, f"person_{person_id}", timestamp, relative_time]
                        row.extend(flat_kp)
                        
                        writer.writerow(row)
                    except Exception as e:
                        print(f"Error processing keypoint for {npy_path.stem} person {person_id}: {e}")
                        # Write row with frame number and NaN values for all keypoints on error
                        row = [npy_path.stem, f"person_{person_id}", timestamp, relative_time] + ['NaN'] * (num_keypoints * 3)
                        writer.writerow(row)
                
                # Visualize if requested
                if args.visualize:
                    try:
                        # If we're using smoothing, we need to update the model's keypoints with the smoothed ones for visualization
                        if args.use_smoothing:
                            # Create a copy of the original keypoints
                            original_keypoints = model._keypoints.copy()
                            
                            # Update with smoothed keypoints
                            for person_id, kp in keypoints_dict.items():
                                if isinstance(kp, np.ndarray) and kp.ndim >= 2:
                                    model._keypoints[person_id] = kp
                        
                        # Draw keypoints on image
                        vis_img = model.draw(show_yolo=True, confidence_threshold=args.conf_threshold)
                        
                        # Save visualization
                        vis_path = os.path.join(vis_dir, f"{npy_path.stem}.png")
                        Image.fromarray(vis_img).save(vis_path)
                        
                        # Restore original keypoints if we modified them
                        if args.use_smoothing and 'original_keypoints' in locals():
                            model._keypoints = original_keypoints
                    except Exception as e:
                        print(f"Warning: Could not generate visualization for {npy_path.stem}: {e}")
                        # 如果可视化失败，尝试保存原始图像
                        try:
                            vis_path = os.path.join(vis_dir, f"{npy_path.stem}.png")
                            Image.fromarray(img).save(vis_path)
                        except:
                            pass
            except Exception as e:
                print(f"Error processing file {npy_path}: {e}")
    
    print(f">>> Skeleton data saved to {output_csv}")
    if args.visualize:
        print(f">>> Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch process multiple subjects for skeleton extraction')
    parser.add_argument('--root-dir', type=str, default='/Volumes/Disk/mmwave_cam2.11',
                        help='Root directory containing subject folders')
    parser.add_argument('--model', type=str, default='vitpose-s-coco.pth',
                        help='Checkpoint path of the model')
    parser.add_argument('--yolo', type=str, default='yolo11n.pt',
                        help='Checkpoint path of the yolo model')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Name of the dataset. If None it\'s extracted from the file name. '
                             '["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
    parser.add_argument('--det-class', type=str, default=None,
                        help='["human", "cat", "dog", "horse", "sheep", '
                             '"cow", "elephant", "bear", "zebra", "giraffe", "animals"]')
    parser.add_argument('--model-name', type=str, default='s', choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, default=320,
                        help='YOLOv8 image size during inference')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Minimum confidence for keypoints to be considered valid. [0, 1] range')
    parser.add_argument('--single-pose', action='store_true',
                        help='Only extract the most confident pose in each image')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of detected skeletons')
    parser.add_argument('--use-smoothing', action='store_true',
                        help='Apply OneEuroFilter to smooth keypoints across frames')
    parser.add_argument('--min-cutoff', type=float, default=0.8,
                        help='OneEuroFilter min_cutoff parameter (lower = more smoothing, default: 0.8)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='OneEuroFilter beta parameter (higher = less lag, default: 0.1)')
    
    args = parser.parse_args()
    
    # Check if root directory exists
    root_dir = Path(args.root_dir)
    assert root_dir.exists() and root_dir.is_dir(), f"Root directory {root_dir} does not exist"
    
    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        use_cuda = torch.cuda.is_available()
        try:
            import onnxruntime
            has_onnx = True
        except ModuleNotFoundError:
            has_onnx = False
        yolo = 'easy_ViTPose/' + ('yolov8s' + ('.onnx' if has_onnx and not (use_mps or use_cuda) else '.pt'))
    
    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=False,
                         single_pose=args.single_pose)
    print(f">>> Model loaded: {args.model}")
    
    # Initialize smoothing filters for each person if smoothing is enabled
    if args.use_smoothing:
        print(f">>> Using OneEuroFilter for smoothing (min_cutoff={args.min_cutoff}, beta={args.beta})")
    
    # Find all timestamp folders
    subject_folders = []
    for item in root_dir.iterdir():
        if item.is_dir() and is_timestamp_folder(item.name):
            subject_folders.append(item)
    
    if not subject_folders:
        print(f"No timestamp folders found in {root_dir}")
        return
    
    print(f"Found {len(subject_folders)} subject folders")
    
    # Process each subject folder
    for subject_folder in subject_folders:
        process_subject_folder(subject_folder, model, args)
    
    print(">>> Batch processing completed")


if __name__ == "__main__":
    main()
