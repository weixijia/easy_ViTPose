#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.post_processing.one_euro_filter import OneEuroFilter


def natural_sort_key(s):
    """
    Sort strings that contain numbers in a natural way.
    For example: ['img1.jpg', 'img10.jpg', 'img2.jpg'] -> ['img1.jpg', 'img2.jpg', 'img10.jpg']
    """
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def extract_frame_number(filename):
    """
    Extract frame number from filename.
    Assumes filenames are numbers like '1.npy', '2.npy', etc.
    """
    # Remove file extension and convert to integer
    return int(Path(filename).stem)


def main():
    parser = argparse.ArgumentParser(description='Extract skeletons from .npy files and save to CSV')
    parser.add_argument('--input-dir', type=str, default='/Volumes/Disk/camera',
                        help='Directory containing input .npy files')
    parser.add_argument('--output-csv', type=str, default='/Volumes/Disk/camera/skeletons.csv',
                        help='Path to output CSV file')
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
    parser.add_argument('--single-pose', default=True, action='store_true',
                        help='Only extract the most confident pose in each image')
    parser.add_argument('--sort-by-timestamp', default=True, action='store_true',
                        help='Sort images by timestamp in filename (if available)')
    parser.add_argument('--visualize', default=True, action='store_true',
                        help='Save visualization of detected skeletons')
    parser.add_argument('--vis-dir', type=str, default='/Volumes/Disk/camera/visualizations',
                        help='Directory to save visualization images (if --visualize is used)')
    parser.add_argument('--use-smoothing', action='store_true',
                        help='Apply OneEuroFilter to smooth keypoints across frames')
    parser.add_argument('--min-cutoff', type=float, default=0.8,
                        help='OneEuroFilter min_cutoff parameter (lower = more smoothing, default: 0.8)')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='OneEuroFilter beta parameter (higher = less lag, default: 0.1)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    input_dir = Path(args.input_dir)
    assert input_dir.exists() and input_dir.is_dir(), f"Input directory {input_dir} does not exist"
    
    # Create output directory if needed
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = args.vis_dir if args.vis_dir else str(output_csv.parent / 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get list of .npy files
    npy_files = list(input_dir.glob('*.npy'))
    
    assert len(npy_files) > 0, f"No .npy files found in {input_dir}"
    print(f"Found {len(npy_files)} .npy files in {input_dir}")
    
    # Sort .npy files by frame number
    npy_files.sort(key=lambda x: extract_frame_number(x.name))
    print(f">>> Files sorted by frame number")
    
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
    smoothing_filters = {}
    if args.use_smoothing:
        print(f">>> Using OneEuroFilter for smoothing (min_cutoff={args.min_cutoff}, beta={args.beta})")
    
    # Get keypoint names from the model
    from easy_ViTPose.vit_utils.visualization import joints_dict
    dataset = model.dataset
    keypoint_names = joints_dict()[dataset]['keypoints']
    num_keypoints = len(keypoint_names)
    
    # Prepare CSV header
    header = ['image_name', 'person_id']
    for kp_name in keypoint_names:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_score'])
    
    # Process .npy files and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for npy_path in tqdm.tqdm(npy_files):
            # Load numpy array (image)
            img = np.load(npy_path)
            
            # Run inference
            # VitInference returns a dictionary where keys are person IDs and values are keypoint arrays
            keypoints_dict = model.inference(img)
            
            # If no keypoints detected or empty
            if not keypoints_dict or len(keypoints_dict) == 0:
                # Write row with frame number, default person ID and NaN values for all keypoints
                row = [npy_path.stem, 'none'] + ['NaN'] * (num_keypoints * 3)
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
                        row = [f"{npy_path.stem}", f"person_{person_id}"] + ['NaN'] * (num_keypoints * 3)
                        writer.writerow(row)
                        continue
                    
                    # Apply smoothing filter if enabled
                    if args.use_smoothing:
                        # Create a filter for this person if it doesn't exist yet
                        person_key = f"person_{person_id}"
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
                    
                    # Create row with frame number, person ID and keypoints
                    row = [npy_path.stem, f"person_{person_id}"]
                    row.extend(flat_kp)
                    
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error processing keypoint for {npy_path.stem} person {person_id}: {e}")
                    # Write row with frame number and NaN values for all keypoints on error
                    row = [npy_path.stem, f"person_{person_id}"] + ['NaN'] * (num_keypoints * 3)
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
    
    print(f">>> Skeleton data saved to {output_csv}")
    if args.visualize:
        print(f">>> Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
