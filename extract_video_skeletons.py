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
import cv2

from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.post_processing.one_euro_filter import OneEuroFilter


def main(args):
    # Check if input video exists
    input_video = Path(args.input_video)
    assert input_video.exists(), f"Input video {input_video} does not exist"
    
    # Create output directory if needed
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = args.vis_dir if args.vis_dir else str(output_csv.parent / 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_video}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {total_frames} frames at {fps} FPS")
    
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
                         args.yolo_size, is_video=True,
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
    header = ['frame_number', 'person_id']
    for kp_name in keypoint_names:
        header.extend([f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_score'])
    
    # Process video frames and write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for frame_idx in tqdm.tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            keypoints_dict = model.inference(frame_rgb)
            
            # If no keypoints detected or empty
            if not keypoints_dict or len(keypoints_dict) == 0:
                # Write row with frame number, default person ID and NaN values for all keypoints
                row = [frame_idx, 'none'] + ['NaN'] * (num_keypoints * 3)
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
                    print(f"Warning: Error processing keypoints for frame {frame_idx}: {e}")
                    if len(keypoints_dict) > 0:
                        first_id = list(keypoints_dict.keys())[0]
                        keypoints_dict = {first_id: keypoints_dict[first_id]}
            
            # For each detected person, write a row to CSV
            for person_id, kp in keypoints_dict.items():
                try:
                    # Check if kp is a valid array
                    if not isinstance(kp, np.ndarray) or kp.ndim < 2:
                        print(f"Warning: Invalid keypoint format for frame {frame_idx} person {person_id}")
                        row = [frame_idx, f"person_{person_id}"] + ['NaN'] * (num_keypoints * 3)
                        writer.writerow(row)
                        continue
                    
                    # Apply smoothing filter if enabled
                    if args.use_smoothing:
                        person_key = f"person_{person_id}"
                        if person_key not in smoothing_filters:
                            smoothing_filters[person_key] = OneEuroFilter(
                                x0=kp[:, :2],
                                min_cutoff=args.min_cutoff,
                                beta=args.beta,
                                d_cutoff=fps
                            )
                        
                        smoothed_xy = smoothing_filters[person_key](kp[:, :2])
                        smoothed_kp = kp.copy()
                        smoothed_kp[:, :2] = smoothed_xy
                        kp = smoothed_kp
                    
                    # Flatten keypoints array
                    flat_kp = kp.reshape(-1).tolist()
                    
                    # Create row with frame number, person ID and keypoints
                    row = [frame_idx, f"person_{person_id}"]
                    row.extend(flat_kp)
                    
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error processing keypoint for frame {frame_idx} person {person_id}: {e}")
                    row = [frame_idx, f"person_{person_id}"] + ['NaN'] * (num_keypoints * 3)
                    writer.writerow(row)
            
            # Visualize if requested
            if args.visualize:
                try:
                    if args.use_smoothing:
                        original_keypoints = model._keypoints.copy()
                        for person_id, kp in keypoints_dict.items():
                            if isinstance(kp, np.ndarray) and kp.ndim >= 2:
                                model._keypoints[person_id] = kp
                    
                    vis_img = model.draw(show_yolo=True, confidence_threshold=args.conf_threshold)
                    
                    vis_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.png")
                    Image.fromarray(vis_img).save(vis_path)
                    
                    if args.use_smoothing and 'original_keypoints' in locals():
                        model._keypoints = original_keypoints
                except Exception as e:
                    print(f"Warning: Could not generate visualization for frame {frame_idx}: {e}")
                    try:
                        vis_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.png")
                        Image.fromarray(frame_rgb).save(vis_path)
                    except:
                        pass
    
    # Release video capture
    cap.release()
    
    print(f">>> Skeleton data saved to {output_csv}")
    if args.visualize:
        print(f">>> Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract skeletons from video file and save to CSV')
    parser.add_argument('--input-video', type=str, default="excise_one.mov",
                        help='Path to input video file')
    parser.add_argument('--output-csv', type=str, default="excise_one_reference_skeletons.csv",
                        help='Path to output CSV file')
    parser.add_argument('--model', type=str, default='vitpose-s-coco.pth',
                        help='Checkpoint path of the model')
    parser.add_argument('--yolo', type=str, default='yolo11n.pt',
                        help='Checkpoint path of the yolo model')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Name of the dataset')
    parser.add_argument('--det-class', type=str, default=None,
                        help='Detection class')
    parser.add_argument('--model-name', type=str, default='s',
                        help='Model size')
    parser.add_argument('--yolo-size', type=int, default=320,
                        help='YOLO image size')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--single-pose', action='store_true', default=True,
                        help='Only detect single pose')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations')
    parser.add_argument('--vis-dir', type=str, default="reference_skeletons_vis",
                        help='Visualization directory')
    parser.add_argument('--use-smoothing', action='store_true', default=False,
                        help='Use smoothing filter')
    parser.add_argument('--min-cutoff', type=float, default=0.8,
                        help='Smoothing min cutoff')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Smoothing beta')
    
    args = parser.parse_args()
    main(args) 