import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
import numpy as np
import math
import pygame
import pygame.sndarray
from ultralytics import YOLO
import time
import copy
import os
from concurrent.futures import ThreadPoolExecutor

class MediaPipeProcessor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize YOLO model for person detection
        self.yolo_model = YOLO('yolo11n.pt', verbose=False)
        
        # 预分配缓冲区用于图像处理
        self.process_frame_buffer = np.empty((360, 640, 3), dtype=np.uint8)  # PROCESS_HEIGHT, PROCESS_WIDTH
        self.process_frame_rgb_buffer = np.empty((360, 640, 3), dtype=np.uint8)
        
        # 添加帧计数器和缓存
        self.frame_count = 0
        self.detection_interval = 2  # 每隔2帧进行一次检测
        self.last_boxes = []  # 缓存上一次的检测结果
        
        # 添加线程池
        self.max_workers = min(4, (os.cpu_count() or 1))  # 限制最大线程数
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.processing_lock = threading.Lock()  # 添加线程锁用于同步
        
        # Initialize list to store multiple holistic processors
        self.holistic_processors = []
        self.processor_dimensions = []  # Store bounding boxes for tracking
        
        # Initialize first processor
        self.add_new_processor()
        
        self.wave_phase = 0  # 用于动画效果
        
        # Initialize pygame mixer for sound
        pygame.mixer.init(44100, -16, 1, 1024)
        self.sample_rate = 44100
        self.current_sound = None
        self.is_playing = False
        
        # 噪声相关设置
        self.noise_enabled = False
        self.noise_type = 'constant'  # 'gaussian', 'uniform', 'constant'
        self.noise_params = {
            'gaussian': {'mean': 0, 'std': 0.01},
            'uniform': {'low': -0.01, 'high': 0.01},
            'constant': {'value': 0.01}
        }
        
        # 显示RGB图像的开关
        self.show_rgb = True
        
        # 定义关键点组
        self.landmark_groups = {
            'face': list(range(0, 11)),
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'torso': [23, 24],
            'left_leg': [25, 27, 29, 31],
            'right_leg': [26, 28, 30, 32]
        }
        
        # 设置需要添加噪声的关键点组
        self.noise_landmark_groups = []  # 默认不选择任何关键点组
        
        # 检测参数
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.6
        
        # 处理分辨率设置
        self.PROCESS_WIDTH = 640
        self.PROCESS_HEIGHT = 360
        self.DISPLAY_WIDTH = 1280
        self.DISPLAY_HEIGHT = 720
        
        # 人物追踪相关属性
        self.tracked_people = {}
        self.next_person_id = 0
        self.max_tracking_age = 30
        self.max_id_distance = 0.3
        
        # 颜色设置
        self.id_colors = {
            0: {'pose': (0, 255, 0),    'left_hand': (255, 0, 0),   'right_hand': (0, 0, 255),   'wave': (0, 0, 255)},
            1: {'pose': (0, 255, 255),  'left_hand': (255, 0, 255), 'right_hand': (255, 0, 255), 'wave': (255, 0, 255)},
            2: {'pose': (255, 255, 0),  'left_hand': (255, 128, 0), 'right_hand': (255, 128, 0), 'wave': (255, 128, 0)},
            3: {'pose': (255, 0, 255),  'left_hand': (128, 0, 255), 'right_hand': (128, 0, 255), 'wave': (128, 0, 255)},
        }
        
        # 添加关键点平滑
        self.previous_landmarks = {}  # 存储上一帧的关键点
        self.smoothing_factor = 0.3   # 平滑因子，越大越平滑
        
        # 性能监控
        self.processing_times = []
        self.max_processing_times = 30
        self.last_fps_update = time.time()
        self.fps = 0
        self.fps_update_interval = 0.5

    def add_new_processor(self):
        processor = self.mp_holistic.Holistic(
            static_image_mode=False,  # 改为False以启用跟踪模式
            model_complexity=1,       # 使用更复杂的模型以提高准确性
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,  # 降低检测置信度阈值以提高检测率
            min_tracking_confidence=0.3    # 降低跟踪置信度阈值以维持跟踪
        )
        self.holistic_processors.append(processor)
        self.processor_dimensions.append(None)
        return len(self.holistic_processors) - 1
    
    def detect_people(self, frame):
        frame_height, frame_width = frame.shape[:2]
        results = self.yolo_model(frame, classes=[0], conf=0.2)
        current_boxes = []
        
        # 降低检测置信度阈值
        DETECTION_CONFIDENCE = 0.15
        
        # 降低边缘区域阈值
        EDGE_THRESHOLD = 0.01
        edge_pixels_x = int(frame_width * EDGE_THRESHOLD)
        edge_pixels_y = int(frame_height * EDGE_THRESHOLD)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf > DETECTION_CONFIDENCE:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 计算边界框的原始尺寸
                    original_width = x2 - x1
                    original_height = y2 - y1
                    
                    # 进一步放宽人体高宽比的限制
                    aspect_ratio = original_height / original_width
                    if not (0.3 < aspect_ratio < 5.0):
                        continue
                    
                    # 降低人体大小的要求
                    relative_height = original_height / frame_height
                    if relative_height < 0.05:
                        continue
                    
                    # 减小边界框扩展范围
                    box_width = x2 - x1
                    box_height = y2 - y1
                    padding_x = box_width * 0.05
                    padding_y = box_height * 0.05
                    
                    # 确保扩展后的框不会超出图像边界
                    x1_padded = max(0, x1 - padding_x)
                    y1_padded = max(0, y1 - padding_y)
                    x2_padded = min(frame_width, x2 + padding_x)
                    y2_padded = min(frame_height, y2 + padding_y)
                    
                    # 检查扩展后的框是否仍然合理
                    padded_width = x2_padded - x1_padded
                    padded_height = y2_padded - y1_padded
                    if padded_width < box_width * 0.8 or padded_height < box_height * 0.8:
                        continue
                    
                    area = padded_width * padded_height
                    center = ((x1_padded + x2_padded) / 2, (y1_padded + y2_padded) / 2)
                    
                    # 创建新的检测框
                    new_box = {
                        'box': [int(x1_padded), int(y1_padded), int(x2_padded), int(y2_padded)],
                        'area': area,
                        'confidence': float(box.conf),
                        'center': center,
                        'width': padded_width,
                        'height': padded_height,
                        'original_width': original_width,
                        'original_height': original_height
                    }
                    
                    # 检查与现有框的重叠
                    should_add = True
                    overlapped_boxes = []
                    for existing_box in current_boxes[:]:
                        overlap = self.calculate_overlap(new_box['box'], existing_box['box'])
                        
                        if overlap > 0.2:
                            overlapped_boxes.append((existing_box, overlap))
                    
                    if overlapped_boxes:
                        for existing_box, overlap in overlapped_boxes:
                            vertical_overlap = self.calculate_vertical_overlap(new_box['box'], existing_box['box'])
                            
                            if vertical_overlap > 0.7:
                                if new_box['confidence'] > existing_box['confidence']:
                                    current_boxes.remove(existing_box)
                                else:
                                    should_add = False
                                break
                            elif vertical_overlap > 0.3:
                                center_distance = abs(new_box['center'][0] - existing_box['center'][0])
                                if center_distance < min(new_box['width'], existing_box['width']) * 0.5:
                                    if new_box['confidence'] > existing_box['confidence']:
                                        current_boxes.remove(existing_box)
                                    else:
                                        should_add = False
                        break
            
                    if should_add:
                        current_boxes.append(new_box)
        
        # 按置信度排序并保留最高置信度的框
        current_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        current_boxes = current_boxes[:4]
        
        return [box['box'] for box in current_boxes]
    
    def calculate_overlap(self, box1, box2):
        # Calculate intersection over union (IoU) of two boxes
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def calculate_vertical_overlap(self, box1, box2):
        """Calculate vertical overlap ratio between two boxes."""
        _, y1_1, _, y2_1 = box1
        _, y1_2, _, y2_2 = box2
        
        # Calculate vertical intersection
        y1_i = max(y1_1, y1_2)
        y2_i = min(y2_1, y2_2)
        
        if y2_i <= y1_i:
            return 0.0
        
        intersection = y2_i - y1_i
        
        # Calculate vertical spans
        height1 = y2_1 - y1_1
        height2 = y2_2 - y1_2
        
        # Use minimum height for ratio
        min_height = min(height1, height2)
        return intersection / min_height if min_height > 0 else 0.0
    
    def crop_person(self, frame, box, padding=10):
        x1, y1, x2, y2 = box
        
        # Calculate center of the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Add extra padding for partially visible people
        if x1 < padding or y1 < padding or x2 > frame.shape[1] - padding or y2 > frame.shape[0] - padding:
            padding = int(padding * 1.5)  # Increase padding for edge cases
        
        # Add padding and ensure within frame bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        # If person is cut off at edges, try to maintain aspect ratio
        if x1 == 0 or y1 == 0 or x2 == frame.shape[1] or y2 == frame.shape[0]:
            aspect_ratio = box_width / box_height
            
            # If cut off at sides, adjust height
            if x1 == 0 or x2 == frame.shape[1]:
                desired_height = int(box_width / aspect_ratio)
                y1 = max(0, center_y - desired_height // 2)
                y2 = min(frame.shape[0], center_y + desired_height // 2)
            
            # If cut off at top/bottom, adjust width
            if y1 == 0 or y2 == frame.shape[0]:
                desired_width = int(box_height * aspect_ratio)
                x1 = max(0, center_x - desired_width // 2)
                x2 = min(frame.shape[1], center_x + desired_width // 2)
        
        # Use np.ascontiguousarray to ensure the array is contiguous
        cropped = frame[y1:y2, x1:x2].copy()
        return np.ascontiguousarray(cropped), (x1, y1)
    
    def add_noise_to_landmark(self, landmark, noise_type='gaussian', noise_params=None):
        """
        为关键点添加噪声，并确保坐标保持在有效范围内
        
        参数:
        - landmark: mediapipe关键点
        - noise_type: 噪声类型 ('gaussian', 'uniform', 'constant')
        - noise_params: 噪声参数
            - gaussian: {'mean': 0, 'std': 0.01}  # 降低标准差
            - uniform: {'low': -0.01, 'high': 0.01}  # 降低范围
            - constant: {'value': 0.01}  # 降低固定值
        """
        if noise_params is None:
            noise_params = {}
        
        # 保存原始坐标
        orig_x, orig_y = landmark.x, landmark.y
        
        # 将噪声值归一化到0-1范围
        if noise_type == 'gaussian':
            mean = noise_params.get('mean', 0)
            std = noise_params.get('std', 0.01)  # 降低默认标准差
            noise_x = np.random.normal(mean, std)
            noise_y = np.random.normal(mean, std)
        elif noise_type == 'uniform':
            low = noise_params.get('low', -0.01)  # 降低默认范围
            high = noise_params.get('high', 0.01)
            noise_x = np.random.uniform(low, high)
            noise_y = np.random.uniform(low, high)
        elif noise_type == 'constant':
            value = noise_params.get('value', 0.01)  # 降低默认固定值
            noise_x = value
            noise_y = 0  # 对于constant noise，只在x方向添加偏移
        else:
            return landmark
        
        # 添加噪声并确保坐标在0-1范围内
        landmark.x = np.clip(orig_x + noise_x, 0.0, 1.0)
        landmark.y = np.clip(orig_y + noise_y, 0.0, 1.0)
        
        return landmark

    def set_noise_landmarks(self, groups):
        """
        设置需要添加噪声的关键点组
        
        参数:
        - groups: 字符串列表，可以包含 'face', 'left_arm', 'right_arm', 'torso', 'left_leg', 'right_leg', 'all'
        """
        if not isinstance(groups, list):
            groups = [groups]
        
        valid_groups = set(self.landmark_groups.keys())
        for group in groups:
            if group not in valid_groups:
                raise ValueError(f"Invalid landmark group: {group}. Valid groups are: {valid_groups}")
        
        self.noise_landmark_groups = groups

    def should_add_noise_to_landmark(self, landmark_idx):
        """
        判断是否应该对指定索引的关键点添加噪声
        """
        if 'all' in self.noise_landmark_groups:
            return True
            
        for group in self.noise_landmark_groups:
            if landmark_idx in self.landmark_groups[group]:
                return True
        return False

    def process_landmarks_with_noise(self, landmarks, noise_condition, noise_group=None):
        """处理关键点并添加噪声"""
        if not landmarks:
            return None
            
        if self.noise_enabled and noise_condition:
            landmarks = copy.deepcopy(landmarks)
            for landmark in landmarks.landmark:
                if not noise_group or noise_group in self.noise_landmark_groups:
                    landmark = self.add_noise_to_landmark(
                        landmark,
                        self.noise_type,
                        self.noise_params[self.noise_type]
                    )
        return landmarks

    def draw_landmarks_with_color(self, frame, landmarks, connections, colors, landmark_type):
        """使用指定颜色绘制关键点"""
        if not landmarks:
            return
            
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            connections,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=colors[landmark_type],
                thickness=2,
                circle_radius=2
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=colors[landmark_type],
                thickness=2
            )
        )

    def process_single_person(self, frame, processor_idx, results, box):
        """在裁剪的子帧上处理单个人的骨架"""
        x1, y1, x2, y2 = box
        
        # 裁剪人物区域
        person_frame = frame[y1:y2, x1:x2].copy()
        if person_frame.size == 0:
            return frame
        
        # 获取颜色设置
        colors = self.get_person_colors(processor_idx)
        
        # 存储手腕位置
        left_wrist = None
        right_wrist = None
        
        # 处理姿势关键点
        pose_landmarks = self.process_landmarks_with_noise(
            results.pose_landmarks,
            any(self.should_add_noise_to_landmark(i) for i in range(33))
        )
        self.draw_landmarks_with_color(
            person_frame,
            pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            colors,
            'pose'
        )
        
        # 处理左手关键点
        left_hand_landmarks = self.process_landmarks_with_noise(
            results.left_hand_landmarks,
            True,
            'left_arm'
        )
        self.draw_landmarks_with_color(
            person_frame,
            left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            colors,
            'left_hand'
        )
        if left_hand_landmarks:
            left_wrist = left_hand_landmarks.landmark[0]
            left_wrist = self.convert_landmark_to_pixel(left_wrist, frame.shape)
        
        # 处理右手关键点
        right_hand_landmarks = self.process_landmarks_with_noise(
            results.right_hand_landmarks,
            True,
            'right_arm'
        )
        self.draw_landmarks_with_color(
            person_frame,
            right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            colors,
            'right_hand'
        )
        if right_hand_landmarks:
            right_wrist = right_hand_landmarks.landmark[0]
            right_wrist = self.convert_landmark_to_pixel(right_wrist, frame.shape)
        
        # 将处理后的人物帧放回原始位置
        frame[y1:y2, x1:x2] = person_frame
        
        # 如果检测到两只手，绘制sin wave
        if left_wrist and right_wrist:
            self.draw_sin_wave(frame, left_wrist, right_wrist, colors['wave'])
        
        return frame
    
    def validate_pose_landmarks(self, pose_landmarks, is_partially_visible):
        """Validate pose landmarks based on visibility and position."""
        if not pose_landmarks:
            return False
            
        # Key points we want to check (using MediaPipe's pose landmark indices)
        key_points = {
            'shoulders': [11, 12],  # Left and right shoulders
            'hips': [23, 24],      # Left and right hips
            'knees': [25, 26]      # Left and right knees
        }
        
        # Minimum confidence threshold
        min_confidence = 0.5
        if is_partially_visible:
            min_confidence = 0.3  # Lower threshold for partially visible people
        
        # Check visibility and confidence of key points
        visible_parts = {part: 0 for part in key_points}
        
        for part, indices in key_points.items():
            for idx in indices:
                landmark = pose_landmarks.landmark[idx]
                # Check if landmark is within reasonable bounds and has good visibility
                if (0 <= landmark.x <= 1 and 
                    0 <= landmark.y <= 1 and 
                    landmark.visibility > min_confidence):
                    visible_parts[part] += 1
        
        # For partially visible people, require fewer visible parts
        if is_partially_visible:
            # Require at least one shoulder and one hip to be visible
            return (visible_parts['shoulders'] >= 1 and 
                   visible_parts['hips'] >= 1)
        else:
            # For fully visible people, require most key points to be visible
            return (visible_parts['shoulders'] >= 2 and 
                   visible_parts['hips'] >= 2 and 
                   visible_parts['knees'] >= 1)  # At least one knee
    
    def convert_to_global_coordinates(self, landmark, box, frame_shape):
        """将局部坐标转换为全局坐标"""
        x1, y1, x2, y2 = box
        landmark.x = (landmark.x * (x2 - x1) + x1) / frame_shape[1]
        landmark.y = (landmark.y * (y2 - y1) + y1) / frame_shape[0]
        return landmark

    def convert_landmark_to_pixel(self, landmark, frame_shape):
        """将归一化坐标转换为像素坐标"""
        return (
            int(landmark.x * frame_shape[1]),
            int(landmark.y * frame_shape[0])
        )

    def process_landmarks(self, landmarks, box, frame_shape, should_add_noise=False, person_id=None, landmark_type='pose'):
        """处理一组关键点，包括坐标转换、噪声添加和平滑处理"""
        if not landmarks:
            return None
        
        # 只在需要修改时进行一次深拷贝
        if should_add_noise or person_id is not None:
            landmarks = copy.deepcopy(landmarks)
            
        # 一次循环处理所有操作
        for i, landmark in enumerate(landmarks.landmark):
            # 1. 坐标转换
            self.convert_to_global_coordinates(landmark, box, frame_shape)
            
            # 2. 添加噪声（如果需要）
            if should_add_noise:
                landmark = self.add_noise_to_landmark(
                    landmark,
                    self.noise_type,
                    self.noise_params[self.noise_type]
                )
            
            # 3. 确保坐标在有效范围内
            landmark.x = max(0.0, min(1.0, landmark.x))
            landmark.y = max(0.0, min(1.0, landmark.y))
            if hasattr(landmark, 'z'):
                landmark.z = max(-1.0, min(1.0, landmark.z))
            if hasattr(landmark, 'visibility'):
                landmark.visibility = max(0.0, min(1.0, landmark.visibility))
        
        # 4. 时间平滑处理
        if person_id is not None:
            key = f"{person_id}_{landmark_type}"
            if key not in self.previous_landmarks:
                self.previous_landmarks[key] = copy.deepcopy(landmarks)
            else:
                previous = self.previous_landmarks[key]
                for i, landmark in enumerate(landmarks.landmark):
                    # 使用更保守的平滑因子
                    smooth_factor = 0.5 if landmark_type == 'pose' else 0.7
                    landmark.x = smooth_factor * previous.landmark[i].x + (1 - smooth_factor) * landmark.x
                    landmark.y = smooth_factor * previous.landmark[i].y + (1 - smooth_factor) * landmark.y
                    if hasattr(landmark, 'z'):
                        landmark.z = smooth_factor * previous.landmark[i].z + (1 - smooth_factor) * landmark.z
                    if hasattr(landmark, 'visibility'):
                        landmark.visibility = smooth_factor * previous.landmark[i].visibility + (1 - smooth_factor) * landmark.visibility
            self.previous_landmarks[key] = copy.deepcopy(landmarks)
                
        return landmarks

    def process_person_parallel(self, args):
        """并行处理单个人的骨架检测"""
        try:
            idx, box, display_frame = args
            x1, y1, x2, y2 = box
            
            # 裁剪并处理人物区域，增加padding以包含更多手部区域
            padding_x = int((x2 - x1) * 0.2)  # 增加水平padding
            padding_y = int((y2 - y1) * 0.2)  # 增加垂直padding
            
            # 确保padding后的坐标不超出图像边界
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(display_frame.shape[1], x2 + padding_x)
            y2_pad = min(display_frame.shape[0], y2 + padding_y)
            
            person_frame = display_frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if person_frame.size == 0:
                return None
            
            person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            person_frame_rgb.flags.writeable = False
            
            # MediaPipe处理
            with self.processing_lock:  # 使用锁保护MediaPipe处理
                results = self.holistic_processors[idx].process(person_frame_rgb)
            
            if not results.pose_landmarks:
                return None
            
            # 处理所有关键点
            should_add_noise = self.noise_enabled
            pose_landmarks = self.process_landmarks(
                results.pose_landmarks, 
                [x1_pad, y1_pad, x2_pad, y2_pad],  # 使用padding后的坐标
                display_frame.shape, 
                should_add_noise and any(self.should_add_noise_to_landmark(i) for i in range(33)),
                person_id=idx,
                landmark_type='pose'
            )
            
            # 处理手部关键点
            left_hand_landmarks = None
            if results.left_hand_landmarks:
                left_hand_landmarks = copy.deepcopy(results.left_hand_landmarks)
                for landmark in left_hand_landmarks.landmark:
                    # 转换到全局坐标，考虑padding
                    landmark.x = (landmark.x * (x2_pad - x1_pad) + x1_pad) / display_frame.shape[1]
                    landmark.y = (landmark.y * (y2_pad - y1_pad) + y1_pad) / display_frame.shape[0]
            
            right_hand_landmarks = None
            if results.right_hand_landmarks:
                right_hand_landmarks = copy.deepcopy(results.right_hand_landmarks)
                for landmark in right_hand_landmarks.landmark:
                    # 转换到全局坐标，考虑padding
                    landmark.x = (landmark.x * (x2_pad - x1_pad) + x1_pad) / display_frame.shape[1]
                    landmark.y = (landmark.y * (y2_pad - y1_pad) + y1_pad) / display_frame.shape[0]
            
            return {
                'person_idx': idx,
                'box': box,
                'pose_landmarks': pose_landmarks,
                'left_hand_landmarks': left_hand_landmarks,
                'right_hand_landmarks': right_hand_landmarks,
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def scale_box_coordinates(self, box, original_size, target_size):
        """将边界框坐标从一个尺寸缩放到另一个尺寸"""
        x1, y1, x2, y2 = box
        scale_x = target_size[1] / original_size[1]
        scale_y = target_size[0] / original_size[0]
        return [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]

    def process(self, frame):
        if frame is None:
            return None
        
        process_start = time.time()
        
        # 根据 show_rgb 属性决定是使用原始帧还是黑色背景
        if self.show_rgb:
            display_frame = frame.copy()  # 使用原始帧的副本
        else:
            display_frame = np.zeros(frame.shape, dtype=np.uint8)  # 创建黑色背景
        
        # 使用预分配的缓冲区进行缩放
        cv2.resize(frame, (self.PROCESS_WIDTH, self.PROCESS_HEIGHT), dst=self.process_frame_buffer)
        
        # 确保缓冲区可写
        self.process_frame_rgb_buffer.flags.writeable = True
        cv2.cvtColor(self.process_frame_buffer, cv2.COLOR_BGR2RGB, dst=self.process_frame_rgb_buffer)
        self.process_frame_rgb_buffer.flags.writeable = False
        
        # 隔帧检测人物
        if self.frame_count % self.detection_interval == 0:
            self.last_boxes = self.detect_people(self.process_frame_rgb_buffer)
        self.frame_count += 1
        
        # 使用缓存的检测结果
        people_boxes = self.last_boxes
        
        # 将检测到的框转换回原始尺寸
        process_size = (self.PROCESS_HEIGHT, self.PROCESS_WIDTH)
        display_size = display_frame.shape[:2]
        scaled_boxes = [
            self.scale_box_coordinates(box, process_size, display_size)
            for box in people_boxes
        ]
        
        # 确保有足够的处理器
        while len(self.holistic_processors) < len(scaled_boxes):
            self.add_new_processor()
        
        # 准备并行处理参数
        process_args = [(idx, box, frame) for idx, box in enumerate(scaled_boxes)]  # 使用原始帧进行处理
        
        # 并行处理所有检测到的人
        all_skeletons = []
        futures = []
        
        # 提交任务到线程池
        for args in process_args:
            future = self.thread_pool.submit(self.process_person_parallel, args)
            futures.append(future)
        
        # 收集处理结果
        for future in futures:
            result = future.result()
            if result is not None:
                all_skeletons.append(result)
        
        # 更新人物追踪
        tracked_people = self.update_person_tracking(all_skeletons, frame.shape)
        
        # 为每个检测到的框分配追踪ID和对应的颜色
        for skeleton in all_skeletons:
            person_id = skeleton['person_idx']
            colors = self.get_person_colors(person_id)
            
            # 使用对应的颜色绘制骨架
            if skeleton['pose_landmarks']:
                self.mp_drawing.draw_landmarks(
                    display_frame,  # 在黑色背景上绘制
                    skeleton['pose_landmarks'],
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['pose'],
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['pose'],
                        thickness=2
                    )
                )
            
            # 使用对应的颜色绘制左手
            if skeleton['left_hand_landmarks']:
                self.mp_drawing.draw_landmarks(
                    display_frame,  # 在黑色背景上绘制
                    skeleton['left_hand_landmarks'],
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['left_hand'],
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['left_hand'],
                        thickness=2
                    )
                )
            
            # 使用对应的颜色绘制右手
            if skeleton['right_hand_landmarks']:
                self.mp_drawing.draw_landmarks(
                    display_frame,  # 在黑色背景上绘制
                    skeleton['right_hand_landmarks'],
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['right_hand'],
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=colors['right_hand'],
                        thickness=2
                    )
                )
            
            # 使用对应的颜色绘制sin wave
            if skeleton['left_hand_landmarks'] and skeleton['right_hand_landmarks']:
                left_wrist = skeleton['left_hand_landmarks'].landmark[0]
                right_wrist = skeleton['right_hand_landmarks'].landmark[0]
                left_point = (int(left_wrist.x * display_frame.shape[1]), 
                            int(left_wrist.y * display_frame.shape[0]))
                right_point = (int(right_wrist.x * display_frame.shape[1]), 
                             int(right_wrist.y * display_frame.shape[0]))
                self.draw_sin_wave(display_frame, left_point, right_point, colors['wave'])
            
            # 使用对应的颜色绘制边界框
            x1, y1, x2, y2 = skeleton['box']
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), colors['pose'], 2)
        
        # 更新性能统计
        process_time = time.time() - process_start
        self.update_performance_stats(process_time)
        
        return display_frame

    def __del__(self):
        self.cleanup()
    
    def draw_sin_wave(self, frame, start_point, end_point, color=(0, 0, 255)):
        # 直接使用实际坐标
        x1, y1 = start_point
        x2, y2 = end_point
        
        # 检查坐标是否有效
        if not (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and 
                0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
            return
        
        # 计算两点之间的距离和角度
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 如果距离太小，不绘制正弦波
        if distance < 20:
            return
            
        angle = math.atan2(dy, dx)
        
        # 调整正弦波参数
        base_amplitude = min(20, distance * 0.1)  # 根据距离动态调整振幅
        amplitude = base_amplitude * min(distance / 200, 1.5)  # 限制最大振幅
        
        # 根据距离调整频率和点的数量
        frequency = 1.5  # 降低基础频率
        num_points = max(int(distance), 50)  # 增加点的数量使曲线更平滑
        
        # 生成正弦波点
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # 使用平滑的正弦函数
            x_offset = t * distance
            y_offset = amplitude * math.sin(2 * math.pi * frequency * t + self.wave_phase)
            
            # 使用矩阵变换进行旋转，提高精度
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            x = x1 + (x_offset * cos_angle - y_offset * sin_angle)
            y = y1 + (x_offset * sin_angle + y_offset * cos_angle)
            
            # 确保点在图像范围内并四舍五入
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                points.append((int(round(x)), int(round(y))))
        
        # 绘制正弦波
        if len(points) > 1:
            # 使用抗锯齿线条绘制
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], color, 2, cv2.LINE_AA)
            
            # 在手腕位置绘制端点标记
            cv2.circle(frame, (int(round(x1)), int(round(y1))), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(round(x2)), int(round(y2))), 4, color, -1, cv2.LINE_AA)
        
        # 降低相位变化速度
        self.wave_phase += 0.1  # 降低动画速度

    def generate_sine_tone(self, frequency):
        # Generate a sine wave sound
        duration = 0.1  # 100ms buffer
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Apply fade in/out to avoid clicks
        fade_length = int(0.005 * self.sample_rate)  # 5ms fade
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        tone[:fade_length] *= fade_in
        tone[-fade_length:] *= fade_out
        
        # Convert to 16-bit integer samples
        samples = (tone * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(samples)
    
    def update_sound(self, frequency):
        base_audio_freq = 220  # Base frequency in Hz (A3 note)
        max_audio_freq = 880   # Max frequency in Hz (A5 note)
        
        # Map the visual frequency to an audible frequency range
        audio_freq = base_audio_freq + (frequency * 1000)  # Scale the frequency
        audio_freq = min(max_audio_freq, max(base_audio_freq, audio_freq))
        
        # Generate and play new sound
        if self.current_sound is not None:
            self.current_sound.stop()
        
        self.current_sound = self.generate_sine_tone(audio_freq)
        self.current_sound.play(-1)  # -1 means loop indefinitely
        self.is_playing = True

    def update_person_tracking(self, current_boxes, frame_shape):
        """更新人物追踪状态"""
        height, width = frame_shape[:2]
        current_time = time.time()
        matched_ids = set()
        new_tracked_people = {}
        
        # 计算图像对角线长度用于归一化距离
        diagonal = math.sqrt(width * width + height * height)
        
        # 为每个当前检测到的人物框寻找最佳匹配的追踪ID
        for box_info in current_boxes:
            box = box_info['box']
            center = box_info['center']
            best_match_id = None
            best_match_distance = float('inf')
            
            # 计算与所有已追踪人物的距离
            for person_id, person_info in self.tracked_people.items():
                if person_id in matched_ids:
                    continue
                    
                prev_center = person_info['center']
                # 计算归一化距离
                distance = math.sqrt(
                    ((center[0] - prev_center[0]) / width) ** 2 +
                    ((center[1] - prev_center[1]) / height) ** 2
                )
                
                if distance < self.max_id_distance and distance < best_match_distance:
                    best_match_id = person_id
                    best_match_distance = distance
            
            # 如果找到匹配的ID，更新追踪信息
            if best_match_id is not None:
                matched_ids.add(best_match_id)
                new_tracked_people[best_match_id] = {
                    'box': box,
                    'center': center,
                    'last_seen': current_time
                }
            else:
                # 分配新的ID
                new_id = self.next_person_id
                self.next_person_id = (self.next_person_id + 1) % len(self.id_colors)
                new_tracked_people[new_id] = {
                    'box': box,
                    'center': center,
                    'last_seen': current_time
                }
        
        # 更新追踪状态
        self.tracked_people = new_tracked_people
        return self.tracked_people
    
    def get_person_colors(self, person_id):
        """根据人物ID获取对应的颜色设置"""
        return self.id_colors[person_id % len(self.id_colors)]

    def cleanup(self):
        """统一的资源清理"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            if hasattr(self, 'current_sound') and self.current_sound is not None:
                self.current_sound.stop()
            pygame.mixer.quit()
            if hasattr(self, 'holistic_processors'):
                for processor in self.holistic_processors:
                    processor.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def update_performance_stats(self, process_time):
        """统一的性能统计更新"""
        self.processing_times.append(process_time)
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
        
        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.fps = self.frame_count / self.fps_update_interval
            self.frame_count = 0
            self.last_fps_update = current_time

    def resize_frame(self, frame, target_width, target_height):
        """统一的帧大小调整函数"""
        if target_width > 1 and target_height > 1:
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if target_width / target_height > aspect_ratio:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            else:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return frame

    def process_person(self, args, is_parallel=True):
        """统一的人物处理函数"""
        try:
            idx, box, display_frame = args
            x1, y1, x2, y2 = box
            
            # 裁剪并处理人物区域
            padding_x = int((x2 - x1) * 0.2)
            padding_y = int((y2 - y1) * 0.2)
            
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(display_frame.shape[1], x2 + padding_x)
            y2_pad = min(display_frame.shape[0], y2 + padding_y)
            
            person_frame = display_frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if person_frame.size == 0:
                return None
                
            person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            person_frame_rgb.flags.writeable = False
            
            # MediaPipe处理
            if is_parallel:
                with self.processing_lock:
                    results = self.holistic_processors[idx].process(person_frame_rgb)
            else:
                results = self.holistic_processors[idx].process(person_frame_rgb)
            
            if not results.pose_landmarks:
                return None
                
            # 处理所有关键点
            should_add_noise = self.noise_enabled
            pose_landmarks = self.process_landmarks(
                results.pose_landmarks, 
                [x1_pad, y1_pad, x2_pad, y2_pad],
                display_frame.shape, 
                should_add_noise and any(self.should_add_noise_to_landmark(i) for i in range(33)),
                person_id=idx,
                landmark_type='pose'
            )
            
            # 处理手部关键点
            left_hand_landmarks = None
            if results.left_hand_landmarks:
                left_hand_landmarks = copy.deepcopy(results.left_hand_landmarks)
                for landmark in left_hand_landmarks.landmark:
                    landmark.x = (landmark.x * (x2_pad - x1_pad) + x1_pad) / display_frame.shape[1]
                    landmark.y = (landmark.y * (y2_pad - y1_pad) + y1_pad) / display_frame.shape[0]
            
            right_hand_landmarks = None
            if results.right_hand_landmarks:
                right_hand_landmarks = copy.deepcopy(results.right_hand_landmarks)
                for landmark in right_hand_landmarks.landmark:
                    landmark.x = (landmark.x * (x2_pad - x1_pad) + x1_pad) / display_frame.shape[1]
                    landmark.y = (landmark.y * (y2_pad - y1_pad) + y1_pad) / display_frame.shape[0]
            
            return {
                'person_idx': idx,
                'box': box,
                'pose_landmarks': pose_landmarks,
                'left_hand_landmarks': left_hand_landmarks,
                'right_hand_landmarks': right_hand_landmarks,
                'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

class CameraApp:
    def __init__(self, camera_index=0):
        self.root = tk.Tk()
        self.root.title("Neon Skeleton")
        
        # 设置窗口最小尺寸
        self.root.minsize(800, 600)
        
        # 设置窗口初始大小为屏幕的80%
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # 居中窗口
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 设置窗口背景为黑色
        self.root.configure(bg='black')
        
        # 配置根窗口的网格权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)  # 视频区域可扩展
        
        # 创建左侧控制面板（固定宽度）
        self.control_frame = tk.Frame(self.root, bg='black', width=300)
        self.control_frame.grid(row=0, column=0, sticky='ns', padx=10, pady=10)
        self.control_frame.grid_propagate(False)  # 防止frame被内容压缩
        
        # 创建视频显示区域（可扩展）
        self.video_frame = tk.Frame(self.root, bg='black')
        self.video_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        # 配置视频框架的网格权重
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        # 创建视频标签并使其填充整个视频框架
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky='nsew')
        
        # 添加显示控制
        self.setup_display_controls()
        
        # 添加噪声控制
        self.setup_noise_controls()
        
        # 设置摄像头
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 获取实际分辨率
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        # 初始化处理器和性能监控
        self.frame_queue = queue.Queue(maxsize=3)
        self.last_frame_time = time.time()
        self.frame_interval = 1.0 / 30
        self.frame_count = 0
        
        self.processor = MediaPipeProcessor()
        self.is_running = True
        
        # 启动摄像头捕获线程
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # 开始显示更新
        self.update_display()
        
        # 窗口置前
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()

    def capture_frames(self):
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            
            # Control frame rate
            if elapsed >= self.frame_interval:
                ret, frame = self.cap.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put({
                        'frame': frame,
                        'timestamp': current_time,
                        'frame_number': self.frame_count
                    })
                    
                    self.frame_count += 1
                    self.last_frame_time = current_time
            else:
                # Avoid CPU overuse
                time.sleep(max(0, self.frame_interval - elapsed))

    def update_display(self):
        try:
            frame_data = self.frame_queue.get_nowait()
            if frame_data is not None:
                frame = frame_data['frame']
                current_time = time.time()
                
                # 处理帧
                frame = cv2.flip(frame, 1)
                frame = self.processor.process(frame)
                
                # 调整帧大小
                video_width = self.video_frame.winfo_width()
                video_height = self.video_frame.winfo_height()
                frame = self.processor.resize_frame(frame, video_width, video_height)
                
                # 添加性能信息显示
                avg_process_time = sum(self.processor.processing_times) / len(self.processor.processing_times) if self.processor.processing_times else 0
                cv2.putText(frame, f"FPS: {int(self.processor.fps)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Process Time: {avg_process_time*1000:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 转换并显示图像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.config(image=photo)
                self.video_label.image = photo
                
        except queue.Empty:
            pass
        
        # 安排下一次更新
        if self.is_running:
            self.root.after(max(1, int(self.frame_interval * 1000)), self.update_display)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.processor.cleanup()  # 使用新的清理方法
        self.root.destroy()

    def setup_noise_controls(self):
        """Setup noise control interface"""
        # Create noise control area
        noise_frame = tk.LabelFrame(self.control_frame, text="Noise Control", bg='black', fg='white')
        noise_frame.pack(fill='x', padx=5, pady=5)
        
        # Noise switch
        self.noise_var = tk.BooleanVar(value=False)
        noise_cb = tk.Checkbutton(noise_frame, text="Enable Noise", 
                                 variable=self.noise_var,
                                 bg='black', fg='white',
                                 selectcolor='black',
                                 activebackground='black',
                                 activeforeground='white')
        noise_cb.pack(anchor='w', padx=5, pady=2)
        
        # Noise type selection
        type_frame = tk.Frame(noise_frame, bg='black')
        type_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Label(type_frame, text="Noise Type:", bg='black', fg='white').pack(anchor='w')
        self.noise_type_var = tk.StringVar(value='constant')
        for noise_type in ['constant', 'gaussian', 'uniform']:
            rb = tk.Radiobutton(type_frame, text=noise_type,
                               variable=self.noise_type_var,
                               value=noise_type,
                               command=self.update_noise_params,
                               bg='black', fg='white',
                               selectcolor='black',
                               activebackground='black',
                               activeforeground='white')
            rb.pack(anchor='w')
        
        # Landmark group selection
        points_frame = tk.LabelFrame(noise_frame, text="Landmark Groups", bg='black', fg='white')
        points_frame.pack(fill='x', padx=5, pady=5)
        
        # Define landmark group names mapping
        group_names = {
            'face': 'Face',
            'left_arm': 'Left Arm',
            'right_arm': 'Right Arm',
            'torso': 'Torso',
            'left_leg': 'Left Leg',
            'right_leg': 'Right Leg'
        }
        
        self.landmark_vars = {}
        for group_key, group_name in group_names.items():
            self.landmark_vars[group_key] = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(points_frame, text=group_name,
                               variable=self.landmark_vars[group_key],
                               bg='black', fg='white',
                               selectcolor='black',
                               activebackground='black',
                               activeforeground='white')
            cb.pack(anchor='w')
        
        # Parameter settings area
        self.param_frame = tk.LabelFrame(noise_frame, text="Parameter Settings", bg='black', fg='white')
        self.param_frame.pack(fill='x', padx=5, pady=5)
        
        # Create parameter inputs
        self.setup_param_inputs()
        
        # Button area
        button_frame = tk.Frame(noise_frame, bg='black')
        button_frame.pack(fill='x', padx=5, pady=5)
        
        # Apply button
        apply_button = tk.Button(button_frame, text="Apply Settings",
                               command=self.apply_noise_settings,
                               bg='#2962FF',  # Material Blue
                               fg='white',
                               activebackground='#1565C0',
                               activeforeground='white',
                               relief='flat',
                               padx=10,
                               pady=5)
        apply_button.pack(side='left', padx=2)
        
        # Reset button
        reset_button = tk.Button(button_frame, text="Reset",
                               command=self.reset_noise_settings,
                               bg='#FF1744',  # Material Red
                               fg='white',
                               activebackground='#D50000',
                               activeforeground='white',
                               relief='flat',
                               padx=10,
                               pady=5)
        reset_button.pack(side='left', padx=2)

    def setup_param_inputs(self):
        """Setup parameter input area"""
        # Clear existing parameter inputs
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        noise_type = self.noise_type_var.get()
        
        if noise_type == 'constant':
            tk.Label(self.param_frame, text="Value:", bg='black', fg='white').pack(anchor='w')
            self.value_var = tk.StringVar(value='0.01')
            tk.Entry(self.param_frame, textvariable=self.value_var,
                    bg='gray20', fg='white').pack(fill='x', padx=5)
            
        elif noise_type == 'gaussian':
            tk.Label(self.param_frame, text="Mean:", bg='black', fg='white').pack(anchor='w')
            self.mean_var = tk.StringVar(value='0')
            tk.Entry(self.param_frame, textvariable=self.mean_var,
                    bg='gray20', fg='white').pack(fill='x', padx=5)
            
            tk.Label(self.param_frame, text="Std Dev:", bg='black', fg='white').pack(anchor='w')
            self.std_var = tk.StringVar(value='0.01')
            tk.Entry(self.param_frame, textvariable=self.std_var,
                    bg='gray20', fg='white').pack(fill='x', padx=5)
            
        elif noise_type == 'uniform':
            tk.Label(self.param_frame, text="Min Value:", bg='black', fg='white').pack(anchor='w')
            self.min_var = tk.StringVar(value='-0.01')
            tk.Entry(self.param_frame, textvariable=self.min_var,
                    bg='gray20', fg='white').pack(fill='x', padx=5)
            
            tk.Label(self.param_frame, text="Max Value:", bg='black', fg='white').pack(anchor='w')
            self.max_var = tk.StringVar(value='0.01')
            tk.Entry(self.param_frame, textvariable=self.max_var,
                    bg='gray20', fg='white').pack(fill='x', padx=5)

    def update_noise_params(self):
        """Update noise parameter settings interface"""
        self.setup_param_inputs()

    def apply_noise_settings(self):
        """Apply noise settings"""
        # Update noise switch state
        self.processor.noise_enabled = self.noise_var.get()
        
        # Update noise type
        noise_type = self.noise_type_var.get()
        self.processor.noise_type = noise_type
        
        # Update noise parameters
        if noise_type == 'constant':
            self.processor.noise_params['constant']['value'] = float(self.value_var.get())
        elif noise_type == 'gaussian':
            self.processor.noise_params['gaussian']['mean'] = float(self.mean_var.get())
            self.processor.noise_params['gaussian']['std'] = float(self.std_var.get())
        elif noise_type == 'uniform':
            self.processor.noise_params['uniform']['low'] = float(self.min_var.get())
            self.processor.noise_params['uniform']['high'] = float(self.max_var.get())
        
        # Update landmark groups
        selected_groups = []
        for group, var in self.landmark_vars.items():
            if var.get():
                selected_groups.append(group)
        self.processor.noise_landmark_groups = selected_groups

    def reset_noise_settings(self):
        """Reset all noise settings"""
        # Reset noise switch
        self.noise_var.set(False)
        self.processor.noise_enabled = False
        
        # Reset noise type
        self.noise_type_var.set('constant')
        self.processor.noise_type = 'constant'
        
        # Reset landmark group selection
        for var in self.landmark_vars.values():
            var.set(False)
        self.processor.noise_landmark_groups = []
        
        # Reset parameter values
        self.processor.noise_params = {
            'gaussian': {'mean': 0, 'std': 0.01},
            'uniform': {'low': -0.01, 'high': 0.01},
            'constant': {'value': 0.01}
        }
        
        # Update parameter inputs
        self.update_noise_params()

    def setup_display_controls(self):
        """Setup display control interface"""
        # Create display control area
        display_frame = tk.LabelFrame(self.control_frame, text="Display Control", bg='black', fg='white')
        display_frame.pack(fill='x', padx=5, pady=5)
        
        # RGB display switch
        self.show_rgb_var = tk.BooleanVar(value=True)
        show_rgb_cb = tk.Checkbutton(display_frame, text="Show RGB Image", 
                                   variable=self.show_rgb_var,
                                   bg='black', fg='white',
                                   selectcolor='black',
                                   activebackground='black',
                                   activeforeground='white',
                                   command=self.toggle_rgb_display)
        show_rgb_cb.pack(anchor='w', padx=5, pady=2)
    
    def toggle_rgb_display(self):
        """Toggle RGB display on/off"""
        self.processor.show_rgb = self.show_rgb_var.get()

if __name__ == "__main__":
    # Directly create main application instance, using default camera (usually 0)
    app = CameraApp(camera_index=0)
    
    # Set noise parameter example (optional)
    app.processor.noise_enabled = False  # Default noise off
    app.processor.noise_type = 'gaussian'  # Set noise type to Gaussian noise
    app.processor.noise_params['gaussian']['std'] = 15  # Set Gaussian noise standard deviation
    
    # Set landmark groups to add noise to (optional)
    app.processor.set_noise_landmarks(['left_arm', 'right_arm'])
    
    # Run application
    app.run() 