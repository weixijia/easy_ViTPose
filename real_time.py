import cv2
import time
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from easy_ViTPose import VitInference

# Global variable to track privacy mode status
show_rgb = True  # Default: show RGB background

class ViTPoseApp:
    def __init__(self, camera_index=0):
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("ViTPose Real-time")
        
        # Set window size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Configure the grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create left control panel (fixed width)
        self.control_frame = tk.Frame(self.root, width=250, bg='#2c2c2c')
        self.control_frame.grid(row=0, column=0, sticky='ns')
        self.control_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Create video display area (expandable)
        self.video_frame = tk.Frame(self.root, bg='black')
        self.video_frame.grid(row=0, column=1, sticky='nsew')
        
        # Configure video frame grid
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        # Create video label
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky='nsew')
        
        # Add control elements
        self.setup_control_panel()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        # Initialize model
        self.init_model()
        
        # Initialize variables for FPS calculation
        self.prev_time = 0
        self.curr_time = 0
        self.display_prev_time = 0  # 用于计算显示帧率
        self.fps_values = []
        self.display_fps_values = []  # 用于显示帧率的平滑
        self.skip_frames = 0
        self.process_every_n_frames = 2  # 减少帧跳过，提高流畅度
        self.fps_print_interval = 30
        self.frame_count = 0
        
        # Initialize black background for privacy mode
        self.black_bg = None
        
        # Start the video processing
        self.is_running = True
        self.update()
        
        # Focus the window
        self.root.focus_force()
        
    def setup_control_panel(self):
        # Add title
        title_label = tk.Label(self.control_frame, text="ViTPose Controls", 
                              font=("Arial", 16, "bold"), bg='#2c2c2c', fg='white')
        title_label.pack(pady=20)
        
        # Add privacy mode control
        privacy_frame = tk.LabelFrame(self.control_frame, text="Privacy Settings", 
                                     bg='#2c2c2c', fg='white', font=("Arial", 12))
        privacy_frame.pack(fill='x', padx=10, pady=10)
        
        # Privacy mode toggle
        self.show_rgb_var = tk.BooleanVar(value=True)
        privacy_cb = tk.Checkbutton(privacy_frame, text="Show RGB Background", 
                                   variable=self.show_rgb_var,
                                   command=self.toggle_privacy_mode,
                                   bg='#2c2c2c', fg='white', 
                                   selectcolor='#2c2c2c',
                                   activebackground='#2c2c2c',
                                   activeforeground='white')
        privacy_cb.pack(anchor='w', padx=10, pady=10)
        
        # Add status display
        status_frame = tk.LabelFrame(self.control_frame, text="Status", 
                                    bg='#2c2c2c', fg='white', font=("Arial", 12))
        status_frame.pack(fill='x', padx=10, pady=10)
        
        # FPS display
        fps_label = tk.Label(status_frame, text="FPS:", bg='#2c2c2c', fg='white')
        fps_label.grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.fps_value_label = tk.Label(status_frame, text="0", bg='#2c2c2c', fg='white')
        self.fps_value_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        # Mode display
        mode_label = tk.Label(status_frame, text="Mode:", bg='#2c2c2c', fg='white')
        mode_label.grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.mode_value_label = tk.Label(status_frame, text="Normal Mode", bg='#2c2c2c', fg='white')
        self.mode_value_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Processing time display
        process_label = tk.Label(status_frame, text="Process Time:", bg='#2c2c2c', fg='white')
        process_label.grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.process_value_label = tk.Label(status_frame, text="0 ms", bg='#2c2c2c', fg='white')
        self.process_value_label.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        
        # Add quit button
        quit_button = tk.Button(self.control_frame, text="Quit", 
                               command=self.on_closing,
                               bg='#FF5252', fg='white',
                               activebackground='#FF1744',
                               activeforeground='white',
                               relief='flat',
                               font=("Arial", 12),
                               padx=10, pady=5)
        quit_button.pack(side='bottom', pady=20)
        
    def toggle_privacy_mode(self):
        global show_rgb
        show_rgb = self.show_rgb_var.get()
        mode_text = "Normal Mode" if show_rgb else "Privacy Mode"
        self.mode_value_label.config(text=mode_text)
        
    def init_model(self):
        # Model configuration
        model_path = 'vitpose-s-coco.pth'
        yolo_path = 'yolo11n.pt'
        dataset = 'coco'

        # Create model with specified device and improved tracking parameters
        self.model = VitInference(
            model_path, 
            yolo_path, 
            model_name='s',
            yolo_size=320,  # Keep YOLO size for detection quality
            is_video=True, 
            device=None,  # Let VitInference handle device detection
            dataset=dataset, 
            yolo_step=3,  # 每3帧运行一次 YOLO 检测，平衡性能和稳定性
            tracker_max_age=15,  # 减少最大追踪帧数
            tracker_min_hits=1,  # 降低开始跟踪所需的连续检测次数
            tracker_iou_threshold=0.2  # 保持较低的IOU阈值以维持跟踪稳定性
        )
        self.confidence_threshold = 0.4  # 提高置信度阈值，只显示更可靠的关键点
        
    def resize_image_for_display(self, img):
        """调整图像大小以适应显示区域"""
        video_width = self.video_frame.winfo_width()
        video_height = self.video_frame.winfo_height()
        if video_width > 1 and video_height > 1:  # 确保有效尺寸
            img_width, img_height = img.size
            # 计算保持宽高比的新尺寸
            if video_width / video_height > img_width / img_height:
                new_height = video_height
                new_width = int(img_width * (video_height / img_height))
            else:
                new_width = video_width
                new_height = int(img_height * (video_width / img_width))
            img = img.resize((new_width, new_height), Image.LANCZOS)
        return img
        
    def update(self):
        if not self.is_running:
            return
            
        # Read a frame
        ret, frame = self.cap.read()
        if not ret:
            print("Unable to get video frame")
            return
            
        self.frame_count += 1
        
        # 计算显示帧率（每一帧都计算，无论是否处理）
        curr_display_time = time.time()
        if self.display_prev_time > 0:
            display_fps = 1 / (curr_display_time - self.display_prev_time)
            self.display_fps_values.append(display_fps)
            if len(self.display_fps_values) > 10:
                self.display_fps_values.pop(0)
        self.display_prev_time = curr_display_time
        
        # Skip frame processing for higher FPS
        self.skip_frames += 1
        if self.skip_frames % self.process_every_n_frames != 0:
            # 在跳过的帧中也显示上一帧的结果，避免闪烁
            if hasattr(self, 'last_display_frame'):
                frame_rgb = cv2.cvtColor(self.last_display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = self.resize_image_for_display(img)
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self.photo)
                
                # Update FPS display even in skipped frames
                if len(self.display_fps_values) > 0:
                    avg_display_fps = sum(self.display_fps_values) / len(self.display_fps_values)
                    self.fps_value_label.config(text=f"{avg_display_fps:.1f}")
            
            # Schedule next update
            self.root.after(1, self.update)
            return
            
        # Process the frame
        t_start = time.time()
            
        # Convert BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_convert = time.time()
        
        # Infer keypoints
        with torch.no_grad():
            keypoints = self.model.inference(frame_rgb)
        t_inference = time.time()
        
        # Set face keypoint confidence to 0 (don't show face keypoints)
        if hasattr(self.model, '_keypoints'):
            for idx in self.model._keypoints:
                self.model._keypoints[idx][23:91, 2] = 0
        
        # Check privacy mode status
        privacy_mode = not self.show_rgb_var.get()
        
        try:
            if privacy_mode:
                # PRIVACY MODE: Draw skeleton on black background
                if self.black_bg is None or self.black_bg.shape != frame_rgb.shape:
                    self.black_bg = np.zeros_like(frame_rgb)
                self.model._img = self.black_bg
                normal_result = self.model.draw(confidence_threshold=self.confidence_threshold, show_yolo=True)
            else:
                # Normal mode: Draw skeleton on RGB image
                self.model._img = frame_rgb
                normal_result = self.model.draw(confidence_threshold=self.confidence_threshold, show_yolo=True)
            
            # 转换回BGR并保存当前帧用于跳帧显示
            display_frame = cv2.cvtColor(normal_result, cv2.COLOR_RGB2BGR)
            self.last_display_frame = display_frame  # 无需额外复制
        except Exception as e:
            print(f"Error in drawing: {e}")
            display_frame = frame
            
        t_draw = time.time()
        
        # Calculate processing FPS (only for processed frames)
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        
        # Smooth FPS calculation for processing
        self.fps_values.append(fps)
        if len(self.fps_values) > 10:
            self.fps_values.pop(0)
        avg_fps = sum(self.fps_values) / len(self.fps_values)
        
        # Use display FPS for UI update
        avg_display_fps = sum(self.display_fps_values) / len(self.display_fps_values) if self.display_fps_values else 0
        self.fps_value_label.config(text=f"{avg_display_fps:.1f}")
        process_time = (t_draw - t_start) * 1000
        self.process_value_label.config(text=f"{process_time:.1f} ms")
        
        # Convert to PIL format for tkinter
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = self.resize_image_for_display(img)
        
        # Convert to PhotoImage and update display
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo)
        
        # Schedule next update
        self.root.after(1, self.update)
        
    def run(self):
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Start the main loop
        self.root.mainloop()
        
    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.model.reset()  # Reset tracker
        self.root.destroy()

if __name__ == "__main__":
    app = ViTPoseApp(camera_index=0)
    app.run()