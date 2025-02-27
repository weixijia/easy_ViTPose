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
        self.fps_values = []
        self.skip_frames = 0
        self.process_every_n_frames = 2  # 减少帧跳过，提高流畅度
        self.fps_print_interval = 30
        self.frame_count = 0
        
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
        print(f"Privacy mode toggled: show_rgb = {self.show_rgb_var.get()}")
        
    def init_model(self):
        # Model configuration
        model_path = 'vitpose-s-wholebody.pth'
        yolo_path = 'yolov8n.pt'
        dataset = 'wholebody'

        # Create model with specified device and improved tracking parameters
        self.model = VitInference(
            model_path, 
            yolo_path, 
            model_name='s',
            yolo_size=320,
            is_video=True, 
            device=None,
            dataset=dataset, 
            yolo_step=1,
            tracker_max_age=30,
            tracker_min_hits=2,
            tracker_iou_threshold=0.2
        )
        self.confidence_threshold = 0.4
        
        # 添加关键点平滑
        self.smooth_factor = 0.3  # EMA平滑因子 (0-1), 越大平滑效果越强
        self.prev_keypoints = None  # 存储上一帧的关键点
        
    def smooth_keypoints(self, current_keypoints):
        """使用指数移动平均对关键点进行平滑"""
        if self.prev_keypoints is None:
            self.prev_keypoints = current_keypoints
            return current_keypoints
            
        # 对每个检测到的人进行平滑
        for person_id in current_keypoints:
            if person_id in self.prev_keypoints:
                # 只平滑置信度高的关键点
                mask = current_keypoints[person_id][:, 2] > self.confidence_threshold
                current_keypoints[person_id][mask, :2] = (
                    self.smooth_factor * self.prev_keypoints[person_id][mask, :2] +
                    (1 - self.smooth_factor) * current_keypoints[person_id][mask, :2]
                )
        
        self.prev_keypoints = current_keypoints.copy()
        return current_keypoints
        
    def update(self):
        if not self.is_running:
            return
            
        # Read a frame
        ret, frame = self.cap.read()
        if not ret:
            print("Unable to get video frame")
            return
            
        self.frame_count += 1
        
        # Skip frame processing for higher FPS
        self.skip_frames += 1
        if self.skip_frames % self.process_every_n_frames != 0:
            if hasattr(self, 'last_display_frame'):
                frame_rgb = cv2.cvtColor(self.last_display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize to fit the video frame
                video_width = self.video_frame.winfo_width()
                video_height = self.video_frame.winfo_height()
                if video_width > 1 and video_height > 1:
                    img_width, img_height = img.size
                    if video_width / video_height > img_width / img_height:
                        new_height = video_height
                        new_width = int(img_width * (video_height / img_height))
                    else:
                        new_width = video_width
                        new_height = int(img_height * (video_width / img_width))
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self.photo)
            
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
            
            # 对关键点进行平滑
            if hasattr(self.model, '_keypoints'):
                self.model._keypoints = self.smooth_keypoints(self.model._keypoints)
                
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
                black_bg = np.zeros_like(frame_rgb)
                self.model._img = black_bg
                normal_result = self.model.draw(confidence_threshold=self.confidence_threshold, show_yolo=False)
                display_frame = cv2.cvtColor(normal_result, cv2.COLOR_RGB2BGR)
            else:
                # Normal mode: Draw skeleton on RGB image
                self.model._img = frame_rgb
                normal_result = self.model.draw(confidence_threshold=self.confidence_threshold, show_yolo=False)
                display_frame = cv2.cvtColor(normal_result, cv2.COLOR_RGB2BGR)
            
            # 保存当前帧用于跳帧显示
            self.last_display_frame = display_frame.copy()
        except Exception as e:
            print(f"Error in drawing: {e}")
            display_frame = frame
            
        t_draw = time.time()
        
        # Calculate FPS
        self.curr_time = time.time()
        fps = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        
        # Smooth FPS calculation
        self.fps_values.append(fps)
        if len(self.fps_values) > 10:
            self.fps_values.pop(0)
        avg_fps = sum(self.fps_values) / len(self.fps_values)
        
        # Update status labels
        self.fps_value_label.config(text=f"{avg_fps:.1f}")
        process_time = (t_draw - t_start) * 1000
        self.process_value_label.config(text=f"{process_time:.1f} ms")
        
        # Print performance info every few frames
        if self.frame_count % self.fps_print_interval == 0:
            privacy_status = "Normal Mode" if self.show_rgb_var.get() else "Privacy Mode"
            print(f"FPS: {avg_fps:.1f} | {privacy_status} | Convert: {(t_convert-t_start)*1000:.1f}ms | "
                  f"Inference: {(t_inference-t_convert)*1000:.1f}ms | Draw: {(t_draw-t_inference)*1000:.1f}ms")
        
        # Convert to PIL format for tkinter
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit the video frame
        video_width = self.video_frame.winfo_width()
        video_height = self.video_frame.winfo_height()
        if video_width > 1 and video_height > 1:  # Ensure valid dimensions
            img_width, img_height = img.size
            # Calculate new size while maintaining aspect ratio
            if video_width / video_height > img_width / img_height:
                new_height = video_height
                new_width = int(img_width * (video_height / img_height))
            else:
                new_width = video_width
                new_height = int(img_height * (video_width / img_width))
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
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