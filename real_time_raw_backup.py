import cv2
import time
import torch
import numpy as np
from easy_ViTPose import VitInference

# 初始化网络摄像头
cap = cv2.VideoCapture(0)  # 使用默认摄像头，如果有多个摄像头，可以尝试不同的索引

# 设置摄像头分辨率为较高值以保证识别精度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开网络摄像头")
    exit()

# 自动检测可用的设备
if torch.cuda.is_available():
    device = 'cuda'
    print("使用 CUDA 加速")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print("使用 MPS 加速")
    # 为MPS设备启用更多优化
    if hasattr(torch.backends.mps, 'enable_linalg'):
        torch.backends.mps.enable_linalg = True
    if hasattr(torch.backends.mps, 'enable_math'):
        torch.backends.mps.enable_math = True
else:
    device = 'cpu'
    print("使用 CPU 运行")

# 启用 PyTorch 优化
torch.backends.cudnn.benchmark = True
if device == 'cuda':
    # 对于CUDA设备的额外优化
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition
model_path = 'vitpose-s-wholebody.pth'
yolo_path = 'yolo11n.pt'  # 使用 YOLOv11 模型

use_wholebody = True  

if use_wholebody:
    #model_path = 'vitpose-s-wholebody.pth'  
    model_path = 'vitpose-s-wholebody.pth'  
    dataset = 'wholebody'
else:
    dataset = None  # 自动从模型名称推断

# 优化参数
yolo_size = 320  # 增大 YOLO 输入尺寸以提高检测精度
yolo_step = 5    # 每5帧运行一次 YOLO 检测，其他帧使用跟踪器
confidence_threshold = 0.3  # 保持置信度阈值

# 创建模型时指定设备
model = VitInference(model_path, yolo_path, model_name='s', yolo_size=yolo_size, 
                    is_video=True, device=device, dataset=dataset, yolo_step=yolo_step)

# 用于计算FPS
prev_time = 0
curr_time = 0
fps_values = []  # 存储最近的FPS值
skip_frames = 0  # 跳帧计数器
process_every_n_frames = 2  # 每2帧处理一次，提高处理频率但不影响太多性能
fps_print_interval = 30  # 每30帧打印一次FPS
frame_count = 0

# 预热模型以提高初始性能
print("预热模型中...")
dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
for _ in range(5):
    model.inference(dummy_img)
print("预热完成，开始实时姿态估计")

try:
    print("开始实时姿态估计，按'q'键退出")
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
        
        frame_count += 1
        
        # 跳帧处理，提高FPS
        skip_frames += 1
        if skip_frames % process_every_n_frames != 0:
            # 显示上一帧处理结果
            if 'result_bgr' in locals():
                # 更新FPS计算
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time
                
                # 显示结果（不显示FPS）
                cv2.imshow('ViTPose Real-time', result_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue
            
        # 直接使用原始分辨率进行推理，不降低尺寸
        # 性能监控开始
        t_start = time.time()
            
        # 将BGR转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_convert = time.time()
        
        # 推理关键点
        with torch.no_grad():  # 使用no_grad模式减少内存使用并提高速度
            keypoints = model.inference(frame_rgb)
        t_inference = time.time()
        
        # 将面部关键点的置信度设为0（不显示面部关键点）
        if dataset == 'wholebody':
            for idx in model._keypoints:
                # 面部关键点索引为23-90
                model._keypoints[idx][23:91, 2] = 0
        
        # 使用原生的draw方法
        result_rgb = model.draw(confidence_threshold=confidence_threshold)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        t_draw = time.time()
        
        # 计算FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # 平滑FPS计算
        fps_values.append(fps)
        if len(fps_values) > 10:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        # 每隔一定帧数打印一次FPS和性能信息
        if frame_count % fps_print_interval == 0:
            print(f"FPS: {avg_fps:.1f} | 转换: {(t_convert-t_start)*1000:.1f}ms | 推理: {(t_inference-t_convert)*1000:.1f}ms | 绘制: {(t_draw-t_inference)*1000:.1f}ms")
        
        # 显示结果（不显示FPS）
        cv2.imshow('ViTPose Real-time', result_bgr)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    # 释放资源
    model.reset()  # 重置跟踪器
    cap.release()
    cv2.destroyAllWindows()