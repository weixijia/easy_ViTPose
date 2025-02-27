import cv2
import time
import torch
from easy_ViTPose import VitInference

# 初始化网络摄像头
cap = cv2.VideoCapture(0)  # 使用默认摄像头，如果有多个摄像头，可以尝试不同的索引

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
else:
    device = 'cpu'
    print("使用 CPU 运行")

# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition
model_path = 'vitpose-s-wholebody.pth'
yolo_path = 'yolov8n.pt'  # 使用YOLO latest v11 model

# 是否使用 wholebody 数据集（包含手部关键点）
# 注意：要使用 wholebody 数据集，您需要相应的模型权重文件，例如 'vitpose-b-wholebody.pth'
use_wholebody = True  # 如果您有 wholebody 模型权重，可以设置为 True

if use_wholebody:
    model_path = 'vitpose-s-wholebody.pth'  # 您需要下载这个模型权重文件
    dataset = 'wholebody'
else:
    dataset = None  # 自动从模型名称推断

# If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
# If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
# dataset and det_class parameters can be inferred from the ckpt name, but you can specify them.
model = VitInference(model_path, yolo_path, model_name='s', yolo_size=320, is_video=True, 
                    device=None, dataset=dataset)

# 用于计算FPS
prev_time = 0
curr_time = 0

try:
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
            
        # 将BGR转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 推理关键点
        keypoints = model.inference(frame_rgb)
        
        # 绘制结果
        result_rgb = model.draw(show_yolo=True)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # 计算并显示FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(result_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('ViTPose Real-time', result_bgr)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    # 释放资源
    model.reset()  # 重置跟踪器
    cap.release()
    cv2.destroyAllWindows()