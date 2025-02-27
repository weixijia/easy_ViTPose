import os
from typing import Optional
import typing

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .configs.ViTPose_common import data_cfg
from .sort import Sort
from .vit_models.model import ViTPose
from .vit_utils.inference import draw_bboxes, pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class VitInference:
    """用于使用 ViTPose 模型进行人体姿态估计的类,集成了 YOLOv8 人体检测和 SORT 跟踪。

    Args:
        model (str): ViT模型文件路径(.pth)
        yolo (str): YOLOv8模型路径
        model_name (str): ViT模型架构名称 ('s', 'b', 'l', 'h')
        dataset (str): 数据集名称,默认'wholebody'
        yolo_size (int): YOLOv8输入图像大小,默认320
        device (str): 推理设备,默认自动选择
        is_video (bool): 是否为视频输入,默认False
        single_pose (bool): 是否只检测单人姿态,默认False
        yolo_step (int): YOLO检测频率,默认每帧检测
        tracker_max_age (int): 跟踪器最大存活帧数
        tracker_min_hits (int): 跟踪器最小命中数
        tracker_iou_threshold (float): 跟踪器IOU阈值
    """
    def __init__(self, model: str,
                 yolo: str,
                 model_name: Optional[str] = None,
                 dataset: Optional[str] = 'wholebody',
                 yolo_size: Optional[int] = 320,
                 device: Optional[str] = None,
                 is_video: Optional[bool] = False,
                 single_pose: Optional[bool] = False,
                 yolo_step: Optional[int] = 1,
                 tracker_max_age: Optional[int] = None,
                 tracker_min_hits: Optional[int] = None,
                 tracker_iou_threshold: Optional[float] = 0.3):
        
        assert os.path.isfile(model), f'模型文件 {model} 不存在'
        assert os.path.isfile(yolo), f'YOLO模型 {yolo} 不存在'

        # 设备优先级: cuda > mps > cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.yolo = YOLO(yolo, task='detect')
        self.yolo_size = yolo_size
        self.yolo_step = yolo_step
        self.is_video = is_video
        self.single_pose = single_pose
        
        # 存储跟踪器参数
        self.tracker_max_age = tracker_max_age
        self.tracker_min_hits = tracker_min_hits
        self.tracker_iou_threshold = tracker_iou_threshold
        self.reset()

        # 推理过程中的状态保存
        self.save_state = True
        self._img = None
        self._yolo_res = None
        self._tracker_res = None
        self._keypoints = None

        # 设置数据集
        self.dataset = dataset
        # 只检测人类的类别ID
        self.yolo_classes = [0]  # 人类在COCO数据集中的类别ID为0

        # 加载模型配置
        assert model_name in ['s', 'b', 'l', 'h'], f'模型名称 {model_name} 无效'
        model_cfg = dyn_model_import(self.dataset, model_name)

        # 设置目标尺寸
        self.target_size = data_cfg['image_size']
        
        # 加载ViTPose模型
        self._vit_pose = ViTPose(model_cfg)
        self._vit_pose.eval()
        
        # 加载模型权重
        ckpt = torch.load(model, map_location='cpu', weights_only=True)
        if 'state_dict' in ckpt:
            self._vit_pose.load_state_dict(ckpt['state_dict'])
        else:
            self._vit_pose.load_state_dict(ckpt)
        self._vit_pose.to(torch.device(device))

    def reset(self):
        """重置推理类,准备处理新的视频。这将重置内部帧计数器,对视频这是必要的以重置跟踪器。"""
        use_tracker = self.is_video and not self.single_pose
        if use_tracker:
            max_age = self.tracker_max_age if self.tracker_max_age is not None else self.yolo_step
            min_hits = self.tracker_min_hits if self.tracker_min_hits is not None else (3 if self.yolo_step == 1 else 1)
            iou_threshold = self.tracker_iou_threshold
            
            self.tracker = Sort(
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold
            )
        else:
            self.tracker = None
        self.frame_counter = 0

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        """后处理热图以获得关键点及其概率"""
        points, prob = keypoints_from_heatmaps(
            heatmaps=heatmaps,
            center=np.array([[org_w // 2, org_h // 2]]),
            scale=np.array([[org_w, org_h]]),
            unbiased=True, 
            use_udp=True
        )
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        """对输入图像进行推理"""
        # 使用YOLOv8进行人体检测
        res_pd = np.empty((0, 5))
        results = None
        if (self.tracker is None or
           (self.frame_counter % self.yolo_step == 0 or self.frame_counter < 3)):
            results = self.yolo(img[..., ::-1], verbose=False, imgsz=self.yolo_size,
                              device=self.device if self.device != 'cuda' else 0,
                              classes=self.yolo_classes)[0]
            res_pd = np.array([r[:5].tolist() for r in results.boxes.data.cpu().numpy() 
                             if r[4] > 0.35]).reshape((-1, 5))
        self.frame_counter += 1

        frame_keypoints = {}
        scores_bbox = {}
        ids = None
        if self.tracker is not None:
            res_pd = self.tracker.update(res_pd)
            ids = res_pd[:, 5].astype(int).tolist()

        # 准备用于推理的边界框
        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        pad_bbox = 10

        if ids is None:
            ids = range(len(bboxes))

        for bbox, id, score in zip(bboxes, ids, scores):
            # 扩大边界框
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

            # 裁剪图像并填充到3/4宽高比
            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            # 进行关键点检测
            with torch.no_grad():
                img_input, org_h, org_w = self.pre_img(img_inf)
                img_input = torch.from_numpy(img_input).to(torch.device(self.device))
                heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
                keypoints = self.postprocess(heatmaps, org_w, org_h)[0]

            # 将关键点转换回原始图像坐标
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints[id] = keypoints
            scores_bbox[id] = score

        if self.save_state:
            self._img = img
            self._yolo_res = results
            self._tracker_res = (bboxes, ids, scores)
            self._keypoints = frame_keypoints
            self._scores_bbox = scores_bbox

        return frame_keypoints

    def draw(self, show_yolo=False, show_raw_yolo=False, confidence_threshold=0.5):
        """在图像上绘制关键点和边界框"""
        img = self._img.copy()
        bboxes, ids, scores = self._tracker_res

        if self._yolo_res is not None and (show_raw_yolo or (self.tracker is None and show_yolo)):
            img = np.array(self._yolo_res.plot())[..., ::-1]

        if show_yolo and self.tracker is not None:
            img = draw_bboxes(img, bboxes, ids, scores)

        img = np.array(img)[..., ::-1]  # RGB到BGR转换
        for idx, k in self._keypoints.items():
            img = draw_points_and_skeleton(img.copy(), k,
                                         joints_dict()[self.dataset]['skeleton'],
                                         person_index=idx,
                                         points_color_palette='gist_rainbow',
                                         skeleton_color_palette='jet',
                                         points_palette_samples=10,
                                         confidence_threshold=confidence_threshold)
        return img[..., ::-1]  # 返回RGB格式

    def pre_img(self, img):
        """预处理输入图像"""
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w