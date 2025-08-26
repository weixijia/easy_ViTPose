#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class SkeletonDataset(Dataset):
    def __init__(self, csv_file, sequence_length=32, frame_skip=2, scaler=None):
        """
        Args:
            csv_file: 带标签的骨骼点CSV文件路径
            sequence_length: 序列长度，用于滑动窗口
            frame_skip: 跳帧数，每隔几帧采样一次
            scaler: 特征标准化器，如果为None则创建新的
        """
        self.df = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        
        # 提取特征列（所有x,y坐标和置信度分数）
        feature_cols = [col for col in self.df.columns if any(x in col for x in ['_x', '_y', '_score'])]
        raw_features = self.df[feature_cols].values
        
        # 检查原始数据
        if np.any(np.isnan(raw_features)):
            print("警告：原始数据中包含 NaN 值，将被替换为 0")
            raw_features = np.nan_to_num(raw_features, nan=0.0)
        
        # 计算额外的运动特征
        print("计算运动特征...")
        try:
            motion_features = self._compute_motion_features(raw_features)
            print("特征计算完成")
        except Exception as e:
            print(f"特征计算出错: {str(e)}")
            raise
        
        # 合并原始特征和运动特征
        self.features = motion_features
        print(f"特征维度: {self.features.shape[1]}")
        
        # 标准化特征（使用稳健的方法）
        if scaler is None:
            self.scaler = StandardScaler(with_mean=True, with_std=True)
            # 先移除异常值
            valid_mask = np.all(np.abs(self.features) < 1e6, axis=1)
            if not np.all(valid_mask):
                print(f"警告：检测到 {np.sum(~valid_mask)} 个异常值")
            self.features = self.scaler.fit_transform(self.features[valid_mask])
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        
        # 再次检查是否有无效值
        if np.any(np.isnan(self.features)) or np.any(np.isinf(self.features)):
            print("警告：标准化后数据中包含无效值，将被替换为 0")
            self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['action_label'])
        self.num_classes = len(self.label_encoder.classes_)
        print(f"\n类别数量: {self.num_classes}")
        print("类别映射:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label} -> {i}")
        
        # 创建滑动窗口序列，使用跳帧采样
        self.sequences = []
        self.sequence_labels = []
        
        # 使用跳帧后的序列长度
        effective_length = sequence_length * frame_skip
        
        for i in range(0, len(self.features) - effective_length + 1, frame_skip):
            # 采样帧
            sequence = self.features[i:i + effective_length:frame_skip]
            if len(sequence) == sequence_length:  # 确保序列长度正确
                # 使用序列中最后一帧的标签作为整个序列的标签
                label = self.labels[i + effective_length - 1]
                self.sequences.append(sequence)
                self.sequence_labels.append(label)
        
        print(f"\n数据统计:")
        print(f"- 原始帧数: {len(self.features)}")
        print(f"- 采样间隔: {frame_skip} 帧")
        print(f"- 序列长度: {sequence_length} (实际覆盖 {effective_length} 帧)")
        print(f"- 总序列数: {len(self.sequences)}")
        
        # 打印每个类别的序列数量
        unique_labels, counts = np.unique(self.sequence_labels, return_counts=True)
        print("\n各类别序列数量:")
        for label, count in zip(unique_labels, counts):
            print(f"  {self.label_encoder.inverse_transform([label])[0]}: {count}")
    
    def _compute_motion_features(self, raw_features):
        """计算运动相关的特征"""
        n_frames = len(raw_features)
        n_keypoints = raw_features.shape[1] // 3  # 每个关键点有x,y,score三个值
        
        # 提取坐标和置信度分数
        coords = raw_features[:, :(n_keypoints * 2)].reshape(n_frames, n_keypoints, 2)  # (frames, keypoints, 2)
        scores = raw_features[:, (n_keypoints * 2):].reshape(n_frames, n_keypoints, 1)  # (frames, keypoints, 1)
        
        # 使用置信度分数作为权重
        valid_mask = (scores > 0.3).astype(np.float32)  # 忽略低置信度的关键点
        
        features = []
        
        # 1. 速度特征（相邻帧之间的差异）
        velocities = np.zeros_like(coords)
        velocities[1:] = coords[1:] - coords[:-1]
        velocities = velocities * valid_mask[:, :, :2]  # 只应用于x,y坐标
        features.append(velocities.reshape(n_frames, -1))
        
        # 2. 加速度特征
        accelerations = np.zeros_like(coords)
        accelerations[2:] = velocities[2:] - velocities[1:-1]
        accelerations = accelerations * valid_mask[:, :, :2]
        features.append(accelerations.reshape(n_frames, -1))
        
        # 3. 相对位置特征
        # 使用重心作为参考点
        center = np.sum(coords * valid_mask[:, :, :2], axis=1, keepdims=True) / (np.sum(valid_mask[:, :, :2], axis=1, keepdims=True) + 1e-6)
        relative_pos = coords - center
        relative_pos = relative_pos * valid_mask[:, :, :2]
        features.append(relative_pos.reshape(n_frames, -1))
        
        # 4. 关节角度特征
        angles = np.zeros((n_frames, n_keypoints))
        # 计算相对于垂直方向的角度
        angles = np.arctan2(relative_pos[:, :, 1], relative_pos[:, :, 0])
        angles = np.where(np.isnan(angles), 0, angles)  # 处理无效值
        angles = angles * valid_mask[:, :, 0]  # 使用第一个维度的mask
        features.append(angles.reshape(n_frames, -1))
        
        # 5. 关节间距离
        distances = np.sqrt(np.sum(relative_pos ** 2, axis=2))
        distances = np.where(np.isnan(distances), 0, distances)  # 处理无效值
        distances = distances * valid_mask[:, :, 0]
        features.append(distances.reshape(n_frames, -1))
        
        # 6. 运动幅度（速度的大小）
        motion_magnitude = np.sqrt(np.sum(velocities ** 2, axis=2))
        motion_magnitude = np.where(np.isnan(motion_magnitude), 0, motion_magnitude)
        motion_magnitude = motion_magnitude * valid_mask[:, :, 0]
        features.append(motion_magnitude.reshape(n_frames, -1))
        
        # 合并所有特征
        all_features = np.concatenate(features, axis=1)
        
        # 处理无效值
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 添加原始特征
        final_features = np.concatenate([raw_features, all_features], axis=1)
        
        # 确保没有无效值
        assert not np.any(np.isnan(final_features)), "特征中包含 NaN 值"
        assert not np.any(np.isinf(final_features)), "特征中包含 Inf 值"
        
        return final_features
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = self.sequence_labels[idx]
        return sequence, label

class SkeletonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(SkeletonLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size)  # 添加批标准化
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 正确初始化权重
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
            elif 'fc.weight' in name:
                nn.init.xavier_normal_(param)
            elif 'fc.bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 只使用最后一个时间步的输出
        out = self.bn(out[:, -1, :])  # 应用批标准化
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for sequences, labels in train_pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for sequences, labels in val_pbar:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
            }, 'best_model.pth')
            print(f'>>> 保存最佳模型 (Val Loss: {val_loss:.4f})')
    
    return train_losses, val_losses

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查可用设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用 CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 MPS")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    
    # 加载数据
    dataset = SkeletonDataset('reference_skeletons_with_labels.csv', sequence_length=32)
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n数据集大小:")
    print(f"- 训练集: {len(train_dataset)}")
    print(f"- 验证集: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 模型参数
    input_size = dataset.features.shape[1]  # 特征维度
    hidden_size = 128
    num_layers = 2
    num_classes = dataset.num_classes
    
    print(f"\n模型参数:")
    print(f"- 输入维度: {input_size}")
    print(f"- 隐藏层大小: {hidden_size}")
    print(f"- LSTM层数: {num_layers}")
    print(f"- 类别数: {num_classes}")
    
    # 创建模型
    model = SkeletonLSTM(input_size, hidden_size, num_layers, num_classes)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    print("\n训练完成！模型和训练曲线已保存")

if __name__ == "__main__":
    main() 