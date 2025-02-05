"""深度学习图像分类器基类和具体实现"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

class KLineImageDataset(Dataset):
    """K线图像数据集"""
    def __init__(self, data_frame: pd.DataFrame, transform=None):
        self.data_frame = data_frame
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['image']
        label = self.data_frame.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class BaseImageClassifier(ABC):
    """图像分类器基类"""
    def __init__(self, 
                 model_dir: str,
                 num_classes: int = 2,
                 learning_rate: float = 1e-4,
                 batch_size: int = 16,
                 num_epochs: int = 50):
        
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # 初始化模型（由子类实现）
        self.model = self._create_model(num_classes)
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
    @abstractmethod
    def _create_model(self, num_classes: int) -> nn.Module:
        """创建模型（由子类实现）"""
        pass
        
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        return optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
    def train(self, data_dir: str) -> Dict:
        """训练模型
        
        Args:
            data_dir: 数据目录
            
        Returns:
            训练结果字典
        """
        print(f"\n开始训练模型:")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.batch_size}")
        print(f"学习率: {self.learning_rate}")
        
        try:
            # 准备数据
            data_info = self._prepare_data(data_dir)
            train_dataset = KLineImageDataset(data_info['train_df'])
            val_dataset = KLineImageDataset(data_info['val_df'])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # 训练循环
            best_val_acc = 0.0
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            for epoch in range(self.num_epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
                for images, labels in train_bar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    train_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })
                
                train_loss = train_loss / len(train_loader)
                train_acc = 100. * correct / total
                
                # 验证阶段
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
                    for images, labels in val_bar:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        
                        val_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{100.*correct/total:.2f}%'
                        })
                
                val_loss = val_loss / len(val_loader)
                val_acc = 100. * correct / total
                
                # 更新学习率
                self.scheduler.step(val_acc)
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                    }, os.path.join(self.model_dir, 'best_model.pth'))
                
                # 记录训练过程
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # 绘制训练曲线
            self._plot_training_curves(train_losses, val_losses, train_accs, val_accs)
            
            return {
                'best_val_acc': best_val_acc,
                'final_train_acc': train_acc,
                'final_val_acc': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            
        except Exception as e:
            print(f"训练过程出错: {str(e)}")
            raise
            
    def _prepare_data(self, data_dir: str) -> Dict:
        """准备训练数据
        
        Args:
            data_dir: 特征数据目录
            
        Returns:
            数据信息字典
        """
        try:
            # 加载标签数据
            labels = np.load(os.path.join(data_dir, "labels.npy"))
            
            # 加载特征数据
            features = np.load(os.path.join(data_dir, "features.npy"))
            
            # 加载特征元数据
            with open(os.path.join(data_dir, "feature_meta.json"), "r") as f:
                feature_meta = json.load(f)
                
            # 找到kline_image字段的索引
            kline_image_idx = None
            for idx, (name, _) in enumerate(feature_meta.items()):
                if name == 'kline_image':
                    kline_image_idx = idx
                    break
                    
            if kline_image_idx is None:
                raise ValueError("在特征元数据中未找到 kline_image 字段")
                
            # 创建训练数据DataFrame
            data = []
            for idx, (label, feature) in enumerate(zip(labels, features)):
                image_name = feature[kline_image_idx]
                image_name = str(int(image_name))
                image_path = os.path.join(data_dir, "images", f"{image_name}.png")
                
                if os.path.exists(image_path):
                    data.append({
                        'image': image_path,
                        'label': int(label)
                    })
                else:
                    print(f"警告: 图像不存在: {image_path}")
                    
            if not data:
                raise ValueError(f"未找到任何有效的图像数据在 {data_dir}")
                
            df = pd.DataFrame(data)
            
            # 打印数据统计
            print(f"\n数据统计:")
            print(f"总样本数: {len(df)}")
            print(f"标签分布:\n{df['label'].value_counts()}")
            
            # 添加数据验证
            if len(df) < self.batch_size:
                raise ValueError(f"数据样本数({len(df)})小于批次大小({self.batch_size})")
                
            # 确保标签分布均衡
            label_counts = df['label'].value_counts()
            print("\n标签分布:")
            for label, count in label_counts.items():
                print(f"标签 {label}: {count} 样本 ({count/len(df):.2%})")
                
            # 划分训练集和验证集
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
            return {
                'train_df': train_df,
                'val_df': val_df
            }
            
        except Exception as e:
            print(f"数据准备出错: {str(e)}")
            raise

    def predict(self, image_path: str) -> Tuple[int, float]:
        """预测单个图像的类别和概率
        
        Args:
            image_path: 图像路径
            
        Returns:
            (预测类别, 预测概率)
        """
        self.model.eval()
        
        try:
            # 加载和预处理图像
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0][pred_class].item()
                
            return pred_class, pred_prob
            
        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            raise
            
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[int, float]]:
        """批量预测多个图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            预测结果列表，每个元素为 (预测类别, 预测概率)
        """
        self.model.eval()
        results = []
        
        try:
            # 创建数据加载器
            dataset = KLineImageDataset(
                pd.DataFrame({'image': image_paths, 'label': [0]*len(image_paths)})
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # 批量预测
            for images, _ in tqdm(dataloader, desc='Predicting'):
                images = images.to(self.device)
                with torch.no_grad():
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    pred_classes = torch.argmax(probs, dim=1)
                    pred_probs = probs[range(len(probs)), pred_classes]
                    
                    for cls, prob in zip(pred_classes, pred_probs):
                        results.append((cls.item(), prob.item()))
                        
            return results
            
        except Exception as e:
            print(f"批量预测过程出错: {str(e)}")
            raise
            
    @classmethod
    def load(cls, model_path: str, **kwargs) -> 'BaseImageClassifier':
        """加载已训练的模型
        
        Args:
            model_path: 模型文件路径
            **kwargs: 其他参数
            
        Returns:
            分类器实例
        """
        try:
            # 确保 model_dir 参数存在
            if 'model_dir' not in kwargs:
                kwargs['model_dir'] = os.path.dirname(model_path)
            
            # 创建分类器实例
            classifier = cls(**kwargs)
            
            # 加载模型权重
            checkpoint = torch.load(
                model_path,
                map_location=classifier.device
            )
            
            # 加载模型状态
            classifier.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 可选：加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            print(f"模型加载成功: {model_path}")
            print(f"验证准确率: {checkpoint.get('val_acc', 'N/A')}")
            
            return classifier
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
            
    def save(self, save_path: str, **extra_info):
        """保存模型
        
        Args:
            save_path: 保存路径
            **extra_info: 额外信息
        """
        try:
            # 准备保存的数据
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': {
                    'num_classes': self.model.fc.out_features,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                }
            }
            
            # 添加额外信息
            save_data.update(extra_info)
            
            # 保存模型
            torch.save(save_data, save_path)
            print(f"模型保存成功: {save_path}")
            
        except Exception as e:
            print(f"模型保存失败: {str(e)}")
            raise

class ResNet50Classifier(BaseImageClassifier):
    """ResNet50 分类器"""
    def _create_model(self, num_classes: int) -> nn.Module:
        model = models.resnet50(weights='IMAGENET1K_V1')
        
        # 冻结部分层
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        # 修改最后一层
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        return model

class ResNet18Classifier(BaseImageClassifier):
    """ResNet18 分类器 - 更轻量级"""
    def _create_model(self, num_classes: int) -> nn.Module:
        model = models.resnet18(weights='IMAGENET1K_V1')
        
        # 冻结部分层
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        return model

class EfficientNetClassifier(BaseImageClassifier):
    """EfficientNet 分类器"""
    def _create_model(self, num_classes: int) -> nn.Module:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # 冻结部分层
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-1].parameters():
            param.requires_grad = True
            
        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        
        return model

class FasterRCNNClassifier(BaseImageClassifier):
    """Faster R-CNN 分类器"""
    def _create_model(self, num_classes: int) -> nn.Module:
        # 加载预训练的 Faster R-CNN 模型
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # 修改分类器头部以适应我们的类别数
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 冻结主干网络
        for param in model.backbone.parameters():
            param.requires_grad = False
            
        return model
        
    def _create_optimizer(self) -> optim.Optimizer:
        """为检测任务创建优化器"""
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)
        
    def _prepare_detection_data(self, data_dir: str) -> Dict:
        """准备检测任务的训练数据
        
        Args:
            data_dir: 特征数据目录
            
        Returns:
            数据信息字典
        """
        try:
            # 加载标签数据
            labels = np.load(os.path.join(data_dir, "labels.npy"))
            
            # 加载特征数据
            features = np.load(os.path.join(data_dir, "features.npy"))
            
            # 加载特征元数据
            with open(os.path.join(data_dir, "feature_meta.json"), "r") as f:
                feature_meta = json.load(f)
                
            # 找到kline_image字段的索引
            kline_image_idx = None
            for idx, (name, _) in enumerate(feature_meta.items()):
                if name == 'kline_image':
                    kline_image_idx = idx
                    break
                    
            if kline_image_idx is None:
                raise ValueError("在特征元数据中未找到 kline_image 字段")
                
            # 创建训练数据DataFrame
            data = []
            for idx, (label, feature) in enumerate(zip(labels, features)):
                image_name = feature[kline_image_idx]
                image_name = str(int(image_name))
                image_path = os.path.join(data_dir, "images", f"{image_name}.png")
                
                if os.path.exists(image_path):
                    data.append({
                        'image': image_path,
                        'label': int(label)
                    })
                else:
                    print(f"警告: 图像不存在: {image_path}")
                    
            if not data:
                raise ValueError(f"未找到任何有效的图像数据在 {data_dir}")
                
            df = pd.DataFrame(data)
            
            # 打印数据统计
            print(f"\n数据统计:")
            print(f"总样本数: {len(df)}")
            print(f"标签分布:\n{df['label'].value_counts()}")
            
            # 添加数据验证
            if len(df) < self.batch_size:
                raise ValueError(f"数据样本数({len(df)})小于批次大小({self.batch_size})")
                
            # 确保标签分布均衡
            label_counts = df['label'].value_counts()
            print("\n标签分布:")
            for label, count in label_counts.items():
                print(f"标签 {label}: {count} 样本 ({count/len(df):.2%})")
                
            # 划分训练集和验证集
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # 为每个样本添加检测框信息
            def add_detection_info(row):
                # 读取图像获取尺寸
                image = Image.open(row['image'])
                w, h = image.size
                # 添加检测框坐标
                row['box'] = [0, 0, w, h]  # 使用整个图像作为目标区域
                return row
                
            train_df = train_df.apply(add_detection_info, axis=1)
            val_df = val_df.apply(add_detection_info, axis=1)
            
            return {
                'train_df': train_df,
                'val_df': val_df
            }
            
        except Exception as e:
            print(f"数据准备出错: {str(e)}")
            raise
        
    def train(self, data_dir: str) -> Dict:
        """训练检测模型"""
        print("\n开始训练 Faster R-CNN 模型:")
        print(f"设备: {self.device}")
        
        try:
            # 准备数据
            data_info = self._prepare_detection_data(data_dir)
            train_dataset = KLineDetectionDataset(data_info['train_df'])
            val_dataset = KLineDetectionDataset(data_info['val_df'])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=self._detection_collate_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=self._detection_collate_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # 训练循环
            best_val_map = 0.0
            train_losses = []
            val_maps = []
            
            for epoch in range(self.num_epochs):
                # 训练阶段
                self.model.train()
                epoch_loss = 0.0
                train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
                
                for images, targets in train_bar:
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    self.optimizer.zero_grad()
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    losses.backward()
                    self.optimizer.step()
                    
                    epoch_loss += losses.item()
                    train_bar.set_postfix({'loss': f'{losses.item():.4f}'})
                
                epoch_loss = epoch_loss / len(train_loader)
                train_losses.append(epoch_loss)
                
                # 验证阶段
                val_map = self._evaluate_detection(val_loader)
                val_maps.append(val_map)
                
                # 更新学习率
                self.scheduler.step(val_map)
                
                # 保存最佳模型
                if val_map > best_val_map:
                    best_val_map = val_map
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_map': val_map,
                    }, os.path.join(self.model_dir, 'best_model.pth'))
                
                print(f'\nEpoch {epoch+1}/{self.num_epochs}:')
                print(f'Train Loss: {epoch_loss:.4f}')
                print(f'Val mAP: {val_map:.4f}')
            
            return {
                'best_val_map': best_val_map,
                'train_losses': train_losses,
                'val_maps': val_maps
            }
            
        except Exception as e:
            print(f"训练过程出错: {str(e)}")
            raise
            
    def _evaluate_detection(self, val_loader) -> float:
        """评估检测模型性能"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Evaluating'):
                images = [image.to(self.device) for image in images]
                predictions = self.model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
        # 计算mAP
        map_score = self._calculate_map(all_predictions, all_targets)
        return map_score
        
    def _calculate_map(self, predictions, targets) -> float:
        """计算平均精度均值(mAP)"""
        # 简化版mAP计算
        total_ap = 0
        num_classes = 2  # 二分类问题
        
        for pred, target in zip(predictions, targets):
            # 计算IoU
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            target_boxes = target['boxes']
            
            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                continue
                
            # 使用0.5作为IoU阈值
            ious = self._box_iou(pred_boxes, target_boxes)
            max_ious, _ = ious.max(dim=1)
            
            # 计算AP
            ap = ((max_ious > 0.5) * pred_scores).sum() / len(target_boxes)
            total_ap += ap.item()
            
        return total_ap / len(predictions) if predictions else 0
        
    @staticmethod
    def _box_iou(boxes1, boxes2):
        """计算两组框之间的IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        
        return inter / union
        
    @staticmethod
    def _detection_collate_fn(batch):
        """数据批次整理函数"""
        images = []
        targets = []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        return images, targets

class KLineDetectionDataset(Dataset):
    """K线图像检测数据集"""
    def __init__(self, data_frame: pd.DataFrame, transform=None):
        self.data_frame = data_frame
        self.transform = transform or transforms.Compose([
            transforms.Resize((800, 800)),  # Faster R-CNN 推荐的输入尺寸
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['image']
        label = self.data_frame.iloc[idx]['label']
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 获取图像尺寸
        w, h = image.size
        
        # 转换图像
        if self.transform:
            image = self.transform(image)
            
        # 创建检测目标
        # 这里我们使用整个图像作为目标区域
        boxes = torch.tensor([[0, 0, w, h]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, target

# 工厂类用于创建分类器
class ClassifierFactory:
    """分类器工厂类"""
    _classifiers = {
        'resnet50': ResNet50Classifier,
        'resnet18': ResNet18Classifier,
        'efficientnet': EfficientNetClassifier,
        'fasterrcnn': FasterRCNNClassifier
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> BaseImageClassifier:
        """创建分类器实例
        
        Args:
            model_type: 模型类型
            **kwargs: 其他参数
            
        Returns:
            分类器实例
        """
        if model_type not in cls._classifiers:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        return cls._classifiers[model_type](**kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用的模型类型列表"""
        return list(cls._classifiers.keys())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练或预测K线图像')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                      help='运行模式：train-训练，predict-预测')
    parser.add_argument('--data', type=str, required=True,
                      help='数据目录(训练模式)或图像路径(预测模式)')
    parser.add_argument('--model', type=str, default='resnet50',
                      choices=ClassifierFactory.get_available_models(),
                      help='模型类型')
    parser.add_argument('--model-path', type=str,
                      help='模型文件路径(预测模式)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--output', type=str, default='image_models',
                      help='模型输出目录(训练模式)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 训练模式
        classifier = ClassifierFactory.create(
            args.model,
            model_dir=args.output,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr
        )
        results = classifier.train(args.data)
        
    else:
        # 预测模式
        if not args.model_path:
            raise ValueError("预测模式需要指定模型文件路径(--model-path)")
            
        # 加载模型
        model_dir = os.path.dirname(args.model_path)
        classifier = ClassifierFactory.create(
            args.model,
            model_dir=model_dir  # 确保传入 model_dir
        )
        classifier = type(classifier).load(args.model_path, model_dir=model_dir)
        
        # 预测
        if os.path.isdir(args.data):
            # 批量预测目录下的所有图像
            images_dir = os.path.join(args.data, 'images')
            image_paths = [
                os.path.join(images_dir, f) for f in os.listdir(images_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            results = classifier.predict_batch(image_paths)
            
            # 检查是否存在标签文件
            label_file = os.path.join(args.data, 'labels.npy')
            if os.path.exists(label_file):
                labels = np.load(label_file)
                
                # 统计指标
                correct = 0
                total = 0
                confusion_matrix = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
                
                print("\n预测结果与标签对比:")
                for i, (img_path, (pred_class, prob)) in enumerate(zip(image_paths, results)):
                    true_label = int(labels[i])
                    is_correct = pred_class == true_label
                    correct += int(is_correct)
                    total += 1
                    confusion_matrix[true_label][pred_class] += 1
                    
                    print(f"图像: {os.path.basename(img_path)}")
                    print(f"真实标签: {true_label}, 预测类别: {pred_class}, 概率: {prob:.2%}")
                    print(f"预测{'正确' if is_correct else '错误'}\n")
                
                # 输出统计结果
                print("\n统计结果:")
                
                # 计算混淆矩阵基础值
                tn = confusion_matrix[0][0]
                fp = confusion_matrix[0][1]
                fn = confusion_matrix[1][0] 
                tp = confusion_matrix[1][1]
                
                # 计算基础评估指标
                def safe_divide(x, y):
                    return x / y if y > 0 else 0
                    
                def calculate_basic_metrics(tp, fp, fn, tn):
                    precision = safe_divide(tp, tp + fp)
                    recall = safe_divide(tp, tp + fn)
                    f1 = safe_divide(2 * precision * recall, precision + recall)
                    balanced_acc = (safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp)) / 2
                    
                    # MCC计算
                    numerator = (tp * tn - fp * fn)
                    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
                    mcc = safe_divide(numerator, denominator)
                    
                    return precision, recall, f1, balanced_acc, mcc
                
                # 计算ROC AUC
                def calculate_roc_auc(y_true, y_scores):
                    n_pos = sum(y_true)
                    n_neg = len(y_true) - n_pos
                    
                    if n_pos == 0 or n_neg == 0:
                        return 0
                        
                    pos_scores = [s for t, s in zip(y_true, y_scores) if t == 1]
                    neg_scores = [s for t, s in zip(y_true, y_scores) if t == 0]
                    
                    correct_pairs = sum(1 for pos in pos_scores for neg in neg_scores if pos > neg)
                    return correct_pairs / (n_pos * n_neg)
                
                # 计算所有指标
                precision, recall, f1, balanced_acc, mcc = calculate_basic_metrics(tp, fp, fn, tn)
                
                y_true = [1 if i == 1 else 0 for i in labels]
                y_scores = [results[i][1] for i in range(len(results))]
                roc_auc = calculate_roc_auc(y_true, y_scores)
                
                # 输出结果
                print(f"ROC AUC: {roc_auc:.4f}")
                print(f"准确率: {correct/total:.2%} ({correct}/{total})")
                print(f"平衡准确率: {balanced_acc:.4f}")
                print(f"精确率: {precision:.4f}")
                print(f"召回率: {recall:.4f}")
                print(f"MCC: {mcc:.4f}")
                print(f"F1分数: {f1:.4f}")
                
                print("\n混淆矩阵:")
                print("┌─────────────────────────┐")
                print("│ 真实\\预测  0   1 │")
                print(f"│ 类别 0          {confusion_matrix[0][0]:<3} {confusion_matrix[0][1]:<3} │")
                print(f"│ 类别 1          {confusion_matrix[1][0]:<3} {confusion_matrix[1][1]:<3} │")
                print("└─────────────────────────┘")
            
            else:
                # 无标签文件时只输出预测结果
                print("\n预测结果:")
                for img_path, (pred_class, prob) in zip(image_paths, results):
                    print(f"图像: {os.path.basename(img_path)}")
                    print(f"预测类别: {pred_class}, 概率: {prob:.2%}\n")
                
        else:
            # 预测单张图像
            pred_class, prob = classifier.predict(args.data)
            print("\n预测结果:")
            print(f"图像: {os.path.basename(args.data)}")
            print(f"预测类别: {pred_class}, 概率: {prob:.2%}") 