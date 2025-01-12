"""模型训练和评估相关功能"""
from typing import Dict, Tuple, List, Optional
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

class ModelTrainer:
    """模型训练器"""
    def __init__(self, params: Dict = None):
        """初始化训练器"""
        self.params = params or {
            'max_depth': 3,
            'learning_rate': 0.1,  # 使用learning_rate替代eta
            'objective': 'binary:logistic',
            'eval_metric': 'auc',  # 简化评估指标
            'tree_method': 'hist',
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'n_jobs': 4,  # 使用n_jobs替代nthread
            'random_state': 42  # 使用random_state替代seed
        }
        self.model = None
        self.feature_importance = {}
        self.best_iteration = None
        self.train_metrics = {}
        self.val_metrics = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
             val_size: float = 0.2) -> Dict:
        """训练模型"""
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42, stratify=y
        )
        
        # 创建并训练模型
        model = XGBClassifier(**self.params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        self.model = model
        self.best_iteration = getattr(model, 'best_iteration', None)
        
        # 计算特征重要性
        importance = dict(zip(feature_names, model.feature_importances_))
        self.feature_importance = importance
        
        # 计算评估指标
        metrics = {
            'train': self._calculate_metrics(y_train, model.predict_proba(X_train)[:, 1]),
            'val': self._calculate_metrics(y_val, model.predict_proba(X_val)[:, 1])
        }
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """计算评估指标"""
        pred_labels = y_pred > 0.5
        return {
            'auc': roc_auc_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, pred_labels),
            'precision': precision_score(y_true, pred_labels),
            'recall': recall_score(y_true, pred_labels),
            'f1': f1_score(y_true, pred_labels)
        }
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not hasattr(self, 'model') or self.model is None:
            print("警告: 模型尚未训练，无法绘制训练曲线")
            return
        
        if not self.train_metrics and not self.val_metrics:
            print("警告: 没有训练过程的指标记录，无法绘制训练曲线")
            return
        
        # 如果没有训练指标，则跳过绘图
        if not self.train_metrics:
            return
        
        metrics = list(self.train_metrics.keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            ax.plot(self.train_metrics[metric], label='train')
            if metric in self.val_metrics:
                ax.plot(self.val_metrics[metric], label='validation')
            if hasattr(self, 'best_iteration') and self.best_iteration is not None:
                ax.axvline(x=self.best_iteration, color='r', linestyle='--', label='best iteration')
            ax.set_title(f'Training Curve - {metric}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
    
    def plot_feature_importance(self, top_n: int = 20):
        """绘制特征重要性图"""
        if not self.feature_importance:
            raise ValueError("请先训练模型")
            
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        names = [x[0] for x in sorted_features]
        values = [x[1] for x in sorted_features]
        
        plt.figure(figsize=(10, max(6, len(names)*0.3)))
        plt.barh(range(len(names)), values)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def save_model(self, filename: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        self.model.save_model(filename)
    
    def tune_parameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """简化的参数调优"""
        # 分割训练集、验证集和测试集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        best_params = {}
        best_score = 0
        
        # 简化的参数网格
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'min_child_weight': [1, 3]
        }
        
        # 手动搜索
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for min_child_weight in param_grid['min_child_weight']:
                        current_params = {
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'min_child_weight': min_child_weight,
                            'objective': 'binary:logistic',
                            'tree_method': 'hist',
                            'eval_metric': 'auc',
                            'random_state': 42,
                            'n_estimators': 100  # 固定树的数量
                        }
                        
                        # 训练模型
                        model = XGBClassifier(**current_params)
                        model.fit(X_train, y_train)
                        
                        # 在验证集上评估
                        y_pred = model.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, y_pred)
                        
                        print(f"参数: {current_params}")
                        print(f"验证集AUC: {score:.4f}\n")
                        
                        if score > best_score:
                            best_score = score
                            best_params = current_params.copy()
        
        # 使用最佳参数在测试集上评估
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        test_score = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
        
        print(f"\n最佳参数 (验证集得分: {best_score:.4f}, 测试集得分: {test_score:.4f}):")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        return best_params
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """手动实现交叉验证评估模型"""
        from sklearn.model_selection import KFold
        
        # 初始化结果存储
        metrics = {
            'train_auc_mean': 0, 'train_auc_std': 0,
            'val_auc_mean': 0, 'val_auc_std': 0,
            'train_precision_mean': 0, 'train_precision_std': 0,
            'val_precision_mean': 0, 'val_precision_std': 0,
            'train_recall_mean': 0, 'train_recall_std': 0,
            'val_recall_mean': 0, 'val_recall_std': 0,
            'train_f1_mean': 0, 'train_f1_std': 0,
            'val_f1_mean': 0, 'val_f1_std': 0
        }
        
        # 存储每折的指标
        train_metrics = []
        val_metrics = []
        
        # 创建交叉验证分割器
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 进行交叉验证
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n训练折 {fold}/{cv}")
            
            # 分割数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            model = XGBClassifier(**self.params)
            model.fit(X_train, y_train)
            
            # 预测
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # 计算指标
            train_metric = self._calculate_metrics(y_train, train_pred)
            val_metric = self._calculate_metrics(y_val, val_pred)
            
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
            # 打印当前折的结果
            print(f"验证集 AUC: {val_metric['auc']:.4f}")
        
        # 计算平均值和标准差
        for metric in ['auc', 'precision', 'recall', 'f1']:
            # 训练集指标
            train_values = [m[metric] for m in train_metrics]
            metrics[f'train_{metric}_mean'] = np.mean(train_values)
            metrics[f'train_{metric}_std'] = np.std(train_values)
            
            # 验证集指标
            val_values = [m[metric] for m in val_metrics]
            metrics[f'val_{metric}_mean'] = np.mean(val_values)
            metrics[f'val_{metric}_std'] = np.std(val_values)
        
        return metrics
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """特征选择"""
        model = XGBClassifier(**self.params)
        selector = SelectFromModel(model, prefit=False)
        selector.fit(X, y)
        
        # 获取选中的特征
        selected_features = [
            feature_names[i] 
            for i in range(len(feature_names)) 
            if selector.get_support()[i]
        ]
        
        print("\n选中的特征:")
        for feat in selected_features:
            print(f"- {feat}")
            
        return selected_features
    
    def plot_model_evaluation(self, X: np.ndarray, y: np.ndarray):
        """绘制模型评估图"""
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        # 训练模型并获取预测概率
        model = XGBClassifier(**self.params)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.close()
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 5))
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig('pr_curve.png')
        plt.close()
    
    def predict_with_confidence(self, X: np.ndarray, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """使用Bootstrap方法估计预测的置信区间
        
        Args:
            X: 特征矩阵
            n_iterations: bootstrap迭代次数
            
        Returns:
            mean_pred: 平均预测值
            conf_interval: 95%置信区间
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        predictions = []
        n_samples = X.shape[0]
        
        for _ in range(n_iterations):
            # 随机采样训练数据
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            
            # 直接使用XGBClassifier的predict_proba
            pred = self.model.predict_proba(X_sample)[:, 1]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 计算置信区间
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        conf_interval = 1.96 * std_pred  # 95%置信区间
        
        return mean_pred, conf_interval
    
    def analyze_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """分析特征之间的相关性和与目标变量的关系
        
        Args:
            X: 特征矩阵
            y: 标签数组
            feature_names: 特征名列表
        """
        # 创建特征DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # 计算特征相关性
        corr = df.corr()
        
        # 绘制相关性热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        plt.close()
        
        # 分析每个特征与目标变量的关系
        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, feature in enumerate(feature_names, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.boxplot(x='target', y=feature, data=df)
            plt.title(f'{feature} vs Target')
        
        plt.tight_layout()
        plt.savefig('feature_target_analysis.png')
        plt.close()
        
        # 输出特征统计信息
        print("\n特征统计信息:")
        stats = df[feature_names].describe()
        print(stats)
        
        # 计算特征与目标变量的互信息
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({'Feature': feature_names, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values('MI Score', ascending=False)
        print("\n特征互信息得分:")
        print(mi_df)
    
    def explain_predictions(self, X: np.ndarray, feature_names: List[str], sample_idx: int = None):
        """解释模型预测
        
        Args:
            X: 特征矩阵
            feature_names: 特征名列表
            sample_idx: 要解释的样本索引，None则随机选择
        """
        import shap
        
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(self.model)
        
        if sample_idx is None:
            # 随机选择一个样本
            sample_idx = np.random.randint(X.shape[0])
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X)
        
        # 绘制单个预测的解释
        plt.figure(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx,:],
            X[sample_idx,:],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'Sample {sample_idx} Prediction Explanation')
        plt.tight_layout()
        plt.savefig('prediction_explanation.png')
        plt.close()
        
        # 绘制特征重要性摘要图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('shap_importance.png')
        plt.close()

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, model_file: str, threshold: float = 0.6):
        """初始化评估器
        
        Args:
            model_file: 模型文件路径
            threshold: 预测概率阈值
        """
        self.model = xgb.Booster()
        self.model.load_model(model_file)
        self.threshold = threshold
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率值
        """
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
        
    def evaluate_trades(self, trades: List[Dict], bsp_academy: List[int]) -> Dict:
        """评估交易结果"""
        high_prob_trades = [t for t in trades if t['prob'] > self.threshold]
        
        # 基本统计
        stats = self._calculate_basic_stats(trades, high_prob_trades, bsp_academy)
        
        # 准确率指标
        stats.update(self._calculate_accuracy_metrics(high_prob_trades, bsp_academy))
        
        # 分类统计
        stats.update(self._calculate_category_stats(high_prob_trades, bsp_academy))
        
        return stats
    
    def _calculate_basic_stats(self, trades, high_prob_trades, bsp_academy):
        """计算基本统计信息"""
        return {
            'total_signals': len(trades),
            'high_prob_signals': len(high_prob_trades),
            'buy_signals': sum(1 for t in trades if t['is_buy']),
            'sell_signals': sum(1 for t in trades if not t['is_buy']),
            'high_prob_buy': sum(1 for t in high_prob_trades if t['is_buy']),
            'high_prob_sell': sum(1 for t in high_prob_trades if not t['is_buy']),
            'actual_points': len(bsp_academy)
        }
    
    def _calculate_accuracy_metrics(self, high_prob_trades, bsp_academy):
        """计算准确率指标"""
        correct_predictions = sum(1 for t in high_prob_trades if t['idx'] in bsp_academy)
        precision = correct_predictions / len(high_prob_trades) if high_prob_trades else 0
        recall = correct_predictions / len(bsp_academy) if bsp_academy else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_category_stats(self, high_prob_trades, bsp_academy):
        """计算分类统计"""
        buy_correct = sum(1 for t in high_prob_trades if t['idx'] in bsp_academy and t['is_buy'])
        sell_correct = sum(1 for t in high_prob_trades if t['idx'] in bsp_academy and not t['is_buy'])
        
        return {
            'buy_precision': buy_correct / sum(1 for t in high_prob_trades if t['is_buy']) \
                if sum(1 for t in high_prob_trades if t['is_buy']) > 0 else 0,
            'sell_precision': sell_correct / sum(1 for t in high_prob_trades if not t['is_buy']) \
                if sum(1 for t in high_prob_trades if not t['is_buy']) > 0 else 0
        } 