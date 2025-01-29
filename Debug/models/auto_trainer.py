from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import optuna
import xgboost as xgb

class AutoTrainer:
    """增强版AutoML训练器"""
    def __init__(self, time_limit=3600, eval_metric='roc_auc'):
        self.time_limit = time_limit
        self.eval_metric = eval_metric
        self.predictor = None
        self.best_model = None
        self.feature_importance = None
        
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], model_dir: str):
        """训练模型"""
        # 转换为DataFrame
        train_data = pd.DataFrame(X, columns=feature_names)
        train_data['label'] = y
        
        # AutoGluon训练
        self.predictor = TabularPredictor(
            label='label',
            eval_metric=self.eval_metric,
            problem_type='binary',
            path=model_dir+"/autogluon"  # 设置保存路径
        )
        
        # 训练模型
        self.predictor.fit(
            train_data,
            time_limit=self.time_limit,
            presets='best_quality',
            num_bag_folds=5,
            num_stack_levels=2,
            hyperparameters={  # 添加模型配置
                'GBM': [  # LightGBM
                    {'num_boost_round': 100},
                    {'num_boost_round': 200},
                ],
                'CAT': {},  # CatBoost
                'RF': [  # Random Forest
                    {'n_estimators': 100},
                    {'n_estimators': 200},
                ],
                'XT': {},  # Extra Trees
                'XGB': {},  # XGBoost
            },
            ag_args_fit={  # 添加训练配置
                'num_gpus': 0,  # 不使用GPU
                'num_cpus': 4,  # 限制CPU使用数量
            }
        )
        
        # 获取模型性能
        leaderboard = self.predictor.leaderboard()
        print("\n模型性能排行:")
        print(leaderboard)
        
        # 获取最佳模型
        self.best_model = self.predictor.model_best
        
        # 计算特征重要性
        # try:
        #     self.feature_importance = self.predictor.feature_importance(data=train_data)
        # except:
        #     print("无法计算特征重要性")
            
        return self.predictor
        
    def get_model_insights(self):
        """获取模型分析"""
        if self.predictor is None:
            return None
            
        # 获取模型性能排行
        try:
            leaderboard = self.predictor.leaderboard()
            leaderboard_dict = {
                'model_types': leaderboard.index.tolist(),
                'scores': leaderboard['score_val'].tolist(),
                'fit_times': leaderboard['fit_time'].tolist()
            }
        except Exception as e:
            print(f"获取模型性能排行失败: {str(e)}")
            leaderboard_dict = None
            
        # 获取训练摘要
        try:
            # 获取基本信息
            model_names = self.predictor.model_names()
            best_model = self.predictor.model_best
            
            fit_summary = {
                'num_models': len(model_names),
                'model_names': list(model_names),  # 转换为普通列表
                'best_model': best_model,  # 直接使用模型名称
                'problem_type': str(self.predictor.problem_type),  # 转换为字符串
                'eval_metric': str(self.predictor.eval_metric)  # 转换为字符串
            }
            
            # 从 leaderboard 中获取训练时间
            if leaderboard_dict:
                total_time = sum(leaderboard_dict['fit_times'])
                fit_summary['total_time'] = float(total_time)
                
        except Exception as e:
            print(f"获取训练摘要失败: {str(e)}")
            fit_summary = None
            
        insights = {
            'leaderboard': leaderboard_dict,
            'fit_summary': fit_summary,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
        }
        return insights
        
    @classmethod
    def load_model(cls, path: str):
        """加载模型"""
        trainer = cls()
        trainer.predictor = TabularPredictor.load(path)
        return trainer
        
    def predict(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """预测"""
        if self.predictor is None:
            raise ValueError("No model to predict")
            
        test_data = pd.DataFrame(X, columns=feature_names)
        return self.predictor.predict_proba(test_data) 