import json
from typing import Dict, TypedDict

import numpy as np
import xgboost as xgb
import os,sys

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotlyDriver import CPlotlyDriver
from models.trainer import ModelTrainer
from models.feature_engine import FeatureEngine, FeatureType
from models.model_manager import ModelManager


class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 400,
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotlyDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.figure.show()
    # plot_driver.save2img("label.png")


def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }

if __name__ == "__main__":
    """
    本示例旨在展示如何收集策略生成的买卖点特征
    并将这些特征作为样本，用于训练模型（以XGB为示例）
    从而预测买卖点的准确性

    注意：在本示例中，训练和预测都使用同一份数据，这在实际应用中是不合理的，仅作为示例
    """
    code = "BTC/USDT"
    begin_time = "2020-01-01"
    end_time = "2021-01-01"
    # end_time = "2024-01-01"
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_60M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,1p',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    # 初始化特征引擎
    feature_engine = FeatureEngine()

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征
    kline_data = []  # 存储K线数据用于后续分析
    
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        kline_data.append(last_klu)
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type:
            continue

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            # 使用特征引擎计算特征
            market_features = feature_engine.get_features(
                kline_data, 
                len(kline_data)-1,
                bsp_list
            )
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(market_features)

    # 生成特征数据
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    plot_marker, feature_meta, X, y = feature_engine.save_features(bsp_dict, bsp_academy)
    
    # 画图检查label是否正确
    # plot(chan, plot_marker)
    
    # 训练模型
    trainer = ModelTrainer()
    
    # 特征选择
    selected_features = trainer.select_features(X, y, list(feature_meta.keys()))
    
    # 参数调优
    best_params = trainer.tune_parameters(X, y)
    trainer.params.update(best_params)
    
    # 交叉验证
    cv_metrics = trainer.cross_validate(X, y)
    print("\n交叉验证结果:")
    for metric, value in cv_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 训练最终模型
    metrics = trainer.train(X, y, selected_features)
    
    # 绘制评估图
    trainer.plot_model_evaluation(X, y)
    
    # 输出训练和验证集评估指标
    print("\n训练集评估指标:")
    for name, value in metrics['train'].items():
        # 添加中文说明
        metric_name = {
            'auc': 'AUC值(auc)',
            'accuracy': '准确率(accuracy)',
            'precision': '精确率(precision)',
            'recall': '召回率(recall)',
            'f1': 'F1值(f1)',
            'fpr': '假正例率(fpr)',
            'mcc': '马修斯相关系数(mcc)'
        }.get(name, name)
        print(f"{metric_name}: {value:.4f}")
        
    print("\n验证集评估指标:")
    for name, value in metrics['val'].items():
        metric_name = {
            'auc': 'AUC值(auc)',
            'accuracy': '准确率(accuracy)',
            'precision': '精确率(precision)',
            'recall': '召回率(recall)',
            'f1': 'F1值(f1)',
            'fpr': '假正例率(fpr)',
            'mcc': '马修斯相关系数(mcc)'
        }.get(name, name)
        print(f"{metric_name}: {value:.4f}")
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    # 绘制特征重要性图
    trainer.plot_feature_importance()
    
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 创建新的模型目录
    model_dir = model_manager.create_model_dir()
    
    # 先保存到当前目录
    if hasattr(trainer.model, '_Booster'):
        trainer.model._Booster.save_model("model.json")
    else:
        trainer.model.save_model("model.json")
        
    # 复制文件到模型目录
    model_manager.save_model(
        model_dir=model_dir,
        model_path="model.json",
        meta_path="feature.meta",
        processor_path="feature_processor.joblib",
        chan_config={
            "trigger_step": True,
            "bi_strict": True,
            "skip_step": 0,
            "divergence_rate": float("inf"),
            "bsp2_follow_1": False,
            "bsp3_follow_1": False,
            "min_zs_cnt": 0,
            "bs1_peak": False,
            "macd_algo": "peak",
            "bs_type": '1,1p',
            "print_warning": True,
            "zs_algo": "normal",
        },
        train_info={
            "code": code,
            "kl_type": str(lv_list[0]),
            "begin_time": begin_time,
            "end_time": end_time,
            "data_src": str(data_src)
        }
    )
    
    print(f"\n模型已保存到: {model_dir}")
    
    # 分析特征
    trainer.analyze_features(X, y, list(feature_meta.keys()))
    
    # 计算预测置信区间
    mean_pred, conf_interval = trainer.predict_with_confidence(X)
    print("\n预测置信区间示例:")
    for i in range(min(5, len(mean_pred))):
        print(f"样本 {i}: {mean_pred[i]:.4f} ± {conf_interval[i]:.4f}")
    
    # # 暂时注释掉SHAP分析，需要安装shap包才能使用
    # trainer.explain_predictions(X, list(feature_meta.keys()))