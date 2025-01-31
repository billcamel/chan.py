import json
from platform import processor
from typing import Dict, TypedDict
from datetime import datetime

import numpy as np
import xgboost as xgb
import os,sys
import matplotlib.pyplot as plt


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
from models.feature_generator import CFeatureGenerator
from models.auto_trainer import AutoTrainer
from models.feature_processor import FeatureProcessor


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
    end_time = "2022-01-01"
    # end_time = "2024-01-01"
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_60M]

    my_config = {
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": 999999999,  # 使用一个大数字代替 inf
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,1p',
        "print_warning": True,
        "zs_algo": "normal",
    }
    config = CChanConfig(my_config)

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
    # 初始化特征生成器实例
    feature_set = CFeatureGenerator()
    # 一键添加所有特征
    feature_set.add_all_features()

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        kline_data.append(last_klu)
        bsp_list = chan_snapshot.get_bsp(0)
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
            market_features = {
                **feature_engine.get_features(kline_data, len(kline_data)-1, chan_snapshot),
                **feature_set.generate_features(chan_snapshot)
            }
            # if len(kline_data) > 300:
            #     print(market_features)
            #     break

            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(market_features)

    print(market_features)
    # print(market_features.keys())

    # 生成特征数据
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0) 
                   if BSP_TYPE.T1 in bsp.type or BSP_TYPE.T1P in bsp.type]  # 只考虑一类买卖点
    plot_marker, feature_meta, X, y = feature_engine.save_features(bsp_dict, bsp_academy)

    print("市场特征中的特征:", market_features.keys())
    print("特征元数据中的特征:", feature_meta.keys())
    print("差异特征:", feature_meta.keys() - market_features.keys())
    
    # 初始化特征处理器
    processor = FeatureProcessor()
    processor.fit(X, list(feature_meta.keys()))
    X_processed = processor.transform(X)
    # X_processed = X
    
    # 打印标签分布
    print("\n标签分布:")
    print(f"特征维度: {X_processed.shape}")
    print(f"总样本数: {len(y)}")
    print(f"正样本数: {sum(y)}")
    print(f"负样本数: {len(y) - sum(y)}")
    print(f"正样本比例: {sum(y)/len(y):.2%}")
    
    # 检查是否有无效特征
    invalid_features = []
    for feat_name, feat_idx in feature_meta.items():
        feat_values = X_processed[:, feat_idx]
        # 检查特征是否全为0、NaN或无穷大
        if np.all(feat_values == 0):
            invalid_features.append((feat_name, "全为0"))
        elif np.all(np.isnan(feat_values)):
            invalid_features.append((feat_name, "全为NaN"))
        elif np.all(np.isinf(feat_values)):
            invalid_features.append((feat_name, "全为无穷大"))
        elif np.all(np.abs(feat_values) > 1e10):
            invalid_features.append((feat_name, "数值过大"))
            
    if invalid_features:
        print("\n发现无效特征:")
        for feat_name, reason in invalid_features:
            print(f"- {feat_name}: {reason}")
    # 画图检查label是否正确
    # plot(chan, plot_marker)
    # 保存模型
    model_manager = ModelManager()
    model_dir = os.path.join(model_manager.base_dir, 
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    # 初始化训练器
    time_limit = 100  # 1小时训练时间限制
    trainer = AutoTrainer(time_limit=time_limit)
    trainer.train(X_processed, y, list(feature_meta.keys()),model_dir)  # 使用处理后的特征
    
    print("训练完成, 开始获取模型分析")
    # 获取模型分析
    insights = trainer.get_model_insights()
    print("模型分析完成")
    
    # 创建训练信息
    train_info = {
        'train_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'time_limit': time_limit,
        'data_info': {
            'code': code,
            'begin_time': begin_time,
            'end_time': end_time,
            'kl_type': str(lv_list[0])
        },
        'chan_config': my_config,  # 直接使用原始配置字典
        'performance': insights['leaderboard'],
        'fit_summary': insights['fit_summary']
    }
    
    
    model_manager.save_model(
        model_dir=model_dir,
        feature_meta=feature_meta,
        processor=processor,  # 传入FeatureProcessor实例
        train_info=train_info,
    )
    
    print(f"\n模型已保存到: {model_dir}")