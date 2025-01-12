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
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from features import get_market_features, save_features, safe_div


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
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")


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
    code = "sz.000001"
    begin_time = "2010-01-01"
    end_time = "2020-01-01"
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY]

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
        "bs_type": '1,2,3a,1p,2s,3b',
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

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time,
            }
            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(get_market_features(kline_data, len(kline_data)-1))  # 开仓K线特征
            # print(last_bsp.klu.time, last_bsp.is_buy)

    # 生成特征数据
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    plot_marker, feature_meta, X, y = save_features(bsp_dict, bsp_academy)
    
    # 画图检查label是否正确
    plot(chan, plot_marker)
    
    # 训练模型参数调整
    param = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist',
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'nthread': 4,
        'seed': 42
    }
    
    # 加载数据并训练
    try:
        # 直接使用numpy数组创建DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # 训练模型
        evals_result = {}
        bst = xgb.train(
            param,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train")],
            evals_result=evals_result,
            verbose_eval=10,
            early_stopping_rounds=20
        )
        
        # 保存模型
        bst.save_model("model.json")
        
        # 输出特征重要性
        importance = bst.get_score(importance_type='gain')
        print("\nFeature Importance:")
        for fname, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            feat_idx = int(fname.replace('f', ''))
            for name, idx in feature_meta.items():
                if idx == feat_idx:
                    print(f"{name}: {imp:.2f}")
        
        # 预测并计算评估指标
        predictions = bst.predict(dtrain)
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(y, predictions)
        acc = accuracy_score(y, predictions > 0.5)
        print(f"\nAUC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        
        # 保存预测结果
        np.save("predictions.npy", predictions)
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        raise