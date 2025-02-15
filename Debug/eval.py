import json
from typing import Dict, TypedDict
import os,sys
import numpy as np
import xgboost as xgb

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime

# 修改导入路径
from models.trade_analyzer import TradeAnalyzer
from models import FeatureProcessor
from models.trainer import ModelEvaluator
from models.feature_engine import FeatureEngine
from models.model_manager import ModelManager

from Debug.models.trade_analyzer import TradeAnalyzer


def predict_bsp(model: xgb.Booster, features: Dict[str, float], feature_meta: Dict[str, int], processor: FeatureProcessor) -> float:
    """预测买卖点的概率
    
    Args:
        model: 训练好的XGBoost模型
        features: 特征字典
        feature_meta: 特征映射信息
        processor: 特征处理器
        
    Returns:
        float: 预测的概率值
    """
    # 特征归一化
    scaled_features = processor.transform_dict(features)
    
    # 构建特征向量
    X = np.zeros(len(feature_meta))
    for feat_name, value in scaled_features.items():
        if feat_name in feature_meta:
            X[feature_meta[feat_name]] = value
            
    # 转换为DMatrix
    dtest = xgb.DMatrix(X.reshape(1, -1))
    
    # 预测
    return model.predict(dtest)[0]

if __name__ == "__main__":
    """
    本示例展示了如何将策略生成的买卖点与离线模型集成，以进行实盘交易
    """
    code = "BTC/USDT"
    begin_time = "2024-01-01"
    end_time = None
    # end_time = "2024-01-01"
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_15M]

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

    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 获取最新模型目录
    model_dir = model_manager.get_latest_model()
    if model_dir is None:
        raise ValueError("未找到可用的模型")
    
    # 加载模型文件
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.json"))
    
    # 加载特征映射和处理器
    with open(os.path.join(model_dir, "feature.meta"), "r") as f:
        feature_meta = json.load(f)
    processor = FeatureProcessor.load(os.path.join(model_dir, "feature_processor.joblib"))
    
    # 加载训练信息
    with open(os.path.join(model_dir, "train_info.json"), "r") as f:
        train_info = json.load(f)
        
    print(f"\n加载模型: {model_dir}")
    print("训练信息:")
    print(f"品种: {train_info['code']}")
    print(f"周期: {train_info['kl_type']}")
    print(f"训练数据: {train_info['begin_time']} - {train_info['end_time']}")
    
    # 初始化特征引擎
    feature_engine = FeatureEngine()

    treated_bsp_idx = set()
    kline_data = []  # 存储K线数据用于后续分析
    
    # 记录交易结果
    trades = []
    
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        kline_data.append(last_klu)
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx in treated_bsp_idx or cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        # 使用特征引擎计算特征
        features = feature_engine.get_features(
            kline_data, 
            len(kline_data)-1,
            chan_snapshot  # 传入缠论快照
        )
        
        # 预测买卖点的概率
        prob = predict_bsp(model, features, feature_meta, processor)
        
        # 记录交易信息
        trade_info = {
            'time': last_bsp.klu.time.to_str(),
            'is_buy': last_bsp.is_buy,
            'prob': prob,
            'price': last_klu.close,
            'idx': last_bsp.klu.idx
        }
        trades.append(trade_info)
        
        # 打印交易信息
        trade_type = "买入" if last_bsp.is_buy else "卖出"
        print(f"{trade_info['time']}: {trade_type} 信号, 预测概率={prob:.2%}, 价格={trade_info['price']:.2f}")
        
        treated_bsp_idx.add(last_bsp.klu.idx)
    
    # 获取实际买卖点
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
    
    # 评估结果
    evaluator = ModelEvaluator("model.json", threshold=0.6)
    stats = evaluator.evaluate_trades(trades, bsp_academy)
    
    # 输出评估结果
    print("\n交易统计:")
    print(f"总信号数: {stats['total_signals']}")
    print(f"买入信号: {stats['buy_signals']}")
    print(f"卖出信号: {stats['sell_signals']}")
    
    print(f"\n高概率信号 (>{evaluator.threshold:.0%}):")
    print(f"总数: {stats['high_prob_signals']}")
    print(f"买入: {stats['high_prob_buy']}")
    print(f"卖出: {stats['high_prob_sell']}")
    
    print("\n预测准确性分析:")
    print(f"实际买卖点数量: {stats['actual_points']}")
    print(f"高概率信号准确率: {stats['precision']:.2%}")
    print(f"实际信号召回率: {stats['recall']:.2%}")
    print(f"F1分数: {stats['f1']:.2%}")
    
    print("\n买卖点分类准确率:")
    print(f"买入点准确率: {stats['buy_precision']:.2%}")
    print(f"卖出点准确率: {stats['sell_precision']:.2%}")
    
    # 按时间顺序输出详细的高概率信号分析
    print("\n高概率信号详细分析:")
    print("时间\t\t类型\t概率\t是否正确\tK线索引")
    print("-" * 60)
    high_prob_trades = [t for t in trades if t['prob'] > evaluator.threshold]
    for trade in high_prob_trades:
        trade_type = "买入" if trade['is_buy'] else "卖出"
        is_correct = trade['idx'] in bsp_academy
        correct_mark = "√" if is_correct else "×"
        print(f"{trade['time']}\t{trade_type}\t{trade['prob']:.2%}\t{correct_mark}\t{trade['idx']}")

    # 分析交易曲线
    analyzer = TradeAnalyzer(initial_capital=100000)
    stats = analyzer.analyze_trades(trades, threshold=evaluator.threshold)
    analyzer.plot_equity_curve()
    
    # 打印交易统计
    print("\n交易统计:")
    print(f"初始资金: {stats['initial_capital']:,.0f}")
    print(f"最终资金: {stats['final_capital']:,.0f}")
    print(f"总收益率: {stats['total_return']:.2f}%")
    print(f"胜率: {stats['win_rate']:.2f}%")
    print(f"最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"盈亏比: {stats['profit_ratio']:.2f}")
    print(f"总交易次数: {stats['trade_count']}")
    print(f"买入次数: {stats['buy_count']}")
    print(f"卖出次数: {stats['sell_count']}")


