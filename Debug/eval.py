import json
from typing import Dict, List, TypedDict
import os,sys
import numpy as np
import xgboost as xgb
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef


cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE,BSP_TYPE
from Common.CTime import CTime

# 修改导入路径
from models.trade_analyzer import TradeAnalyzer
from models import FeatureProcessor
from models.trainer import ModelEvaluator
from models.feature_engine import FeatureEngine
from models.model_manager import ModelManager
from models.auto_trainer import AutoTrainer

from Debug.models.trade_analyzer import TradeAnalyzer
from Debug.models.feature_generator import CFeatureGenerator


def predict_bsp(model, features: Dict, feature_meta: Dict, processor: FeatureProcessor) -> float:
    """预测买卖点的概率"""
    # 将特征字典转换为有序的特征列表
    feature_list = []
    missing_features = []
    for feat_name in feature_meta.keys():
        if feat_name not in features:
            missing_features.append(feat_name)
            feature_list.append(0)  # 使用默认值
        else:
            feature_list.append(features[feat_name])
    
    # 只在第一次遇到缺失特征时打印警告
    if missing_features and not hasattr(predict_bsp, 'warned_features'):
        print(f"警告: 以下特征在预测时不存在: {missing_features}")
        print("这可能会影响模型的预测效果")
        predict_bsp.warned_features = True
            
    # 转换为numpy数组并处理
    X = np.array([feature_list])
    
    # 使用特征处理器进行处理
    X_processed = processor.transform(X)
    
    # 转换为DataFrame
    df = pd.DataFrame(X_processed, columns=list(feature_meta.keys()))
    # print(df.head())
    # 使用模型预测
    y_pred_proba = model.predict_proba(df)
    # print(y_pred_proba)
    
    # 检查预测结果的格式并返回正类的概率
    if isinstance(y_pred_proba, pd.DataFrame):
        return y_pred_proba.iloc[0, 1]
    elif isinstance(y_pred_proba, np.ndarray):
        return y_pred_proba[0, 1]
    else:
        raise ValueError(f"Unexpected prediction format: {type(y_pred_proba)}")

def evaluate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
    """计算各项评估指标"""
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'fpr': 1 - recall_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def load_and_evaluate(model_dir: str, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
    """加载模型并评估性能"""
    # 加载模型和相关文件
    model_manager = ModelManager()
    model, feature_meta, processor = model_manager.load_model(model_dir)
    
    # 加载训练信息
    with open(os.path.join(model_dir, "train_info.json"), "r") as f:
        train_info = json.load(f)
    
    # 处理特征
    X_processed = processor.transform(X)
    
    # 预测
    test_data = pd.DataFrame(X_processed, columns=feature_names)
    y_pred_proba = model.predict_proba(test_data)
    
    # 计算评估指标
    metrics = evaluate_metrics(y, y_pred_proba)
    
    # 打印评估结果
    print("\n模型评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}值({metric}): {value:.4f}")
        
    # 打印模型信息
    print("\n模型信息:")
    print(model.get_model_best())  # 获取最佳模型信息
    print(model.leaderboard())     # 显示所有模型的性能排名
    
    # 打印数据信息
    print("\n数据信息:")
    print(f"代码: {train_info['data_info']['code']}")
    print(f"开始时间: {train_info['data_info']['begin_time']}")
    print(f"结束时间: {train_info['data_info']['end_time']}")
    print(f"K线类型: {train_info['data_info']['kl_type']}")
    
    return {
        'metrics': metrics,
        'train_info': train_info
    }

if __name__ == "__main__":
    """
    本示例展示了如何将策略生成的买卖点与离线模型集成，以进行实盘交易
    """
    code = "BTC/USDT"
    begin_time = "2024-12-01"
    end_time = None
    # end_time = "2022-01-01"
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_5M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": 999999999,
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
    
    # 加载模型和相关文件
    model_dir = model_manager.get_latest_model_dir()
    if not model_dir:
        raise ValueError("未找到可用的模型")
        
    model, feature_meta, processor = model_manager.load_model(model_dir)
    
    # 加载训练信息
    with open(os.path.join(model_dir, "train_info.json"), "r") as f:
        train_info = json.load(f)
    print(f"使用模型: {model_dir}")
    print(f"训练时间: {train_info['train_time']}")
    print(f"训练数据: {train_info['data_info']}")
    print(f"decision_threshold: {model.decision_threshold}")
    
    # 初始化特征引擎
    feature_engine = FeatureEngine()

    treated_bsp_idx = set()
    kline_data = []  # 存储K线数据用于后续分析
    
    # 记录交易结果
    trades = []
    # 初始化特征生成器实例
    feature_set = CFeatureGenerator()
    # 一键添加所有特征
    feature_set.add_all_features()
    
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
        if last_bsp.klu.idx not in treated_bsp_idx and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
            features = last_bsp.features
            # 使用特征引擎计算特征
            market_features = {
                **feature_engine.get_features(kline_data, len(kline_data)-1, chan_snapshot),
                **feature_set.generate_features(chan_snapshot)
            }
            features.add_feat(market_features)
            
            # 预测买卖点的概率
            prob = predict_bsp(model, features.to_dict(), feature_meta, processor)
            
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
    
    # 创建评估器并寻找最佳阈值
    evaluator = ModelEvaluator()
    best_metrics = evaluator.find_best_threshold(trades, bsp_academy)
    
    # 使用最佳阈值的评估结果已经在evaluator中
    stats = evaluator.evaluate_trades(trades, bsp_academy)
    
    # 输出评估结果
    print("\n使用最佳阈值的交易统计:")
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

    # # 获取最新的模型目录
    # model_manager = ModelManager()
    # model_dir = model_manager.get_latest_model_dir()
    
    # if model_dir is None:
    #     print("未找到模型目录")
    #     exit(1)
        
    # print(f"加载模型目录: {model_dir}")
    
    # try:
    #     # 生成测试数据
    #     # TODO: 这里需要替换为实际的测试数据生成逻辑
    #     X_test = np.array([features for features in feature_engine.get_features(kline_data, len(kline_data)-1, chan_snapshot) for _ in range(100)])
    #     y_test = np.array([1 if bsp.is_buy else 0 for bsp in chan.get_bsp()])
    #     feature_names = [f"feature_{i}" for i in range(len(features))]
        
    #     # 评估模型
    #     results = load_and_evaluate(model_dir, X_test, y_test, feature_names)
        
    #     # 保存评估结果
    #     eval_dir = os.path.join(model_dir, "eval_results")
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
            
    #     eval_file = os.path.join(eval_dir, 
    #                             f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    #     with open(eval_file, "w") as f:
    #         json.dump(results, f, indent=2)
            
    #     print(f"\n评估结果已保存到: {eval_file}")
        
    # except Exception as e:
    #     print(f"评估过程出错: {str(e)}")
    #     # 打印详细的错误堆栈
    #     import traceback
    #     traceback.print_exc()


