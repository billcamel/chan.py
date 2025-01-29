import os,sys
import json
import numpy as np
import xgboost as xgb
from typing import Dict
cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from DataAPI.BaoStockAPI import CBaoStock
from DataAPI.pickleAPI import PICKLE_API
from models.trade_analyzer import TradeAnalyzer
from models import FeatureProcessor
from models.feature_engine import FeatureEngine
from models.model_manager import ModelManager


def predict_bsp(model: xgb.Booster, features: Dict[str, float], feature_meta: Dict[str, int], processor: FeatureProcessor) -> float:
    """预测买卖点的概率"""
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
    一个极其弱智的策略，只交易一类买卖点，底分型形成后就开仓，直到一类卖点顶分型形成后平仓
    只用做展示如何自己实现策略，做回测用~
    相比于strategy_demo.py，本代码演示如何从CChan外部喂K线来触发内部缠论计算
    """
    code = "BTC/USDT"
    begin_time = "2024-01-01"
    end_time = None
    # end_time = "2024-01-01"
    data_src_type = DATA_SRC.PICKLE
    kl_type = KL_TYPE.K_5M
    lv_list = [kl_type]

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
        begin_time=begin_time,  # 已经没啥用了这一行
        end_time=end_time,  # 已经没啥用了这一行
        data_src=data_src_type,  # 已经没啥用了这一行
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,  # 已经没啥用了这一行
    )
    PICKLE_API.do_init()
    data_src = PICKLE_API(code, k_type=kl_type, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)  # 初始化数据源类

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

    # 初始化交易分析器
    analyzer = TradeAnalyzer(initial_capital=10000)
    trades = []  # 存储交易记录
    kline_data = []  # 存储K线数据用于特征计算

    is_hold = False
    last_buy_price = None
    for klu in data_src.get_kl_data():  # 获取单根K线
        chan.trigger_load({kl_type: [klu]})  # 喂给CChan新增k线
        kline_data.append(klu)  # 保存K线数据
        
        bsp_list = chan.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1] # 最后一个买卖点
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type:
            continue

        cur_lv_chan = chan[0]

        if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx: # 如果买卖点不是当前缠论的倒数第二个k线，则跳过
            continue
            
        # 计算特征并预测概率
        features = feature_engine.get_features(
            kline_data, 
            len(kline_data)-1,
            chan  # 传入缠论快照
        )
        prob = predict_bsp(model, features, feature_meta, processor)
        
        # 顶底分型时进行交易
        if cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy and not is_hold and prob > 0.5:  # 只在高概率时买入
            last_buy_price = cur_lv_chan[-1][-1].close
            print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}, prob = {prob:.2%}')
            is_hold = True
            trades.append({
                'time': cur_lv_chan[-1][-1].time,
                'is_buy': True,
                'price': last_buy_price,
                'prob': prob
            })
        elif cur_lv_chan[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy and is_hold and prob > 0.5:  # 只在高概率时卖出
            sell_price = cur_lv_chan[-1][-1].close
            profit_rate = (sell_price-last_buy_price)/last_buy_price*100
            print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, prob = {prob:.2%}, profit rate = {profit_rate:.2f}%')
            is_hold = False
            trades.append({
                'time': cur_lv_chan[-1][-1].time,
                'is_buy': False,
                'price': sell_price,
                'prob': prob
            })

    PICKLE_API.do_close()

    # 分析交易结果
    stats = analyzer.analyze_trades(trades, threshold=0.5)
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
