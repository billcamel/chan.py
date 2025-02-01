import json
from typing import Dict, TypedDict
from datetime import datetime
import numpy as np
import os,sys
cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from ChanModel.Features import CFeatures
from Common.CTime import CTime
from Plot.PlotlyDriver import CPlotlyDriver
from models.model_manager import ModelManager
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

def train_model(data_path: str, time_limit: int = 3600):
    """训练模型
    
    Args:
        data_path: 特征数据路径
        time_limit: 训练时间限制(秒)
    """
    # 加载数据
    with open(os.path.join(data_path, "data_info.json"), "r") as f:
        data_info = json.load(f)
        
    print("\n数据信息:")
    print(f"币种: {data_info['code']}")
    print(f"时间范围: {data_info['begin_time']} 到 {data_info['end_time']}")
    print(f"周期: {data_info['period']}")
    print(f"样本数: {data_info['sample_count']}")
    print(f"正样本比例: {data_info['positive_ratio']:.2%}")
    print(f"特征数: {data_info['feature_count']}")
    
    print("\n缠论配置:")
    for key, value in data_info['chan_config'].items():
        print(f"{key}: {value}")
        
    print("\n特征引擎配置:")
    print(f"启用的特征类型: {', '.join(data_info['feature_engine_config']['enabled_types'])}")
    print(f"归一化窗口: {data_info['feature_engine_config']['normalize_window']}")
    
    X = np.load(os.path.join(data_path, "features.npy"))
    y = np.load(os.path.join(data_path, "labels.npy"))
    with open(os.path.join(data_path, "feature_meta.json"), "r") as f:
        feature_meta = json.load(f)
        
    # 特征处理
    processor = FeatureProcessor()
    processor.fit(X, list(feature_meta.keys()))
    X_processed = processor.transform(X)
    
    # 训练模型
    model_manager = ModelManager()
    model_dir = os.path.join(model_manager.base_dir, 
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    trainer = AutoTrainer(time_limit=time_limit)
    trainer.train(X_processed, y, list(feature_meta.keys()), model_dir)
    
    # 保存模型
    insights = trainer.get_model_insights()
    train_info = {
        'train_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'time_limit': time_limit,
        'data_info': data_info,
        'performance': insights['leaderboard'],
        'fit_summary': insights['fit_summary']
    }
    
    model_manager.save_model(
        model_dir=model_dir,
        feature_meta=feature_meta,
        processor=processor,
        train_info=train_info,
    )
    
    print(f"模型已保存到: {model_dir}")
    return model_dir

if __name__ == "__main__":
    # 获取数据目录
    data_dir = "feature_data"
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        sys.exit(1)
        
    # 过滤掉隐藏文件和非目录
    data_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) and 
                 not d.startswith('.')]
    
    if not data_dirs:
        print(f"错误: 在 {data_dir} 中未找到有效的数据目录")
        sys.exit(1)
        
    # 打印可用的数据目录
    print("\n可用的数据目录:")
    for i, d in enumerate(sorted(data_dirs, 
                               key=lambda x: os.path.getctime(os.path.join(data_dir, x)), 
                               reverse=True)):
        ctime = datetime.fromtimestamp(os.path.getctime(os.path.join(data_dir, d)))
        print(f"{i+1}. {d} (创建于 {ctime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 检查指定的目录
    if data_path:
        # 如果指定的是完整路径
        if os.path.exists(data_path):
            print(f"\n使用指定数据目录: {data_path}")
        else:
            # 如果只指定了目录名，检查是否在data_dir中
            dir_name = os.path.basename(data_path)
            full_path = os.path.join(data_dir, dir_name)
            if os.path.exists(full_path):
                data_path = full_path
                print(f"\n使用指定数据目录: {data_path}")
            else:
                print(f"警告: 指定的数据目录 {data_path} 不存在，将使用最新的数据目录")
                data_path = os.path.join(data_dir, 
                                       max(data_dirs, 
                                           key=lambda x: os.path.getctime(os.path.join(data_dir, x))))
                print(f"使用最新数据目录: {data_path}")
    else:
        # 使用最新的目录
        data_path = os.path.join(data_dir, 
                                max(data_dirs, 
                                    key=lambda x: os.path.getctime(os.path.join(data_dir, x))))
        print(f"\n使用最新数据目录: {data_path}")
    
    train_model(data_path)