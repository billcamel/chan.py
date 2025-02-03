import os
import json
import numpy as np
from typing import Tuple
from datetime import datetime
from Chan import CChan
from Common.CEnum import BSP_TYPE
from models.feature_engine import FeatureEngine
from models.feature_generator import CFeatureGenerator
from models.image_feature_engine import KLineImageEngine

class DataGenerator:
    """特征数据生成器"""
    def __init__(self, data_dir: str = "feature_data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
    def generate_features(self, chan: CChan, chan_config: dict, 
                         feature_engine: FeatureEngine, 
                         feature_set: CFeatureGenerator) -> str:
        """生成特征数据并保存
        
        Args:
            chan: 缠论对象
            feature_engine: 特征引擎
            feature_set: 特征生成器
            
        Returns:
            数据保存路径
        """
        # 生成有意义的目录名
        code = chan.code.replace('/', '_')  # 替换币种中的斜杠
        begin_time = chan.begin_time.replace('-', '')
        end_time = chan.end_time.replace('-', '') if chan.end_time else 'now'
        period = str(chan.lv_list[0]).split('.')[-1]  # 获取周期名称
        
        dir_name = f"{code}_{begin_time}_{end_time}_{period}"
        data_path = os.path.join(self.data_dir, dir_name)
        
        # 如果目录已存在，添加时间戳
        if os.path.exists(data_path):
            timestamp = datetime.now().strftime('_%H%M%S')
            data_path = os.path.join(self.data_dir, dir_name + timestamp)
            
        os.makedirs(data_path)
        
        # 创建图像目录
        images_dir = os.path.join(data_path, "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            
        # 创建图像生成器
        image_engine = KLineImageEngine(
            window_size=60,
            image_width=512,
            image_height=512,
            output_dir=images_dir
        )
        
        print("\n生成特征数据:")
        print(f"交易对: {chan.code}")
        print(f"开始时间: {chan.begin_time}")
        print(f"结束时间: {chan.end_time or '当前'}")
        print(f"K线周期: {chan.lv_list[0]}")
        print(f"数据源: {chan.data_src}")
        
        # 收集特征数据
        bsp_dict = {}
        kline_data = []
        
        print("开始生成特征数据...")
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
                bsp_dict[last_bsp.klu.idx] = {
                    "feature": last_bsp.features,
                    "is_buy": last_bsp.is_buy,
                    "open_time": last_klu.time,
                }
                market_features = {
                    # **feature_engine.get_features(kline_data, len(kline_data)-1, chan_snapshot),
                    # **feature_set.generate_features(chan_snapshot)
                }
                
                # 生成图像特征
                image_feature, image_name = image_engine.generate_feature(kline_data, len(kline_data)-1)
                if image_feature is not None and image_name is not None:
                    market_features['kline_image'] = len(kline_data)-1  # 保存图片的kline id
                
                bsp_dict[last_bsp.klu.idx]['feature'].add_feat(market_features)
        
        # 生成标签数据
        bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp(0) 
                      if BSP_TYPE.T1 in bsp.type or BSP_TYPE.T1P in bsp.type]
        plot_marker, feature_meta, X, y = feature_engine.save_features(bsp_dict, bsp_academy)
        
        # 保存数据
        np.save(os.path.join(data_path, "features.npy"), X)
        np.save(os.path.join(data_path, "labels.npy"), y)
        with open(os.path.join(data_path, "feature_meta.json"), "w") as f:
            json.dump(feature_meta, f, indent=2)
        # 保存数据信息
        data_info = {
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'code': chan.code,
            'begin_time': chan.begin_time,
            'end_time': chan.end_time if chan.end_time else 'now',
            'period': str(chan.lv_list[0]),
            'feature_count': X.shape[1],
            'sample_count': len(y),
            'positive_count': int(sum(y)),
            'positive_ratio': float(sum(y)/len(y)),
            'feature_names': list(feature_meta.keys()),
            'chan_config': chan_config,  # 添加缠论配置
            'feature_engine_config': {  # 添加特征引擎配置
                'enabled_types': [t.name for t in feature_engine.enabled_types],
                'normalize_window': feature_engine.normalize_window
            },
            'image_features': {
                'window_size': image_engine.window_size,
                'image_width': image_engine.image_width,
                'image_height': image_engine.image_height,
                'image_count': len(os.listdir(images_dir))
            }
        }
        with open(os.path.join(data_path, "data_info.json"), "w") as f:
            json.dump(data_info, f, indent=2)
            
        print(f"特征数据已保存到: {data_path}")
        print(f"K线图像已保存到: {images_dir}")
        print(f"生成了 {data_info['image_features']['image_count']} 张K线图像")
        
        return data_path 
