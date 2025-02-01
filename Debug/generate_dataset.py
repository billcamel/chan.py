import os,sys
cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from models.feature_engine import FeatureEngine
from models.feature_generator import CFeatureGenerator
from models.feature_data_generator import DataGenerator

if __name__ == "__main__":
    # 配置参数
    code = "BTC/USDT"
    begin_time = "2020-01-01"
    end_time = "2022-01-01"
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_5M]

    # 缠论配置
    chan_config = {
        "trigger_step": True,
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": 999999,
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": "1,1p",
        "print_warning": True,
        "zs_algo": "normal",
    }
    # 初始化对象
    chan = CChan(code=code, begin_time=begin_time, end_time=end_time,
                data_src=data_src, lv_list=lv_list, config=CChanConfig(chan_config.copy()),
                autype=AUTYPE.QFQ)
    
    # 特征引擎
    feature_engine = FeatureEngine()
    feature_set = CFeatureGenerator()
    feature_set.add_all_features()
    
    # 生成特征数据
    data_generator = DataGenerator()

    data_path = data_generator.generate_features(chan,chan_config, feature_engine, feature_set) 