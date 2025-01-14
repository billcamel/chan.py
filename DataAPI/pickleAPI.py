from datetime import datetime
import os

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi
import numpy as np
import pandas as pd

def create_item_dict(data, column_name):
    for i in range(len(data)):
        data.iloc[i] = parse_time_column(data.iloc[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else str2float(data.iloc[i])
    return dict(zip(column_name, data))


def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
    if len(inp) == 10:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = minute = 0
    elif len(inp) == 17:
        year = int(inp[:4])
        month = int(inp[4:6])
        day = int(inp[6:8])
        hour = int(inp[8:10])
        minute = int(inp[10:12])
    elif len(inp) == 19:
        year = int(inp[:4])
        month = int(inp[5:7])
        day = int(inp[8:10])
        hour = int(inp[11:13])
        minute = int(inp[14:16])
    else:
        raise Exception(f"unknown time column from csv:{inp}")
    return CTime(year, month, day, hour, minute)


class PICKLE_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(PICKLE_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        
        # 周期到文件夹名称的映射
        period_map = {
            KL_TYPE.K_DAY: "daily",
            KL_TYPE.K_60M: "60min", 
            KL_TYPE.K_30M: "30min",
            KL_TYPE.K_15M: "15min",
            KL_TYPE.K_5M: "5min",
            KL_TYPE.K_1M: "1min"
        }
        
        # 获取对应的文件夹名称
        period = period_map.get(self.k_type)
        if period is None:
            raise CChanException(f"Unsupported k_type: {self.k_type}", ErrCode.PARA_ERROR)
        
        # 根据代码类型和周期生成文件路径
        if self.code.find("/") >= 0 or self.code.find("-") >= 0:  # 加密货币对
            codestr = self.code.replace("/","-") 
            file_path = f"/Users/zhoulj/work/qdata/data/CRYPTO/{period}/{codestr}.pkl"
        elif self.code[:1].isdigit():  # A股
            file_path = f"/Users/zhoulj/work/qdata/data/CN/{period}/{self.code}.pkl"
        else:  # 美股
            file_path = f"/Users/zhoulj/work/qdata/data/US/{period}/{self.code}.pkl"

        if not os.path.exists(file_path):
            raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)
        
        # 读取并处理数据
        df = pd.read_pickle(file_path)
        df.reset_index(inplace=True)
        
        # 定义列名映射关系
        column_mapping = {
            'time': DATA_FIELD.FIELD_TIME,
            'date': DATA_FIELD.FIELD_TIME,
            'datetime': DATA_FIELD.FIELD_TIME,
            'open': DATA_FIELD.FIELD_OPEN,
            'high': DATA_FIELD.FIELD_HIGH, 
            'low': DATA_FIELD.FIELD_LOW,
            'close': DATA_FIELD.FIELD_CLOSE,
            'volume': DATA_FIELD.FIELD_VOLUME,
            'amount': DATA_FIELD.FIELD_TURNOVER,
            'turnover': DATA_FIELD.FIELD_TURNRATE
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 选择需要的列
        df = df[self.columns]
        
        # 将时间列转换为字符串格式
        time_format = '%Y-%m-%d' if self.k_type == KL_TYPE.K_DAY else '%Y-%m-%d %H:%M:%S'
        df[DATA_FIELD.FIELD_TIME] = df[DATA_FIELD.FIELD_TIME].dt.strftime(time_format)

        for line_number, data in df.iterrows():
            if len(data) != len(self.columns):
                raise CChanException(f"file format error: {file_path}", ErrCode.SRC_DATA_FORMAT_ERROR)
            if self.begin_date is not None and data.iloc[self.time_column_idx] < self.begin_date:
                continue
            if self.end_date is not None and data.iloc[self.time_column_idx] > self.end_date:
                continue
            yield CKLine_Unit(create_item_dict(data, self.columns))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
