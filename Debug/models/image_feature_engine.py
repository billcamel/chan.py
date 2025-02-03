import numpy as np
import pandas as pd
from typing import List, Any, Optional, Tuple
from PIL import Image, ImageDraw
import os, sys

cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE

class KLineImageEngine:
    """K线图像特征引擎"""
    def __init__(self, 
                 window_size: int = 30,
                 image_width: int = 512,
                 image_height: int = 512,
                 output_dir: Optional[str] = None):
        """初始化
        
        Args:
            window_size: 每张图包含的K线数量
            image_width: 图像宽度
            image_height: 图像高度
            output_dir: 图像保存目录(用于调试)
        """
        self.window_size = window_size
        self.image_width = image_width
        self.image_height = image_height
        self.output_dir = output_dir
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def _normalize_price_data(self, prices: np.ndarray) -> np.ndarray:
        """归一化价格数据到0-1范围"""
        min_price = np.min(prices)
        max_price = np.max(prices)
        if max_price == min_price:
            return np.full_like(prices, 0.5)
        return (prices - min_price) / (max_price - min_price)
    
    def _draw_kline(self, 
                    image: Image.Image,
                    kline_data: List[Any],
                    start_idx: int) -> None:
        """在图像上绘制K线
        
        Args:
            image: PIL图像对象
            kline_data: K线数据
            start_idx: 起始K线索引
        """
        draw = ImageDraw.Draw(image)
        
        # 提取价格数据 - 只使用开盘和收盘价
        prices = np.array([
            [k.open, k.close] 
            for k in kline_data[start_idx:start_idx+self.window_size]
        ])
        
        # 归一化价格
        norm_prices = self._normalize_price_data(prices.flatten()).reshape(prices.shape)
        
        # 计算每根K线的宽度和位置
        bar_width = self.image_width // self.window_size
        
        for i in range(self.window_size):
            x = i * bar_width
            # 在PIL图像中,坐标系原点(0,0)位于左上角,y轴向下为正
            # 而在金融图表中,价格向上增长,所以需要用1减去归一化的价格来反转y轴
            open_y = int((1 - norm_prices[i, 0]) * self.image_height)
            close_y = int((1 - norm_prices[i, 1]) * self.image_height)
            
            # 绘制实体
            if close_y < open_y:  # 阴线(收盘价低于开盘价)
                draw.rectangle([(x, close_y), (x + bar_width, open_y)],
                             outline='white', fill='green')
            else:  # 阳线(收盘价高于等于开盘价) 
                draw.rectangle([(x, open_y), (x + bar_width, close_y)],
                             outline='white', fill='red')
    def generate_feature(self, 
                        kline_data: List[Any], 
                        idx: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """生成图像特征
        
        Args:
            kline_data: K线数据列表
            idx: 当前K线索引
            
        Returns:
            Tuple[特征数组, 图片名称]，如果数据不足返回 (None, None)
        """
        if idx < self.window_size:
            return None, None
            
        # 创建空白图像
        image = Image.new('RGB', (self.image_width, self.image_height), 'white')
        
        # 绘制K线
        self._draw_kline(image, kline_data, idx - self.window_size)
        
        # 生成图片名称
        time_str = kline_data[idx].time.to_str()
        time_str = time_str.replace('-','').replace(' ','_').replace(':','')
        time_str = time_str.replace('/','')
        image_name = f"{idx}.png"
        
        # 保存调试图像
        if self.output_dir:
            image.save(os.path.join(self.output_dir, image_name))
        
        # 转换为numpy数组并归一化
        feature = np.array(image) / 255.0
        return feature, image_name
        
    def get_feature_shape(self) -> Tuple[int, int]:
        """获取特征形状"""
        return (self.image_height, self.image_width)

    def generate_dataset(self, kline_data: List[Any]) -> int:
        """生成K线图像数据集
        
        Args:
            kline_data: K线数据列表
            
        Returns:
            生成的图像数量
        """
        image_count = 0
        
        # 遍历每个时间点
        for i in range(len(kline_data)):
            # 生成图像特征
            feature, image_name = self.generate_feature(kline_data, i)
            if feature is not None:
                image_count += 1
                if image_count % 100 == 0:
                    print(f"已生成 {image_count} 张图像")
        
        return image_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='生成K线图像数据集')
    parser.add_argument('--code', type=str, default='BTC/USDT',
                      help='交易对代码')
    parser.add_argument('--begin', type=str, default='2024-01-01',
                      help='开始时间 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-05-01',
                      help='结束时间 (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, default='K_DAY',
                      choices=['K_1M', 'K_5M', 'K_15M', 'K_30M', 'K_60M', 'K_DAY'],
                      help='K线周期')
    parser.add_argument('--window', type=int, default=60,
                      help='窗口大小')
    parser.add_argument('--output', type=str, default='kline_images',
                      help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化缠论对象获取数据
    chan = CChan(
        code=args.code,
        begin_time=args.begin,
        end_time=args.end,
        data_src=DATA_SRC.PICKLE,
        lv_list=[getattr(KL_TYPE, args.period)],
        config=CChanConfig({
            "trigger_step": True,
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
        }),
        autype=AUTYPE.QFQ
    )
    
    # 收集K线数据
    kline_data = []
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        kline_data.append(last_klu)
    
    print(f"\n开始生成K线图像:")
    print(f"交易对: {args.code}")
    print(f"时间范围: {args.begin} 到 {args.end}")
    print(f"K线周期: {args.period}")
    print(f"窗口大小: {args.window}")
    print(f"K线数量: {len(kline_data)}")
    
    # 创建图像生成器并生成数据集
    engine = KLineImageEngine(
        window_size=args.window,
        image_width=512,
        image_height=512,
        output_dir=args.output
    )
    
    image_count = engine.generate_dataset(kline_data)
    
    print(f"\n图像生成完成:")
    print(f"总共生成了 {image_count} 张图像")
    print(f"图像保存在: {args.output}") 
