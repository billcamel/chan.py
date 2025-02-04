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
    
    def _calculate_ma(self, kline_data: List[Any], start_idx: int, period: int) -> np.ndarray:
        """计算移动平均线
        
        Args:
            kline_data: K线数据列表
            start_idx: 起始K线索引
            period: 均线周期
            
        Returns:
            移动平均线数据
        """
        if start_idx < period:
            return None
        
        prices = np.array([
            k.close for k in kline_data[start_idx-period:start_idx+self.window_size]
        ])
        
        ma = np.convolve(prices, np.ones(period)/period, mode='valid')
        return ma[-self.window_size:]

    def _calculate_macd(self, kline_data: List[Any], start_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算MACD指标
        
        Args:
            kline_data: K线数据列表
            start_idx: 起始K线索引
            
        Returns:
            (DIF, DEA, MACD)元组
        """
        # 获取收盘价
        closes = np.array([k.close for k in kline_data[:start_idx+self.window_size]])
        
        # 计算EMA
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
        
        # 计算DIF
        dif = ema12 - ema26
        
        # 计算DEA
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        
        # 计算MACD
        macd = 2 * (dif - dea)
        
        # 返回窗口内的数据
        return (dif[-self.window_size:], 
                dea[-self.window_size:], 
                macd[-self.window_size:])

    def _draw_kline(self, 
                    image: Image.Image,
                    kline_data: List[Any],
                    start_idx: int) -> None:
        """在图像上绘制K线"""
        draw = ImageDraw.Draw(image)
        
        # 提取价格数据和成交量数据
        prices = np.array([
            [k.open, k.close] 
            for k in kline_data[start_idx:start_idx+self.window_size]
        ])
        
        volumes = np.array([
            k.trade_info.metric.get('volume', 0) 
            for k in kline_data[start_idx:start_idx+self.window_size]
        ])
        
        # 计算MA10和MA20
        ma10 = self._calculate_ma(kline_data, start_idx, 10)
        ma20 = self._calculate_ma(kline_data, start_idx, 20)
        
        # 合并所有价格数据进行归一化
        all_prices = prices.flatten()
        if ma10 is not None:
            all_prices = np.concatenate([all_prices, ma10])
        if ma20 is not None:
            all_prices = np.concatenate([all_prices, ma20])
        
        # 归一化价格和成交量
        norm_prices = self._normalize_price_data(all_prices)
        norm_volumes = self._normalize_price_data(volumes)
        
        # 分离归一化后的价格数据
        norm_kline = norm_prices[:len(prices.flatten())].reshape(prices.shape)
        norm_ma10 = norm_prices[len(prices.flatten()):len(prices.flatten())+self.window_size] if ma10 is not None else None
        norm_ma20 = norm_prices[-self.window_size:] if ma20 is not None else None
        
        # 计算每根K线的宽度和位置
        bar_width = self.image_width // self.window_size
        
        # 分配显示区域
        price_height = int(self.image_height * 0.6)  # K线占60%
        volume_height = int(self.image_height * 0.15)  # 成交量占15%
        macd_height = int(self.image_height * 0.15)  # MACD占15%
        
        # 计算各区域的起始和结束位置
        price_start = 0
        price_end = price_height
        
        volume_start = price_end + 10  # 留出10像素间隔
        volume_end = volume_start + volume_height
        
        macd_start = volume_end + 10  # 留出10像素间隔
        macd_end = macd_start + macd_height
        
        # 绘制K线实体
        for i in range(self.window_size):
            x = i * bar_width
            
            # K线部分
            open_y = int((1 - norm_kline[i, 0]) * price_height)
            close_y = int((1 - norm_kline[i, 1]) * price_height)
            
            # 绘制K线实体
            if close_y < open_y:  # 阴线
                draw.rectangle([(x, close_y), (x + bar_width, open_y)],
                             outline='white', fill='green')
            else:  # 阳线
                draw.rectangle([(x, open_y), (x + bar_width, close_y)],
                             outline='white', fill='red')
            
            # 绘制成交量
            volume_height_px = int(norm_volumes[i] * volume_height)
            if volume_height_px > 0:  # 确保有成交量才绘制
                volume_bottom = volume_end  # 使用volume_end作为底部
                volume_top = volume_bottom - volume_height_px
                
                # 确保不超出成交量区域
                volume_top = max(volume_top, volume_start)
                
                # 成交量柱状图颜色跟随K线颜色
                volume_color = 'green' if close_y < open_y else 'red'
                draw.rectangle([(x, volume_top), (x + bar_width, volume_bottom)],
                             outline='white', fill=volume_color)
        
        # 绘制均线
        if norm_ma10 is not None:
            points = []
            for i in range(self.window_size):
                x = i * bar_width + bar_width // 2
                y = int((1 - norm_ma10[i]) * price_height)
                points.append((x, y))
            # 绘制MA10
            for i in range(len(points)-1):
                draw.line([points[i], points[i+1]], fill='black', width=2)
            
        if norm_ma20 is not None:
            points = []
            for i in range(self.window_size):
                x = i * bar_width + bar_width // 2
                y = int((1 - norm_ma20[i]) * price_height)
                points.append((x, y))
            # 绘制MA20
            for i in range(len(points)-1):
                draw.line([points[i], points[i+1]], fill='blue', width=2)
            
        # 计算并绘制MACD
        dif, dea, macd = self._calculate_macd(kline_data, start_idx)
        
        # 归一化MACD数据
        all_macd_values = np.concatenate([dif, dea, macd])
        norm_macd_values = self._normalize_price_data(all_macd_values)
        
        # 分离归一化后的数据
        norm_dif = norm_macd_values[:self.window_size]
        norm_dea = norm_macd_values[self.window_size:2*self.window_size]
        norm_macd = norm_macd_values[2*self.window_size:]
        
        # 绘制MACD
        macd_zero = macd_start + macd_height // 2  # 零线位置
        
        # 绘制MACD柱状图
        for i in range(self.window_size):
            x = i * bar_width
            macd_value = norm_macd[i]
            
            if macd_value > 0.5:  # MACD > 0
                macd_top = int(macd_zero - (macd_value - 0.5) * macd_height)
                draw.rectangle([(x, macd_top), (x + bar_width, macd_zero)],
                             outline='white', fill='red')
            else:  # MACD < 0
                macd_bottom = int(macd_zero + (0.5 - macd_value) * macd_height)
                draw.rectangle([(x, macd_zero), (x + bar_width, macd_bottom)],
                             outline='white', fill='green')
        
        # 绘制DIF和DEA线
        for i in range(self.window_size-1):
            x1 = i * bar_width + bar_width // 2
            x2 = (i + 1) * bar_width + bar_width // 2
            
            # DIF线(白色)
            y1 = int(macd_start + (1 - norm_dif[i]) * macd_height)
            y2 = int(macd_start + (1 - norm_dif[i+1]) * macd_height)
            draw.line([(x1, y1), (x2, y2)], fill='black', width=1)
            
            # DEA线(黄色)
            y1 = int(macd_start + (1 - norm_dea[i]) * macd_height)
            y2 = int(macd_start + (1 - norm_dea[i+1]) * macd_height)
            draw.line([(x1, y1), (x2, y2)], fill='blue', width=1)
        
        # 绘制分割线
        draw.line([(0, price_end), (self.image_width, price_end)], 
                 fill='gray', width=1)
        draw.line([(0, volume_end), (self.image_width, volume_end)],
                 fill='gray', width=1)

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
