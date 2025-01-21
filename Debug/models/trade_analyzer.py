import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from datetime import datetime
import matplotlib as mpl

# 设置中文字体
try:
    # macOS
    plt.rcParams['font.family'] = ['Arial Unicode MS']
except:
    try:
        # Windows
        plt.rcParams['font.family'] = ['Microsoft YaHei']
    except:
        # Linux
        plt.rcParams['font.family'] = ['DejaVu Sans']

# 设置正常显示负号
plt.rcParams['axes.unicode_minus'] = False

class TradeAnalyzer:
    """交易分析器"""
    
    def __init__(self, initial_capital: float = 10000):
        """初始化
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = 0  # 持仓数量
        self.trade_history = []  # 交易历史
        self.equity_curve = []  # 资金曲线
        self.trade_times = []  # 交易时间点
        
    def process_trade(self, trade: Dict):
        """处理单次交易
        
        Args:
            trade: 交易信息字典，包含time, is_buy, price等信息
        """
        # 尝试不同的时间格式
        time_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            '%Y-%m-%d %H:%M',
            '%Y/%m/%d',  # 添加日期格式
            '%Y-%m-%d'
        ]
        
        time_str = str(trade['time'])
        for time_format in time_formats:
            try:
                # 如果只有日期，添加时间部分
                if len(time_str) <= 10:  # 只有日期的情况
                    time_str += " 00:00"
                time = datetime.strptime(time_str, time_format)
                break
            except ValueError:
                continue
        else:
            # 如果所有格式都失败，尝试解析CTime对象
            try:
                if hasattr(trade['time'], 'to_str'):
                    time_str = trade['time'].to_str()
                    time = datetime.strptime(time_str, '%Y/%m/%d %H:%M')
                else:
                    raise ValueError(f"无法解析时间格式: {time_str}")
            except:
                raise ValueError(f"无法解析时间格式: {time_str}")
        
        price = trade['price']
        
        if trade['is_buy']:
            # 买入信号，全仓买入
            self.positions = self.current_capital / price
            trade['position'] = self.positions
            trade['capital'] = self.current_capital
        else:
            # 卖出信号，清仓
            if self.positions > 0:
                self.current_capital = self.positions * price
                trade['capital'] = self.current_capital
                self.positions = 0
            
        self.trade_history.append(trade)
        self.equity_curve.append(self.current_capital)
        self.trade_times.append(time)
        
    def analyze_trades(self, trades: List[Dict], threshold: float = 0.6) -> Dict:
        """分析交易结果
        
        Args:
            trades: 交易列表
            threshold: 交易概率阈值
            
        Returns:
            分析结果字典
        """
        # 重置状态
        self.current_capital = self.initial_capital
        self.positions = 0
        self.trade_history = []
        self.equity_curve = []
        self.trade_times = []
        
        # 处理每笔交易
        high_prob_trades = [t for t in trades if t['prob'] > threshold]
        for trade in high_prob_trades:
            self.process_trade(trade)
            
        # 计算交易统计
        returns = np.array(self.equity_curve) / self.initial_capital - 1
        
        stats = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital - 1) * 100,
            'win_rate': self._calculate_win_rate(),
            'max_drawdown': self._calculate_max_drawdown() * 100,
            'profit_ratio': self._calculate_profit_ratio(),
            'trade_count': len(high_prob_trades),
            'buy_count': sum(1 for t in high_prob_trades if t['is_buy']),
            'sell_count': sum(1 for t in high_prob_trades if not t['is_buy'])
        }
        
        return stats
    
    def plot_equity_curve(self):
        """绘制资金曲线"""
        import matplotlib.pyplot as plt
        
        # 计算资金曲线
        capital = self.initial_capital
        equity_data = []  # 存储时间和资金数据
        trade_points = []
        last_buy_price = None
        
        for trade in self.trade_history:
            if trade['prob'] <= 0.6:
                continue
                
            if trade['is_buy']:
                # 买入时记录点位和价格
                last_buy_price = trade['price']
                trade_points.append({
                    'time': trade['time'],
                    'capital': capital,
                    'type': 'buy'
                })
            else:
                # 卖出时更新资金
                if last_buy_price is not None:  # 确保有买入价格
                    profit = (trade['price'] - last_buy_price) / last_buy_price
                    capital *= (1 + profit)
                    equity_data.append({
                        'time': trade['time'],
                        'capital': capital
                    })
                    trade_points.append({
                        'time': trade['time'],
                        'capital': capital,
                        'type': 'sell'
                    })
                    last_buy_price = None  # 重置买入价格
        
        if not equity_data:  # 如果没有交易数据，直接返回
            print("没有足够的交易数据来绘制资金曲线")
            return
            
        # 绘制资金曲线
        plt.figure(figsize=(12, 6))
        
        # 绘制连续的资金曲线
        times = [d['time'] for d in equity_data]
        capitals = [d['capital'] for d in equity_data]
        plt.plot(times, capitals, 'b-', label='资金曲线')
        
        # 标记买卖点
        for point in trade_points:
            if point['type'] == 'buy':
                plt.plot(point['time'], point['capital'], 'g^', markersize=8, 
                        label='买入点' if '买入点' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.plot(point['time'], point['capital'], 'rv', markersize=8,
                        label='卖出点' if '卖出点' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title('交易资金曲线')
        plt.xlabel('时间')
        plt.ylabel('资金')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trade_history:
            return 0
            
        profits = []
        entry_capital = None
        
        for trade in self.trade_history:
            if trade['is_buy']:
                entry_capital = trade['capital']
            else:
                if entry_capital is not None:
                    profits.append(trade['capital'] - entry_capital)
                    entry_capital = None
                    
        if not profits:
            return 0
            
        win_count = sum(1 for p in profits if p > 0)
        return (win_count / len(profits)) * 100
        
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.equity_curve:
            return 0
            
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
        
    def _calculate_profit_ratio(self) -> float:
        """计算盈亏比"""
        if not self.trade_history:
            return 0
            
        profits = []
        entry_capital = None
        
        for trade in self.trade_history:
            if trade['is_buy']:
                entry_capital = trade['capital']
            else:
                if entry_capital is not None:
                    profits.append(trade['capital'] - entry_capital)
                    entry_capital = None
                    
        if not profits:
            return 0
            
        win_trades = [p for p in profits if p > 0]
        loss_trades = [p for p in profits if p < 0]
        
        avg_win = np.mean(win_trades) if win_trades else 0
        avg_loss = abs(np.mean(loss_trades)) if loss_trades else 0
        
        return avg_win / avg_loss if avg_loss != 0 else 0 