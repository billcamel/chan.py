import pandas as pd

class MASS:
    def __init__(self, period_ema: int = 9, period_mass: int = 25):
        """
        初始化 MASS 指标计算类
        :param period_ema: EMA 的周期（默认 9）
        :param period_mass: 累积和周期（默认 25）
        """
        self.period_ema = period_ema
        self.period_mass = period_mass
        self.ema1 = None  # 第一层 EMA
        self.ema2 = None  # 第二层 EMA
        self.mass_values = []  # 存储 MASS 指标值

    def calculate_ema(self, values: pd.Series, span: int) -> pd.Series:
        """
        计算指数移动平均线 (EMA)
        :param values: 数据序列
        :param span: EMA 周期
        :return: EMA 序列
        """
        return values.ewm(span=span, adjust=False).mean()

    def add(self, high: float, low: float) -> float:
        """
        增量计算 MASS 指标
        :param high: 当前 K 线最高价
        :param low: 当前 K 线最低价
        :return: 当前 MASS 值
        """
        # 高低价差
        high_low = high - low

        # 初始化第一层 EMA
        if self.ema1 is None:
            self.ema1 = high_low
            self.ema2 = high_low
        else:
            # 增量更新 EMA1 和 EMA2
            self.ema1 = (self.ema1 * (self.period_ema - 1) + high_low) / self.period_ema
            self.ema2 = (self.ema2 * (self.period_ema - 1) + self.ema1) / self.period_ema

        # 计算比率
        if self.ema2 != 0:
            ratio = self.ema1 / self.ema2
        else:
            ratio = 0

        # 更新 MASS 累积和
        self.mass_values.append(ratio)
        if len(self.mass_values) > self.period_mass:
            self.mass_values.pop(0)

        # 计算 MASS 指标
        mass = sum(self.mass_values)
        return mass
