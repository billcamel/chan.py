class ATR:
    def __init__(self, period: int = 14):
        self.period = period
        self.tr_list = []  # 存储 True Range 值
        self.atr = None  # 当前 ATR 值

    def add(self, high: float, low: float, close_prev: float) -> float:
        """
        计算并返回 ATR。
        :param high: 当前 K 线的最高价
        :param low: 当前 K 线的最低价
        :param close_prev: 前一根 K 线的收盘价
        :return: 当前 ATR 值
        """
        # 计算 True Range (TR)
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        self.tr_list.append(tr)

        # 限制列表长度为周期长度
        if len(self.tr_list) > self.period:
            self.tr_list.pop(0)

        # 计算 ATR
        if len(self.tr_list) < self.period:
            self.atr = sum(self.tr_list) / len(self.tr_list)  # 使用已有数据平均
        else:
            if self.atr is None:
                # 初始 ATR 为 TR 平均值
                self.atr = sum(self.tr_list) / self.period
            else:
                # 增量计算 ATR
                self.atr = (self.atr * (self.period - 1) + tr) / self.period

        return self.atr
