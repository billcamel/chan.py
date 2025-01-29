class ROC:
    def __init__(self, period: int = 14):
        """
        计算变动率（ROC）指标
        :param period: 计算变动率的周期
        """
        self.period = period
        self.prices = []

    def add(self, current_price: float) -> float:
        """
        增量计算 ROC 指标
        :param current_price: 当前收盘价
        :return: 变动率值（ROC）
        """
        self.prices.append(current_price)

        # 只保存最近 `period` 个价格
        if len(self.prices) > self.period:
            self.prices.pop(0)

        # 如果数据不足以计算 ROC，则返回 0
        if len(self.prices) < self.period:
            return 0.0

        # 计算 ROC
        price_n_periods_ago = self.prices[0]
        roc = ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100
        return roc
