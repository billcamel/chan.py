import os,sys
cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
sys.path.append(cpath+"/chan.py")

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, FX_TYPE, KL_TYPE
from DataAPI.BaoStockAPI import CBaoStock
from DataAPI.pickleAPI import PICKLE_API
from models.trade_analyzer import TradeAnalyzer


if __name__ == "__main__":
    """
    一个极其弱智的策略，只交易一类买卖点，底分型形成后就开仓，直到一类卖点顶分型形成后平仓
    只用做展示如何自己实现策略，做回测用~
    相比于strategy_demo.py，本代码演示如何从CChan外部喂K线来触发内部缠论计算
    """
    code = "BTC/USDT"
    begin_time = "2020-01-01"
    end_time = None
    # end_time = "2024-01-01"
    data_src_type = DATA_SRC.PICKLE
    kl_type = KL_TYPE.K_15M
    lv_list = [kl_type]

    config = CChanConfig({
        "trigger_step": True,
        "divergence_rate": 0.8,
        "min_zs_cnt": 1,
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,  # 已经没啥用了这一行
        end_time=end_time,  # 已经没啥用了这一行
        data_src=data_src_type,  # 已经没啥用了这一行
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,  # 已经没啥用了这一行
    )
    PICKLE_API.do_init()
    data_src = PICKLE_API(code, k_type=kl_type, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)  # 初始化数据源类

    # 初始化交易分析器
    analyzer = TradeAnalyzer(initial_capital=1000)
    trades = []  # 存储交易记录

    is_hold = False
    last_buy_price = None
    for klu in data_src.get_kl_data():  # 获取单根K线
        chan.trigger_load({kl_type: [klu]})  # 喂给CChan新增k线
        bsp_list = chan.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]
        if BSP_TYPE.T1 not in last_bsp.type and BSP_TYPE.T1P not in last_bsp.type:
            continue

        cur_lv_chan = chan[0]
        if last_bsp.klu.klc.idx != cur_lv_chan[-2].idx:
            continue
        if cur_lv_chan[-2].fx == FX_TYPE.BOTTOM and last_bsp.is_buy and not is_hold:
            last_buy_price = cur_lv_chan[-1][-1].close
            print(f'{cur_lv_chan[-1][-1].time}:buy price = {last_buy_price}')
            is_hold = True
            # 记录买入交易
            trades.append({
                'time': cur_lv_chan[-1][-1].time,  # 直接传入CTime对象
                'is_buy': True,
                'price': last_buy_price,
                'prob': 1.0  # 这里简单设置为1.0，因为是确定性交易
            })
        elif cur_lv_chan[-2].fx == FX_TYPE.TOP and not last_bsp.is_buy and is_hold:
            sell_price = cur_lv_chan[-1][-1].close
            profit_rate = (sell_price-last_buy_price)/last_buy_price*100
            print(f'{cur_lv_chan[-1][-1].time}:sell price = {sell_price}, profit rate = {profit_rate:.2f}%')
            is_hold = False
            # 记录卖出交易
            trades.append({
                'time': cur_lv_chan[-1][-1].time,  # 直接传入CTime对象
                'is_buy': False,
                'price': sell_price,
                'prob': 1.0  # 这里简单设置为1.0，因为是确定性交易
            })

    PICKLE_API.do_close()

    # 分析交易结果
    stats = analyzer.analyze_trades(trades, threshold=0.5)  # 阈值设为0.5因为我们的prob都是1.0
    analyzer.plot_equity_curve()

    # 打印交易统计
    print("\n交易统计:")
    print(f"初始资金: {stats['initial_capital']:,.0f}")
    print(f"最终资金: {stats['final_capital']:,.0f}")
    print(f"总收益率: {stats['total_return']:.2f}%")
    print(f"胜率: {stats['win_rate']:.2f}%")
    print(f"最大回撤: {stats['max_drawdown']:.2f}%")
    print(f"盈亏比: {stats['profit_ratio']:.2f}")
    print(f"总交易次数: {stats['trade_count']}")
    print(f"买入次数: {stats['buy_count']}")
    print(f"卖出次数: {stats['sell_count']}")