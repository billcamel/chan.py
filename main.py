from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from BuySellPoint.BS_Point import CBS_Point
from BuySellPoint.BSPointList import CBSPointList
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver

if __name__ == "__main__":
    code = "BTC/USDT"
    begin_time = "2024-01-01"
    end_time = None
    data_src = DATA_SRC.PICKLE
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig({
        "bi_strict": True,
        "trigger_step": False,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    plot_para = {
        "seg": {
            # "plot_trendline": True,
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
        },
        "figure": {
            "x_range": 200,
        },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        }
    }
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )
    for kl_type, ckline in chan.kl_datas.items():
        for bs in ckline.bs_point_lst.bsp1_list:
            text = "Buy:" if bs.is_buy else "Sell:"
            print(text+" " +bs.type2str()+": "+str(bs.klu.time))
        for bs in ckline.seg_bs_point_lst.bsp1_list:
            text = "seg Buy:" if bs.is_buy else "seg Sell:"
            print(text+" " +bs.type2str()+": "+str(bs.klu.time))


    # if not config.trigger_step:
    #     plot_driver = CPlotDriver(
    #         chan,
    #         plot_config=plot_config,
    #         plot_para=plot_para,
    #     )
    #     plot_driver.figure.show()
    #     plot_driver.save2img("./test2.png")
    # else:
    #     CAnimateDriver(
    #         chan,
    #         plot_config=plot_config,
    #         plot_para=plot_para,
    #     )
