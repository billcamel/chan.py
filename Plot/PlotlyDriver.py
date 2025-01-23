import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Literal, Optional, Tuple, Union

from Chan import CChan
from Common.CEnum import BI_DIR, FX_TYPE, KL_TYPE, KLINE_DIR, TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Math.Demark import T_DEMARK_INDEX, CDemarkEngine

from .PlotMeta import CBi_meta, CChanPlotMeta, CZS_meta


def reformat_plot_config(plot_config: Dict[str, bool]):
    """
    兼容不填写`plot_`前缀的情况
    """
    def _format(s):
        return s if s.startswith("plot_") else f"plot_{s}"

    return {_format(k): v for k, v in plot_config.items()}


def parse_single_lv_plot_config(plot_config: Union[str, dict, list]) -> Dict[str, bool]:
    """
    返回单一级别的plot_config配置
    """
    if isinstance(plot_config, dict):
        return reformat_plot_config(plot_config)
    elif isinstance(plot_config, str):
        return reformat_plot_config(dict([(k.strip().lower(), True) for k in plot_config.split(",")]))
    elif isinstance(plot_config, list):
        return reformat_plot_config(dict([(k.strip().lower(), True) for k in plot_config]))
    else:
        raise CChanException("plot_config only support list/str/dict", ErrCode.PLOT_ERR)


def parse_plot_config(plot_config: Union[str, dict, list], lv_list: List[KL_TYPE]) -> Dict[KL_TYPE, Dict[str, bool]]:
    """
    支持：
        - 传入字典
        - 传入字符串，逗号分割
        - 传入数组，元素为各个需要画的笔的元素
        - 传入key为各个级别的字典
        - 传入key为各个级别的字符串
        - 传入key为各个级别的数组
    """
    if isinstance(plot_config, dict):
        if all(isinstance(_key, str) for _key in plot_config.keys()):  # 单层字典
            return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}
        elif all(isinstance(_key, KL_TYPE) for _key in plot_config.keys()):  # key为KL_TYPE
            for lv in lv_list:
                assert lv in plot_config
            return {lv: parse_single_lv_plot_config(plot_config[lv]) for lv in lv_list}
        else:
            raise CChanException("plot_config if is dict, key must be str/KL_TYPE", ErrCode.PLOT_ERR)
    return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}


class CPlotlyDriver:
    def __init__(self, chan: CChan, plot_config: Union[str, dict, list] = '', plot_para=None):
        if plot_para is None:
            plot_para = {}
        figure_config: dict = plot_para.get('figure', {})

        plot_config = parse_plot_config(plot_config, chan.lv_list)
        plot_metas = self._get_plot_meta(chan, figure_config)
        self.lv_lst = chan.lv_list[:len(plot_metas)]

        x_range = self._get_real_xrange(figure_config, plot_metas[0])
        plot_macd: Dict[KL_TYPE, bool] = {kl_type: conf.get("plot_macd", False) for kl_type, conf in plot_config.items()}

        # 创建plotly figure
        rows = sum([2 if plot_macd[lv] else 1 for lv in self.lv_lst])
        row_heights = [0.8, 0.2] if rows == 2 else [1]
        specs = [[{"secondary_y": True}] for _ in range(rows)]
        self.figure = make_subplots(rows=rows, cols=1, vertical_spacing=0.05, row_heights=row_heights, shared_xaxes=True, specs=specs)

        # 设置默认布局参数
        self.figure.update_layout(
            showlegend=True,  # 显示图例
            xaxis_rangeslider_visible=False,  # 禁用rangeslider
            plot_bgcolor='white',  # 设置白色背景
            paper_bgcolor='white',  # 设置白色背景
        )

        # 设置网格线样式和边框
        self.figure.update_xaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',  # 浅灰色网格线
            showline=True,  # 显示边框线
            linewidth=1,  # 边框线宽度
            linecolor='black',  # 边框线颜色
            mirror=True  # 对称显示边框
        )
        self.figure.update_yaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',  # 浅灰色网格线
            showline=True,  # 显示边框线
            linewidth=1,  # 边框线宽度
            linecolor='black',  # 边框线颜色
            mirror=True  # 对称显示边框
        )

        # 存储每个级别对应的subplot行号
        self.axes_dict = {}
        current_row = 1
        for lv in self.lv_lst:
            if plot_macd[lv]:
                self.axes_dict[lv] = [current_row, current_row + 1]
                current_row += 2
            else:
                self.axes_dict[lv] = [current_row]
                current_row += 1

        sseg_begin = 0
        slv_seg_cnt = plot_para.get('seg', {}).get('sub_lv_cnt', None)
        sbi_begin = 0
        slv_bi_cnt = plot_para.get('bi', {}).get('sub_lv_cnt', None)
        srange_begin = 0
        assert slv_seg_cnt is None or slv_bi_cnt is None, "you can set at most one of seg_sub_lv_cnt/bi_sub_lv_cnt"

        for meta, lv in zip(plot_metas, self.lv_lst):
            row = self.axes_dict[lv][0]
            macd_row = self.axes_dict[lv][1] if len(self.axes_dict[lv]) > 1 else None

            # 设置标题
            self.figure.update_layout(
                title_text=f"{chan.code}/{lv.name.split('K_')[1]}",
                title_x=0,
                title_font=dict(size=16, color='red')
            )

            # 计算x轴范围
            x_limits = self._cal_x_limit(meta, x_range)
            if lv != self.lv_lst[0]:
                if sseg_begin != 0 or sbi_begin != 0:
                    x_limits[0] = max(sseg_begin, sbi_begin)
                elif srange_begin != 0:
                    x_limits[0] = srange_begin

            # 绘制元素
            self._draw_element(plot_config[lv], meta, row, lv, plot_para, macd_row, x_limits)

            if lv != self.lv_lst[-1]:
                if slv_seg_cnt is not None:
                    sseg_begin = meta.sub_last_kseg_start_idx(slv_seg_cnt)
                if slv_bi_cnt is not None:
                    sbi_begin = meta.sub_last_kbi_start_idx(slv_bi_cnt)
                if x_range != 0:
                    srange_begin = meta.sub_range_start_idx(x_range)

    @staticmethod
    def _get_plot_meta(chan: CChan, figure_config) -> List[CChanPlotMeta]:
        plot_metas = [CChanPlotMeta(chan[kl_type]) for kl_type in chan.lv_list]
        if figure_config.get("only_top_lv", False):
            plot_metas = [plot_metas[0]]
        return plot_metas

    def _get_real_xrange(self, figure_config, meta: CChanPlotMeta):
        x_range = figure_config.get("x_range", 0)
        bi_cnt = figure_config.get("x_bi_cnt", 0)
        seg_cnt = figure_config.get("x_seg_cnt", 0)
        x_begin_date = figure_config.get("x_begin_date", 0)

        if x_range != 0:
            assert bi_cnt == 0 and seg_cnt == 0 and x_begin_date == 0, "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            return x_range
        if bi_cnt != 0:
            assert x_range == 0 and seg_cnt == 0 and x_begin_date == 0, "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            X_LEN = meta.klu_len
            if len(meta.bi_list) < bi_cnt:
                return 0
            x_range = X_LEN - meta.bi_list[-bi_cnt].begin_x
            return x_range
        if seg_cnt != 0:
            assert x_range == 0 and bi_cnt == 0 and x_begin_date == 0, "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            X_LEN = meta.klu_len
            if len(meta.seg_list) < seg_cnt:
                return 0
            x_range = X_LEN - meta.seg_list[-seg_cnt].begin_x
            return x_range
        if x_begin_date != 0:
            assert x_range == 0 and bi_cnt == 0 and seg_cnt == 0, "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            x_range = 0
            for date_tick in meta.datetick[::-1]:
                if date_tick >= x_begin_date:
                    x_range += 1
                else:
                    break
            return x_range
        return x_range

    @staticmethod
    def _cal_x_limit(meta: CChanPlotMeta, x_range):
        X_LEN = meta.klu_len
        return [X_LEN - x_range, X_LEN - 1] if x_range and X_LEN > x_range else [0, X_LEN - 1]

    def _draw_element(self, plot_config: Dict[str, bool], meta: CChanPlotMeta, row: int, lv, plot_para, macd_row: Optional[int], x_limits):
        if plot_config.get("plot_kline", False):
            self.draw_klu(meta, row, **plot_para.get('kl', {}))
        if plot_config.get("plot_kline_combine", False):
            self.draw_klc(meta, row, **plot_para.get('klc', {}))
        if plot_config.get("plot_bi", False):
            self.draw_bi(meta, row, lv, **plot_para.get('bi', {}))
        if plot_config.get("plot_seg", False):
            self.draw_seg(meta, row, lv, **plot_para.get('seg', {}))
        if plot_config.get("plot_segseg", False):
            self.draw_segseg(meta, row, **plot_para.get('segseg', {}))
        if plot_config.get("plot_eigen", False):
            self.draw_eigen(meta, row, **plot_para.get('eigen', {}))
        if plot_config.get("plot_segeigen", False):
            self.draw_segeigen(meta, row, **plot_para.get('segeigen', {}))
        if plot_config.get("plot_zs", False):
            self.draw_zs(meta, row, **plot_para.get('zs', {}))
        if plot_config.get("plot_segzs", False):
            self.draw_segzs(meta, row, **plot_para.get('segzs', {}))
        if plot_config.get("plot_macd", False) and macd_row is not None:
            self.draw_macd(meta, macd_row, x_limits, **plot_para.get('macd', {}))
        if plot_config.get("plot_mean", False):
            self.draw_mean(meta, row, **plot_para.get('mean', {}))
        if plot_config.get("plot_channel", False):
            self.draw_channel(meta, row, **plot_para.get('channel', {}))
        if plot_config.get("plot_boll", False):
            self.draw_boll(meta, row, **plot_para.get('boll', {}))
        if plot_config.get("plot_bsp", False):
            self.draw_bs_point(meta, row, **plot_para.get('bsp', {}))
        if plot_config.get("plot_segbsp", False):
            self.draw_seg_bs_point(meta, row, **plot_para.get('seg_bsp', {}))
        if plot_config.get("plot_demark", False):
            self.draw_demark(meta, row, **plot_para.get('demark', {}))
        if plot_config.get("plot_marker", False):
            self.draw_marker(meta, row, **plot_para.get('marker', {'markers': {}}))
        if plot_config.get("plot_rsi", False):
            self.draw_rsi(meta, row, **plot_para.get('rsi', {}))
        if plot_config.get("plot_kdj", False):
            self.draw_kdj(meta, row, **plot_para.get('kdj', {}))

    def draw_klu(self, meta: CChanPlotMeta, row: int, width=0.4, rugd=True, plot_mode="kl"):
        # rugd: red up green down
        up_color = 'red' if rugd else 'green'
        down_color = 'green' if rugd else 'red'

        if plot_mode == "kl":
            opens = []
            highs = []
            lows = []
            closes = []
            dates = []
            colors = []

            for kl in meta.klu_iter():
                opens.append(kl.open)
                highs.append(kl.high)
                lows.append(kl.low)
                closes.append(kl.close)
                dates.append(kl.time.to_str())
                colors.append(up_color if kl.close > kl.open else down_color)

            self.figure.add_trace(
                go.Candlestick(
                    x=dates,
                    open=opens,
                    high=highs,
                    low=lows,
                    close=closes,
                    increasing_line_color=up_color,
                    decreasing_line_color=down_color,
                    line=dict(width=width)
                ),
                row=row,
                col=1
            )
            # 设置x轴格式
            self.figure.update_xaxes(
                row=row,
                col=1,
                tickangle=45,
                type='category',
                rangeslider=dict(visible=False),
                nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
            )
        else:
            x = []
            y = []
            for kl in meta.klu_iter():
                x.append(kl.time.to_str())
                if plot_mode == "close":
                    y.append(kl.close)
                elif plot_mode == "high":
                    y.append(kl.high)
                elif plot_mode == "low":
                    y.append(kl.low)
                elif plot_mode == "open":
                    y.append(kl.open)
                else:
                    raise CChanException(f"unknow plot mode={plot_mode}, must be one of kl/close/open/high/low", ErrCode.PLOT_ERR)

            self.figure.add_trace(
                go.Scatter(x=x, y=y, mode='lines'),
                row=row,
                col=1
            )
            # 设置x轴格式
            self.figure.update_xaxes(
                row=row,
                col=1,
                tickangle=45,
                type='category',
                nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
            )

    def draw_klc(self, meta: CChanPlotMeta, row: int, width=0.4, plot_single_kl=True):
        color_type = {FX_TYPE.TOP: 'red', FX_TYPE.BOTTOM: 'blue', KLINE_DIR.UP: 'green', KLINE_DIR.DOWN: 'green'}

        # 获取所有K线的时间点
        dates = [klu.time.to_str() for klu in meta.klu_iter()]

        for klc_meta in meta.klc_list:
            if klc_meta.end_idx == klc_meta.begin_idx and not plot_single_kl:
                continue

            klu_list = list(meta.klu_iter())
            begin_time = klu_list[klc_meta.begin_idx].time.to_str()
            end_time = klu_list[klc_meta.end_idx].time.to_str()
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, begin_time, end_time, end_time],
                    y=[klc_meta.low, klc_meta.high, klc_meta.high, klc_meta.low],
                    fill="toself",
                    mode='lines',
                    line=dict(color=color_type[klc_meta.type]),
                    showlegend=False
                ),
                row=row,
                col=1
            )
            # 设置x轴格式
            self.figure.update_xaxes(
                row=row,
                col=1,
                tickangle=45,
                type='category',
                nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
            )

    def draw_bi(self, meta: CChanPlotMeta, row, lv, width=2, color="black", sub_lv_cnt=None, facecolor='green', alpha=0.1, disp_end=False, end_color='green', end_fontsize=13, show_num=False, num_fontsize=15, num_color="red"):
        # 获取x轴范围
        x_begin = 0  # 默认从0开始
        if hasattr(self.figure.layout, f'xaxis{row}'):
            x_range = getattr(self.figure.layout, f'xaxis{row}').range
            if x_range:
                x_begin = int(x_range[0])

        klu_list = list(meta.klu_iter())
        for bi_idx, bi in enumerate(meta.bi_list):
            if bi.end_x < x_begin:
                continue
            begin_time = klu_list[bi.begin_x].time.to_str()
            end_time = klu_list[bi.end_x].time.to_str()
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, end_time],
                    y=[bi.begin_y, bi.end_y],
                    mode='lines',
                    line=dict(color=color, width=width),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 显示笔的编号
            if show_num:
                self.figure.add_trace(
                    go.Scatter(
                        x=[(begin_time + end_time) / 2],
                        y=[(bi.begin_y + bi.end_y) / 2],
                        mode='text',
                        text=[str(bi.idx)],
                        textfont=dict(size=num_fontsize, color=num_color),
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

            # 显示端点值
            if disp_end:
                if bi_idx == 0:
                    self.figure.add_trace(
                        go.Scatter(
                            x=[begin_time],
                            y=[bi.begin_y],
                            mode='text',
                            text=[f'{bi.begin_y:.5f}'],
                            textfont=dict(size=end_fontsize, color=end_color),
                            textposition='top center' if bi.dir == BI_DIR.UP else 'bottom center',
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )
                self.figure.add_trace(
                    go.Scatter(
                        x=[end_time],
                        y=[bi.end_y],
                        mode='text',
                        text=[f'{bi.end_y:.5f}'],
                        textfont=dict(size=end_fontsize, color=end_color),
                        textposition='top center' if bi.dir == BI_DIR.DOWN else 'bottom center',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

        # 绘制子级别的背景
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            if sub_lv_cnt >= len(meta.bi_list):
                return
            begin_time = klu_list[meta.bi_list[-sub_lv_cnt].begin_x].time.to_str()
            end_time = klu_list[-1].time.to_str()
            y_range = [0, 0]  # 默认值
            if hasattr(self.figure.layout, f'yaxis{row}'):
                y_range = getattr(self.figure.layout, f'yaxis{row}').range or y_range

            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, end_time, end_time, begin_time],
                    y=[y_range[0], y_range[0], y_range[1], y_range[1]],
                    fill="toself",
                    fillcolor=facecolor,
                    opacity=alpha,
                    mode='none',
                    showlegend=False
                ),
                row=row,
                col=1
            )

    def draw_segseg(self, meta: CChanPlotMeta, row: int, width=7, color="brown", disp_end=False, end_color='brown', end_fontsize=15):
        klu_list = list(meta.klu_iter())
        for seg_idx, seg_meta in enumerate(meta.segseg_list):
            # 绘制线段的线段
            begin_time = klu_list[seg_meta.begin_x].time.to_str()
            end_time = klu_list[seg_meta.end_x].time.to_str()
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, end_time],
                    y=[seg_meta.begin_y, seg_meta.end_y],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=width,
                        dash='dash' if not seg_meta.is_sure else 'solid'
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 显示端点值
            if disp_end:
                if seg_idx == 0:
                    self.figure.add_trace(
                        go.Scatter(
                            x=[begin_time],
                            y=[seg_meta.begin_y],
                            mode='text',
                            text=[f'{seg_meta.begin_y:.2f}'],
                            textfont=dict(size=end_fontsize, color=end_color),
                            textposition='top center' if seg_meta.dir == BI_DIR.UP else 'bottom center',
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )
                self.figure.add_trace(
                    go.Scatter(
                        x=[end_time],
                        y=[seg_meta.end_y],
                        mode='text',
                        text=[f'{seg_meta.end_y:.2f}'],
                        textfont=dict(size=end_fontsize, color=end_color),
                        textposition='top center' if seg_meta.dir == BI_DIR.DOWN else 'bottom center',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    def draw_seg(self, meta: CChanPlotMeta, row: int, lv, width=5, color="green", sub_lv_cnt=None,
                 facecolor='green', alpha=0.1, disp_end=False, end_color='green', end_fontsize=13,
                 plot_trendline=False, trendline_color='r', trendline_width=3):
        klu_list = list(meta.klu_iter())
        for seg_idx, seg_meta in enumerate(meta.seg_list):
            # 绘制线段
            begin_time = klu_list[seg_meta.begin_x].time.to_str()
            end_time = klu_list[seg_meta.end_x].time.to_str()
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, end_time],
                    y=[seg_meta.begin_y, seg_meta.end_y],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=width,
                        dash='dash' if not seg_meta.is_sure else 'solid'
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 显示端点值
            if disp_end:
                if seg_idx == 0:
                    self.figure.add_trace(
                        go.Scatter(
                            x=[begin_time],
                            y=[seg_meta.begin_y],
                            mode='text',
                            text=[f'{seg_meta.begin_y:.5f}'],
                            textfont=dict(size=end_fontsize, color=end_color),
                            textposition='top center' if seg_meta.dir == BI_DIR.UP else 'bottom center',
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )
                self.figure.add_trace(
                    go.Scatter(
                        x=[end_time],
                        y=[seg_meta.end_y],
                        mode='text',
                        text=[f'{seg_meta.end_y:.2f}'],
                        textfont=dict(size=end_fontsize, color=end_color),
                        textposition='top center' if seg_meta.dir == BI_DIR.DOWN else 'bottom center',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

            # 绘制趋势线
            if plot_trendline:
                if seg_meta.tl.get('support'):
                    tl_meta = seg_meta.format_tl(seg_meta.tl['support'])
                    tl_begin_time = klu_list[int(tl_meta[0])].time.to_str()
                    tl_end_time = klu_list[int(tl_meta[2])].time.to_str()
                    self.figure.add_trace(
                        go.Scatter(
                            x=[tl_begin_time, tl_end_time],
                            y=[tl_meta[1], tl_meta[3]],
                            mode='lines',
                            line=dict(color=trendline_color, width=trendline_width),
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )
                if seg_meta.tl.get('resistance'):
                    tl_meta = seg_meta.format_tl(seg_meta.tl['resistance'])
                    tl_begin_time = klu_list[int(tl_meta[0])].time.to_str()
                    tl_end_time = klu_list[int(tl_meta[2])].time.to_str()
                    self.figure.add_trace(
                        go.Scatter(
                            x=[tl_begin_time, tl_end_time],
                            y=[tl_meta[1], tl_meta[3]],
                            mode='lines',
                            line=dict(color=trendline_color, width=trendline_width),
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )

        # 绘制子级别的背景
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            if sub_lv_cnt >= len(meta.seg_list):
                return
            begin_time = klu_list[meta.seg_list[-sub_lv_cnt].begin_x].time.to_str()
            end_time = klu_list[-1].time.to_str()
            y_range = [0, 0]  # 默认值
            if hasattr(self.figure.layout, f'yaxis{row}'):
                y_range = getattr(self.figure.layout, f'yaxis{row}').range or y_range
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, end_time, end_time, begin_time],
                    y=[y_range[0], y_range[0], y_range[1], y_range[1]],
                    fill="toself",
                    fillcolor=facecolor,
                    opacity=alpha,
                    mode='none',
                    showlegend=False
                ),
                row=row,
                col=1
            )
            # 设置x轴格式
            self.figure.update_xaxes(
                row=row,
                col=1,
                tickangle=45,
                type='category',
                nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
            )

    def draw_zs(self, meta: CChanPlotMeta, row: int, color='orange', linewidth=2, sub_linewidth=0.5, show_text=False, fontsize=14, text_color='orange', draw_one_bi_zs=False):
        klu_list = list(meta.klu_iter())
        for zs_meta in meta.zs_lst:
            if not draw_one_bi_zs and zs_meta.is_onebi_zs:
                continue

            begin_time = klu_list[zs_meta.begin].time.to_str()
            end_time = klu_list[zs_meta.begin + zs_meta.w].time.to_str()
            # 绘制主中枢矩形
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, begin_time, end_time, end_time, begin_time],
                    y=[zs_meta.low, zs_meta.high, zs_meta.high, zs_meta.low, zs_meta.low],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=linewidth,
                        dash='dash' if not zs_meta.is_sure else 'solid'
                    ),
                    fill='toself',
                    fillcolor='rgba(0,0,0,0)',
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制子中枢
            for sub_zs_meta in zs_meta.sub_zs_lst:
                sub_begin_time = klu_list[sub_zs_meta.begin].time.to_str()
                sub_end_time = klu_list[sub_zs_meta.begin + sub_zs_meta.w].time.to_str()
                self.figure.add_trace(
                    go.Scatter(
                        x=[sub_begin_time, sub_begin_time, sub_end_time, sub_end_time, sub_begin_time],
                        y=[sub_zs_meta.low, sub_zs_meta.high, sub_zs_meta.high, sub_zs_meta.low, sub_zs_meta.low],
                        mode='lines',
                        line=dict(
                            color=color,
                            width=sub_linewidth,
                            dash='dash' if not sub_zs_meta.is_sure else 'solid'
                        ),
                        fill='toself',
                        fillcolor='rgba(0,0,0,0)',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

                # 显示中枢文本
                if show_text:
                    self.figure.add_trace(
                        go.Scatter(
                            x=[(sub_begin_time + sub_end_time) / 2],
                            y=[sub_zs_meta.high],
                            mode='text',
                            text=[f'ZG:{sub_zs_meta.high:.2f}\nZD:{sub_zs_meta.low:.2f}'],
                            textfont=dict(size=fontsize, color=text_color),
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )

            # 显示主中枢文本
            if show_text:
                self.figure.add_trace(
                    go.Scatter(
                        x=[(begin_time + end_time) / 2],
                        y=[zs_meta.high],
                        mode='text',
                        text=[f'ZG:{zs_meta.high:.2f}\nZD:{zs_meta.low:.2f}'],
                        textfont=dict(size=fontsize, color=text_color),
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    def draw_macd(self, meta: CChanPlotMeta, row: int, x_limits, width=0.4):
        macd_lst = [klu.macd for klu in meta.klu_iter()]
        assert macd_lst[0] is not None, "you can't draw macd until you delete macd_metric=False"

        x_begin = x_limits[0]
        dates = [klu.time.to_str() for klu in meta.klu_iter()][x_begin:]
        dif_line = [macd.DIF for macd in macd_lst[x_begin:]]
        dea_line = [macd.DEA for macd in macd_lst[x_begin:]]
        macd_bar = [macd.macd for macd in macd_lst[x_begin:]]

        # 绘制DIF线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=dif_line,
                mode='lines',
                line=dict(color='#FFA500'),
                name='DIF',
                showlegend=False
            ),
            row=row,
            col=1
        )

        # 绘制DEA线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=dea_line,
                mode='lines',
                line=dict(color='#0000ff'),
                name='DEA',
                showlegend=False
            ),
            row=row,
            col=1
        )

        # 绘制MACD柱状图
        colors = ['rgba(255,20,20,1)' if val >= 0 else 'rgba(0,180,0,1)' for val in macd_bar]
        self.figure.add_trace(
            go.Bar(
                x=dates,
                y=macd_bar,
                marker_color=colors,
                width=width,
                name='MACD',
                showlegend=False
            ),
            row=row,
            col=1
        )

        # 设置x轴格式
        self.figure.update_xaxes(
            row=row,
            col=1,
            tickangle=45,
            type='category',
            nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
        )

    def save2html(self, path):
        self.figure.write_html(path)

    def draw_mean(self, meta: CChanPlotMeta, row: int):
        mean_lst = [klu.trend[TREND_TYPE.MEAN] for klu in meta.klu_iter()]
        Ts = list(mean_lst[0].keys())

        for T in Ts:
            mean_arr = [mean_dict[T] for mean_dict in mean_lst]
            dates = [klu.time.to_str() for klu in meta.klu_iter()]
            self.figure.add_trace(
                go.Scatter(
                    x=dates,
                    y=mean_arr,
                    mode='lines',
                    name=f'{T} meanline',
                    showlegend=True
                ),
                row=row,
                col=1
            )

    def draw_channel(self, meta: CChanPlotMeta, row: int, T=None, top_color="red", bottom_color="blue", linewidth=3, linestyle="solid"):
        max_lst = [klu.trend[TREND_TYPE.MAX] for klu in meta.klu_iter()]
        min_lst = [klu.trend[TREND_TYPE.MIN] for klu in meta.klu_iter()]
        config_T_lst = sorted(list(max_lst[0].keys()))
        if T is None:
            T = config_T_lst[-1]
        elif T not in max_lst[0]:
            raise CChanException(f"plot channel of T={T} is not setted in CChanConfig.trend_metrics = {config_T_lst}", ErrCode.PLOT_ERR)

        dates = [klu.time.to_str() for klu in meta.klu_iter()]
        top_array = [_d[T] for _d in max_lst]
        bottom_array = [_d[T] for _d in min_lst]

        # 绘制上轨道线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=top_array,
                mode='lines',
                line=dict(color=top_color, width=linewidth, dash='solid' if linestyle == 'solid' else 'dash'),
                name=f'{T}-TOP-channel',
                showlegend=True
            ),
            row=row,
            col=1
        )

        # 绘制下轨道线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=bottom_array,
                mode='lines',
                line=dict(color=bottom_color, width=linewidth, dash='solid' if linestyle == 'solid' else 'dash'),
                name=f'{T}-BOTTOM-channel',
                showlegend=True
            ),
            row=row,
            col=1
        )

    def draw_boll(self, meta: CChanPlotMeta, row: int, mid_color="black", up_color="blue", down_color="purple"):
        dates = [klu.time.to_str() for klu in meta.klu_iter()]
        try:
            ma = [klu.boll.MID for klu in meta.klu_iter()]
            up = [klu.boll.UP for klu in meta.klu_iter()]
            down = [klu.boll.DOWN for klu in meta.klu_iter()]
        except AttributeError as e:
            raise CChanException("you can't draw boll until you set boll_n in CChanConfig", ErrCode.PLOT_ERR) from e

        # 绘制中轨线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=ma,
                mode='lines',
                line=dict(color=mid_color),
                name='BOLL-MID',
                showlegend=True
            ),
            row=row,
            col=1
        )

        # 绘制上轨线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=up,
                mode='lines',
                line=dict(color=up_color),
                name='BOLL-UP',
                showlegend=True
            ),
            row=row,
            col=1
        )

        # 绘制下轨线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=down,
                mode='lines',
                line=dict(color=down_color),
                name='BOLL-DOWN',
                showlegend=True
            ),
            row=row,
            col=1
        )

    def draw_rsi(self, meta: CChanPlotMeta, row: int, color='blue'):
        dates = [klu.time.to_str() for klu in meta.klu_iter()]
        data = [klu.rsi for klu in meta.klu_iter()]

        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=data,
                mode='lines',
                line=dict(color=color),
                name='RSI',
                showlegend=True
            ),
            row=row,
            col=1,
            secondary_y=True
        )

        # 设置x轴格式
        self.figure.update_xaxes(
            row=row,
            col=1,
            tickangle=45,
            type='category',
            nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
        )

        # 设置第二个y轴的标题和范围
        self.figure.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            secondary_y=True,
            row=row,
            col=1
        )

    def draw_bs_point(self, meta: CChanPlotMeta, row: int, buy_color='red', sell_color='green', fontsize=15, arrow_l=0.15, arrow_h=0.2, arrow_w=1):
        klu_list = list(meta.klu_iter())
        for bsp in meta.bs_point_lst:
            color = buy_color if bsp.is_buy else sell_color
            arrow_dir = 1 if bsp.is_buy else -1
            time = klu_list[bsp.x].time.to_str()

            # 获取y轴范围
            y_range = self.figure.layout[f'yaxis{row}'].range
            if y_range is None:
                # 如果y_range未设置，使用当前数据的最大最小值
                y_values = [bsp.y for bsp in meta.bs_point_lst]
                y_min = min(y_values) if y_values else 0
                y_max = max(y_values) if y_values else 1
                y_range = [y_min, y_max]
            arrow_len = arrow_l * (y_range[1] - y_range[0])
            arrow_head = arrow_len * arrow_h

            # 绘制箭头线
            self.figure.add_trace(
                go.Scatter(
                    x=[time, time],
                    y=[bsp.y - arrow_len * arrow_dir, bsp.y - arrow_head * arrow_dir],
                    mode='lines',
                    line=dict(color=color, width=arrow_w),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制箭头头部
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if bsp.is_buy else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 添加文本标签
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y - arrow_len * arrow_dir],
                    mode='text',
                    text=[bsp.desc()],
                    textfont=dict(size=fontsize, color=color),
                    textposition='top center' if not bsp.is_buy else 'bottom center',
                    showlegend=False
                ),
                row=row,
                col=1
            )

    def draw_seg_bs_point(self, meta: CChanPlotMeta, row: int, buy_color='red', sell_color='green', fontsize=18, arrow_l=0.2, arrow_h=0.25, arrow_w=1.2):
        klu_list = list(meta.klu_iter())
        for bsp in meta.seg_bsp_lst:
            color = buy_color if bsp.is_buy else sell_color
            arrow_dir = 1 if bsp.is_buy else -1
            time = klu_list[bsp.x].time.to_str()

            # 获取y轴范围
            y_range = self.figure.layout[f'yaxis{row}'].range
            if y_range is None:
                # 如果y_range未设置，使用当前数据的最大最小值
                y_values = [bsp.y for bsp in meta.bs_point_lst]
                y_min = min(y_values) if y_values else 0
                y_max = max(y_values) if y_values else 1
                y_range = [y_min, y_max]
            arrow_len = arrow_l * (y_range[1] - y_range[0])
            arrow_head = arrow_len * arrow_h

            # 绘制箭头线
            self.figure.add_trace(
                go.Scatter(
                    x=[time, time],
                    y=[bsp.y - arrow_len * arrow_dir, bsp.y - arrow_head * arrow_dir],
                    mode='lines',
                    line=dict(color=color, width=arrow_w),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制箭头头部
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if bsp.is_buy else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 添加文本标签
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y - arrow_len * arrow_dir],
                    mode='text',
                    text=[bsp.desc()],
                    textfont=dict(size=fontsize, color=color),
                    textposition='top center' if not bsp.is_buy else 'bottom center',
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 添加箭头
            y_range = self.figure.layout[f'yaxis{row}'].range
            if y_range is None:
                # 如果y_range未设置，使用当前数据的最大最小值
                y_values = [bsp.y for bsp in meta.bs_point_lst]
                y_min = min(y_values) if y_values else 0
                y_max = max(y_values) if y_values else 1
                y_range = [y_min, y_max]
            arrow_len = arrow_l * (y_range[1] - y_range[0])
            arrow_head = arrow_len * arrow_h

            # 绘制箭头线
            self.figure.add_trace(
                go.Scatter(
                    x=[time, time],
                    y=[bsp.y - arrow_len * arrow_dir, bsp.y - arrow_head * arrow_dir],
                    mode='lines',
                    line=dict(color=color, width=arrow_w),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制箭头头部
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if bsp.is_buy else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 添加文本标签
            self.figure.add_trace(
                go.Scatter(
                    x=[time],
                    y=[bsp.y - arrow_len * arrow_dir],
                    mode='text',
                    text=[bsp.desc()],
                    textfont=dict(size=fontsize, color=color),
                    textposition='top center' if not bsp.is_buy else 'bottom center',
                    showlegend=False
                ),
                row=row,
                col=1
            )

    def draw_eigen(self, meta: CChanPlotMeta, row: int, color_top="red", color_bottom="blue", aplha=0.5, only_peak=False):
        klu_list = list(meta.klu_iter())
        for eigenfx_meta in meta.eigenfx_lst:
            # 获取x轴范围
            x_begin = 0
            if hasattr(self.figure.layout, f'xaxis{row}'):
                x_range = getattr(self.figure.layout, f'xaxis{row}').range
                if x_range:
                    x_begin = int(x_range[0])

            # 设置颜色
            color = color_top if eigenfx_meta.fx == FX_TYPE.TOP else color_bottom

            # 绘制特征序列
            for idx, eigen_meta in enumerate(eigenfx_meta.ele):
                if eigen_meta.begin_x + eigen_meta.w < x_begin:
                    continue
                if only_peak and idx != 1:
                    continue

                # 创建矩形区域
                self.figure.add_trace(
                    go.Scatter(
                        x=[klu_list[eigen_meta.begin_x].time.to_str(),
                           klu_list[eigen_meta.begin_x + eigen_meta.w].time.to_str(),
                           klu_list[eigen_meta.begin_x + eigen_meta.w].time.to_str(),
                           klu_list[eigen_meta.begin_x].time.to_str(),
                           klu_list[eigen_meta.begin_x].time.to_str()],
                        y=[eigen_meta.begin_y,
                           eigen_meta.begin_y,
                           eigen_meta.begin_y + eigen_meta.h,
                           eigen_meta.begin_y + eigen_meta.h,
                           eigen_meta.begin_y],
                        fill="toself",
                        fillcolor=color,
                        opacity=aplha,
                        mode='lines',
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    def draw_segzs(self, meta: CChanPlotMeta, row: int, color='red', linewidth=10, sub_linewidth=4):
        klu_list = list(meta.klu_iter())
        for zs_meta in meta.segzs_lst:
            begin_time = klu_list[zs_meta.begin].time.to_str()
            end_time = klu_list[zs_meta.begin + zs_meta.w].time.to_str()
            # 绘制主中枢矩形
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_time, begin_time, end_time, end_time, begin_time],
                    y=[zs_meta.low, zs_meta.high, zs_meta.high, zs_meta.low, zs_meta.low],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=linewidth,
                        dash='dash' if not zs_meta.is_sure else 'solid'
                    ),
                    fill='toself',
                    fillcolor='rgba(0,0,0,0)',
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制子中枢
            for sub_zs_meta in zs_meta.sub_zs_lst:
                sub_begin_time = klu_list[sub_zs_meta.begin].time.to_str()
                sub_end_time = klu_list[sub_zs_meta.begin + sub_zs_meta.w].time.to_str()
                self.figure.add_trace(
                    go.Scatter(
                        x=[sub_begin_time, sub_begin_time, sub_end_time, sub_end_time, sub_begin_time],
                        y=[sub_zs_meta.low, sub_zs_meta.high, sub_zs_meta.high, sub_zs_meta.low, sub_zs_meta.low],
                        mode='lines',
                        line=dict(
                            color=color,
                            width=sub_linewidth,
                            dash='dash' if not sub_zs_meta.is_sure else 'solid'
                        ),
                        fill='toself',
                        fillcolor='rgba(0,0,0,0)',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    def draw_segeigen(self, meta: CChanPlotMeta, row: int, color_top="red", color_bottom="blue", aplha=0.5, only_peak=False):
        klu_list = list(meta.klu_iter())
        for eigenfx_meta in meta.seg_eigenfx_lst:
            # 获取x轴范围
            x_begin = 0
            if hasattr(self.figure.layout, f'xaxis{row}'):
                x_range = getattr(self.figure.layout, f'xaxis{row}').range
                if x_range:
                    x_begin = int(x_range[0])

            # 设置颜色
            color = color_top if eigenfx_meta.fx == FX_TYPE.TOP else color_bottom

            # 绘制特征序列
            for idx, eigen_meta in enumerate(eigenfx_meta.ele):
                if eigen_meta.begin_x + eigen_meta.w < x_begin:
                    continue
                if only_peak and idx != 1:
                    continue

                # 创建矩形区域
                self.figure.add_trace(
                    go.Scatter(
                        x=[klu_list[eigen_meta.begin_x].time.to_str(),
                           klu_list[eigen_meta.begin_x + eigen_meta.w].time.to_str(),
                           klu_list[eigen_meta.begin_x + eigen_meta.w].time.to_str(),
                           klu_list[eigen_meta.begin_x].time.to_str(),
                           klu_list[eigen_meta.begin_x].time.to_str()],
                        y=[eigen_meta.begin_y,
                           eigen_meta.begin_y,
                           eigen_meta.begin_y + eigen_meta.h,
                           eigen_meta.begin_y + eigen_meta.h,
                           eigen_meta.begin_y],
                        fill="toself",
                        fillcolor=color,
                        opacity=aplha,
                        mode='lines',
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

    def draw_kdj(self, meta: CChanPlotMeta, row: int, k_color='orange', d_color='blue', j_color='pink'):
        dates = [klu.time.to_str() for klu in meta.klu_iter()]
        kdj = [klu.kdj for klu in meta.klu_iter()]

        # 绘制K线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=[x.k for x in kdj],
                mode='lines',
                line=dict(color=k_color),
                name='KDJ-K',
                showlegend=True
            ),
            row=row,
            col=1,
            secondary_y=True
        )

        # 绘制D线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=[x.d for x in kdj],
                mode='lines',
                line=dict(color=d_color),
                name='KDJ-D',
                showlegend=True
            ),
            row=row,
            col=1,
            secondary_y=True
        )

        # 绘制J线
        self.figure.add_trace(
            go.Scatter(
                x=dates,
                y=[x.j for x in kdj],
                mode='lines',
                line=dict(color=j_color),
                name='KDJ-J',
                showlegend=True
            ),
            row=row,
            col=1,
            secondary_y=True
        )

        # 设置x轴格式
        self.figure.update_xaxes(
            row=row,
            col=1,
            tickangle=45,
            type='category',
            nticks=min(20, max(5, int(len(dates) / 10)))  # 根据数据点数量动态调整刻度数
        )

        # 设置第二个y轴的标题和范围
        self.figure.update_yaxes(
            title_text="KDJ",
            range=[0, 100],
            secondary_y=True,
            row=row,
            col=1
        )

    def draw_marker(self, meta: CChanPlotMeta, row: int, markers: Dict[CTime | str, Tuple[str, Literal['up', 'down'], str] | Tuple[str, Literal['up', 'down']]], arrow_l=0.15, arrow_h_r=0.2, arrow_w=1, fontsize=14, default_color='blue'):
        # 获取x轴范围
        x_range = None
        if hasattr(self.figure.layout, f'xaxis{row}'):
            x_range = getattr(self.figure.layout, f'xaxis{row}').range

        # 获取y轴范围
        y_range = None
        if hasattr(self.figure.layout, f'yaxis{row}'):
            y_range = getattr(self.figure.layout, f'yaxis{row}').range

        # 处理日期映射
        datetick_dict = {date: idx for idx, date in enumerate([klu.time.to_str() for klu in meta.klu_iter()])}

        # 处理新的标记点
        new_marker = {}
        for klu in meta.klu_iter():
            for date, marker in markers.items():
                date_str = date.to_str() if isinstance(date, CTime) else date
                if klu.include_sub_lv_time(date_str) and klu.time.to_str() != date_str:
                    new_marker[klu.time.to_str()] = marker
        new_marker.update(markers)

        # 获取K线数据
        kl_dict = {klu.time.to_str(): klu for klu in meta.klu_iter()}

        # 计算箭头参数
        if y_range is None:
            # 如果y_range未设置，使用当前数据的最大最小值
            y_values = [klu.high for klu in meta.klu_iter()] + [klu.low for klu in meta.klu_iter()]
            y_min = min(y_values)
            y_max = max(y_values)
            y_range = [y_min, y_max]

        y_span = y_range[1] - y_range[0]
        arrow_len = arrow_l * y_span
        arrow_h = arrow_len * arrow_h_r

        # 绘制标记
        for date, marker in new_marker.items():
            if date not in datetick_dict or date not in kl_dict:
                continue

            if x_range is not None and (datetick_dict[date] < x_range[0] or datetick_dict[date] > x_range[1]):
                continue

            # 解析标记参数
            if len(marker) == 2:
                marker_content, position = marker
                color = default_color
            else:
                marker_content, position, color = marker

            # 确定箭头方向和位置
            dir_multiplier = -1 if position == 'up' else 1
            base_y = kl_dict[date].high if position == 'up' else kl_dict[date].low

            # 绘制箭头线
            self.figure.add_trace(
                go.Scatter(
                    x=[date, date],
                    y=[base_y - arrow_len * dir_multiplier, base_y - arrow_h * dir_multiplier],
                    mode='lines',
                    line=dict(color=color, width=arrow_w),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 绘制箭头头部
            self.figure.add_trace(
                go.Scatter(
                    x=[date],
                    y=[base_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if position == 'up' else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    showlegend=False
                ),
                row=row,
                col=1
            )

            # 添加文本标签
            self.figure.add_trace(
                go.Scatter(
                    x=[date],
                    y=[base_y - arrow_len * dir_multiplier],
                    mode='text',
                    text=[marker_content],
                    textfont=dict(size=fontsize, color=color),
                    textposition='top center' if position == 'down' else 'bottom center',
                    showlegend=False
                ),
                row=row,
                col=1
            )

    def draw_demark(
        self,
        meta: CChanPlotMeta,
        row: int,
        setup_color='blue',
        countdown_color='red',
        fontsize=12,
        min_setup=9,
        max_countdown_background='yellow',
        begin_line_color: Optional[str] = 'purple',
        begin_line_style='dash',
    ):
        klu_list = list(meta.klu_iter())
        # 获取x轴范围
        x_range = None
        if hasattr(self.figure.layout, f'xaxis{row}'):
            x_range = getattr(self.figure.layout, f'xaxis{row}').range

        # 获取日期映射
        dates = [klu.time.to_str() for klu in meta.klu_iter()]
        datetick_dict = {date: idx for idx, date in enumerate(dates)}

        # 遍历K线单元
        for klu in meta.klu_iter():
            if x_range is not None and datetick_dict[klu.time.to_str()] < x_range[0]:
                continue

            # 处理setup指标
            for demark_idx in klu.demark.get_setup():
                if demark_idx['series'].idx < min_setup or not demark_idx['series'].setup_finished:
                    continue

                # 绘制TDST线
                if begin_line_color is not None and demark_idx['series'].TDST_peak is not None:
                    if demark_idx['series'].countdown is not None:
                        end_date = klu_list[demark_idx['series'].countdown.kl_list[-1].idx].time.to_str()
                    else:
                        end_date = klu_list[demark_idx['series'].kl_list[-1].idx].time.to_str()
                    start_date = klu_list[demark_idx['series'].kl_list[CDemarkEngine.SETUP_BIAS].idx].time.to_str()

                    self.figure.add_trace(
                        go.Scatter(
                            x=[start_date, end_date],
                            y=[demark_idx['series'].TDST_peak, demark_idx['series'].TDST_peak],
                            mode='lines',
                            line=dict(color=begin_line_color, dash=begin_line_style),
                            showlegend=False
                        ),
                        row=row,
                        col=1
                    )

                # 添加setup数字
                y_pos = klu.low if demark_idx['dir'] == BI_DIR.DOWN else klu.high
                self.figure.add_trace(
                    go.Scatter(
                        x=[klu.time.to_str()],
                        y=[y_pos],
                        mode='text',
                        text=[str(demark_idx['idx'])],
                        textfont=dict(size=fontsize, color=setup_color),
                        textposition='bottom center' if demark_idx['dir'] == BI_DIR.DOWN else 'top center',
                        showlegend=False
                    ),
                    row=row,
                    col=1
                )

            # 处理countdown指标
            for demark_idx in klu.demark.get_countdown():
                y_pos = klu.low if demark_idx['dir'] == BI_DIR.DOWN else klu.high
                text_props = dict(
                    x=[klu.time.to_str()],
                    y=[y_pos],
                    mode='text',
                    text=[str(demark_idx['idx'])],
                    textfont=dict(size=fontsize, color=countdown_color),
                    textposition='bottom center' if demark_idx['dir'] == BI_DIR.DOWN else 'top center',
                    showlegend=False
                )

                # 为最大countdown值添加背景
                if demark_idx['idx'] == CDemarkEngine.MAX_COUNTDOWN:
                    text_props.update({
                        'mode': 'text+markers',
                        'marker': dict(
                            size=fontsize * 1.5,
                            color=max_countdown_background,
                            symbol='square'
                        )
                    })

                self.figure.add_trace(
                    go.Scatter(**text_props),
                    row=row,
                    col=1
                )
