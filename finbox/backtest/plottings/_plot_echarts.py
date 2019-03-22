import warnings
from datetime import date
from typing import Dict, List, Optional, Tuple, Union

import empyrical as ep
import numpy as np
import pandas as pd
import seaborn as sns
from pyecharts import Grid, HeatMap, Line
from pyecharts.chart import Chart
from pyfolio import pos, timeseries
from pyfolio.utils import APPROX_BDAYS_PER_MONTH

from .. import pyfolio as pf
from ._plot_meta import PlottingConfig


class Number:
    pass


def plot_interactive_rolling_returns(returns: pd.Series,
                                     factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                                     live_start_date: Optional[str] = None,
                                     cone_std: Optional[List[float]] = None) -> Chart:  # noqa
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.
    """

    def tooltip_format(params):
        def get_color(color):
            return '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' + color + '"></span>'  # noqa E501

        res = params[0].axisValue
        for i in params:
            if i.seriesName != 'CI' and not Number.isNaN(i.data[1]):
                value = (i.data[1] - 1) * 100
                res = res + '<br>' + \
                    get_color(i.color) + ' ' + i.seriesName + \
                    ': ' + value.toFixed(2) + '%'

        return res

    def get_ymin(metrics: Dict) -> float:
        """
        Try to find the global minimum to set up the y_min value, as echart
        always starts from 0 on the yAxis
        """
        mini_list = [np.min(metrics['is_cum_returns'])]
        if metrics['cum_factor_returns'] is not None:
            mini_list.append(np.min(metrics['cum_factor_returns']))
        if metrics['oos_cum_returns'] is not None:
            mini_list.append(np.min(metrics['oos_cum_returns']))
        if metrics['cone_bounds'] is not None:
            mini_list.append(min(np.min(metrics['cone_bounds'])))
        return np.round(min(mini_list) - 0.05, 2)

    def get_CI(metrics: Dict) -> Tuple[List, List]:
        """
        If prediction interval exists, construct the upper and lower bounds
        """
        u_list, d_list = [], []
        if metrics['cone_bounds'] is not None:
            for cone_value in metrics['cone_bounds'].columns.tolist():
                if cone_value > 0:
                    u_list.append(metrics['cone_bounds'][cone_value])
                else:
                    d_list.append(metrics['cone_bounds'][cone_value])
        return u_list, d_list

    def get_attr(metrics: Dict) -> List[str]:
        """
        Get the xAxis datetime str series
        """
        # Only In-Sample series is garanteed to have values
        return metrics['is_cum_returns'].index.strftime('%Y-%m-%d').tolist()

    def append_benchmarks(metrics: Dict, chart: Chart,
                          color_index: int) -> Tuple[int, List[str]]:
        """
        Set up all benchmark lines given various input types, and return the
        benchmark names as legends
        """
        # Add Benchmark Line
        benchmark_lines = []
        if metrics['cum_factor_returns'] is not None and\
                isinstance(metrics['cum_factor_returns'], pd.Series):
            benchmark_lines = [metrics['cum_factor_returns']]
        elif metrics['cum_factor_returns'] is not None and\
                isinstance(metrics['cum_factor_returns'], list):
            benchmark_lines = metrics['cum_factor_returns']

        bench_colors = sns.color_palette(
            'Greys', len(benchmark_lines)).as_hex()
        for i, benchmark in enumerate(benchmark_lines):
            chart.add(benchmark.name, get_attr(metrics),
                      np.round(benchmark, 4).tolist(),
                      line_color=bench_colors[i],
                      is_datazoom_show=True, datazoom_range=[0, 100],
                      **PlottingConfig().BENCH_KWARGS)
            chart._option['color'][color_index] = bench_colors[i]
            color_index += 1

        return color_index, [bench.name for bench in benchmark_lines]

    def append_CI(u_list: List[float], d_list: List[float],
                  chart: Chart, color_index: int) -> None:
        """
        Given the calculated confidence interval range, plot them on chart
        """
        ci_index = 0
        area_color = PlottingConfig.CI_AREA_COLOR
        for u, d in zip(u_list, d_list):
            chart.add("CI", get_attr(metrics), np.round(d, 4).tolist(),
                      line_opacity=0, is_stack=True, is_label_emphasis=False,
                      is_symbol_show=False)
            chart.add("CI", get_attr(metrics), np.round(u-d, 4).tolist(),
                      line_opacity=0, is_label_emphasis=False,
                      area_color=area_color, area_opacity=0.15, is_stack=True,
                      is_symbol_show=False)
            chart._option['series'][ci_index]['stack'] = 'CI{}'.format(
                ci_index)
            chart._option['series'][ci_index +
                                    1]['stack'] = 'CI{}'.format(ci_index)
            ci_index += 2

        if len(u_list) > 0:
            chart._option['color'][color_index] = area_color
            return color_index + 1
        else:
            return color_index

    # Get calculated returns series

    metrics = pf.get_rolling_returns(returns, factor_returns=factor_returns,
                                     live_start_date=live_start_date,
                                     cone_std=cone_std)

    yaxis_min = get_ymin(metrics)
    u_list, d_list = get_CI(metrics)

    # Initalize echart object
    color_index = 0
    line = Line("Cumulative Returns")

    color_index = append_CI(u_list, d_list, line, color_index)
    color_index, bench_names = append_benchmarks(metrics, line, color_index)

    # Add OOS line
    if metrics['oos_cum_returns'] is not None:
        line.add("Out Of Sample", get_attr(metrics),
                 np.round(metrics['oos_cum_returns'], 4).tolist(),
                 **PlottingConfig().LINE_KWARGS)
        line._option['color'][color_index] = PlottingConfig().PINK_RED
        color_index += 1

    # Add IS Line
    line.add("In Sample", get_attr(metrics),
             np.round(metrics['is_cum_returns'], 4).tolist(),
             tooltip_formatter=tooltip_format, yaxis_min=yaxis_min,
             **PlottingConfig().LINE_KWARGS)
    line._option['color'][color_index] = PlottingConfig().GREEN
    color_index += 1

    # Change legends
    line._option['legend'][0]['data'] = bench_names + \
        ['In Sample', 'Out Of Sample', 'CI']

    return line


def plot_interactive_rolling_sharpes(returns: pd.Series,
                                     factor_returns: Union[None, pd.Series, List[pd.Series]] = None) -> Chart:  # noqa
    """
    Plots the rolling Sharpe ratio versus date.
    """
    line = Line("Rolling Sharpe Ratio (6 Months)")

    sharpe = pf.get_rolling_sharpe(returns)

    # ratio used to determine the default zoom in view
    valid_ratio = np.round(sharpe.count() / sharpe.shape[0], 3)
    attr = sharpe.index.strftime("%Y-%m-%d")

    # benchmark handler
    benchmarks = []
    if isinstance(factor_returns, pd.Series):
        benchmarks = [factor_returns]
    elif isinstance(factor_returns, list):
        benchmarks = factor_returns

    color_index = 0
    if len(benchmarks) > 0:
        colors = sns.color_palette('Greys', len(benchmarks)).as_hex()
        for bench, color in zip(benchmarks, colors):
            bench_sharpe = pf.get_rolling_sharpe(bench)
            line.add(bench.name, attr, np.round(bench_sharpe, 3).tolist(),
                     is_datazoom_show=True, mark_line=["average"],
                     datazoom_range=[(1 - valid_ratio) * 100, 100],
                     **PlottingConfig.BENCH_KWARGS)
            line._option['color'][color_index] = color
            line._option["series"][color_index]["markLine"]["lineStyle"] = \
                {"width": 1}
            color_index += 1

    line.add("Strategy", attr, np.round(sharpe, 3).tolist(),
             mark_line=["average"], **PlottingConfig.LINE_KWARGS)
    line._option['color'][color_index] = PlottingConfig.ORANGE
    line._option["series"][color_index]["markLine"]["lineStyle"] = {"width": 2}

    return line


def plot_interactive_drawdown_underwater(returns: pd.Series, top: int = 5) -> Chart:  # noqa
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.
    """
    # Get underwater data and max drawdown tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
        df_drawdowns = pf.timeseries.gen_drawdown_table(returns, top=top)
        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -100 * ((running_max - df_cum_rets) / running_max)

    line = Line("Top 5 Drawdowns")
    attr = df_cum_rets.index.strftime("%Y-%m-%d")

    # Plot out the returns
    line.add("Cumulative Returns", attr, np.round(df_cum_rets, 3).tolist(),
             line_color='#fff', is_datazoom_show=True, datazoom_range=[0, 100],
             yaxis_min=0.7, datazoom_xaxis_index=[0, 1],
             **PlottingConfig.BENCH_KWARGS)

    line._option["color"][0] = PlottingConfig.ORANGE
    line._option["series"][0]["markArea"] = {"data": []}

    # Highligh each drawdown area
    color_sets = ['#355C7D', '#6C5B7B', '#C06C84', '#F67280', '#F8B195']
    for i, (peak, recovery) in df_drawdowns[
            ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = returns.index[-1]
        peak = peak.strftime("%Y-%m-%d")
        recovery = recovery.strftime("%Y-%m-%d")

        line._option["series"][0]["markArea"]["data"].append(
            [{
                "name": str(i+1),
                "xAxis": peak,
                "itemStyle": {
                    "opacity": 0.9,
                    "color": color_sets[i]
                }
            }, {
                "xAxis": recovery
            }]
        )

    # Separately plot out underwaters
    line2 = Line("Underwater Plot", title_top="50%")
    line2.add("DrawDown", attr, np.round(underwater, 3).tolist(),
              is_datazoom_show=True, area_color=PlottingConfig.ORANGE,
              yaxis_formatter='%', area_opacity=0.5, datazoom_range=[0, 100],
              legend_top="50%", **PlottingConfig.BENCH_KWARGS)

    grid = Grid()
    grid.add(line, grid_bottom="57%")
    grid.add(line2, grid_top="57%")

    grid._option["color"][1] = PlottingConfig.ORANGE
    grid._option["axisPointer"] = {"link": {"xAxisIndex": 'all'}}

    return grid


def plot_interactive_rolling_betas(returns: pd.Series,
                                   factor_returns: pd.Series) -> Chart:
    """
    Plots the rolling 6-month and 12-month beta versus date.
    """
    if factor_returns is None:
        raise Exception('`factor_returns` must be provided when calculating '
                        'betas')
    elif isinstance(factor_returns, list):
        factor_returns = factor_returns[0]

    rb_1 = pf.timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    rb_2 = pf.timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)

    attr = rb_1.index.strftime("%Y-%m-%d")
    valid_ratio = np.round(rb_2.count() / rb_2.shape[0], 3)

    line = Line("Rolling Betas")
    line.add("6-Month", attr, np.round(rb_1, 3).tolist(),
             is_datazoom_show=True, mark_line=["average"],
             datazoom_range=[(1 - valid_ratio) * 100, 100],
             **PlottingConfig.LINE_KWARGS)
    line.add("12-Month", attr, np.round(rb_2, 3).tolist(),
             mark_line=["average"], **PlottingConfig.BENCH_KWARGS)

    line._option['color'][0] = PlottingConfig.CI_AREA_COLOR
    line._option['color'][1] = "grey"
    line._option["series"][0]["markLine"]["lineStyle"] = {"width": 2}

    return line


def plot_interactive_rolling_vol(returns: pd.Series,
                                 factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                                 rolling_window: int = APPROX_BDAYS_PER_MONTH * 6) -> Chart:  # noqa
    """
    Plots the rolling volatility versus date.
    """

    rolling_vol_ts = pf.timeseries.rolling_volatility(returns, rolling_window)
    attr = rolling_vol_ts.index.strftime("%Y-%m-%d")
    valid_ratio = np.round(rolling_vol_ts.count() / rolling_vol_ts.shape[0], 3)

    line = Line("Rolling Volatility (6-Month)")
    line.add("Volatility", attr, np.round(rolling_vol_ts, 3).tolist(),
             is_datazoom_show=True, mark_line=["average"],
             datazoom_range=[(1 - valid_ratio) * 100, 100],
             **PlottingConfig.LINE_KWARGS)
    line._option['color'][0] = PlottingConfig.ORANGE
    line._option["series"][0]["markLine"]["lineStyle"] = {"width": 2}

    if isinstance(factor_returns, pd.Series):
        factor_returns = [factor_returns]

    color_index = 1
    if isinstance(factor_returns, list):
        colors = sns.color_palette('Greys', len(factor_returns))
        for bench, color in zip(factor_returns, colors):
            bench_vol = pf.timeseries.rolling_volatility(bench, rolling_window)
            line.add(bench.name, attr,
                     np.around(bench_vol[:len(attr)], 3).tolist(),
                     mark_line=["average"], **PlottingConfig.BENCH_KWARGS)
            line._option['color'][color_index] = 'grey'
            line._option["series"][color_index]["markLine"]["lineStyle"] = \
                {"width": 1}
            color_index += 1

    return line


def plot_interactive_monthly_heatmap(returns: pd.Series) -> Chart:
    """
    Plots a heatmap of returns by month.
    """
    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    monthly_ret_table = np.round(monthly_ret_table * 100, 2)
    lim = lim = max(np.max(np.abs(monthly_ret_table)))

    y_axis = [date(1000, i, 1).strftime('%b') for i in range(1, 13)]
    x_axis = [str(y) for y in monthly_ret_table.index.tolist()]
    data = data = [[x_axis[i], y_axis[j], monthly_ret_table.values[i][j]]
                   for i in range(monthly_ret_table.shape[0])
                   for j in range(monthly_ret_table.shape[1])]

    heatmap = HeatMap("Monthly Returns")
    heatmap.add(
        "Monthly Returns",
        x_axis,
        y_axis,
        data,
        is_visualmap=True,
        is_datazoom_show=True,
        datazoom_orient='horizontal',
        datazoom_range=[0, 100],
        visual_range=[-lim, lim],
        visual_text_color="#000",
        visual_range_color=['#D73027', '#FFFFBF', '#1A9641'],
        visual_orient="vertical",
        is_toolbox_show=False,
        is_label_show=True,
        label_pos="inside",
        label_text_color="black",
        tooltip_formatter="{c}%"
    )
    return heatmap


def plot_interactive_exposures(returns: pd.Series,
                               positions: pd.DataFrame) -> Chart:
    """
    Plots a cake chart of the long and short exposure.
    """

    pos_no_cash = positions.drop('cash', axis=1)

    l_exp = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
    s_exp = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
    net_exp = pos_no_cash.sum(axis=1) / positions.sum(axis=1)

    line = Line("Exposure")
    attr = l_exp.index.strftime("%Y-%m-%d")

    line.add("Long", attr, np.round(l_exp, 3).tolist(),
             is_datazoom_show=True, datazoom_range=[0, 100],
             is_step=True, area_opacity=0.7, tooltip_trigger="axis",
             is_symbol_show=False, line_opacity=0)
    line.add("Short", attr, np.round(s_exp, 3).tolist(),
             is_datazoom_show=True, datazoom_range=[0, 100],
             is_step=True, area_opacity=0.7, tooltip_trigger="axis",
             is_symbol_show=False, line_opacity=0)
    line.add("Net", attr, np.round(net_exp, 3).tolist(),
             is_datazoom_show=True, datazoom_range=[0, 100],
             tooltip_trigger="axis", is_symbol_show=False)

    line._option['color'][0] = PlottingConfig.GREEN
    line._option['color'][1] = PlottingConfig.ORANGE
    line._option['color'][2] = 'black'

    return line


def plot_interactive_exposures_by_asset(positions: pd.DataFrame) -> Chart:
    """
    plots the exposures of the held positions of all time.
    """

    pos_alloc = pos.get_percent_alloc(positions)
    pos_alloc_no_cash = pos_alloc.drop('cash', axis=1)

    attr = pos_alloc_no_cash.index.strftime("%Y-%m-%d")
    line = Line("Exposures By Asset")

    for col in pos_alloc_no_cash.columns:
        line.add(col, attr, np.round(pos_alloc_no_cash[col], 2).tolist(),
                 is_more_utils=True, is_datazoom_show=True,
                 line_width=2, line_opacity=0.7, is_symbol_show=False,
                 datazoom_range=[0, 100], tooltip_trigger="axis")

    return line


def plot_interactive_gross_leverage(positions: pd.DataFrame) -> Chart:
    """TODO: NotImplementedYet"""

    gl = timeseries.gross_lev(positions)
    line = Line("Gross Leverage")

    line.add("Gross Leverage", gl.index.strftime("%Y-%m-%d"),
             np.round(gl, 3).tolist(), is_datazoom_show=True,
             mark_line=["average"], datazoom_range=[0, 100],
             **PlottingConfig.LINE_KWARGS)

    line._option['color'][0] = PlottingConfig.GREEN
    line._option["series"][0]["markLine"]["lineStyle"] = {"width": 2}

    return line


def plot_interactive_interesting_periods(returns: pd.Series,
                                         benchmark_rets: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                                         periods: Optional[List[Dict]] = None,
                                         override: bool = False) -> Tuple[Chart, int]:  # noqa
    """
    plots the exposures of positions of all time.
    """
    rets_interesting = pf.extract_interesting_date_ranges(
        returns, periods, override)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    # returns = clip_returns_to_benchmark(returns, benchmark_rets)

    if isinstance(benchmark_rets, pd.Series):
        benchmark_rets = [benchmark_rets]

    if isinstance(benchmark_rets, list):
        bmark_int = []
        for bmark in benchmark_rets:
            bmark_int.append(
                pf.extract_interesting_date_ranges(bmark, periods, override))

    num_plots = len(rets_interesting)
    num_rows = int((num_plots + 1) / 2.0)
    height = num_rows*180

    grid = Grid(height=height)

    up_maxi = 3
    down_maxi = 3
    gap = 2

    margin_step = np.round(((100 - up_maxi - down_maxi - gap * (num_rows - 1))
                            / num_rows), 1)

    t_margins = [up_maxi + i * (margin_step + gap) for i in range(num_rows)]
    b_margins = [down_maxi + i * (margin_step + gap)
                 for i in range(num_rows)][::-1]

    for i, (name, rets_period) in enumerate(rets_interesting.items()):
        top_margin = t_margins[i // 2]
        bottom_margin = b_margins[i // 2]
        left_margin = 55 if i % 2 == 1 else 5
        right_margin = 0 if i % 2 == 1 else 55

        if i != 0:
            is_legend_show = False
        else:
            is_legend_show = True

        line = Line(name, title_top="{}%".format(top_margin),
                    title_pos="{}%".format(left_margin))
        cum_rets = ep.cum_returns(rets_period)
        line.add("Algo", cum_rets.index.strftime("%Y-%m-%d"),
                 np.round(cum_rets, 3).tolist(), is_splitline_show=False,
                 is_legend_show=is_legend_show, is_symbol_show=False,
                 tooltip_trigger='axis', line_width=2, line_opacity=0.8)
        if benchmark_rets is not None:
            for bmark, b in zip(bmark_int, benchmark_rets):
                cum_bech = ep.cum_returns(bmark[name][cum_rets.index])
                line.add(b.name, cum_bech.index.strftime("%Y-%m-%d"),
                         np.round(cum_bech, 3).tolist(),
                         is_splitline_show=False,
                         is_legend_show=is_legend_show, is_symbol_show=False,
                         tooltip_trigger='axis', line_opacity=0.8)

        grid.add(line,
                 grid_top="{}%".format(top_margin + gap),
                 grid_bottom="{}%".format(bottom_margin),
                 grid_left="{}%".format(left_margin),
                 grid_right="{}%".format(right_margin))

    grid._option["color"][0] = PlottingConfig.GREEN
    grid._option["color"][1] = "grey"

    if benchmark_rets is not None:
        colors = sns.color_palette('Greys', len(benchmark_rets)).as_hex()
        for i in range(1, len(benchmark_rets) + 1):
            grid._option["color"][i] = colors[i-1]

    return grid, height
