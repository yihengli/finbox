from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from pyecharts import Line
from pyecharts.chart import Chart

from .. import pyfolio as pf
from ._plot_meta import PlottingConfig


class Number:
    pass


def plot_interactive_rolling_returns(returns,
                                     factor_returns=None,
                                     live_start_date=None,
                                     cone_std=None):
    """
    Plots cumulative rolling returns versus some benchmarks'.

    Backtest returns are in green, and out-of-sample (live trading)
    returns are in red.

    Additionally, a non-parametric cone plot may be added to the
    out-of-sample returns region.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series, optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    live_start_date : datetime, optional
        The date when the strategy began live trading, after
        its backtest period. This date should be normalized.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.

    Returns
    -------
    pyecharts.chart.Chart
        The axes that were plotted on.
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

        chart._option['color'][color_index] = area_color
        return color_index + 1

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
