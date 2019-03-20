import warnings
from datetime import date
from enum import Enum
from typing import List, Optional, Union

import empyrical as ep
import numpy as np
import pandas as pd
from IPython.core.display import HTML, display
from matplotlib.axes import Axes
from pyecharts import Grid, HeatMap, Line
from pyecharts.chart import Chart
from pyfolio import pos, timeseries
from pyfolio.utils import APPROX_BDAYS_PER_MONTH, clip_returns_to_benchmark

from . import pyfolio as pf
from .plottings import _plot_echarts as _pe


class ChartType(Enum):
    matplotlib = 'matplotlib'
    echarts = 'echarts'


class PlottingConfig:

    # Colors
    CI_AREA_COLOR = '#0082c8'
    GREEN = '#3cb44b'
    PINK_RED = '#e6194b'
    ORANGE = '#f58231'

    # Line Styles
    LINE_KWARGS = {
        "line_width": 3,
        "line_opacity": 0.8,
        "tooltip_trigger": "axis",
        "is_symbol_show": False
    }

    BENCH_KWARGS = {
        "line_width": 2,
        "line_opacity": 0.8,
        "tooltip_trigger": "axis",
        "is_symbol_show": False
    }


def print_table(table, float_format='{0:.2f}'.format, formatters=None,
                jupyter=False, header_rows=None):
    html = table.to_html(float_format=float_format, formatters=formatters)

    if header_rows is not None:
        # Count the number of columns for the text to span
        n_cols = html.split('<thead>')[1].split('</thead>')[0].count('<th>')

        # Generate the HTML for the extra rows
        rows = ''
        for name, value in header_rows.items():
            rows += ('\n    <tr style="text-align: right;"><th>%s</th>' +
                     '<td colspan=%d>%s</td></tr>') % (name, n_cols, value)

        # Inject the new HTML
        html = html.replace('<thead>', '<thead>' + rows)

    if jupyter:
        display(HTML(html))
    else:
        html = html.replace(
            '<table border="1" class="dataframe">',
            '<table class="table table-sm table-hover table-striped">'
        )
        html = html.replace(' style="text-align: right;', '')

        return html


def plot_rolling_returns(returns: pd.Series,
                         factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                         live_start_date: Optional[str] = None,
                         cone_std: Optional[pd.DataFrame] = None,
                         chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
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
    factor_returns : Union[None, pd.Series, List[pd.Series]], optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
         - If a list of benchmarks are given, they will be plotted in order
    cone_std : Optional[pd.DataFrame], optional
        The calculated prediction intervals

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == ChartType['matplotlib']:
        pass
    elif chart_type == ChartType['echarts']:
        return _pe.plot_interactive_rolling_returns(
            returns, factor_returns, live_start_date, cone_std)


def plot_interactive_rolling_sharpes(returns,
                                     factor_returns):
    line = Line("Rolling Sharpe Ratio (6 Months)")

    sharpe = pf.get_rolling_sharpe(returns)
    bench_sharpe = pf.get_rolling_sharpe(factor_returns)
    valid_ratio = np.round(sharpe.count() / sharpe.shape[0], 3)

    attr = sharpe.index.strftime("%Y-%m-%d")

    line.add("Benchmark", attr, np.round(bench_sharpe, 3).tolist(),
             is_datazoom_show=True, mark_line=["average"],
             datazoom_range=[(1 - valid_ratio) * 100, 100],
             **PlottingConfig.BENCH_KWARGS)
    line.add("Strategy", attr, np.round(sharpe, 3).tolist(),
             mark_line=["average"], **PlottingConfig.LINE_KWARGS)

    line._option['color'][0] = 'grey'
    line._option['color'][1] = PlottingConfig.ORANGE

    line._option["series"][0]["markLine"]["lineStyle"] = {"width": 1}
    line._option["series"][1]["markLine"]["lineStyle"] = {"width": 2}

    return line


def plot_interactive_drawdown_underwater(returns, top=5):
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
        df_drawdowns = pf.timeseries.gen_drawdown_table(returns, top=top)
        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -100 * ((running_max - df_cum_rets) / running_max)

    line = Line("Top 5 Drawdowns")
    attr = df_cum_rets.index.strftime("%Y-%m-%d")

    line.add("Cumulative Returns", attr, np.round(df_cum_rets, 3).tolist(),
             line_color='#fff', is_datazoom_show=True, datazoom_range=[0, 100],
             yaxis_min=0.7, datazoom_xaxis_index=[0, 1],
             **PlottingConfig.BENCH_KWARGS)

    line._option["color"][0] = PlottingConfig.ORANGE
    line._option["series"][0]["markArea"] = {"data": []}

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


def plot_interactive_rolling_betas(returns, factor_returns):
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


def plot_interactive_rolling_vol(returns, factor_returns,
                                 rolling_window=APPROX_BDAYS_PER_MONTH * 6):
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

    if factor_returns is not None:
        rolling_vol_ts_factor = pf.timeseries.rolling_volatility(
            factor_returns, rolling_window)
        line.add("Benchmark Volatiltiy", attr,
                 np.around(rolling_vol_ts_factor[:len(attr)], 3).tolist(),
                 mark_line=["average"], **PlottingConfig.BENCH_KWARGS)
        line._option['color'][1] = 'grey'
        line._option["series"][1]["markLine"]["lineStyle"] = {"width": 1}

    return line


def plot_interactive_monthly_heatmap(returns):

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


def plot_interactive_exposures(returns, positions):

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


def plot_interactive_exposures_by_asset(positions):
    pos_alloc = pos.get_percent_alloc(positions)
    pos_alloc_no_cash = pos_alloc.drop('cash', axis=1)

    attr = pos_alloc_no_cash.index.strftime("%Y-%m-%d")

    line = Line("Exposures By Asset")

    for col in pos_alloc_no_cash.columns:
        line.add(col, attr, np.round(pos_alloc_no_cash[col], 2).tolist(),
                 is_more_utils=True, is_datazoom_show=True,
                 line_width=2, line_opacity=0.7,
                 datazoom_range=[0, 100], tooltip_trigger="axis")

    return line


def plot_interactive_gross_leverage(positions):

    gl = timeseries.gross_lev(positions)
    line = Line("Gross Leverage")

    line.add("Gross Leverage", gl.index.strftime("%Y-%m-%d"),
             np.round(gl, 3).tolist(), is_datazoom_show=True,
             mark_line=["average"], datazoom_range=[0, 100],
             **PlottingConfig.LINE_KWARGS)

    line._option['color'][0] = PlottingConfig.GREEN
    line._option["series"][0]["markLine"]["lineStyle"] = {"width": 2}

    return line


def plot_interactive_interesting_periods(returns, benchmark_rets=None,
                                         periods=None, override=False):

    rets_interesting = pf.extract_interesting_date_ranges(
        returns, periods, override)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    returns = clip_returns_to_benchmark(returns, benchmark_rets)
    bmark_interesting = pf.extract_interesting_date_ranges(
        benchmark_rets, periods, override)

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
            cum_bech = ep.cum_returns(bmark_interesting[name])
            line.add("Bench", cum_bech.index.strftime("%Y-%m-%d"),
                     np.round(cum_bech, 3).tolist(), is_splitline_show=False,
                     is_legend_show=is_legend_show, is_symbol_show=False,
                     tooltip_trigger='axis', line_opacity=0.8)

        grid.add(line,
                 grid_top="{}%".format(top_margin + gap),
                 grid_bottom="{}%".format(bottom_margin),
                 grid_left="{}%".format(left_margin),
                 grid_right="{}%".format(right_margin))

    grid._option["color"][0] = PlottingConfig.GREEN
    grid._option["color"][1] = "grey"

    return grid, height
