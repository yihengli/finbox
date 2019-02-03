import warnings
import numpy as np
import pandas as pd
import empyrical as ep
from pyecharts import Grid, Line, HeatMap
from . import pyfolio as pf
from pyfolio.utils import APPROX_BDAYS_PER_MONTH
from IPython.core.display import display, HTML
from datetime import date


class Number:
    pass


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
    logy : bool, optional
        Whether to log-scale the y-axis.
    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots
         - See timeseries.forecast_cone_bounds for more details.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    volatility_match : bool, optional
        Whether to normalize the volatility of the returns to those of the
        benchmark returns. This helps compare strategies with different
        volatilities. Requires passing of benchmark_rets.
    cone_function : function, optional
        Function to use when generating forecast probability cone.
        The function signiture must follow the form:
        def cone(in_sample_returns (pd.Series),
                 days_to_project_forward (int),
                 cone_std= (float, or tuple),
                 starting_value= (int, or float))
        See timeseries.forecast_cone_bootstrap for an example.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
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

    metrics = pf.get_rolling_returns(returns, factor_returns=factor_returns,
                                     live_start_date=live_start_date,
                                     cone_std=cone_std)

    mini_list = [np.min(metrics['cum_factor_returns']),
                 np.min(metrics['is_cum_returns'])]

    if metrics['oos_cum_returns'] is not None:
        mini_list.append(np.min(metrics['oos_cum_returns']))
    if metrics['cone_bounds'] is not None:
        mini_list.append(min(np.min(metrics['cone_bounds'])))
    yaxis_min = np.round(min(mini_list) - 0.05, 2)

    u_list, d_list = [], []
    if metrics['cone_bounds'] is not None:
        for cone_value in metrics['cone_bounds'].columns.tolist():
            if cone_value > 0:
                u_list.append(metrics['cone_bounds'][cone_value])
            else:
                d_list.append(metrics['cone_bounds'][cone_value])

    attr = metrics['cum_factor_returns'].index.strftime('%Y-%m-%d').tolist()
    line = Line("Cumulative Returns")

    # Set Up Color Variables
    area_color = PlottingConfig.CI_AREA_COLOR

    # Add Benchmark Line
    line.add("Benchmark", attr,
             np.round(metrics['cum_factor_returns'], 4).tolist(),
             is_datazoom_show=True, datazoom_range=[0, 100],
             **PlottingConfig().BENCH_KWARGS)
    line._option['color'][0] = 'grey'

    # Add Confidence Interval
    ci_index = 1
    for u, d in zip(u_list, d_list):
        line.add("CI", attr, np.round(d, 4).tolist(), line_opacity=0,
                 is_stack=True, is_label_emphasis=False, is_symbol_show=False)
        line.add("CI", attr, np.round(u-d, 4).tolist(), line_opacity=0,
                 is_label_emphasis=False, area_color=area_color,
                 area_opacity=0.15, is_stack=True, is_symbol_show=False)
        line._option['series'][ci_index]['stack'] = 'CI{}'.format(ci_index)
        line._option['series'][ci_index+1]['stack'] = 'CI{}'.format(ci_index)
        ci_index += 2

    # Add OOS Line
    if metrics['oos_cum_returns'] is not None:
        line.add("Out Of Sample", attr,
                 np.round(metrics['oos_cum_returns'], 4).tolist(),
                 **PlottingConfig().LINE_KWARGS)
        line._option['color'][2] = PlottingConfig.PINK_RED
        line._option['color'][3] = PlottingConfig.GREEN
    else:
        line._option['color'][1] = PlottingConfig.GREEN

    # Add IS Line
    line.add("In Sample", attr,
             np.round(metrics['is_cum_returns'], 4).tolist(),
             tooltip_formatter=tooltip_format, yaxis_min=yaxis_min,
             **PlottingConfig().LINE_KWARGS)

    line._option['legend'][0]['data'] = [
        'Benchmark', 'In Sample', 'Out Of Sample', 'CI']

    # Set up colors
    if metrics['oos_cum_returns'] is None:
        line._option['color'][1] = PlottingConfig.GREEN
    elif metrics['oos_cum_returns'] is not None and \
            metrics['cone_bounds'] is None:
        line._option['color'][1] = PlottingConfig.PINK_RED
        line._option['color'][2] = PlottingConfig.GREEN
    else:
        line._option['color'][2] = PlottingConfig.PINK_RED
        line._option['color'][3] = PlottingConfig.GREEN

    return line


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
