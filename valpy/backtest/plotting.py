from pyecharts import Line
from . import pyfolio as pf
import numpy as np


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
