import warnings
from typing import List, Optional, Union

import empyrical as ep
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pyecharts import Grid, Line
from pyecharts.chart import Chart
from pyfolio import timeseries
from pyfolio.utils import clip_returns_to_benchmark

from . import pyfolio as pf
from .plottings import _plot_echarts as _pe
from .plottings import _plot_matplotlib as _pm

mpl.style.use('seaborn-paper')


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
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_rolling_returns(returns, factor_returns,
                                        live_start_date, cone_std)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_rolling_returns(returns, factor_returns,
                                                    live_start_date, cone_std)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_rolling_sharpes(returns: pd.Series,
                         factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                         chart_type: str = 'matplotlib') -> Union[Axes, Chart]:
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : Union[None, pd.Series, List[pd.Series]], optional
        Daily noncumulative returns of the benchmark factor for which the
        benchmark rolling Sharpe is computed. Usually a benchmark such as
        market returns.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_rolling_sharpes(returns, factor_returns)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_rolling_sharpes(returns, factor_returns)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_drawdown_underwater(returns: pd.Series, top: int = 5,
                             chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    top : int, optional
        Only analyze the top N drwadowns as highlighted areas
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_drawdown_underwater(returns, top)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_drawdown_underwater(returns, top)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_rolling_betas(returns: pd.Series,
                       factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                       chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
    """
    Plots the rolling 6-month and 12-month beta versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : Union[None, pd.Series, List[pd.Series]], optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_rolling_betas(returns, factor_returns)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_rolling_betas(returns, factor_returns)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_rolling_vol(returns: pd.Series,
                     factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                     chart_type: str = 'matplotlib') -> Union[Axes, Chart]:
    """
    Plots the rolling volatility versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : Union[None, pd.Series, List[pd.Series]], optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_rolling_vol(returns, factor_returns)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_rolling_vol(returns, factor_returns)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_monthly_heatmap(returns: pd.Series, chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
    """
    Plots a heatmap of returns by month.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_monthly_heatmap(returns)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_monthly_heatmap(returns)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_exposures(returns: pd.Series, positions: pd.DataFrame,
                   chart_type: str = 'matplotlib') -> Union[Axes, Chart]:
    """
    Plots a cake chart of the long and short exposure.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    positions : pd.DataFrame
        Portfolio allocation of positions.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_exposures(returns, positions)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_exposures(returns, positions)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


def plot_exposures_by_assets(positions: pd.DataFrame, chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
    """
    plots the exposures of positions of all time.

    Parameters
    ----------
    positions : pd.DataFrame
        Portfolio allocation of positions.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Union[mpl.axes.Axes, pyecharts.chart.Chart]
        Either matplotlib plots or Echarts plot object
    """
    if chart_type == 'matplotlib':
        return _pm.plot_exposures_by_assets(positions)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_exposures_by_asset(positions)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')


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
