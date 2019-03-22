from typing import List, Optional, Union, Dict

import matplotlib as mpl
import pandas as pd
from matplotlib.axes import Axes
from pyecharts.chart import Chart

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


def plot_interesting_periods(returns: pd.Series,
                             factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                             periods: Optional[List[Dict]] = None,
                             override: bool = False,
                             chart_type: str = 'matplotlib') -> Union[Axes, Chart]:  # noqa
    """
    Plot out the cumulative returns over several selected periods such as 08
    financial crisis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
    factor_returns : Union[None, pd.Series, List[pd.Series]], optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
    periods : Optional[List[Dict]], optional
        A list of dict which describes the periods' start and end dates:
        {
            'Example Name' : (pd.Timestamp('20080101'),
                              pd.Timestamp('20080301))
        }
    override : bool, optional
        If True, this function will only use given periods to plot out returns,
        otherwise, the given periods will be appended to the original periods.
    chart_type : str, optional
        Plot Engine (the default is 'matplotlib', otherwise 'echarts')

    Returns
    -------
    Tuple[Union[mpl.axes.Axes, pyecharts.chart.Chart], Union[None, int]]
        Either matplotlib plots or Echarts plot object, the second output
        will only be used by the report object to decide its allocated heights
    """
    if chart_type == 'matplotlib':
        return _pm.plot_interesting_periods(returns, factor_returns,
                                            periods, override)
    elif chart_type == 'echarts':
        return _pe.plot_interactive_interesting_periods(
            returns, factor_returns, periods, override)
    else:
        raise NotImplementedError('`chart_type` can only be `matplotlib` '
                                  'or `echarts`')
