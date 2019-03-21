import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from pyfolio import plotting, utils

from .. import pyfolio as pf


def plot_rolling_returns(returns: pd.Series,
                         factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                         live_start_date: Optional[str] = None,
                         cone_std: Optional[List[float]] = None,
                         ax: Optional[Axes] = None) -> Axes:  # noqa
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
    matplotlib.axes.Axes
        The axes that were plotted on.
    """

    metrics = pf.get_rolling_returns(returns, factor_returns=factor_returns,
                                     live_start_date=live_start_date,
                                     cone_std=cone_std)

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative returns')

    if isinstance(metrics['cum_factor_returns'], pd.Series):
        metrics['cum_factor_returns'] = [metrics['cum_factor_returns']]

    if metrics['cum_factor_returns'] is not None:
        bench_colors = sns.color_palette('Greys', len(metrics['cum_factor_returns'])).as_hex()  # noqa
        for factor, color in zip(metrics['cum_factor_returns'], bench_colors):
            factor.plot(lw=2, color=color, label=factor.name,
                        alpha=0.60, ax=ax)

    metrics['is_cum_returns'].plot(lw=3, color='forestgreen', alpha=0.6,
                                   label='Strategy', ax=ax)
    if metrics['oos_cum_returns'] is not None:
        metrics['oos_cum_returns'].plot(lw=4, color='red', alpha=0.6,
                                        label='Live', ax=ax)

    if metrics['cone_bounds'] is not None:
        for std in cone_std:
            ax.fill_between(metrics['cone_bounds'].index,
                            metrics['cone_bounds'][float(std)],
                            metrics['cone_bounds'][float(-std)],
                            color='steelblue', alpha=0.5)

    ax.legend(loc='best', frameon=True, framealpha=0.5)
    ax.axhline(1.0, linestyle='--', color='black', lw=2)

    return ax


def plot_rolling_sharpes(returns: pd.Series,
                         factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                         ax: Optional[Axes] = None) -> Axes:
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('')

    sharpe = pf.get_rolling_sharpe(returns)
    sharpe.name = 'Strategy'
    sharpe.plot(alpha=.7, lw=3, color='orangered', ax=ax)

    # benchmark handler
    benchmarks, bench_means = [], []
    if isinstance(factor_returns, pd.Series):
        benchmarks = [factor_returns]
    elif isinstance(factor_returns, list):
        benchmarks = factor_returns

    if len(benchmarks) > 0:
        colors = sns.color_palette('Greys', len(benchmarks)).as_hex()
        for bench, color in zip(benchmarks, colors):
            bench_sharpe = pf.get_rolling_sharpe(bench)
            bench_sharpe.plot(alpha=.7, lw=3, color=color, ax=ax)
            bench_means.append(bench_sharpe.mean())

        for u, color in zip(bench_means, colors):
            ax.axhline(u, color=color, linestyle=':', lw=2)

    ax.axhline(sharpe.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=3)

    ax.legend(loc='best', frameon=True, framealpha=0.5)
    return ax


def plot_drawdown_underwater(returns: pd.Series, top: int = 5,
                             axes: Optional[Tuple[Axes, Axes]] = None) -> Axes:
    """
    Plots how far underwaterr returns are over time, or plots current
    drawdown vs. date.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)

    # Get underwater data and max drawdown tables
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotting.plot_drawdown_periods(returns, top, axes[0])
        plotting.plot_drawdown_underwater(returns, axes[1])

    fig.tight_layout()
    return axes
