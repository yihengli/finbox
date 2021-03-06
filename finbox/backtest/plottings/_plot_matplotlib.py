import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import empyrical as ep
from pyfolio import plotting, pos, utils, timeseries
from pyfolio.utils import APPROX_BDAYS_PER_MONTH

from .. import pyfolio as pf


def plot_rolling_returns(returns: pd.Series,
                         factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                         live_start_date: Optional[str] = None,
                         cone_std: Optional[List[float]] = None,
                         ax: Optional[Axes] = None) -> Axes:  # noqa
    """
    Plots cumulative rolling returns versus some benchmarks'.
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
    """
    Plots the rolling Sharpe ratio versus date.
    """
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
            bench_sharpe.plot(alpha=.7, lw=2, color=color, ax=ax)
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

    if 'fig' in locals():
        fig.tight_layout()

    return axes


def plot_rolling_betas(returns: pd.Series, factor_returns: pd.Series,
                       ax: Optional[Axes] = None) -> Axes:
    """
    Plots the rolling 6-month and 12-month beta versus date.
    """
    if factor_returns is None:
        raise Exception('`factor_returns` must be provided when calculating '
                        'betas')
    elif isinstance(factor_returns, list):
        factor_returns = factor_returns[0]

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    ax.set_title("Rolling portfolio beta to " + str(factor_returns.name))
    ax.set_ylabel('Beta')
    rb_1 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
    rb_1.plot(color='steelblue', lw=3, alpha=0.6, ax=ax)
    rb_2 = timeseries.rolling_beta(
        returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)
    rb_2.plot(color='grey', lw=3, alpha=0.4, ax=ax)
    ax.axhline(rb_1.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)

    ax.set_xlabel('')
    ax.legend(['6-mo',
               '12-mo'],
              loc='best', frameon=True, framealpha=0.5)

    return ax


def plot_rolling_vol(returns: pd.Series,
                     factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                     rolling_window: int = APPROX_BDAYS_PER_MONTH * 6,
                     ax: Optional[Axes] = None) -> Axes:
    """
    Plots the rolling volatility versus date.
    """
    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(utils.two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.set_title('Rolling volatility (6-month)')
    ax.set_ylabel('Volatility')
    ax.set_xlabel('')

    rolling_vol_ts = pf.timeseries.rolling_volatility(returns, rolling_window)
    rolling_vol_ts.plot(alpha=.7, lw=3, color='orangered', ax=ax,
                        label='Strategy')

    if isinstance(factor_returns, pd.Series):
        factor_returns = [factor_returns]

    color_index = 1
    if isinstance(factor_returns, list):
        bench_means = []
        colors = sns.color_palette('Greys', len(factor_returns))
        for bench, color in zip(factor_returns, colors):
            bench_vol = pf.timeseries.rolling_volatility(bench, rolling_window)
            bench_vol.plot(alpha=.7, lw=3, color=color, ax=ax,
                           label=bench.name)
            color_index += 1
            bench_means.append(bench_vol.mean())

        for u in bench_means:
            ax.axhline(u, color=color, linestyle=':', lw=2)

    ax.axhline(rolling_vol_ts.mean(), color='steelblue', linestyle='--', lw=3)
    ax.axhline(0.0, color='black', linestyle='-', lw=2)
    ax.legend(loc='best', frameon=True, framealpha=0.5)

    return ax


def plot_monthly_heatmap(returns: pd.Series,
                         ax: Optional[Axes] = None) -> Axes:
    """
    Plots a heatmap of returns by month.
    """
    return plotting.plot_monthly_returns_heatmap(returns, ax=ax)


def plot_exposures(returns: pd.Series, positions: pd.DataFrame,
                   ax: Optional[Axes] = None) -> Axes:
    """
    Plots a cake chart of the long and short exposure.
    """
    return plotting.plot_exposures(returns, positions, ax=ax)


def plot_exposures_by_assets(positions: pd.DataFrame,
                             ax: Optional[Axes] = None) -> Axes:
    """
    plots the exposures of the held positions of all time.
    """
    pos_alloc = pos.get_percent_alloc(positions)
    pos_alloc_no_cash = pos_alloc.drop('cash', axis=1)

    if ax is None:
        ax = plt.gca()

    pos_alloc_no_cash.plot(
        title='Portfolio allocation over time',
        alpha=0.5, ax=ax)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', frameon=True, framealpha=0.5,
              bbox_to_anchor=(0.5, -0.14), ncol=5)

    ax.set_ylabel('Exposure by holding')
    ax.set_xlabel('')
    return ax


def plot_interesting_periods(returns: pd.Series,
                             benchmark_rets: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                             periods: Optional[List[Dict]] = None,
                             override: bool = False,
                             axes: Optional[List[Axes]] = None) -> Tuple[Axes, None]:  # noqa
    """
    Plot out the cumulative returns over several selected periods such as 08
    financial crisis.
    """
    rets_interesting = pf.extract_interesting_date_ranges(
        returns, periods, override)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any'
                      'interesting times.', UserWarning)
        return

    if isinstance(benchmark_rets, pd.Series):
        benchmark_rets = [benchmark_rets]

    if isinstance(benchmark_rets, list):
        bmark_int = []
        for bmark in benchmark_rets:
            bmark_int.append(
                pf.extract_interesting_date_ranges(bmark, periods, override))

    num_plots = len(rets_interesting)
    num_rows = int((num_plots + 1) / 2.0)
    height = num_rows*3

    if axes is None:
        fig, axes = plt.subplots(num_rows, 2, figsize=(10, height))
        axes = axes.flatten()

    for i, (name, ret_period) in enumerate(rets_interesting.items()):

        cum_rets = ep.cum_returns(ret_period)
        cum_rets.plot(lw=3, color='forestgreen', alpha=0.6, label='Algo',
                      ax=axes[i])

        if benchmark_rets is not None:
            colors = sns.color_palette('Greys', len(benchmark_rets)).as_hex()

            for bmark, b, color in zip(bmark_int, benchmark_rets, colors):
                cum_bech = ep.cum_returns(bmark[name][cum_rets.index])
                cum_bech.plot(lw=2, color=color, label=b.name,
                              alpha=0.60, ax=axes[i])

        axes[i].legend(loc='best', frameon=True, framealpha=0.5)
        axes[i].set_title(name, fontweight=700)
        axes[i].set_ylabel('Returns')
        axes[i].set_xlabel('')
        axes[i].axhline(0.0, linestyle='--', color='black', lw=1)

    if num_plots < len(axes):
        axes[-1].set_visible(False)

    if 'fig' in locals():
        fig.tight_layout()

    return axes
