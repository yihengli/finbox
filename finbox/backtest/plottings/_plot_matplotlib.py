from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

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
