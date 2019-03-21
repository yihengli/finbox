import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import backtrader as bt
import empyrical as ep
import numpy as np
import pandas as pd
from backtrader.analyzers.leverage import GrossLeverage
from backtrader.analyzers.positions import PositionsValue
from backtrader.analyzers.timereturn import TimeReturn
from backtrader.analyzers.transactions import Transactions
from backtrader.utils.py3 import iteritems
from IPython.core.display import HTML, display
from pyfolio import timeseries
from pyfolio.utils import APPROX_BDAYS_PER_MONTH

from .interesting_periods import PERIODS

STAT_FUNCS_PCT = [
    'Annual return',
    'Cumulative returns',
    'Annual volatility',
    'Max drawdown',
    'Daily value at risk',
    'Daily turnover'
]


class PyFolio(bt.Analyzer):
    '''
    This analyzer is orignally develeped from Backtrader. We modified some
    of internal processings to make it better align with our pyfolio use cases.

    This analyzer uses 4 children analyzers to collect data and transforms it
    in to a data set compatible with ``pyfolio``
    Children Analyzer
      - ``TimeReturn``
        Used to calculate the returns of the global portfolio value
      - ``PositionsValue``
        Used to calculate the value of the positions per data. It sets the
        ``headers`` and ``cash`` parameters to ``True``
      - ``Transactions``
        Used to record each transaction on a data (size, price, value). Sets
        the ``headers`` parameter to ``True``
      - ``GrossLeverage``
        Keeps track of the gross leverage (how much the strategy is invested)
    Params:
      These are passed transparently to the children
      - timeframe (default: ``bt.TimeFrame.Days``)
        If ``None`` then the timeframe of the 1st data of the system will be
        used
      - compression (default: `1``)
        If ``None`` then the compression of the 1st data of the system will be
        used
    Both ``timeframe`` and ``compression`` are set following the default
    behavior of ``pyfolio`` which is working with *daily* data and upsample it
    to obtaine values like yearly returns.
    Methods:
      - get_analysis
        Returns a dictionary with returns as values and the datetime points for
        each return as keys
    '''
    params = (
        ('timeframe', bt.TimeFrame.Days),
        ('compression', 1)
    )

    def __init__(self):
        dtfcomp = dict(timeframe=self.p.timeframe,
                       compression=self.p.compression)

        self._returns = TimeReturn(**dtfcomp)
        self._positions = PositionsValue(headers=True, cash=True)
        self._transactions = Transactions(headers=True)
        self._gross_lev = GrossLeverage()

    def stop(self):
        super(PyFolio, self).stop()
        self.rets['returns'] = self._returns.get_analysis()
        self.rets['positions'] = self._positions.get_analysis()
        self.rets['transactions'] = self._transactions.get_analysis()
        self.rets['gross_lev'] = self._gross_lev.get_analysis()

    def get_pf_items(self):
        '''Returns a tuple of 4 elements which can be used for further processing with
          ``pyfolio``
          returns, positions, transactions, gross_leverage
        Because the objects are meant to be used as direct input to ``pyfolio``
        this method makes a local import of ``pandas`` to convert the internal
        *backtrader* results to *pandas DataFrames* which is the expected input
        by, for example, ``pyfolio.create_full_tear_sheet``
        The method will break if ``pandas`` is not installed
        '''
        # keep import local to avoid disturbing installations with no pandas
        import pandas
        from pandas import DataFrame as DF

        #
        # Returns
        cols = ['index', 'return']
        returns = DF.from_records(iteritems(self.rets['returns']),
                                  index=cols[0], columns=cols)
        returns.index = pandas.to_datetime(returns.index)
        returns.index = returns.index.tz_localize('UTC')
        rets = returns['return']
        #
        # Positions

        pss = self.rets['positions']
        ps = [[k] + v[:] for k, v in iteritems(pss)]

        cols = ps.pop(0)  # headers are in the first entry
        positions = DF.from_records(ps, index=cols[0], columns=cols)
        positions.index = pandas.to_datetime(positions.index)
        positions.index = positions.index.tz_localize('UTC')

        #
        # Transactions
        txss = self.rets['transactions']
        txs = list()
        # The transactions have a common key (date) and can potentially happend
        # for several assets. The dictionary has a single key and a list of
        # lists. Each sublist contains the fields of a transaction
        # Hence the double loop to undo the list indirection
        for k, v in iteritems(txss):
            for v2 in v:
                txs.append([k] + v2)

        cols = txs.pop(0)  # headers are in the first entry
        transactions = DF.from_records(txs, index=cols[0], columns=cols)
        transactions.index = pandas.to_datetime(transactions.index)
        transactions.index = transactions.index.tz_localize('UTC')

        # Gross Leverage
        cols = ['index', 'gross_lev']
        gross_lev = DF.from_records(iteritems(self.rets['gross_lev']),
                                    index=cols[0], columns=cols)

        gross_lev.index = pandas.to_datetime(gross_lev.index)
        gross_lev.index = gross_lev.index.tz_localize('UTC')
        glev = gross_lev['gross_lev']

        # Return all together
        return rets, positions, transactions, glev


def show_worst_drawdown_table(returns, top=5, jupyter=False, pandas=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = timeseries.gen_drawdown_table(returns, top=top)

    if pandas:
        return table

    return print_table(table, jupyter=jupyter)


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


def show_perf_stats(returns: pd.Series,
                    factor_returns: Union[pd.Series, List[pd.Series], None] = None,  # noqa
                    positions: Optional[pd.DataFrame] = None,
                    transactions: Optional[pd.DataFrame] = None,
                    turnover_denom: str = 'AGB',
                    live_start_date: Optional[str] = None,
                    bootstrap: bool = False,
                    header_rows: Optional[Dict] = None,
                    jupyter: bool = False,
                    pandas: bool = False) -> Union[None, str, pd.DataFrame]:
    """
    Prints some performance metrics of the strategy.

    - Shows amount of time the strategy has been run in backtest and
      out-of-sample (in live trading).

    - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
      stability, Sharpe ratio, annual volatility, alpha, and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    factor_returns : pd.Series or List[pd.Series], optional
        Daily noncumulative returns of the benchmark factor to which betas are
        computed. Usually a benchmark such as market returns.
         - This is in the same style as returns.
    positions : pd.DataFrame, optional
        Daily net position values.
         - See full explanation in create_full_tear_sheet.
    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.
        - See full explanation in tears.create_full_tear_sheet
    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.
        - See full explanation in txn.get_turnover.
    live_start_date : datetime, optional
        The point in time when the strategy began live trading, after
        its backtest period.
    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics.
         - For more information, see timeseries.perf_stats_bootstrap
    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the displayed table.
    jupyter : bool, optional
        If True, output None and display contents is adaptable with Jupyter
        window, otherwise output raw HTML code
    pandas : bool, optional
        If True, output format will be dataframe instead of HTML codes
    """

    if bootstrap:
        perf_func = timeseries.perf_stats_bootstrap
    else:
        perf_func = timeseries.perf_stats

    # Benchmark data handler
    perf_stats_benchmarks = []
    if factor_returns is not None and isinstance(factor_returns, list):
        for factor in factor_returns:
            perf_stats_benchmarks.append(perf_func(factor))
            perf_stats_benchmarks[-1].name = factor.name
        factor_returns = factor_returns[0]
    elif factor_returns is not None and isinstance(factor_returns, pd.Series):
        perf_stats_benchmarks.append(perf_func(factor_returns))
        perf_stats_benchmarks[-1].name = factor_returns.name

    # Strategy data handler
    perf_stats_all = perf_func(
        returns,
        factor_returns=factor_returns,
        positions=positions,
        transactions=transactions,
        turnover_denom=turnover_denom)

    date_rows = OrderedDict()
    if len(returns.index) > 0:
        date_rows['Start date'] = returns.index[0].strftime('%Y-%m-%d')
        date_rows['End date'] = returns.index[-1].strftime('%Y-%m-%d')

    # Break down into In-Sample and Out-of-sample
    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        returns_is = returns[returns.index < live_start_date]
        returns_oos = returns[returns.index >= live_start_date]

        positions_is = None
        positions_oos = None
        transactions_is = None
        transactions_oos = None

        if positions is not None:
            positions_is = positions[positions.index < live_start_date]
            positions_oos = positions[positions.index >= live_start_date]
            if transactions is not None:
                transactions_is = transactions[(transactions.index <
                                                live_start_date)]
                transactions_oos = transactions[(transactions.index >
                                                 live_start_date)]

        perf_stats_is = perf_func(
            returns_is,
            factor_returns=factor_returns,
            positions=positions_is,
            transactions=transactions_is,
            turnover_denom=turnover_denom)

        perf_stats_oos = perf_func(
            returns_oos,
            factor_returns=factor_returns,
            positions=positions_oos,
            transactions=transactions_oos,
            turnover_denom=turnover_denom)
        if len(returns.index) > 0:
            date_rows['In-sample months'] = int(len(returns_is) /
                                                APPROX_BDAYS_PER_MONTH)
            date_rows['Out-of-sample months'] = int(len(returns_oos) /
                                                    APPROX_BDAYS_PER_MONTH)

        perf_stats = pd.concat(OrderedDict([
            ('In-sample', perf_stats_is),
            ('Out-of-sample', perf_stats_oos),
            ('Strategy', perf_stats_all),
        ]), axis=1)
    else:
        if len(returns.index) > 0:
            date_rows['Total months'] = int(len(returns) /
                                            APPROX_BDAYS_PER_MONTH)
        perf_stats = pd.DataFrame(perf_stats_all, columns=['Strategy'])

    if perf_stats_benchmarks:
        for perf_bench in perf_stats_benchmarks:
            perf_stats = pd.merge(perf_stats, perf_bench.to_frame(),
                                  left_index=True, right_index=True,
                                  how='left')

    # Format the numerical outputs
    for column in perf_stats.columns:
        for stat, value in perf_stats[column].iteritems():
            if stat in STAT_FUNCS_PCT and not np.isnan(value):
                perf_stats.loc[stat, column] = str(np.round(value * 100,
                                                            1)) + '%'
    if header_rows is None:
        header_rows = date_rows
    else:
        header_rows = OrderedDict(header_rows)
        header_rows.update(date_rows)

    # Print Table
    if isinstance(perf_stats, pd.Series):
        table = pd.DataFrame(perf_stats)
    else:
        table = perf_stats

    if pandas:
        return table

    name = None

    if name is not None:
        table.columns.name = name

    return print_table(table, header_rows=header_rows, jupyter=jupyter)


def get_rolling_returns(returns: pd.Series,
                        factor_returns: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                        live_start_date: Optional[str] = None,
                        logy: bool = False,
                        cone_std: Union[float, Tuple, None] = None,
                        volatility_match: bool = False,
                        cone_function: Callable = timeseries.forecast_cone_bootstrap) -> Dict:  # noqa
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
    factor_returns : pd.Series or List[pd.Series], optional
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

    Returns
    -------
    Dict
         {
            'cum_factor_returns': Union[None, pd.Series, List[pd.Series]],
            'is_cum_returns': pd.Series,
            'oos_cum_returns': Union[None, pd.Series],
            'cone_bounds': Union[None, pd.DataFrame]
        }
    """

    if volatility_match and factor_returns is None:
        raise ValueError('volatility_match requires passing of '
                         'factor_returns.')
    elif volatility_match and factor_returns is not None:
        bmark_vol = factor_returns.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    cum_rets = ep.cum_returns(returns, 1.0)

    # Construct benchmark returns
    if factor_returns is not None and isinstance(factor_returns, pd.Series):
        cum_factor_returns = ep.cum_returns(
            factor_returns[cum_rets.index], 1.0)
        cum_factor_returns.name = factor_returns.name
    elif factor_returns is not None and isinstance(factor_returns, list):
        cum_factor_returns = list(
            map(lambda x: ep.cum_returns(x[cum_rets.index], 1.0),
                factor_returns))
        for a, b in zip(cum_factor_returns, factor_returns):
            a.name = b.name
    else:
        cum_factor_returns = None

    # Construct In-Sample and Out-of-Sample returns, if `live_start_date` not
    # given, then only In-Sample is constructed
    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)
        is_cum_returns = cum_rets.loc[cum_rets.index < live_start_date]
        oos_cum_returns = cum_rets.loc[cum_rets.index >= live_start_date]
    else:
        is_cum_returns = cum_rets
        oos_cum_returns = pd.Series([])

    # Construct prediction intervals for the out-of-sample returns
    if len(oos_cum_returns) > 0:
        if cone_std is not None:
            if isinstance(cone_std, (float, int)):
                cone_std = [cone_std]

            is_returns = returns.loc[returns.index < live_start_date]
            cone_bounds = cone_function(
                is_returns,
                len(oos_cum_returns),
                cone_std=cone_std,
                starting_value=is_cum_returns[-1])
            cone_bounds.index = oos_cum_returns.index
        else:
            cone_bounds = None
    else:
        cone_bounds = None

    is_cum_returns = pd.Series(is_cum_returns, index=cum_rets.index)
    if len(oos_cum_returns) > 0:
        oos_cum_returns = pd.Series(oos_cum_returns,
                                    index=cum_rets.index)
    else:
        oos_cum_returns = None

    if cone_bounds is not None:
        cone_bounds = pd.DataFrame(cone_bounds, index=cum_rets.index)

    return {
        'cum_factor_returns': cum_factor_returns,
        'is_cum_returns': is_cum_returns,
        'oos_cum_returns': oos_cum_returns,
        'cone_bounds': cone_bounds
    }


def get_rolling_sharpe(returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6):
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.

    Returns
    -------
    pandas.Sieries
        Rolling sharpe ratios
    """
    rolling_sharpe_ts = timeseries.rolling_sharpe(returns, rolling_window)
    return rolling_sharpe_ts


def extract_interesting_date_ranges(returns, periods=None, override=False):
    """
    Extracts returns based on interesting events. See
    gen_date_range_interesting.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    periods : list[Dict], optional
        A list of interesting events with key as names, and value as a tuple of
        pandas timestamps,
    override : bool, optional
        if True, the original Periods will be overriden with provided periods

    Returns
    -------
    ranges : OrderedDict
        Date ranges, with returns, of all valid events.
    """
    if periods is None:
        periods = PERIODS
    elif periods is not None and not override:
        periods_tmp = PERIODS.copy()
        for period in periods:
            periods_tmp.update(period)
        periods = periods_tmp

    returns_dupe = returns.copy()
    returns_dupe.index = returns_dupe.index.map(pd.Timestamp)
    ranges = OrderedDict()
    for name, (start, end) in periods.items():
        try:
            period = returns_dupe.loc[start:end]
            if len(period) == 0:
                continue
            ranges[name] = period
        except BaseException:
            continue

    return ranges
