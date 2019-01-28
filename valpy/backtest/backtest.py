import backtrader as bt
import pandas as pd

from ..data.equity import get_history
from .pyfolio import PyFolio


class BacktestLogger(object):
    """
    A set of logger function that can replace the default backtrader's
    behaviour to print out the detailed strategy execution information at
    transaction basis. Normally for debuging purpose.
    """

    @staticmethod
    def log(strategy, txt, dt=None):
        ''' Logging function for this strategy'''
        if strategy.params.debug:
            dt = dt or strategy.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    @staticmethod
    def notify_order(strategy, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                strategy.log(
                    'BUY EXECUTED, Price: %.2f, Size: %.1f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.size,
                     order.executed.comm))

                strategy.buyprice = order.executed.price
                strategy.buycomm = order.executed.comm
            else:  # Sell
                strategy.log('SELL EXECUTED, Price: %.2f, Size: %.1f, '
                             'Comm %.2f' %
                             (order.executed.price,
                                 order.executed.size,
                                 order.executed.comm))

            strategy.bar_executed = len(strategy)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            strategy.log('Order Canceled/Margin/Rejected: Order Price: %.2f, '
                         'Size: %.2f' %
                         (order.created.price, order.created.size))

        # Write down: no pending order
        strategy.order = None

    @staticmethod
    def notify_trade(strategy, trade):
        if not trade.isclosed:
            return

        strategy.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))


def build_single_signal_strategy(ticker, signal, is_debug=False, leverage=1,
                                 max_no_signal_days=1, coc=True, dataset=None,
                                 initial_cash=100000., commission=0.):
    """
    Given a specific ticker (currently only supports Equity) and a signal
    (currently only support daily resolution) with -1, 0 and 1. The backtest
    strategy will backtest based on simple long an short operations.

    Parameters
    ----------
    ticker : str
        An acceptable equity ticker such as 'SPY'. If `dataset` is provided
        then this function will use given dataset
    signal : pandas.DataFrame
        One column dataframe with datetime index and -1, 0, 1 as value per cell
    is_debug : bool, optional
        Toggle Debugging mode (the default is False, which not print out any
        transaction details)
    leverage : int, optional
        Leverage level for the strategy (the default is 1, which means no
        leverage is applied)
    max_no_signal_days : int, optional
        The maximum days the strategy will hold its position when there is no
        long / short signals (the default is 1, which means it will immediately
        clear positions when no signal comes in)
    coc : bool, optional
        Cheat on Close Price Option (the default is True, which means the
        backtest assumes you can always buy or sell at the close price per day)
    dataset : pandas.DataFrame
        If provided, this function will use given dataset
    initial_cash : float, optional
        The initial capital when starting the strategy (the default is 100000)
    commission : int, optional
        [description] (the default is 0, which [default_description])

    Returns
    -------
    [type]
        [description]
    """

    btlogger = BacktestLogger()

    class SignalPandasData(bt.feeds.PandasData):
        lines = tuple(["signal"],)
        params = (('datetime', None),) + tuple(
            [(col, -1) for col in bt.feeds.PandasData.datafields[1:]
                + ["signal"]])
        datafields = bt.feeds.PandasData.datafields[1:] + (["signal"])

    # Single Signal Strategy
    class SingleSignalStrategy(bt.Strategy):
        """
        1: step weights instead of default all-in strategy
        2: split max_long_days and max_short_days
        """

        params = (
            ('max_no_signal_days', max_no_signal_days),
            ('debug', is_debug)
        )

        def __init__(self):
            self.no_signal_hold_days = 0

        def log(self, txt, dt=None):
            btlogger.log(self, txt, dt)

        def notify_order(self, order):
            btlogger.notify_order(self, order)

        def notify_trade(self, trade):
            btlogger.notify_trade(self, trade)

        def next(self):
            self.log("CASH = {}".format(self.broker.get_cash()))
            self.log("Portfolio = {}".format(self.broker.get_value()))

            if self.data.signal[0] > 0:
                self.log('+++ BUY SIGNAL TRIGGERED +++')
                self.order_target_percent(target=.99)
                self.no_signal_hold_days = 0
            elif self.data.signal[0] < 0:
                self.log('--- SELL SIGNAL TRIGGERED ---')
                self.order_target_percent(target=-.99)
                self.no_signal_hold_days = 0
            elif self.no_signal_hold_days >= self.params.max_no_signal_days:
                self.log('~~~ CLEARN POSITION ({} days no signals) ~~~'.format(
                    self.no_signal_hold_days))
                self.order_target_percent(target=0.)
            else:
                self.no_signal_hold_days += 1

    def get_pandas_signal_feed(ticker, signal):
        """ Get the pandas data feed with a signal column"""

        signal.columns = ["signal"]
        fromdate = signal.index.min().strftime('%Y-%m-%d')
        todate = signal.index.max().strftime('%Y-%m-%d')

        if dataset is None:
            data = get_history(ticker, fromdate=fromdate,
                               todate=todate, set_index=True)
        else:
            data = dataset
        data = pd.merge(data, signal, left_index=True,
                        right_index=True, how="left").fillna(0)

        return SignalPandasData(dataname=data)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SingleSignalStrategy)

    data = get_pandas_signal_feed(ticker, signal)
    cerebro.adddata(data, name=ticker)
    cerebro.broker.setcash(initial_cash)

    cerebro.broker.set_coc(coc)
    cerebro.addanalyzer(PyFolio)

    cerebro.broker.setcommission(commission=commission, leverage=leverage)

    # Run over everything
    strats = cerebro.run()
    strat0 = strats[0]

    return strat0
