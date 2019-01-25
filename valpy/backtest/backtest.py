import backtrader as bt
import pandas as pd

from ..data.equity import get_history


class BacktestLogger(object):

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
                                 max_no_signal_days=1, coc=True,
                                 initial_cash=100000, commission=0):

    btlogger = BacktestLogger()

    class SignalPandasData(bt.feeds.PandasData):
        lines = tuple(["signal"],)
        params = (('datetime', None),) + tuple(
            [(col, -1) for col in bt.feeds.PandasData.datafields[1:]
                + ["signal"]])
        datafields = bt.feeds.PandasData.datafields[1:] + (["signal"])

    def get_pandas_signal_feed(ticker, signal):
        """ Get the pandas data feed with a signal column"""

        signal.columns = ["signal"]
        fromdate = signal.index.min().strftime('%Y-%m-%d')
        todate = signal.index.max().strftime('%Y-%m-%d')

        data = get_history(ticker, fromdate=fromdate,
                           todate=todate, set_index=True)
        data = pd.merge(data, signal, left_index=True,
                        right_index=True, how="left").fillna(0)

        return SignalPandasData(dataname=data)

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
            elif self.data.signal[0] <= 0:
                self.log('--- SELL SIGNAL TRIGGERED ---')
                self.order_target_percent(target=-.99)
                self.no_signal_hold_days = 0
            elif self.no_signal_hold_days >= self.params.max_no_signal_days:
                self.log('~~~ CLEARN POSITION ({} days no signals) ~~~'.format(
                    self.hold_days_without_signal))
                self.order_target_percent(target=0.)
            else:
                self.no_signal_hold_days += 1

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SingleSignalStrategy)

    data = get_pandas_signal_feed(ticker, signal)
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)

    cerebro.broker.set_coc(coc)
    cerebro.addanalyzer(bt.analyzers.PyFolio)

    cerebro.broker.setcommission(commission=commission, leverage=leverage)

    # Run over everything
    strats = cerebro.run()
    strat0 = strats[0]

    return strat0
