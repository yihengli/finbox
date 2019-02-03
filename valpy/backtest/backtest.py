import backtrader as bt
import pandas as pd
import numpy as np

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


def get_customized_pandasfeed(data, col_name, adjclose=True):
    """
    Based on Backtrader's default pandas datafeed, create a customized pandas
    feed with an additional column which can be a predefined signal or
    allocation weights.

    Parameters
    ----------
    data : pandas.DataFrame
        Similar data table as Yahoo inputs, standard OLHCV etc.
    col_name : str
        The additional column's column name, will be used as the line name.
    adjclose : bool, optional
        Whether to use the dividend/split adjusted close and adjust all values
        according to it.

    Returns
    -------
    [type]
        [description]
    """
    df = data.copy()

    if adjclose:
        adjfactor = df["Close"] / df["Adj Close"]
        for col in ["Open", "High", "Low"]:
            df[col] /= adjfactor

        df["Close"] = df["Adj Close"]
        df["Volume"] *= adjfactor

    class CustomizedPandasData(bt.feeds.PandasData):
        lines = tuple([col_name], )
        params = (('datetime', None),) + tuple(
            [(col, -1) for col in bt.feeds.PandasData.datafields[1:]
                + [col_name]])
        datafields = bt.feeds.PandasData.datafields[1:] + ([col_name])

    return CustomizedPandasData(dataname=df)


def build_single_signal_strategy(ticker, signal, is_debug=False,
                                 max_no_signal_days=1, coc=True, dataset=None,
                                 initial_cash=100000., adjclose=True,
                                 commission_settings=None):
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
    adjclose : bool, optional
        Whether to use the dividend/split adjusted close and adjust all values
        according to it.
    commission_settings : list[dict]
        A list of commission settings including commision, interest, leverage,
        that will be applied to the backtrader engine in order.

    Returns
    -------
    [type]
        [description]
    """

    btlogger = BacktestLogger()

    # class SignalPandasData(bt.feeds.PandasData):
    #     lines = tuple(["signal"],)
    #     params = (('datetime', None),) + tuple(
    #         [(col, -1) for col in bt.feeds.PandasData.datafields[1:]
    #             + ["signal"]])
    #     datafields = bt.feeds.PandasData.datafields[1:] + (["signal"])

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
            elif self.no_signal_hold_days >= \
                    self.params.max_no_signal_days - 1:
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

        return get_customized_pandasfeed(data, "signal")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SingleSignalStrategy)

    data = get_pandas_signal_feed(ticker, signal)
    cerebro.adddata(data, name=ticker)
    cerebro.broker.setcash(initial_cash)

    cerebro.broker.set_coc(coc)
    cerebro.addanalyzer(PyFolio)

    if commission_settings is not None:
        for setting in commission_settings:
            cerebro.broker.setcommission(**setting)

    # Run over everything
    strats = cerebro.run()
    strat0 = strats[0]

    return strat0


def build_weights_rebalance_strategy(tickers, weights, datasets=None,
                                     is_debug=False, lazy_rebalance=False,
                                     coc=True, dataset=None,
                                     initial_cash=100000., adjclose=True,
                                     commission_settings=None):
    """
    Given a specific ticker (currently only supports Equity) and a signal
    (currently only support daily resolution) with -1, 0 and 1. The backtest
    strategy will backtest based on simple long an short operations.

    Parameters
    ----------
    tickers : list[str]
        A list of acceptable equity tickers such as 'SPY'. If `datasets` is
        provided then this function will use given dataset instead of pulling
        from `get_history`
    weights : pandas.DataFrame
        A dataframe with datetime index and each column as a sereis of weights
        per asset. 1 means 100% capital allocated on it. The column order
        should match the order of tickers
    datasets : list[pandas.DataFrame] or None
        A list of OLHCV tables and the order should match tickers. If given,
        will use the given datasets instead of downloading new data
    lazy_rebalance : bool, optional
        If True, if weights are not changing from two continuous days, no
        trades will be made. Otherwise, the rebalance is based on changing
        values per asset, you should expect more frequent trades.
    is_debug : bool, optional
        Toggle Debugging mode (the default is False, which not print out any
        transaction details)
    coc : bool, optional
        Cheat on Close Price Option (the default is True, which means the
        backtest assumes you can always buy or sell at the close price per day)
    initial_cash : float, optional
        The initial capital when starting the strategy (the default is 100000)
    adjclose : bool, optional
        Whether to use the dividend/split adjusted close and adjust all values
        according to it.
    commission_settings : list[dict]
        A list of commission settings including commision, interest, leverage,
        that will be applied to the backtrader engine in order.

    Returns
    -------
    [type]
        [description]
    """

    btlogger = BacktestLogger()

    def get_pandas_weights_feeds(tickers, weights, datasets):
        """
        Get the a set of pandas data feed with a weights dataframe
        """

        if len(tickers) != weights.shape[1] or len(tickers) != len(datasets):
            raise Exception("`tickers`, `datasets` and `weights`'s columns '"
                            "must have the same length and in the same order")

        fromdate = weights.index.min().strftime('%Y-%m-%d')
        todate = weights.index.max().strftime('%Y-%m-%d')
        weights.columns = tickers

        # Download data if `datasets` are not given
        if datasets is None:
            datasets = []
            for ticker in tickers:
                datasets.append(get_history(ticker, fromdate=fromdate,
                                            todate=todate, set_index=True))

        # Force each asset start from the same start date
        maxi_fromdate = max([d.index.min().strftime('%Y-%m-%d')
                             for d in datasets])

        # Build a list of pandas datafeeds
        data_feed_list = []
        for data, ticker in zip(datasets, tickers):
            df_w = weights[[ticker]]
            df_w.columns = ["weight"]
            data = pd.merge(data, df_w, left_index=True, right_index=True,
                            how="left").fillna(method='ffill')
            data = data[data.index >= maxi_fromdate]
            data_feed_list.append(
                get_customized_pandasfeed(data, "weight", adjclose))

        return data_feed_list, maxi_fromdate

    class WeightsRebalanceStrategy(bt.Strategy):
        """
        A strategy to rebalnce weights per day based on given weights sheet
        """

        params = (
            ('lazy_rebalance', False),
            ('debug', True),
            ('tickers', None),
            ('fromdate', None),
        )

        def log(self, txt, dt=None):
            btlogger.log(self, txt, dt)

        def notify_order(self, order):
            btlogger.notify_order(self, order)

        def notify_trade(self, trade):
            btlogger.notify_trade(self, trade)

        def _is_first_day(self):
            """
            Return True, if the function was executed as the first day of
            backtesting. It's necessary for `lazy_rebalance` when first day
            is not all 0 positions, otherwise the strategy will not rebalance
            from the begining.
            """

            if self.params.fromdate is not None:
                dt = self.datas[0].datetime.datetime().strftime("%Y-%m-%d")
                return dt == self.params.fromdate
            return False

        def _get_delta_value_and_cost(self, ticker, cur_w):
            """
            Given a required current weight, calculate how much values need to
            change to maintain the weights as reuqired
            """
            data = self.getdatabyname(ticker)

            # Price is the bar's close or open depending on `coc` option
            if self.broker.p.coc:
                price = data.lines.close[0]
            else:
                price = data.lines.open[0]

            # Calculate current value for holdings of the given asset
            cur_pos = self.getpositionbyname(ticker)
            adjbase = cur_pos.adjbase if cur_pos.adjbase is not None else 0
            cur_value = adjbase * cur_pos.size

            # Calculate how much values will be changed to holding given weight
            target_value = self.broker.getvalue() * cur_w
            delta_value = target_value - cur_value

            # Caculate the transaction cost if `delta_value` is executed
            cominfo = self.broker.getcommissioninfo(data)
            cost = cominfo.getcommission(np.abs(delta_value) // price, price)

            # Calculate interests caused due to short positions or ETF similar
            cur_pos.datetime = data.datetime.datetime()
            try:
                dt1 = data.datetime.datetime(1)
            except IndexError:
                dt1 = data.datetime.datetime()
            interest = cominfo.get_credit_interest(data, cur_pos, dt1)

            self.log("<%s> Interest Fees: %f" % (ticker, interest))
            cost = cost + interest

            return delta_value, cost, price

        def _build_action_list(self):
            """
            Build the action list object, where each action item contains the
            delta weights, delta values and related delta costs information for
            each asset at the given time
            """
            action_list = []
            for ticker in self.params.tickers:
                action = {}
                action["name"] = ticker

                data = self.getdatabyname(ticker)
                action["cur_w"] = np.round(data.weight[0], 3)
                action["last_w"] = np.round(data.weight[-1], 3)
                action["delta_value"], action["cost"], action["price"] = \
                    self._get_delta_value_and_cost(ticker, action["cur_w"])

                action_list.append(action)

            return action_list

        def report_actual_weights(self, tickers):
            """
            Report the realized current weights of each asset
            """
            for ticker in tickers:
                cur_pos = self.getpositionbyname(ticker)
                adjbase = cur_pos.adjbase if cur_pos.adjbase is not None else 0

                cur_value = adjbase * cur_pos.size
                cur_w = cur_value / self.broker.getvalue()
                self.log("   {}: {:.2%}".format(ticker, cur_w))

        def smart_execution(self, action_list):
            """
            Smartly execute a list of actions:
            1. Sell before Buy, to make sure buy orders can be funded
            2. When target weight is 0, garantee the clear order is executed
               with required values.
            3. Adjust other orders' values with transaction costs, make sure
               other orders are executed, though values may float a bit.
            """

            clear_actions = list(
                filter(lambda x: x["cur_w"] == 0, action_list))

            clear_cost = sum([x["cost"] for x in clear_actions])
            num_other_actions = len(action_list) - len(clear_actions)

            # Evenly distribute clear order's costs to other orders
            if num_other_actions > 0:
                clear_cost_per_order = clear_cost / num_other_actions
            else:
                clear_cost_per_order = 0

            # Make sure SELL happens before BUY
            action_list.sort(key=lambda x: x["delta_value"])

            # In `lazy_rebalance` mode, only take action when weights change
            # or the first trading day
            for action in action_list:
                if not self.params.lazy_rebalance or self._is_first_day() or \
                        (self.params.lazy_rebalance and
                         action["last_w"] != action["cur_w"]):
                    self.log(
                        "<{}> WEIGHTS CHANGE From {:.3f} TO {:.3f}".format(
                            action["name"], action["last_w"], action["cur_w"]))
                    self.smart_order_with_action(action, clear_cost_per_order)

        def smart_order_with_action(self, action, additional_cost):
            """
            Smartly execute the order to buy or sell with given value while
            also considering related costs
            """

            w = action["cur_w"]
            v = action["delta_value"]
            p = action["price"]
            ticker = action["name"]

            if w != 0:

                # For action that's not clear order, include clear cost
                cost = action["cost"] + additional_cost

                # Adjust each order's value by decreasing size to fullfill cost
                if v - cost > 0:
                    order = self.buy(data=ticker, size=(v - cost) // p)
                elif v - cost < 0:
                    order = self.sell(data=ticker, size=(v - cost) // p)
            else:
                order = self.order_target_value(data=ticker, target=0)

            if order is not None:
                signal = "BUY" if order.isbuy() else "SELL"
                self.log("<%s> %s CREATED: size %.1f | price %.3f" %
                         (ticker, signal, order.created.size,
                          order.created.price))
            return order

        def next(self):
            self.log("CASH: {:.2f} | Portfolio {:.2f}".format(
                self.broker.get_cash(), self.broker.get_value()))

            # Record the current positions
            for ticker in self.params.tickers:
                self.last_positions[ticker] = \
                    self.broker.getposition(self.getdatabyname(ticker))

            # Build weight list and sort weight list
            # action_list_item = [
            #    {"name", "cur_w", "last_w", "delta_value", "cost", "price"}]
            action_list = self._build_action_list()

            # Execute the rebalance job
            self.smart_execution(action_list)

            if self.params.debug:
                self.report_actual_weights(self.params.tickers)
            self.log("---" * 10)

    cerebro = bt.Cerebro()

    data_feeds, maxi_fromdate = get_pandas_weights_feeds(tickers,
                                                         weights,
                                                         datasets)
    cerebro.addstrategy(WeightsRebalanceStrategy, tickers=tickers,
                        debug=is_debug, lazy_rebalance=lazy_rebalance,
                        fromdate=maxi_fromdate)

    for data, ticker in zip(data_feeds, tickers):
        cerebro.adddata(data, name=ticker)

    cerebro.broker.setcash(initial_cash)

    cerebro.broker.set_coc(coc)
    cerebro.addanalyzer(PyFolio)

    cerebro.broker.set_checksubmit(False)

    if commission_settings is not None:
        for setting in commission_settings:
            cerebro.broker.setcommission(**setting)

    # Run over everything
    strats = cerebro.run()
    return strats[0]
