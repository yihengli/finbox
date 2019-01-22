# ValPy

This is a Python client for a set of Finance related tools.

## Data APIs

The goal is to easily access finance data as pandas dataframe from various data vendors or ingesting data and pull data from internal data storages. (this may be useful for API that has request limits)

### Currently Supported

* Equity
  * Yahoo

Working on ORATS optins, IEX, Alpha Vantage etc.

### How to use

```python
from valpy.data.equity import get_history

spy = get_history('SPY', todate='2019-01-01', fromdate='2018-01-01', 
                  freq='d', api_mode='yahoo'
```

## Backtesting

The goal is to have a set of higher level wrappers to do backtest on a set of simple trading signals or pre-calculated portfolio rebalance weights.

The back-end engine is currently based on `backtrader` and `pyfolio`. Going forward, may also support other commercial engines to backtest assets other than equities. And this library may also support some low level performance reporting mechanisms to integrate into a reporting WEB App, or REST API.
