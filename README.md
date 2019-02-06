# ValPy

This is a Python client for a set of Finance related tools.

## Data APIs

The goal is to easily access finance data as pandas dataframe from various data vendors or ingesting data and pull data from internal data storages. (this may be useful for API that has request limits)

### Currently Supported

* Equity
  * Yahoo
* Options
  * ORATS

### Yahoo

```python
from valpy.data.equity import get_history

spy = get_history('SPY', todate='2019-01-01', fromdate='2018-01-01', 
                  freq='d', api_mode='yahoo')
```

| Date       | Open       | High       | Low        | Close      | Adj Close  | Volume   |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | -------- |
| 2018-01-02 | 267.839996 | 268.809998 | 267.399994 | 268.769989 | 263.759949 | 86655700 |
| 2018-01-03 | 268.959991 | 270.640015 | 268.959991 | 270.470001 | 265.428253 | 90070400 |

### ORATS

For the detailed information of backend API, please check https://docs.orats.io/

```python
from valpy.data.options import OratsData

od = OratsData(token='YOUR_TOKEN_FROM_ORATS')
df = od.get_history_data(name='strikes', ticker='SPY', fromdate='2019-01-01',
                         todate='2019-02-01')
df.head(2)
```

| tradeDate  | strike | ticker | callAskPrice | callBidPrice | ...  |
| ---------- | ------ | ------ | ------------ | ------------ | ---- |
| 2019-01-02 | 100.0  | SPY    | 149.68       | 149.16       | ...  |
| 2019-01-02 | 105.0  | SPY    | 144.69       | 144.16       | ...  |

Alternatively, you can specify the specific `fields` or even filter the data using `filters`, this will minimize the data pulled from API and save memories.

For more detailed definitions, please check https://docs.orats.io/data-api-guide/definitions.html

```python
data = od.get_history_data(
    name='strikes', ticker='SPY',
    dates=['2018-05-01', '2019-02-01'],
    fields=['tradeDate', 'strike', 'dte', 'callVolume', 'putVolume'],
    filters=[("dte", 20, 40)],
    concat=False)
data['2019-02-01'].head(2)
```

| callVolume | dte  | PutVolume | Strike | tradeDate  |
| ---------- | ---- | --------- | ------ | ---------- |
| 0          | 21   | 50        | 205.0  | 2019-02-01 |
| 0          | 21   | 210.0     | 205.0  | 2019-02-01 |

* `dates` allows to pull granular dates instead of a sequence of dates
* `concat` True then return a concatenated dataframe, otherwise a dictionary of `{"date": dataframe}`
* `filters` must be a list of `(metric, lower_bound, upper_bound)`. For the possible metrics, please check ORATS documentations. For example, `strikes` only allows users to filter on `dte` and `delta`

> This function currently only supports `HIST` data, with the following codes:
>
> - "strikes", "volatility", "summaries", "cores general", "cores earn", "dividend", "earnings", "splits", "ivrank", "monies implied", "monies forecast"

## Backtesting

The goal is to have a set of higher level wrappers to do backtest on a set of simple trading signals or pre-calculated portfolio rebalance weights.

The back-end engine is currently based on `backtrader` and `pyfolio`. Going forward, may also support other commercial engines to backtest assets other than equities. And this library may also support some low level performance reporting mechanisms to integrate into a reporting WEB App, or REST API.
