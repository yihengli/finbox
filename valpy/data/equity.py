"""
A set of Python Client functions to get equity related data

Yahoo related logics are credited to Backtrader.
"""

import pandas as pd
import requests
import io
from datetime import date
from dateutil.relativedelta import relativedelta
from enum import Enum


class EquityApiMode(Enum):
    yahoo = 'yahoo'


def _validate_and_transform_dates(todate, fromdate):
    """
    Make sure the input dates are valid
    """

    if isinstance(todate, str):
        todate = pd.to_datetime(todate).date()
    if isinstance(fromdate, str):
        fromdate = pd.to_datetime(fromdate).date()

    if type(todate) == type(fromdate) and todate <= fromdate:
        raise Exception('todate must be greater than fromdate')

    return todate, fromdate


def get_history(ticker, todate=None, fromdate=None, freq='d', api_mode='yahoo',
                proxies=None, set_index=False):
    """
    Get the history prices over different equity assets from various data
    vendors.

    Parameters
    ----------
    ticker : str
        Valid symbol such as `SPY` or `^VIX`
    todate : str, optional
        Format as `YYYY-MM-DD`, if not given, default to pull last 5 years
    fromdate : str, optional
        Format as `YYYY-MM-DD`, if not given, default to pull till today's data
    freq : str, optional
        Currently only supports 'd' (daily), '1wk' (weekly) and '1mo' (monthly)
        (the default is 'd')
    api_mode : str, optional
        Currently only supports data from Yahoo (the default is 'yahoo')
    proxies : dict{str: str}, optional
        A dict indicating which proxy to go through for the download as in
        {'http': 'http://myproxy.com'} or {'http': 'http://127.0.0.1:8080'}
        (the default is None)
    set_index : bool, optional
        [description] (the default is False, which [default_description])

    Returns
    -------
    pandas.DataFrame
        Standard OHLCV table.
    """

    if todate is None:
        todate = date.today()

    if fromdate is None:
        fromdate = todate - relativedelta(years=5)

    todate, fromdate = _validate_and_transform_dates(todate, fromdate)

    if EquityApiMode(api_mode) == EquityApiMode('yahoo'):
        data = _get_history_yahoo(ticker, todate, fromdate, freq, proxies)

    if set_index:
        data.set_index("Date", inplace=True)
        data.index = pd.to_datetime(data.index)

    return data


def _get_history_yahoo(ticker, todate, fromdate, freq, proxies):
    """
    Pull data from Yahoo v7 API
    """

    urlhist = 'https://finance.yahoo.com/quote/{}/history'
    urldown = 'https://query1.finance.yahoo.com/v7/finance/download'
    retries = 3
    proxies = {}
    posix = date(1970, 1, 1)
    timeframe = '1' + freq  # '1wk' or '1mo'

    url = urlhist.format(ticker)
    sesskwargs = dict()
    if proxies:
        sesskwargs['proxies'] = proxies

    # Get cookies
    crumb = None
    sess = requests.Session()

    for i in range(retries + 1):
        resp = sess.get(url, **sesskwargs)
        if resp.status_code != requests.codes.ok:
            continue

        txt = resp.text
        i = txt.find('CrumbStore')
        if i == -1:
            continue
        i = txt.find('crumb', i)
        if i == -1:
            continue
        istart = txt.find('"', i + len('crumb') + 1)
        if istart == -1:
            continue
        istart += 1
        iend = txt.find('"', istart)
        if iend == -1:
            continue

        crumb = txt[istart:iend]
        crumb = crumb.encode('ascii').decode('unicode-escape')
        break

    if crumb is None:
        raise Exception('Crumb not found')
        f = None

    # urldown/ticker?period1=posix1&period2=posix2&interval=1d&events=history&crumb=crumb
    urld = '{}/{}'.format(urldown, ticker)

    urlargs = []
    period2 = (todate - posix).total_seconds()
    period1 = (fromdate - posix).total_seconds()

    urlargs.append('period2={}'.format(int(period2)))
    urlargs.append('period1={}'.format(int(period1)))
    urlargs.append('interval={}'.format(timeframe))
    urlargs.append('events=history')
    urlargs.append('crumb={}'.format(crumb))

    # Download data
    urld = '{}?{}'.format(urld, '&'.join(urlargs))
    f = None
    for i in range(retries + 1):  # at least once
        resp = sess.get(urld, **sesskwargs)
        if resp.status_code != requests.codes.ok:
            continue

        ctype = resp.headers['Content-Type']
        if 'text/csv' not in ctype:
            raise Exception('Wrong content type: %s' % ctype)
            # HTML returned? wrong url?
            continue

        # buffer everything from the socket into a local buffer
        try:
            # r.encoding = 'UTF-8'
            f = pd.read_csv(io.StringIO(resp.text, newline=None))
        except Exception:
            # try again if possible
            continue

        break

    return f
