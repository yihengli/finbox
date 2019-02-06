import pandas as pd
import json
import requests
from enum import Enum
from ..utils import utils


class OratsDataTypes(Enum):
    strikes = 'strikes'
    hvs = 'volatility'
    summaries = 'summaries'
    cores_general = 'cores general'
    cores_earn = 'cores earn'
    divs = 'dividend'
    earnings = 'earnings'
    splits = 'splits'
    ivrank = 'ivrank'
    monies_implied = 'monies implied'
    monies_forecast = 'monies forecast'


class OratsData:
    """
    The set of functions to pull data from ORATS Option data:
    (https://www.orats.com/)
    """

    api_address = 'https://api.orats.io/data/'

    def __init__(self, token):
        """
        Initialize an OratsData Object.

        Parameters
        ----------
        token : str
            The credential get from ORATS
        """

        self.token = token

    def get_history_data(self, name, ticker, fromdate=None, todate=None,
                         fields=None, filters=None, concat=True, dates=None):
        """
        Get historical data through ORATS API (https://api.orats.io/data) and
        more information could found at https://docs.orats.io/data-api-guide/

        Parameters
        ----------
        name : str
            The supported api methods, such as `strikes`
        ticker : str
            The supported underlying assets' tickers, such as `AAPL`
        fromdate : str
            Format as `YYYY-MM-DD`
        todate : str, optional
            Format as `YYYY-MM-DD`
        fields : list[str], optional
            If given, only given fields will be pulled from API, this would
            be useful if requried data size is large. By default, all fields
            are pulled.
        filters : list(tuple), optional
            If given, data will be filtered from API, this would be useful if
            required data size is large. By default, all data is pulled.

            The format is as
                ("field_name", lower_bound_float, upper_bound_float)
        concat : bool, optional
            If True, results will be catatenated vertically as one dataframe.
            Otherwise, a dict with key as date and value as a dataframe.
        dates : list[str], optional
            A list of 'YYYY-MM-DD' dates. If provided, only given dates will
            be pulled and `fromdate`, `todate` will be ignored.

        Returns
        -------
        pandas.DataFrame or Dict{str: pd.DataFrame}
            The returned values from ORATS per day
        """

        logger = utils.get_logger()

        if dates is None:
            if fromdate is None or todate is None:
                raise Exception("`fromdate`, `todate` must be provided")
            dates = pd.date_range(fromdate, todate, freq='B')\
                .strftime('%Y-%m-%d').tolist()

        df_dict = {}

        tqdm = utils.get_tqdm()
        for date in tqdm(dates):
            data_per_day = self._get_history_data(name, ticker, date,
                                                  fields, filters)
            if data_per_day is None:
                logger.warning("ORATS Request [{} | {} | {}] Failed".format(
                    name, ticker, date))
            else:
                df_dict[date] = data_per_day

        if concat and len(df_dict) > 0:
            return pd.concat(df_dict)
        else:
            return df_dict

    def _get_history_data(self, name, ticker, trade_date, fields=None,
                          filters=None):
        address = self.api_address + 'hist'
        address += '/%s' % OratsDataTypes(name).name.replace('_', '/')
        address += '?ticker={}&tradeDate={}'.format(ticker, trade_date)

        if fields is not None:
            address += '&fields[{}]={}'.format(
                OratsDataTypes(name).name.replace('_', '-'),
                ','.join(fields))

        if filters is not None:
            # flt in format as ("metric", lower, upper)
            for flt in filters:
                address += '&{}={},{}'.format(*flt)

        r = requests.get(address, headers={'Authorization': self.token})

        if r.status_code != 200:
            return None

        return pd.DataFrame(json.loads(r.text)['data'])
