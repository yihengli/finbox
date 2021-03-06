from collections import OrderedDict
from typing import Dict, List, Optional, Union
import warnings

import backtrader as bt
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display

from . import plotting, pyfolio


class ReportBuilder(object):

    template = """
<!DOCTYPE html>
<html>

<head>
  <!-- Required meta tags -->
  <meta charset="utf-8">
  <meta name="viewport"
        content="width=device-width, initial-scale=1, shrink-to-fit=no">

  <!-- Echarts JS -->
  <script src="http://code.jquery.com/jquery-1.11.0.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/4.1.0/echarts-en.min.js"></script>

  <!-- Bootstrap CSS -->
  <link rel="stylesheet"
        href="https://stackpath.bootstrapcdn.com/bootstrap/4.2.1/css/bootstrap.min.css"
        integrity="sha384-GJzZqFGwb1QTTN6wy59ffF1BuGJpLSa9DkKMp0DgiMDm4iYMj70gZWKYbI706tWS"
        crossorigin="anonymous">

  <title>Report</title>
</head>

<body class="bg-light">

    <nav class="navbar navbar-expand-lg sticky-top navbar-{nav_font_color}"
        style="background-color: {nav_bg_color};">
        <a class="navbar-brand "
        href="#">
            <img src="{nav_logo_address}"
                width="{nav_logo_width}"
                height="{nav_log_height}"
                class="d-inline-block align-top"
                alt="">
        </a>

        <div class="collapse navbar-collapse"
            id="navbarTogglerDemo02">
            <ul class="navbar-nav mr-auto mt-2 mt-lg-0">
                <li class="nav-item">
                    <a class="nav-link"
                    href="#performance">Performance</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link"
                    href="#returns">Returns</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link"
                    href="#drawdown">Drawdowns</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link"
                    href="#positions">Positions</a>
                </li>
            </ul>
        </div>
    </nav>

  <div class="container">
    <h1>{report_name}</h1>
    <h2><a name="performance"></a>Performance Table</h2>
    {table}
    <h2><a name="returns"></a>Returns Analysis</h2>
      
      <div id="rolling_cum_returns" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartRCT = echarts.init(document.getElementById('rolling_cum_returns'));
        {rct_func}
        var optionRCT = {rct_option};
        chartRCT.setOption(optionRCT);
        $(window).on('resize', function(){{
          if(chartRCT != null && chartRCT != undefined){{chartRCT.resize();}} }});
      </script>

      <div id="rolling_vols" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartRVOL = echarts.init(document.getElementById('rolling_vols'));
        var optionRVOL = {rvol_option};
        chartRVOL.setOption(optionRVOL);
        $(window).on('resize', function(){{
          if(chartRVOL != null && chartRVOL != undefined){{chartRVOL.resize();}} }});
      </script>
      
      <div id="rolling_cum_sharpes" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartRCS = echarts.init(document.getElementById('rolling_cum_sharpes'));
        var optionRCS = {rcs_option};
        chartRCS.setOption(optionRCS);
        $(window).on('resize', function(){{
          if(chartRCS != null && chartRCS != undefined){{chartRCS.resize();}} }});
      </script>

      <div id="rolling_betas" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartRBETA = echarts.init(document.getElementById('rolling_betas'));
        var optionRBETA = {rbeta_option};
        chartRBETA.setOption(optionRBETA);
        $(window).on('resize', function(){{
          if(chartRBETA != null && chartRBETA != undefined){{chartRBETA.resize();}} }});
      </script>

      <div id="monthly_returns_heatmap" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartMRH = echarts.init(document.getElementById('monthly_returns_heatmap'));
        var optionMRH = {mrh_option};
        chartMRH.setOption(optionMRH);
        $(window).on('resize', function(){{
          if(chartMRH != null && chartMRH != undefined){{chartMRH.resize();}} }});
      </script>

    <h2><a name="drawdown"></a>Drawdown Analysis</h2>
      {table_dd}
      <div id="drawdown_and_underwater" style="width: 100%;height:500px;"></div>
      <script type="text/javascript">
        var chartDAU = echarts.init(document.getElementById('drawdown_and_underwater'));
        var optionDAU = {dau_option};
        chartDAU.setOption(optionDAU);
        $(window).on('resize', function(){{
          if(chartDAU != null && chartDAU != undefined){{chartDAU.resize();}} }});
      </script>

    <hr>
    <h2><a name="positions"></a>Position Analysis</h2>
      <div id="exposure_positions" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartEXP = echarts.init(document.getElementById('exposure_positions'));
        var optionEXP = {exp_option};
        chartEXP.setOption(optionEXP);
        $(window).on('resize', function(){{
          if(chartEXP != null && chartEXP != undefined){{chartEXP.resize();}} }});
      </script>
      <div id="exposure_by_asset" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartEXPBA = echarts.init(document.getElementById('exposure_by_asset'));
        var optionEXPBA = {expba_option};
        chartEXPBA.setOption(optionEXPBA);
        $(window).on('resize', function(){{
          if(chartEXPBA != null && chartEXPBA != undefined){{chartEXPBA.resize();}} }});
      </script>
      <!--
      <div id="gross_leverages" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartGL = echarts.init(document.getElementById('gross_leverages'));
        var optionGL = {gl_option};
        chartGL.setOption(optionGL);
        $(window).on('resize', function(){{
          if(chartGL != null && chartGL != undefined){{chartGL.resize();}} }});
      </script> -->

    {interesting_periods_section}
  </div>
</body>
</html>
"""  # noqa E501

    def __init__(self,
                 strat: bt.strategy.Strategy,
                 benchmark_rets: Union[None, pd.Series, List[pd.Series]] = None,  # noqa
                 live_start_date: Optional[str] = None,
                 report_name: str = 'Report',
                 custom_interesting_periods: Optional[List[OrderedDict]] = None,  # noqa
                 custom_interesting_periods_overide: bool = False,
                 returns: Optional[pd.Series] = None,
                 positions: Optional[pd.DataFrame] = None,
                 transactions: Optional[pd.DataFrame] = None,
                 gross_lev: Optional[pd.DataFrame] = None,
                 navbar_settings: Optional[Dict] = None):

        def _check_and_fix_tz(bench: pd.Series) -> pd.Series:
            if bench.index.tzinfo is None or bench.index.tzinfo.utcoffset(bench.index) is None:  # noqa
                bench.index = bench.index.tz_localize('UTC')
            return bench

        def _validate_returns(ret: pd.Series) -> None:
            if len(ret[ret.isnull()]) > 0:
                warnings.warn('%s series contains nan values' % ret.name,
                              UserWarning)

        # Handle Backtrader Strategy Object or directly use return data..
        if strat is not None:
            pyfoliozer = strat.analyzers.getbyname('pyfolio')
            returns, positions, transactions, gross_lev = \
                pyfoliozer.get_pf_items()
        else:
            if returns is None or positions is None or transactions is None:
                raise Exception("Either a `strat` object or `returns, "
                                "positions, transactions` should be provided")

        if isinstance(benchmark_rets, pd.Series):
            _validate_returns(benchmark_rets)
            benchmark_rets = _check_and_fix_tz(benchmark_rets)
            benchmark_rets = pd.merge(pd.DataFrame(returns),
                                      pd.DataFrame(benchmark_rets),
                                      left_index=True, right_index=True,
                                      how="left").fillna(0).iloc[:, 1]
        elif isinstance(benchmark_rets, list):
            for bench in benchmark_rets:
                _validate_returns(bench)
                bench = _check_and_fix_tz(bench)
                bench = pd.merge(pd.DataFrame(returns),
                                 pd.DataFrame(bench),
                                 left_index=True, right_index=True,
                                 how="left").fillna(0).iloc[:, 1]

        # Define Attributes
        _validate_returns(returns)
        self.returns = returns
        self.positions = positions
        self.transactions = transactions
        self.gross_lev = gross_lev

        self.benchmark_rets = benchmark_rets
        self.live_start_date = live_start_date
        self.report_name = report_name

        self.periods = custom_interesting_periods
        self.override = custom_interesting_periods_overide

        if navbar_settings is None:
            navbar_settings = {}

        self.navbar_settings = navbar_settings
        nav_items = ["nav_font_color", "nav_bg_color", "logo", "logo_width",
                     "logo_height"]
        nav_defaults = ["light", "#fff", "https://cdn1.iconfinder.com/data/icons/social-messaging-ui-color-shapes/128/document-circle-blue-512.png", 30, 30]  # noqa E501
        for item, default in zip(nav_items, nav_defaults):
            if item not in navbar_settings.keys():
                self.navbar_settings[item] = default

    def build_report(self, dest: Optional[str] = None,
                     chart_type: str = 'echarts') -> None:
        jupyter = True if dest is None else False

        if dest is not None and chart_type != 'echarts':
            raise NotImplementedError('Currently report only supports `echarts` engine when outputting reports as HTML')  # noqa

        table = self.get_performance_table(jupyter=jupyter)
        table_drawdowns = self.get_drawdown_table(jupyter=jupyter)

        if jupyter and chart_type == 'echarts':
            display(self.get_rolling_returns(jupyter=jupyter))
            display(self.get_rolling_vol(jupyter=jupyter))
            display(self.get_rolling_sharpes(jupyter=jupyter))
            try:
                display(self.get_interactive_rolling_betas(jupyter=jupyter))
            except Exception:
                print('Benchmark returns are needed for beta returns')
            display(self.get_monthly_heatmap(jupyter=jupyter))
            display(self.get_drawdown_and_underwater(jupyter))
            display(self.get_exposures(jupyter=jupyter))
            display(self.get_exposures_by_asset(jupyter))
            # display(self.get_interactive_gross_leverage(jupyter=jupyter))
            display(self.get_interesting_periods(jupyter))
        elif jupyter and chart_type == 'matplotlib':
            _ = self.get_rolling_returns(jupyter=jupyter,
                                         chart_type=chart_type)
            plt.show()

            _ = self.get_rolling_vol(jupyter=jupyter,
                                     chart_type=chart_type)
            plt.show()

            _ = self.get_rolling_sharpes(jupyter=jupyter,
                                         chart_type=chart_type)
            plt.show()

            try:
                _ = self.get_rolling_betas(jupyter=jupyter,
                                           chart_type=chart_type)
                plt.show()
            except Exception:
                print('Benchmark returns are needed for beta returns')

            _ = self.get_monthly_heatmap(jupyter=jupyter,
                                         chart_type=chart_type)
            plt.show()

            _ = self.get_drawdown_and_underwater(jupyter=jupyter,
                                                 chart_type=chart_type)
            plt.show()

            _ = self.get_exposures(jupyter=jupyter,
                                   chart_type=chart_type)
            plt.show()

            _ = self.get_exposures_by_asset(jupyter=jupyter,
                                            chart_type=chart_type)
            plt.show()

            _ = self.get_interesting_periods(jupyter=jupyter,
                                             chart_type=chart_type)
            plt.show()
        if dest is not None:
            rct_f, rct_o = self.get_rolling_returns(
                jupyter=jupyter)
            rvol_o = self.get_rolling_vol(jupyter=jupyter)
            try:
                rbeta_o = self.get_rolling_betas(jupyter=jupyter)
            except Exception:
                rbeta_o = ''
            rcs_o = self.get_rolling_sharpes(jupyter=jupyter)
            mrh_o = self.get_monthly_heatmap(jupyter=jupyter)
            dau_o = self.get_drawdown_and_underwater(jupyter)
            exp_o = self.get_exposures(jupyter=jupyter)
            expba_o = self.get_exposures_by_asset(jupyter=jupyter)
            # gl_o = self.get_gross_leverage(jupyter=jupyter)

            interesting_periods = self.get_interesting_periods_section()

            with open(dest, 'w') as report:
                report.write(self.template.format(
                    report_name=self.report_name,
                    nav_font_color=self.navbar_settings["nav_font_color"],
                    nav_bg_color=self.navbar_settings["nav_bg_color"],
                    nav_logo_address=self.navbar_settings["logo"],
                    nav_logo_width=self.navbar_settings["logo_width"],
                    nav_log_height=self.navbar_settings["logo_height"],
                    table=table,
                    table_dd=table_drawdowns,
                    rct_func=rct_f,
                    rct_option=rct_o,
                    rvol_option=rvol_o,
                    rbeta_option=rbeta_o,
                    rcs_option=rcs_o,
                    mrh_option=mrh_o,
                    dau_option=dau_o,
                    exp_option=exp_o,
                    expba_option=expba_o,
                    gl_option='',
                    interesting_periods_section=interesting_periods))

    def get_performance_table(self, jupyter=True):
        return pyfolio.show_perf_stats(returns=self.returns,
                                       factor_returns=self.benchmark_rets,
                                       positions=self.positions,
                                       transactions=self.transactions,
                                       live_start_date=self.live_start_date,
                                       jupyter=jupyter)

    def get_drawdown_table(self, jupyter=True):
        return pyfolio.show_worst_drawdown_table(returns=self.returns,
                                                 jupyter=jupyter, pandas=False)

    def get_rolling_returns(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_rolling_returns(
            returns=self.returns,
            factor_returns=self.benchmark_rets,
            live_start_date=self.live_start_date,
            cone_std=[1, 1.5, 2],
            chart_type=chart_type
        )

        if jupyter:
            return plot
        else:
            html_text = plot._repr_html_()
            chart_id = plot.chart_id

            func_start = html_text.find('function tooltip_format(params)')
            func_end = html_text.find('var option_{}'.format(chart_id))
            option_start = html_text.find(
                '{} = {{\n    "title": '.format(chart_id))
            option_end = html_text.find('\nmyChart_{}'.format(chart_id))

            func_str = html_text[func_start:func_end]
            option_str = html_text[option_start + len(chart_id) + 3:option_end]

            return func_str, option_str

    def _echart_option_extract(self, plot):
        html_text = plot._repr_html_()
        chart_id = plot.chart_id
        option_start = html_text.find(
            '{} = {{\n    "title": '.format(chart_id))
        option_end = html_text.find('\nmyChart_{}'.format(chart_id))
        option_str = html_text[option_start + len(chart_id) + 3:option_end]
        return option_str

    def get_rolling_vol(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_rolling_vol(
            returns=self.returns, factor_returns=self.benchmark_rets,
            chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_rolling_betas(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_rolling_betas(
            returns=self.returns, factor_returns=self.benchmark_rets,
            chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_rolling_sharpes(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_rolling_sharpes(
            returns=self.returns, factor_returns=self.benchmark_rets,
            chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_monthly_heatmap(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_monthly_heatmap(
            returns=self.returns, chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_drawdown_and_underwater(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_drawdown_underwater(
            returns=self.returns, chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_exposures(self, jupyter=True, chart_type='echarts'):

        plot = plotting.plot_exposures(
            returns=self.returns, positions=self.positions,
            chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_exposures_by_asset(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_exposures_by_assets(
            positions=self.positions, chart_type=chart_type)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_gross_leverage(self, jupyter=True, chart_type='echarts'):
        pass

    def get_interesting_periods(self, jupyter=True, chart_type='echarts'):
        plot = plotting.plot_interesting_periods(
            returns=self.returns, benchmark_rets=self.benchmark_rets,
            periods=self.periods, override=self.override,
            chart_type=chart_type)

        if jupyter:
            if plot is None:
                return "No Interesting Periods Overlapped"
            return plot[0]
        else:
            if plot is None:
                return None
            return self._echart_option_extract(plot[0]), plot[1]

    def get_interesting_periods_section(self):
        template = """
<hr>
<h2>Interesting Periods Analysis</h2>
    <div id="interesting_periods_analysis" style="width: 100%;height:{size}px;"></div>
    <script type="text/javascript">
    var chartIPA = echarts.init(document.getElementById('interesting_periods_analysis'));
    var optionIPA = {ipa_option};
    chartIPA.setOption(optionIPA);
    $(window).on('resize', function(){{
        if(chartIPA != null && chartIPA != undefined){{chartIPA.resize();}} }});
    </script>
"""  # noqa E501
        res = self.get_interesting_periods(jupyter=False, chart_type='echarts')
        if res is None:
            return ""
        else:
            return template.format(size=res[1], ipa_option=res[0])
