from . import pyfolio
from . import plotting
from IPython.core.display import display
import pandas as pd


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
  <div class="container">
    <h1>{report_name}</h1>
    <h2>Performance Table</h2>
    {table}
    <h2>Returns Analysis</h2>
      
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

    <h2>Drawdown Analysis</h2>
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
    <h2>Position Analysis</h2>
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
      <div id="gross_leverages" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartGL = echarts.init(document.getElementById('gross_leverages'));
        var optionGL = {gl_option};
        chartGL.setOption(optionGL);
        $(window).on('resize', function(){{
          if(chartGL != null && chartGL != undefined){{chartGL.resize();}} }});
      </script>

    {interesting_periods_section}
  </div>
</body>
</html>
""" # noqa E501

    def __init__(self, strat, benchmark_rets, live_start_date,
                 report_name='Report', custom_interesting_periods=None,
                 custom_interesting_periods_overide=False):
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

        benchmark_rets = pd.merge(pd.DataFrame(returns),
                                  pd.DataFrame(benchmark_rets),
                                  left_index=True, right_index=True,
                                  how="left").fillna(0).iloc[:, 1]

        self.returns = returns
        self.positions = positions
        self.transactions = transactions
        self.gross_lev = gross_lev

        self.benchmark_rets = benchmark_rets
        self.live_start_date = live_start_date
        self.report_name = report_name

        self.periods = custom_interesting_periods
        self.override = custom_interesting_periods_overide

    def build_report(self, dest=None):
        jupyter = True if dest is None else False

        table = self.get_performance_table(jupyter=jupyter)
        table_drawdowns = self.get_drawdown_table(jupyter=jupyter)

        if jupyter:
            display(self.get_interactive_rolling_returns(jupyter=jupyter))
            display(self.get_interactive_rolling_vol(jupyter=jupyter))
            display(self.get_interactive_rolling_sharpes(jupyter=jupyter))
            # display(self.get_interactive_rolling_betas(jupyter=jupyter))
            display(self.get_interactive_monthly_heatmap(jupyter=jupyter))
            display(self.get_interactive_drawdown_and_underwater(jupyter))
            display(self.get_interactive_exposures(jupyter=jupyter))
            display(self.get_interactive_exposures_by_asset(jupyter))
            display(self.get_interactive_gross_leverage(jupyter=jupyter))
            display(self.get_interactive_interesting_periods(jupyter))

        if dest is not None:
            rct_f, rct_o = self.get_interactive_rolling_returns(
                jupyter=jupyter)
            rvol_o = self.get_interactive_rolling_vol(jupyter=jupyter)
            rbeta_o = self.get_interactive_rolling_betas(jupyter=jupyter)
            rcs_o = self.get_interactive_rolling_sharpes(jupyter=jupyter)
            mrh_o = self.get_interactive_monthly_heatmap(jupyter=jupyter)
            dau_o = self.get_interactive_drawdown_and_underwater(jupyter)
            exp_o = self.get_interactive_exposures(jupyter=jupyter)
            expba_o = self.get_interactive_exposures_by_asset(jupyter=jupyter)
            gl_o = self.get_interactive_gross_leverage(jupyter=jupyter)

            interesting_periods = self.get_interesting_periods_section()

            with open(dest, 'w') as report:
                report.write(self.template.format(
                    report_name=self.report_name,
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
                    gl_option=gl_o,
                    interesting_periods_section=interesting_periods))

    def get_performance_table(self, jupyter=True):
        return pyfolio.show_perf_stats(returns=self.returns,
                                       positions=self.positions,
                                       transactions=self.transactions,
                                       live_start_date=self.live_start_date,
                                       jupyter=jupyter)

    def get_drawdown_table(self, jupyter=True):
        return pyfolio.show_worst_drawdown_table(returns=self.returns,
                                                 jupyter=jupyter, pandas=False)

    def get_interactive_rolling_returns(self, jupyter=True):
        plot = plotting.plot_interactive_rolling_returns(
            returns=self.returns,
            factor_returns=self.benchmark_rets,
            live_start_date=self.live_start_date,
            cone_std=[1, 1.5, 2]
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

    def get_interactive_rolling_vol(self, jupyter=True):
        plot = plotting.plot_interactive_rolling_vol(
            returns=self.returns, factor_returns=self.benchmark_rets)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_rolling_betas(self, jupyter=True):
        plot = plotting.plot_interactive_rolling_betas(
            returns=self.returns, factor_returns=self.benchmark_rets)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_rolling_sharpes(self, jupyter=True):
        plot = plotting.plot_interactive_rolling_sharpes(
            returns=self.returns, factor_returns=self.benchmark_rets)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_monthly_heatmap(self, jupyter=True):
        plot = plotting.plot_interactive_monthly_heatmap(returns=self.returns)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_drawdown_and_underwater(self, jupyter=True):
        plot = plotting.plot_interactive_drawdown_underwater(
            returns=self.returns)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_exposures(self, jupyter=True):

        plot = plotting.plot_interactive_exposures(
            returns=self.returns, positions=self.positions)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_exposures_by_asset(self, jupyter=True):
        plot = plotting.plot_interactive_exposures_by_asset(
            positions=self.positions)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_gross_leverage(self, jupyter=True):
        plot = plotting.plot_interactive_gross_leverage(
            positions=self.positions)

        if jupyter:
            return plot
        else:
            return self._echart_option_extract(plot)

    def get_interactive_interesting_periods(self, jupyter=True):
        plot = plotting.plot_interactive_interesting_periods(
            returns=self.returns, benchmark_rets=self.benchmark_rets,
            periods=self.periods, override=self.override)

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
""" # noqa E501
        res = self.get_interactive_interesting_periods(jupyter=False)
        if res is None:
            return ""
        else:
            return template.format(size=res[1], ipa_option=res[0])
