from . import pyfolio
from . import plotting
from IPython.core.display import display


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
      
      <div id="rolling_cum_sharpes" style="width: 100%;height:400px;"></div>
      <script type="text/javascript">
        var chartRCS = echarts.init(document.getElementById('rolling_cum_sharpes'));
        var optionRCS = {rcs_option};
        chartRCS.setOption(optionRCS);
        $(window).on('resize', function(){{
          if(chartRCS != null && chartRCS != undefined){{chartRCS.resize();}} }});
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
  </div>
</body>
</html>
""" # noqa E501

    def __init__(self, strat, benchmark_rets, live_start_date,
                 report_name='Report'):
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()

        self.returns = returns
        self.positions = positions
        self.transactions = transactions
        self.gross_lev = gross_lev

        self.benchmark_rets = benchmark_rets
        self.live_start_date = live_start_date
        self.report_name = report_name

    def build_report(self, dest=None):
        jupyter = True if dest is None else False

        table = self.get_performance_table(jupyter=jupyter)
        table_drawdowns = self.get_drawdown_table(jupyter=jupyter)

        if jupyter:
            display(self.get_interactive_rolling_returns(jupyter=jupyter))
            display(self.get_interactive_rolling_sharpes(jupyter=jupyter))
            display(self.get_interactive_drawdown_and_underwater(jupyter))

        if dest is not None:
            rct_f, rct_o = self.get_interactive_rolling_returns(
                jupyter=jupyter)
            rcs_o = self.get_interactive_rolling_sharpes(jupyter=jupyter)
            dau_o = self.get_interactive_drawdown_and_underwater(jupyter)

            with open(dest, 'w') as report:
                report.write(self.template.format(
                    report_name=self.report_name,
                    table=table,
                    table_dd=table_drawdowns,
                    rct_func=rct_f,
                    rct_option=rct_o,
                    rcs_option=rcs_o,
                    dau_option=dau_o))

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

    def get_interactive_rolling_sharpes(self, jupyter=True):
        plot = plotting.plot_interactive_rolling_sharpes(
            returns=self.returns, factor_returns=self.benchmark_rets)

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
