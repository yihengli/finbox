from . import pyfolio


class ReportBuilder(object):

    template = """
<!DOCTYPE html>
<html>

<head>
  <!-- Required meta tags -->
  <meta charset="utf-8">
  <meta name="viewport"
        content="width=device-width, initial-scale=1, shrink-to-fit=no">

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
  </div>
</body>
</html>
"""

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

        if dest is not None:
            with open(dest, 'w') as report:
                report.write(self.template.format(
                    report_name=self.report_name,
                    table=table))

    def get_performance_table(self, jupyter=True):
        return pyfolio.show_perf_stats(returns=self.returns,
                                       positions=self.positions,
                                       transactions=self.transactions,
                                       live_start_date=self.live_start_date,
                                       jupyter=jupyter)
