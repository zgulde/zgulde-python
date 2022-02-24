import argparse

import pandas as pd
import pandas_profiling as pp

parser = argparse.ArgumentParser()
parser.add_argument("csv_file")
parser.add_argument("output_file", default="report.html")
args = parser.parse_args()

df = pd.read_csv(args.csv_file)
report = pp.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
report.to_file(args.output_file)
