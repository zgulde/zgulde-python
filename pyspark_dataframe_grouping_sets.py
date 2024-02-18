'''
emulates the behavior of Spark SQL's [GROUPING SETS][1] by making multiple dataframes and
using pyspark.sql.DataFrame.unionByName

[1]:https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-groupby.html

Examples

>>> import pyspark.sql
>>> import plotly.express as px # for datasets
>>> spark = pyspark.sql.SparkSession.builder.getOrCreate()
>>> df = spark.createDataFrame(px.data.tips())
>>> df.groupingSets([], ['time'], ['day']).aggExpr('SUM(total_bill) AS sales').show()
    +------+----+------------------+
    |  time| day|             sales|
    +------+----+------------------+
    |  NULL|NULL| 4827.769999999999|
    |Dinner|NULL|3660.2999999999997|
    | Lunch|NULL|           1167.47|
    |  NULL| Sun|1627.1599999999999|
    |  NULL| Sat|1778.3999999999996|
    |  NULL|Thur|           1096.33|
    |  NULL| Fri|            325.88|
    +------+----+------------------+
>>> df.groupingSets('day', 'smoker').pivot('time').aggExpr('SUM(total_bill) AS sales').show()
    +----+------+------------------+-------+
    | day|smoker|            Dinner|  Lunch|
    +----+------+------------------+-------+
    |Thur|  NULL|             18.78|1077.55|
    | Sun|  NULL|1627.1599999999999|   NULL|
    | Sat|  NULL|1778.3999999999996|   NULL|
    | Fri|  NULL|235.95999999999998|  89.92|
    |NULL|    No|           2130.14| 767.29|
    |NULL|   Yes|           1530.16| 400.18|
    +----+------+------------------+-------+
'''
from functools import reduce
import pyspark.sql
import pyspark.sql.functions as F

class GroupingSetsAggregations:
    def __init__(self, df, sets):
        self.df = df
        for s in sets:
            if isinstance(s, str) or isinstance(s, list):
                continue
            else:
                raise TypeError(f'GroupingSets expects sets to be strings or lists of strings, instead got {s=} {type(s)=}')
        self.sets = [[s] if isinstance(s, str) else s for s in sets]
        self._pivot = None

    def pivot(self, *cols):
        self._pivot = cols
        return self

    def aggExpr(self, *exprs):
        return self.agg(*[F.expr(e) for e in exprs])

    def agg(self, *aggs):
        frames = []
        for s in self.sets:
            if self._pivot:
                frames.append(self.df.groupby(*s).pivot(*self._pivot).agg(*aggs))
            else:
                frames.append(self.df.groupby(*s).agg(*aggs))
        df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)

        grouping_cols = set(reduce(lambda a, b: a + b, self.sets, []))
        other_cols = [c for c in df.columns if c not in grouping_cols]
        return df.select(*grouping_cols, *other_cols)

pyspark.sql.DataFrame.groupingSets = lambda df, *sets: GroupingSetsAggregations(df, sets)
