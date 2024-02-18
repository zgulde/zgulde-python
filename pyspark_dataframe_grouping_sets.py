# emulates the behavior of Spark SQL's [GROUPING SETS][1] by making multiple dataframes and
# using pyspark.sql.DataFrame.unionByName
#
# [1]:https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-groupby.html
class GroupingSetsAggregations:
    def __init__(self, df, sets):
        self.df = df
        for s in sets:
            if isinstance(s, str) or isinstance(s, list):
                continue
            else:
                raise TypeError(f'GroupingSets expects sets to be strings or lists of strings, instead got {s=} {type(s)=}')
        self.sets = [[s] if isinstance(s, str) else s for s in sets]

    def pivot(self, *cols):
        self._pivot = cols
        return self

    def aggExpr(self, *exprs):
        return self.agg(*[F.expr(e) for e in exprs])

    def agg(self, *aggs):
        frames = []
        for set in self.sets:
            if self._pivot:
                frames.append(self.df.groupby(*set).pivot(*self._pivot).agg(*aggs))
            else:
                frames.append(self.df.groupby(*set).agg(*aggs))
        df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)

        grouping_cols = __builtins__.set(__builtins__.sum(self.sets, []))
        other_cols = [c for c in df.columns if c not in grouping_cols]
        return df.select(*grouping_cols, *other_cols)

pyspark.sql.DataFrame.groupingSets = lambda df, *sets: GroupingSetsAggregations(df, sets)
