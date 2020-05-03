def coalesce(a: pd.Series, *rest: pd.Series) -> pd.Series:
    for b in rest:
        a = np.where(np.isnan(a), b, a)
    return pd.Series(a)


a = pd.Series([1, np.nan, 3, np.nan, np.nan, 6])
b = pd.Series([100, 2, 100, np.nan, np.nan, 100])
c = pd.Series([1000, 1000, 1000, 4, 5, 1000])

df = pd.DataFrame(dict(a=a, b=b, c=c))
df
