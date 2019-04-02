
def model(X, y, m, **kwargs):
    return m(**kwargs).fit(X, y)
