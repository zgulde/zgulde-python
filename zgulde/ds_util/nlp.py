from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(documents: Iterable[str]) -> pd.DataFrame:
    """
    >>> docs = ['first article', 'second article', 'mary had a little lamb, little lamb']
    >>> bag_of_words(docs)
       article  first  had  lamb  little  mary  second
    0        1      1    0     0       0     0       0
    1        1      0    0     0       0     0       1
    2        0      0    1     2       2     1       0
    """
    cv = CountVectorizer()
    m = cv.fit_transform(documents).todense()
    return pd.DataFrame(m, columns=cv.get_feature_names())
