from typing import Iterable, Union, Tuple

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from toolz import comp, partial


def tfidfs(
    documents: Iterable[str], return_vectorizer=False
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, sklearn.feature_extraction.text.TfidfVectorizer]
]:
    """
    >>> docs = [
    ...     "doc number one",
    ...     "this is the second one",
    ...     "this is the third doc"
    ... ]
    >>> tfidfs(docs)
            doc        is    number       one    second       the     third      this
    0  0.517856  0.000000  0.680919  0.517856  0.000000  0.000000  0.000000  0.000000
    1  0.000000  0.417796  0.000000  0.417796  0.549351  0.417796  0.000000  0.417796
    2  0.417796  0.417796  0.000000  0.000000  0.000000  0.417796  0.549351  0.417796
    """
    vectorizer = TfidfVectorizer()
    m = vectorizer.fit_transform(documents).todense()
    df = pd.DataFrame(m, columns=vectorizer.get_feature_names())
    if return_vectorizer:
        return df, vectorizer
    else:
        return df


def bag_of_words(
    documents: Iterable[str], return_cv=False
) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, sklearn.feature_extraction.text.CountVectorizer]
]:
    """
    >>> docs = ["first article", "second article", "mary had a little lamb, little lamb"]
    >>> bag_of_words(docs)
       article  first  had  lamb  little  mary  second
    0        1      1    0     0       0     0       0
    1        1      0    0     0       0     0       1
    2        0      0    1     2       2     1       0
    """
    cv = CountVectorizer()
    m = cv.fit_transform(documents).todense()
    bow = pd.DataFrame(m, columns=cv.get_feature_names())
    if return_cv:
        return bow, cv
    else:
        return bow


def top_n_ngrams(s: pd.Series, top_n=3, ngrams=3):
    """
    Extract the top_n number of ngrams (n=ngrams) from s. s is assumed to be a
    pandas Series full of text data.
    """
    f = comp(pd.Series, partial(nltk.ngrams, n=ngrams), str.split)
    return s.apply(f).stack().value_counts().head(top_n)
