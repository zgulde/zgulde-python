import re
from typing import Iterable, Tuple, Union

import nltk
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

    >>> s = pd.Series(['a b a b c', 'a b c'])
    >>> top_n_ngrams(s, top_n=3, ngrams=2)
    (a, b)    3
    (b, c)    2
    (b, a)    1
    dtype: int64
    """
    f = comp(pd.Series, partial(nltk.ngrams, n=ngrams), str.split)
    return s.apply(f).stack().value_counts().head(top_n)


def tokenize(string):
    """
    >>> tokenize("Hey, what's going on?")
    "Hey , what ' s going on ?"
    """
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)


def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    return " ".join(stems)


def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    return " ".join(lemmas)


STOPWORDS = stopword_list = nltk.corpus.stopwords.words("english")


def clean(text: str, ascii=False) -> str:
    if ascii:
        text = text.encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub("\s+", " ", text)
    text = re.sub("[^\s\w]", "", text)
    # contractions
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    return text
