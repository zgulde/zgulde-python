"""
A module full of general utility functions.
"""
import collections
import hashlib
import itertools as it
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)


def md5(s):
    return hashlib.md5(s.encode("utf8")).hexdigest()


def sha256(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def slurp(fp):
    with open(fp) as f:
        return f.read()


def spit(fp, content, mode="w+"):
    with open(fp) as f:
        f.write(content)


def pluck(d: Dict, *ks: str):
    """
    Pluck specified values from a dictionary.

    This can make for concise code together with unpacking syntax.

    Example
    -------

    >>> d = {'x': 1, 'y': 2}
    >>> pluck(d, 'x')
    [1]
    >>> pluck(d, 'y', 'x')
    [2, 1]
    >>> x, y = pluck(d, 'x', 'y')
    >>> x
    1
    >>> y
    2
    """
    return [d[k] for k in ks]


def comp(*fns: Callable):
    """
    Returns a function that is the passed functions composed together. Functions
    are applied from right to left.

    comp(f, g)(x) == f(g(x))
    comp(f, g, h)(x) == f(g(h(x)))

    Example
    -------

    >>> def double(n):
    ...     print(f'Doubling {n}...')
    ...     return n * 2
    >>> def inc(n):
    ...     print(f'Incrementing {n}...')
    ...     return n + 1
    >>> double_plus_one = comp(double, inc)
    >>> double_plus_one(3)
    Incrementing 3...
    Doubling 4...
    8
    >>> double_plus_one(5)
    Incrementing 5...
    Doubling 6...
    12
    """
    return lambda x: reduce(lambda x, f: f(x), reversed(fns), x)


def partition(xs: Sequence, chunksize: int):
    """
    Partition a sequence into smaller subsequences

    Returns a generator that yields the smaller partitions

    >>> letters = 'abcdefghijklm'
    >>> partition(letters, 2)
    <generator object partition at ...>
    >>> list(partition(letters, 2))
    ['ab', 'cd', 'ef', 'gh', 'ij', 'kl', 'm']
    >>> list(partition(letters, 3))
    ['abc', 'def', 'ghi', 'jkl', 'm']
    """
    for i in range(0, len(xs), chunksize):
        yield xs[i : i + chunksize]


chunk = partition


def pipe(v: Any, *fns: Callable):
    """
    Thread a value through one or more functions.

    Functions are applied left to right

    >>> def double(n):
    ...     print(f'Doubling {n}...')
    ...     return n * 2
    >>> def inc(n):
    ...     print(f'Incrementing {n}...')
    ...     return n + 1
    >>> pipe(3, inc, double)
    Incrementing 3...
    Doubling 4...
    8
    """

    return reduce(lambda x, f: f(x), fns, v)


def take(xs, n):
    return list(it.islice(xs, n))


def tail(xs, n):
    return list(collections.deque(xs, maxlen=n))


def drop(xs, n):
    return it.islice(xs, n, None)


def prepend(xs, v):
    return it.chain([v], xs)


def append(xs, v):
    return it.chain(xs, [v])


A = TypeVar("A")


def and_prev(xs: Iterable[A]) -> Iterator[Tuple[Optional[A], A]]:
    """
    Return each item of the iterable xs, along with the previous item.

    The first previous item is None.

    Returns
    -------

    An iterable of tuples, where each tuple is the previous item and the current
    item.

    >>> list(and_prev([1, 2, 3]))
    [(None, 1), (1, 2), (2, 3)]
    """
    return zip(prepend(xs, None), xs)


def and_next(xs: Iterable[A]) -> Iterator[Tuple[A, Optional[A]]]:
    """
    Return each of the items in the iterable xs, along with the next item.

    When the iterable is exhausted, the last item is None.

    Returns
    -------

    An iterable of tuples, where each tuple is the current item and the next
    item.

    >>> list(and_next([1, 2, 3]))
    [(1, 2), (2, 3), (3, None)]
    """
    return it.zip_longest(xs, drop(xs, 1))


def prev_and_next(xs: Iterable[A]) -> Iterator[Tuple[Optional[A], A, Optional[A]]]:
    """
    Return each of the items in the iterable xs, along with the previous and
    next elements.

    For the first item, the previous is None, for the last item, the next is
    None.

    Returns
    -------

    An iterable of tuples, where each tuple is (prev, current, next)

    >>> list(prev_and_next([1, 2, 3]))
    [(None, 1, 2), (1, 2, 3), (2, 3, None)]
    """
    prev, current, next = it.tee(xs, 3)
    return zip(prepend(prev, None), current, drop(append(next, None), 1))


class MyRange:
    """
    A simple class that defines a convenient (if not conventional) range syntax

    >>> r = MyRange()
    >>> r[1:10]
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> r[1:10:2]
    [1, 3, 5, 7, 9]
    """

    def __getitem__(_, aslice: slice):
        start = aslice.start if aslice.start is not None else 0
        step = aslice.step if aslice.step is not None else 1
        stop = aslice.stop + 1 if aslice.stop is not None else 1
        return list(range(start, stop, step))
