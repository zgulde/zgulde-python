from typing import Callable, List, Iterable, Dict, Any, TypeVar, Tuple, Optional, Sequence
import itertools as it
from functools import reduce, partial

def pluck(d: Dict, *ks):
    '''
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
    '''
    return [d[k] for k in ks]

def comp(*fns: Iterable[Callable]):
    '''
    Returns a function that is the passed functions composed together. Functions
    are applied from right to left.

    comp(f, g)(x) == f(g(x))

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
    '''
    return partial(reduce, lambda x, f: f(x), reversed(fns))

def partition(xs: Sequence, chunksize: int):
    '''
    Partition a sequence into smaller subsequences

    Returns a generator that yields the smaller partitions

    >>> letters = 'abcdefghijklm'
    >>> partition(letters, 2)
    <generator object partition at ...>
    >>> list(partition(letters, 2))
    ['ab', 'cd', 'ef', 'gh', 'ij', 'kl', 'm']
    >>> list(partition(letters, 3))
    ['abc', 'def', 'ghi', 'jkl', 'm']
    '''
    for i in range(0, len(xs), chunksize):
        yield xs[i:i + chunksize]

chunk = partition

def pipe(v: Any, *fns: Callable):
    '''
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
    '''

    return reduce(lambda x, f: f(x), fns, v)

A = TypeVar('A')
def and_prev(xs: Iterable[A]) -> Iterable[Tuple[Optional[A], A]]:
    '''
    Return each item of the iterable xs, along with the previous item.

    The first previous item is None.

    Returns
    -------

    An iterable of tuples, where each tuple is the previous item and the current
    item.

    >>> list(and_prev([1, 2, 3]))
    [(None, 1), (1, 2), (2, 3)]
    '''
    return zip(it.chain([None], xs), xs)

def and_next(xs: Iterable[A]) -> Iterable[Tuple[A, Optional[A]]]:
    '''
    Return each of the items in the iterable xs, along with the next item.

    When the iterable is exhausted, the last item is None.

    Returns
    -------

    An iterable of tuples, where each tuple is the current item and the next
    item.

    >>> list(and_next([1, 2, 3]))
    [(1, 2), (2, 3), (3, None)]
    '''
    return it.zip_longest(xs, it.islice(xs, 1, None))

def prev_and_next(xs: Iterable[A]) -> Iterable[Tuple[Optional[A], A, Optional[A]]]:
    '''
    Return each of the items in the iterable xs, along with the previous and
    next elements.

    For the first item, the previous is None, for the last item, the next is
    None.

    Returns
    -------

    An iterable of tuples, where each tuple is (prev, current, next)

    >>> list(prev_and_next([1, 2, 3]))
    [(None, 1, 2), (1, 2, 3), (2, 3, None)]
    '''
    prev, current, next = it.tee(xs, 3)
    return zip(it.chain([None], prev),
               current,
               it.chain(it.islice(next, 1, None), [None]))

