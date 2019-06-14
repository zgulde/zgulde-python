from typing import Callable, List, Iterable, Dict, Any
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

def partition(xs: Iterable, chunksize: int):
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

def pipe(v: Any, *fns: List[Callable]):
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
