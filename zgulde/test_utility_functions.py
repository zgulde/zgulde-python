import pytest
from zgulde import *

def test_plucking():
    d = {'x': 1, 'y': 2}

    assert pluck(d, 'x') == [1]
    assert pluck(d, 'y', 'x') == [2, 1]

    x, y = pluck(d, 'x', 'y')
    assert x == 1
    assert y == 2

    assert pluck(d) == []

def test_compose_functions():
    def increment(n):
        return n + 1
    def double(n):
        return n * 2

    assert comp(increment, double)(4) == 9
    assert comp(double, increment)(4) == 10

    assert comp(increment, increment, increment)(4) == 7
    assert comp(increment, double, increment)(4) == 11
    assert comp(increment, increment, double)(4) == 10
    assert comp(double, increment, increment)(4) == 12

    assert comp(double, double, double, double)(4) == 64

    assert comp(increment)(1) == 2

    assert comp()(4) == 4

def test_partition():
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    g = partition(l, 3)
    assert next(g) == [1, 2, 3]
    assert next(g) == [4, 5, 6]
    assert next(g) == [7, 8, 9]
    with pytest.raises(StopIteration):
        next(g)

    assert list(partition(l, 1)) == [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
    assert list(partition(l, 2)) == [[1, 2], [3, 4], [5, 6], [7, 8], [9]]
    assert list(partition(l, 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list(partition(l, 4)) == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]

    assert list(partition(l, 8)) == [[1, 2, 3, 4, 5, 6, 7, 8], [9]]
    assert list(partition(l, 9)) == [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
    assert list(partition(l, 10)) == [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

    assert chunk == partition, 'It has another name'

def test_pipe():
    def increment(n):
        return n + 1
    def double(n):
        return n * 2

    assert pipe(3) == 3

    assert pipe(3, increment) == 4
    assert pipe(3, double) == 6

    assert pipe(3, increment, increment) == 5
    assert pipe(3, double, double) == 12
    assert pipe(3, increment, double) == 8
    assert pipe(3, double, increment) == 7

    assert pipe(3, increment, double, increment, double) == 18
    assert pipe(3, double, double, increment, increment) == 14
    assert pipe(3, increment, increment, double, double) == 20

def test_and_prev():
    l = [1, 2, 3, 4, 5]

    g = and_prev(l)
    assert next(g) == (None, 1)
    assert next(g) == (1, 2)
    assert next(g) == (2, 3)
    assert next(g) == (3, 4)
    assert next(g) == (4, 5)

    assert list(and_prev(l)) == [(None, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

def test_and_next():
    l = [1, 2, 3, 4, 5]

    g = and_next(l)
    assert next(g) == (1, 2)
    assert next(g) == (2, 3)
    assert next(g) == (3, 4)
    assert next(g) == (4, 5)
    assert next(g) == (5, None)

    assert list(and_next(l)) == [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]

def test_prev_and_next():
    l = [1, 2, 3]

    g = prev_and_next(l)
    assert next(g) == (None, 1, 2)
    assert next(g) == (1, 2, 3)
    assert next(g) == (2, 3, None)

    assert list(prev_and_next([1, 2, 3])) == [(None, 1, 2), (1, 2, 3), (2, 3, None)]
