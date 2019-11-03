from .ds_util import MyRange
import pytest

@pytest.fixture()
def r():
    return MyRange()

def test_it_generates_a_range(r):
    assert r[5:8] == [5, 6, 7, 8]
    assert r[5:8:2] == [5, 7]
    assert r[:5] == [0, 1, 2, 3, 4, 5]
    assert r[5:] == []
