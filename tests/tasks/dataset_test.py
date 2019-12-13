import pytest

from nnlab.tasks.dataset import rgb_tup, one_hot_tup

def test_rgb_tup():
    assert rgb_tup(0x0000Ff) == (0,0,255)

def test_one_hot_tup():
    assert one_hot_tup(4, 1 << 0) == (0,0,0,1)
    assert one_hot_tup(4, 1 << 1) == (0,0,1,0)
    assert one_hot_tup(4, 1 << 2) == (0,1,0,0)
    assert one_hot_tup(4, 1 << 3) == (1,0,0,0)

@pytest.mark.xfail(raises=AssertionError)
def test_one_hot_tup_assert():
    assert one_hot_tup(4, 1 << 4)
