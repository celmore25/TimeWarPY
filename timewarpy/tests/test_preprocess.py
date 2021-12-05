from .. import preprocess

def test_increment():
    assert preprocess.increment(3) == 4

def test_decrement():
    assert preprocess.decrement(3) == 4