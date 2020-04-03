from acrobotics.path.util import is_in_range


def test_is_in_range():
    assert is_in_range(2, -2, 3)
    assert not is_in_range(2, -2, 1)
    assert is_in_range(-5, -6, -3)
    assert not is_in_range(-5, -4, -3)
