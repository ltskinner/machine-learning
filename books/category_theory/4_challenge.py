
import math
import numpy as np


class Optional(object):
    def __init__(self, value=None):
        if value is None:
            self.isValid = False
            self.value = value
        elif isinstance(value, Optional):
            self.isValid = value.isValid
            self.value = value.value
        else:
            self.isValid = True
            self.value = value


def identity(x):
    return Optional(x)


# bc non tuples we dont need to over engineer this
compose = lambda m1, m2: (
    lambda x: m2(
        m1(x).value
    )
)


def safe_root(x: float) -> Optional:
    if isinstance(x, (int, float)) and x >= 0:
        return Optional(math.sqrt(x))
    return Optional()

def safe_reciprocal(x: float) -> Optional:

    if isinstance(x, (int, float)) and x != 0:
        return Optional(1/x)
    return Optional()

safe_root_reciprocal = compose(safe_root, safe_reciprocal)


if __name__ == '__main__':
    # stolen test cases from:
    # https://danshiebler.com/2018-11-10-category-solutions/


    assert not safe_root(-1).isValid
    assert np.isclose(safe_root(4).value, 2.0)

    assert not safe_reciprocal(0).isValid
    assert np.isclose(safe_reciprocal(4).value, 0.25)

    assert not safe_root_reciprocal(0).isValid
    assert not safe_root_reciprocal(-5).isValid
    assert np.isclose(safe_root_reciprocal(0.25).value, 2)
