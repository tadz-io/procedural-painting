import pytest  # noqa
import numpy as np
from hypothesis import given
from hypothesis.strategies import tuples, integers
from optimizers import Parameter


def test_update_single_step():
    # single parameter, one step size
    single_param = Parameter(initial=100, step_size=10, min_value=0, max_value=255)
    # multi parameter, one step size
    multi_param = Parameter(
        initial=(100, 100), step_size=10, min_value=(0, 0), max_value=(255, 255)
    )
    # update parameter(s)
    single_param.update(np.random.randint)
    multi_param.update(np.random.randint)
    # checks
    assert isinstance(single_param.value, int)
    assert isinstance(multi_param.value, tuple)
    assert multi_param.value[0] == multi_param.value[1]


def test_update_multi_step():
    # multi parameter, multi step size
    multi_param = Parameter(
        initial=(100, 100), step_size=(10, 20), min_value=(0, 0), max_value=(255, 255)
    )
    multi_param.update(np.random.randint)
    # NOTE: this assertion can fail by chance in case both step sizes turn out to be the same
    assert multi_param.value[0] != multi_param.value[1]


@given(
    initial=tuples(integers(min_value=0, max_value=255), integers(min_value=0, max_value=255)),
    step_size=tuples(integers(min_value=1, max_value=50), integers(min_value=1, max_value=50)),
)
def test_value_bounds(initial, step_size):
    min_value = 0
    max_value = 255
    multi_param = Parameter(
        initial=initial,
        step_size=step_size,
        min_value=(min_value, min_value),
        max_value=(max_value, max_value),
    )

    # Update the parameter
    multi_param.update(np.random.randint)

    assert 0 <= multi_param.value[0] <= 255
    assert 0 <= multi_param.value[1] <= 255
