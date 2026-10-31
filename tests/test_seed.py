import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascadelens.seed import derive_case_seed, lightgbm_seed_params, set_seed


def test_set_seed_repeats_python_and_numpy_draws() -> None:
    set_seed(7)
    first = (random.random(), np.random.random())
    set_seed(7)

    assert (random.random(), np.random.random()) == first


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_set_seed_rejects_out_of_range(seed: int) -> None:
    with pytest.raises(ValueError):
        set_seed(seed)


def test_lightgbm_params_carry_seed() -> None:
    assert lightgbm_seed_params(3) == {"seed": 3, "deterministic": True, "force_row_wise": True}


@given(
    case_id=st.text(),
    perturbation=st.text(),
    rate=st.floats(allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_case_seed_is_deterministic_and_in_range(
    case_id: str, perturbation: str, rate: float, seed: int
) -> None:
    derived = derive_case_seed(case_id, perturbation, {"rate": rate}, seed)

    assert derived == derive_case_seed(case_id, perturbation, {"rate": rate}, seed)
    assert 0 <= derived < 2**31


def test_case_seed_ignores_param_order() -> None:
    assert derive_case_seed("c", "P1", {"a": 1, "b": 2}, 0) == derive_case_seed(
        "c", "P1", {"b": 2, "a": 1}, 0
    )


def test_case_seed_is_a_fixed_value() -> None:
    # hash() would fail this across processes because of string hash randomization.
    assert derive_case_seed("case-1", "P1", {"rate": 0.3}, 0) == 490034621


@pytest.mark.parametrize(
    ("case_id", "perturbation", "rate", "seed"),
    [
        ("case-2", "P1", 0.3, 0),
        ("case-1", "P2", 0.3, 0),
        ("case-1", "P1", 0.5, 0),
        ("case-1", "P1", 0.3, 1),
    ],
)
def test_case_seed_depends_on_every_input(
    case_id: str, perturbation: str, rate: float, seed: int
) -> None:
    assert derive_case_seed(case_id, perturbation, {"rate": rate}, seed) != derive_case_seed(
        "case-1", "P1", {"rate": 0.3}, 0
    )
