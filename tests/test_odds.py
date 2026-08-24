import numpy as np
import pytest

from mma_efficiency.odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    devig_power,
    devig_proportional,
    devig_shin,
    overround,
    prob_to_american,
)


def test_american_to_decimal_known_values():
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-100) == pytest.approx(2.0)
    assert american_to_decimal(250) == pytest.approx(3.5)
    assert american_to_decimal(-250) == pytest.approx(1.4)


def test_round_trips():
    odds = np.array([-1000.0, -300.0, -150.0, -101.0, 100.0, 150.0, 300.0, 1000.0])
    assert np.allclose(decimal_to_american(american_to_decimal(odds)), odds)
    p = american_to_prob(odds)
    assert np.allclose(american_to_prob(prob_to_american(p)), p)


def test_prob_direction():
    # Favorites imply > 50%, underdogs < 50%
    assert american_to_prob(-200) == pytest.approx(2 / 3)
    assert american_to_prob(200) == pytest.approx(1 / 3)


def test_overround_positive_for_real_prices():
    # A typical UFC book: -300 / +240
    r = overround(american_to_prob(-300), american_to_prob(240))
    assert 1.0 < r < 1.1


def test_devig_proportional_sums_to_one():
    p_a = american_to_prob(np.array([-300.0, -150.0, -110.0]))
    p_b = american_to_prob(np.array([240.0, 130.0, -110.0]))
    pi_a, pi_b = devig_proportional(p_a, p_b)
    assert np.allclose(pi_a + pi_b, 1.0)
    # De-vigged probs keep the ordering of raw probs
    assert np.all((pi_a > pi_b) == (p_a > p_b))


def test_devig_power_sums_to_one():
    p_a = american_to_prob(np.array([-500.0, -120.0]))
    p_b = american_to_prob(np.array([380.0, 100.0]))
    pi_a, pi_b = devig_power(p_a, p_b)
    assert np.allclose(pi_a + pi_b, 1.0, atol=1e-9)


def test_devig_shin_sums_to_one_and_shifts_toward_favorite():
    p_a = american_to_prob(np.array([-400.0, -250.0, -110.0]))
    p_b = american_to_prob(np.array([320.0, 205.0, -110.0]))
    pi_a, pi_b, z = devig_shin(p_a, p_b)
    assert np.allclose(pi_a + pi_b, 1.0, atol=1e-9)
    assert np.all(z >= 0)
    bas_a, _ = devig_proportional(p_a, p_b)
    # Shin's insider-trading correction gives the favorite at least as much
    # probability as proportional scaling does.
    fav_shift = pi_a - bas_a
    assert np.all(fav_shift[p_a > p_b] > -1e-12)


def test_devig_shin_symmetric_prices():
    # Symmetric book: both methods must give 50/50
    p = american_to_prob(np.array([-110.0]))
    pi_a, pi_b, _ = devig_shin(p, p)
    assert pi_a[0] == pytest.approx(0.5, abs=1e-9)
    assert pi_b[0] == pytest.approx(0.5, abs=1e-9)
