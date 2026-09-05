import itertools

import numpy as np
import pytest
from p_kit.library.quantum import TransverseFieldIsing
from p_kit.solver import CaSuDaSolver
from p_kit.solver.annealing import constant


def _make(n=3, R=4, gamma=0.5, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    J_q = rng.uniform(-1, 1, (n, n))
    J_q = (J_q + J_q.T) / 2
    np.fill_diagonal(J_q, 0)
    h_q = rng.uniform(-1, 1, n)
    return TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R), n, R


def test_shape():
    c, n, R = _make()
    assert c.n_pbits == n * R
    assert c.J.shape == (n * R, n * R)
    assert c.h.shape == (n * R,)


def test_symmetry():
    c, _, _ = _make()
    np.testing.assert_array_almost_equal(c.J, c.J.T)


def test_zero_diagonal():
    c, n, R = _make()
    np.testing.assert_array_equal(np.diag(c.J), np.zeros(n * R))


def test_intra_replica_scaling():
    n, R, beta = 2, 3, 2.0
    J_q = np.array([[0.0, 1.0], [1.0, 0.0]])
    h_q = np.array([0.5, -0.5])
    c = TransverseFieldIsing(J_q, h_q, gamma=0.0, beta=beta, n_replicas=R)
    scale = beta / R
    for tau in range(R):
        s = tau * n
        np.testing.assert_array_almost_equal(c.J[s:s + n, s:s + n], scale * J_q)
        np.testing.assert_array_almost_equal(c.h[s:s + n], scale * h_q)


def test_no_inter_replica_when_gamma_zero():
    n, R = 2, 3
    J_q = np.array([[0.0, 1.0], [1.0, 0.0]])
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=0.0, n_replicas=R)
    for tau in range(R):
        tau_next = (tau + 1) % R
        assert c.J[tau * n, tau_next * n] == 0.0


def test_inter_replica_coupling_value():
    n, R = 2, 4
    beta, gamma = 1.0, 0.5
    J_q = np.zeros((n, n))
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R)
    K = -0.5 * np.log(np.tanh(beta * gamma / R))
    # qubit 0: replica 0 connected to replica 1 at index [0, n]
    assert c.J[0, n] == pytest.approx(K)
    assert c.J[n, 0] == pytest.approx(K)


def test_r2_no_double_counting():
    """For R=2 the ring has one unique temporal bond per qubit — not two."""
    n, R = 2, 2
    beta, gamma = 1.0, 0.5
    J_q = np.zeros((n, n))
    h_q = np.zeros(n)
    c = TransverseFieldIsing(J_q, h_q, gamma=gamma, beta=beta, n_replicas=R)
    K = -0.5 * np.log(np.tanh(beta * gamma / R))
    assert c.J[0, n] == pytest.approx(K)   # not 2*K


def test_solve():
    c, _, _ = _make(n=2, R=4)
    solver = CaSuDaSolver(Nt=100, dt=0.1667, i0=0.8, seed=42)
    _, all_m, _ = solver.solve(c)
    assert all_m.shape[-1] == c.n_pbits


def _brute_force_qubo_min(qubo):
    n = qubo.shape[0]
    best = None
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        val = x @ qubo @ x
        if best is None or val < best:
            best = val
    return best


@pytest.mark.parametrize("qubo", [
    np.array([[1.0, -2.0], [0.0, 1.0]]),
    np.random.default_rng(0).standard_normal((4, 4)),
])
def test_from_qubo_h_matches_energy(qubo):
    """h_q/j_q must encode -x^T Q x up to an additive constant: the
    "maximize" energy F(s) = h@s + 0.5*s@J@s (the same form used by
    CaSuDaSolver/GibbsSolver) has to differ from -x^T Q x by a constant
    across every spin configuration, not just at the optimum.
    """
    n = qubo.shape[0]
    circuit = TransverseFieldIsing.from_qubo(qubo, gamma=0.0, n_replicas=1)
    J, h = circuit.J, circuit.h

    offsets = []
    for bits in itertools.product([-1, 1], repeat=n):
        s = np.array(bits, dtype=float)
        x = (1 + s) / 2
        F = h @ s + 0.5 * s @ J @ s
        offsets.append(F + x @ qubo @ x)

    np.testing.assert_allclose(offsets, offsets[0])


def test_from_qubo_solver_finds_optimum():
    qubo = np.array([
        [1.0, -2.0, 0.0],
        [0.0, 1.0, -2.0],
        [0.0, 0.0, 1.0],
    ])
    true_min = _brute_force_qubo_min(qubo)

    circuit = TransverseFieldIsing.from_qubo(qubo, gamma=0.0, n_replicas=1)
    solver = CaSuDaSolver(Nt=2000, dt=0.1667, i0=2.0, seed=7)
    _, all_m, _ = solver.solve(circuit, annealing_func=constant)

    x_samples = (all_m[-500:] + 1) / 2
    values = np.einsum("ij,jk,ik->i", x_samples, qubo, x_samples)
    assert np.isclose(values.min(), true_min)
    assert np.mean(np.isclose(values, true_min)) > 0.3
