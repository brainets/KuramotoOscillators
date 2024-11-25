##
import numpy as np
from .models_setup import _set_nodes, _set_nodes_delayed

"""
FUNCTIONS LOOP MIGHT BE COMPILED WITH JAX SCAN LATER
"""


def _ode(Z: np.complex128, a: float, w: float):
    return Z * (a + 1j * w - np.abs(Z) ** 2)


def _loop(carry, t, dt):

    N, A, Iext, omegas, a, eta, phases_history = carry

    phases_t = phases_history.squeeze().copy()

    phase_differences = phases_history - phases_t

    # Input to each node
    Input = (A * phase_differences).sum(axis=1) + Iext[:, t] * np.exp(
        1j * np.angle(phases_t)
    )

    phases_history = (
        phases_t
        + dt * (_ode(phases_t, a, omegas) + Input)
        + eta * (np.random.normal(size=N) + 1j * np.random.normal(size=N))
    )
    return phases_history.reshape(N, 1)


def _loop_delayed(carry, t, dt):

    N, A, D, omegas, a, eta, phases_history = carry

    phases_t = phases_history[:, -1].copy()

    """
    THE IDEA IS TO VECTORIZE THIS
    """

    def _return_phase_differences(n, d):
        return phases_history[np.indices(d.shape)[0], d - 1] * phases_t[n]

    phase_differences = np.stack(
        [_return_phase_differences(n, d) for n, d in enumerate(D)]
    )

    # Input to each node
    Input = (A * phase_differences).sum(axis=1)

    # Slide History only if the delays are > 0
    phases_history[:, :-1] = phases_history[:, 1:]

    phases_history = (
        phases_t
        + dt * (_ode(phases_t, a, omegas) + Input)[:, None]
        + eta * (np.random.normal(size=N) + 1j * np.random.normal(size=N))
    )

    return phases_history


def KuramotoOscillators(
    A: np.ndarray,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    D: np.ndarray = None,
    Iext: np.ndarray = None,
):

    if isinstance(D, np.ndarray) and D.any():
        N, A, D, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)
        carry = [N, A, D, omegas, a, eta * np.sqrt(dt), phases_history]
        _loop_fun = _loop_delayed
    else:
        N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)
        carry = [N, A, Iext, omegas, a, eta * np.sqrt(dt), phases_history]
        _loop_fun = _loop

    # Stored phases of each node
    phases = np.zeros((N, T), dtype=np.complex128)

    for t in range(T):
        # Update
        phases_history = _loop_fun(carry, t, dt)
        # print(phases_history.shape)
        # Store
        phases[:, t] = phases_history[:, -1]
        # New carry only phases history changes
        carry[-1] = phases_history

    return phases, True


###

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ### Simulated data

    # Parameters
    fs = 600
    time = np.arange(-0.5, 1, 1 / fs)
    T = len(time)
    C = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]).T

    # Derived parameters
    N = C.shape[0]
    D = 10 * np.ones((N, N))  # Fixed delay matrix divided by 1000
    C = C / np.mean(C[np.ones((N, N)) - np.eye(N) > 0])
    f = 40  # Node natural frequency in Hz
    K = 10  # Global coupling strength

    # Generate random placeholder data for TS with shape (3, Npoints)
    TS, dt_save = KuramotoOscillators(K * C, f, fs, 0 * 3.5, T, D, np.ones(T))

    # Extract time series data
    x, y, z = TS[0], TS[1], TS[2]

    plt.plot(time, x)
    plt.plot(time, y)
    # plt.plot(time, z)
