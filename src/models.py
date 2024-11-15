##
import numpy as np
from .models_setup import _set_nodes, _set_nodes_delayed


def _loop(carry, t):

    N, A, omegas, eta, coupling, phases_history = carry

    phases_t = phases_history.squeeze().copy()

    phase_differences = phases_history - phases_t
    phase_differences = np.sin(-phase_differences)

    # Input to each node
    Input = (A * phase_differences * coupling[t]).sum(axis=1)

    phases_history = phases_t + omegas + Input + eta * np.random.normal(0, 1, size=N)
    return phases_history.reshape(N, 1)


def _loop_delayed(carry, t):

    N, A, D, omegas, eta, coupling, phases_history = carry

    phases_t = phases_history[:, -1].copy()

    def _return_phase_differences(n, d):
        phase_differences = phases_history[np.indices(d.shape)[0], d - 1] - phases_t[n]
        return np.sin(-phase_differences)

    phase_differences = np.stack(
        [_return_phase_differences(n, d) for n, d in enumerate(D)]
    )

    # Input to each node
    Input = (A * phase_differences * coupling[t]).sum(axis=1)

    # Slide History only if the delays are > 0
    phases_history[:, :-1] = phases_history[:, 1:]

    phases_history[:, -1] = (
        phases_t + omegas + Input + eta * np.random.normal(0, 1, size=N)
    )

    return phases_history


def KuramotoOscillators(
    A: np.ndarray,
    f: float,
    fs: float,
    eta: float,
    T: float,
    D: np.ndarray = None,
    coupling: np.ndarray = None,
):

    if not isinstance(coupling, (list, tuple, np.ndarray)):
        coupling = np.ones(T)
    else:
        coupling = np.asarray(coupling)

    if isinstance(D, np.ndarray) and D.any():
        N, A, D, omegas, phases_history, dt = _set_nodes_delayed(A, D, f, fs)
        carry = [N, A, D, omegas, eta * np.sqrt(dt), coupling, phases_history]
        _loop_fun = _loop_delayed
    else:
        N, A, omegas, phases_history, dt = _set_nodes(A, f, fs)
        carry = [N, A, omegas, eta * np.sqrt(dt), coupling, phases_history]
        _loop_fun = _loop

    # Stored phases of each node
    phases = np.zeros((N, T))

    for t in range(T):
        # Update
        phases_history = _loop_fun(carry, t)
        # Store
        phases[:, t] = phases_history[:, -1]
        # New carry only phases history changes
        carry[-1] = phases_history

    phases_fft = np.fft.fft(np.sin(phases), n=T, axis=1)
    TS = np.real(np.fft.ifft(phases_fft))

    return TS, phases


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
