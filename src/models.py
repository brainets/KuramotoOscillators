import numpy as np
from .models_setup import _setup_nodes


def Kuramoto(A: np.ndarray, f: float, fs: float, eta: float, T: float):

    # Call setup to initialize nodes and scale A and D matrices
    N, A, omegas, phases_history, dt = _setup_nodes(A, f, fs)

    # Rescale noise with \sqrt{dt}
    eta = np.sqrt(dt) * eta

    # Stored phases of each node
    phases = np.zeros((N, T))

    carry = (phases, phases_history)

    def _loop(carry, t):

        phases, phases_history = carry

        phases_t = phases_history[:, 1].copy()

        phase_differences = phases_history[:, 0][:, np.newaxis] - phases_t
        phase_differences = -np.sin(phase_differences)

        # Input to each node
        Input = (A * phase_differences).sum(1)

        phases_history[:, 1] = (
            phases_t + omegas + Input + eta * np.random.normal(0, 1, size=N)
        )

        # phases[:, t] = phases_history[:, -1]
        return phases_history[:, -1], phases_history

    for t in range(T):
        phases[:, t], phases_history = _loop(carry, t)
        carry = (phases, phases_history)

    phases_fft = np.fft.fft(np.sin(phases), n=T, axis=1)
    TS = np.real(np.fft.ifft(phases_fft))

    return TS, phases
