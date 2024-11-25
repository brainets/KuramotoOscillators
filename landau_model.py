import numpy as np


def _ode(Z: np.complex128, a: float, w: float):
    return Z * (a + 1j * w - np.abs(Z * Z))


if __name__ == "__main__":

    # Model patameters
    w_0 = 2 * np.pi * 40
    a = -5
    beta = 1

    # Simulation parameters
    dt = 0.0001
    T = np.arange(0, 1, dt)

    Z = np.ones(T.shape, dtype=np.complex128)

    for i, t in enumerate(T[:-1]):
        Z[i + 1] = (
            Z[i]
            + dt * _ode(Z[i], a, w_0)
            + np.sqrt(dt) * beta * (np.random.normal() + 1j * np.random.normal())
        )
