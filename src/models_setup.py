import numpy as np


def _set_nodes(A: np.ndarray, f: float, fs: float, a: float):
    """
    Setup nodes for Kuramoto simulation without time delays.

    Parameters
    ----------
    A : np.ndarray
        Binary or weighted adjacency matrix.
    f : float or array_like
        Natural oscillating frequency [in Hz] of each node.
        If float all Kuramoto oscillatiors have the same frequency
        otherwise each oscillator has its own frequency.
    fs: float
        Sampling frequency for simulating the network.
    a: float
        Branching parameter

    Returns
    -------
    N: int
        Number of nodes
    A : np.ndarray
        Adjacency matrix rescaled with dt.
    phases: np.ndarray
        Initialize container with phase values.
    dt: float
        Integration time-step
    a: float
        Branching parameter
    """

    # Integration time-step
    dt = 1 / fs

    # Make sure elays(C==0)=0;they are arrays
    A = np.asarray(A)

    # Number of nodes in the network
    N = A.shape[0]

    # If float convert to array
    if isinstance(f, float):
        f = f * np.ones(N)
    else:
        f = np.asarray(f)

    # If float convert to array
    if isinstance(a, float):
        a = a * np.ones(N)
    else:
        a = np.asarray(a)

    omegas = 2 * np.pi * f

    # Randomly initialize phases and keeps it only up to max delay
    # phases = 2 * np.pi * np.random.rand(N, 1) + omegas * np.ones((N, 1)) * np.arange(1)
    phases = 1e-4 * (np.random.rand(N, 1) + 1j * np.random.rand(N, 1))

    # From 0 to 2\pi
    # phases = phases % (2 * np.pi)

    return N, A, omegas, phases, dt, a


def _set_nodes_delayed(A: np.ndarray, D: np.ndarray, f: float, fs: float, a: float):
    """
    Setup nodes for Kuramoto simulation with time delays.

    Parameters
    ----------
    A : np.ndarray
        Binary or weighted adjacency matrix.
    D : np.ndarray
        Contain the delay if connections among nodes in seconds.
    f : float or array_like
        Natural oscillating frequency [in Hz] of each node.
        If float all Kuramoto oscillatiors have the same frequency
        otherwise each oscillator has its own frequency.
    fs: float
        Sampling frequency for simulating the network.
    a: float
        Branching parameter

    Returns
    -------
    N: int
        Number of nodes
    A : np.ndarray
        Adjacency matrix rescaled with dt.
    D: np.ndarray
        Delays in timesteps.
    phases: np.ndarray
        Initialize container with phase values.
    dt: float
        Integration time-step
    a: float
        Branching parameter
    """

    # Check dimensions
    assert A.shape == D.shape

    # Call config for nodes without delay
    N, A, omegas, _, dt, a = _set_nodes(A, f, fs, a)

    # Work on the delay matrix
    D = np.asarray(D)

    # Zero delay if there is no connection and convert to time-step
    D = np.round(D * (A > 0) / dt).astype(int)

    # Maximum delay
    max_delay = int(np.max(D) + 1)
    # Revert the Delays matrix such that it contains the index of the History
    # that we need to retrieve at each dt
    D = max_delay - D

    # Randomly initialize phases and keeps it only up to max delay
    phases = 2 * np.pi * np.random.rand(N, 1) + omegas * np.ones(
        (N, max_delay)
    ) * np.arange(max_delay)

    # From 0 to 2\pi
    # phases = phases % (2 * np.pi)

    return N, A, D, omegas, phases.astype(np.complex128), dt, a
