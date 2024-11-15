import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
from mne.time_frequency.tfr import tfr_array_morlet
from frites.core import gcmi_nd_cc

###


def _setup_nodes(A: np.ndarray, D: np.ndarray, f: float, fs: float):
    """
    Setup nodes for Kuramoto simulation.

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
    """

    # Check dimensions
    assert A.shape == D.shape

    # Integration time-step
    dt = 1 / fs

    # Make sure they are arrays
    A = np.asarray(A)
    D = np.asarray(D)

    # Number of nodes in the network
    N = A.shape[0]

    # If float convert to array
    if isinstance(f, float):
        f = f * np.ones(N)
    else:
        f = np.asarray(f)

    # Zero delay if there is no connection and convert to time-step
    D = (D * (A > 0) / dt).astype(int)

    # Maximum delay
    max_delay = int(np.max(D) + 1)
    # Revert the Delays matrix such that it contains the index of the History
    # that we need to retrieve at each dt
    D = max_delay - D

    # Scale with dt to avoid doing it in each time-step
    # during numerical integation
    omegas = 2 * np.pi * f * dt
    A = A * dt

    # Randomly initialize phases and keeps it only up to max delay
    phases = 2 * np.pi * np.random.rand(N) + omegas * np.ones(
        (N, max_delay)
    ) * np.arange(max_delay)

    # From 0 to 2\pi
    phases = phases % (2 * np.pi)

    return N, A, D, omegas, phases, dt


def Simulate_Kuramoto_Delay(
    A: np.ndarray, D: np.ndarray, f: float, fs: float, eta: float, T: float, icoup
):

    # Call setup to initialize nodes and scale A and D matrices
    N, A, D, omegas, phases_history, dt = _setup_nodes(A, D, f, fs)

    eta = np.sqrt(dt) * eta

    # Stored phases of each node
    phases = np.zeros((N, T))

    carry = (phases, phases_history)

    def _loop(carry, t):

        phases, phases_history = carry

        phases_t = phases_history[:, -1].copy()

        # Input to each node
        Input = np.zeros(N)

        # Input from coupled units
        for n in range(N):
            I_n = 0  #  Initialize total coupling received into node n
            for p in range(N):
                if A[n, p]:
                    I_n = I_n + A[n, p] * np.sin(
                        phases_history[p, D[n, p]] - phases_t[n]
                    )
            Input[n] = I_n

        phases_history[:, :-1] = phases_history[:, 1:]

        phases_history[:, -1] = (
            phases_t + omegas + icoup[t] * Input + eta * np.random.normal(0, 1, size=N)
        )

        # phases[:, t] = phases_history[:, -1]
        return phases_history[:, -1], phases_history

    for t in range(T):
        phases[:, t], phases_history = _loop(carry, t)
        carry = (phases, phases_history)

    phases_fft = np.fft.fft(np.sin(phases), n=T, axis=1)
    TS = np.real(np.fft.ifft(phases_fft))

    return TS, phases


### Simulated data

# Parameters
ntrials = 500
fs = 600
time = np.arange(-0.5, 1, 1 / fs)
T = len(time)
C = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]).T

data = np.zeros((C.shape[1], ntrials, T))

# Derived parameters
N = C.shape[0]
D = np.ones((N, N)) / 1000  # Fixed delay matrix divided by 1000
C = C / np.mean(C[np.ones((N, N)) - np.eye(N) > 0])
f = 40  # Node natural frequency in Hz
MD = 0.0  # Mean Delay in seconds
K = 50  # Global coupling strength
s = 8 / (2 * np.pi * f)

# Coupling time window
time_start = 0
time_end = 0.4
timestim = time[(time > time_start) & (time < time_end)] - (time_end - time_start) / 2
ind = np.where((time > time_start) & (time < time_end))[0]
gaussian = np.exp(-(timestim**2) / (2 * s**2))
coupling = np.zeros_like(time)
coupling[ind] = gaussian

# Coupling strength array (linearly spaced from 1 to 100)
CS = np.linspace(1, 100, ntrials)

# Placeholder simulation loop with random data
for itrials in tqdm(range(ntrials)):
    # Generate random placeholder data for TS with shape (3, Npoints)
    TS, dt_save = Simulate_Kuramoto_Delay(
        K * C * CS[itrials], D, f, fs, 3.5, T, coupling
    )

    # Extract time series data
    x, y, z = TS[0], TS[1], TS[2]
    data[0, itrials, :] = x
    data[1, itrials, :] = y
    data[2, itrials, :] = z

# Output the shapes of data and datah for verification
data.shape


### Convert to xarray

data = xr.DataArray(
    data.transpose(1, 0, 2),
    dims=("trials", "roi", "times"),
    coords=(CS, range(3), time),
)

### Measuring stimulus-specific information

freqs = np.linspace(10, 80, 50)

W = tfr_array_morlet(
    data.values,
    fs,
    freqs,
    freqs / 7,
    output="complex",
    n_jobs=10,
    zero_mean=False,
)

W = xr.DataArray(
    W,
    dims=("trials", "roi", "freqs", "times"),
    coords={"freqs": freqs, "times": data.times.values},
)

power_amplitude = np.abs(W)
complex_phase = W / power_amplitude
complex_phase_diff = complex_phase.isel(roi=0) * np.conj(complex_phase.isel(roi=2))


power_edge = np.stack(
    (power_amplitude.isel(roi=0), power_amplitude.isel(roi=1)), axis=1
)
phase_edge = np.stack((complex_phase_diff.real, complex_phase_diff.imag), axis=1)

power_phase_edge = np.stack(
    (
        power_amplitude.isel(roi=0),
        power_amplitude.isel(roi=1),
        complex_phase_diff.real,
        complex_phase_diff.imag,
    ),
    axis=1,
)

stim = data.trials.values
stim = np.expand_dims(stim, axis=(0, 1))
stim = np.tile(stim, (len(freqs), W.sizes["times"], 1, 1)).transpose(3, 2, 0, 1)

power_encoding = gcmi_nd_cc(power_edge, stim, mvaxis=1, traxis=0)
phase_encoding = gcmi_nd_cc(phase_edge, stim, mvaxis=1, traxis=0)
power_phase_encoding = gcmi_nd_cc(power_phase_edge, stim, mvaxis=1, traxis=0)

# Redundancy
red = np.minimum(phase_encoding, power_encoding)

# Uniques
unique_power = power_encoding - red
unique_phase = phase_encoding - red

# Synergy
syn = power_phase_encoding - unique_power - unique_phase - red

power_encoding = xr.DataArray(
    power_encoding,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="power encoding [bits]",
).squeeze()

phase_encoding = xr.DataArray(
    phase_encoding,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="phase encoding [bits]",
).squeeze()

unique_power = xr.DataArray(
    unique_power,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="phase encoding [bits]",
).squeeze()

unique_phase = xr.DataArray(
    unique_phase,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="phase encoding [bits]",
).squeeze()

syn = xr.DataArray(
    syn,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="phase encoding [bits]",
).squeeze()

red = xr.DataArray(
    red,
    dims=("freqs", "times"),
    coords=(freqs, W.times.values),
    name="phase encoding [bits]",
).squeeze()


### Plot

plt.figure(figsize=(15, 5))
plt.subplot(121)
power_encoding.plot.imshow(cmap="turbo", vmin=0)
plt.title("Power only")
plt.subplot(122)
phase_encoding.plot.imshow(cmap="turbo", vmin=0)
plt.title("Phase only")
plt.show()

plt.figure(figsize=(15, 10))
plt.subplot(221)
unique_power.plot.imshow(cmap="turbo", vmin=0, vmax=0.5)
plt.title("Unique power")
plt.subplot(222)
unique_phase.plot.imshow(cmap="turbo", vmin=0, vmax=0.5)
plt.title("Unique phase")
plt.subplot(223)
red.plot.imshow(cmap="turbo", vmin=0, vmax=0.5)
plt.title("Redundance")
plt.subplot(224)
syn.plot.imshow(cmap="turbo", vmin=0, vmax=0.5)
plt.title("Synergy")

plt.show()
