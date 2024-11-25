import numpy as np
import xarray as xr
from mne.time_frequency.tfr import tfr_array_morlet
from src.models import KuramotoOscillators
from frites.core import mi_nd_gg, copnorm_nd
from frites.core import mi_model_nd_gd
import matplotlib.pyplot as plt


# Parameters
ntrials = 200
fsamp = 600
time = np.arange(-0.5, 1, 1 / fsamp)
Npoints = len(time)
C = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
data = np.zeros((C.shape[1], ntrials, Npoints))

# Derived parameters
N = C.shape[0]
D = np.ones((N, N)) / 1000  # Fixed delay matrix divided by 1000
C = C / np.mean(C[np.ones((N, N)) - np.eye(N) > 0])
f = 40  # Node natural frequency in Hz
K = 10  # Global coupling strength
s = 8 / (2 * np.pi * f)

time_start = 0
time_end = 0.4
timestim = time[(time > time_start) & (time < time_end)] - (time_end - time_start) / 2
ind = np.where((time > time_start) & (time < time_end))[0]
gaussian = np.exp(-(timestim**2) / (2 * s**2))
coupling = np.ones((N, Npoints))
coupling[:, ind] = gaussian

# Coupling strength array (linearly spaced from 1 to 100)
CS = np.linspace(1, 100, ntrials)

# Placeholder simulation loop with random data
for itrials in range(ntrials):
    # Generate random placeholder data for TS with shape (3, Npoints)
    TS, dt_save = KuramotoOscillators(
        K * C, f, fsamp, 5, Npoints, None, CS[itrials] * coupling
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

### Decompose in time-frequency domain

freqs = np.linspace(10, 80, 50)

S = tfr_array_morlet(
    data.values,
    fsamp,
    freqs,
    freqs / 7,
    output="complex",
    n_jobs=10,
    zero_mean=False,
)

S = xr.DataArray(
    S,
    dims=("trials", "roi", "freqs", "times"),
    coords={"freqs": freqs, "times": data.times.values},
)

xy = S.sel(roi=0) * np.conj(S.sel(roi=1))
yz = S.sel(roi=0) * np.conj(S.sel(roi=2))

xy = xr.concat((xy.real, xy.imag), "mvaxis").transpose(
    "trials", "mvaxis", "freqs", "times"
)
yz = xr.concat((yz.real, yz.imag), "mvaxis").transpose(
    "trials", "mvaxis", "freqs", "times"
)

x_ = xr.concat((xy, yz), "roi").transpose("trials", "roi", "mvaxis", "freqs", "times")


# Simple Feature Specific Information

mvaxis = 1
x = x_.values
y = data.trials.values
y_ = np.atleast_2d(y)[np.newaxis, ...]
y_ = np.tile(y, (x.shape[0], 1, 1))

n_delays = 5
# Quantity of delays for which to comput FIT
max_delay = n_delays / fsamp
n_delays = int(np.round(max_delay * fsamp))

# Copula-normalization
x = copnorm_nd(x, axis=-1)

x_s = x[:, 0]
x_t = x[:, 1]

mi_s = mi_model_nd_gd(x_s, y, mvaxis=mvaxis, traxis=0)
mi_t = mi_model_nd_gd(x_t, y, mvaxis=mvaxis, traxis=0)

mi_sp_tf = np.zeros((n_delays, n_freqs, n_times), dtype=np.float32)
mi_tp_tf = np.zeros((n_delays, n_freqs, n_times), dtype=np.float32)

for n_d in range(n_delays):
    # define indices
    idx_past = slice(n_d, n_d + n_times - n_delays - 1)
    idx_pres = slice(n_delays + 1, n_times)

    # source past; target past; target present
    _sp = x_s[..., idx_past]
    _tp = x_t[..., idx_past]
    _tf = x_t[..., idx_pres]

    mi_tp_tf[n_d, :, idx_pres] = mi_nd_gg(_tp, _tf, mvaxis=mvaxis, traxis=0)

# time indices for target roi
t_start = list(range(n_delays, n_times))
fit = np.zeros((n_delays, n_freqs, n_times - n_delays), dtype=np.float32)

# I(target_pres; cue)
mi_t_pres = mi_t[..., t_start]

# I(source_past; target_pres)
mi_sp_tf_pres = mi_sp_tf[..., t_start]

# I(target_past; target_pres) = mi_x_t
mi_tp_tf_pres = mi_tp_tf[..., t_start]

for n_d in range(n_delays):

    delays = list(range(n_d, n_times - n_delays + n_d))

    # PID with cue as target var

    # I(target_{past}; cue)
    mi_t_past = mi_t[..., delays]
    # I(source_{past}; cue)
    mi_s_past = mi_s[..., delays]

    # redundancy between sources and target about S (MMI-based)
    red_s_t = np.minimum(mi_s_past, mi_t_pres)
    # redundancy between sources, target present and target past about S
    red_all = np.minimum(red_s_t, mi_t_past)
    # first term of FIT with the cue as target var
    fit_cue = red_s_t - red_all

    # PID with target pres as target var
    # redundancy between sources and target about target pres (MMI-based)
    red_t = np.minimum(mi_t_pres, mi_sp_tf_pres[n_d, :])
    # redundancy between sources, target present and target past about S
    red_all = np.minimum(red_t, mi_t_pres[n_d, :])
    # second term of FIT with x pres as target var
    fit_t_pres = red_t - red_all

    fit += np.minimum(fit_cue, fit_t_pres)


fit_ = xr.DataArray(
    fit.mean(0), dims=("freqs", "times"), coords=(freqs, ar.times.values[n_delays:])
)

fit_.plot.imshow(cmap="turbo")
plt.show()
