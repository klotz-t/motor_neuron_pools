
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, irfft
from scipy.ndimage import map_coordinates
from typing import Literal

def convolutive_mixture(
        muaps: np.ndarray, 
        spikes: dict,
        t_dur: float, 
        fsamp: float = 2048, 
        method: Literal["sparse", "fft_conv"] = "sparse"
):
    """
    Obtain EMG through a convolutive mixture
    of spikes and muaps 

    Parameters
    ----------
    muaps : np.ndarray 
        Array of muaps with shape ``(units, channels, samples)`` 

    spikes : dict
        Dictonary of spike times (in seconds)

    t_dur : float
        Duration of the signal in seconds

    fsamp : float , default 2048
        Sampling frequency in Hz

    method : {"sparse", "fft_conv"}  , default sparse
        Method used for the convolution. If "sparse" (default), 
        the function loops over all spikes and inserts 
        the given waveform templates. If "fft_conv" the signal 
        is reconstructed by convolving (fft-based)
        the waveform with the spike train.  

    Returns
    -------

    emg : np.ndarray 
        EMG signal of shape ``(n_channels, n_samples)``

    """

    nunits = muaps.shape[0]
    nchan = muaps.shape[1]
    width = muaps.shape[2]
    nsamples = int(t_dur * fsamp) + 1
    emg = np.zeros((nchan, nsamples))

    for mu_idx in range(nunits):

        mu_spikes = (np.array(spikes[mu_idx]) * fsamp).astype(int)

        if len(mu_spikes) == 0:
            continue

        if method == "sparse":
            comp_sig = np.zeros_like(emg)
            for s in mu_spikes:
                comp_sig[:, s:s+width] = muaps[mu_idx, :, :]

        elif method == "fft_conv":
            firings = np.zeros(emg.shape[1])
            firings[mu_spikes] = 1

            # Zero-pad waveform to match signal shape
            L = emg.shape[1]
            pad_len = L - width
            waveform_padded = np.pad(
                muaps[mu_idx, :, :], ((0, 0), (0, pad_len)), mode="constant"
            )

            # FFT of firings (same for all channels)
            fft_firings = rfft(firings)

            # FFT of waveform for each channel
            fft_waveform = rfft(waveform_padded, axis=1)

            # Multiply in frequency domain (broadcasting firings to each channel)
            fft_product = fft_waveform * fft_firings

            # IFFT to get time domain component signal
            comp_sig = irfft(fft_product, n=L, axis=1)

            # # Correct time shift due to FFT convolution (center of kernel)
            # shift = (waveform.shape[1] - 1) // 2
            # comp_sig = np.roll(comp_sig, -shift, axis=1)

        emg += comp_sig

    return emg

def dynamic_convolutive_mixture(
        muaps: np.ndarray, 
        spikes: dict,
        position: np.ndarray, 
        fsamp: float = 2048, 
):
    """
    Obtain dynamic EMG through a convolutive mixture
    of spikes and position dependent muaps 

    Parameters
    ----------
    muaps : np.ndarray 
        Array of muaps with shape ``(units, positions, channels, samples)`` 

    spikes : dict
        Dictonary of spike times (in seconds)

    position : np.ndarray
        Array encoding the position index 
    
    fsamp : float , default 2048
        Sampling frequency in Hz

    method : {"sparse", "fft_conv"}  , default sparse
        Method used for the convolution. If "sparse" (default), 
        the function loops over all spikes and inserts 
        the given waveform templates. If "fft_conv" the signal 
        is reconstructed by convolving (fft-based)
        the waveform with the spike train.  

    Returns
    -------

    emg : np.ndarray 
        EMG signal of shape ``(n_channels, n_samples)``

    """


    nunits = muaps.shape[0]
    nchan = muaps.shape[1]
    width = muaps.shape[2]
    nsamples = position.shape[0] 
    emg = np.zeros((nchan, nsamples))

    for mu_idx in range(nunits):

        mu_spikes = (np.array(spikes[mu_idx]) * fsamp).astype(int)

        if len(mu_spikes) == 0:
            continue

        comp_sig = np.zeros_like(emg)
        for s in mu_spikes:
            pos_idx = position[s]
            comp_sig[:, s:s+width] = muaps[mu_idx, pos_idx :, :]

        emg += comp_sig

    return emg

def generate_emg_noise(
    n_channels,
    n_samples,
    fs,
    med_psd,
    alpha=0.0,
    line_freq=50.0,
    line_snr_db=20.0,
    bandwidth=(5, 600),
    rng=42,
    independent_line_phase=True,
):
    """
    Generate 1/f**alpha colored noise plus sinusoidal power-line interference.

    Parameters
    ----------
    n_channels : int
        Number of channels

    n_samples : int
        Number of samples

    fs : float
        Sampling frequency in Hz

    med_psd : float
        Target mean one-sided PSD in the 20-500 Hz band,
        in signal_unit**2/Hz (e.g. µV²/Hz)

    alpha : float , default 0
        PSD shape exponent: 0=white, 1=pink, 2=brown-like.

    line_freq : float , default 50
        Power-line frequency in Hz

    line_snr_db : float , default 20
        Sinusoidal power relative to the local noise PSD
        in a 1-Hz reference bandwidth, in dB

    bandwidth : tuple of float or None, default (5, 600)
        Bandwidth of the noise. If None, the full bandwith 
        is considered

    rng : int , default 42
        Random seed or NumPy random generator

    independent_line_phase : bool , default True
        If True, each channel has an independent line phase.

    Returns
    -------
    
    noise : ndarray
        Noise with shape ``(n_channels, n_samples)``
    """
    if n_channels < 1 or n_samples < 2:
        raise ValueError(
            "Invalid channel or sample count."
        )
    if fs < 1000:
        raise ValueError(
            "fs must be at least 1000 Hz for the 20-500 Hz band."
        )
    if med_psd < 0 or alpha < 0:
        raise ValueError(
            "mean_psd and alpha must be nonnegative."
        )
    if not 0 < line_freq < fs / 2:
        raise ValueError(
            "line_freq must lie below Nyquist."
        )
    if not (bandwidth is None):
        if bandwidth[0] < 2.5:
            raise ValueError(
                "Lower bound bandwidth must be at least 2.5 Hz."
            )
        if bandwidth[1] > fs/2:
            raise ValueError(
                "Upper bound bandwidth can't exceed the Nyquits frequency."
            )
    

    rng = np.random.default_rng(rng)
    f = np.fft.rfftfreq(n_samples, d=1 / fs)
    n_freq = len(f)

    # Shape the PSD, excluding DC.
    shape = np.zeros(n_freq)
    pos = f > 0
    shape[pos] = f[pos] ** (-alpha)

    band = (f >= 20) & (f <= 500)
    if not np.any(band):
        raise ValueError("Record is too short to resolve the 20-500 Hz band.")

    psd = shape * (med_psd / np.median(shape[band]))

    # Random Fourier coefficients with the desired one-sided PSD.
    X = np.zeros((n_channels, n_freq), dtype=complex)
    interior = np.ones(n_freq, dtype=bool)
    interior[0] = False
    if n_samples % 2 == 0:
        interior[-1] = False

    X[:, interior] = (
        rng.normal(size=(n_channels, interior.sum()))
        + 1j * rng.normal(size=(n_channels, interior.sum()))
    ) * np.sqrt(n_samples * fs * psd[interior] / 4)[None, :]

    # Real-valued Nyquist coefficient for even-length records.
    if n_samples % 2 == 0:
        X[:, -1] = (
            rng.normal(size=n_channels)
            * np.sqrt(n_samples * fs * psd[-1])
        )

    colored = np.fft.irfft(X, n=n_samples, axis=1)

    # Set line power relative to local background PSD × 1 Hz.
    idx = np.argmin(np.abs(f - line_freq))
    line_power = psd[idx] * 10 ** (line_snr_db / 10)
    line_peak = np.sqrt(2 * line_power)

    t = np.arange(n_samples) / fs
    if independent_line_phase:
        phase = rng.uniform(0, 2 * np.pi, (n_channels, 1))
    else:
        phase = 0.0

    line = line_peak * np.sin(
        2 * np.pi * line_freq * t[None, :] + phase
    )

    # Apply a bandpass filter
    if not (bandwidth is None):
        b, a = butter(2, [bandwidth[0], bandwidth[1]], fs=fs, btype="band")
        noise = filtfilt(
            b, a, colored + line, axis=1
        )
    else:
        noise = colored + line

    return noise


def moving_grid(
    emg,
    x_center,
    y_center,
    sx=13,
    sy=5,
    reference_spacing=4.0,
    moving_spacing=4.0,
):
    """
    Obtain non-stationary EMG signals by virtually moving a small
    EMG grid over a larger reference grid.

    Parameters
    ----------
    emg : ndarray
        Reference EMG data of shape (rows, columns, samples).

    x_center : ndarray
        X coordinate of the virtual grid center in mm,
        with shape (samples,).

    y_center : ndarray
        Y coordinate of the virtual grid center in mm,
        with shape (samples,).

    sx : int, default 13
        Number of rows in the virtual grid.

    sy : int, default 5
        Number of columns in the virtual grid.

    reference_spacing : float, default 4.0
        Interelectrode distance of the reference grid in mm.

    moving_spacing : float, default 4.0
        Interelectrode distance of the virtual moving grid in mm.

    Returns
    -------
    new_emg : ndarray
        Interpolated EMG data of shape ``(sx, sy, samples)``.
        Values outside the reference grid are set to NaN.
    """
    _, _, nt = emg.shape

    x_center = np.asarray(x_center)
    y_center = np.asarray(y_center)

    if x_center.shape != (nt,) or y_center.shape != (nt,):
        raise ValueError(
            "x_center and y_center must each have shape (samples,)."
        )

    if reference_spacing <= 0 or moving_spacing <= 0:
        raise ValueError("Grid spacings must be positive.")

    # Virtual electrode coordinates relative to the grid center (mm).
    dx = (np.arange(sx) - (sx - 1) / 2) * moving_spacing
    dy = (np.arange(sy) - (sy - 1) / 2) * moving_spacing

    # Physical coordinates of each virtual electrode (mm).
    x = x_center[:, None, None] + dx[None, :, None]
    y = y_center[:, None, None] + dy[None, None, :]

    # Convert physical coordinates to reference-grid indices.
    ix = x / reference_spacing
    iy = y / reference_spacing

    new_emg = np.empty((sx, sy, nt), dtype=float)

    for t in range(nt):
        coords = np.array([
            np.broadcast_to(ix[t], (sx, sy)),
            np.broadcast_to(iy[t], (sx, sy)),
        ])

        new_emg[:, :, t] = map_coordinates(
            emg[:, :, t],
            coords,
            order=1,
            mode="constant",
            cval=np.nan,
        )

    return new_emg



