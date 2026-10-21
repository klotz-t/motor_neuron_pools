import numpy as np
from scipy.signal import butter, filtfilt

def constant_drive(
        n_mu: int = 100, 
        t_dur: float = 10000,  
        dt: float = 0.025, 
        mean_CI: float = 9, 
        CCoV: float = 20, 
        ICoV: float = 5, 
        bwC: tuple = (15, 35),
        bwI: tuple = (0, 100),
        seed: float = 42,
):
    """
    Return the cortical drive (in nano Ampere) 
    for a constant effort profile 

    Parameters
    ----------

    n_mu : int , default 100
        Number of simulated motor neurons

    t_dur : float , default 10000
        Duartation of the segment in milliseconds

    dt : float , default 0.025
        Time step size in milliseconds

    mean_CI : float , default 9
        Mean amplitude of the cortical drive in
        nano ampere

    CCoV : float , default 20
        Coefficent of variation (std/mean) of the common drive  

    ICoV : float , default 5
        Coefficent of variation (std/mean) of the 
        independent drive  

    bwC : tuple of float , default (0, 35)
        Bandwidth of the common noise in Hz 

    bwI : tuple of float , default (0, 100)
            Bandwidth of the independent noise in Hz 

    seed : float , default 42
        Seed of the random number generator


    """

    rng = np.random.default_rng(seed)
    fsamp = 1 / dt * 1000
    ndt = int(t_dur / dt) + 1

    # Common noise (shared for all MNs)
    if bwC[0] > 0:
        b_c, a_c = butter(2, [bwC[0], bwC[1]], fs=fsamp, btype="band")
    else:
        b_c, a_c = butter(2, bwC[1], fs=fsamp, btype="lowpass")
    comNoise = filtfilt(b_c, a_c, rng.standard_normal((1, ndt)), axis=1, padlen=int(fsamp), padtype="even")
    comNoise = comNoise / np.std(comNoise) * (CCoV / 100 * mean_CI)

    # Independent noise (per MN)
    if bwI[0] > 0:
        b_i, a_i = butter(2, [bwI[0], bwI[1]], fs=fsamp, btype="band")
    else:
        b_i, a_i = butter(2, bwI[1], fs=fsamp, btype="lowpass")
    indNoise = filtfilt(b_i, a_i, rng.standard_normal((n_mu, ndt)), axis=1, padlen=int(fsamp), padtype="even")
    indNoise = indNoise / np.std(indNoise) * (ICoV / 100 * mean_CI)

    return mean_CI + comNoise + indNoise

def trapezoid_drive(
        n_mu: int = 100, 
        t_plateau: float = 10000,  
        t_ramp = 3000,
        dt: float = 0.025, 
        max_CI: float = 9, 
        min_CI: float = 6,
        CCoV: float = 20, 
        ICoV: float = 5, 
        bwC: tuple = (15, 35),
        bwI: tuple = (0, 100),
        seed: float = 42,
):
    """
    Return the cortical drive (in nano Ampere) 
    for a constant effort profile 

    Parameters
    ----------

    n_mu : int , default 100
        Number of simulated motor neurons

    t_plateau : float , default 10000
        Duartation of the plateau phase in milliseconds

    t_ramp : float , default 3000
        Duartation of the rising and falling ramps
        in milliseconds

    dt : float , default 0.025
        Time step size in milliseconds

    max_CI : float , default 9
        Max amplitude of the cortical drive in
        nano ampere (during the plateau)

    min_CI : float , default 9
            Minimum amplitude of the cortical drive in
            nano ampere (during the plateau)

    CCoV : float , default 20
        Coefficent of variation (std/mean) of the common drive  

    ICoV : float , default 5
        Coefficent of variation (std/mean) of the 
        independent drive  

    bwC : tuple of float , default (0, 35)
        Bandwidth of the common noise in Hz 

    bwI : tuple of float , default (0, 100)
            Bandwidth of the independent noise in Hz 

    seed : float , default 42
        Seed of the random number generator


    """

    rng = np.random.default_rng(seed)
    fsamp = 1 / dt * 1000
    t_dur = t_plateau + 2*t_ramp
    ndt = int(t_dur / dt) + 1


    mean_CI = np.zeros(ndt)
    idx0 = int(t_ramp / dt)
    idx1 = int((t_ramp + t_plateau) / dt)
    idx2 = int((2 * t_ramp + t_plateau) / dt)
    mean_CI[:idx0] = np.linspace(min_CI, max_CI, idx0)
    mean_CI[idx0:idx1] = max_CI
    mean_CI[idx1:idx2] = np.linspace(max_CI, min_CI, idx0)

    # Common noise (shared for all MNs)
    if bwC[0] > 0:
        b_c, a_c = butter(2, [bwC[0], bwC[1]], fs=fsamp, btype="band")
    else:
        b_c, a_c = butter(2, bwC[1], fs=fsamp, btype="lowpass")
    comNoise = filtfilt(b_c, a_c, rng.standard_normal((1, ndt)), axis=1, padlen=int(fsamp), padtype="even")
    comNoise = comNoise / np.std(comNoise) * (CCoV / 100 * max_CI)

    # Independent noise (per MN)
    if bwI[0] > 0:
        b_i, a_i = butter(2, [bwI[0], bwI[1]], fs=fsamp, btype="band")
    else:
        b_i, a_i = butter(2, bwI[1], fs=fsamp, btype="lowpass")
    indNoise = filtfilt(b_i, a_i, rng.standard_normal((n_mu, ndt)), axis=1, padlen=int(fsamp), padtype="even")
    indNoise = indNoise / np.std(indNoise) * (ICoV / 100 * max_CI)

    return mean_CI + comNoise + indNoise