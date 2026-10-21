import numpy as np
#from openNMS.mn_pool.core import get_cortical_input

class LIFModel:
    """
    Class for predicting motor neuron spike times 
    using a leaky integrate-and-fire (LIF) model.
    The LIF model is solved using a custom backward
    Euler implementation

    Properties
    ----------

    n_mn : int , default 100
        Number of motor neurons in the pool

    dist_exp : float , default 2
        Size distribution exponent 

    param : dict
        Dictonary of model parameters

    References
    ----------

    .. [1] Caillet et al., "Mathematical relationships between
           spinal motorneuron properties", eLife, 2022

    """ 

    def __init__(
            self, 
            n_mn: int = 100,
            d_soma: tuple[float] = (50e-6, 100e-6), # unit: meter
            dist_exp: float = 2
    ):

        self.n_mn = n_mn
        self.dist_exp = dist_exp

        # Get soma diameters and rheobase current from the given parameters
        mni = np.arange(n_mn)

        if n_mn == 1:
            soma_diam = np.array(0.5 * ([d_soma[0]] + [d_soma[1]]))
        else:
            soma_diam = d_soma[0] + (d_soma[1]- d_soma[0]) * ((mni / (n_mn - 1)) ** dist_exp)

        # Define parameter dictionary
        self.param = {
            "soma_diam": soma_diam,
            "V_reset": -70e-3,  # V
            "V_e": -70e-3,      # V
            "V_th": -43e-3 * np.ones(n_mn),  # V (can be array for flexibility)
            "tref": 2.7e-8 / (soma_diam ** 1.51) * 0.2,  # [s] 20% of AHP (see Dernocourt et al 2025)
            "Rm": 2*5.1e-5/(soma_diam**2.43) # A factor of two is hard coded (seem's more realistic)
        }
        self.param["tau_m"] = 7.9e-5 * soma_diam * self.param["Rm"]
        
    
    def simulate(self, CI, dt=0.1):

        """
        Solve the leaky integrate-and-fire model with 
        an backward Euler method.

        Parameters
        ----------

        CI : ndarray 
            Cortical input of shape ``(n_mn, ndt)`` 
            with unit nano ampere

        dt : float , default 0.1
            Time step size in milli seconds

        Returns
        -------
            spike_times (list of np.ndarray): list with spike indices per MN
            Vm (ndarray): membrane potentials (n_mn x len(T))
        """

        # Scale the input current in ampere
        CI = CI * 1e-9

        # Scale the time step in seconds
        dt = dt * 1e-3

        # Unpack temporal parameters
        fsamp = 1 / dt  # in Hz              
        T = np.linspace(0, CI.shape[1]*fsamp, CI.shape[1]) 

        # Initalize the output variables
        n_time = len(T)
        v = np.zeros((self.n_mn, n_time))
        spike_times = [ [] for _ in range(self.n_mn) ]

        # Discretize using an implicit Euler method
        dL = dt * self.param["V_e"] / self.param["tau_m"]
        dI = dt * self.param["Rm"] / self.param["tau_m"]
        iM = 1.0 / (1.0 + dt / self.param["tau_m"])
        tr = np.zeros(self.n_mn, dtype=int)

        # Add peak values of the spikes (only relevant for nicer visualisation)
        spike_peak = 0.03

        # Initalize the membrane potential
        v[:, 0] = self.param["V_reset"]

        # Loop over the time steps
        for t in range(n_time - 1):
            # Integrate input currents
            v[:, t+1] = iM * (v[:, t] + dL + dI * CI[:, t])

            # Apply refractory period
            refractory_idx = np.where(tr > 0)[0]
            v[refractory_idx, t+1] = self.param["V_reset"]
            tr[refractory_idx] -= 1

            # Detect spikes, set spike peak and update buffer variable tracking the refractory time
            spiking_idx = np.where(v[:, t+1] > self.param["V_th"])[0]
            v[spiking_idx, t+1] = spike_peak
            tr[spiking_idx] = np.round(self.param["tref"][spiking_idx] / dt).astype(int)

        # Detect spike times (time sample index)
        for mu_idx in range(self.n_mn):
            spike_indices = np.where(v[mu_idx, :] == spike_peak)[0]
            spike_times[mu_idx] = spike_indices / fsamp

        # Use millivolt scale
        v = v * 1e3
        
        return v, spike_times