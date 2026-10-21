import numpy as np
import jax.numpy as jnp
import jaxley as jx

from jaxley.channels import Channel
from jaxley.solver_gate import solve_gate_exponential

from scipy.signal import find_peaks


class NegroFarina:
    """
    Class for predicting motor neuron spike times using 
    a two-compartment Hodkin-Huxley model. The 
    implementation of the biophyical neuron models make 
    use of the Jaxley framewok.

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
    .. [1] Negro and Farina, "Decorrelation of cortical inputs 
           and motoneuron output", Journal of Neurophysiology,  2011
    .. [2] Cisi and Kohn, "Simulation system of spinal cord motor 
           nuclei and associated nerves and muscles, in a 
           Web-based architecture", Journal of Computational 
           Neuroscience, 2008
    .. [3] Schmid et al., "Postinhibitory excitation in 
           motoneurons can be facilitated by hyperpolarization-activated
           inward currents: A simulation study", PLOS Computational
           Biology, 2024
    .. [4] Deistler et al., "Jaxley: differentiable simulation enables 
           large-scale training of detailed biophysical models of 
           neural dynamics", Nature Methods, 2025

    """ 

    def __init__(self, 
                 n_mn: int = 100,
                 dist_exp: float = 2,
                 r_s: tuple = (77.5/2, 113/2),  # Radius of soma compartment in micro meter
                 l_s: tuple = (77.5, 113),      # Length of soma compartment in micro meter
                 r_d: tuple = (41.5/2, 92.5/2), # Radius of dendritic compartment in micro meter
                 l_d: tuple = (0.55e4, 1.06e4), # Length of dendritic compartment in micro meter
                 rL_s: tuple = (1.15, 0.65),    # Soma specific membrane resistacne in kilo Ohm square cm
                 rL_d: tuple = (14.4, 6.05)     # Dendritic sepecific membrane resistance in kilo Ohm square cm
    ):

        self.n_mn = n_mn
        self.dist_exp = 2

        # Index for MNs
        mni = np.arange(self.n_mn)

        # Define parameter dictionary
        if n_mn == 1:
            self.param = {
                "r_s": np.array(0.5 * ([r_s[0]] + [r_s[1]])), 
                "l_s": np.array(0.5 * ([l_s[0]] + [l_s[1]])),      
                "r_d": np.array(0.5 * ([r_d[0]] + [r_d[1]])),  
                "l_d": np.array(0.5 * ([l_d[0]] + [l_d[1]])),  
                "rL_s": np.array(0.5 * ([rL_s[0]] + [rL_s[1]])),
                "rL_d": np.array(0.5 * ([rL_d[0]] + [rL_d[1]]))
            }
        else:
            self.param = {
                "r_s": r_s[0] + (r_s[1]- r_s[0]) * ((mni / (n_mn - 1)) ** dist_exp), 
                "l_s": l_s[0] + (l_s[1]- l_s[0]) * ((mni / (n_mn - 1)) ** dist_exp),      
                "r_d": r_d[0] + (r_d[1]- r_d[0]) * ((mni / (n_mn - 1)) ** dist_exp),  
                "l_d": l_d[0] + (l_d[1]- l_d[0]) * ((mni / (n_mn - 1)) ** dist_exp),  
                "rL_s": rL_s[0] + (rL_s[1]- rL_s[0]) * ((mni / (n_mn - 1)) ** dist_exp),
                "rL_d": rL_d[0] + (rL_d[1]- rL_d[0]) * ((mni / (n_mn - 1)) ** dist_exp)
            }

        # Build the model
        self._make_pool()


    def _make_cell(
                self,
                r_s=77.5/2,
                l_s=77.5,
                r_d=41.5/2,
                l_d=0.55e4,
                rL_s=1.15,
                rL_d=14.4
        ):
            """
            Helper function to generate a single two compartement 
            motor neuron 

            """
    
    
            # Create compartments
            soma_comp = jx.Compartment()
            dend_comp = jx.Compartment()
    
            # Create branches
            soma = jx.Branch(soma_comp, ncomp=1)
            dend = jx.Branch(dend_comp, ncomp=1)
    
            soma.set("length", l_s)
            soma.set("radius", r_s)
    
            dend.set("length", l_d)
            dend.set("radius", r_d)
    
            # Build a cell morphology
            # branch 0 = soma
            # branch 1 = dendrite attached to branch 0
    
    
            cell = jx.Cell(
                [soma, dend],
                parents=[-1, 0]
            )
    
            cell.branch(0).insert(_MNFastNa())
            cell.branch(0).insert(_MNFastK())
            cell.branch(0).insert(_MNSlowK())
            cell.branch(0).insert(_MNLeak())
    
            cell.branch(1).insert(_MNLeak())
    
            cell.set("axial_resistivity", 70.0)
    
            cell.set("v", 0.0)
    
    
            cell.branch(0).set("gLeak", 1e-3 / rL_s)
            cell.branch(1).set("gLeak", 1e-3 / rL_d)
    
            return cell

    def _make_pool(self):
            """
            Set up the motor neuron pool model as 
            a Jaxley-network

            """
        
            neurons = [
                self._make_cell(
                    r_s = self.param["r_s"][idx],
                    l_s = self.param["l_s"][idx],
                    r_d = self.param["r_d"][idx],
                    l_d = self.param["l_d"][idx],
                    rL_s = self.param["rL_s"][idx],
                    rL_d = self.param["rL_d"][idx]
                )
                for idx in range(self.n_mn)
            ]
    
            self.pool = jx.Network(neurons)
    

    def simulate(self, CI, dt=0.025):

        """
        Solve the motor neuron pool model given 
        the specified injected input current

        Parameters
        ----------

        CI : ndarray 
            Cortical drive of shape ``(n_mn, ndt)`` and
            unit nano Ampere

        dt : float , default 0.025
            Time step that is used to integrate the motor
            neuron pool model in milliseconds

        Returns
        -------

        v : ndarray
            Array of shape ``(n_mn, ndt)`` representing the
            transmembrane potential in milli volts

        spike_times : dict
            Dictonary of spike times

        """

        # Duration of the simulation
        t_dur = (CI.shape[1]-1) * dt

        # Init the model
        self.pool.init_states()
        self.pool.delete_stimuli()

        # Set stimuli
        stim_sites = self.pool.cell(np.arange(self.n_mn)).branch(0).comp(0)   

        # Output the soma membrane voltage of each MN
        for idx in range(self.n_mn):
            self.pool.cell(idx).branch(0).comp(0).record("v")

        # Apply the stimuls
        CI = jnp.array(CI, dtype=jnp.float32)
        data_stimuli = stim_sites.data_stimulate(CI)

        v = jx.integrate(
            self.pool,
            delta_t=dt,
            t_max=t_dur,
            data_stimuli=data_stimuli
        )
        v.block_until_ready()

        spike_times = self._get_spikes(v, dt=dt)

        return v, spike_times

    def _get_spikes(self, v, dt=0.025, min_dist=2):
        """
        Extract the spike times of the motor neurons
        given the simulated membrane voltage
        
        """

        fsamp = 1 / dt * 1000 # Unit: Hz

        spikes = {i: [] for i in range(self.n_mn)}

        for i in range(self.n_mn):

            pks, _ = find_peaks(v[i, :], height=20, distance=int(min_dist/dt))
            if len(pks) > 0:
                spikes[i] = pks / fsamp

        return spikes


class _MNFastNa(Channel):

    def __init__(self):
        self.current_is_in_mA_per_cm2 = True

        super().__init__()

        self.channel_params = {
            "Na_gNa": 30.0e-3,   # S/cm2
            "eNa": 120,    # mV
        }

        self.channel_states = {
            "m": 0.0293,
            "h": 0.9959,
        }
        self.current_name = "i_Na"


    def update_states(self, states, dt, v, params):

        m = states["m"]
        h = states["h"]

        alpha_m = 0.32 * (13 - v) / (jnp.exp((13 - v) / 5) - 1) 
        beta_m = 0.28 * (v - 40) / (jnp.exp((v - 40) / 5) - 1) 
        tau_m =  1 / (alpha_m + beta_m)
        m_inf = alpha_m * tau_m
        m_new = m_inf + (m - m_inf) * jnp.exp(-dt / tau_m)

        alpha_h = 0.128 * jnp.exp((17 - v) / 18)
        beta_h  = 4 / (jnp.exp((40 - v) / 5)+ 1)
        tau_h = 1 / (alpha_h + beta_h)
        h_inf = alpha_h * tau_h
        h_new = h_inf + (h - h_inf) * jnp.exp(-dt / tau_h)

        return {
            "m": m_new,
            "h": h_new,
        }


    def compute_current(self, states, v, params):

        INa = (
            params["Na_gNa"]
            * states["m"]**3
            * states["h"]
            * (v-params["eNa"])
        )

        # mS/cm2 * mV = uA/cm2
        # convert to mA/cm2
        return INa 
    
    def init_state(self, states, v, params, delta_t):
        return {"m": 0.0293, "h": 0.9959}
    
class _MNFastK(Channel):

    def __init__(self):
        self.current_is_in_mA_per_cm2 = True
        super().__init__()

        self.channel_params = {
            "Kf_gKf": 4e-3, #4.0e-3, # S/cm^2
            "eKf": -10,
        }

        self.channel_states = {
            "n": 0.0381,
        }
        self.current_name = "i_Kf"

    def update_states(self, states, dt, v, params):


        n = states["n"]

        alpha_n = (0.032 * (15 - v)) / (jnp.exp((15 - v) / 5) - 1) 
        beta_n = 0.5 * jnp.exp((10 - v) / 40)
        tau_n = 1 / (alpha_n + beta_n)
        n_inf = alpha_n * tau_n
        n_new = n_inf + (n - n_inf) * jnp.exp(-dt / tau_n)

        return {
            "n": n_new
        }  


    def compute_current(self, states, v, params):

        IK = (
            params["Kf_gKf"]
            * states["n"]**4
            * (v-params["eKf"])
        )

        return IK 
    
    def init_state(self, states, v, params, delta_t):
        return {"n": 0.0381}
    
class _MNSlowK(Channel):
    """ 
    Slow potassium Channel (Traub et al 1991)
    """

    def __init__(self):
        self.current_is_in_mA_per_cm2 = True
        super().__init__()

        self.channel_params = {
            "Ks_gKs": 16e-3,#16.0e-3, # S/cm^2
            "eKs": -10,
        }

        self.channel_states = {
            "q": 1.5155e-04,
        }
        self.current_name = "i_Ks"

    def update_states(self, states, dt, v, params):


        q = states["q"]

        alpha_q = 3.5 / (jnp.exp((55 -v) / 4) + 1)
        beta_q = 0.025
        tau_q = 1.0 / (alpha_q + beta_q)
        q_inf = alpha_q * tau_q

        q_new = q_inf + (q - q_inf) * jnp.exp(-dt / tau_q)

        # dq = alpha_q * (1.0 - q) - beta_q * q
        # q_new = q + dt * dq

        return {
            "q": q_new
        }  
        # return q_new  

    def compute_current(self, states, v, params):

        IK = (
            params["Ks_gKs"]
            * states["q"]**2
            * (v-params["eKs"])
        )

        return IK 
    
    def init_state(self, states, v, params, delta_t):
        return {"q": 1.5155e-04}    


class _MNLeak(Channel):

    def __init__(self):
        self.current_is_in_mA_per_cm2 = True

        super().__init__()

        self.channel_params = {
            "gLeak": 1.0e-3,      # S/cm2
            "eLeak": 0,    # mV
        }

        # Required: no state variables
        self.channel_states = {}
        self.current_name= "i_Leak"

    def update_states(self, states, dt, v, params):
        return {}   


    def compute_current(self, states, v, params):

        ILeak = (
            params["gLeak"]
            * (v - params["eLeak"])
        )

        return ILeak   
    
    def init_state(self, states, v, params, delta_t):
        return {}