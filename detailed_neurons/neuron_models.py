import numpy as np

from scipy.linalg import block_diag
from scipy.integrate import ode
from scipy.optimize import minimize

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.params import Default, NumberParam
from nengo.dists import Uniform
from nengo.neurons import *
from nengo.builder.neurons import *
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver
from nengo.base import ObjView
from nengo.builder import Builder, Operator, Signal
from nengo.exceptions import BuildError
from nengo.builder.connection import build_decoders, BuiltConnection
from nengo.utils.builder import full_transform

from nengolib.signal import s, z, nrmse, LinearSystem
from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim

import os

import warnings

import neuron
neuron.h.load_file('stdrun.hoc')
neuron.h.load_file('NEURON/cells.hoc')
# neuron.h.load_file('NEURON/durstewitzDArandom.hoc')
# neuron.h.load_file('NEURON/durstewitzDA.hoc')
# neuron.h.load_file('NEURON/durstewitz.hoc')


__all__ = ['LIF', 'AdaptiveLIFT', 'WilsonEuler', 'WilsonRungeKutta', 'DurstewitzNeuron',
    'SimNeuronNeurons', 'TransmitSpikes', 'BiasSpikes', 'TransmitSignal', 'reset_neuron']

class AMPA(nengo.synapses.Synapse):
    def __init__(self):
        super().__init__()
        
class GABA(nengo.synapses.Synapse):
    def __init__(self):
        super().__init__()

class NMDA(nengo.synapses.Synapse):
    def __init__(self):
        super().__init__()

class LIF(LIF):

    probeable = ("spikes", "voltage", "refractory_time")

    def __init__(self, max_x=1.0, tau_rc=0.02, tau_ref=0.002, amplitude=1):
        super().__init__(tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        spiked = np.zeros_like(gain)
        voltage = np.zeros_like(gain)
        refractory_time = np.zeros_like(gain)

        return settled_firingrate(
            self.step_math, J, states=[voltage, refractory_time],
            dt=0.001, settle_time=0.1, sim_time=1.0)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > 1
        spiked[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = 1 and solve for t to compute the spike time
        t_spike = dt + self.tau_rc * np.log1p(
            -(voltage[spiked_mask] - 1) / (J[spiked_mask] - 1)
        )

        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike



class AdaptiveLIFT(LIF):
    
    ''' Aaron Voelker, https://github.com/nengo/nengo/issues/1423'''
    
    probeable = ('spikes', 'voltage', 'refractory_time', 'threshold')

    min_voltage = NumberParam('min_voltage', high=0)
    tau_adapt = NumberParam('tau_adapt', low=0)
    inc_adapt = NumberParam('inc_adapt', low=0)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, min_voltage=0,
                 amplitude=1, tau_adapt=0.1, inc_adapt=0.1):
        super(AdaptiveLIFT, self).__init__(tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)
        # self.min_voltage = min_voltage
        self.tau_adapt = tau_adapt
        self.inc_adapt = inc_adapt
        
    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        voltage = np.zeros_like(gain)
        refractory_time = np.zeros_like(gain)
        threshold = np.ones_like(gain)

        return settled_firingrate(
            self.step_math, J, [voltage, refractory_time, threshold],
            dt=0.001, settle_time=0.3, sim_time=1.0)
    
    def gain_bias(self, max_rates, intercepts):
        return NeuronType.gain_bias(self, max_rates, intercepts)

    def max_rates_intercepts(self, gain, bias):
        return NeuronType.max_rates_intercepts(self, gain, bias)

    def step_math(self, dt, J, spiked, voltage, refractory_time, threshold):
        # reduce all refractory times by dt
        refractory_time -= dt

        # compute effective dt for each neuron, based on remaining time.
        # note that refractory times that have completed midway into this
        # timestep will be given a partial timestep, and moreover these will
        # be subtracted to zero at the next timestep (or reset by a spike)
        delta_t = (dt - refractory_time).clip(0, dt)

        # update voltage using discretized lowpass filter
        # since v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) assuming
        # J is constant over the interval [t, t + dt)
        voltage -= (J - voltage) * np.expm1(-delta_t / self.tau_rc)

        # determine which neurons spiked (set them to 1/dt, else 0)
        spiked_mask = voltage > threshold
        spiked[:] = spiked_mask * (self.amplitude / dt)

        # set v(0) = threshold and solve for t to compute the spike time
        # TODO: not sure if this mask is the right way to handle log domain errors
        threshold_spiked = threshold[spiked_mask]
        m = (voltage[spiked_mask] - threshold_spiked) / (J[spiked_mask] - threshold_spiked)
        t_spike = np.zeros_like(m)
        t_spike[m < 1] = dt + self.tau_rc * np.log1p(-m[m < 1])

        # update threshold using discretized lowpass filter
        # applied to the input 1 + spiked * inc_adapt 
        threshold -= ((1 + self.inc_adapt * spiked - threshold) *
                      np.expm1(-dt / self.tau_adapt))
        
        # set spiked voltages to zero, refractory times to tau_ref, and
        # rectify negative voltages to a floor of min_voltage
        voltage[voltage < self.min_voltage] = self.min_voltage
        voltage[spiked_mask] = 0
        refractory_time[spiked_mask] = self.tau_ref + t_spike


class WilsonEuler(NeuronType):
    '''
    Todo: nice description
    '''
    probeable = ('spikes', 'voltage', 'recovery', 'conductance', 'AP')
    threshold = NumberParam('threshold')
    tau_V = NumberParam('tau_V')
    tau_R = NumberParam('tau_R')
    tau_H = NumberParam('tau_H')
    
    _v0 = -0.754  # initial voltage
    _r0 = 0.279  # initial recovery
    _maxJ = 2.0  # clip input current at this maximum to avoid catastrophic shutdown
    
    def __init__(self, threshold=-0.20, tau_V=0.00097, tau_R=0.0056, tau_H=0.0990):
        super(WilsonEuler, self).__init__()
        self.threshold = threshold
        self.tau_V = tau_V
        self.tau_R = tau_R
        self.tau_H = tau_H
        
    @property
    def _argreprs(self):
        args = []
        def add(attr, default):
            if getattr(self, attr) != default:
                args.append("%s=%s" %(attr, getattr(self, attr)))
        add("threshold", -0.20)
        add("tau_V", 0.00097)
        add("tau_R", 0.0056)
        add("tau_H", 0.0990)
        return args


    def gain_bias(self, max_rates, intercepts):
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
        J_steps = 201  # Odd number so that 0 is a sample
        max_rate = max_rates.max()
        # Find range of J that will achieve max rates (assume monotonic)
        J_threshold = None
        J_max = None
        Jr = 1.0
        for _ in range(10):
            J = np.linspace(-Jr, Jr, J_steps)
            rate = self.rates(J, np.ones(J_steps), np.zeros(J_steps))
#             print('J', J, 'euler rate', rate)
            if J_threshold is None and (rate <= 0).any():
                J_threshold = J[np.where(rate <= 0)[0][-1]]
            if J_max is None and (rate >= max_rate).any():
                J_max = J[np.where(rate >= max_rate)[0][0]]
            if J_threshold is not None and J_max is not None:
                break
            else:
                Jr *= 2
        else:
            if J_threshold is None:
                raise RuntimeError("Could not find firing threshold")
            if J_max is None:
                raise RuntimeError("Could not find max current")

        J = np.linspace(J_threshold, J_max, J_steps)
        rate = self.rates(J, np.ones(J_steps), np.zeros(J_steps))
        gain = np.zeros_like(max_rates)
        bias = np.zeros_like(max_rates)
        J_tops = np.interp(max_rates, rate, J)
        gain[:] = (J_threshold - J_tops) / (intercepts - 1)
        bias[:] = J_tops - gain
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        return np.zeros_like(gain), np.zeros_like(bias)

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        voltage = self._v0*np.ones_like(J)
        recovery = self._r0*np.ones_like(J)
        conductance = np.zeros_like(J)
        AP = np.zeros_like(J, dtype=bool)

        return settled_firingrate(
            self.step_math, J, [voltage, recovery, conductance, AP],
            dt=0.000025, settle_time=0.1, sim_time=1.0)


    def step_math(self, dt, J, spiked, V, R, H, AP):
        # must use dt<0.0001 to avoid numerical errors
        # if np.abs(J).any() >= 2.0:
        #     warnings.warn("input current exceeds failure point; clipping")
        #     J = J.clip(max=self._maxJ)
        dV = -(17.81 + 47.58*V + 33.80*np.square(V))*(V - 0.48) - 26*R*(V + 0.95) - 13*H*(V + 0.95) + J
        dR = -R + 1.29*V + 0.79 + 3.30*np.square(V + 0.38)
        dH = -H + 11*(V + 0.754)*(V + 0.69)
        V[:] = (V + dV * dt/self.tau_V).clip(-0.9, 0.3)
        R[:] = (R + dR * dt/self.tau_R)  # .clip(0.18, 0.42)
        H[:] = (H + dH * dt/self.tau_H)  # .clip(0, 0.23)
        spiked[:] = (V > self.threshold) & (~AP)
        spiked /= dt
        AP[:] = V > self.threshold
        return spiked, V, R, H, AP


class WilsonRungeKutta(NeuronType):

    probeable = ('spikes', 'voltage', 'recovery', 'conductance')
    threshold = NumberParam('threshold')
    tau_V = NumberParam('tau_V')
    tau_R = NumberParam('tau_R')
    tau_H = NumberParam('tau_H')
    
    _v0 = -0.754  # initial voltage
    _r0 = 0.279  # initial recovery
    _maxJ = 2.0  # clip input current at this maximum to avoid catastrophic shutdown
    
    def __init__(self, threshold=-0.20, tau_V=0.00097, tau_R=0.0056, tau_H=0.0990):
        super(WilsonRungeKutta, self).__init__()
        self.threshold = threshold
        self.tau_V = tau_V
        self.tau_R = tau_R
        self.tau_H = tau_H
        
        # TODO(arvoelke): Try replacing this solver with something like
        # http://www2.gsu.edu/~matrhc/PyDSTool.htm
        # The main consideration is that we need a callback to count spikes
        from scipy.integrate import ode
        self.solver = ode(self._ode_fun).set_integrator(
            'dopri5', first_step=0.000025, nsteps=100,
            rtol=1e-2, atol=1e-3)  # runge-kutta method of order (4)5
        
    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        voltage = self._v0*np.ones_like(J)
        recovery = self._r0*np.ones_like(J)
        conductance = np.zeros_like(J)

        return settled_firingrate(
            self.step_math, J, [voltage, recovery, conductance],
            dt=0.001, settle_time=0.1, sim_time=1.0)

    def _ode_fun(self, dummy_t, y, J):  # first argument to scipy.integrate.ode
        V, R, H = np.split(y, 3)
        dV = (-(17.81 + 47.58*V + 33.80*np.square(V))*(V - 0.48) -
              26*R*(V + 0.95) - 13*H*(V + 0.95) + J)
        dR = -R + 1.29*V + 0.79 + 3.30*np.square(V + 0.38)
        dH = -H + 11*(V + 0.754)*(V + 0.69)
        return np.concatenate((
            dV / self.tau_V, dR / self.tau_R, dH / self.tau_H))

    def step_math(self, dt, J, spiked, V, R, H):
        # It's a little silly to be reinitializing the solver on
        # every time-step, but any other ways that I could think of would 
        # violate the nengo builder's assumption that the neuron's state is
        # encapsulated by the signals in SimNeurons
        self.solver.set_initial_value(np.concatenate((V, R, H)))
        self.solver.set_f_params(J.clip(max=self._maxJ))
        
        spiked[:] = 0
        AP = V > self.threshold
        def spike_detector(dummy_t, y):  # callback for each sub-step
            V_t = y[:len(V)] > self.threshold
            spiked[:] += V_t & (~AP)  # note the "+="
            AP[:] = V_t
        self.solver.set_solout(spike_detector)

        V[:], R[:], H[:] = np.split(self.solver.integrate(self.solver.t + dt), 3)
        if not self.solver.successful():
            raise ValueError("ODE solver failed with status code: %d" % (
                self.solver.get_return_code()))
        spiked[:] /= dt

        return spiked, V, R, H




class BioNeuron(NeuronType):

    probeable = ('spikes', 'voltage')

    def __init__(self, cell_type, DA=lambda t: 0, dt_neuron=0.1):
        super(BioNeuron, self).__init__()
        self.cell_type = cell_type
        self.DA = DA
        assert callable(DA), "DA attribute of bioneuron must be a lambda function"
        self.dt_neuron = dt_neuron
        self.max_rates = np.array([])
        self.intercepts = np.array([])
        
    def gain_bias(self, max_rates, intercepts):
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)
        return max_rates, intercepts
        
    def max_rates_intercepts(self, gain, bias):
        return self.max_rates, self.intercepts
    
    def step_math(self, neurons, v_recs, spk_vecs, spk_recs, spk_before, DA, voltage, spiked, time, dt):
        n_neurons = voltage.shape[0]
        if neuron.h.t < time*1000:  # Nengo starts at t=dt
            neuron.h.tstop = time*1000
            neuron.h.continuerun(neuron.h.tstop)
        for n in range(n_neurons):
            volts = [v_recs[n][-i] for i in range(100)] if time > 0.1 else [-65.0]
            voltage[n] = -65 if np.any(np.isnan(volts)) else v_recs[n][-1]
            dopamine = DA(time)
            if type(dopamine) == np.ndarray: dopamine = dopamine[0]  # todo: more elegant
            neurons[n].set_DA(dopamine)  # update dopamine levels
        spk_after = [list(spk_vecs[n]) for n in range(n_neurons)]
        for n in range(n_neurons):
            spiked[n] = (len(spk_after[n]) - len(spk_before[n])) / dt
            spk_before[n] = list(spk_after[n])

class SimNeuronNeurons(Operator):
    def __init__(self, neuron_type, n_neurons, J, output, states, dt):
        super(SimNeuronNeurons, self).__init__()
        self.neuron_type = neuron_type
        rng = np.random.RandomState(seed=0)
        rGeos = rng.normal(1, 0.2, size=(n_neurons,))
        rCms = rng.normal(1, 0.05, size=(n_neurons,))
        rRs = rng.normal(1, 0.1, size=(n_neurons,))
        self.neurons = []
        for n in range(n_neurons):
            if self.neuron_type.cell_type == 'Pyramidal':
                self.neurons.append(neuron.h.Pyramidal(rGeos[n], rCms[n], rRs[n]))
            elif self.neuron_type.cell_type == 'Interneuron':
                self.neurons.append(neuron.h.Interneuron(rGeos[n], rCms[n], rRs[n]))
            else:
                raise "Cell Type %s not understood"%neuron_type.cell_type
        self.reads = [states[0], J]
        self.sets = [output, states[1]]
        self.updates = []
        self.incs = []
        self.v_recs = []
        self.spk_vecs = []
        self.spk_recs = []
        self.spk_before = [[] for n in range(n_neurons)]
        self.DA = neuron_type.DA
        for n in range(n_neurons):
            self.v_recs.append(neuron.h.Vector())
            self.v_recs[n].record(self.neurons[n].soma(0.5)._ref_v)
            self.spk_vecs.append(neuron.h.Vector())
            self.spk_recs.append(neuron.h.APCount(self.neurons[n].soma(0.5)))
            self.spk_recs[n].record(neuron.h.ref(self.spk_vecs[n]))
        neuron.h.dt = self.neuron_type.dt_neuron
        neuron.h.tstop = 0
    def make_step(self, signals, dt, rng):
        J = signals[self.current]
        output = signals[self.output]
        voltage = signals[self.voltage]
        time = signals[self.time]
        def step_nrn():
            self.neuron_type.step_math(
                self.neurons, self.v_recs, self.spk_vecs, self.spk_recs, self.spk_before, self.DA, voltage, output, time, dt)
        return step_nrn
    @property
    def time(self):
        return self.reads[0]
    @property
    def current(self):
        return self.reads[1]
    @property
    def output(self):
        return self.sets[0]
    @property
    def voltage(self):
        return self.sets[1]

class TransmitSpikes(Operator):
    def __init__(self, neurons, netcons, spikes, DA, states, dt):
        super(TransmitSpikes, self).__init__()
        self.neurons = neurons
        self.dt = dt
        self.time = states[0]
        self.reads = [spikes, states[0]]
        self.updates = []
        self.sets = []
        self.incs = []
        self.netcons = netcons
        self.DA = DA
    def make_step(self, signals, dt, rng):
        spikes = signals[self.spikes]
        time = signals[self.time]
        def step():
            t_neuron = time.item()*1000
            for pre in range(spikes.shape[0]):
                if spikes[pre] > 0:
                    for post in range(len(self.neurons)):
                        # update dopamine levels
                        if hasattr(self.netcons[pre, post].syn(), 'DA'):
                            self.netcons[pre, post].syn().DA = self.DA(time.item())
                        # deliver spike
                        self.netcons[pre, post].event(t_neuron)
        return step
    @property
    def spikes(self):
        return self.reads[0]

@Builder.register(nengo.Connection)
def build_connection(model, conn):
    is_ens = isinstance(conn.post_obj, nengo.Ensemble)
    is_neurons = isinstance(conn.post_obj, nengo.ensemble.Neurons)
    is_NEURON = False
    
    if is_ens:
        if isinstance(conn.post_obj.neuron_type, BioNeuron):
            is_NEURON = True
        else:
            is_NEURON = False
    elif is_neurons:
        if isinstance(conn.post_obj.ensemble.neuron_type, BioNeuron):
            is_NEURON = True
    else:
        is_ens = False
        is_neurons = False
        is_NEURON = False

    if is_NEURON:
        model.sig[conn]['in'] = model.sig[conn.pre_obj]['out']
        assert isinstance(conn.pre_obj, nengo.Ensemble) and 'spikes' in conn.pre_obj.neuron_type.probeable
        post_obj = conn.post_obj if is_ens else conn.post_obj.ensemble
        pre_obj = conn.pre_obj
        
        if isinstance(conn.synapse, AMPA):
            taus = "AMPA"
        elif isinstance(conn.synapse, GABA):
            taus = "GABA"
        elif isinstance(conn.synapse, NMDA):
            taus = "NMDA"
        elif isinstance(conn.synapse, LinearSystem):
            taus = -1.0/np.array(conn.synapse.poles) * 1000  # convert to ms
        else:
            raise "synapse type %s not understood (connection %s)"%(conn.synapse, conn)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        
            conn.rng = np.random.RandomState(model.seeds[conn])
            # conn.e = conn.rng.uniform(-1, 1, size=(pre_obj.n_neurons, post_obj.n_neurons, post_obj.dimensions))
            conn.e = np.zeros((pre_obj.n_neurons, post_obj.n_neurons, post_obj.dimensions))
            conn.locations = conn.rng.uniform(0, 1, size=(pre_obj.n_neurons, post_obj.n_neurons))
            if post_obj.neuron_type.cell_type == "Pyramidal":
                conn.compartments = conn.rng.randint(0, 3, size=(pre_obj.n_neurons, post_obj.n_neurons))  # 0=proximal, 1=distal, 2=basal
            elif post_obj.neuron_type.cell_type == "Interneuron":
                conn.compartments = np.zeros(shape=(pre_obj.n_neurons, post_obj.n_neurons))  # 0=dendrite
            conn.synapses = np.zeros((pre_obj.n_neurons, post_obj.n_neurons), dtype=list)
            conn.netcons = np.zeros((pre_obj.n_neurons, post_obj.n_neurons), dtype=list)
            conn.weights = np.zeros((pre_obj.n_neurons, post_obj.n_neurons))
            conn.v_recs = []
            transform = full_transform(conn, slice_pre=False)
            eval_points, d, solver_info = model.build(conn.solver, conn, conn.rng, transform)
            conn.d = d.T
            conn.transmitspike = None
        for post in range(post_obj.n_neurons):
            nrn = model.params[post_obj.neurons][post]
            for pre in range(pre_obj.n_neurons):
                conn.weights[pre, post] = np.dot(conn.d[pre], conn.e[pre, post])
                reversal = 0.0 if conn.weights[pre, post] > 0 else -70.0
                if conn.compartments[pre, post] == 0:
                    if post_obj.neuron_type.cell_type == "Pyramidal":
                        loc = nrn.prox(conn.locations[pre, post])
                    elif post_obj.neuron_type.cell_type == "Interneuron":
                        loc = nrn.dendrite(conn.locations[pre, post])
                elif conn.compartments[pre, post] == 1:
                    loc = nrn.dist(conn.locations[pre, post])
                else:
                    loc = nrn.basal(conn.locations[pre, post])
                if type(taus) == str:
                    if taus == "AMPA":
                        syn = neuron.h.ampa(loc)
                    elif taus == "GABA":
                        syn = neuron.h.gaba(loc)
                    elif taus == "NMDA":
                        syn = neuron.h.nmda(loc)
                    else:
                        raise "synapse %s not understood"%taus
                elif len(taus) == 1:
                    syn = neuron.h.ExpSyn(loc)
                    syn.tau = taus[0]
                    syn.e = reversal
                elif len(taus) == 2:
#                     syn = neuron.h.Exp2Syn(loc)
                    syn = neuron.h.doubleexp(loc)
                    syn.tauRise = np.min(taus)
                    syn.tauFall = np.max(taus)
                    syn.e = reversal
                conn.synapses[pre, post] = syn
                conn.netcons[pre, post] = neuron.h.NetCon(None, conn.synapses[pre, post])
                conn.netcons[pre, post].weight[0] = np.abs(conn.weights[pre, post])
            conn.v_recs.append(neuron.h.Vector())
            conn.v_recs[post].record(nrn.soma(0.5)._ref_v)
        transmitspike = TransmitSpikes(model.params[post_obj.neurons], conn.netcons,
            model.sig[conn.pre_obj]['out'], DA=post_obj.neuron_type.DA, states=[model.time], dt=model.dt)
        model.add_op(transmitspike)
        conn.transmitspike = transmitspike
        model.params[conn] = BuiltConnection(
            eval_points=eval_points, solver_info=solver_info, transform=transform, weights=d)
    
    else:
        c = nengo.builder.connection.build_connection(model, conn)
        model.sig[conn]['weights'].readonly = False
        return c
    
def reset_neuron(sim, model):
    for key in list(sim.model.params.keys()):
        if type(key) == nengo.ensemble.Neurons:
            del(sim.model.params[key])
    for op in sim.model.operators:
        if isinstance(op, SimNeuronNeurons):
            for v_rec in op.v_recs:
                v_rec.play_remove()
            for spk_vec in op.spk_vecs:
                spk_vec.play_remove()
            del(op.neurons)
        if isinstance(op, TransmitSpikes):
            del(op.neurons)
            del(op.netcons)
    for conn in model.connections:
        if hasattr(conn, 'v_recs'):
            for v_rec in conn.v_recs:
                v_rec.play_remove()
        if hasattr(conn, 'synapses'):
            del(conn.synapses)
        if hasattr(conn, 'netcons'):
            del(conn.netcons)
        if hasattr(conn, 'transmitspikes'):
            del(conn.transmitspikes.neurons)
            del(conn.transmitspikes.netcons)
            del(conn.transmitspikes)


@Builder.register(LIF)
def build_lif(model, lif, neurons):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))

@Builder.register(AdaptiveLIFT)
def build_alift(model, lif, neurons):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['threshold'] = Signal(
        np.ones(neurons.size_in), name="%s.threshold" % neurons)
    model.add_op(SimNeurons(
        neurons=lif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time'],
                model.sig[neurons]['threshold']]))


@Builder.register(WilsonEuler)
def build_wilsonneuron(model, neuron_type, neurons):
    model.sig[neurons]['voltage'] = Signal(
        neuron_type._v0*np.ones(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['recovery'] = Signal(
        neuron_type._r0*np.ones(neurons.size_in), name="%s.recovery" % neurons)
    model.sig[neurons]['conductance'] = Signal(
        np.zeros(neurons.size_in), name="%s.conductance" % neurons)
    model.sig[neurons]['AP'] = Signal(
        np.zeros(neurons.size_in, dtype=bool), name="%s.AP" % neurons)
    model.add_op(SimNeurons(
        neurons=neuron_type,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
            model.sig[neurons]['recovery'],
            model.sig[neurons]['conductance'],
            model.sig[neurons]['AP']]))

@Builder.register(WilsonRungeKutta)
def build_wilsonneuron(model, neuron_type, neurons):
    model.sig[neurons]['voltage'] = Signal(
        neuron_type._v0*np.ones(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['recovery'] = Signal(
        neuron_type._r0*np.ones(neurons.size_in), name="%s.recovery" % neurons)
    model.sig[neurons]['conductance'] = Signal(
        np.zeros(neurons.size_in), name="%s.conductance" % neurons)
    model.add_op(SimNeurons(
        neurons=neuron_type,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['recovery'],
                model.sig[neurons]['conductance']]))
    
@Builder.register(BioNeuron)
def build_neuronneuron(model, neuron_type, neurons):
    model.sig[neurons]['voltage'] = Signal(
#         neuron_type.v0*np.ones(neurons.size_in), name="%s.voltage"%neurons)
        np.random.RandomState(seed=0).uniform(-75, -35, neurons.size_in), name="%s.voltage"%neurons)
    neuronop = SimNeuronNeurons(
        neuron_type=neuron_type,
        n_neurons=neurons.size_in,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.time, model.sig[neurons]['voltage']],
        dt=model.dt)
    model.params[neurons] = neuronop.neurons
    model.add_op(neuronop)