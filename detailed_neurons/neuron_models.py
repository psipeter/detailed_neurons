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
neuron.h.load_file('NEURON/durstewitz.hoc')
neuron.h.load_file('stdrun.hoc')


__all__ = ['LIFNorm', 'AdaptiveLIFT', 'WilsonEuler', 'DurstewitzNeuron',
    'SimNeuronNeurons', 'TransmitSpikes', 'BiasSpikes', 'TransmitSignal', 'reset_neuron']


class LIFNorm(LIF):
    probeable = ("spikes", "voltage", "refractory_time")

    min_voltage = NumberParam("min_voltage", high=0)

    max_x = nengo.params.NumberParam('max_x')

    def __init__(self, max_x=1.0, tau_rc=0.02, tau_ref=0.002, amplitude=1):
        super().__init__(tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)
        self.max_x = max_x

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        inv_tau_ref = 1. / self.tau_ref if self.tau_ref > 0 else np.inf
        if np.any(max_rates > inv_tau_ref):
            raise ValidationError("Max rates must be below the inverse "
                                  "refractory period (%0.3f)" % inv_tau_ref,
                                  attr='max_rates', obj=self)

        x = -1 / np.expm1((self.tau_ref - 1 / max_rates) / self.tau_rc)
        normalizer = 0.5 * self.max_x + 0.5  # == 1 when max_x == 1, 0.5*max_x when max_x >> 1
        gain = (x - 1) * normalizer / (self.max_x - intercepts)
        bias = 1 - gain * intercepts
        return gain, bias


class AdaptiveLIFT(LIFRate):
    
    '''
    Aaron Voelker
    https://github.com/nengo/nengo/issues/1423
    '''
    
    probeable = ('spikes', 'voltage', 'refractory_time', 'threshold')

    min_voltage = NumberParam('min_voltage', high=0)
    tau_adapt = NumberParam('tau_adapt', low=0)
    inc_adapt = NumberParam('inc_adapt', low=0)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, min_voltage=0,
                 amplitude=1, tau_adapt=0.1, inc_adapt=0.05):
        super(AdaptiveLIFT, self).__init__(
            tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)
        self.min_voltage = min_voltage
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

    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        voltage = self._v0*np.ones_like(J)
        recovery = self._r0*np.ones_like(J)
        conductance = np.zeros_like(J)
        AP = np.zeros_like(J, dtype=bool)

        return settled_firingrate(
            self.step_math, J, [voltage, recovery, conductance, AP],
            dt=0.00005, settle_time=0.1, sim_time=1.0)


    def step_math(self, dt, J, spiked, V, R, H, AP, ddt=0.000025):
        if np.abs(J).any() >= 2.0:
            warnings.warn("input current exceeds failure point; clipping")
            J = J.clip(max=self._maxJ)
        for step in range(int(dt/ddt)):
            dV = -(17.81 + 47.58*V + 33.80*np.square(V))*(V - 0.48) - 26*R*(V + 0.95) - 13*H*(V + 0.95) + J
            dR = -R + 1.29*V + 0.79 + 3.30*np.square(V + 0.38)
            dH = -H + 11*(V + 0.754)*(V + 0.69)
        
            V[:] = (V + dV * ddt/self.tau_V).clip(-0.9, 0.3)
            R[:] = (R + dR * ddt/self.tau_R)  # .clip(0.18, 0.42)
            H[:] = (H + dH * ddt/self.tau_H)  # .clip(0, 0.23)
        spiked[:] = (V > self.threshold) & (~AP)
        spiked /= dt
        AP[:] = V > self.threshold
        return spiked, V, R, H, AP


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



class DurstewitzNeuron(NeuronType):
    '''
    todo: nice description
    '''
    probeable = ('spikes', 'voltage')

    def __init__(self, v0=-65.0, dt_neuron=0.025, DA=False):
        super(DurstewitzNeuron, self).__init__()
        self.v0 = v0
        self.dt_neuron = dt_neuron
        self.DA = DA  # dopaminergic modulation ON/OFF
        self.max_rates = np.array([])
        self.intercepts = np.array([])
        
    def gain_bias(self, max_rates, intercepts):
        return np.ones_like(max_rates), np.zeros_like(intercepts)
        
    def max_rates_intercepts(self, gain, bias):
        """Measure firing rate at each eval_point using neuron_rates, estimate x_int and rate at y_int"""
        return self.max_rates, self.intercepts
    
    def step_math(self, v_recs, spk_vecs, spk_recs, spk_before, voltage, spiked, time, dt):
        n_neurons = voltage.shape[0]
#         spk_before = np.array([np.array(spk_vecs[n]) for n in range(n_neurons)])
        if neuron.h.t < time*1000:  # Nengo starts at t=dt
            neuron.h.tstop = time*1000
            neuron.h.continuerun(neuron.h.tstop)
        for n in range(n_neurons):
            if not np.isfinite(v_recs[n][-1]):
#                 warnings.warn('neuron %s returned nan voltage at t=%s' %(n, neuron.h.t*1000))
                voltage[n] = 0
            else:
                voltage[n] = v_recs[n][-1]
        spk_after = [list(spk_vecs[n]) for n in range(n_neurons)]
        for n in range(n_neurons):
            spiked[n] = (len(spk_after[n]) - len(spk_before[n])) / dt
            spk_before[n] = list(spk_after[n])
        
@Builder.register(DurstewitzNeuron)
def build_neuronneuron(model, neuron_type, neurons):
    model.sig[neurons]['voltage'] = Signal(
        neuron_type.v0*np.ones(neurons.size_in), name="%s.voltage"%neurons)
    neuronop = SimNeuronNeurons(
        neuron_type=neuron_type,
        n_neurons=neurons.size_in,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.time, model.sig[neurons]['voltage']],
        dt=model.dt)
    model.params[neurons] = neuronop.neurons
    model.add_op(neuronop)
    
class SimNeuronNeurons(Operator):
    def __init__(self, neuron_type, n_neurons,  J, output, states, dt):
        super(SimNeuronNeurons, self).__init__()
        self.neuron_type = neuron_type
        self.neurons = [neuron.h.Durstewitz() for n in range(n_neurons)]
        self.reads = [states[0], J]
        self.sets = [output, states[1]]
        self.updates = []
        self.incs = []
        self.v_recs = []
        self.spk_vecs = []
        self.spk_recs = []
        self.spk_before = [[] for n in range(n_neurons)]
        for n in range(n_neurons):
            self.v_recs.append(neuron.h.Vector())
            self.v_recs[n].record(self.neurons[n].soma(0.5)._ref_v, dt*1000)
            self.spk_vecs.append(neuron.h.Vector())
            self.spk_recs.append(neuron.h.APCount(self.neurons[n].soma(0.5)))
            self.spk_recs[n].record(neuron.h.ref(self.spk_vecs[n]), dt*1000)
        neuron.h.dt = self.neuron_type.dt_neuron
        neuron.h.tstop = 0
    def make_step(self, signals, dt, rng):
        J = signals[self.current]
        output = signals[self.output]
        voltage = signals[self.voltage]
        time = signals[self.time]
        def step_nrn():
            self.neuron_type.step_math(
                self.v_recs, self.spk_vecs, self.spk_recs, self.spk_before,
                voltage, output, time, dt)
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
    def __init__(self, neurons, taus, weights, spikes, states, dt):
        super(TransmitSpikes, self).__init__()
        self.neurons = neurons
        self.taus = taus
        self.dt = dt
        self.weights = weights
        self.time = states[0]
        self.reads = [spikes, states[0]]
        self.updates = []
        self.sets = []
        self.incs = []
        self.synapses = np.zeros((self.weights.shape), dtype=list)
        for pre in range(self.synapses.shape[0]):
            for post in range(self.synapses.shape[1]):
                self.synapses[pre, post] = []
                for loc in [self.neurons[post].basal(0.5),
                            self.neurons[post].prox(0.5),
                            self.neurons[post].dist(0.5)]:
                    if len(self.taus) == 1:
                        syn = neuron.h.ExpSyn(loc)
                        syn.tau = self.taus[0]*1000
                    elif len(taus) == 2:
                        syn = neuron.h.Exp2Syn(loc)
                        syn.tau1 = np.min(self.taus)*1000
                        syn.tau2 = np.max(self.taus)*1000
                    syn.e = 0.0 if self.weights[pre, post] > 0 else -70.0
                    self.synapses[pre, post].append(syn)
        self.netcons = np.zeros((self.weights.shape), dtype=list)
        for pre in range(self.synapses.shape[0]):
            for post in range(self.synapses.shape[1]):
                # NetCon(source, target, threshold, delay, weight)
                self.netcons[pre, post] = []
                for compt in range(len(self.synapses[pre, post])):
                    syn = self.synapses[pre, post][compt]
                    w = np.abs(self.weights[pre, post])
                    nc = neuron.h.NetCon(None, syn, 0, 0, w)
#                     nc.active(0)  # turn off until build finishes
                    self.netcons[pre, post].append(nc)
    def make_step(self, signals, dt, rng):
        spikes = signals[self.spikes]
        time = signals[self.time]
        def step():
            t_neuron = time.item()*1000
            for pre in range(spikes.shape[0]):
                if spikes[pre] > 0:
                    for post in range(len(self.neurons)):
                        for nc in self.netcons[pre, post]:
                            nc.event(t_neuron)
        return step
    @property
    def spikes(self):
        return self.reads[0]
    
class BiasSpikes(Operator):
    def __init__(self, neurons, bias):
        super(BiasSpikes, self).__init__()
        self.neurons = neurons
        self.tau = 0.1
        self.bias = bias
        self.reads = []
        self.updates = []
        self.sets = []
        self.incs = []
        self.stims = []
        self.syns = []
        self.ncs = []
        for n in range(len(self.neurons)):
            self.stims.append(neuron.h.NetStim())
            self.syns.append(neuron.h.ExpSyn(neurons[n].soma(0.5)))
            self.syns[-1].tau = self.tau * 1000  # time constant of synapse on spiking bias input
            self.syns[-1].e = 0.0 if self.bias[n] > 0 else -70.0
            self.ncs.append(neuron.h.NetCon(self.stims[-1], self.syns[-1], 0, 0, np.abs(self.bias[n])))
            self.ncs[-1].pre().start = 0
            self.ncs[-1].pre().number = 1e10
            self.ncs[-1].pre().interval = 1
            self.ncs[-1].pre().noise = 0  # 0 for regular spikes at rate, 1 for poisson spikes at rate
#                 self.ncs[-1].active(0)  # turn off until build finishes
    def make_step(self, signals, dt, rng):
        def step():
            pass
        return step
    
class TransmitSignal(Operator):
    def __init__(self, neurons, encoders, signal, states, dt):
        super(TransmitSignal, self).__init__()
        self.neurons = neurons
        self.encoders = encoders
        self.dt = dt
        self.time = states[0]
        self.reads = [signal, states[0]]
        self.updates = []
        self.sets = []
        self.incs = []
        self.Iclamps = np.zeros((len(self.neurons)), dtype=list)
        for post, nrn in enumerate(self.neurons):
            self.Iclamps[post] = []
            for loc in [nrn.basal(0.5), nrn.prox(0.5), nrn.dist(0.5)]:
                iclamp = neuron.h.IClamp(loc)
                iclamp.dur = 1e9
                iclamp.amp = 1e0
                self.Iclamps[post].append(iclamp)
    def make_step(self, signals, dt, rng):
        sig = signals[self.signal]
        time = signals[self.time]
        def step():
            for post in range(len(self.Iclamps)):
                for iclamp in self.Iclamps[post]:
                    iclamp.amp = 5e0*np.dot(sig, self.encoders[post])  # compensate for conductance syn?
        return step
    @property
    def signal(self):
        return self.reads[0]

@Builder.register(nengo.Connection)
def build_connection(model, conn):
    is_ens = isinstance(conn.post_obj, nengo.Ensemble)
    is_neurons = isinstance(conn.post_obj, nengo.ensemble.Neurons)
    is_durstewitz = False
    
    if is_ens:
        if isinstance(conn.post_obj.neuron_type, DurstewitzNeuron):
            is_durstewitz = True
        else:
            is_durstewitz = False
    elif is_neurons:
        if isinstance(conn.post_obj.ensemble.neuron_type, DurstewitzNeuron):
            is_durstewitz = True
    else:
        is_ens = False
        is_neurons = False
        is_durstewitz = False

    if is_durstewitz:
        rng = np.random.RandomState(model.seeds[conn])
        model.sig[conn]['in'] = model.sig[conn.pre_obj]['out']
        post_obj = conn.post_obj if is_ens else conn.post_obj.ensemble
        # load previously optimized gain/bias
        encoders = model.params[post_obj].encoders
        dim = post_obj.dimensions

        # connect spiking poisson input to conn.post
        if not hasattr(conn, 'biasspike'):
            if hasattr(conn, 'bias'):
                bias = np.array(conn.bias).ravel()
            else:
                bias = np.zeros((post_obj.n_neurons))
            biasspike = BiasSpikes(model.params[post_obj.neurons], bias)
            model.add_op(biasspike)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conn.biasspike = biasspike
                
        # normalize by area under synapse curve, integral PSC(t) |b-a = F(b) - F(a), b=1s, a=0s
        assert isinstance(conn.synapse, LinearSystem), "only nengolib synapses supported"
        assert len(conn.synapse.num) == 0, "only poles supported"
        assert 0 < len(conn.synapse.den) <= 2, "only exponential and double exponential synapses supported"
        if hasattr(conn, 'gain'):
            gain = np.array(conn.gain)
        else:
            gain = 1e-2 * np.ones((post_obj.n_neurons, 1))
        taus = -1.0/np.array(conn.synapse.poles)
        tf = 10.0
        if len(taus) == 1:
            gain /= (taus[0] * (-np.exp(-tf/taus[0]) + 1))
        elif len(taus) == 2:
            gain /= np.max(taus)/(np.max(taus)-np.min(taus)) * \
                (np.min(taus) + np.max(taus) - \
                 np.min(taus)*np.exp(-tf/np.min(taus)) - np.max(taus)*np.exp(-tf/np.max(taus)))
        scaled_enc = encoders * gain
        
        if isinstance(conn.pre_obj, nengo.Ensemble) and 'spikes' in conn.pre_obj.neuron_type.probeable:
            # connect spikes from conn.pre to conn.post with computed weight matrix
            transform = full_transform(conn, slice_pre=False)
            eval_points, d, solver_info = model.build(conn.solver, conn, rng, transform)
            weights = np.dot(d.T, scaled_enc.T)
            transmitspike = TransmitSpikes(model.params[post_obj.neurons], taus, weights,
                model.sig[conn.pre_obj]['out'], states=[model.time], dt=model.dt)
            model.add_op(transmitspike)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conn.transmitspike = transmitspike
            model.params[conn] = BuiltConnection(
                eval_points=eval_points, solver_info=solver_info, transform=transform, weights=d)
        elif isinstance(conn.pre_obj, nengo.Node) and is_ens:
            transmitsignal = TransmitSignal(model.params[post_obj.neurons], scaled_enc,
                model.sig[conn.pre_obj]['out'], states=[model.time], dt=model.dt)
            model.add_op(transmitsignal)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                conn.transmitsignal = transmitsignal
            model.params[conn] = BuiltConnection(
                eval_points=None, solver_info=None, transform=None, weights=None)
        else:
            raise NotImplementedError("conn.pre_obj must be spiking ensemble or node")
    
    else:
        c = nengo.builder.connection.build_connection(model, conn)
        model.sig[conn]['weights'].readonly = False
        return c

    
def reset_neuron(sim):
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
        if isinstance(op, BiasSpikes):
            del(op.neurons)
            del(op.stims)
            del(op.ncs)
        if isinstance(op, TransmitSpikes):
            del(op.neurons)
            del(op.netcons)
        if isinstance(op, TransmitSignal):
            del(op.neurons)
            del(op.Iclamps)