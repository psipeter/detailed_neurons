import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train import norms, df_opt, gb_opt, gb_opt2, d_opt
from neuron_models import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(n_neurons=10, t=10, m=Uniform(10, 20), i=Uniform(-1, 1), seed=1, dt=0.001, f=Lowpass(0.1),
       stim_func=lambda t: np.sin(t), g=None, b=None, norm_val=1.2, mirror=False):

    norm = norms(t, dt, stim_func, f=f, value=norm_val)
    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        model.T = t
        def flip(t, x):
            if t<model.T/2: return x
            elif t>=model.T/2: return -1.0*x
        u_raw = nengo.Node(stim_func)
        u = nengo.Node(output=flip, size_in=1) if mirror else nengo.Node(size_in=1)
        nengo.Connection(u_raw, u, synapse=None, transform=norm)

        # Ensembles
        pre = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed)
        lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        # alif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed)
        wilson = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed)
        durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, tar, synapse=f, seed=seed)
        nengo.Connection(pre, lif, synapse=f, seed=seed, label='pre_lif')
        # nengo.Connection(pre, alif, synapse=f, seed=seed, label='pre_alif')
        nengo.Connection(pre, wilson, synapse=f, seed=seed, label='pre_wilson')
        pre_durstewitz = nengo.Connection(pre, durstewitz, synapse=f, seed=seed, label='pre_durstewitz')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_pre = nengo.Probe(pre, synapse=f)
        p_tar = nengo.Probe(tar, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        # p_alif = nengo.Probe(alif.neurons, synapse=None)
        # p_wilson = nengo.Probe(wilson.neurons, synapse=None)
        p_v = nengo.Probe(wilson.neurons, 'voltage', synapse=None)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre_durstewitz.gain = g
        pre_durstewitz.bias = b

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim) 
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        pre=sim.data[p_pre],
        tar=sim.data[p_tar],
        # nef=sim.data[p_nef],
        lif=sim.data[p_lif],
        # alif=sim.data[p_alif],
        # wilson=sim.data[p_wilson],
        wilson_voltage=sim.data[p_v],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)


def run(n_neurons=5, t=10, f=Lowpass(0.1), dt=0.000025, m=Uniform(10, 20), i=Uniform(-1, 0.8), gb_evals=20, load_gb=False):

    g = 1e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    f.tau = 0.1

    # stim_func = lambda t: np.sin(t)
    # data = go(n_neurons=n_neurons, t=t, f=f, g=g, b=b, dt=dt, stim_func=stim_func, m=m, i=i)
    # fig, ax = plt.subplots()
    # ax.plot(data['times'], data['wilson_voltage'])
    # plt.show()
    # raise

    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
    else:
        for gb in range(gb_evals):
            print("gain/bias evaluation #%s"%gb)
            # stim_func = nengo.processes.WhiteSignal(period=t, high=1, rms=0.5, seed=gb)
            omega = np.random.RandomState(seed=gb).uniform(0, 2*np.pi)
            stim_func = lambda t: np.sin(t + omega)
            # stim_func = lambda t: np.sin(t)
            data = go(n_neurons=n_neurons, t=t, f=f, g=g, b=b, dt=dt, stim_func=stim_func, m=m, i=i)
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['pre'], data['enc'], g, b,
            	dt=dt, name="plots/tuning/tuning_eval%s_"%gb)
        np.savez('data/gb_tuning.npz', g=g, b=b)
run()