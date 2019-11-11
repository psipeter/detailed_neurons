import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train import norms, df_opt, LearningNode
from neuron_models2 import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(n_neurons=10, t=10, m=Uniform(20, 40), i=Uniform(-0.8, 0.8), seed=1, dt=0.001,  weights=None, f=Lowpass(0.1), stim_func=lambda t: np.sin(t), g=None, b=None, norm_val=1.3, mirror=False, learn=False):

#     norm = norms(t, dt, stim_func, f=f, value=norm_val)
    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)

        # Ensembles
        pre = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed)
        lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, tar, synapse=f, seed=seed)
        nengo.Connection(pre, lif, synapse=f, seed=seed, label='pre_lif')
        conn = nengo.Connection(pre, durstewitz, synapse=f, seed=seed, label='pre_durstewitz')

        if learn:
            node = LearningNode(n_neurons, 100, 1, conn)
            nengo.Connection(pre.neurons, node[0:100], synapse=f)
            nengo.Connection(durstewitz.neurons, node[100:100+n_neurons], synapse=f)
            nengo.Connection(lif.neurons, node[100+n_neurons: 100+2*n_neurons], synapse=f)
            nengo.Connection(u, node[-1:], synapse=f)

        # Probes
        p_u = nengo.Probe(u, synapse=f)
        p_tar = nengo.Probe(tar, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=0.1)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=0.1)
        p_v_durstewitz = nengo.Probe(durstewitz.neurons, 'voltage', synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(weights):
            for pre in range(100):
                for post in range(n_neurons):
                    conn.weights[pre, post] = weights[pre, post]
                    conn.netcons[pre, post].weight[0] = np.abs(weights[pre, post])
                    conn.netcons[pre, post].syn().e = 0 if weights[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        tar=sim.data[p_tar],
        lif=sim.data[p_lif],
        durstewitz=sim.data[p_durstewitz],
        v_durstewitz=sim.data[p_v_durstewitz],
        weights=conn.weights)

t = 30
stim_func = nengo.processes.WhiteSignal(period=t, high=1, rms=0.5, seed=0)
# stim_func = lambda t: np.sin(t)
data = go(n_neurons=5, t=t, f=Lowpass(0.1), dt=0.001, stim_func=stim_func, learn=True)
np.savez('data/tuning2.npz', weights=data['weights'])

fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
# ax.plot(data['times'], data['v_durstewitz'])
ax.plot(data['times'], data['u'])
ax2.plot(data['times'], data['lif'], alpha=0.5)
ax3.plot(data['times'], data['durstewitz'], alpha=0.5)
ax2.set(ylim=((0, 40)))
ax3.set(xlabel='time', ylim=((0, 40)))
plt.savefig('plots/tuning2_train.png')

# weights = np.load('data/tuning2.npz')['weights']
weights = data['weights']
stim_func = nengo.processes.WhiteSignal(period=10, high=1, rms=0.5, seed=1)
data = go(n_neurons=5, t=10, f=Lowpass(0.1), dt=0.001, stim_func=stim_func, learn=False, weights=weights)

fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
# ax.plot(data['times'], data['v_durstewitz'])
ax.plot(data['times'], data['u'])
ax2.plot(data['times'], data['lif'], alpha=0.5)
ax3.plot(data['times'], data['durstewitz'], alpha=0.5)
ax2.set(ylim=((0, 40)))
ax3.set(xlabel='time', ylim=((0, 40)))
plt.savefig('plots/tuning2_test.png')