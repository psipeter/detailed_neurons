import numpy as np

from scipy.optimize import curve_fit

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.networks import LinearNetwork
from nengolib.signal import s, nrmse, LinearSystem, Identity
from nengolib.synapses import ss2sim

from train2 import norms, df_opt, LearningNode
from neuron_models2 import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')


def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=Lowpass(0.1), neuron_type=nengo.LIF(), w_supv=None, w_ens=None, learn_supv=False, learn_ens=False, freq=1):

    w = 2*np.pi*freq
    A = [[0, -w], [w, 0]]
    B = [[1], [0]]
    C = [[1, 0]]
    D = [[0]]
    sys = LinearSystem((A, B, C, D))
    A_nef = 0.1*np.array(A)+np.eye(2)
    
    with nengo.Network(seed=seed) as model:
                    
        x = nengo.Node(lambda t: [np.sin(w*t), np.cos(w*t)])
        if not learn_supv and not learn_ens:
            off = nengo.Node(lambda t: -1e3*(t>0.1))
        off_nef = nengo.Node(lambda t: -1e3*(t>0.1))

        # Ensembles
        pre = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='pre')
        supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2, label='supv')
        ens = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2, label='ens')
        pre_nef = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='supv_nef')
        nef = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='nef')

        nengo.Connection(x, pre, synapse=None, seed=seed)
        nengo.Connection(x, pre_nef, synapse=None, seed=seed)
        if not learn_supv and not learn_ens:
            nengo.Connection(off, pre.neurons, synapse=None, transform=np.ones((pre.n_neurons, 1)))
        nengo.Connection(off_nef, pre_nef.neurons, synapse=None, transform=np.ones((pre_nef.n_neurons, 1)))
        nengo.Connection(pre_nef, nef, synapse=f, transform=1, seed=seed, label='pre_nef-nef')
        nengo.Connection(nef, nef, synapse=f, transform=A_nef, seed=seed, label='nef-nef')
        pre_supv = nengo.Connection(pre, supv, synapse=f, seed=seed, label='pre_supv')
        if learn_ens:
            supv_ens = nengo.Connection(supv, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens) , label='supv_ens')
        if not learn_supv and not learn_ens:
            pre_ens = nengo.Connection(pre, ens, synapse=f, seed=seed, label='pre_ens')
#             supv_ens = nengo.Connection(supv, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens) , label='supv_ens')
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens), label='ens-ens')
        
        if learn_supv and isinstance(neuron_type, DurstewitzNeuron):
            node_supv = LearningNode(n_neurons, pre.n_neurons, 1, pre_supv)
            nengo.Connection(pre.neurons, node_supv[0:pre.n_neurons], synapse=f)
            nengo.Connection(supv.neurons, node_supv[pre.n_neurons:pre.n_neurons+n_neurons], synapse=f)
            nengo.Connection(nef.neurons, node_supv[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f)
            nengo.Connection(x, node_supv[-1], synapse=f)
            
        if learn_ens and isinstance(neuron_type, DurstewitzNeuron):
            node_ens = LearningNode(n_neurons, n_neurons, 1, supv_ens)
            nengo.Connection(supv.neurons, node_ens[0:n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node_ens[n_neurons:2*n_neurons], synapse=f)
            nengo.Connection(supv.neurons, node_ens[2*n_neurons: 3*n_neurons], synapse=f)
            nengo.Connection(x, node_ens[-1], synapse=f)

        # Probes
        p_x = nengo.Probe(x, synapse=None)
        p_pre = nengo.Probe(pre, synapse=0.01)
        p_nef = nengo.Probe(nef, synapse=f)
        p_nef_neurons = nengo.Probe(nef.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_supv):
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    pre_supv.weights[pre, post] = w_pre[pre, post]
                    pre_supv.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post])
                    pre_supv.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
        if np.any(w_ens):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    if learn_ens:
                        supv_ens.weights[pre, post] = w_ens[pre, post]
                        supv_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                        supv_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
                    if not learn_ens and not learn_supv:
                        ens_ens.weights[pre, post] = w_ens[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
    
    return dict(
        times=sim.trange(),
        x=sim.data[p_x],
        nef=sim.data[p_nef],
        nef_neurons=sim.data[p_nef_neurons],
        ens=sim.data[p_ens],
        supv=sim.data[p_supv],
        pre=sim.data[p_pre],
        w_supv=pre_supv.weights if isinstance(neuron_type, DurstewitzNeuron) else None,
        w_ens=supv_ens.weights if isinstance(neuron_type, DurstewitzNeuron) and learn_ens else None,
    )

def sinusoid(x, a, b, c):
    return a*np.sin(b*x+c)

def run(n_neurons=200, t=20, t_test=20, f=Lowpass(0.1), dt=0.001, dt_sample=0.001, seed=0, m=Uniform(20, 40), i=Uniform(-0.8, 0.8), freq=1, neuron_type=nengo.LIF(), load_fd=False, load_fd_out=None):

    d_ens = np.zeros((n_neurons, 2))
    f_ens = f
    w_supv = None
    w_ens = None

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_supv = np.load(load_w)['w_supv']
        else:
            print('optimizing encoders into DurstewitzNeuron supv')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001,
                neuron_type=neuron_type, w_supv=w_supv, w_ens=w_ens, learn_supv=True)
            w_supv = data['w_supv']
            np.savez('data/w_oscillate.npz', w_supv=w_supv)

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = Lowpass(taus_ens[0]) if len(taus_ens) == 1 else DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for supv')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=dt, neuron_type=neuron_type, w_supv=w_supv, w_ens=w_ens, learn_supv=True)
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(data['x'], data['supv'], f, order=2, df_evals=100, dt=dt, reg=1e-1, name='oscillate_%s'%neuron_type)
        np.savez('data/fd_oscillate_%s.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label="f_ens, nonzero d: %s/%s"%(np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/oscillate_filters_%s.png"%neuron_type)

        a_supv = f_ens.filt(data['supv'], dt=dt)
    #     a_ens = f_ens.filt(data['ens'], dt=dt)
        target = f.filt(data['x'],dt=dt)
        xhat_supv = np.dot(a_supv, d_ens)
    #     xhat_ens = np.dot(a_ens, d_ens)
        xhat_nef = data['nef']
        nrmse_supv = nrmse(xhat_supv, target=target)
    #     nrmse_ens = nrmse(xhat_ens, target=target)
        nrmse_nef = nrmse(xhat_nef, target=target)

        fig, ax = plt.subplots(figsize=((18, 18)))
        ax.plot(data['times'], target, linestyle="--", label='target')
#         ax.plot(data['times'], data['pre'], label='pre')
        ax.plot(data['times'], xhat_supv, label='supv, nrmse=%.3f' %nrmse_supv)
    #     ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.plot(data['times'], xhat_nef, label='nef, nrmse=%.3f' %nrmse_nef)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train supv")
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillate_train_supv_%s.png"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_supv = np.load(load_w)['w_supv']
        else:
            print('optimizing encoders into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001,
                neuron_type=neuron_type, w_supv=w_supv, w_ens=w_ens, learn_ens=True)
            w_ens = data['w_ens']
            np.savez('data/w_oscillate.npz', w_supv=w_supv, w_ens=w_ens)
            
    if load_fd_out:
        load = np.load(load_fd_out)
        d_ens_out = load['d_ens_out']
        taus_ens_out = load['taus_ens_out']
        f_ens_out = Lowpass(taus_ens_out[0]) if len(taus_ens_out) == 1 else DoubleExp(taus_ens_out[0], taus_ens_out[1])
        d_nef_out = load['d_nef_out']
        taus_nef_out = load['taus_nef_out']
        f_nef_out = Lowpass(taus_nef_out[0]) if len(taus_nef_out) == 1 else DoubleExp(taus_nef_out[0], taus_nef_out[1])
    else:
        print('gathering filter/decoder training data for readout')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=dt, neuron_type=neuron_type, w_supv=w_supv, w_ens=w_ens)
        print('optimizing filters and decoders')
        d_ens_out, f_ens_out, taus_ens_out = df_opt(data['x'][10000:], data['ens'][10000:], f, order=2, df_evals=100, dt=dt, reg=1e-1, name='oscillate_out_%s'%neuron_type)
        d_nef_out, f_nef_out, taus_nef_out = df_opt(data['x'][10000:], data['nef_neurons'][10000:], f, order=2, df_evals=100, dt=dt, reg=1e-1, name='oscillate_out_nef')
        np.savez('data/fd_oscillate_out_%s.npz'%neuron_type, d_ens_out=d_ens_out, taus_ens_out=taus_ens_out, d_nef_out=d_nef_out, taus_nef_out=taus_nef_out)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens_out.impulse(len(times), dt=0.0001), label="f_ens_out, nonzero d: %s/%s"%(np.count_nonzero(d_ens_out), n_neurons))
        ax.plot(times, f_nef_out.impulse(len(times), dt=0.0001), label="f_nef_out, nonzero d: %s/%s"%(np.count_nonzero(d_nef_out), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/oscillate_filters_out_%s.png"%neuron_type)

#         a_supv = f_ens.filt(data['supv'], dt=dt)
        a_ens = f_ens_out.filt(data['ens'], dt=dt)
        a_nef = f_nef_out.filt(data['nef_neurons'], dt=dt)
        target = f.filt(data['x'],dt=dt)
#         xhat_supv = np.dot(a_supv, d_ens)
        xhat_ens = np.dot(a_ens, d_ens_out)
        xhat_nef = np.dot(a_nef, d_nef_out)
#         nrmse_supv = nrmse(xhat_supv, target=target)
        nrmse_ens = nrmse(xhat_ens[10000:], target=target[10000:])
        nrmse_nef = nrmse(xhat_nef[10000:], target=target[10000:])

        fig, ax = plt.subplots(figsize=((18, 18)))
        ax.plot(data['times'][10000:], target[10000:], linestyle="--", label='target')
#         ax.plot(data['times'], data['pre'], label='pre')
#         ax.plot(data['times'], xhat_supv, label='supv, nrmse=%.3f' %nrmse_supv)
        ax.plot(data['times'][10000:], xhat_ens[10000:], label='ens, nrmse=%.3f' %nrmse_ens)
        ax.plot(data['times'][10000:], xhat_nef[10000:], label='nef, nrmse=%.3f' %nrmse_nef)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train out")
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillate_train_out_%s.png"%neuron_type)
        
    print('test')
    data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=Lowpass(0.1), dt=dt, neuron_type=neuron_type,
        w_supv=w_supv, w_ens=w_ens)
    
#     a_supv = f_ens.filt(data['supv'], dt=dt)
    a_ens = f_ens_out.filt(data['ens'], dt=dt)
    a_nef = f_nef_out.filt(data['nef_neurons'], dt=dt)
    target = f.filt(data['x'], dt=dt)
#     xhat_supv = np.dot(a_supv, d_ens)
    xhat_ens = np.dot(a_ens, d_ens_out)
    xhat_nef = np.dot(a_nef, d_nef_out)
#     nrmse_supv = nrmse(xhat_supv, target=target)
    nrmse_ens = nrmse(xhat_ens[10000:], target=target[10000:])
    nrmse_nef = nrmse(xhat_nef[10000:], target=target[10000:])

    fig, ax = plt.subplots(figsize=((18, 18)))
    ax.plot(data['times'][10000:], target[10000:], linestyle="--", label='target')
#     ax.plot(data['times'], data['pre'], label='pre')
#     ax.plot(data['times'], xhat_supv, label='supv, nrmse=%.3f' %nrmse_supv)
    ax.plot(data['times'][10000:], xhat_ens[10000:], label='ens, nrmse=%.3f' %nrmse_ens)
    ax.plot(data['times'][10000:], xhat_nef[10000:], label='nef, nrmse=%.3f' %nrmse_nef)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillate_test_%s.png"%neuron_type)
    
    print('nrmses: ', nrmse_nef, nrmse_ens)
    np.savez('data/results_oscillate_%s.npz'%neuron_type,
        nrmse_ens=nrmse_ens,
        nrmse_nef=nrmse_nef)
    return nrmse_nef, nrmse_ens

_, nrmse_wilson = run(neuron_type=WilsonEuler(), dt=0.000025,
    load_fd="data/fd_oscillate_WilsonEuler().npz",
    load_fd_out="data/fd_oscillate_out_WilsonEuler().npz")
_, nrmse_alif = run(neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1))
nrmse_nef, nrmse_lif = run(neuron_type=nengo.LIF())

nrmses = np.array([nrmse_nef, nrmse_lif, nrmse_alif])
nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.barplot(data=nrmses)
ax.set(ylabel='NRMSE')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.tight_layout()
plt.savefig("plots/oscillate_nrmses.png")