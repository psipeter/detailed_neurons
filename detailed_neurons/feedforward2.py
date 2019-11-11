import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train2 import df_opt, LearningNode
from neuron_models2 import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=Lowpass(0.1), neuron_type=nengo.LIF(), w_ens=None, w_ens2=None, learn=False, learn2=False, stim_func=lambda t: np.sin(t)):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)

        # Ensembles
        pre = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed)
        nef = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        nef2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed+1)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed+1)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(pre, nef, synapse=f, seed=seed, label='pre_nef')
        conn = nengo.Connection(pre, ens, synapse=f, seed=seed, label='pre_ens')
        nengo.Connection(u, tar, synapse=f, seed=seed)
        nengo.Connection(nef, nef2, synapse=f, seed=seed+1, label='nef-nef2')
        conn2 = nengo.Connection(ens, ens2, synapse=f_ens, seed=seed+1, solver=NoSolver(d_ens), label='ens-ens2')
        nengo.Connection(tar, tar2, synapse=f, seed=seed+1, label='tar-tar2')

        if learn and isinstance(neuron_type, DurstewitzNeuron):
            node = LearningNode(n_neurons, 100, 1, conn)
            nengo.Connection(pre.neurons, node[0:100], synapse=f)
            nengo.Connection(ens.neurons, node[100:100+n_neurons], synapse=f)
            nengo.Connection(nef.neurons, node[100+n_neurons: 100+2*n_neurons], synapse=f)
            nengo.Connection(u, node[-1:], synapse=f)
            
        if learn2 and isinstance(neuron_type, DurstewitzNeuron):
            node2 = LearningNode(n_neurons, n_neurons, 1, conn2)
            nengo.Connection(ens.neurons, node2[0:n_neurons], synapse=f)
            nengo.Connection(ens2.neurons, node2[n_neurons:2*n_neurons], synapse=f)
            nengo.Connection(nef2.neurons, node2[2*n_neurons: 3*n_neurons], synapse=f)
            nengo.Connection(tar, node2[-1:], synapse=f)
            
        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_pre = nengo.Probe(pre, synapse=None)
        p_nef = nengo.Probe(nef, synapse=f)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_tar = nengo.Probe(tar, synapse=None)
        p_nef2 = nengo.Probe(nef2, synapse=f)
        p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        p_tar2 = nengo.Probe(tar2, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ens):
            for pre in range(100):
                for post in range(n_neurons):
                    conn.weights[pre, post] = w_ens[pre, post]
                    conn.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                    conn.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        if np.any(w_ens2):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    conn2.weights[pre, post] = w_ens2[pre, post]
                    conn2.netcons[pre, post].weight[0] = np.abs(w_ens2[pre, post])
                    conn2.netcons[pre, post].syn().e = 0 if w_ens2[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        nef=sim.data[p_nef],
        ens=sim.data[p_ens],
        tar=sim.data[p_tar],
        nef2=sim.data[p_nef2],
        ens2=sim.data[p_ens2],
        tar2=sim.data[p_tar2],
        w_ens=conn.weights if isinstance(neuron_type, DurstewitzNeuron) else None,
        w_ens2=conn2.weights if isinstance(neuron_type, DurstewitzNeuron) else None,
    )


def run(n_neurons=100, t=10, t_test=10, f=Lowpass(0.1), dt=0.001, n_tests=10, neuron_type=nengo.LIF(), load_fd=None, load_fd2=None):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_ens = None
    w_ens2 = None
    stim_func = nengo.processes.WhiteSignal(period=t, high=1, rms=0.5, seed=0)

    if isinstance(neuron_type, DurstewitzNeuron):
        print('optimizing encoders into DurstewitzNeuron ens')
        # stim_func = lambda t: np.sin(t)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001, stim_func=stim_func, neuron_type=neuron_type, w_ens=w_ens, w_ens2=w_ens2, learn=True)
        w_ens = data['w_ens']
        np.savez('data/w_feedforward.npz', w_ens=w_ens)
        # todo: load weights

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d']
        taus_ens = load['taus']
        f_ens = Lowpass(taus_ens[0]) if len(taus_ens) == 1 else DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens1')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2)
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(data['tar'], data['ens'], f, order=2, df_evals=100, dt=dt, name='feedforward_%s'%id(neuron_type))
        np.savez('data/fd_feedforward_%s.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

    times = np.arange(0, 1, 0.0001)
    fig, ax = plt.subplots(figsize=((6, 6)))
    ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
    ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label="f_ens, nonzero d: %s/%s"%(np.count_nonzero(d_ens), n_neurons))
    ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plots/feedforward_filters_%s.png"%neuron_type)

    a_ens = f_ens.filt(data['ens'], dt=dt)
    target = f.filt(data['tar'], dt=dt)
    xhat_nef = data['nef']
    xhat_ens = np.dot(a_ens, d_ens)
    nrmse_nef = nrmse(xhat_nef, target=target)
    nrmse_ens = nrmse(xhat_ens, target=target)

    fig, ax = plt.subplots(figsize=((6, 6)))
    ax.plot(data['times'], target, linestyle="--", label='target')
    ax.plot(data['times'], xhat_nef, label='nef, nrmse=%.3f' %nrmse_nef)
    ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train1")
    plt.legend(loc='upper right')
    plt.savefig("plots/feedforward2_train1_%s.png"%neuron_type)
        
    if isinstance(neuron_type, DurstewitzNeuron):
        print('optimizing encoders into DurstewitzNeuron ens2')
        # stim_func = lambda t: np.sin(t)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2, learn2=True)
        w_ens2 = data['w_ens2']
        np.savez('data/w_feedforward.npz', w_ens=w_ens, w_ens2=w_ens2)
        # todo: load weights

    if load_fd2:
        load = np.load(load_fd2)
        d_ens = load['d']
        taus_ens = load['taus']
        f_ens = Lowpass(taus_ens[0]) if len(taus_ens) == 1 else DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens2')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2)
        print('optimizing filters and decoders for ens2')
        d_ens2, f_ens2, taus_ens2  = df_opt(data['tar2'], data['ens2'], f, order=2, df_evals=100, dt=dt, name='feedforward2_%s'%id(neuron_type))
        np.savez('data/fd_feedforward_%s.npz'%id(neuron_type), d_ens=d_ens, taus_ens=taus_ens, d_ens2=d_ens2, taus_ens2=taus_ens2)

    times = np.arange(0, 1, 0.0001)
    fig, ax = plt.subplots(figsize=((6, 6)))
    ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
    ax.plot(times, f_ens2.impulse(len(times), dt=0.0001), label="f_ens2, nonzero d2: %s/%s"%(np.count_nonzero(d_ens2), n_neurons))
    ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plots/feedforward_filters2_%s.png"%neuron_type)

    a_ens2 = f_ens2.filt(data['ens2'], dt=dt)
    target2 = f.filt(data['tar2'], dt=dt)
    xhat_nef2 = data['nef2']
    xhat_ens2 = np.dot(a_ens2, d_ens2)
    nrmse_nef2 = nrmse(xhat_nef2, target=target2)
    nrmse_ens2 = nrmse(xhat_ens2, target=target2)

    fig, ax = plt.subplots(figsize=((6, 6)))
    ax.plot(data['times'], target2, linestyle="--", label='target')
    ax.plot(data['times'], xhat_nef2, label='nef2, nrmse=%.3f' %nrmse_nef2)
    ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train2")
    plt.legend(loc='upper right')
    plt.savefig("plots/feedforward2_train2_%s.png"%neuron_type)
    
    nrmses_nef = np.zeros((n_tests))
    nrmses_ens = np.zeros((n_tests))
    nrmses_nef2 = np.zeros((n_tests))
    nrmses_ens2 = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t_test, high=1, rms=0.5, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        target = f.filt(data['tar'], dt=dt)
        xhat_nef = data['nef']
        xhat_ens = np.dot(a_ens, d_ens)
        nrmse_nef = nrmse(xhat_nef, target=target)
        nrmse_ens = nrmse(xhat_ens, target=target)
        a_ens2 = f_ens2.filt(data['ens2'], dt=dt)
        target2 = f.filt(data['tar2'], dt=dt)
        xhat_nef2 = data['nef2']
        xhat_ens2 = np.dot(a_ens2, d_ens2)
        nrmse_nef2 = nrmse(xhat_nef2, target=target2)
        nrmse_ens2 = nrmse(xhat_ens2, target=target2)
        nrmses_nef[test] = nrmse_nef
        nrmses_ens[test] = nrmse_ens
        nrmses_nef2[test] = nrmse_nef2
        nrmses_ens2[test] = nrmse_ens2        
    
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef, label='nef, nrmse=%.3f' %nrmse_nef)
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test1")
        plt.legend(loc='upper right')
        plt.savefig("plots/feedforward2_test_%s_%s.png"%(test, neuron_type))
        
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(data['times'], target2, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef2, label='nef2, nrmse=%.3f' %nrmse_nef2)
        ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test2")
        plt.legend(loc='upper right')
        plt.savefig("plots/feedforward2_test2_%s_%s.png"%(test, neuron_type))

    mean_nef = np.mean(nrmses_nef)
    mean_ens = np.mean(nrmses_ens)
    mean_nef2 = np.mean(nrmses_nef2)
    mean_ens2 = np.mean(nrmses_ens2)
    CI_nef = sns.utils.ci(nrmses_nef)
    CI_ens = sns.utils.ci(nrmses_ens)
    CI_nef2 = sns.utils.ci(nrmses_nef2)
    CI_ens2 = sns.utils.ci(nrmses_ens2)

    print('nrmses: ', nrmses_nef, nrmses_ens, nrmses_nef2, nrmses_ens2)
    print('means: ', mean_nef, mean_ens, mean_nef2, mean_ens2)
    print('confidence intervals: ', CI_nef, CI_ens, CI_nef2, CI_ens2)
    np.savez('data/results_feedforward2_%s.npz'%neuron_type,
        nrmses_nef=nrmses_nef,
        nrmses_ens=nrmses_ens,
        nrmses_nef2=nrmses_nef2,
        nrmses_ens2=nrmses_ens2)
    return nrmses_nef2, nrmses_ens2

nrmses_nef, nrmses_lif = run(n_neurons=30, t=30, n_tests=10, neuron_type=nengo.LIF())
_, nrmses_alif = run(n_neurons=30, t=30, n_tests=10, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1))
# _, nrmses_wilson = run(n_neurons=30, t=30, n_tests=10, dt=0.001, neuron_type=WilsonEuler())
# _, nrmses_durstewitz = run(n_neurons=30, t=30, n_tests=10, dt=0.001, neuron_type=DurstewitzNeuron())

# nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
# nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sns.barplot(data=nrmses.T)
# ax.set(ylabel='NRMSE')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/feedforward2_nrmses.png")