import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train2 import norms, df_opt, LearningNode
from neuron_models2 import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=Lowpass(0.1),
        neuron_type=nengo.LIF(), w_pre=None, w_ens=None, learn_supv=False, learn_ens=False, stim_func=lambda t: np.sin(t)):

    norm = norms(t/2, dt, stim_func, f=1/s, value=1.0)
    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed)
        pre_x = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed)
        nef = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        tar = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre_u, synapse=None, seed=seed)
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        nengo.Connection(x, pre_x, synapse=None, seed=seed)
        nengo.Connection(pre_u, nef, synapse=f, seed=seed, transform=0.1, label='pre_nef')
        pre_ens = nengo.Connection(pre_u, ens, synapse=f, transform=0.1, seed=seed, label='pre_ens')
        pre_supv = nengo.Connection(pre_x, supv, synapse=f, seed=seed, label='pre_supv')
        nengo.Connection(nef, nef, synapse=f, seed=seed, label='nef-nef')
        if learn_ens:
            supv_ens = nengo.Connection(supv, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens), label='supv-ens')
        if not learn_supv and not learn_ens:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens), label='ens-ens')

        if learn_supv and isinstance(neuron_type, DurstewitzNeuron):
            node_supv = LearningNode(n_neurons, pre_x.n_neurons, 1, pre_supv)
            nengo.Connection(pre_x.neurons, node_supv[0:pre_x.n_neurons], synapse=f)
            nengo.Connection(supv.neurons, node_supv[pre_x.n_neurons:pre_x.n_neurons+n_neurons], synapse=f)
            nengo.Connection(nef.neurons, node_supv[pre_x.n_neurons+n_neurons: pre_x.n_neurons+2*n_neurons], synapse=f)
            nengo.Connection(x, node_supv[-1], synapse=f)
            
        if learn_ens and isinstance(neuron_type, DurstewitzNeuron):
            node_ens = LearningNode(n_neurons, n_neurons, 1, supv_ens)
            nengo.Connection(supv.neurons, node_ens[0:n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node_ens[n_neurons:2*n_neurons], synapse=f)
            nengo.Connection(supv.neurons, node_ens[2*n_neurons: 3*n_neurons], synapse=f)
            nengo.Connection(x, node_ens[-1], synapse=f)
            
        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_pre_u = nengo.Probe(pre_u, synapse=None)
        p_pre_x = nengo.Probe(pre_x, synapse=None)
        p_nef = nengo.Probe(nef, synapse=f)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_ens_v = nengo.Probe(ens.neurons, 'voltage', synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_pre):
            for pre in range(pre_u.n_neurons):
                for post in range(n_neurons):
                    pre_supv.weights[pre, post] = w_pre[pre, post]
                    pre_supv.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post])
                    pre_supv.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
                    pre_ens.weights[pre, post] = w_pre[pre, post]
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post])
                    pre_ens.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
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
        u=sim.data[p_u],
        x=sim.data[p_x],
        nef=sim.data[p_nef],
        ens=sim.data[p_ens],
        ens_v=sim.data[p_ens_v],
        supv=sim.data[p_supv],
        w_pre=pre_supv.weights if isinstance(neuron_type, DurstewitzNeuron) else None,
        w_ens=supv_ens.weights if learn_ens else None,
    )


def run(n_neurons=100, t=10, t_test=10, f=Lowpass(0.1), dt=0.001, n_trains=1, n_tests=10, neuron_type=nengo.LIF(),
        load_w=None, load_fd=None):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_pre = None
    w_ens = None

    print('Creating input signal from concatenated, normalized, flipped white noise')
    u_list = np.zeros((int(t/dt)+1, 1))
    for n in range(n_trains):
        stim_func = nengo.processes.WhiteSignal(period=t/n_trains/2, high=1, rms=0.5, seed=n)
        with nengo.Network() as model:
            model.t_half = t/n_trains/2
            def flip(t, x):
                return x if t<model.t_half else -1.0*x
            u_raw = nengo.Node(stim_func)
            u = nengo.Node(output=flip, size_in=1)   
            nengo.Connection(u_raw, u, synapse=None)
            p_u = nengo.Probe(u, synapse=None)
            p_x = nengo.Probe(u, synapse=1/s)
        with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
            sim.run(t/n_trains, progress_bar=False)
        u_list[n*int(t/n_trains/dt): (n+1)*int(t/n_trains/dt)] = sim.data[p_u] / np.max(np.abs(sim.data[p_x]))
    stim_func = lambda t: u_list[int(t/dt)]

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_pre = np.load(load_w)['w_pre']
        else:
            print('optimizing encoders into DurstewitzNeuron supv')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001, stim_func=stim_func,
                neuron_type=neuron_type, w_pre=w_pre, w_ens=w_ens, learn_supv=True, )
            w_pre = data['w_pre']
            np.savez('data/w_integrate.npz', w_pre=w_pre)

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = Lowpass(taus_ens[0]) if len(taus_ens) == 1 else DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens1')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=dt, stim_func=stim_func,
            neuron_type=neuron_type, w_pre=w_pre, w_ens=w_ens, )
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(f.filt(data['x'], dt=dt), data['supv'], f, order=2, df_evals=100, dt=dt, reg=1e-2, name='integrate_%s'%neuron_type)
        np.savez('data/fd_integrate_%s.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)


        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label="f_ens, nonzero d: %s/%s"%(np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/integrate_filters_%s.png"%neuron_type)

        a_supv = f_ens.filt(data['supv'], dt=dt)
        target = f.filt(f.filt(data['x'], dt=dt), dt=dt)
        xhat_supv = np.dot(a_supv, d_ens)
        nrmse_supv = nrmse(xhat_supv, target=target)

        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_supv, label='supv, nrmse=%.3f' %nrmse_supv)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train supv")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_train_supv_%s.png"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ens = np.load(load_w)['w_ens']
        else:
            print('optimizing encoders into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=Lowpass(0.1), dt=0.001, stim_func=stim_func,
                neuron_type=neuron_type, w_pre=w_pre, w_ens=w_ens, learn_ens=True, )
            w_ens = data['w_ens']
            np.savez('data/w_integrate.npz', w_pre=w_pre, w_ens=w_ens)

    nrmses_ens = np.zeros((n_tests))
    nrmses_nef = np.zeros((n_tests))
    nrmses_supv = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t_test, high=1, rms=0.5, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt,
            neuron_type=neuron_type, stim_func=stim_func, w_pre=w_pre, w_ens=w_ens, )

        a_ens = f_ens.filt(data['ens'], dt=dt)
        a_supv = f_ens.filt(data['supv'], dt=dt)
        target = f.filt(f.filt(data['x'], dt=dt), dt=dt)
        xhat_nef = data['nef']
        xhat_ens = np.dot(a_ens, d_ens)
        xhat_supv = np.dot(a_supv, d_ens)
        nrmse_nef = nrmse(xhat_nef, target=target)
        nrmse_ens = nrmse(xhat_ens, target=target)
        nrmse_supv = nrmse(xhat_supv, target=target)
        nrmses_ens[test] = nrmse_ens
        nrmses_nef[test] = nrmse_nef
        nrmses_supv[test] = nrmse_supv

        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(data['times'], target, linestyle="--", label='test1')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.plot(data['times'], xhat_nef, label='nef, nrmse=%.3f' %nrmse_nef)
        ax.plot(data['times'], xhat_supv, label='supv, nrmse=%.3f' %nrmse_supv)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_test_%s_%s.png"%(test, neuron_type))
        
#         fig, ax = plt.subplots()
#         ax.plot(data['times'], data['ens_v'])
#         plt.savefig('plots/integrate_v.png')

    mean_ens = np.mean(nrmses_ens)
    mean_nef = np.mean(nrmses_nef)
    mean_supv = np.mean(nrmses_supv)
    CI_ens = sns.utils.ci(nrmses_ens)
    CI_nef = sns.utils.ci(nrmses_nef)
    CI_supv = sns.utils.ci(nrmses_supv)

    print('nrmses: ', nrmses_nef, nrmses_ens, nrmses_supv)
    print('means: ', mean_nef, mean_ens, mean_supv)
    print('confidence intervals: ', CI_nef, CI_ens, CI_supv)
    np.savez('data/results_integrate_%s.npz'%neuron_type,
        nrmses_ens=nrmses_ens,
        nrmses_nef=nrmses_nef,
        nrmses_supv=nrmses_supv)
    return nrmses_nef, nrmses_ens

# nrmses_nef, nrmses_lif = run(n_neurons=200, t=30, n_trains=3, n_tests=3, neuron_type=nengo.LIF())
# _, nrmses_alif = run(n_neurons=200, t=30, n_trains=3, n_tests=3, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1))
# _, nrmses_wilson = run(n_neurons=200, t=30, n_trains=3, n_tests=3, dt=0.000025, neuron_type=WilsonEuler())
_, nrmses_durstewitz = run(n_neurons=200, t=30, n_trains=3, n_tests=3, neuron_type=DurstewitzNeuron())
# , load_fd="data/fd_integrate_DurstewitzNeuron().npz"

nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.barplot(data=nrmses.T)
ax.set(ylabel='NRMSE')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.tight_layout()
plt.savefig("plots/integrate_nrmses.png")