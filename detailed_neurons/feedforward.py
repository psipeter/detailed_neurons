import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train import norms, df_opt, gb_opt, d_opt
from neuron_models import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(n_neurons=100, t=10, m=Uniform(10, 20), i=Uniform(-1, 1), seed=1, dt=0.001, f=Lowpass(0.1),
       stim_func=lambda t: np.sin(t), g=None, b=None, norm_val=1.0, mirror=True):

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
        pre = nengo.Ensemble(100, 1, radius=norm, max_rates=m, intercepts=i, seed=seed)
        nef = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIFNorm(max_x=norm_val), seed=seed)
        alif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed)
        wilson = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, tar, synapse=f, seed=seed)
        nengo.Connection(pre, nef, synapse=f, seed=seed, label='pre_nef')
        nengo.Connection(pre, lif, synapse=f, seed=seed, label='pre_lif')
        nengo.Connection(pre, alif, synapse=f, seed=seed, label='pre_alif')
        nengo.Connection(pre, wilson, synapse=f, seed=seed, label='pre_wilson')
        pre_durstewitz = nengo.Connection(pre, durstewitz, synapse=f, seed=seed, label='pre_durstewitz')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_pre = nengo.Probe(pre, synapse=None)
        p_tar = nengo.Probe(tar, synapse=None)
        p_nef = nengo.Probe(nef, synapse=f)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        p_alif = nengo.Probe(alif.neurons, synapse=None)
        p_wilson = nengo.Probe(wilson.neurons, synapse=None)
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
        tar=sim.data[p_tar],
        nef=sim.data[p_nef],
        lif=sim.data[p_lif],
        alif=sim.data[p_alif],
        wilson=sim.data[p_wilson],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)


def run(n_neurons=100, t=10, f=Lowpass(0.1), dt=0.001, n_tests=10,
        gb_evals=10, df_evals=100, order=1, load_gb=None, load_fd=None):

    g = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    
    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
    else:
        for gb in range(gb_evals):
            print("gain/bias evaluation #%s"%gb)
            # stim_func = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.5, seed=gb)
            omega = np.random.RandomState(seed=gb).uniform(0, 2*np.pi)
            stim_func = lambda t: np.sin(t + omega)
            data = go(n_neurons=n_neurons, t=t, f=f, g=g, b=b, dt=dt, stim_func=stim_func, norm_val=1.2, mirror=False)
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['tar'], data['enc'], g, b,
            	dt=dt, name="plots/tuning/feedforward_eval%s_"%gb)
        np.savez('data/gb_feedforward.npz', g=g, b=b)

    if load_fd:
        d_lif = np.load(load_fd)['d_lif']
        d_alif = np.load(load_fd)['d_alif']
        d_wilson = np.load(load_fd)['d_wilson']
        d_durstewitz = np.load(load_fd)['d_durstewitz']
        f_lif = Lowpass(np.load(load_fd)['tau_lif'])[0]
        f_alif = Lowpass(np.load(load_fd)['tau_alif'])[0]
        f_wilson = Lowpass(np.load(load_fd)['tau_wilson'])[0]
        f_durstewitz = Lowpass(np.load(load_fd)['tau_durstewitz'])[0]
    else:
        if load_gb:
            print('gathering filter/decoder training data')
            stim_func = lambda t: np.sin(t)
            data = go(n_neurons=n_neurons, t=t, f=f, g=g, b=b, dt=dt, stim_func=stim_func, mirror=False)
        if df_evals:
            print('optimizing filters and decoders')
            d_lif, f_lif  = df_opt(data['tar'], data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='feedforward_lif')
            d_alif, f_alif  = df_opt(data['tar'], data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='feedforward_alif')
            d_wilson, f_wilson  = df_opt(data['tar'], data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='feedforward_wilson')
            d_durstewitz, f_durstewitz  = df_opt(data['tar'], data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='feedforward_durstewitz')
        else:
            f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
            d_lif = d_opt(data['tar'], data['lif'], f_lif, f, dt=dt)
            d_alif = d_opt(data['tar'], data['alif'], f_alif, f, dt=dt)
            d_wilson = d_opt(data['tar'], data['wilson'], f_wilson, f, dt=dt)
            d_durstewitz = d_opt(data['tar'], data['durstewitz'], f_durstewitz, f, dt=dt)
        np.savez('data/fd_feedforward.npz',
            d_lif=d_lif,
            d_alif=d_alif,
            d_wilson=d_wilson,
            d_durstewitz=d_durstewitz,
            tau_lif=-1.0/f_lif.poles,
            tau_alif=-1.0/f_alif.poles,
            tau_wilson=-1.0/f_wilson.poles,
            tau_durstewitz=-1.0/f_durstewitz.poles)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_lif.impulse(len(times), dt=0.0001), label="f_lif, nonzero d: %s/%s"%(np.count_nonzero(d_lif), n_neurons))
        ax.plot(times, f_alif.impulse(len(times), dt=0.0001), label="f_alif, nonzero d: %s/%s"%(np.count_nonzero(d_alif), n_neurons))
        ax.plot(times, f_wilson.impulse(len(times), dt=0.0001), label="f_wilson, nonzero d: %s/%s"%(np.count_nonzero(d_wilson), n_neurons))
        ax.plot(times, f_durstewitz.impulse(len(times), dt=0.0001), label="f_durstewitz, nonzero d: %s/%s"%(np.count_nonzero(d_durstewitz), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/feedforward_filters.png")
        
    print('running experimental tests')
    nrmses = np.zeros((5, n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t, high=1, rms=0.5, seed=100+test)
        data = go(n_neurons=n_neurons, t=t, f=f, g=g, b=b, dt=dt, stim_func=stim_func)

        a_lif = f_lif.filt(data['lif'], dt=dt)
        a_alif = f_alif.filt(data['alif'], dt=dt)
        a_wilson = f_wilson.filt(data['wilson'], dt=dt)
        a_durstewitz = f_durstewitz.filt(data['durstewitz'], dt=dt)
        target = f.filt(data['tar'], dt=dt)
        xhat_nef = data['nef']
        xhat_lif = np.dot(a_lif, d_lif)
        xhat_alif = np.dot(a_alif, d_alif)
        xhat_wilson = np.dot(a_wilson, d_wilson)
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)
        nrmses[0, test] = nrmse(xhat_nef, target=target)
        nrmses[1, test] = nrmse(xhat_lif, target=target)
        nrmses[2, test] = nrmse(xhat_alif, target=target)
        nrmses[3, test] = nrmse(xhat_wilson, target=target)
        nrmses[4, test] = nrmse(xhat_durstewitz, target=target)
        
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f' %nrmses[0, test])
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f' %nrmses[1, test])
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f' %nrmses[2, test])
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f' %nrmses[3, test])
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f' %nrmses[4, test])
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test %s"%test)
        plt.legend(loc='upper right')
        plt.savefig("plots/feedforward_states_test%s.png"%test)
            
    if n_tests > 1:
        nt_names =  ['LIF\n(static)', 'LIF\n(temporal)', 'ALIF', 'Wilson', 'Durstewitz']
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        sns.barplot(data=nrmses.T)
        ax.set(ylabel='NRMSE')
        plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
        plt.savefig("plots/feedforward_nrmses.png")
    means = np.mean(nrmses, axis=1)
    CIs = np.zeros((5, 2))
    for nt in range(nrmses.shape[0]):
        CIs[nt] = sns.utils.ci(nrmses[nt])
    print('nrmses: ', nrmses)
    print('means: ', means)
    print('confidence intervals: ', CIs)
    np.savez('data/nrmses_feedforward.npz', nrmses=nrmses, means=means, CIs=CIs)

run(t=10, n_neurons=100, gb_evals=15, n_tests=10, df_evals=200, load_gb='data/gb_feedforward.npz')