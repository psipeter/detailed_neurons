import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from nengolib.synapses import ss2sim

from train import norms, df_opt, gb_opt, d_opt
from neuron_models import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')


def go(d_supv, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=100, t_supv=10, t=10, m=Uniform(10, 20), i=Uniform(-1, 0.8), seed=0, dt=0.001, f=Lowpass(0.1),
        g=None, b=None, freq=1):

    solver_lif = NoSolver(d_lif)
    solver_alif = NoSolver(d_alif)
    solver_wilson = NoSolver(d_wilson)
    solver_durstewitz = NoSolver(d_durstewitz)

    w = 2*np.pi*freq
    A = [[0, -w], [w, 0]]
    B = [[1], [0]]
    C = [[1, 0]]
    D = [[0]]
    sys = LinearSystem((A, B, C, D))
    msys = ss2sim(sys, synapse=~s, dt=dt)
    
    with nengo.Network(seed=seed) as model:
                    
        # Stimulus and Nodes
        u = nengo.Node(lambda t: 1/dt if t<=dt else 0)
        x = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        # Ensembles
        supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='pre_u')
        lif = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, radius=2, label='lif')
        alif = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, radius=2, label='alif')
        wilson = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed, radius=2, label='wilson')
        durstewitz = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed, radius=2, label='durstewitz')

        # Target connections
        nengo.Connection(u, x, synapse=~s, transform=msys.B)
        nengo.Connection(x, x, synapse=~s, transform=msys.A)
        nengo.Connection(x, supv, synapse=None)

        # Feedforward connections
        ff_lif = nengo.Connection(supv, lif, synapse=f, solver=NoSolver(d_supv), seed=seed, label='ff_lif')
        ff_alif = nengo.Connection(supv, alif, synapse=f, solver=NoSolver(d_supv), seed=seed, label='ff_alif')
        ff_wilson = nengo.Connection(supv, wilson, synapse=f, solver=NoSolver(d_supv), seed=seed, label='ff_wilson')
        ff_durstewitz = nengo.Connection(supv, durstewitz, synapse=f, solver=NoSolver(d_supv), seed=seed, label='ff_durstewitz')

        # Feedback Connections
        fb_lif = nengo.Connection(lif, lif, synapse=f_lif, solver=solver_lif, seed=seed, label='fb_lif')
        fb_alif = nengo.Connection(alif, alif, synapse=f_alif, solver=solver_alif, seed=seed, label='fb_alif')
        fb_wilson = nengo.Connection(wilson, wilson, synapse=f_wilson, solver=solver_wilson, seed=seed, label='fb_wilson')
        fb_durstewitz = nengo.Connection(durstewitz, durstewitz, synapse=f_durstewitz, solver=solver_durstewitz, seed=seed, label='fb_durstewitz')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        p_alif = nengo.Probe(alif.neurons, synapse=None)
        p_wilson = nengo.Probe(wilson.neurons, synapse=None)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff_durstewitz.gain = g
            ff_durstewitz.bias = b
            fb_durstewitz.gain = g
            fb_durstewitz.bias = b

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        '''Add supervision and remove recurrence'''
        # for conn in model.connections:
        #     if conn == ff_durstewitz:
        #         op = conn.transmitspike
        #         for pre in range(op.synapses.shape[0]):
        #             for post in range(op.synapses.shape[1]):
        #                 for compt in range(len(op.synapses[pre, post])):
        #                     op.netcons[pre, post][compt].weight[0] = np.abs(op.weights[pre, post])
        #     if conn == fb_durstewitz:
        #         op = conn.transmitspike
        #         for pre in range(op.synapses.shape[0]):
        #             for post in range(op.synapses.shape[1]):
        #                 for compt in range(len(op.synapses[pre, post])):
        #                     op.netcons[pre, post][compt].weight[0] = 0
        sim.signals[sim.model.sig[ff_lif]['weights']][:] = d_supv.T
        sim.signals[sim.model.sig[ff_alif]['weights']][:] = d_supv.T
        sim.signals[sim.model.sig[ff_wilson]['weights']][:] = d_supv.T
        sim.signals[sim.model.sig[fb_lif]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_alif]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_wilson]['weights']][:] = 0
        sim.run(t_supv, progress_bar=True)
        '''Remove supervision and add recurrence'''
        # for conn in model.connections:
        #     if conn == ff_durstewitz:
        #         op = conn.transmitspike
        #         for pre in range(op.synapses.shape[0]):
        #             for post in range(op.synapses.shape[1]):
        #                 for compt in range(len(op.synapses[pre, post])):
        #                     op.netcons[pre, post][compt].weight[0] = 0
        #     if conn == fb_durstewitz:
        #         op = conn.transmitspike
        #         for pre in range(op.synapses.shape[0]):
        #             for post in range(op.synapses.shape[1]):
        #                 for compt in range(len(op.synapses[pre, post])):
        #                     op.netcons[pre, post][compt].weight[0] = np.abs(op.weights[pre, post])
        sim.signals[sim.model.sig[ff_lif]['weights']][:] = 0
        sim.signals[sim.model.sig[ff_alif]['weights']][:] = 0
        sim.signals[sim.model.sig[ff_wilson]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_lif]['weights']][:] = d_lif.T
        sim.signals[sim.model.sig[fb_alif]['weights']][:] = d_alif.T
        sim.signals[sim.model.sig[fb_wilson]['weights']][:] = d_wilson.T
        sim.run(t, progress_bar=True)
        reset_neuron(sim) 
    
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        lif=sim.data[p_lif],
        alif=sim.data[p_alif],
        wilson=sim.data[p_wilson],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)


def run(n_neurons=100, t_supv=10, t=10, f=Lowpass(0.1), dt=0.001, seed=0,
        m=Uniform(10, 20), i=Uniform(-1, 0.8), freq=1,
        gb_evals=10, df_evals=100, order=1, load_gb=False, load_fd=False):

    g = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
    d_lif, d_alif, d_wilson, d_durstewitz = np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2))

    print('gathering feedforward weights')
    with nengo.Network(seed=seed) as model:
        supv = nengo.Ensemble(n_neurons, 2, neuron_type=nengo.SpikingRectifiedLinear(), max_rates=m, intercepts=i, seed=seed, radius=2)
        ens = nengo.Ensemble(n_neurons, 2, neuron_type=nengo.LIF(), max_rates=m, intercepts=i, seed=seed, radius=2)
        ff = nengo.Connection(supv, ens, synapse=f, seed=seed, label='ff')
    with nengo.Simulator(model, dt=dt, seed=0) as sim:
        d_supv = sim.data[ff].weights.T

    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
    else:
        for gb in range(gb_evals):
            print("gain1/bias1 evaluation #%s"%gb)
            data = go(d_supv, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t_supv=t_supv, t=0, f=f, m=m, i=i, dt=0.001, g=g, b=b, freq=freq, seed=seed)
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['x'], data['enc'], g, b,
                dt=0.001, name="plots/tuning/oscillator_eval%s_"%gb)
        # np.savez('data/gb_oscillator.npz', g=g, b=b)

    # Run many short trials to generate training data for decoders and filters without large drift.
    if load_fd:
        load = np.load(load_fd)
        d_lif = load['d_lif']
        d_alif = load['d_alif']
        d_wilson = load['d_wilson']
        d_durstewitz = load['d_durstewitz']
        if len(load['taus_lif']) == 1:
            f_lif = Lowpass(load['taus_lif'][0])
            f_alif = Lowpass(load['taus_alif'][0])
            f_wilson = Lowpass(load['taus_wilson'][0])
            f_durstewitz = Lowpass(load['taus_durstewitz'][0])
        elif len(load['taus_lif']) == 2:
            f_lif = DoubleExp(load['taus_lif'][0], load['taus_lif'][1])
            f_alif = DoubleExp(load['taus_alif'][0], load['taus_alif'][1])
            f_wilson = DoubleExp(load['taus_wilson'][0], load['taus_wilson'][1])
            f_durstewitz = DoubleExp(load['taus_durstewitz'][0], load['taus_durstewitz'][1])
    else:
        print('gathering filter and decoder training data')
        data = go(d_supv, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
            n_neurons=n_neurons, t_supv=t_supv, t=0, f=f, dt=dt, g=g, b=b, m=m, i=i, freq=freq, seed=seed)    
        if df_evals:
            print('optimizing filters and decoders')
            d_lif, f_lif, taus_lif = df_opt(data['x'], data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='oscillator_lif')
            d_alif, f_alif, taus_alif = df_opt(data['x'], data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='oscillator_alif')
            d_wilson, f_wilson, taus_wilson = df_opt(data['x'], data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='oscillator_wilson')
            d_durstewitz, f_durstewitz, taus_durstewitz = df_opt(data['x'], data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='oscillator_durstewitz')
        else:
            d_lif = d_opt(data['x'], data['lif'], f_lif, f, dt=dt)
            d_alif = d_opt(data['x'], data['alif'], f_alif, f, dt=dt)
            d_wilson = d_opt(data['x'], data['wilson'], f_wilson, f, dt=dt)
            d_durstewitz = d_opt(data['x'], data['durstewitz'], f_durstewitz, f, dt=dt)

        # np.savez('data/fd_oscillator.npz',
        #     d_lif=d_lif,
        #     d_alif=d_alif,
        #     d_wilson=d_wilson,
        #     d_durstewitz=d_durstewitz,
        #     taus_lif=taus_lif,
        #     taus_alif=taus_alif,
        #     taus_wilson=taus_wilson,
        #     taus_durstewitz=taus_durstewitz)
        
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
        plt.savefig("plots/oscillator_filters.png")
            
        a_lif = f_lif.filt(data['lif'], dt=dt)
        a_alif = f_alif.filt(data['alif'], dt=dt)
        a_wilson = f_wilson.filt(data['wilson'], dt=dt)
        a_durstewitz = f_durstewitz.filt(data['durstewitz'], dt=dt)
        target = f.filt(data['x'], dt=dt)
        xhat_lif = np.dot(a_lif, d_lif)
        xhat_alif = np.dot(a_alif, d_alif)
        xhat_wilson = np.dot(a_wilson, d_wilson)
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f' %nrmse(xhat_lif, target=target))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f' %nrmse(xhat_alif, target=target))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f' %nrmse(xhat_wilson, target=target))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f' %nrmse(xhat_durstewitz, target=target))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens1)")
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillator_states_train.png")

    print('running experimental test')
    nrmses = np.zeros((5, 1))
    data = go(d_supv, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=n_neurons, t_supv=1, t=t, f=f, dt=dt, g=g, b=b, m=m, i=i, freq=freq, seed=seed)

    a_lif = f_lif.filt(data['lif'], dt=dt)
    a_alif = f_alif.filt(data['alif'], dt=dt)
    a_wilson = f_wilson.filt(data['wilson'], dt=dt)
    a_durstewitz = f_durstewitz.filt(data['durstewitz'], dt=dt)
    target = f.filt(data['x'], dt=dt)
    xhat_lif = np.dot(a_lif, d_lif)
    xhat_alif = np.dot(a_alif, d_alif)
    xhat_wilson = np.dot(a_wilson, d_wilson)
    xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)

    fig, ax = plt.subplots(figsize=((12, 8)))
    ax.plot(data['times'], target, linestyle="--", label='target')
    ax.plot(data['times'], xhat_lif, label='LIF')
    ax.plot(data['times'], xhat_alif, label='ALIF')
    ax.plot(data['times'], xhat_wilson, label='Wilson')
    ax.plot(data['times'], xhat_durstewitz, label='Durstewitz')
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$')
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillator_states_time.png")

    fig, ax = plt.subplots(figsize=((8, 8)))
    ax.plot(target[:,0], target[:,1], alpha=0.5, linestyle="--", label='target')
    ax.plot(xhat_lif[:,0], xhat_lif[:,1], alpha=0.5, label='LIF')
    ax.plot(xhat_alif[:,0], xhat_alif[:,1], alpha=0.5, label='ALIF')
    ax.plot(xhat_wilson[:,0], xhat_wilson[:,1], alpha=0.5, label='Wilson')
    ax.plot(xhat_durstewitz[:,0], xhat_durstewitz[:,1], alpha=0.5, label='Durstewitz')
    ax.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{y}$', xlim=((-1, 1)), ylim=((-1, 1)))
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillator_states_test.png")

run(n_neurons=200, t_supv=10, t=10, gb_evals=0, order=2, df_evals=100, dt=0.000025)