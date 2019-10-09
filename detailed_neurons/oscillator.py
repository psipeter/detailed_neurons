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

from train import norms, downsample_spikes, fit_sinusoid, df_opt, gb_opt, d_opt
from neuron_models import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')


def gb_train(n_neurons=100, t=10, m=Uniform(10, 20), i=Uniform(-0.7, 0.7), seed=0, dt=0.001, f=Lowpass(0.1),
    stim_func0=lambda t: np.sin(t), stim_func1=lambda t: np.sin(t+np.pi), g=None, b=None):

    with nengo.Network(seed=seed) as model:
        x0 = nengo.Node(stim_func0)
        x1 = nengo.Node(stim_func1)
        x = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='supv')
        lif = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, radius=2, label='lif')
        durstewitz = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed, radius=2, label='durstewitz')

        nengo.Connection(x0, x[0], synapse=None)
        nengo.Connection(x1, x[1], synapse=None)
        nengo.Connection(x, supv, synapse=None)
        ff_lif = nengo.Connection(supv, lif, synapse=f, seed=seed, label='ff_lif')
        ff_durstewitz = nengo.Connection(supv, durstewitz, synapse=f, seed=seed, label='ff_durstewitz')

        p_x = nengo.Probe(x, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff_durstewitz.gain = g
            ff_durstewitz.bias = b

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(t, progress_bar=True)
 
    return dict(
        times=sim.trange(),
        x=sim.data[p_x],
        lif=sim.data[p_lif],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)

def go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=100, t_supv=10, t=10, m=Uniform(10, 20), i=Uniform(-0.7, 0.7), seed=0, dt=0.001, f=Lowpass(0.1),
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
    # msys = ss2sim(sys, synapse=~s, dt=dt)
    
    with nengo.Network(seed=seed) as model:
                    
        # Stimulus and Nodes
        u = nengo.Node(lambda t: 1/dt*(t<=dt))
        # u2 = nengo.Node(lambda t: 1/dt*(t<=0.1))
        x = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        # Ensembles
        supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='supv')
        # nef = LinearNetwork(sys=sys, n_neurons_per_ensemble=n_neurons, max_rates=m, intercepts=i, radius=np.sqrt(2), synapse=f_nef, dt=dt, seed=seed, realizer=Identity())
        nef = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=2, label='nef')
        lif = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, radius=2, label='lif')
        alif = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, radius=2, label='alif')
        wilson = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed, radius=2, label='wilson')
        durstewitz = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed, radius=2, label='durstewitz')

        # Target connections
        nengo.Connection(u, x, synapse=~s, transform=sys.B)
        nengo.Connection(x, x, synapse=~s, transform=sys.A)
        nengo.Connection(x, supv, synapse=None)

        # Feedforward connections
        # ff_nef = nengo.Connection(u2, nef.input, synapse=None, seed=seed, label='ff_nef')
        ff_nef = nengo.Connection(supv, nef, synapse=f, seed=seed, transform=1.5, label='ff_nef')
        ff_lif = nengo.Connection(supv, lif, synapse=f, seed=seed, label='ff_lif')
        ff_alif = nengo.Connection(supv, alif, synapse=f, seed=seed, label='ff_alif')
        ff_wilson = nengo.Connection(supv, wilson, synapse=f, seed=seed, label='ff_wilson')
        ff_durstewitz = nengo.Connection(supv, durstewitz, synapse=f, seed=seed, label='ff_durstewitz')

        # Feedback Connections
        # fb_nef = nengo.Connection(nef, nef, synapse=f_nef, solver=solver_nef, seed=seed, label='fb_nef')
        fb_nef = nengo.Connection(nef, nef, synapse=f, transform=0.1*np.array(A)+np.eye(2), seed=seed, label='fb_lif')
        fb_lif = nengo.Connection(lif, lif, synapse=f_lif, solver=solver_lif, seed=seed, label='fb_lif')
        fb_alif = nengo.Connection(alif, alif, synapse=f_alif, solver=solver_alif, seed=seed, label='fb_alif')
        fb_wilson = nengo.Connection(wilson, wilson, synapse=f_wilson, solver=solver_wilson, seed=seed, label='fb_wilson')
        fb_durstewitz = nengo.Connection(durstewitz, durstewitz, synapse=f_durstewitz, solver=solver_durstewitz, seed=seed, label='fb_durstewitz')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        # p_nef = nengo.Probe(nef.state.output, synapse=f_nef)
        p_nef = nengo.Probe(nef, synapse=f)
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
        d_nef = sim.data[fb_nef].weights.T
        '''Add supervision and remove recurrence'''
        if isinstance(durstewitz.neuron_type, DurstewitzNeuron):
            for conn in model.connections:
                if conn == fb_durstewitz:
                    op = conn.transmitspike
                    for pre in range(op.synapses.shape[0]):
                        for post in range(op.synapses.shape[1]):
                            for compt in range(len(op.synapses[pre, post])):
                                op.netcons[pre, post][compt].active(0)
        sim.signals[sim.model.sig[fb_nef]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_lif]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_alif]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_wilson]['weights']][:] = 0
        sim.run(t_supv, progress_bar=True)
        '''Remove supervision and add recurrence'''
        if isinstance(durstewitz.neuron_type, DurstewitzNeuron):
            for conn in model.connections:
                if conn == ff_durstewitz:
                    op = conn.transmitspike
                    for pre in range(op.synapses.shape[0]):
                        for post in range(op.synapses.shape[1]):
                            for compt in range(len(op.synapses[pre, post])):
                                op.netcons[pre, post][compt].active(0)
                if conn == fb_durstewitz:
                    op = conn.transmitspike
                    for pre in range(op.synapses.shape[0]):
                        for post in range(op.synapses.shape[1]):
                            for compt in range(len(op.synapses[pre, post])):
                                op.netcons[pre, post][compt].active(1)  # np.abs(op.weights[pre, post])
        sim.signals[sim.model.sig[ff_nef]['weights']][:] = 0
        sim.signals[sim.model.sig[ff_lif]['weights']][:] = 0
        sim.signals[sim.model.sig[ff_alif]['weights']][:] = 0
        sim.signals[sim.model.sig[ff_wilson]['weights']][:] = 0
        sim.signals[sim.model.sig[fb_nef]['weights']][:] = d_nef.T
        sim.signals[sim.model.sig[fb_lif]['weights']][:] = d_lif.T
        sim.signals[sim.model.sig[fb_alif]['weights']][:] = d_alif.T
        sim.signals[sim.model.sig[fb_wilson]['weights']][:] = d_wilson.T
        sim.run(t, progress_bar=True)
        reset_neuron(sim) 
    
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        nef=sim.data[p_nef],
        lif=sim.data[p_lif],
        alif=sim.data[p_alif],
        wilson=sim.data[p_wilson],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)

def sinusoid(x, a, b, c):
    return a*np.sin(b*x+c)

def run(n_neurons=200, t_train=10, t=10, f=Lowpass(0.1), dt=0.001, dt_sample=0.001, seed=0,
        m=Uniform(10, 20), i=Uniform(-0.7, 0.7), freq=1, t_transient_train=0, t_transient_test=5,
        gb_evals=0, df_evals=100, order=1, load_gb=False, load_fd=False):

    g = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    # f_nef = DoubleExp(0.05, 0.05)
    f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
    d_lif, d_alif, d_wilson, d_durstewitz = np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2)), np.zeros((n_neurons, 2))

    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
    else:
        for gb in range(gb_evals):
            print("gain/bias evaluation #%s"%gb)
            data = gb_train(n_neurons, t_train, m=m, i=i, seed=seed, dt=0.001, f=f, g=g, b=b,
                stim_func0=nengo.processes.WhiteSignal(period=t_train, high=0.5, rms=0.6, seed=gb),
                stim_func1=nengo.processes.WhiteSignal(period=t_train, high=0.5, rms=0.6, seed=100+gb))
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['x'], data['enc'], g, b, f=f,
                t_transient=t_transient_train, dt=0.001, name="plots/tuning/oscillator_eval%s_"%gb)
            np.savez('data/gb_oscillator.npz', g=g, b=b)

    # Run many short trials to generate training data for decoders and filters without large drift.
    # NOTE: ALIF/Wilson needs to have startup transients discarded before optimizing filters/decoders
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
        data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
            n_neurons=n_neurons, t_supv=t_train, t=0, f=f, dt=dt, g=g, b=b, m=m, i=i, freq=freq, seed=seed)    
        if df_evals:
            print('optimizing filters and decoders')
            d_lif, f_lif, taus_lif = df_opt(
                data['x'][::int(dt_sample/dt)],
                downsample_spikes(data['lif'], dt=dt, dt_sample=dt_sample),
                f, order=order, df_evals=df_evals, dt=dt_sample, name='oscillator_lif')
            d_alif, f_alif, taus_alif = df_opt(
                data['x'][::int(dt_sample/dt)][int(t_transient_train/dt_sample):],
                downsample_spikes(data['alif'], dt=dt, dt_sample=dt_sample)[int(t_transient_train/dt_sample):],
                f, order=order, df_evals=df_evals, dt=dt_sample, name='oscillator_alif')
            d_wilson, f_wilson, taus_wilson = df_opt(
                data['x'][::int(dt_sample/dt)][int(t_transient_train/dt_sample):],
                downsample_spikes(data['wilson'], dt=dt, dt_sample=dt_sample)[int(t_transient_train/dt_sample):],
                f, order=order, df_evals=df_evals, dt=dt_sample, name='oscillator_wilson')
            d_durstewitz, f_durstewitz, taus_durstewitz = df_opt(
                data['x'][::int(dt_sample/dt)],
                downsample_spikes(data['durstewitz'], dt=dt, dt_sample=dt_sample),
                f, order=order, df_evals=df_evals, dt=dt_sample, name='oscillator_durstewitz')

        np.savez('data/fd_oscillator.npz',
            d_lif=d_lif,
            d_alif=d_alif,
            d_wilson=d_wilson,
            d_durstewitz=d_durstewitz,
            taus_lif=taus_lif,
            taus_alif=taus_alif,
            taus_wilson=taus_wilson,
            taus_durstewitz=taus_durstewitz)
        
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
            
        a_lif = f_lif.filt(downsample_spikes(data['lif'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
        a_alif = f_alif.filt(downsample_spikes(data['alif'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
        a_wilson = f_wilson.filt(downsample_spikes(data['wilson'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
        a_durstewitz = f_durstewitz.filt(downsample_spikes(data['durstewitz'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
        target = f.filt(data['x'][::int(dt_sample/dt)], dt=dt_sample)[int(t_transient_train/dt_sample):]
        times = data['times'][::int(dt_sample/dt)][int(t_transient_train/dt_sample):]
        xhat_nef = data['nef'][::int(dt_sample/dt)][int(t_transient_train/dt_sample):]
        xhat_lif = np.dot(a_lif, d_lif)[int(t_transient_train/dt_sample):]
        xhat_alif = np.dot(a_alif, d_alif)[int(t_transient_train/dt_sample):]
        xhat_wilson = np.dot(a_wilson, d_wilson)[int(t_transient_train/dt_sample):]
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)[int(t_transient_train/dt_sample):]

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, target[:,0], linestyle="--", label='target')
        ax.plot(times, xhat_nef[:,0], label='NEF')
        ax.plot(times, xhat_lif[:,0], label='LIF')
        ax.plot(times, xhat_alif[:,0], label='ALIF')
        ax.plot(times, xhat_wilson[:,0], label='Wilson')
        ax.plot(times, xhat_durstewitz[:,0], label='Durstewitz')
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$')
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillator_states0_time_train.png")

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, target[:,1], linestyle="--", label='target')
        ax.plot(times, xhat_nef[:,1], label='NEF')
        ax.plot(times, xhat_lif[:,1], label='LIF')
        ax.plot(times, xhat_alif[:,1], label='ALIF')
        ax.plot(times, xhat_wilson[:,1], label='Wilson')
        ax.plot(times, xhat_durstewitz[:,1], label='Durstewitz')
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$')
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillator_states1_time_train.png")

        fig, ax = plt.subplots(figsize=((8, 8)))
        ax.plot(target[:,0], target[:,1], alpha=0.5, linestyle="--", label='target')
        ax.plot(xhat_nef[:,0], xhat_nef[:,1], alpha=0.5, label='NEF')
        ax.plot(xhat_lif[:,0], xhat_lif[:,1], alpha=0.5, label='LIF')
        ax.plot(xhat_alif[:,0], xhat_alif[:,1], alpha=0.5, label='ALIF')
        ax.plot(xhat_wilson[:,0], xhat_wilson[:,1], alpha=0.5, label='Wilson')
        ax.plot(xhat_durstewitz[:,0], xhat_durstewitz[:,1], alpha=0.5, label='Durstewitz')
        ax.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{y}$', xlim=((-1, 1)), ylim=((-1, 1)))
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillator_states_train.png")

    print('test')
    nrmses = np.zeros((5, 1))
    data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=n_neurons, t_supv=2.5, t=t, f=f, dt=dt, g=g, b=b, m=m, i=i, freq=freq, seed=seed)

    a_lif = f_lif.filt(downsample_spikes(data['lif'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
    a_alif = f_alif.filt(downsample_spikes(data['alif'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
    a_wilson = f_wilson.filt(downsample_spikes(data['wilson'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
    a_durstewitz = f_durstewitz.filt(downsample_spikes(data['durstewitz'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
    target = f.filt(data['x'][::int(dt_sample/dt)], dt=dt_sample)[int(t_transient_test/dt_sample):]
    times = data['times'][::int(dt_sample/dt)][int(t_transient_test/dt_sample):]
    xhat_nef = data['nef'][::int(dt_sample/dt)][int(t_transient_test/dt_sample):]
    xhat_lif = np.dot(a_lif, d_lif)[int(t_transient_test/dt_sample):]
    xhat_alif = np.dot(a_alif, d_alif)[int(t_transient_test/dt_sample):]
    xhat_wilson = np.dot(a_wilson, d_wilson)[int(t_transient_test/dt_sample):]
    xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)[int(t_transient_test/dt_sample):]

    p0 = [1, 2*np.pi, 0]
    bounds = ((0, np.pi, 0), (1, 3*np.pi, 2*np.pi))
    p_best_target_0, _ = curve_fit(sinusoid, times, target[:,0], p0=p0, bounds=bounds)
    p_best_target_1, _ = curve_fit(sinusoid, times, target[:,1], p0=p0, bounds=bounds)
    p_best_nef_0, _ = curve_fit(sinusoid, times, xhat_nef[:,0], p0=p0, bounds=bounds)
    p_best_nef_1, _ = curve_fit(sinusoid, times, xhat_nef[:,1], p0=p0, bounds=bounds)
    p_best_lif_0, _ = curve_fit(sinusoid, times, xhat_lif[:,0], p0=p0, bounds=bounds)
    p_best_lif_1, _ = curve_fit(sinusoid, times, xhat_lif[:,1], p0=p0, bounds=bounds)
    p_best_alif_0, _ = curve_fit(sinusoid, times, xhat_alif[:,0], p0=p0, bounds=bounds)
    p_best_alif_1, _ = curve_fit(sinusoid, times, xhat_alif[:,1], p0=p0, bounds=bounds)
    p_best_wilson_0, _ = curve_fit(sinusoid, times, xhat_wilson[:,0], p0=p0, bounds=bounds)
    p_best_wilson_1, _ = curve_fit(sinusoid, times, xhat_wilson[:,1], p0=p0, bounds=bounds)
    p_best_durstewitz_0, _ = curve_fit(sinusoid, times, xhat_durstewitz[:,0], p0=p0, bounds=bounds)
    p_best_durstewitz_1, _ = curve_fit(sinusoid, times, xhat_durstewitz[:,1], p0=p0, bounds=bounds)
    error_nef_0 = np.abs(p_best_target_0[0] - p_best_nef_0[0]) + np.abs(p_best_target_0[2] - p_best_nef_0[2])/(2*np.pi*freq)
    error_nef_1 = np.abs(p_best_target_1[0] - p_best_nef_1[0]) + np.abs(p_best_target_1[2] - p_best_nef_1[2])/(2*np.pi*freq)
    error_lif_0 = np.abs(p_best_target_0[0] - p_best_lif_0[0]) + np.abs(p_best_target_0[2] - p_best_lif_0[2])/(2*np.pi*freq)
    error_lif_1 = np.abs(p_best_target_1[0] - p_best_lif_1[0]) + np.abs(p_best_target_1[2] - p_best_lif_1[2])/(2*np.pi*freq)
    error_alif_0 = np.abs(p_best_target_0[0] - p_best_alif_0[0]) + np.abs(p_best_target_0[2] - p_best_alif_0[2])/(2*np.pi*freq)
    error_alif_1 = np.abs(p_best_target_1[0] - p_best_alif_1[0]) + np.abs(p_best_target_1[2] - p_best_alif_1[2])/(2*np.pi*freq)
    error_wilson_0 = np.abs(p_best_target_0[0] - p_best_wilson_0[0]) + np.abs(p_best_target_0[2] - p_best_wilson_0[2])/(2*np.pi*freq)
    error_wilson_1 = np.abs(p_best_target_1[0] - p_best_wilson_1[0]) + np.abs(p_best_target_1[2] - p_best_wilson_1[2])/(2*np.pi*freq)
    error_durstewitz_0 = np.abs(p_best_target_0[0] - p_best_durstewitz_0[0]) + np.abs(p_best_target_0[2] - p_best_durstewitz_0[2])/(2*np.pi*freq)
    error_durstewitz_1 = np.abs(p_best_target_1[0] - p_best_durstewitz_1[0]) + np.abs(p_best_target_1[2] - p_best_durstewitz_1[2])/(2*np.pi*freq)
    nrmses = np.array([
        error_nef_0+error_nef_1,
        error_lif_0+error_lif_1,
        error_alif_0+error_alif_1,
        error_wilson_0+error_wilson_1,
        error_durstewitz_0+error_durstewitz_1])
    np.savez("data/nrmses_oscillator.npz", nrmses=nrmses)
    print('nrmses:', nrmses)

    # a_nef_0, b_nef_0, c_nef_0, error_nef_0 = fit_sinusoid(xhat_nef[:,0], times, freq=freq)
    # a_lif_0, b_lif_0, c_lif_0, error_lif_0 = fit_sinusoid(xhat_lif[:,0], times, freq=freq)
    # a_alif_0, b_alif_0, c_alif_0, error_alif_0 = fit_sinusoid(xhat_alif[:,0], times, freq=freq)
    # a_wilson_0, b_wilson_0, c_wilson_0, error_wilson_0 = fit_sinusoid(xhat_wilson[:,0], times, freq=freq)
    # a_durstewitz_0, b_durstewitz_0, c_durstewitz_0, error_durstewitz_0 = fit_sinusoid(xhat_durstewitz[:,0], times, freq=freq)
    # a_nef_1, b_nef_1, c_nef_1, error_nef_1 = fit_sinusoid(xhat_nef[:,1], times, freq=freq)
    # a_lif_1, b_lif_1, c_lif_1, error_lif_1 = fit_sinusoid(xhat_lif[:,1], times, freq=freq)
    # a_alif_1, b_alif_1, c_alif_1, error_alif_1 = fit_sinusoid(xhat_alif[:,1], times, freq=freq)
    # a_wilson_1, b_wilson_1, c_wilson_1, error_wilson_1 = fit_sinusoid(xhat_wilson[:,1], times, freq=freq)
    # a_durstewitz_1, b_durstewitz_1, c_durstewitz_1, error_durstewitz_1 = fit_sinusoid(xhat_durstewitz[:,1], times, freq=freq)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    ax1.plot(times, target[:,0], label='target')
    ax1.plot(times, sinusoid(times, *p_best_target_0), label='target fit')
    ax1.plot(times, sinusoid(times, *p_best_nef_0), label='NEF, error=%.3f' %error_nef_0)
    ax1.plot(times, sinusoid(times, *p_best_lif_0), label='LIF, error=%.3f' %error_lif_0)
    ax1.plot(times, sinusoid(times, *p_best_alif_0), label='ALIF, error=%.3f' %error_alif_0)
    ax1.plot(times, sinusoid(times, *p_best_wilson_0), label='Wilson, error=%.3f' %error_wilson_0)
    ax1.plot(times, sinusoid(times, *p_best_durstewitz_0), label='Durstewitz, error=%.3f' %error_durstewitz_0)
    ax1.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title='dimension 0')
    ax1.legend()
    ax2.plot(times, target[:,1], label='target')
    ax2.plot(times, sinusoid(times, *p_best_target_1), label='target fit')
    ax2.plot(times, sinusoid(times, *p_best_nef_1), label='NEF, error=%.3f' %error_nef_1)
    ax2.plot(times, sinusoid(times, *p_best_lif_1), label='LIF, error=%.3f' %error_lif_1)
    ax2.plot(times, sinusoid(times, *p_best_alif_1), label='ALIF, error=%.3f' %error_alif_1)
    ax2.plot(times, sinusoid(times, *p_best_wilson_1), label='Wilson, error=%.3f' %error_wilson_1)
    ax2.plot(times, sinusoid(times, *p_best_durstewitz_1), label='Durstewitz, error=%.3f' %error_durstewitz_1)
    ax2.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title='dimension 1')
    ax2.legend()
    plt.savefig("plots/oscillator_fits.png")

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax1.plot(times, xhat_nef[:,0])
    ax1.plot(times, sinusoid(times, *p_best_nef_0))
    ax1.set(ylabel=r'$\mathbf{x}$', title="NEF, error=%.3f" %error_nef_0)
    ax2.plot(times, xhat_lif[:,0])
    ax2.plot(times, sinusoid(times, *p_best_lif_0))
    ax2.set(ylabel=r'$\mathbf{x}$', title="LIF, error=%.3f" %error_lif_0)
    ax3.plot(times, xhat_alif[:,0])
    ax3.plot(times, sinusoid(times, *p_best_alif_0))
    ax3.set(ylabel=r'$\mathbf{x}$', title="ALIF, error=%.3f" %error_alif_0)
    ax4.plot(times, xhat_wilson[:,0])
    ax4.plot(times, sinusoid(times, *p_best_wilson_0))
    ax4.set(ylabel=r'$\mathbf{x}$', title="Wilson, error=%.3f" %error_wilson_0)
    ax5.plot(times, xhat_durstewitz[:,0])
    ax5.plot(times, sinusoid(times, *p_best_durstewitz_0))
    ax5.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="Durstewitz, error=%.3f" %error_durstewitz_0)
    plt.savefig("plots/oscillator_fits2.png")

    fig, ax = plt.subplots(figsize=((12, 8)))
    ax.plot(times, target[:,0], linestyle="--", label='target')
    ax.plot(times, xhat_nef[:,0], label='NEF, nrmse=%.3f'%error_nef_0)
    ax.plot(times, xhat_lif[:,0], label='LIF, nrmse=%.3f'%error_lif_0)
    ax.plot(times, xhat_alif[:,0], label='ALIF, nrmse=%.3f'%error_alif_0)
    ax.plot(times, xhat_wilson[:,0], label='Wilson, nrmse=%.3f'%error_wilson_0)
    ax.plot(times, xhat_durstewitz[:,0], label='Durstewitz, nrmse=%.3f'%error_durstewitz_0)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$')
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillator_states0_time_test.png")

    fig, ax = plt.subplots(figsize=((12, 8)))
    ax.plot(times, target[:,1], linestyle="--", label='target')
    ax.plot(times, xhat_nef[:,1], label='NEF, nrmse=%.3f'%error_nef_1)
    ax.plot(times, xhat_lif[:,1], label='LIF, nrmse=%.3f'%error_lif_1)
    ax.plot(times, xhat_alif[:,1], label='ALIF, nrmse=%.3f'%error_alif_1)
    ax.plot(times, xhat_wilson[:,1], label='Wilson, nrmse=%.3f'%error_wilson_1)
    ax.plot(times, xhat_durstewitz[:,1], label='Durstewitz, nrmse=%.3f'%error_durstewitz_1)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$')
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillator_states1_time_test.png")

    fig, ax = plt.subplots(figsize=((8, 8)))
    ax.plot(target[:,0], target[:,1], alpha=0.5, linestyle="--", label='target')
    ax.plot(xhat_nef[:,0], xhat_nef[:,1], alpha=0.5, label='NEF')
    ax.plot(xhat_lif[:,0], xhat_lif[:,1], alpha=0.5, label='LIF')
    ax.plot(xhat_alif[:,0], xhat_alif[:,1], alpha=0.5, label='ALIF')
    ax.plot(xhat_wilson[:,0], xhat_wilson[:,1], alpha=0.5, label='Wilson')
    ax.plot(xhat_durstewitz[:,0], xhat_durstewitz[:,1], alpha=0.5, label='Durstewitz')
    ax.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{y}$', xlim=((-1, 1)), ylim=((-1, 1)))
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillator_states_test.png")

run(n_neurons=100, t_train=10, t=10, t_transient_train=2.5, t_transient_test=2.5, order=2, gb_evals=0, df_evals=100, dt=0.000025)
    # load_fd="data/fd_oscillator.npz")