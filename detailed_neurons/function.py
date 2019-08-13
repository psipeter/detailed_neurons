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


def go(fx, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=100, t=10, seed=1, dt=0.001, f=Lowpass(0.1), m=Uniform(10, 20), i=Uniform(-1, 1),
        stim_func=lambda t: np.sin(t), g=None, b=None, g2=None, b2=None, norm_val=1.0, mirror=False):

    solver_lif = NoSolver(d_lif)
    solver_alif = NoSolver(d_alif)
    solver_wilson = NoSolver(d_wilson)
    solver_durstewitz = NoSolver(d_durstewitz)
    t_norm = t/2 if mirror else t
    norm = norms(t_norm, dt, stim_func, f=f, value=norm_val)
    # e2 = nengo.dists.Choice([[-1], [1]]).sample(n_neurons, rng=np.random.RandomState(seed=seed))
    # i2 = Uniform(0.2, 0.8).sample(n_neurons, rng=np.random.RandomState(seed=seed)) * e2.ravel()
 
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
        pre = nengo.Ensemble(100, 1, max_rates=m, intercepts=i, seed=seed, label='pre')
        nef = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, label='nef')
        lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIFNorm(max_x=norm_val), seed=seed, label='lif')
        alif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, label='alif')
        wilson = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed, label='wilson')
        durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=DurstewitzNeuron(), seed=seed, label='durstewitz')
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        nef2 = nengo.Ensemble(n_neurons, 1, max_rates=m, neuron_type=nengo.LIF(), seed=seed, label='nef2')
        lif2 = nengo.Ensemble(n_neurons, 1, max_rates=m, neuron_type=LIFNorm(max_x=norm_val), seed=seed, label='lif2')
        alif2 = nengo.Ensemble(n_neurons, 1, max_rates=m, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, label='alif2')
        wilson2 = nengo.Ensemble(n_neurons, 1, max_rates=m, neuron_type=WilsonEuler(), seed=seed, label='wilson2')
        durstewitz2 = nengo.Ensemble(n_neurons, 1, max_rates=m, neuron_type=DurstewitzNeuron(), seed=seed, label='durstewitz2')
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, tar, synapse=f, seed=seed)
        nengo.Connection(pre, nef, synapse=f, seed=seed, label='pre_nef')
        nengo.Connection(pre, lif, synapse=f, seed=seed, label='pre_lif')
        nengo.Connection(pre, alif, synapse=f, seed=seed, label='pre_alif')
        nengo.Connection(pre, wilson, synapse=f, seed=seed, label='pre_wilson')
        pre_durstewitz = nengo.Connection(pre, durstewitz, synapse=f, seed=seed, label='pre_durstewitz')
        nengo.Connection(tar, tar2, synapse=f, seed=seed, function=fx)
        nengo.Connection(nef, nef2, synapse=f, function=fx, seed=seed, label='nef_nef2')
        nengo.Connection(lif, lif2, synapse=f_lif, solver=solver_lif, seed=seed, label='lif_lif2')
        nengo.Connection(alif, alif2, synapse=f_alif, solver=solver_alif, seed=seed, label='alif_alif2')
        nengo.Connection(wilson, wilson2, synapse=f_wilson, solver=solver_wilson, seed=seed, label='wilson_wilson2')
        durstewitz_durstewitz2 = nengo.Connection(durstewitz, durstewitz2, synapse=f_durstewitz, solver=solver_durstewitz, seed=seed, label='durstewitz_durstewitz2')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_pre = nengo.Probe(pre, synapse=f)
        p_tar = nengo.Probe(tar, synapse=None)
        p_nef = nengo.Probe(nef, synapse=f)
        p_nef_spike = nengo.Probe(nef.neurons, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        p_alif = nengo.Probe(alif.neurons, synapse=None)
        p_wilson = nengo.Probe(wilson.neurons, synapse=None)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=None)
        p_tar2 = nengo.Probe(tar2, synapse=None)
        p_nef2 = nengo.Probe(nef2, synapse=f)
        p_nef2_spike = nengo.Probe(nef2.neurons, synapse=None)
        p_lif2 = nengo.Probe(lif2.neurons, synapse=None)
        p_alif2 = nengo.Probe(alif2.neurons, synapse=None)
        p_wilson2 = nengo.Probe(wilson2.neurons, synapse=None)
        p_durstewitz2 = nengo.Probe(durstewitz2.neurons, synapse=None)  

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre_durstewitz.gain = g
        pre_durstewitz.bias = b
        durstewitz_durstewitz2.gain = g2
        durstewitz_durstewitz2.bias = b2

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim) 
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        pre=sim.data[p_pre],
        tar=sim.data[p_tar],
        tar2=sim.data[p_tar2],
        nef=sim.data[p_nef],
        nef_spike=sim.data[p_nef_spike],
        lif=sim.data[p_lif],
        alif=sim.data[p_alif],
        wilson=sim.data[p_wilson],
        durstewitz=sim.data[p_durstewitz],
        nef2=sim.data[p_nef2],
        nef2_spike=sim.data[p_nef2_spike],
        lif2=sim.data[p_lif2],
        alif2=sim.data[p_alif2],
        wilson2=sim.data[p_wilson2],
        durstewitz2=sim.data[p_durstewitz2],
        enc=sim.data[durstewitz].encoders)


def run(fx, n_neurons=100, t_train=10, t=10, f=Lowpass(0.1), dt=0.001,
        n_tests=10, gb_evals=10, gb_evals2=0, df_evals=100, order=1,
        load_gb=False, load_gb2=False, load_fd=False, load_fd2=False):

    g = 2e-3 * np.ones((n_neurons, 1))
    g2 = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    b2 = np.zeros((n_neurons, 1))
    f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
    d_lif, d_alif, d_wilson, d_durstewitz = np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1))
    # omega = np.random.RandomState(seed=gb).uniform(0, 2*np.pi)
    stim_func = lambda t: np.sin(t)

    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
        g2 = g
        b2 = b
    else:
        for gb in range(gb_evals):
            print("gain1/bias1 evaluation #%s"%gb)
            data = go(fx, d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func, norm_val=1.0)
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['tar'], data['enc'], g, b,
            	dt=dt, name="plots/tuning/function_eval%s_"%gb)
        np.savez('data/gb_function.npz', g=g, b=b, g2=g, b2=b)

    if load_fd:
        load = np.load(load_fd)
        d_lif_out = load['d_lif_out']
        d_alif_out = load['d_alif_out']
        d_wilson_out = load['d_wilson_out']
        d_durstewitz_out = load['d_durstewitz_out']
        d_lif_fx = load['d_lif_fx']
        d_alif_fx = load['d_alif_fx']
        d_wilson_fx = load['d_wilson_fx']
        d_durstewitz_fx = load['d_durstewitz_fx']
        f_lif_out = Lowpass(load['tau_lif_out'])[0]
        f_alif_out = Lowpass(load['tau_alif_out'])[0]
        f_wilson_out = Lowpass(load['tau_wilson_out'])[0]
        f_durstewitz_out = Lowpass(load['tau_durstewitz_out'])[0]
        f_lif_fx = Lowpass(load['tau_lif_fx'])[0]
        f_alif_fx = Lowpass(load['tau_alif_fx'])[0]
        f_wilson_fx = Lowpass(load['tau_wilson_fx'])[0]
        f_durstewitz_fx = Lowpass(load['tau_durstewitz_fx'])[0]
    else:
        if load_gb or gb_evals == 0:
            print('gathering filter/decoder training data')
            data = go(fx, d_lif, d_alif, d_wilson, d_durstewitz,
                f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, stim_func=stim_func, norm_val=1.0,
                g=g, b=b, g2=g2, b2=b2)            
        if df_evals:
            print('optimizing filters/decoders for readout')
            d_lif_out, f_lif_out= df_opt(
                data['tar'], data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='function_lif')
            d_alif_out, f_alif_out = df_opt(
                data['tar'], data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='function_alif')
            d_wilson_out, f_wilson_out = df_opt(
                data['tar'], data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='function_wilson')
            d_durstewitz_out, f_durstewitz_out = df_opt(
                data['tar'], data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='function_durstewitz')
            print('optimizing filters/decoders for fx')        
            d_lif_fx, f_lif_fx  = df_opt(
                fx(data['tar']), data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='function_fx_lif')
            d_alif_fx, f_alif_fx  = df_opt(
                fx(data['tar']), data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='function_fx_alif')
            d_wilson_fx, f_wilson_fx  = df_opt(
                fx(data['tar']), data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='function_fx_wilson')
            d_durstewitz_fx, f_durstewitz_fx  = df_opt(
                fx(data['tar']), data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='function_fx_durstewitz')
        else:
            d_lif_out = d_opt(data['tar'], data['lif'], f_lif, f, dt=dt)
            d_alif_out = d_opt(data['tar'], data['alif'], f_alif, f, dt=dt)
            d_wilson_out = d_opt(data['tar'], data['wilson'], f_wilson, f, dt=dt)
            d_durstewitz_out = d_opt(data['tar'], data['durstewitz'], f_durstewitz, f, dt=dt)
            d_lif_fx = d_opt(fx(data['tar']), data['lif'], f_lif, f, dt=dt)
            d_alif_fx = d_opt(fx(data['tar']), data['alif'], f_alif, f, dt=dt)
            d_wilson_fx = d_opt(fx(data['tar']), data['wilson'], f_wilson, f, dt=dt)
            d_durstewitz_fx = d_opt(fx(data['tar']), data['durstewitz'], f_durstewitz, f, dt=dt)
        np.savez('data/fd_function.npz',
            d_lif_out=d_lif_out,
            d_alif_out=d_alif_out,
            d_wilson_out=d_wilson_out,
            d_durstewitz_out=d_durstewitz_out,
            d_lif_fx=d_lif_fx,
            d_alif_fx=d_alif_fx,
            d_wilson_fx=d_wilson_fx,
            d_durstewitz_fx=d_durstewitz_fx,
            tau_lif_out=-1.0/f_lif_out.poles,
            tau_alif_out=-1.0/f_alif_out.poles,
            tau_wilson_out=-1.0/f_wilson_out.poles,
            tau_durstewitz_out=-1.0/f_durstewitz_out.poles,
            tau_lif_fx=-1.0/f_lif_fx.poles,
            tau_alif_fx=-1.0/f_alif_fx.poles,
            tau_wilson_fx=-1.0/f_wilson_fx.poles,
            tau_durstewitz_fx=-1.0/f_durstewitz_fx.poles)
        
        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_lif_out.impulse(len(times), dt=0.0001), label="f_lif readout, nonzero d: %s/%s"%(np.count_nonzero(d_lif_out), n_neurons))
        ax.plot(times, f_alif_out.impulse(len(times), dt=0.0001), label="f_alif readout, nonzero d: %s/%s"%(np.count_nonzero(d_alif_out), n_neurons))
        ax.plot(times, f_wilson_out.impulse(len(times), dt=0.0001), label="f_wilson readout, nonzero d: %s/%s"%(np.count_nonzero(d_wilson_out), n_neurons))
        ax.plot(times, f_durstewitz_out.impulse(len(times), dt=0.0001), label="f_durstewitz readout, nonzero d: %s/%s"%(np.count_nonzero(d_durstewitz_out), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/function_filters_readout.png")

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_lif_fx.impulse(len(times), dt=0.0001), label="f_lif function, nonzero d: %s/%s"%(np.count_nonzero(d_lif_fx), n_neurons))
        ax.plot(times, f_alif_fx.impulse(len(times), dt=0.0001), label="f_alif function, nonzero d: %s/%s"%(np.count_nonzero(d_alif_fx), n_neurons))
        ax.plot(times, f_wilson_fx.impulse(len(times), dt=0.0001), label="f_wilson function, nonzero d: %s/%s"%(np.count_nonzero(d_wilson_fx), n_neurons))
        ax.plot(times, f_durstewitz_fx.impulse(len(times), dt=0.0001), label="f_durstewitz function, nonzero d: %s/%s"%(np.count_nonzero(d_durstewitz_fx), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/function_filters_fx.png")
            
        a_nef_out = f.filt(data['nef_spike'], dt=dt)
        a_lif_out = f_lif_out.filt(data['lif'], dt=dt)
        a_alif_out = f_alif_out.filt(data['alif'], dt=dt)
        a_wilson_out = f_wilson_out.filt(data['wilson'], dt=dt)
        a_durstewitz_out = f_durstewitz_out.filt(data['durstewitz'], dt=dt)
        target = f.filt(data['tar'], dt=dt)
        xhat_nef = data['nef']
        xhat_lif = np.dot(a_lif_out, d_lif_out)
        xhat_alif = np.dot(a_alif_out, d_alif_out)
        xhat_wilson = np.dot(a_wilson_out, d_wilson_out)
        xhat_durstewitz = np.dot(a_durstewitz_out, d_durstewitz_out)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_nef, target=target), np.count_nonzero(np.sum(a_nef_out, axis=0))))
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_lif, target=target), np.count_nonzero(np.sum(a_lif_out, axis=0))))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_alif, target=target), np.count_nonzero(np.sum(a_alif_out, axis=0))))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_wilson, target=target), np.count_nonzero(np.sum(a_wilson_out, axis=0))))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_durstewitz, target=target), np.count_nonzero(np.sum(a_durstewitz_out, axis=0))))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens1)")
        plt.legend(loc='upper right')
        plt.savefig("plots/function_states_train_ens1.png")
        
        a_lif_fx = f_lif_fx.filt(data['lif'], dt=dt)
        a_alif_fx = f_alif_fx.filt(data['alif'], dt=dt)
        a_wilson_fx = f_wilson_fx.filt(data['wilson'], dt=dt)
        a_durstewitz_fx = f_durstewitz_fx.filt(data['durstewitz'], dt=dt)
        target = f.filt(fx(data['tar']), dt=dt)
        xhat_lif = np.dot(a_lif_fx, d_lif_fx)
        xhat_alif = np.dot(a_alif_fx, d_alif_fx)
        xhat_wilson = np.dot(a_wilson_fx, d_wilson_fx)
        xhat_durstewitz = np.dot(a_durstewitz_fx, d_durstewitz_fx)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f' %nrmse(xhat_lif, target=target))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f' %nrmse(xhat_alif, target=target))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f' %nrmse(xhat_wilson, target=target))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f' %nrmse(xhat_durstewitz, target=target))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (fx ens1)")
        plt.legend(loc='upper right')
        plt.savefig("plots/function_states_train_fx_ens1.png")
    
    if load_gb2:
        load = np.load(load_gb2)
        g2 = load['g2']
        b2 = load['b2']
    else:
        g2 = g
        b2 = b
        for gb in range(gb_evals2):
            print("gain2/bias2 evaluation #%s"%gb)
            data = go(fx, d_lif_fx, d_alif_fx, d_wilson_fx, d_durstewitz_fx, f_lif_fx, f_alif_fx, f_wilson_fx, f_durstewitz_fx,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func, norm_val=1.0)
            g2, b2, losses = gb_opt(data['durstewitz2'], data['lif2'], data['tar2'], data['enc'], g2, b2,
            	xmin=0, xmax=1.0, dt=dt, name="plots/tuning/function2_eval%s_"%gb)
        np.savez('data/gb_function2.npz', g=g, b=b, g2=g2, b2=b2)

    # d_lif2_out = d_lif_out
    # d_alif2_out = d_alif_out
    # d_wilson2_out = d_wilson_out
    # d_durstewitz2_out = d_durstewitz_out
    # f_lif2_out = f_lif_out
    # f_alif2_out = f_alif_out
    # f_wilson2_out = f_wilson_out
    # f_durstewitz2_out = f_durstewitz_out
    if load_fd2:
        load = np.load(load_fd2)
        d_lif2_out = load['d_lif2_out']
        d_alif2_out = load['d_alif2_out']
        d_wilson2_out = load['d_wilson2_out']
        d_durstewitz2_out = load['d_durstewitz2_out']
        f_lif2_out = Lowpass(load['tau_lif2_out'])[0]
        f_alif2_out = Lowpass(load['tau_alif2_out'])[0]
        f_wilson2_out = Lowpass(load['tau_wilson2_out'])[0]
        f_durstewitz2_out = Lowpass(load['tau_durstewitz2_out'])[0]
    else:
        if load_gb2 or gb_evals2 == 0:
            print('gathering filter/decoder training data2')
            data = go(fx, d_lif_fx, d_alif_fx, d_wilson_fx, d_durstewitz_fx,
                f_lif_fx, f_alif_fx, f_wilson_fx, f_durstewitz_fx,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, stim_func=stim_func, norm_val=1.0,
                g=g, b=b, g2=g2, b2=b2)            
        if df_evals:
            print('optimizing filters/decoders for readout2')
            d_lif2_out, f_lif2_out= df_opt(
                data['tar2'], data['lif2'], f, order=order, df_evals=df_evals, dt=dt, name='function_lif2')
            d_alif2_out, f_alif2_out = df_opt(
                data['tar2'], data['alif2'], f, order=order, df_evals=df_evals, dt=dt, name='function_alif2')
            d_wilson2_out, f_wilson2_out = df_opt(
                data['tar2'], data['wilson2'], f, order=order, df_evals=df_evals, dt=dt, name='function_wilson2')
            d_durstewitz2_out, f_durstewitz2_out = df_opt(
                data['tar2'], data['durstewitz2'], f, order=order, df_evals=df_evals, dt=dt, name='function_durstewitz2')
        else:
            d_lif2_out = d_opt(data['tar2'], data['lif2'], f_lif, f, dt=dt)
            d_alif2_out = d_opt(data['tar2'], data['alif2'], f_alif, f, dt=dt)
            d_wilson2_out = d_opt(data['tar2'], data['wilson2'], f_wilson, f, dt=dt)
            d_durstewitz2_out = d_opt(data['tar2'], data['durstewitz2'], f_durstewitz, f, dt=dt)
        np.savez('data/fd_function2.npz',
            d_lif2_out=d_lif2_out,
            d_alif2_out=d_alif2_out,
            d_wilson2_out=d_wilson2_out,
            d_durstewitz2_out=d_durstewitz2_out,
            tau_lif2_out=-1.0/f_lif2_out.poles,
            tau_alif2_out=-1.0/f_alif2_out.poles,
            tau_wilson2_out=-1.0/f_wilson2_out.poles,
            tau_durstewitz2_out=-1.0/f_durstewitz2_out.poles)
        
        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_lif2_out.impulse(len(times), dt=0.0001), label="f_lif2 readout, nonzero d: %s/%s"%(np.count_nonzero(d_lif2_out), n_neurons))
        ax.plot(times, f_alif2_out.impulse(len(times), dt=0.0001), label="f_alif2 readout, nonzero d: %s/%s"%(np.count_nonzero(d_alif2_out), n_neurons))
        ax.plot(times, f_wilson2_out.impulse(len(times), dt=0.0001), label="f_wilson2 readout, nonzero d: %s/%s"%(np.count_nonzero(d_wilson2_out), n_neurons))
        ax.plot(times, f_durstewitz2_out.impulse(len(times), dt=0.0001), label="f_durstewitz2 readout, nonzero d: %s/%s"%(np.count_nonzero(d_durstewitz2_out), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/function_filters_readout2.png")
            
        a_nef2_out = f.filt(data['nef2_spike'], dt=dt)
        a_lif2_out = f_lif2_out.filt(data['lif2'], dt=dt)
        a_alif2_out = f_alif2_out.filt(data['alif2'], dt=dt)
        a_wilson2_out = f_wilson2_out.filt(data['wilson2'], dt=dt)
        a_durstewitz2_out = f_durstewitz2_out.filt(data['durstewitz2'], dt=dt)
        target = f.filt(data['tar2'], dt=dt)
        xhat_nef = f.filt(data['nef2'], dt=dt)
        xhat_lif = np.dot(a_lif2_out, d_lif2_out)
        xhat_alif = np.dot(a_alif2_out, d_alif2_out)
        xhat_wilson = np.dot(a_wilson2_out, d_wilson2_out)
        xhat_durstewitz = np.dot(a_durstewitz2_out, d_durstewitz2_out)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_nef, target=target), np.count_nonzero(np.sum(a_nef2_out, axis=0))))
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_lif, target=target), np.count_nonzero(np.sum(a_lif2_out, axis=0))))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_alif, target=target), np.count_nonzero(np.sum(a_alif2_out, axis=0))))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_wilson, target=target), np.count_nonzero(np.sum(a_wilson2_out, axis=0))))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_durstewitz, target=target), np.count_nonzero(np.sum(a_durstewitz2_out, axis=0))))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens2)")
        plt.legend(loc='upper right')
        plt.savefig("plots/function_states_train_ens2.png")


    print('running experimental tests')
    nrmses = np.zeros((5, n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t/2, high=1, rms=1, seed=100+test)
        data = go(fx, d_lif_fx, d_alif_fx, d_wilson_fx, d_durstewitz_fx,
            f_lif_fx, f_alif_fx, f_wilson_fx, f_durstewitz_fx,
            n_neurons=n_neurons, t=t, f=f, dt=dt, stim_func=stim_func,
            g=g, b=b, g2=g2, b2=b2, mirror=True)

        a_nef = f.filt(data['nef_spike'], dt=dt)
        a_lif = f_lif_out.filt(data['lif'], dt=dt)
        a_alif = f_alif_out.filt(data['alif'], dt=dt)
        a_wilson = f_wilson_out.filt(data['wilson'], dt=dt)
        a_durstewitz = f_durstewitz_out.filt(data['durstewitz'], dt=dt)
        target = f.filt(data['tar'], dt=dt)
        xhat_nef = data['nef']
        xhat_lif = np.dot(a_lif, d_lif_out)
        xhat_alif = np.dot(a_alif, d_alif_out)
        xhat_wilson = np.dot(a_wilson, d_wilson_out)
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz_out)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        # ax.plot(data['times'], data['u'], label='u')
        # ax.plot(data['times'], data['pre'], label='pre')
        ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_nef, target=target), np.count_nonzero(np.sum(a_nef, axis=0))))
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_lif, target=target), np.count_nonzero(np.sum(a_lif, axis=0))))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_alif, target=target), np.count_nonzero(np.sum(a_alif, axis=0))))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_wilson, target=target), np.count_nonzero(np.sum(a_wilson, axis=0))))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_durstewitz, target=target), np.count_nonzero(np.sum(a_durstewitz, axis=0))))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens1)")
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title='ensemble 1, test %s'%test)
        plt.legend(loc='upper right')
        plt.savefig("plots/function_states_test%s_ens1.png"%test)

        a_nef2 = f.filt(data['nef2_spike'], dt=dt)
        a_lif2 = f_lif2_out.filt(data['lif2'], dt=dt)
        a_alif2 = f_alif2_out.filt(data['alif2'], dt=dt)
        a_wilson2 = f_wilson2_out.filt(data['wilson2'], dt=dt)
        a_durstewitz2 = f_durstewitz2_out.filt(data['durstewitz2'], dt=dt)
        target2 = f.filt(data['tar2'], dt=dt)
        xhat_nef = data['nef2']
        xhat_lif = np.dot(a_lif2, d_lif2_out)
        xhat_alif = np.dot(a_alif2, d_alif2_out)
        xhat_wilson = np.dot(a_wilson2, d_wilson2_out)
        xhat_durstewitz = np.dot(a_durstewitz2, d_durstewitz2_out)
        nrmses[0, test] = nrmse(xhat_nef, target=target2)
        nrmses[1, test] = nrmse(xhat_lif, target=target2)
        nrmses[2, test] = nrmse(xhat_alif, target=target2)
        nrmses[3, test] = nrmse(xhat_wilson, target=target2)
        nrmses[4, test] = nrmse(xhat_durstewitz, target=target2)
        
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target2, linestyle="--", label='target')
        ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_nef, target=target2), np.count_nonzero(np.sum(a_nef2, axis=0))))
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_lif, target=target2), np.count_nonzero(np.sum(a_lif2, axis=0))))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_alif, target=target2), np.count_nonzero(np.sum(a_alif2, axis=0))))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_wilson, target=target2), np.count_nonzero(np.sum(a_wilson2, axis=0))))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f, nonzero %s'
            %(nrmse(xhat_durstewitz, target=target2), np.count_nonzero(np.sum(a_durstewitz2, axis=0))))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ensemble 2, test %s"%test)
        plt.legend(loc='upper right')
        plt.savefig("plots/function_states_test%s_ens2.png"%test)
            
    if n_tests > 1:
        nt_names =  ['LIF\n(static)', 'LIF\n(temporal)', 'ALIF', 'Wilson', 'Durstewitz']
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        sns.barplot(data=nrmses.T)
        ax.set(ylabel='NRMSE')
        plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
        plt.savefig("plots/function_nrmses.png")
    means = np.mean(nrmses, axis=1)
    CIs = np.zeros((5, 2))
    for nt in range(nrmses.shape[0]):
        CIs[nt] = sns.utils.ci(nrmses[nt])
    print('nrmses: ', nrmses)
    print('means: ', means)
    print('confidence intervals: ', CIs)
    np.savez('data/nrmses_function.npz', nrmses=nrmses, means=means, CIs=CIs)


def fx(x): return np.square(x)
run(fx, n_neurons=100, t_train=20, gb_evals=0, gb_evals2=0, df_evals=200, order=2, dt=0.001,
    load_gb="data/gb_function.npz") # , load_fd="data/fd_function.npz"   