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


def go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        n_neurons=100, t=10, m=Uniform(10, 20), i=Uniform(-1, 1), seed=0, dt=0.001, f=Lowpass(0.1), T=0.1,
        stim_func=lambda t: np.sin(t), g=None, b=None, g2=None, b2=None, mirror=False, norm_val=1.0, supv=0):

    solver_lif = NoSolver(d_lif)
    solver_alif = NoSolver(d_alif)
    solver_wilson = NoSolver(d_wilson)
    solver_durstewitz = NoSolver(d_durstewitz)
    t_norm = t/2 if mirror else t
    norm = norms(t_norm, dt, stim_func, f=f, value=norm_val)
    
    with nengo.Network(seed=seed) as model:
                    
        # Stimulus and Nodes
        model.T = t
        def flip(t, x):
            if t<model.T/2: return x
            elif t>=model.T/2: return -1.0*x
        u_raw = nengo.Node(stim_func)
        u = nengo.Node(output=flip, size_in=1) if mirror else nengo.Node(size_in=1)
        nengo.Connection(u_raw, u, synapse=None, transform=norm)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, label='pre_u')
        pre_x = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, label='pre_x')
        nef = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, label='nef')
        supv_lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIFNorm(max_x=norm_val), seed=seed, label='supv_lif')
        supv_alif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, label='supv_alif')
        supv_wilson = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed, label='supv_wilson')
        supv_durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, label='supv_durstewitz')
        lif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIFNorm(max_x=norm_val), seed=seed, label='lif')
        alif = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), seed=seed, label='alif')
        wilson = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=WilsonEuler(), seed=seed, label='wilson')
        durstewitz = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, label='durstewitz')

        # Target connections
        nengo.Connection(u, x, synapse=1/s)
        nengo.Connection(u, pre_u, synapse=None, seed=seed)
        nengo.Connection(x, pre_x, synapse=None, seed=seed)
        nengo.Connection(x, tar, synapse=f)

        # Feedforward connections
        pre_u_nef = nengo.Connection(pre_u, nef, synapse=f, transform=T, seed=seed, label='pre_u_nef')
        pre_u_lif = nengo.Connection(pre_u, lif, synapse=f, transform=T, seed=seed, label='pre_u_lif')
        pre_u_alif = nengo.Connection(pre_u, alif, synapse=f, transform=T, seed=seed, label='pre_u_alif')
        pre_u_wilson = nengo.Connection(pre_u, wilson, synapse=f, transform=T, seed=seed, label='pre_u_wilson')
        pre_u_durstewitz = nengo.Connection(pre_u, durstewitz, synapse=f, transform=T, seed=seed, label='pre_u_durstewitz')
        pre_x_supv_lif = nengo.Connection(pre_x, supv_lif, synapse=f, seed=seed, label='pre_x_supv_lif')
        pre_x_supv_alif = nengo.Connection(pre_x, supv_alif, synapse=f, seed=seed, label='pre_x_supv_alif')
        pre_x_supv_wilson = nengo.Connection(pre_x, supv_wilson, synapse=f, seed=seed, label='pre_x_supv_wilson')
        pre_x_supv_durstewitz = nengo.Connection(pre_x, supv_durstewitz, synapse=f, seed=seed, label='pre_x_supv_durstewitz')

        # Feedback Connections
        if supv:
            supv_lif_lif = nengo.Connection(supv_lif, lif, synapse=f_lif, solver=solver_lif, seed=seed, label='supv_lif')
            supv_alif_alif = nengo.Connection(supv_alif, alif, synapse=f_alif, solver=solver_alif, seed=seed, label='supv_alif')
            supv_wilson_wilson = nengo.Connection(supv_wilson, wilson, synapse=f_wilson, solver=solver_wilson, seed=seed, label='supv_wilson')
            supv_durstewitz_durstewitz = nengo.Connection(supv_durstewitz, durstewitz, synapse=f_durstewitz, solver=solver_durstewitz, seed=seed, label='supv_durstewitz')
        else:
            lif_lif = nengo.Connection(lif, lif, synapse=f_lif, solver=solver_lif, seed=seed, label='lif_lif')
            alif_alif = nengo.Connection(alif, alif, synapse=f_alif, solver=solver_alif, seed=seed, label='alif_alif')
            wilson_wilson = nengo.Connection(wilson, wilson, synapse=f_wilson, solver=solver_wilson, seed=seed, label='wilson_wilson')
            durstewitz_durstewitz = nengo.Connection(durstewitz, durstewitz, synapse=f_durstewitz, solver=solver_durstewitz, seed=seed, label='durstewitz_durstewitz')
        nef_nef = nengo.Connection(nef, nef, synapse=f, seed=seed, label='nef_nef')

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_pre_u = nengo.Probe(pre_u.neurons, synapse=None)
        p_pre_x = nengo.Probe(pre_x.neurons, synapse=None)
        p_tar = nengo.Probe(tar, synapse=None)
        p_nef = nengo.Probe(nef, synapse=f)
        # p_nef = nengo.Probe(nef.neurons, synapse=None)
        p_supv_lif = nengo.Probe(supv_lif.neurons, synapse=None)
        p_supv_alif = nengo.Probe(supv_alif.neurons, synapse=None)
        p_supv_wilson = nengo.Probe(supv_wilson.neurons, synapse=None)
        p_supv_durstewitz = nengo.Probe(supv_durstewitz.neurons, synapse=None)
        p_lif = nengo.Probe(lif.neurons, synapse=None)
        p_alif = nengo.Probe(alif.neurons, synapse=None)
        p_wilson = nengo.Probe(wilson.neurons, synapse=None)
        p_durstewitz = nengo.Probe(durstewitz.neurons, synapse=None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre_u_durstewitz.gain = g
            pre_u_durstewitz.bias = b
            pre_x_supv_durstewitz.gain = g
            pre_x_supv_durstewitz.bias = b
            if supv:
                supv_durstewitz_durstewitz.gain = g2
                supv_durstewitz_durstewitz.bias = b2
            else:
                durstewitz_durstewitz.gain = g2
                durstewitz_durstewitz.bias = b2

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim) 
    
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        pre_u=sim.data[p_pre_u],
        pre_x=sim.data[p_pre_x],
        tar=sim.data[p_tar],
        nef=sim.data[p_nef],
        supv_lif=sim.data[p_supv_lif],
        supv_alif=sim.data[p_supv_alif],
        supv_wilson=sim.data[p_supv_wilson],
        supv_durstewitz=sim.data[p_supv_durstewitz],
        lif=sim.data[p_lif],
        alif=sim.data[p_alif],
        wilson=sim.data[p_wilson],
        durstewitz=sim.data[p_durstewitz],
        enc=sim.data[durstewitz].encoders)


def run(n_neurons=100, t_train=10, t=10, f=Lowpass(0.1), dt=0.001, n_trains=1,
        n_tests=10, gb_evals=10, gb_evals2=10, df_evals=100, order=1,
        load_gb=False, load_fd=False, load_gb2=False, load_fd2=False):

    g = 1e-2 * np.ones((n_neurons, 1))
    g2 = 1e-2 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    b2 = np.zeros((n_neurons, 1))
    f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
    d_lif, d_alif, d_wilson, d_durstewitz = np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1)), np.zeros((n_neurons, 1))
    stim_func = lambda t: np.sin(t)

    if load_gb:
        load = np.load(load_gb)
        g = load['g']
        b = load['b']
        g2 = load['g2']
        b2 = load['b2']
    else:
        for gb in range(gb_evals):
            print("gain1/bias1 evaluation #%s"%gb)
            # stim_func = nengo.processes.WhiteSignal(period=t_train/2, high=1, rms=0.5, seed=gb)
            data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func,
                T=1.0, supv=1, norm_val=1.2)
            g, b, losses = gb_opt(data['durstewitz'], data['lif'], data['u'], data['enc'], g, b,
                dt=dt, name="plots/tuning/integrator_eval%s_"%gb)
            g2 = g
            b2 = b
        np.savez('data/gb_integrator.npz', g=g, b=b, g2=g2, b2=b2)

    # Run many short trials to generate training data for decoders and filters without large drift.
    if load_fd:
        load = np.load(load_fd)
        d_lif = load['d_lif']
        d_alif = load['d_alif']
        d_wilson = load['d_wilson']
        d_durstewitz = load['d_durstewitz']
        f_lif = Lowpass(load['tau_lif'])[0]
        f_alif = Lowpass(load['tau_alif'])[0]
        f_wilson = Lowpass(load['tau_wilson'])[0]
        f_durstewitz = Lowpass(load['tau_durstewitz'])[0]
    else:
        if load_gb or gb_evals == 0:
            print('gathering filter/decoder training data')
            data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func,
                T=1.0, norm_val=1.0, supv=1)
        if df_evals:
            print('optimizing filters and decoders')
            d_lif, f_lif  = df_opt(f.filt(data['u'], dt=dt), data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_pre_u_lif')
            d_alif, f_alif  = df_opt(f.filt(data['u'], dt=dt), data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_pre_u_alif')
            d_wilson, f_wilson  = df_opt(f.filt(data['u'], dt=dt), data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_pre_u_wilson')
            d_durstewitz, f_durstewitz  = df_opt(f.filt(data['u'], dt=dt), data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_pre_u_durstewitz')
        else:
            f_lif, f_alif, f_wilson, f_durstewitz = f, f, f, f
            d_lif = d_opt(f.filt(data['u'], dt=dt), data['lif'], f_lif, f, dt=dt)
            d_alif = d_opt(f.filt(data['u'], dt=dt), data['alif'], f_alif, f, dt=dt)
            d_wilson = d_opt(f.filt(data['u'], dt=dt), data['wilson'], f_wilson, f, dt=dt)
            d_durstewitz = d_opt(f.filt(data['u'], dt=dt), data['durstewitz'], f_durstewitz, f, dt=dt)
        # A_lif = np.zeros((n_trains, int(t_train/dt), n_neurons))
        # A_alif = np.zeros((n_trains, int(t_train/dt), n_neurons))
        # A_wilson = np.zeros((n_trains, int(t_train/dt), n_neurons))
        # A_durstewitz = np.zeros((n_trains, int(t_train/dt), n_neurons))
        # nef = np.zeros((n_trains, int(t_train/dt), 1))
        # targets = np.zeros((n_trains, int(t_train/dt), 1))
        # time = np.zeros((n_trains, int(t_train/dt)))
        # for n in range(n_trains):
        #     print("training evaluation #%s"%n)
        #     # stim_func = nengo.processes.WhiteSignal(period=t_train/2, high=1, rms=0.5, seed=n)
        #     data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
        #         n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func, norm_val=1.2, supv=1)
        #     A_lif[n] = data['lif']
        #     A_alif[n] = data['alif']
        #     A_wilson[n] = data['wilson']
        #     A_durstewitz[n] = data['durstewitz']
        #     nef[n] = data['nef']
        #     targets[n] = data['tar']
        #     time[n] = data['times']
        # A_lif = A_lif.reshape((n_trains*int(t_train/dt), n_neurons))
        # A_alif = A_alif.reshape((n_trains*int(t_train/dt), n_neurons))
        # A_wilson = A_wilson.reshape((n_trains*int(t_train/dt), n_neurons))
        # A_durstewitz = A_durstewitz.reshape((n_trains*int(t_train/dt), n_neurons))
        # nef = targets.reshape((n_trains*int(t_train/dt), 1))        
        # targets = targets.reshape((n_trains*int(t_train/dt), 1))        
        # time = time.reshape((n_trains*int(t_train/dt)))        
        # if df_evals:
        #     print('optimizing filters/decoders')
        #     d_lif, f_lif= df_opt(targets, A_lif, f, order=order, df_evals=df_evals, dt=dt, name='integrator_lif')
        #     d_alif, f_alif = df_opt(targets, A_alif, f, order=order, df_evals=df_evals, dt=dt, name='integrator_alif')
        #     d_wilson, f_wilson = df_opt(targets, A_wilson, f, order=order, df_evals=df_evals, dt=dt, name='integrator_wilson')
        #     d_durstewitz, f_durstewitz = df_opt(targets, A_durstewitz, f, order=order, df_evals=df_evals, dt=dt, name='integrator_durstewitz')
        # else:
        #     d_lif = d_opt(targets, A_lif, f_lif, f, dt=dt)
        #     d_alif = d_opt(targets, A_alif, f_alif, f, dt=dt)
        #     d_wilson = d_opt(targets, A_wilson, f_wilson, f, dt=dt)
        #     d_durstewitz = d_opt(targets, A_durstewitz, f_durstewitz, f, dt=dt)
        np.savez('data/fd_integrator.npz',
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
        plt.savefig("plots/integrator_filters.png")
            
        # a_lif = f_lif.filt(A_lif, dt=dt)
        # a_alif = f_alif.filt(A_alif, dt=dt)
        # a_wilson = f_wilson.filt(A_wilson, dt=dt)
        # a_durstewitz = f_durstewitz.filt(A_durstewitz, dt=dt)
        # target = f.filt(targets, dt=dt)
        a_lif = f_lif.filt(data['lif'], dt=dt)
        a_alif = f_alif.filt(data['alif'], dt=dt)
        a_wilson = f_wilson.filt(data['wilson'], dt=dt)
        a_durstewitz = f_durstewitz.filt(data['durstewitz'], dt=dt)
        target = f.filt(data['u'], dt=dt)
        # xhat_nef = data['nef']
        xhat_lif = np.dot(a_lif, d_lif)
        xhat_alif = np.dot(a_alif, d_alif)
        xhat_wilson = np.dot(a_wilson, d_wilson)
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(data['times'], target, linestyle="--", label='target')
        # ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f' %nrmse(xhat_nef, target=target))
        ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f' %nrmse(xhat_lif, target=target))
        ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f' %nrmse(xhat_alif, target=target))
        ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f' %nrmse(xhat_wilson, target=target))
        ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f' %nrmse(xhat_durstewitz, target=target))
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens1)")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrator_states_train.png")

    stim_func = lambda t: np.cos(t)
    if load_gb2:
        load = np.load(load_gb2)
        g = load['g']
        b = load['b']
        g2 = load['g2']
        b2 = load['b2']
    else:
        g2 = g
        b2 = b
        for gb in range(gb_evals2):
            print("gain2/bias2 evaluation #%s"%gb)
            # stim_func = nengo.processes.WhiteSignal(period=t_train/2, high=1, rms=0.5, seed=gb)
            data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
                n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func,
                supv=1, norm_val=1.2)
            inpt = f.filt(data['u'], dt=dt)*0.1 + f.filt(data['x'], dt=dt)  # combined input to ens through pre_u and supv
            g2, b2, losses = gb_opt(data['durstewitz'], data['lif'], inpt, data['enc'], g, b,
                dt=dt, name="plots/tuning/integrator2_eval%s_"%gb)
        np.savez('data/gb_integrator2.npz', g=g, b=b, g2=g2, b2=b2)

    d_lif_out = d_lif
    d_alif_out = d_alif
    d_wilson_out = d_wilson
    d_durstewitz_out = d_durstewitz
    f_lif_out = f_lif
    f_alif_out = f_alif
    f_wilson_out = f_wilson
    f_durstewitz_out = f_durstewitz
    # if load_fd2:
    #     load = np.load(load_fd2)
    #     d_lif_out = load['d_lif_out']
    #     d_alif_out = load['d_alif_out']
    #     d_wilson_out = load['d_wilson_out']
    #     d_durstewitz_out = load['d_durstewitz_out']
    #     f_lif_out = Lowpass(load['tau_lif_out'])[0]
    #     f_alif_out = Lowpass(load['tau_alif_out'])[0]
    #     f_wilson_out = Lowpass(load['tau_wilson_out'])[0]
    #     f_durstewitz_out = Lowpass(load['tau_durstewitz_out'])[0]
    # else:
    #     if load_gb2 or gb_evals2 == 0:
    #         print('gathering filter2/decoder2 training data')
    #         data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
    #             n_neurons=n_neurons, t=t_train, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func,
    #             T=0.1, norm_val=1.0, supv=1)
    #     if df_evals:
    #         print('optimizing filters2 and decoders2')
    #         d_lif_out, f_lif_out  = df_opt(data['tar'], data['lif'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_lif_out')
    #         d_alif_out, f_alif_out  = df_opt(data['tar'], data['alif'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_alif_out')
    #         d_wilson_out, f_wilson_out  = df_opt(data['tar'], data['wilson'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_wilson_out')
    #         d_durstewitz_out, f_durstewitz_out  = df_opt(data['tar'], data['durstewitz'], f, order=order, df_evals=df_evals, dt=dt, name='integrator_durstewitz_out')
    #     else:
    #         f_lif_out, f_alif_out, f_wilson_out, f_durstewitz_out = f, f, f, f
    #         d_lif_out = d_opt(data['tar'], data['lif'], f_lif, f, dt=dt)
    #         d_alif_out = d_opt(data['tar'], data['alif'], f_alif, f, dt=dt)
    #         d_wilson_out = d_opt(data['tar'], data['wilson'], f_wilson, f, dt=dt)
    #         d_durstewitz_out = d_opt(data['tar'], data['durstewitz'], f_durstewitz, f, dt=dt)
    #     np.savez('data/fd_integrator2.npz',
    #         d_lif_out=d_lif_out,
    #         d_alif_out=d_alif_out,
    #         d_wilson_out=d_wilson_out,
    #         d_durstewitz_out=d_durstewitz_out,
    #         tau_lif_out=-1.0/f_lif_out.poles,
    #         tau_alif_out=-1.0/f_alif_out.poles,
    #         tau_wilson_out=-1.0/f_wilson_out.poles,
    #         tau_durstewitz_out=-1.0/f_durstewitz_out.poles)
        
    #     times = np.arange(0, 1, 0.0001)
    #     fig, ax = plt.subplots(figsize=((12, 8)))
    #     ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
    #     ax.plot(times, f_lif_out.impulse(len(times), dt=0.0001), label="f_lif_out, nonzero d: %s/%s"%(np.count_nonzero(d_lif_out), n_neurons))
    #     ax.plot(times, f_alif_out.impulse(len(times), dt=0.0001), label="f_alif_out, nonzero d: %s/%s"%(np.count_nonzero(d_alif_out), n_neurons))
    #     ax.plot(times, f_wilson_out.impulse(len(times), dt=0.0001), label="f_wilson_out, nonzero d: %s/%s"%(np.count_nonzero(d_wilson_out), n_neurons))
    #     ax.plot(times, f_durstewitz_out.impulse(len(times), dt=0.0001), label="f_durstewitz_out, nonzero d: %s/%s"%(np.count_nonzero(d_durstewitz_out), n_neurons))
    #     ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
    #     ax.legend(loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig("plots/integrator_filters_out.png")
            
    #     a_lif = f_lif_out.filt(data['lif'], dt=dt)
    #     a_alif = f_alif_out.filt(data['alif'], dt=dt)
    #     a_wilson = f_wilson_out.filt(data['wilson'], dt=dt)
    #     a_durstewitz = f_durstewitz_out.filt(data['durstewitz'], dt=dt)
    #     target = f.filt(data['tar'], dt=dt)
    #     xhat_nef = data['nef']
    #     xhat_lif = np.dot(a_lif, d_lif_out)
    #     xhat_alif = np.dot(a_alif, d_alif_out)
    #     xhat_wilson = np.dot(a_wilson, d_wilson_out)
    #     xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz_out)

    #     fig, ax = plt.subplots(figsize=((12, 8)))
    #     ax.plot(data['times'], target, linestyle="--", label='target')
    #     ax.plot(data['times'], xhat_nef, label='NEF, nrmse=%.3f' %nrmse(xhat_nef, target=target))
    #     ax.plot(data['times'], xhat_lif, label='LIF, nrmse=%.3f' %nrmse(xhat_lif, target=target))
    #     ax.plot(data['times'], xhat_alif, label='ALIF, nrmse=%.3f' %nrmse(xhat_alif, target=target))
    #     ax.plot(data['times'], xhat_wilson, label='Wilson, nrmse=%.3f' %nrmse(xhat_wilson, target=target))
    #     ax.plot(data['times'], xhat_durstewitz, label='Durstewitz, nrmse=%.3f' %nrmse(xhat_durstewitz, target=target))
    #     ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train (ens1)")
    #     plt.legend(loc='upper right')
    #     plt.savefig("plots/integrator_states_train2.png")


    print('running experimental tests')
    nrmses = np.zeros((5, n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t/2, high=1, rms=1, seed=100+test)
        data = go(d_lif, d_alif, d_wilson, d_durstewitz, f_lif, f_alif, f_wilson, f_durstewitz,
            n_neurons=n_neurons, t=t, f=f, dt=dt, g=g, b=b, g2=g2, b2=b2, stim_func=stim_func, mirror=True)

        a_lif = f_lif_out.filt(data['lif'], dt=dt)
        a_alif = f_alif_out.filt(data['alif'], dt=dt)
        a_wilson = f_wilson_out.filt(data['wilson'], dt=dt)
        a_durstewitz = f_durstewitz_out.filt(data['durstewitz'], dt=dt)
        target = data['tar']
        xhat_nef = data['nef']
        xhat_lif = np.dot(a_lif, d_lif_out)
        xhat_alif = np.dot(a_alif, d_alif_out)
        xhat_wilson = np.dot(a_wilson, d_wilson_out)
        xhat_durstewitz = np.dot(a_durstewitz, d_durstewitz_out)
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
        plt.savefig("plots/integrator_states_test%s.png"%test)
            
    if n_tests > 1:
        nt_names =  ['LIF\n(static)', 'LIF\n(temporal)', 'ALIF', 'Wilson', 'Durstewitz']
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        sns.barplot(data=nrmses.T)
        ax.set(ylabel='NRMSE')
        plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
        plt.savefig("plots/integrator_nrmses.png")
    means = np.mean(nrmses, axis=1)
    CIs = np.zeros((5, 2))
    for nt in range(nrmses.shape[0]):
        CIs[nt] = sns.utils.ci(nrmses[nt])
    print('nrmses: ', nrmses)
    print('means: ', means)
    print('confidence intervals: ', CIs)
    np.savez('data/nrmses_integrator.npz', nrmses=nrmses, means=means, CIs=CIs)


run(n_neurons=200, t_train=20, n_tests=1, gb_evals=0, gb_evals2=0, order=2, df_evals=100, dt=0.001)