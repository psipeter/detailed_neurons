import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import norms, d_opt, df_opt, LearningNode2
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context='paper', style='white')

def make_signal(value=1.0, t=10.0, t_flat=3, dt=0.001, f=Lowpass(0.01), normed='x', seed=0, test=False):
    stim_func = nengo.processes.WhiteSignal(period=t/2, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim_func)   
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    u = f.filt(sim.data[p_u], dt=dt)
    x = f.filt(sim.data[p_x], dt=dt)
    if normed=='u':  
        norm = value / np.max(np.abs(u))
    elif normed=='x':
        norm = value / np.max(np.abs(x))
    vals1 = sim.data[p_u][:int(t/4/dt),0] * norm
    vals2 = sim.data[p_u][int(t/4/dt):,0] * norm
    flat = np.zeros((int(t_flat/dt)))
    if test:
#         result = np.concatenate((flat, vals1, flat))    
        result = np.concatenate((vals1, vals2, -vals1, -vals2, vals1, flat))    
    else:
        result = np.concatenate((flat, vals1, flat, vals2, -vals1, flat, -vals2))
    return result

def go(d_ens, f_ens, n_neurons=30, n_pre=300, t=10, m=Uniform(30, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, T=0.2, neuron_type=LIF(), f=DoubleExp(1e-3, 2e-1), f_smooth=DoubleExp(1e-2, 2e-1), stim_func=lambda t: np.sin(t), stim_func_base=None, w_x=None, e_x=None, w_u=None, e_u=None, w_fb=None, e_fb=None, L_x=False, L_u=False, L_fd=False, L_fb=False, supervised=False):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(n_pre, 1, radius=3, seed=seed)
        pre_x = nengo.Ensemble(n_pre, 1, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)

        # Connections
        u_pre = nengo.Connection(u, pre_u, synapse=None, seed=seed)
        x_pre = nengo.Connection(x, pre_x, synapse=None, seed=seed)
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        if not L_x:
            pre_u_ens = nengo.Connection(pre_u, ens, synapse=f, transform=T, seed=seed)
            pre_u_supv = nengo.Connection(pre_u, supv, synapse=f, transform=T, seed=seed)
        if L_x or L_fd:
#         if L_x or L_u or L_fd:
            pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
            pre_x_supv = nengo.Connection(pre_x, supv, synapse=f, seed=seed)
        if L_u:
            base = nengo.Node(stim_func_base)
            pre_base = nengo.Ensemble(n_pre, 1, seed=seed)
            nengo.Connection(base, pre_base, synapse=None, seed=seed)
            pre_base_ens = nengo.Connection(pre_base, ens, synapse=f, seed=seed)
            pre_base_supv = nengo.Connection(pre_base, supv, synapse=f, seed=seed)

        if L_fb or supervised:
            u_pre.synapse = f
            x_pre.synapse = None
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            pre_x_ens2 = nengo.Connection(pre_x, ens2, synapse=f, seed=seed)
#             pre_x_supv2 = nengo.Connection(pre_x, supv2, synapse=f, seed=seed)
            pre_x_supv2 = nengo.Connection(pre_x, supv2, synapse=f_ens, seed=seed)
            ens2_ens = nengo.Connection(ens2, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            supv2_supv = nengo.Connection(supv2, supv, synapse=f, seed=seed)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
            p_supv_state = nengo.Probe(supv, synapse=f)
            p_supv2_state = nengo.Probe(supv2, synapse=f)
        else:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            if not (L_x or L_u or L_fd):
                supv_supv = nengo.Connection(supv, supv, synapse=f, seed=seed)

        if L_x:
            node = LearningNode2(n_neurons, n_pre, pre_x_ens, k=3e-6)
            nengo.Connection(pre_x.neurons, node[0: n_pre], synapse=f)
            nengo.Connection(ens.neurons, node[n_pre: n_pre+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_smooth)
        if L_u:
            node = LearningNode2(n_neurons, n_pre, pre_u_ens, k=3e-6)
            nengo.Connection(pre_u.neurons, node[0: n_pre], synapse=f)
            nengo.Connection(ens.neurons, node[n_pre: n_pre+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_smooth)
        if L_fb:
            node = LearningNode2(n_neurons, n_neurons, ens2_ens, conn_supv=pre_x_ens2, k=3e-6)
            nengo.Connection(ens2.neurons, node[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(ens2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
#             nengo.Connection(supv.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)


        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_v = nengo.Probe(ens.neurons, 'voltage', synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if np.any(w_x):
            for pre in range(n_pre):
                for post in range(n_neurons):
                    if L_u:
                        pre_base_ens.weights[pre, post] = w_x[pre, post]
                        pre_base_ens.netcons[pre, post].weight[0] = np.abs(w_x[pre, post])
                        pre_base_ens.netcons[pre, post].syn().e = 0 if w_x[pre, post] > 0 else -70
                    elif L_x or L_fd:
#                     if L_u or L_fd:
                        pre_x_ens.weights[pre, post] = w_x[pre, post]
                        pre_x_ens.netcons[pre, post].weight[0] = np.abs(w_x[pre, post])
                        pre_x_ens.netcons[pre, post].syn().e = 0 if w_x[pre, post] > 0 else -70
                    elif L_fb or supervised:
                        pre_x_ens2.weights[pre, post] = w_x[pre, post]
                        pre_x_ens2.netcons[pre, post].weight[0] = np.abs(w_x[pre, post])
                        pre_x_ens2.netcons[pre, post].syn().e = 0 if w_x[pre, post] > 0 else -70
        if np.any(w_u):
            for pre in range(n_pre):
                for post in range(n_neurons):
                    pre_u_ens.weights[pre, post] = w_u[pre, post]
                    pre_u_ens.netcons[pre, post].weight[0] = np.abs(w_u[pre, post])
                    pre_u_ens.netcons[pre, post].syn().e = 0 if w_u[pre, post] > 0 else -70
        if np.any(w_fb):
            if L_fb or supervised:
                for pre in range(n_neurons):
                    for post in range(n_neurons):
                        ens2_ens.weights[pre, post] = w_fb[pre, post]
                        ens2_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens2_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
            else:
                for pre in range(n_neurons):
                    for post in range(n_neurons):
                        ens_ens.weights[pre, post] = w_fb[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
        if np.any(e_x) and L_x:
            pre_x_ens.e = e_x
        if np.any(e_u) and L_u:
            pre_u_ens.e = e_u
        if np.any(e_fb) and L_fb:
            ens2_ens.e = e_fb

        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model)

    if L_x and hasattr(pre_x_ens, 'weights'):
        w_x = pre_x_ens.weights
        e_x = pre_x_ens.e
    if L_u and hasattr(pre_u_ens, 'weights'):
        w_u = pre_u_ens.weights
        e_u = pre_u_ens.e
    if L_fb and hasattr(ens2_ens, 'weights'):
        w_fb = ens2_ens.weights
        e_fb = ens2_ens.e

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        ens=sim.data[p_ens],
        ens2=sim.data[p_ens2] if L_fb or supervised else None,
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2] if L_fb or supervised else None,
        w_x=w_x,
        w_u=w_u,
        w_fb=w_fb,
        e_x=e_x if L_x else None,
        e_u=e_u if L_u else None,
        e_fb=e_fb if L_fb else None,
        supv_state=sim.data[p_supv_state] if L_fb or supervised else None,
        supv2_state=sim.data[p_supv2_state] if L_fb or supervised else None,
    )


def run(n_neurons=100, t=10, t_flat=3, t_test=10, n_train=20, n_test=10,
        reg=1e-1, penalty=0.0, df_evals=100, T=0.2, dt=0.001, neuron_type=LIF(),
        f=DoubleExp(1e-2, 2e-1), f_smooth=DoubleExp(1e-2, 2e-1),
        w_file="data/memory_w.npz", fd_file="data/memory_DurstewitzNeuron()_fd.npz", load_w_x=False, 
        load_w_u=False, load_w_fb=False, load_fd=False, supervised=False):
            
    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_u = None
    w_x = None
    w_fb = None
    DA = None

    if isinstance(neuron_type, DurstewitzNeuron):
        DA = neuron_type.DA
        if load_w_x:
            w_x = np.load(w_file)['w_x']
            e_x = np.load(w_file)['e_x']      
        else:
            print('optimizing encoders from pre_x into ens (white noise)')
            e_x = None
            w_x = None
            for nenc in range(n_train):
                print("encoding trial %s"%nenc)
                u = make_signal(t=t, t_flat=t_flat, f=f, normed='x', seed=nenc)
                t_sim = len(u)*dt - dt
                stim_func = lambda t: u[int(t/dt)]
                data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, e_x=e_x, w_x=w_x, L_x=True)
                w_x = data['w_x']
                e_x = data['e_x']
                np.savez('data/memory_DA%s_w.npz'%DA, w_x=w_x, e_x=e_x)
                a_ens = f_smooth.filt(data['ens'], dt=dt)
                a_supv = f_smooth.filt(data['supv'], dt=dt)
                for n in range(n_neurons):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
                    ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                    ax.set(ylabel='firing rate', ylim=((0, 40)))
                    plt.legend()
                    plt.savefig('plots/tuning/memory_x_DA%s_nenc_%s_activity_%s.pdf'%(DA, nenc, n))
                    plt.close('all')

        if load_w_u:
            w_u = np.load(w_file)['w_u']
            e_u = np.load(w_file)['e_u']
        else:
            print('optimizing encoders from pre_u into ens (white noise)')
            e_u = None
            w_u = None
            bases = np.linspace(-0.5, 0.5, n_train)
            for nenc in range(n_train):
                print("encoding trial %s"%nenc)
                u = make_signal(t=t, t_flat=t_flat, f=f, normed='x', seed=nenc)
                t_sim = len(u)*dt - dt
                stim_func = lambda t: u[int(t/dt)]
                stim_func_base = lambda t: bases[nenc]
                data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, e_u=e_u, w_u=w_u, stim_func_base=stim_func_base, L_u=True)
                w_u = data['w_u']
                e_u = data['e_u']
                np.savez('data/memory_DA%s_w.npz'%DA, w_x=w_x, e_x=e_x, w_u=w_u, e_u=e_u)
                a_ens = f_smooth.filt(data['ens'], dt=dt)
                a_supv = f_smooth.filt(data['supv'], dt=dt)
                for n in range(n_neurons):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
                    ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                    ax.set(ylabel='firing rate', ylim=((0, 40)))
                    plt.legend()
                    plt.savefig('plots/tuning/memory_u_DA%s_nenc_%s_activity_%s.pdf'%(DA, nenc, n))
                    plt.close('all')
    
    if load_fd:
        load = np.load(fd_file)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens (white-flat-white)')
        targets = np.zeros((1, 1))
        spikes = np.zeros((1, n_neurons))
        for ntrn in range(n_train):
            print('filter/decoder iteration %s'%ntrn)
            u = make_signal(t=t, t_flat=t_flat, f=f, normed='x', seed=ntrn, dt=dt)
            t_sim = len(u)*dt - dt
            stim_func = lambda t: u[int(t/dt)]
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, L_fd=True)
            targets = np.append(targets, data['x'], axis=0)
            spikes = np.append(spikes, data['ens'], axis=0)

        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(targets, spikes, f, dt=dt, penalty=penalty, reg=reg, df_evals=df_evals, name='flat_%s'%neuron_type)
        if DA:
            np.savez('data/memory_%s_DA%s_fd.npz'%(neuron_type, DA), d_ens=d_ens, taus_ens=taus_ens)            
        else:
            np.savez('data/memory_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        if DA:
            plt.savefig("plots/memory_%s_DA%s_filters.pdf"%(neuron_type, DA))
        else:
            plt.savefig("plots/memory_%s_filters.pdf"%neuron_type)
        
        a_ens = f_ens.filt(spikes, dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        x = f.filt(targets, dt=dt)
        rmse_ens = rmse(xhat_ens, x)
        fig, ax = plt.subplots()
        ax.plot(x, linestyle="--", label='x')
        ax.plot(xhat_ens, label='ens, rmse=%.3f' %rmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train_fb")
        plt.legend(loc='upper right')
        if DA:
            plt.savefig("plots/memory_%s_DA%s_train_fb.pdf"%(neuron_type, DA))
        else:
            plt.savefig("plots/memory_%s_train_fb.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w_fb:
            w_fb = np.load(w_file)['w_fb']
            e_fb = np.load(w_file)['e_fb']
        else:
            print('optimizing encoders from ens2 into ens (white noise)')
            e_fb = None
            w_fb = None
            for nenc in range(n_train):
                print("encoding trial %s"%nenc)
                u = make_signal(t=t, t_flat=t_flat, f=f, normed='x', seed=nenc)
                t_sim = len(u)*dt - dt
                stim_func = lambda t: u[int(t/dt)]
                data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, e_fb=e_fb, w_fb=w_fb, L_fb=True)
                w_fb = data['w_fb']
                e_fb = data['e_fb']
                np.savez('data/memory_DA%s_w.npz'%DA, w_x=w_x, e_x=e_x, w_u=w_u, e_u=e_u, w_fb=w_fb, e_fb=e_fb)
                
                a_ens = f_smooth.filt(data['ens'], dt=dt)
                a_ens2 = f_smooth.filt(data['ens2'], dt=dt)
                a_supv = f_smooth.filt(data['supv'], dt=dt)
#                 a_supv2 = f_smooth.filt(data['supv2'], dt=dt)
                for n in range(n_neurons):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
#                     ax.plot(data['times'], a_supv2[:,n], alpha=0.5, label='supv2')
                    ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                    ax.plot(data['times'], a_ens2[:,n], alpha=0.5, label='ens2')
                    ax.set(ylabel='firing rate', ylim=((0, 40)))
                    plt.legend()
                    plt.savefig('plots/tuning/memory_fb_DA%s_nenc_%s_activity_%s.pdf'%(DA, nenc, n))
                    plt.close('all')

    errors_flat = np.zeros((n_test))
    errors_final = np.zeros((n_test))
    errors_abs = np.zeros((n_test, int(t_test/dt)))
    for test in range(n_test):
        print('test %s' %test)
        u = make_signal(t=6, t_flat=t_test, f=f, normed='x', seed=test, dt=dt, test=True)
        t_sim = len(u)*dt - dt
        stim_func = lambda t: u[int(t/dt)]
        if supervised:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, w_fb=w_fb, supervised=True)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            a_ens2 = f_ens.filt(data['ens2'], dt=dt)
            x = f.filt(f.filt(data['x'], dt=dt), dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            xhat_ens2 = np.dot(a_ens2, d_ens)
            xhat_supv = data['supv_state']
            xhat_supv2 = data['supv2_state']
            error_ens = rmse(xhat_ens, x)
            error_ens2 = rmse(xhat_ens2, x)
            error_supv = rmse(xhat_supv, x)
            error_supv2 = rmse(xhat_supv2, x)

            fig, ax = plt.subplots()
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, error=%.3f' %error_ens)
            ax.plot(data['times'], xhat_ens2, label='ens2, error=%.3f' %error_ens2)
            ax.plot(data['times'], xhat_supv, label='supv, error=%.3f' %error_supv)
            ax.plot(data['times'], xhat_supv2, label='supv2, error=%.3f' %error_supv2)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised test")
            plt.legend(loc='upper right')
            if DA:
                plt.savefig("plots/memory_%s_DA%s_supervised_test_%s.pdf"%(neuron_type, DA, test))
            else:
                plt.savefig("plots/memory_%s_supervised_test_%s.pdf"%(neuron_type, test))
            plt.close('all')

        else:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_sim, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_u=w_u, w_x=None, w_fb=w_fb)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            u = f.filt(f.filt(T*data['u'], dt=dt), dt=dt)
            x = f.filt(data['x'], dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            error_flat = rmse(xhat_ens[-int(t_test/dt):], x[-int(t_test/dt):])
            error_final = rmse(xhat_ens[-1], x[-1])
            error_abs = np.abs(xhat_ens[-int(t_test/dt):,0] - x[-int(t_test/dt):,0])
            errors_flat[test] = error_flat
            errors_final[test] = error_final
            errors_abs[test] = error_abs

            if test > 10:
                continue
            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label='u')
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='error_flat=%.3f, error_final=%.3f' %(error_flat, error_final))
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
            plt.legend(loc='upper right')
            if DA:
                plt.savefig("plots/memory_%s_DA%s_test_%s.pdf"%(DA, neuron_type, test))
            else:
                plt.savefig("plots/memory_%s_test_%s.pdf"%(neuron_type, test))
            plt.close('all')
            
            fig, ax = plt.subplots()
            ax.plot(data['times'][-int(t_test/dt):], error_abs, linestyle="--", label='error')
            ax.set(xlabel='time (s)', ylabel=r'$|\mathbf{x} - \mathbf{\hat{x}}|$', title="test")
#             plt.legend(loc='upper right')
            if DA:
                plt.savefig("plots/memory_abs_%s_DA%a_test_%s.pdf"%(DA, neuron_type, test))
            else:
                plt.savefig("plots/memory_abs_%s_test_%s.pdf"%(neuron_type, test))
            plt.close('all')
            
    mean_flat = np.mean(errors_flat)
    mean_final = np.mean(errors_final)
    CI_flat = sns.utils.ci(errors_flat)
    CI_final = sns.utils.ci(errors_final)
    if DA:
        np.savez("data/%s_DA%s_errors_abs.npz"%(neuron_type, DA), errors_abs=errors_abs)
    else:
        np.savez("data/%s_errors_abs.npz"%neuron_type, errors_abs=errors_abs)
    
#     errors = np.vstack((errors_flat, errors_final))
#     names =  ['flat', 'final']
#     fig, ax = plt.subplots()
#     sns.barplot(data=errors.T)
#     ax.set(ylabel='RMSE')
#     plt.xticks(np.arange(len(names)), tuple(names), rotation=0)
#     plt.savefig("plots/memory_%s_errors.pdf"%neuron_type)
    
    
    dfs = []
    columns = ("nAvg", "time", "error")
    for a in range(errors_abs.shape[0]):
        for t in range(errors_abs.shape[1]):
            dfs.append(pd.DataFrame([[a, dt*t, errors_abs[a, t]]], columns=columns))
    df = pd.concat([df for df in dfs], ignore_index=True)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="time", y="error", ax=ax)
    ax.set(xlabel='time (s)', ylabel=r'$|\mathbf{x} - \mathbf{\hat{x}}|$')
    fig.tight_layout()
    if DA:
        fig.savefig("plots/memory_abs_%s_DA%s_all.pdf"%(neuron_type, DA))
    else:
        fig.savefig("plots/memory_abs_%s_all.pdf"%(neuron_type))
    plt.close('all')
                    
    print("errors: mean flat=%.3f, mean final=%.3f"%(np.mean(errors_flat), np.mean(errors_final)))
    return errors_flat, errors_final


# eflat_lif, efinal_lif = run(neuron_type=LIF(), load_fd=True, fd_file="data/memory_LIF()_fd.npz", n_test=100)
# eflat_alif, efinal_alif = run(neuron_type=AdaptiveLIFT())
# eflat_wilson, efinal_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)

# eflat_durstewitz, efinal_durstewitz = run(neuron_type=DurstewitzNeuron(DA=1.0),
#      load_w_u=True, load_w_x=True, load_fd=True, load_w_fb=True)

eflat_durstewitz, efinal_durstewitz = run(neuron_type=DurstewitzNeuron(DA=1.0), n_test=100,
    load_w_u=True, load_w_x=True, load_fd=True, load_w_fb=True,
    w_file="data/memory_DA1.0_w.npz", fd_file="data/memory_DurstewitzNeuron()_DA1.0_fd.npz")
