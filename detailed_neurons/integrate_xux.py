import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import norms, d_opt, df_opt, LearningNode2
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def make_normed_flipped(value=1.0, t=1.0, dt=0.001, N=1, f=Lowpass(0.01), normed='x', seed=0):
    print('Creating input signal from concatenated, normalized, flipped white noise')
    stim_length = int(t*N/dt)+1
    u_list = np.zeros((stim_length, 1))
    for n in range(N):
        stim_func = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.5, seed=seed+n)
        with nengo.Network() as model:
            model.t_half = t/2
            def flip(t, x):
                return x if t<model.t_half else -1.0*x
            u_raw = nengo.Node(stim_func)
            u = nengo.Node(output=flip, size_in=1)   
            nengo.Connection(u_raw, u, synapse=None)
            p_u = nengo.Probe(u, synapse=None)
            p_x = nengo.Probe(u, synapse=1/s)
        with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
            sim.run(t, progress_bar=False)
        u = f.filt(sim.data[p_u], dt=dt)
        x = f.filt(sim.data[p_x], dt=dt)
        if normed=='u':  
            norm = value / np.max(np.abs(u))
        elif normed=='x':
            norm = value / np.max(np.abs(x))
        u_list[n*int(t/dt): (n+1)*int(t/dt)] = sim.data[p_u] * norm
    stim_func = lambda t: u_list[int(t/dt)]
    return stim_func

def go(d_ens, f_ens, n_neurons=100, n_pre=300, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.6), seed=0, dt=0.001, T=0.2, neuron_type=LIF(), f=Lowpass(0.2), f_smooth=DoubleExp(1e-3, 2e-1), stim_func=lambda t: np.sin(t), w_x=None, e_x=None, w_u=None, e_u=None, w_fb=None, e_fb=None, L_x=False, L_u=False, L_fd=False, L_fb=False, supervised=False):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(n_pre, 1, radius=3, seed=seed)
        pre_x = nengo.Ensemble(n_pre, 1, radius=1.5, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)

        # Connections
        u_pre = nengo.Connection(u, pre_u, synapse=None, seed=seed)
        x_pre = nengo.Connection(x, pre_x, synapse=None, seed=seed)
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        if not L_x:
            pre_u_ens = nengo.Connection(pre_u, ens, synapse=f, transform=T, seed=seed)
            pre_u_supv = nengo.Connection(pre_u, supv, synapse=f, transform=T, seed=seed)
        if L_x or L_u or L_fd:
            pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
            pre_x_supv = nengo.Connection(pre_x, supv, synapse=f, seed=seed)

        if L_fb or supervised:
            u_pre.synapse = f
            x_pre.synapse = None
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            pre_x_ens2 = nengo.Connection(pre_x, ens2, synapse=f, seed=seed)
            pre_x_supv2 = nengo.Connection(pre_x, supv2, synapse=f, seed=seed)
            ens2_ens = nengo.Connection(ens2, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            supv2_supv = nengo.Connection(supv2, supv, synapse=f, seed=seed)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
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
            node = LearningNode2(n_neurons, n_neurons, ens2_ens, k=3e-6)
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

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_x):
            for pre in range(n_pre):
                for post in range(n_neurons):
                    if L_u or L_fd:
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
            if supervised:
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
        sim.run(t)
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
        ens2=sim.data[p_ens2] if L_fb else None,
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2] if L_fb else None,
        w_x=w_x,
        w_u=w_u,
        w_fb=w_fb,
        e_x=e_x if L_x else None,
        e_u=e_u if L_u else None,
        e_fb=e_fb if L_fb else None,
        v=sim.data[p_v],
    )


def run(n_neurons=30, t=10, t_test=10, dt=0.001, n_trains=30, n_encodes=30, n_tests=30, neuron_type=LIF(), f=DoubleExp(1e-3, 2e-1), load_w=None, load_fd=None, load_w_x=False, load_w_u=False, load_w_fb=False, reg=1e-1, penalty=0.0, T=0.2, supervised=False):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    f_smooth = DoubleExp(1e-3, 2e-1)
    w_x = None
    e_x = None
    w_u = None
    e_u = None
    w_fb = None
    e_fb = None
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w_x:
            w_x = np.load(load_w)['w_x']
            e_x = np.load(load_w)['e_x']      
        else:
            print('optimizing encoders from pre_x into ens')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_encodes, f=f, normed='x')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, e_x=e_x, L_x=True)
            w_x = data['w_x']
            e_x = data['e_x']
            np.savez('data/integrate_xux_w.npz', w_x=w_x, e_x=e_x)

            fig, ax = plt.subplots()
            sns.distplot(np.ravel(w_x), ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_xux_%s_w_x.pdf"%neuron_type)

            a_ens = f_smooth.filt(data['ens'], dt=dt)
            a_supv = f_smooth.filt(data['supv'], dt=dt)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_xux_pre_x_ens_activity_%s.pdf'%n)
                plt.close('all')

            for n in range(n_neurons):
                volt = data['v'][:,n]
                fig, ax, = plt.subplots(1, 1)
                ax.plot(data['times'], data['v'][:,n])
                ax.set(xlabel='time', ylabel='voltage')
                plt.savefig('plots/tuning/integrate_xux_pre_x_voltage_%s.pdf'%n)
                plt.close('all')

        if load_w_u:
            w_u = np.load(load_w)['w_u']
            e_u = np.load(load_w)['e_u']
        else:
            print('optimizing encoders from pre_u into ens')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_encodes, f=f, normed='x')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, e_u=e_u, L_u=True)
            w_u = data['w_u']
            e_u = data['e_u']
            np.savez('data/integrate_xux_w.npz', w_x=w_x, e_x=e_x, w_u=w_u, e_u=e_u)

            fig, ax = plt.subplots()
            sns.distplot(np.ravel(w_u), ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_xux_%s_w_u.pdf"%neuron_type)

            a_ens = f_smooth.filt(data['ens'], dt=dt)
            a_supv = f_smooth.filt(data['supv'], dt=dt)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_xux_pre_u_ens_activity_%s.pdf'%n)
                plt.close('all')
                
            for n in range(n_neurons):
                volt = data['v'][:,n]
                fig, ax, = plt.subplots(1, 1)
                ax.plot(data['times'], data['v'][:,n])
                ax.set(xlabel='time', ylabel='voltage')
                plt.savefig('plots/tuning/integrate_xux_pre_u_voltage_%s.pdf'%n)
                plt.close('all')
    
    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, L_fd=True)            
        target = data['x']
        spikes = data['ens']
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(target, spikes, f, dt=dt, penalty=penalty, reg=reg, df_evals=200, name='integrate_%s'%neuron_type)
        np.savez('data/integrate_xux_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/integrate_xux_%s_filters.pdf"%neuron_type)
        
        a_ens = f_ens.filt(data['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        u = f.filt(T*data['u'], dt=dt)
        x = f.filt(data['x'], dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=x)
        fig, ax = plt.subplots()
#         ax.plot(data['times'], u, linestyle="--", label='u')
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_xux_%s_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w_fb:
            w_fb = np.load(load_w)['w_fb']
            e_fb = np.load(load_w)['e_fb']
        else:
            print('optimizing encoders from ens2 into ens')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_encodes, f=f, normed='x')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, e_fb=e_fb, L_fb=True)
            w_fb = data['w_fb']
            e_fb = data['e_fb']
            np.savez('data/integrate_xux_w.npz', w_x=w_x, e_x=e_x, w_u=w_u, e_u=e_u, w_fb=w_fb, e_fb=e_fb)

            fig, ax = plt.subplots()
            sns.distplot(np.ravel(w_fb), ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_xux_%s_w_fb.pdf"%neuron_type)

            a_ens = f_smooth.filt(data['ens'], dt=dt)
            a_ens2 = f_smooth.filt(data['ens2'], dt=dt)
            a_supv = f_smooth.filt(data['supv'], dt=dt)
#             a_supv2 = f_smooth.filt(data['supv2'], dt=dt)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:20000,n], alpha=0.5, label='supv')
    #                 ax.plot(data['times'][:20000], a_supv2[:20000,n], alpha=0.5, label='supv2')
                ax.plot(data['times'][:20000], a_ens[:20000,n], alpha=0.5, label='ens')
                ax.plot(data['times'][:20000], a_ens2[:20000,n], alpha=0.5, label='ens2')
                ax.set(ylabel='firing rate', ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[-20000:,n], alpha=0.5, label='supv')
    #                 ax2.plot(data['times'][-20000:], a_supv2[-20000:,n], alpha=0.5, label='supv2')
                ax2.plot(data['times'][-20000:], a_ens[-20000:,n], alpha=0.5, label='ens')
                ax2.plot(data['times'][-20000:], a_ens2[-20000:,n], alpha=0.5, label='ens2')
                ax2.set(xlabel='time', ylabel='firing rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_xux_ens2_ens_activity_%s.pdf'%n)
                plt.close('all')

            for n in range(n_neurons):
                fig, ax, = plt.subplots(1, 1)
                ax.plot(data['times'], data['v'][:,n])
                ax.set(xlabel='time', ylabel='voltage')
                plt.savefig('plots/tuning/integrate_xux_ens2_ens_voltage_%s.pdf'%n)
                plt.close('all')

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t_test, dt=dt, N=1, normed='x', f=f, seed=100+test)
        if supervised:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_x=w_x, w_u=w_u, w_fb=w_fb, supervised=True)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            x = f.filt(f.filt(data['x'], dt=dt), dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)

            fig, ax = plt.subplots()
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised test")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_xux_%s_supervised_test_%s.pdf"%(neuron_type, test))
            plt.close('all')
            
            for n in range(n_neurons):
                fig, ax, = plt.subplots(1, 1)
                ax.plot(data['times'], data['v'][:,n])
                ax.set(xlabel='time', ylabel='voltage')
                plt.savefig('plots/tuning/integrate_xux_supervised_%s_voltage_%s.pdf'%(test, n))
                plt.close('all')

        else:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T=T, w_u=w_u, w_x=None, w_fb=w_fb)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            u = f.filt(f.filt(T*data['u'], dt=dt), dt=dt)
            x = f.filt(data['x'], dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)
            nrmses_ens[test] = nrmse_ens

            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label='u')
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_xux_%s_test_%s.pdf"%(neuron_type, test))
            plt.close('all')
            
#             a_ens = f_smooth.filt(data['ens'], dt=dt)
#             a_supv = f_smooth.filt(data['supv'], dt=dt)
#             for n in range(n_neurons):
#                 fig, (ax, ax2) = plt.subplots(2, 1)
#                 ax.plot(data['times'][:20000], a_supv[:20000,n], alpha=0.5, label='supv')
#                 ax.plot(data['times'][:20000], a_ens[:20000,n], alpha=0.5, label='ens')
#                 ax.set(ylabel='firing rate', ylim=((0, 40)))
#                 ax2.plot(data['times'][-20000:], a_supv[-20000:,n], alpha=0.5, label='supv')
#                 ax2.plot(data['times'][-20000:], a_ens[-20000:,n], alpha=0.5, label='ens')
#                 ax2.set(xlabel='time', ylabel='firing rate', ylim=((0, 40)))
#                 plt.legend()
#                 plt.savefig('plots/tuning/integrate_xux_test_%s_activity_%s.pdf'%(test, n))
#                 plt.close('all')
                
#             for n in range(n_neurons):
#                 fig, ax, = plt.subplots(1, 1)
#                 ax.plot(data['times'], data['v'][:,n])
#                 ax.set(xlabel='time', ylabel='voltage')
#                 plt.savefig('plots/tuning/integrate_xux_test_%s_voltage_%s.pdf'%(test, n))
#                 plt.close('all')
                
    if not supervised:
        mean_ens = np.mean(nrmses_ens)
        CI_ens = sns.utils.ci(nrmses_ens)

        fig, ax = plt.subplots()
        sns.barplot(data=nrmses_ens)
        ax.set(ylabel='NRMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens, CI_ens[0], CI_ens[1]))
        plt.xticks()
        plt.savefig("plots/integrate_xux_%s_nrmse.pdf"%neuron_type)

        print('nrmses: ', nrmses_ens)
        print('means: ', mean_ens)
        print('confidence intervals: ', CI_ens)
        np.savez('data/integrate_xux_%s_results.npz'%neuron_type, nrmses_ens=nrmses_ens)
        return nrmses_ens

nrmses_lif = run(neuron_type=LIF())
nrmses_alif = run(neuron_type=AdaptiveLIFT())
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)
nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0))

# nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), n_neurons=30, n_encodes=10, n_tests=3, supervised=True, load_w_u=True, load_w_x=True, load_w_fb=True, load_w="data/integrate_xux_w.npz", load_fd="data/integrate_xux_DurstewitzNeuron()_fd.npz")
# nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), n_neurons=50, n_tests=1, t_test=2, supervised=False, load_w_u=True, load_w_x=True, load_w_fb=True, load_w="data/integrate_xux_w.npz", load_fd="data/integrate_xux_DurstewitzNeuron()_fd.npz")

# nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
# nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sns.barplot(data=nrmses.T)
# ax.set(ylabel='NRMSE')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/integrate_nrmses.pdf")
# 