import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import norms, df_opt, LearningNode
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


def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, neuron_type=LIF(), f=DoubleExp(1e-3, 1e-1), f_smooth=DoubleExp(1e-3, 2e-1), T_ff=0.1, stim_func=lambda t: np.sin(t), w_ff=None, w_ff2=None, w_fb=None, w_fb2=None, learn_fd=False, learn_ff=False, learn_ff2=False, learn_fb=False, learn_fb2=False, supervised=False):

    with nengo.Network(seed=seed) as model:
                    
        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(300, 1, radius=4, seed=seed)
        pre_x = nengo.Ensemble(300, 1, radius=2, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=neuron_type)

        # Connections
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        nengo.Connection(u, pre_u, synapse=None, seed=seed)
        nengo.Connection(x, pre_x, synapse=None, seed=seed)
        pre_u_ens = nengo.Connection(pre_u, ens, synapse=f, transform=T_ff, seed=seed)
        
        if learn_ff:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=LIF())
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            node = LearningNode(n_neurons, pre_u.n_neurons, 1, pre_u_ens, k=3e-6)
            nengo.Connection(pre_u.neurons, node[0: pre_u.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre_u.n_neurons: pre_u.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre_u.n_neurons+n_neurons: pre_u.n_neurons+2*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
        elif learn_ff2:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=LIF())
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            nengo.Connection(pre_x, supv, synapse=f, seed=seed)        
            pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
            node = LearningNode(n_neurons, pre_x.n_neurons, 1, pre_x_ens, k=3e-6)
            nengo.Connection(pre_x.neurons, node[0: pre_x.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre_x.n_neurons: pre_x.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre_x.n_neurons+n_neurons: pre_x.n_neurons+2*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
        elif learn_fd:
            pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
        elif learn_fb:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=neuron_type)
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            nengo.Connection(supv, supv2, synapse=f, seed=seed)
            ens_ens2 = nengo.Connection(ens, ens2, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            node = LearningNode(n_neurons, n_neurons, 1, ens_ens2, k=3e-6)
            nengo.Connection(ens.neurons, node[0: n_neurons], synapse=f_ens)
            nengo.Connection(ens2.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        elif learn_fb2:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=neuron_type)
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            nengo.Connection(pre_x, supv, synapse=f, seed=seed)  
            nengo.Connection(supv, supv2, synapse=f, seed=seed)
            pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
            ens_ens2 = nengo.Connection(ens, ens2, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            ens_ens2_two = nengo.Connection(ens, ens2, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            node = LearningNode(n_neurons, n_neurons, 1, ens_ens2_two, k=3e-6)
            nengo.Connection(ens.neurons, node[0: n_neurons], synapse=f_ens)
            nengo.Connection(ens2.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        elif supervised:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=nengo.LIF())
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=neuron_type)
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            nengo.Connection(supv, supv2, synapse=f, seed=seed)
            if np.any(w_ff2):
                nengo.Connection(pre_x, supv, synapse=f, seed=seed)  
                pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
            ens_ens2 = nengo.Connection(ens, ens2, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            ens_ens2_two = nengo.Connection(ens, ens2, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        else:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)
            ens_ens_two = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(0*d_ens), seed=seed)


        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if np.any(w_ff):
            for pre in range(pre_u.n_neurons):
                for post in range(n_neurons):
                    pre_u_ens.weights[pre, post] = w_ff[pre, post]
                    pre_u_ens.netcons[pre, post].weight[0] = np.abs(w_ff[pre, post])
                    pre_u_ens.netcons[pre, post].syn().e = 0 if w_ff[pre, post] > 0 else -70
        if np.any(w_ff2):
            for pre in range(pre_x.n_neurons):
                for post in range(n_neurons):
                    pre_x_ens.weights[pre, post] = w_ff2[pre, post]
                    pre_x_ens.netcons[pre, post].weight[0] = np.abs(w_ff2[pre, post])
                    pre_x_ens.netcons[pre, post].syn().e = 0 if w_ff2[pre, post] > 0 else -70
        if np.any(w_fb):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    if supervised or learn_fb2:
                        ens_ens2.weights[pre, post] = w_fb[pre, post]
                        ens_ens2.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens_ens2.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
                    else:
                        ens_ens.weights[pre, post] = w_fb[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
        if np.any(w_fb2):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    if supervised:
                        ens_ens2_two.weights[pre, post] = w_fb2[pre, post]
                        ens_ens2_two.netcons[pre, post].weight[0] = np.abs(w_fb2[pre, post])
                        ens_ens2_two.netcons[pre, post].syn().e = 0 if w_fb2[pre, post] > 0 else -70
                    else:
                        ens_ens_two.weights[pre, post] = w_fb2[pre, post]
                        ens_ens_two.netcons[pre, post].weight[0] = np.abs(w_fb2[pre, post])
                        ens_ens_two.netcons[pre, post].syn().e = 0 if w_fb2[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model)

    if learn_ff and hasattr(pre_u_ens, 'weights'):
        w_ff = pre_u_ens.weights
    if learn_ff2 and hasattr(pre_x_ens, 'weights'):
        w_ff2 = pre_x_ens.weights
    if learn_fb and hasattr(ens_ens2, 'weights'):
        w_fb = ens_ens2.weights
    if learn_fb2 and hasattr(ens_ens2, 'weights'):
        w_fb2 = ens_ens2.weights
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        ens=sim.data[p_ens],
        supv=sim.data[p_supv] if learn_ff or learn_ff2 or learn_fb or learn_fb2 or supervised else None,
        ens2=sim.data[p_ens2] if learn_fb or learn_fb2 or supervised else None,
        supv2=sim.data[p_supv2] if learn_fb or learn_fb2 or supervised else None,
        enc=sim.data[ens].encoders,
        w_ff=w_ff,
        w_ff2=w_ff2,
        w_fb=w_fb,
        w_fb2=w_fb2
    )


def run(n_neurons=100, t=10, t_test=10, neuron_type=LIF(), f=DoubleExp(1e-3, 2e-1), dt=0.001, n_trains=10, n_encodes=10, n_tests=30, reg=0, penalty=0.1, T_ff=0.2, load_fd=False, load_w=False, f_smooth=DoubleExp(1e-3, 2e-1), supervised=False):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_ff = None
    w_ff2 = None
    w_fb = None
    w_fb2 = None
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ff = np.load(load_w)['w_ff']
            w_ff2 = np.load(load_w)['w_ff2']
        else:
            print('optimizing encoders from pre_u into ens (T_ff=%.1f)'%T_ff)
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, learn_ff=True)
            w_ff = data['w_ff']
            fig, ax = plt.subplots()
            sns.distplot(w_ff.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_ff.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_ff=w_ff)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_supv = f_smooth.filt(data['supv'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens_ff_activity_%s.pdf'%n)
                plt.close('all')
                
            print('optimizing encoders from pre_x into ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, learn_ff2=True)
            w_ff2 = data['w_ff2']
            fig, ax = plt.subplots()
            sns.distplot(w_ff2.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_ff2.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_ff=w_ff, w_ff2=w_ff2)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_supv = f_smooth.filt(data['supv'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens_ff2_activity_%s.pdf'%n)
                plt.close('all')

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        # filters/decoders must account for both a small ff input (tau*u) and a large fb input (x)
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x')
        data1 = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, learn_fd=False)
        data2 = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, w_ff2=w_ff2, learn_fd=True)
        target = np.vstack((f.filt(T_ff*data1['u'], dt=dt), data2['x']))
        spikes = np.vstack((data1['ens'], data2['ens']))
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(target, spikes, f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)
        np.savez('data/integrate_x_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/integrate_x_%s_filters_ens.pdf"%neuron_type)

        a_ens = f_ens.filt(data1['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        target = f.filt(f.filt(T_ff*data1['u'], dt=dt), dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=target)
        fig, ax = plt.subplots()
        ax.plot(data1['times'], target, linestyle="--", label='target')
        ax.plot(data1['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_x_%s_ens_fd_train1.pdf"%neuron_type)

        a_ens = f_ens.filt(data2['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        target = f.filt(data2['x'], dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=target)
        fig, ax = plt.subplots()
        ax.plot(data2['times'], target, linestyle="--", label='target')
        ax.plot(data2['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_x_%s_ens_fd_train2.pdf"%neuron_type)
        
    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ff = np.load(load_w)['w_ff']
            w_ff2 = np.load(load_w)['w_ff2']
            w_fb = np.load(load_w)['w_fb']
            w_fb2 = np.load(load_w)['w_fb2']
        else:
            print('optimizing encoders from ens into ens2')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x')

            data1 = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, learn_fb=True, w_ff=w_ff)
            w_fb = data1['w_fb']
            fig, ax = plt.subplots()
            sns.distplot(w_fb.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_fb.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_ff=w_ff, w_ff2=w_ff2, w_fb=w_fb)
            a_ens = f_smooth.filt(data1['ens2'], dt=0.001)
            a_supv = f_smooth.filt(data1['supv2'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data1['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data1['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data1['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data1['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens_fb_activity_%s.pdf'%n)
                plt.close('all')

            data2 = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, learn_fb2=True, w_ff=w_ff, w_ff2=w_ff2, w_fb=w_fb)
            w_fb2 = data2['w_fb2']
            fig, ax = plt.subplots()
            sns.distplot(w_fb.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_fb2.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_ff=w_ff, w_ff2=w_ff2, w_fb=w_fb, w_fb2=w_fb2)
            a_ens = f_smooth.filt(data2['ens2'], dt=0.001)
            a_supv = f_smooth.filt(data2['supv2'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data2['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data2['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data2['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data2['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens_fb2_activity_%s.pdf'%n)
                plt.close('all')

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t_test, dt=dt, N=1, normed='x', f=f, seed=100+test)
#         stim_func = lambda t: 1*(1<t<2) + 2*(3<t<4) + 3*(5<t<6)
        if supervised:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, w_fb=w_fb, w_fb2=w_fb2, supervised=True)

            a_ens = f_ens.filt(data['ens'], dt=dt)
            a_ens2 = f_ens.filt(data['ens2'], dt=dt)
            u = f.filt(f.filt(T_ff*data['u'], dt=dt), dt=dt)
            x = f.filt(data['x'], dt=dt)
            u2 = f.filt(u, dt=dt)
            x2 = f.filt(x, dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            xhat_ens2 = np.dot(a_ens2, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=u)
            nrmse_ens2 = nrmse(xhat_ens2, target=u2)

            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label=r"$\tau * u$")
            ax.plot(data['times'], x, linestyle="--", label=r"$x$")
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised (u and x into ens)")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_x_%s_supv_utest_%s.pdf"%(neuron_type, test))

            fig, ax = plt.subplots()
            ax.plot(data['times'], u2, linestyle="--", label='u2')
            ax.plot(data['times'], x2, linestyle="--", label='x2')
            ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised (ens into ens2)")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_x_%s_supv2_utest_%s.pdf"%(neuron_type, test))
        
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, w_ff2=w_ff2, w_fb=w_fb, w_fb2=w_fb2, supervised=True)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            a_ens2 = f_ens.filt(data['ens2'], dt=dt)
            u = f.filt(f.filt(T_ff*data['u'], dt=dt), dt=dt)
            x = f.filt(data['x'], dt=dt)
            u2 = f.filt(u, dt=dt)
            x2 = f.filt(x, dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            xhat_ens2 = np.dot(a_ens2, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)
            nrmse_ens2 = nrmse(xhat_ens2, target=x2)

            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label=r"$\tau * u$")
            ax.plot(data['times'], x, linestyle="--", label=r"$x$")
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised (u and x into ens)")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_x_%s_supv_uxtest_%s.pdf"%(neuron_type, test))

            fig, ax = plt.subplots()
            ax.plot(data['times'], u2, linestyle="--", label='u2')
            ax.plot(data['times'], x2, linestyle="--", label='x2')
            ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised (ens into ens2)")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_x_%s_supv2_uxtest_%s.pdf"%(neuron_type, test))
            
        else:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, w_ff=w_ff, w_fb=w_fb, w_fb2=w_fb2)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            u = f.filt(T_ff*data['u'], dt=dt)
            x = f.filt(data['x'], dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)
            nrmses_ens[test] = nrmse_ens

            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label=r"$\tau * u$")
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="unsupervised")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_x_%s_test_%s.pdf"%(neuron_type, test))

#     mean_ens = np.mean(nrmses_ens)
#     CI_ens = sns.utils.ci(nrmses_ens)

#     fig, ax = plt.subplots()
#     sns.barplot(data=nrmses_ens)
#     ax.set(ylabel='NRMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens, CI_ens[0], CI_ens[1]))
#     plt.xticks()
#     plt.savefig("plots/integrate_x_%s_nrmse.pdf"%neuron_type)

#     print('nrmses: ', nrmses_ens)
#     print('means: ', mean_ens)
#     print('confidence intervals: ', CI_ens)
#     np.savez('data/integrate_x_%s_results.npz'%neuron_type, nrmses_ens=nrmses_ens)
#     return nrmses_ens

# nrmses_lif = run(neuron_type=LIF())#, load_fd="data/integrate_x_LIF()_fd.npz", supervised=True, n_tests=1)
# nrmses_alif = run(neuron_type=AdaptiveLIFT())#, load_fd="data/integrate_x_AdaptiveLIFT()_fd.npz")
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)#, load_fd="data/integrate_x_WilsonEuler()_fd.npz")
nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), load_w="data/integrate_w.npz", load_fd="data/integrate_x_DurstewitzNeuron()_fd.npz", supervised=False, n_tests=1)