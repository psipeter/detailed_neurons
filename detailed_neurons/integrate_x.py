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


def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, neuron_type=LIF(), f=DoubleExp(1e-3, 1e-1), f_smooth=DoubleExp(1e-3, 2e-1), T_ff=0.1, stim_func=lambda t: np.sin(t), w_ff=None, w_fb=None, learn_fd=False, learn_ff=False, learn_fb=False):

    with nengo.Network(seed=seed) as model:
                    
        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(300, 1, radius=3, seed=seed)
        pre_x = nengo.Ensemble(300, 1, radius=3, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=neuron_type)

        # Connections
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        nengo.Connection(u, pre_u, synapse=None, seed=seed)
        nengo.Connection(x, pre_x, synapse=None, seed=seed)
        pre_u_ens = nengo.Connection(pre_u, ens, synapse=f, transform=T_ff, seed=seed)
#         if learn_fd:
#             pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
#         else:
#             ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)

        if learn_fd or learn_fb:
             pre_x_ens = nengo.Connection(pre_x, ens, synapse=f, seed=seed)
        if learn_ff:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=LIF())
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            node = LearningNode(n_neurons, pre_u.n_neurons, 1, pre_u_ens, k=3e-6)
            nengo.Connection(pre_u.neurons, node[0: pre_u.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre_u.n_neurons: pre_u.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre_u.n_neurons+n_neurons: pre_u.n_neurons+2*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
        if learn_fb:
            supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, seed=seed, neuron_type=LIF())
            nengo.Connection(pre_u, supv, synapse=f, transform=T_ff, seed=seed)        
            nengo.Connection(pre_x, supv, synapse=f, seed=seed)        
            node = LearningNode(n_neurons, pre_x.n_neurons, 1, pre_x_ens, k=3e-6)
            nengo.Connection(pre_x.neurons, node[0: pre_x.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre_x.n_neurons: pre_x.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre_x.n_neurons+n_neurons: pre_x.n_neurons+2*n_neurons], synapse=f_smooth)
            p_supv = nengo.Probe(supv.neurons, synapse=None)
        if not learn_fd and not learn_ff and not learn_fb:
             ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)

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
        if np.any(w_fb):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    ens_ens.weights[pre, post] = w_fb[pre, post]
                    ens_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                    ens_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model)

    if learn_ff and hasattr(pre_u_ens, 'weights'):
        w_ff = pre_u_ens.weights
    if learn_fb and hasattr(pre_x_ens, 'weights'):
        w_fb = pre_x_ens.weights
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        ens=sim.data[p_ens],
        supv=sim.data[p_supv] if learn_ff or learn_fb else None,
        enc=sim.data[ens].encoders,
        w_ff=w_ff,
        w_fb=w_fb
    )


def run(n_neurons=100, t=10, t_test=10, neuron_type=LIF(), f=DoubleExp(1e-3, 1e-1), dt=0.001, n_trains=10, n_encodes=10, n_tests=10, reg=0, penalty=0.1, T_ff=0.1, load_fd=False):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_ff = None
    w_fb = None
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ff = np.load(load_w)['w_ff']
            w_fb = np.load(load_w)['w_fb']
        else:
            print('optimizing encoders from pre_u into ens (T_ff=%.1f)'%T_ff)
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=0.0, learn_ff=True)
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
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=0.0, w_ff=w_ff, learn_fb=True)
            w_fb = data['w_fb']
            fig, ax = plt.subplots()
            sns.distplot(w_fb.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_fb.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_ff=w_ff, w_fb=w_fb)
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
                plt.savefig('plots/tuning/integrate_ens_fb_activity_%s.pdf'%n)
                plt.close('all')

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ff=w_ff, w_fb=w_fb, learn_fd=True)
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(data['x'], data['ens'], f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)
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

        a_ens = f_ens.filt(data['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        target = f.filt(data['x'], dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=target)
        fig, ax = plt.subplots()
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_x_%s_ens_train.pdf"%neuron_type)

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=1, normed='x', f=f, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ff=w_ff, w_fb=w_fb)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        x = f.filt(data['x'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        nrmse_ens = nrmse(xhat_ens, target=x)
        nrmses_ens[test] = nrmse_ens

        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_x_%s_test_%s.pdf"%(neuron_type, test))

    mean_ens = np.mean(nrmses_ens)
    CI_ens = sns.utils.ci(nrmses_ens)

    fig, ax = plt.subplots()
    sns.barplot(data=nrmses_ens)
    ax.set(ylabel='NRMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens, CI_ens[0], CI_ens[1]))
    plt.xticks()
    plt.savefig("plots/integrate_x_%s_nrmse.pdf"%neuron_type)

    print('nrmses: ', nrmses_ens)
    print('means: ', mean_ens)
    print('confidence intervals: ', CI_ens)
    np.savez('data/integrate_x_%s_results.npz'%neuron_type, nrmses_ens=nrmses_ens)
    return nrmses_ens

nrmses_lif = run(neuron_type=LIF())
# nrmses_alif = run(neuron_type=AdaptiveLIFT())
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005, load_fd="data/integrate_x_WilsonEuler()_fd.npz")