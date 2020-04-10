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

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, T_ff=0.1, T_fb=1.0, neuron_type=LIF(),
        f=Lowpass(0.1), f_smooth=Lowpass(0.1), stim_func=lambda t: np.sin(t),
        w_pre=None, w_ens=None, learn_pre_ens=False, learn_ens_ens2=False, progress_bar=True):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre = nengo.Ensemble(300, 1, radius=3, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        pre_ens = nengo.Connection(pre, ens, synapse=f, transform=T_ff, seed=seed)
        pre_supv = nengo.Connection(pre, supv, synapse=f, transform=T_ff, seed=seed)

        if learn_pre_ens:
            node = LearningNode(n_neurons, pre.n_neurons, 1, pre_ens, k=3e-6)
            nengo.Connection(pre.neurons, node[0: pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons: pre.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f_smooth)

        if learn_ens_ens2:
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
            ens_ens2 = nengo.Connection(ens, ens2, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            supv_supv2 = nengo.Connection(supv, supv2, synapse=f, seed=seed)
            node = LearningNode(n_neurons, n_neurons, 1, ens_ens2, k=3e-6)
            nengo.Connection(ens.neurons, node[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens2.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        if not learn_pre_ens and not learn_ens_ens2:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), transform=T_fb, seed=seed)

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=progress_bar) as sim:
        if np.any(w_pre) and T_ff:
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    pre_ens.weights[pre, post] = w_pre[pre, post] * T_ff
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post]) * T_ff
                    pre_ens.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
        if np.any(w_ens) and T_fb:
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    ens_ens.weights[pre, post] = w_ens[pre, post]
                    ens_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                    ens_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t, progress_bar=progress_bar)
        reset_neuron(sim, model)

    if learn_pre_ens and hasattr(pre_ens, 'weights'):
        w_pre = pre_ens.weights
    if learn_ens_ens2 and hasattr(ens_ens2, 'weights'):
        w_ens = ens_ens2.weights

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        ens=sim.data[p_ens],
        ens2=sim.data[p_ens2] if learn_ens_ens2 else None,
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2] if learn_ens_ens2 else None,
        enc=sim.data[ens].encoders,
        w_pre=w_pre,
        w_ens=w_ens,
    )


def run(n_neurons=100, t=10, t_test=10, dt=0.001, n_trains=10, n_encodes=10, n_tests=10, neuron_type=LIF(), f=DoubleExp(1e-3, 1e-1), load_w=None, load_fd=None, reg=0, penalty=0.1, T_ff=0.1, T_fb=1.0):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    f_smooth=Lowpass(0.1)
    w_pre = None
    w_ens = None
    print('Neuron Type: %s'%neuron_type)

    stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='u')

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
#             w_pre = np.load(load_w)['w_pre']
            w_pre10 = np.load(load_w)['w_pre10']
            w_pre01 = np.load(load_w)['w_pre01']
        else:
            print('optimizing encoders from pre into ens (T_ff=1.0)')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=1.0, T_fb=0.0, learn_pre_ens=True)
            w_pre10 = data['w_pre']
            e_pre10 = data['e_pre']
            fig, ax = plt.subplots()
            sns.distplot(w_pre10.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_pre10.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_pre10=w_pre10)
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
                plt.savefig('plots/tuning/integrate_ens10_activity_%s.pdf'%n)
                plt.close('all')
            
            print('optimizing encoders from pre into ens (T_ff=0.1)')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func2, T_ff=0.1, T_fb=0.0, learn_pre_ens=True)
            w_pre01 = data['w_pre']
            e_pre01 = data['e_pre']
            fig, ax = plt.subplots()
            sns.distplot(w_pre01.ravel(), ax=ax)
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_pre01.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_pre10=w_pre10, e_pre10=e_pre10,  w_pre01=w_pre01, e_pre01=e_pre01)
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
                plt.savefig('plots/tuning/integrate_ens01_activity_%s.pdf'%n)
                plt.close('all')
    
    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=1.0, T_fb=0.0, w_pre=w_pre)
        print('optimizing filters and decoders')
        # just training decoders for identity: stimulate with u, target is filtered u
        target = f.filt(data['u'], dt=dt)
        d_ens, f_ens, taus_ens = df_opt(target, data['ens'], f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/integrate_%s_filters_ens.pdf"%neuron_type)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        target = f.filt(f.filt(data['u'], dt=dt), dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=target)
        fig, ax = plt.subplots()
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_x_%s_ens_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_pre10 = np.load(load_w)['w_pre10']
            w_pre01 = np.load(load_w)['w_pre01']
            w_ens = np.load(load_w)['w_ens']
        else:
            print('optimizing encoders from ens into ens2')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=1.0, T_fb=0.0, w_pre=w_pre10, learn_ens_ens2=True)
            w_ens = data['w_ens']
            e_ens = data['e_ens']
            fig, ax = plt.subplots()
            sns.distplot(w_ens.ravel(), ax=ax)
            plt.savefig("plots/integrate_%s_w_ens.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_pre10=w_pre10, w_ens=w_ens, w_pre01=w_pre01)
            a_ens2 = f_smooth.filt(data['ens2'], dt=0.001)
            a_supv2 = f_smooth.filt(data['supv2'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv2[:20000,n], alpha=0.5, label='supv2')
                ax.plot(data['times'][:20000], a_ens2[:20000,n], alpha=0.5, label='ens2')
                ax.set(ylabel='firing rate', ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv2[-20000:,n], alpha=0.5, label='supv2')
                ax2.plot(data['times'][-20000:], a_ens2[-20000:,n], alpha=0.5, label='ens2')
                ax2.set(xlabel='time', ylabel='firing rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens_ens2_activity_%s.pdf'%n)
                plt.close('all')

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t_test, dt=dt, N=1, normed='x', f=f, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=1.0, w_pre=w_pre, w_ens=w_ens)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        x = f.filt(data['x'], dt=dt)
#         x = f.filt(0.1*f.filt(data['u'], dt=dt), dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
#         xhat_ens = np.dot(a_ens, d_out)
        nrmse_ens = nrmse(xhat_ens, target=x)
        nrmses_ens[test] = nrmse_ens

        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_%s_test_%s.pdf"%(neuron_type, test))

    mean_ens = np.mean(nrmses_ens)
    CI_ens = sns.utils.ci(nrmses_ens)

    fig, ax = plt.subplots()
    sns.barplot(data=nrmses_ens)
    ax.set(ylabel='NRMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens, CI_ens[0], CI_ens[1]))
    plt.xticks()
    plt.savefig("plots/integrate_%s_nrmse.pdf"%neuron_type)

    print('nrmses: ', nrmses_ens)
    print('means: ', mean_ens)
    print('confidence intervals: ', CI_ens)
    np.savez('data/integrate_%s_results.npz'%neuron_type, nrmses_ens=nrmses_ens)
    return nrmses_ens

nrmses_lif = run(neuron_type=LIF())
nrmses_alif = run(neuron_type=AdaptiveLIFT())
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005,
#     load_fd="data/integrate_WilsonEuler()_fd.npz")
# nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), n_neurons=100, n_tests=10, t=10, reg=1e-2, T_evals=20,
#     load_fd="data/integrate_w/integrate_DurstewitzNeuron()_fd.npz", load_w="data/integrate_w/integrate_w.npz")


# nrmses_durstewitz = run(n_neurons=100, n_encodes=20, n_trains=10, n_tests=1, neuron_type=DurstewitzNeuron())



# nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
# nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sns.barplot(data=nrmses.T)
# ax.set(ylabel='NRMSE')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/integrate_nrmses.pdf")
