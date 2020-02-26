import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import norms, df_opt, LearningNode
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 1), seed=0, dt=0.001, T=0.1, neuron_type=LIF(),
        f_pre=Lowpass(0.01), f=Lowpass(0.1), f_smooth=Lowpass(0.1), stim_func=lambda t: np.sin(t),
        w_pre=None, w_pre2=None, w_ens=None, learn_pre_ens=False, learn_supv_ens=False, learn_fd=False):

    norm = 1.0 if learn_pre_ens or learn_supv_ens or learn_fd else norms(t, dt, stim_func, f=1/s, value=1.0)
    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre_u = nengo.Ensemble(200, 1, max_rates=m, seed=seed)
        pre_x = nengo.Ensemble(200, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
        supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)

        # Connections
        nengo.Connection(u, x, synapse=1/s, transform=norm, seed=seed)
        nengo.Connection(u, pre_u, synapse=None, transform=norm, seed=seed)
        nengo.Connection(x, pre_x, synapse=None, seed=seed)
        pre_ens = nengo.Connection(pre_u, ens, synapse=f, transform=T, seed=seed)

        if not isinstance(neuron_type, DurstewitzNeuron):
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            nengo.Connection(pre_x, ens2, synapse=f, seed=seed)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(ens2.neurons, synapse=None)

        if learn_pre_ens:
            nengo.Connection(pre_u, supv, synapse=f, transform=T, seed=seed)
            node = LearningNode(n_neurons, pre_u.n_neurons, 1, pre_ens, k=1e-4)
            nengo.Connection(pre_u.neurons, node[0:pre_u.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre_u.n_neurons:pre_u.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre_u.n_neurons+n_neurons: pre_u.n_neurons+2*n_neurons], synapse=f_smooth)

            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)            
            pre_ens2 = nengo.Connection(pre_x, ens2, synapse=f_pre, seed=seed)
            nengo.Connection(pre_x, supv2, synapse=f_pre, seed=seed)
            node = LearningNode(n_neurons, pre_x.n_neurons, 1, pre_ens2, k=1e-4)
            nengo.Connection(pre_x.neurons, node[0:pre_x.n_neurons], synapse=f_pre)
            nengo.Connection(ens2.neurons, node[pre_x.n_neurons:pre_x.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node[pre_x.n_neurons+n_neurons: pre_x.n_neurons+2*n_neurons], synapse=f_smooth)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        if learn_fd:
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)            
            nengo.Connection(pre_u, supv, synapse=f, transform=T, seed=seed)
            pre_ens2 = nengo.Connection(pre_x, ens2, synapse=f_pre, seed=seed)
            nengo.Connection(pre_x, supv2, synapse=f_pre, seed=seed)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        if learn_supv_ens:
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)  # acts as rate supv and teacher        
            pre_ens2 = nengo.Connection(pre_x, ens2, synapse=f_pre, seed=seed)
            nengo.Connection(pre_x, supv2, synapse=f_ens, seed=seed)
            supv_ens = nengo.Connection(ens2, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            node = LearningNode(n_neurons, n_neurons, 1, supv_ens, k=1e-7)
            nengo.Connection(ens2.neurons, node[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        if not learn_pre_ens and not learn_supv_ens and not learn_fd:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens), label='ens_ens')
            p_ens2 = nengo.Probe(ens.neurons, synapse=None)  # filler, so probes are defined
            p_supv2 = nengo.Probe(supv.neurons, synapse=None)  # filler, so probes are defined

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_pre):
            for pre in range(pre_u.n_neurons):
                for post in range(n_neurons):
                    pre_ens.weights[pre, post] = w_pre[pre, post] # transform
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post])  # transform
                    pre_ens.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
        if np.any(w_pre2):
            for pre in range(pre_x.n_neurons):
                for post in range(n_neurons):
                    pre_ens2.weights[pre, post] = w_pre2[pre, post]
                    pre_ens2.netcons[pre, post].weight[0] = np.abs(w_pre2[pre, post])
                    pre_ens2.netcons[pre, post].syn().e = 0 if w_pre2[pre, post] > 0 else -70
        if np.any(w_ens):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    ens_ens.weights[pre, post] = w_ens[pre, post]
                    ens_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                    ens_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model)

    if hasattr(pre_ens, 'weights'):
        w_pre = pre_ens.weights
    if learn_pre_ens and hasattr(pre_ens2, 'weights'):
        w_pre2 = pre_ens2.weights
    if learn_supv_ens and hasattr(supv_ens, 'weights'):
        w_ens = supv_ens.weights

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        x=sim.data[p_x],
        ens=sim.data[p_ens],
        ens2=sim.data[p_ens2],
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2],
        enc=sim.data[supv].encoders,
        w_pre=w_pre,
        w_pre2=w_pre2,
        w_ens=w_ens,
    )


def run(n_neurons=200, t=10, t_test=10, t_enc=100, dt=0.001, n_trains=10, n_tests=10, neuron_type=LIF(),
        f=DoubleExp(1e-3, 1e-1), f_out=DoubleExp(1e-3, 1e-1), f_smooth=DoubleExp(1e-2, 2e-1),
        load_w=None, load_fd=None):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_pre = None
    w_pre2 = None
    w_ens = None
    print('Neuron Type: %s'%neuron_type)

    print('Creating input signal from concatenated, normalized, flipped white noise')
    u_list = np.zeros((int(300/dt), 1))
    for n in range(n_trains):
        stim_func = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.5, seed=n)
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
        u_list[n*int(t/dt): (n+1)*int(t/dt)] = sim.data[p_u] / np.max(np.abs(sim.data[p_x]))
    stim_func = lambda t: u_list[int(t/dt)]

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_pre = np.load(load_w)['w_pre']
            w_pre2 = np.load(load_w)['w_pre2']
            # w_ens = np.load(load_w)['w_ens']
        else:
            print('optimizing encoders from pre_u into ens and pre_x into ens2 (supv)')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_enc, f=f, dt=dt, stim_func=stim_func,
                f_smooth=f_smooth, neuron_type=neuron_type, w_pre=w_pre, learn_pre_ens=True)
            w_pre = data['w_pre']
            w_pre2 = data['w_pre2']
            fig, ax = plt.subplots()
            sns.distplot(w_pre.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_pre.pdf"%neuron_type)
            fig, ax = plt.subplots()
            sns.distplot(w_pre2.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/integrate_%s_w_pre2.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_pre=w_pre, w_pre2=w_pre2)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_supv = f_smooth.filt(data['supv'], dt=0.001)
            for n in range(n_neurons):
                fig, ax = plt.subplots()
                ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
                ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                ax.set(xlabel='time', ylim=((0, 40)))
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/tuning/integrate_activity_time_%s.pdf'%n)
                plt.close('all')
            a_ens = f_smooth.filt(data['ens2'], dt=0.001)
            a_supv = f_smooth.filt(data['supv2'], dt=0.001)
            for n in range(n_neurons):
                fig, ax = plt.subplots()
                ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv2')
                ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens2')
                ax.set(xlabel='time', ylim=((0, 40)))
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/tuning/integrate_ens2_activity_%s.pdf'%n)
                plt.close('all')

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens (ens2=supv for Durstewitz)')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, stim_func=stim_func,
            f_smooth=f_smooth, neuron_type=neuron_type, w_pre=w_pre, w_pre2=w_pre2, learn_fd=True)
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(data['x'], data['ens2'], f, dt=dt, penalty=0.1, name='integrate_%s'%neuron_type)
        np.savez('data/integrate_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)
        
        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/integrate_%s_filters_ens.pdf"%neuron_type)


        a_ens2 = f_ens.filt(data['ens2'], dt=dt)
        x = f.filt(data['x'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_ens)
        nrmse_ens2 = nrmse(xhat_ens2, target=x)
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train supv")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_%s_supv_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_pre = np.load(load_w)['w_pre']
            w_pre2 = np.load(load_w)['w_pre2']
            # w_ens = np.load(load_w)['w_ens']
        else:
            print('optimizing encoders from ens2 into ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_enc, f=f, dt=dt, stim_func=stim_func,
                f_smooth=f_smooth, neuron_type=neuron_type, w_pre=w_pre, w_pre2=w_pre2, learn_supv_ens=True)
            w_ens = data['w_ens']
            fig, ax = plt.subplots()
            sns.distplot(w_ens.ravel())
            plt.savefig("plots/integrate_%s_w_ens.pdf"%neuron_type)
            np.savez('data/integrate_w.npz', w_pre=w_pre, w_pre2=w_pre2, w_ens=w_ens)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_ens2 = f_smooth.filt(data['ens2'], dt=0.001)
            a_supv2 = f_smooth.filt(data['supv2'], dt=0.001)
            for n in range(n_neurons):
                fig, ax = plt.subplots()
                ax.plot(data['times'], a_supv2[:,n], alpha=0.5, label='supv2')
                ax.plot(data['times'], a_ens2[:,n], alpha=0.5, label='ens2')
                ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                ax.set(xlabel='time', ylim=((0, 40)))
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/tuning/integrate_ens_supv_activity_%s.pdf'%n)
                plt.close('all')

            x = f.filt(data['x'], dt=dt)
            a_ens = f_ens.filt(data['ens'], dt=0.001)
            a_ens2 = f_ens.filt(data['ens2'], dt=0.001)
            xhat_ens = np.dot(a_ens, d_ens)
            xhat_ens2 = np.dot(a_ens2, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)
            nrmse_ens2 = nrmse(xhat_ens2, target=x)
            fig, ax = plt.subplots(figsize=((12, 12)))
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train enc2")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_%s_enc2_train.pdf"%neuron_type)
            
    # readout filter/decoder optimization?

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t_test, high=1, rms=0.5, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt,
            f_smooth=f_smooth, neuron_type=neuron_type, stim_func=stim_func, w_pre=w_pre, w_pre2=None, w_ens=w_ens)

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

# nrmses_lif = run(neuron_type=LIF())
# nrmses_alif = run(neuron_type=AdaptiveLIFT())
nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)
# nrmses_durstewitz = run(n_neurons=30, t=30, t_enc=100, n_tests=1, neuron_type=DurstewitzNeuron())
    # load_fd="data/fd_integrate_DurstewitzNeuron().npz", load_w="data/w_integrate.npz")

# nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
# nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sns.barplot(data=nrmses.T)
# ax.set(ylabel='NRMSE')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/integrate_nrmses.pdf")
