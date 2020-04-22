import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import norms, d_opt, df_opt, LearningNode
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

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, T_ff=0.2, T_fb=1.0, neuron_type=LIF(), f=Lowpass(0.2), f_smooth=Lowpass(0.2), stim_func=lambda t: np.sin(t), w_ff=None, e_ff=None, w_ff2=None, w_fb=None, e_fb=None, L_ff=False, L_fb=False, supervised=False):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Ensembles
        pre = nengo.Ensemble(300, 1, radius=3, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)

        # Connections
        u_pre = nengo.Connection(u, pre, synapse=None, seed=seed)
        nengo.Connection(u, x, synapse=1/s, seed=seed)
        pre_ens = nengo.Connection(pre, ens, synapse=f, transform=T_ff, seed=seed)
        pre_supv = nengo.Connection(pre, supv, synapse=f, transform=T_ff, seed=seed)

        if L_ff:
            node = LearningNode(n_neurons, pre.n_neurons, 1, pre_ens, k=3e-6)
            nengo.Connection(pre.neurons, node[0: pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons: pre.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f_smooth)

        if supervised:
#             u_pre.synapse = f
            pre2 = nengo.Ensemble(300, 1, radius=3, seed=seed)
            nengo.Connection(x, pre2, synapse=None)
            ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            pre2_ens2 = nengo.Connection(pre2, ens2, synapse=f, seed=seed)
            ens2_ens = nengo.Connection(ens2, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            
        if L_fb:
            node = LearningNode(n_neurons, n_neurons, 1, ens2_ens, k=3e-6)
            nengo.Connection(ens2.neurons, node[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens.neurons, node[n_neurons: 2*n_neurons], synapse=f_smooth)
            nengo.Connection(ens2.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
            
        else:
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), transform=T_fb, seed=seed)

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ff):
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    pre_ens.weights[pre, post] = w_ff[pre, post]
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_ff[pre, post])
                    pre_ens.netcons[pre, post].syn().e = 0 if w_ff[pre, post] > 0 else -70
        if np.any(e_ff):
            pre_ens.e = e_ff
        if np.any(w_ff2):
            for pre in range(300):
                for post in range(n_neurons):
                    pre2_ens2.weights[pre, post] = w_ff2[pre, post]
                    pre2_ens2.netcons[pre, post].weight[0] = np.abs(w_ff2[pre, post])
                    pre2_ens2.netcons[pre, post].syn().e = 0 if w_ff2[pre, post] > 0 else -70
        if np.any(w_fb):
            if supervised:
                for pre in range(n_neurons):
                    for post in range(n_neurons):
                        ens2_ens.weights[pre, post] = w_fb[pre, post]
                        ens2_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens2_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
                if np.any(e_fb):
                    ens2_ens.e = e_fb
            else:
                for pre in range(n_neurons):
                    for post in range(n_neurons):
                        ens_ens.weights[pre, post] = w_fb[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model)

    if L_ff and hasattr(pre_ens, 'weights'):
        w_ff = pre_ens.weights
        e_ff = pre_ens.e
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
        enc=sim.data[ens].encoders,
        w_ff=w_ff,
        w_fb=w_fb,
        e_ff=e_ff if L_ff else None,
        e_fb=e_fb if L_fb else None,
    )


def run(n_neurons=100, t=10, t_test=10, dt=0.001, n_trains=10, n_encodes=10, n_tests=10, neuron_type=LIF(), f=DoubleExp(1e-3, 2e-1), load_w=None, load_fd=None, reg=0, penalty=0.1, T_ff=0.2, T_fb=1.0, supervised=False):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    f_smooth=Lowpass(0.1)
    w_ff = None
    e_ff = None
    w_fb = None
    e_fb = None
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ff = np.load(load_w)['w_ff']
            e_ff = np.load(load_w)['e_ff']
        else:
            print('optimizing encoders from pre into ens (T_ff=1.0)')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='u')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=1.0, T_fb=0.0, e_ff=e_ff, L_ff=True)
            w_ff = data['w_ff']
            e_ff = data['e_ff']
            np.savez('data/integrate_w.npz', w_ff=w_ff, e_ff=e_ff)

            fig, ax = plt.subplots()
            xmin = np.mean(w_ff.ravel()) - 2*np.std(w_ff.ravel())
            xmax = np.mean(w_ff.ravel()) + 2*np.std(w_ff.ravel())
            bins = np.linspace(xmin, xmax, n_neurons)
            sns.distplot(w_ff.ravel(), bins=bins, ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency', xlim=((xmin, xmax)))
            plt.savefig("plots/integrate_%s_w_ff.pdf"%neuron_type)

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
                plt.savefig('plots/tuning/integrate_ens_activity_%s.pdf'%n)
                plt.close('all')
    
    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='u')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=1.0, T_fb=0.0, w_ff=w_ff)            
        target = np.vstack((f.filt(data['u'], dt=dt)))
        spikes = np.vstack((data['ens']))
        print('optimizing filters and decoders')
        d_ens, f_ens, taus_ens = df_opt(target, spikes, f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)
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
        
        a_ens = f_ens.filt(data['ens'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        target = f.filt(f.filt(data['u'], dt=dt), dt=dt)
        nrmse_ens = nrmse(xhat_ens, target=target)
        fig, ax = plt.subplots()
        ax.plot(data['times'], target, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train")
        plt.legend(loc='upper right')
        plt.savefig("plots/integrate_%s_ens_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_fb = np.load(load_w)['w_fb']
            e_fb = np.load(load_w)['e_fb']
        else:
            print('optimizing encoders from ens2 (supv) into ens')
            stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_encodes, f=f, normed='x')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=0.0, w_ff=T_ff*w_ff, w_ff2=w_ff, e_fb=e_fb, L_fb=True, supervised=True)
            w_fb = data['w_fb']
            e_fb = data['e_fb']
            np.savez('data/integrate_w.npz',  w_ff=w_ff, e_ff=e_ff, w_fb=w_fb, e_fb=e_fb)

            fig, ax = plt.subplots()
            xmin = np.mean(w_fb.ravel()) - 2*np.std(w_fb.ravel())
            xmax = np.mean(w_fb.ravel()) + 2*np.std(w_fb.ravel())
            bins = np.linspace(xmin, xmax, n_neurons)
            sns.distplot(w_fb.ravel(), bins=bins, ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency', xlim=((xmin, xmax)))
            plt.savefig("plots/integrate_%s_w_fb.pdf"%neuron_type)

            a_ens2 = f_smooth.filt(data['ens'], dt=0.001)
            a_supv2 = f_smooth.filt(data['ens2'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:40000], a_supv2[:40000,n], alpha=0.5, label='supv2')
                ax.plot(data['times'][:40000], a_ens2[:40000,n], alpha=0.5, label='ens2')
                ax.set(ylabel='firing rate', ylim=((0, 40)))
                ax2.plot(data['times'][-40000:], a_supv2[-40000:,n], alpha=0.5, label='supv2')
                ax2.plot(data['times'][-40000:], a_ens2[-40000:,n], alpha=0.5, label='ens2')
                ax2.set(xlabel='time', ylabel='firing rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/integrate_ens2_ens_activity_%s.pdf'%n)
                plt.close('all')

    nrmses_ens = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t_test, dt=dt, N=1, normed='x', f=f, seed=100+test)
        if supervised:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=1.0, w_ff=T_ff*w_ff if np.any(w_ff) else None, w_ff2=w_ff, w_fb=w_fb, supervised=True)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            a_ens2 = f_ens.filt(data['ens2'], dt=dt)
            u = f.filt(f.filt(f.filt(T_ff*data['u'], dt=dt), dt=dt))
            x = f.filt(f.filt(data['x'], dt=dt), dt=dt)
            xhat_ens = np.dot(a_ens, d_ens)
            xhat_ens2 = np.dot(a_ens2, d_ens)
            nrmse_ens = nrmse(xhat_ens, target=x)  # x2
            nrmse_ens2 = nrmse(xhat_ens2, target=x)

            fig, ax = plt.subplots()
            ax.plot(data['times'], u, linestyle="--", label='u')
            ax.plot(data['times'], x, linestyle="--", label='x')
            ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
            ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
            ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised test")
            plt.legend(loc='upper right')
            plt.savefig("plots/integrate_%s_test_%s_supervised.pdf"%(neuron_type, test))
            plt.close('all')

        else:
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=1.0, w_ff=T_ff*w_ff if np.any(w_ff) else None, w_fb=w_fb)
            a_ens = f_ens.filt(data['ens'], dt=dt)
            u = f.filt(T_ff*data['u'], dt=dt)
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
            plt.savefig("plots/integrate_%s_test_%s.pdf"%(neuron_type, test))
            plt.close('all')

    if not supervised:
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
# nrmses_lif = run(neuron_type=LIF(), load_fd="data/integrate_LIF()_fd.npz", supervised=True, n_tests=3)

# nrmses_alif = run(neuron_type=AdaptiveLIFT())
# nrmses_alif = run(neuron_type=AdaptiveLIFT(), load_fd="data/integrate_AdaptiveLIFT()_fd.npz", supervised=True, n_tests=3)

# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005, load_fd="data/integrate_WilsonEuler()_fd.npz", supervised=False)

# nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), n_tests=3, supervised=True)
nrmses_durstewitz = run(neuron_type=DurstewitzNeuron(0.0), n_tests=3, supervised=False,
   load_w="data/integrate_short_w.npz", load_fd="data/integrate_short_DurstewitzNeuron()_fd.npz")


# nrmses = np.vstack((nrmses_nef, nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
# nt_names =  ['NEF', 'LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# sns.barplot(data=nrmses.T)
# ax.set(ylabel='NRMSE')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/integrate_nrmses.pdf")
