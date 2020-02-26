import numpy as np
from scipy.optimize import curve_fit
import nengo
from nengo import SpikingRectifiedLinear
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.networks import LinearNetwork
from nengolib.signal import s, nrmse, LinearSystem, Identity
from nengolib.synapses import ss2sim
from train import norms, df_opt, LearningNode
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')


def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 1), seed=0, dt=0.001, neuron_type=nengo.LIF(),
        f=Lowpass(0.1), f_smooth=Lowpass(0.1), freq=1, w_pre=None, w_ens=None,
        learn_pre_ens=False, learn_supv_ens=False, learn_fd=False):

    w = 2*np.pi*freq
    A = [[0, -w], [w, 0]]
    B = [[1], [0]]
    C = [[1, 0]]
    D = [[0]]
    sys = LinearSystem((A, B, C, D))
    
    with nengo.Network(seed=seed) as model:
                    
        u = nengo.Node(lambda t: [np.sin(w*t), np.cos(w*t)])

        # Ensembles
        pre = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=SpikingRectifiedLinear(), seed=seed, radius=2)
        ens = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2)
        nengo.Connection(u, pre, synapse=None, seed=seed)
        pre_ens = nengo.Connection(pre, ens, synapse=f, seed=seed)

        if learn_pre_ens and isinstance(neuron_type, DurstewitzNeuron):
            supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed, radius=2)
            nengo.Connection(pre, supv, synapse=f, seed=seed)
            node = LearningNode(n_neurons, pre.n_neurons, 1, pre_ens)
            nengo.Connection(pre.neurons, node[0:pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons:pre.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f_smooth)
            nengo.Connection(u, node[-1], synapse=f)

        if learn_supv_ens and isinstance(neuron_type, DurstewitzNeuron):
            supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2)
            pre_supv = nengo.Connection(pre, supv, synapse=f, seed=seed)
            supv_ens = nengo.Connection(supv, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            node = LearningNode(n_neurons, n_neurons, 1, supv_ens)
            nengo.Connection(supv.neurons, node[0:n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[n_neurons:2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            nengo.Connection(u, node[-1], synapse=f)

        if not learn_fd and not learn_pre_ens and not learn_supv_ens:
            off = nengo.Node(lambda t: (t>0.1))
            nengo.Connection(off, pre.neurons, synapse=None, transform=-1e3*np.ones((pre.n_neurons, 1)))
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None) if learn_pre_ens or learn_supv_ens else None,
        p_ens = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_pre):
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    pre_ens.weights[pre, post] = w_pre[pre, post]
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_pre[pre, post])
                    pre_ens.netcons[pre, post].syn().e = 0 if w_pre[pre, post] > 0 else -70
        if np.any(w_ens):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    if learn_supv_ens:
                        supv_ens.weights[pre, post] = w_ens[pre, post]
                        supv_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                        supv_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
                    if not learn_supv_ens and not learn_pre_ens:
                        ens_ens.weights[pre, post] = w_ens[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
    
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        ens=sim.data[p_ens],
        supv=sim.data[p_supv] if learn_pre_ens or learn_supv_ens else None,
        w_pre=pre_ens.weights if isinstance(neuron_type, DurstewitzNeuron) else None,
        w_ens=supv_ens.weights if isinstance(neuron_type, DurstewitzNeuron) and learn_supv_ens else None,
    )


def run(n_neurons=200, t=30, t_test=10, dt=0.001, dt_sample=0.001, seed=0, m=Uniform(20, 40), i=Uniform(-1, 1),
        f=DoubleExp(1e-3, 1e-1), f_out=DoubleExp(1e-3, 1e-1), f_smooth=DoubleExp(1e-2, 2e-1), reg=1e-2,
        freq=1, neuron_type=LIF(), load_fd=False, load_fd_out=None):

    d_ens = np.zeros((n_neurons, 2))
    f_ens = f
    w_pre = None
    w_ens = None
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_pre = np.load(load_w)['w_pre']
        else:
            print('optimizing encoders from pre into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, f_smooth=f_smooth, dt=0.001,
                neuron_type=neuron_type, w_pre=w_pre, w_ens=w_ens, learn_pre_ens=True)
            w_pre = data['w_pre']
            fig, ax = plt.subplots()
            sns.distplot(w_pre.ravel())
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/oscillate_%s_w_supv.pdf"%neuron_type)
            np.savez('data/oscillate_w.npz', w_pre=w_pre)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_supv = f_smooth.filt(data['supv'], dt=0.001)
            for n in range(n_neurons):
                fig, ax = plt.subplots()
                ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
                ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                ax.set(xlabel='time', ylim=((0, 40)))
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/tuning/oscillate_ens1_activity_%s.pdf'%n)
                plt.close('all')

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('gathering filter/decoder training data for ens')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type,
            f_smooth=f_smooth, w_pre=w_pre, w_ens=w_ens, learn_fd=True)
        d_ens, f_ens, taus_ens = df_opt(data['u'], data['ens'], f, dt=dt, name='oscillate_%s'%neuron_type, reg=reg, penalty=0)
        np.savez('data/oscillate_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/oscillate_%s_filters_ens.pdf"%neuron_type)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        x = f.filt(data['u'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        nrmse_ens = nrmse(xhat_ens, target=x)
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="pre_ens")
        plt.legend(loc='upper right')
        plt.savefig("plots/oscillate_%s_pre_ens_train.pdf"%neuron_type)

        fig, ax = plt.subplots()
        sns.distplot(d_ens.ravel())
        ax.set(xlabel='decoders', ylabel='frequency')
        plt.savefig("plots/oscillate_%s_d_ens.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ens = np.load(load_w)['w_ens']
        else:
            print('optimizing encoders into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=0.001,
                f_smooth=f_smooth, neuron_type=neuron_type, w_pre=w_pre, w_ens=w_ens, learn_supv_ens=True)
            w_ens = data['w_ens']
            np.savez('data/w_oscillate.npz', w_pre=w_pre, w_ens=w_ens)
            
#     print('gathering training data for readout filters and decoders')
#     data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type,
#         f_smooth=f_smooth, w_pre=w_pre, w_ens=w_ens)
#     d_out, f_out1, taus_out = df_opt(data['u'][10000:], data['ens'][10000:], f_out, dt=dt, name='oscillate_out_%s'%neuron_type, reg=reg, penalty=0)
#     np.savez('data/oscillate_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out=d_out, taus_out=taus_out)

#     fig, ax = plt.subplots()
#     sns.distplot(d_out.ravel())
#     ax.set(xlabel='decoders', ylabel='frequency')
#     plt.savefig("plots/oscillate_%s_d_out.pdf"%neuron_type)

#     times = np.arange(0, 1, 0.0001)
#     fig, ax = plt.subplots()
#     ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau_1=%.3f, \tau_2=%.3f$"
#         %(-1./f_out.poles[0], -1./f_out.poles[1]))
#     ax.plot(times, f_out1.impulse(len(times), dt=0.0001), label=r"$f^{out1}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
#        %(-1./f_out1.poles[0], -1./f_out1.poles[1], np.count_nonzero(d_out), n_neurons))
#     ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
#     ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.savefig("plots/oscillate_%s_filters_out.pdf"%neuron_type)

#     x = f_out.filt(data['u'], dt=dt)[10000:]
#     a_ens = f_out1.filt(data['ens'], dt=dt)
#     xhat_ens = np.dot(a_ens, d_out)[10000:]
#     nrmse_ens = nrmse(xhat_ens, target=x)
#     fig, ax = plt.subplots()
#     ax.plot(data['times'][10000:], x, linestyle="--", label='x')
#     ax.plot(data['times'][10000:], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
#     ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ens_ens")
#     plt.legend(loc='upper right')
#     plt.savefig("plots/oscillate_%s_train.pdf"%neuron_type)
    

    print("Testing")
    data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type,
        f_smooth=f_smooth, w_pre=w_pre, w_ens=w_ens)

    x = f_out.filt(data['u'], dt=dt)[10000:]
    a_ens = f_ens.filt(data['ens'], dt=dt)
    xhat_ens = np.dot(a_ens, d_ens)[10000:]
    times = data['times'][10000:]

    # curve fit to a sinusoid of arbitrary frequency, phase, magnitude
    # calculate error as difference of fit sinusoid frequency and target sinusoid frequency

    def sinusoid(t, mag, freq, phase):
        return mag * np.sin(t * 2*np.pi*freq + 2*np.pi*phase)
    p0 = [1, 1, 0]
    bounds = ((0, 0, 0), (2, 2, 1))
    param_0, _ = curve_fit(sinusoid, times, x[:,0], p0=p0, bounds=bounds)
    param_1, _ = curve_fit(sinusoid, times, x[:,1], p0=p0, bounds=bounds)
    freq_error = nrmse(np.array([param_0[1], param_1[1]]), target=np.array([freq, freq]))
        

    fig, ax = plt.subplots()
    ax.plot(times, x, linestyle="--", label='x')
    ax.plot(times, xhat_ens, label='ens, freq_error=%.3f' %freq_error)
    ax.set(xlim=((10, 15)), ylim=((-1, 1)), xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ens_ens")
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillate_%s_test.pdf"%neuron_type)

    print('freq_error: ', freq_error)
    fig, ax = plt.subplots()
    sns.barplot(data=np.array([freq_error]))
    ax.set(ylabel='Frequency Error', title="mean=%.3f"%freq_error)
    plt.xticks()
    plt.savefig("plots/oscillate_%s_freq_error.pdf"%neuron_type)
    np.savez('data/oscillate_%s_results.npz'%neuron_type, freq_error=freq_error)
    return freq_error


freq_error_lif = run(t=50, t_test=50, neuron_type=LIF(), reg=1e-1)
# freq_error_alif = run(t=50, t_test=50, neuron_type=AdaptiveLIFT(), reg=1e-1)
# freq_error_wilson = run(t=50, t_test=50, dt=0.00005, neuron_type=WilsonEuler(), reg=1e-1)

# errors = np.vstack((freq_error_lif, freq_error_alif, freq_error_wilson, freq_error_durstewitz))
# nt_names =  ['LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots()
# sns.barplot(data=errors.T)
# ax.set(ylabel='Frequency Errors')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.savefig("plots/oscillate_all_nrmses.pdf")