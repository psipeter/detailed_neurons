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


def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=0, dt=0.001, neuron_type=nengo.LIF(), f=Lowpass(0.1), f_smooth=Lowpass(0.1), freq=1, w_ff=None, w_fb=None, e_ff=None, e_fb=None, L_ff=False, L_fb=False, supervised=False):

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

        if L_ff:
            supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed, radius=2)
            nengo.Connection(pre, supv, synapse=f, seed=seed)
            node = LearningNode(n_neurons, pre.n_neurons, 1, pre_ens)
            nengo.Connection(pre.neurons, node[0:pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons:pre.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f_smooth)
            nengo.Connection(u, node[-1], synapse=f)

        if L_fb:
            supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2)
            pre_supv = nengo.Connection(pre, supv, synapse=f, seed=seed)
            supv_ens = nengo.Connection(supv, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))
            node = LearningNode(n_neurons, n_neurons, 1, supv_ens)
            nengo.Connection(supv.neurons, node[0:n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[n_neurons:2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_smooth)
            nengo.Connection(u, node[-1], synapse=f)

        if not L_ff and not L_fb and not supervised:
            off = nengo.Node(lambda t: (t>0.1))
            nengo.Connection(off, pre.neurons, synapse=None, transform=-1e3*np.ones((pre.n_neurons, 1)))
            ens_ens = nengo.Connection(ens, ens, synapse=f_ens, seed=seed, solver=NoSolver(d_ens))

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None) if L_ff or L_fb else None,
        p_ens = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ff):
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    pre_ens.weights[pre, post] = w_ff[pre, post]
                    pre_ens.netcons[pre, post].weight[0] = np.abs(w_ff[pre, post])
                    pre_ens.netcons[pre, post].syn().e = 0 if w_ff[pre, post] > 0 else -70
        if np.any(e_ff) and L_ff:
            pre_ens.e = e_ff
        if np.any(w_fb):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    if L_fb:
                        supv_ens.weights[pre, post] = w_fb[pre, post]
                        supv_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        supv_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
                    else:
                        ens_ens.weights[pre, post] = w_fb[pre, post]
                        ens_ens.netcons[pre, post].weight[0] = np.abs(w_fb[pre, post])
                        ens_ens.netcons[pre, post].syn().e = 0 if w_fb[pre, post] > 0 else -70
        if np.any(e_fb) and L_fb:
            supv_ens.e = e_fb

        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 

    if L_ff and hasattr(pre_ens, 'weights'):
        w_ff = pre_ens.weights
        e_ff = pre_ens.e
    if L_fb and hasattr(supv_ens, 'weights'):
        w_fb = supv_ens.weights
        e_fb = supv_ens.e

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        ens=sim.data[p_ens],
        supv=sim.data[p_supv] if L_ff or L_fb else None,
        w_ff=w_ff,
        w_fb=w_fb,
        e_ff=e_ff,
        e_fb=e_fb,
    )


def run(n_neurons=200, t=11, t_test=11, dt=0.001, f=DoubleExp(1e-3, 2e-1), penalty=0, reg=1e-1, freq=1, neuron_type=LIF(), load_fd=False, load_fd_out=None):

    d_ens = np.zeros((n_neurons, 2))
    f_ens = f
    w_ff = None
    w_fb = None
    e_ff = None
    e_fb = None
    f_smooth=DoubleExp(1e-2, 2e-1)
    print('Neuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ff = np.load(load_w)['w_ff']
        else:
            print('optimizing encoders from pre into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=0.001,
                neuron_type=neuron_type, w_ff=w_ff, e_ff=e_ff, L_ff=True)
            w_ff = data['w_ff']
            np.savez('data/oscillate_w.npz', w_ff=w_ff, e_ff=e_ff)

            fig, ax = plt.subplots()
            xmin = np.mean(w_ff.ravel()) - 2*np.std(w_ff.ravel())
            xmax = np.mean(w_ff.ravel()) + 2*np.std(w_ff.ravel())
            bins = np.linspace(xmin, xmax, n_neurons)
            sns.distplot(w_ff.ravel(), bins=bins, ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency', xlim=((xmin, xmax)))
            plt.savefig("plots/oscillate_%s_w_ff.pdf"%neuron_type)
            
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
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, w_ff=w_ff, supervised=True)
        d_ens, f_ens, taus_ens = df_opt(data['u'][1000:], data['ens'][1000:], f, dt=dt, name='oscillate_%s'%neuron_type, reg=reg, penalty=penalty)
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

#         fig, ax = plt.subplots()
#         sns.distplot(d_ens.ravel())
#         ax.set(xlabel='decoders', ylabel='frequency')
#         plt.savefig("plots/oscillate_%s_d_ens.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_fb = np.load(load_w)['w_fb']
        else:
            print('optimizing encoders into DurstewitzNeuron ens')
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=0.001, neuron_type=neuron_type, w_ff=w_ff, w_fb=w_fb, e_fb=e_fb, L_fb=True)
            w_fb = data['w_fb']
            np.savez('data/oscillate_w.npz', w_ff=w_ff, e_ff=e_ff, w_fb=w_fb, e_fb=e_fb)
            
            fig, ax = plt.subplots()
            xmin = np.mean(w_fb.ravel()) - 2*np.std(w_fb.ravel())
            xmax = np.mean(w_fb.ravel()) + 2*np.std(w_fb.ravel())
            bins = np.linspace(xmin, xmax, n_neurons)
            sns.distplot(w_fb.ravel(), bins=bins, ax=ax, kde=False)
            ax.set(xlabel='weights', ylabel='frequency', xlim=((xmin, xmax)))
            plt.savefig("plots/oscillate_%s_w_fb.pdf"%neuron_type)
            
    print("Testing")
    data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, w_ff=w_ff, w_fb=w_fb)

    a_ens = f_ens.filt(data['ens'], dt=dt)
    xhat_ens_0 = np.dot(a_ens, d_ens)[:,0]
    xhat_ens_1 = np.dot(a_ens, d_ens)[:,1]
    x_0 = f.filt(data['u'], dt=dt)[:,0]
    x_1 = f.filt(data['u'], dt=dt)[:,1]
    times = data['times']

    # curve fit to a sinusoid of arbitrary frequency, phase, magnitude
    def sinusoid(t, freq, phase, dt=dt):  # mag
        return f.filt(np.sin(t * 2*np.pi*freq + 2*np.pi*phase), dt=dt)
    p0 = [1, 0]  # [1, 1, 0]
    bounds = ((0, 0, 0), (2, 2, 1))
    param_0, _ = curve_fit(sinusoid, times[1000:], xhat_ens_0[1000:], p0=p0)  # , bounds=bounds
    param_1, _ = curve_fit(sinusoid, times[1000:], xhat_ens_1[1000:], p0=p0)  # , bounds=bounds
    print('param0', param_0)
    print('param1', param_1)
    sinusoid_0 = sinusoid(times, param_0[0], param_0[1])
    sinusoid_1 = sinusoid(times, param_1[0], param_1[1])

    # error is nrmse of xhat and best fit sinusoid times freq error of best fit sinusoid to x
    freq_error_0 = np.abs(freq - param_0[1])
    freq_error_1 = np.abs(freq - param_1[1])
    nrmse_0 = nrmse(xhat_ens_0[1000:], target=sinusoid_0[1000:])
    nrmse_1 = nrmse(xhat_ens_1[1000:], target=sinusoid_1[1000:])
    scaled_nrmse_0 = (1+freq_error_0) * nrmse_0
    scaled_nrmse_1 = (1+freq_error_1) * nrmse_1

    fig, ax = plt.subplots()
    ax.plot(times[1000:], x_0[1000:], linestyle="--", label='x0')
    ax.plot(times[1000:], sinusoid_0[1000:], label='best fit sinusoid_0')
    ax.plot(times[1000:], xhat_ens_0[1000:], label='ens, scaled nrmse=%.3f' %scaled_nrmse_0)
    ax.set(xlim=((1, 11)), ylim=((-1, 1)), xlabel='time (s)', ylabel=r'$\mathbf{x}$')
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillate_%s_test_0.pdf"%neuron_type)
    
    fig, ax = plt.subplots()
    ax.plot(times[1000:], x_1[1000:], linestyle="--", label='x1')
    ax.plot(times[1000:], sinusoid_1[1000:], label='best fit sinusoid_1')
    ax.plot(times[1000:], xhat_ens_1[1000:], label='ens, scaled nrmse=%.3f' %scaled_nrmse_1)
    ax.set(xlim=((1, 11)), ylim=((-1, 1)), xlabel='time (s)', ylabel=r'$\mathbf{x}$')
    plt.legend(loc='upper right')
    plt.savefig("plots/oscillate_%s_test_1.pdf"%neuron_type)

    print('scaled nrmses: ', scaled_nrmse_0, scaled_nrmse_1)
    mean = np.mean([scaled_nrmse_0, scaled_nrmse_1])
    fig, ax = plt.subplots()
    sns.barplot(data=np.array([mean]))
    ax.set(ylabel='Scaled NRMSE', title="mean=%.3f"%mean)
    plt.xticks()
    plt.savefig("plots/oscillate_%s_scaled_nrmse.pdf"%neuron_type)
    np.savez('data/oscillate_%s_results.npz'%neuron_type, scaled_nrmse_0=scaled_nrmse_0, scaled_nrmse_1=scaled_nrmse_1)
    return mean


scaled_nrmse_lif = run(neuron_type=LIF())
scaled_nrmse_alif = run(neuron_type=AdaptiveLIFT())
scaled_nrmse_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)

# errors = np.vstack((freq_error_lif, freq_error_alif, freq_error_wilson, freq_error_durstewitz))
# nt_names =  ['LIF', 'ALIF', 'Wilson', 'Durstewitz']
# fig, ax = plt.subplots()
# sns.barplot(data=errors.T)
# ax.set(ylabel='Frequency Errors')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.savefig("plots/oscillate_all_nrmses.pdf")