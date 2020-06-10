import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import d_opt, df_opt, LearningNode2
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def make_normed_flipped(value=1.0, t=1.0, dt=0.001, N=1, f=Lowpass(0.01), seed=0):
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
        with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
            sim.run(t, progress_bar=False)
        u = f.filt(sim.data[p_u], dt=dt)
        norm = value / np.max(np.abs(u))
        u_list[n*int(t/dt): (n+1)*int(t/dt)] = sim.data[p_u] * norm
    stim_func = lambda t: u_list[int(t/dt)]
    return stim_func

def go(d_ens, f_ens, n_pre=100, n_neurons=30, t=10, m=Uniform(30, 40), i=Uniform(-1, 0.6), seed=1, dt=0.001, f=Lowpass(0.01), f_smooth=DoubleExp(1e-2, 2e-1),
        neuron_type=LIF(), w_ens=None, e_ens=None, w_ens2=None, e_ens2=None, L=False, L2=False, stim_func=lambda t: np.sin(t)):

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)

        # Ensembles
        pre = nengo.Ensemble(n_pre, 1, radius=1.5, seed=seed)
        ens = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed+1)
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
        supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed+1)
        x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        x2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre, synapse=None, seed=seed)
        conn = nengo.Connection(pre, ens, synapse=f, seed=seed)
        conn2 = nengo.Connection(ens, ens2, synapse=f_ens, seed=seed+1, solver=NoSolver(d_ens))
        nengo.Connection(x, supv, synapse=None, seed=seed)
        nengo.Connection(x2, supv2, synapse=None, seed=seed+1)
        nengo.Connection(u, x, synapse=f, seed=seed)
        nengo.Connection(x, x2, synapse=f, seed=seed+1)
            
        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_v = nengo.Probe(ens.neurons, 'voltage', synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        p_v2 = nengo.Probe(ens2.neurons, 'voltage', synapse=None)
        p_x2 = nengo.Probe(x2, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)
        p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        # Bioneurons
        if L and isinstance(neuron_type, DurstewitzNeuron):
            node = LearningNode2(n_neurons, n_pre, conn, k=3e-6)
            nengo.Connection(pre.neurons, node[0:n_pre], synapse=f)
            nengo.Connection(ens.neurons, node[n_pre:n_pre+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_smooth)
        if L2 and isinstance(neuron_type, DurstewitzNeuron):
            node2 = LearningNode2(n_neurons, n_neurons, conn2, k=3e-6)
            nengo.Connection(ens.neurons, node2[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens2.neurons, node2[n_neurons:2*n_neurons], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node2[2*n_neurons: 3*n_neurons], synapse=f_smooth)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ens):
            for pre in range(n_pre):
                for post in range(n_neurons):
                    conn.weights[pre, post] = w_ens[pre, post]
                    conn.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                    conn.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        if np.any(w_ens2):
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    conn2.weights[pre, post] = w_ens2[pre, post]
                    conn2.netcons[pre, post].weight[0] = np.abs(w_ens2[pre, post])
                    conn2.netcons[pre, post].syn().e = 0 if w_ens2[pre, post] > 0 else -70
        if np.any(e_ens):
            conn.e = e_ens
        if np.any(e_ens2):
            conn2.e = e_ens2
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
        
    if L and hasattr(conn, 'weights'):
        w_ens = conn.weights
        e_ens = conn.e
    if L2 and hasattr(conn2, 'weights'):
        w_ens2 = conn2.weights
        e_ens2 = conn2.e

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        ens=sim.data[p_ens],
        v=sim.data[p_v],
        x=sim.data[p_x],
        ens2=sim.data[p_ens2],
        v2=sim.data[p_v2],
        x2=sim.data[p_x2],
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2],
        w_ens=w_ens,
        e_ens=e_ens,
        w_ens2=w_ens2,
        e_ens2=e_ens2,
    )


def run(n_neurons=30, t=30, t_test=10, n_encodes=20, dt=0.001, n_tests=10, neuron_type=LIF(),
        f=DoubleExp(1e-3, 3e-2), f_out=DoubleExp(1e-3, 1e-1), reg=0, penalty=1.0, 
        load_w=None, load_df=None):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    f_smooth = DoubleExp(1e-2, 2e-1)
    w_ens = None
    e_ens = None
    w_ens2 = None
    e_ens2 = None
    print('\nNeuron Type: %s'%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ens = np.load(load_w)['w_ens']
        else:
            print('Optimizing ens1 encoders')
            for nenc in range(n_encodes):
                print("encoding trial %s"%nenc)
                stim_func = make_normed_flipped(value=1.2, t=t, dt=dt, f=f, seed=nenc)
                data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=0.001, stim_func=stim_func,
                    neuron_type=neuron_type, w_ens=w_ens, e_ens=e_ens, L=True)
                e_ens = data['e_ens']
                w_ens = data['w_ens']
                np.savez('data/identity_w.npz', e_ens=e_ens, w_ens=w_ens)
                
                fig, ax = plt.subplots()
                sns.distplot(np.ravel(w_ens), ax=ax, kde=False)
                ax.set(xlabel='weights', ylabel='frequency')
                plt.savefig("plots/tuning/identity_%s_w_ens_nenc_%s.pdf"%(neuron_type, nenc))
                
                a_ens = f_smooth.filt(data['ens'], dt=dt)
                a_supv = f_smooth.filt(data['supv'], dt=dt)
                for n in range(n_neurons):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
                    ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens')
                    ax.set(ylim=((0, 40)))
                    plt.legend()
                    plt.savefig('plots/tuning/identity_ens_nenc_%s_activity_%s.pdf'%(nenc, n))
                    plt.close('all')

    if load_df:
        load = np.load(load_df)
        d_ens = load['d_ens']
        d_out1 = load['d_out1']
        taus_ens = load['taus_ens']
        taus_out1 = load['taus_out1']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
        f_out1 =  DoubleExp(taus_out1[0], taus_out1[1])
    else:
        print('Optimizing ens1 filters and decoders')
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, f=f)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type,
            stim_func=stim_func, w_ens=w_ens)
        d_ens, f_ens, taus_ens = df_opt(data['x'], data['ens'], f, dt=dt, reg=reg, penalty=penalty, name='identity_%s'%neuron_type)
        d_out1, f_out1, taus_out1 = df_opt(data['x'], data['ens'], f_out, dt=dt, reg=0, penalty=0, name='identity_%s'%neuron_type)
        np.savez('data/identity_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out1=d_out1, taus_out1=taus_out1)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/identity_%s_filters_ens.pdf"%neuron_type)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau=%.3f, \tau_2=%.3f$" %(-1./f_out.poles[0], -1./f_out.poles[1]))
        ax.plot(times, f_out1.impulse(len(times), dt=0.0001), label=r"$f^{out1}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_out1.poles[0], -1./f_out1.poles[1], np.count_nonzero(d_out1), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/identity_%s_filters_out1.pdf"%neuron_type)
   
        a_ens = f_out1.filt(data['ens'], dt=dt)
        x = f_out.filt(data['x'], dt=dt)
        xhat_ens = np.dot(a_ens, d_out1)
        rmse_ens = rmse(xhat_ens, x)
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, rmse=%.3f' %rmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/identity_%s_ens1_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ens2 = np.load(load_w)['w_ens2']
        else:
            print('Optimizing ens2 encoders')
            for nenc in range(n_encodes):
                print("encoding trial %s"%nenc)
                stim_func = make_normed_flipped(value=1.2, t=t, dt=dt, f=f, seed=nenc)
                data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, f_smooth=f_smooth, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2, e_ens2=e_ens2, L2=True)
                w_ens2 = data['w_ens2']
                e_ens2 = data['e_ens2']
                
                fig, ax = plt.subplots()
                sns.distplot(np.ravel(w_ens2), ax=ax)
                ax.set(xlabel='weights', ylabel='frequency')
                plt.savefig("plots/tuning/identity_%s_w_ens2_nenc_%s.pdf"%(neuron_type, nenc))
                np.savez('data/identity_w.npz', w_ens=w_ens, w_ens2=w_ens2, e_ens=e_ens, e_ens2=e_ens2)
                
                a_ens = f_smooth.filt(data['ens2'], dt=dt)
                a_supv = f_smooth.filt(data['supv2'], dt=dt)
                for n in range(n_neurons):
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv2')
                    ax.plot(data['times'], a_ens[:,n], alpha=0.5, label='ens2')
                    ax.set(ylim=((0, 40)))
                    plt.legend()
                    plt.savefig('plots/tuning/identity_ens2_nenc_%s_activity_%s.pdf'%(nenc, n))
                    plt.close('all')

    if load_df:
        load = np.load(load_df)
        d_out2 = load['d_out2']
        taus_out2 = load['taus_out2']
        f_out2 = DoubleExp(taus_out2[0], taus_out2[1])
    else:
        print('Optimizing ens2 filters and decoders')
        stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, f=f)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2)
        d_out2, f_out2, taus_out2  = df_opt(data['x2'], data['ens2'], f_out, dt=dt, reg=0, penalty=0, name='identity_%s'%neuron_type)
        np.savez('data/identity_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out1=d_out1, taus_out1=taus_out1, d_out2=d_out2, taus_out2=taus_out2)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau=%.3f, \tau_2=%.3f$" %(-1./f_out.poles[0], -1./f_out.poles[1]))
        ax.plot(times, f_out2.impulse(len(times), dt=0.0001), label=r"$f^{out2}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_out2.poles[0], -1./f_out2.poles[1], np.count_nonzero(d_out2), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/identity_%s_filters_out2.pdf"%neuron_type)

        fig, ax = plt.subplots()
        sns.distplot(np.ravel(d_out2))
        ax.set(xlabel='decoders', ylabel='frequency')
        plt.savefig("plots/identity_%s_d_out2.pdf"%neuron_type)

        a_ens2 = f_out2.filt(data['ens2'], dt=dt)
        x2 = f_out.filt(data['x2'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_out2)
        rmse_ens2 = rmse(xhat_ens2, x2)

        fig, ax = plt.subplots()
        ax.plot(data['times'], x2, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens2, label='ens2, rmse=%.3f' %rmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train ens2")
        plt.legend(loc='upper right')
        plt.savefig("plots/identity_%s_ens2_train.pdf"%neuron_type)

    rmses_ens = np.zeros((n_tests))
    rmses_ens2 = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = make_normed_flipped(value=1.0, t=t_test, dt=dt, f=f, seed=100+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, w_ens=w_ens, w_ens2=w_ens2)

        a_ens = f_out1.filt(data['ens'], dt=dt)
        x = f_out.filt(data['x'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        rmse_ens = rmse(xhat_ens, x)
        a_ens2 = f_out2.filt(data['ens2'], dt=dt)
        x2 = f_out.filt(data['x2'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_out2)
        rmse_ens2 = rmse(xhat_ens2, x2)
        rmses_ens[test] = rmse_ens
        rmses_ens2[test] = rmse_ens2        
    
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, rmse=%.3f' %rmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/identity_%s_ens1_test_%s.pdf"%(neuron_type, test))

        fig, ax = plt.subplots()
        ax.plot(data['times'], x2, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens2, label='ens2, rmse=%.3f' %rmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test ens2")
        plt.legend(loc='upper right')
        plt.savefig("plots/identity_%s_ens2_test_%s.pdf"%(neuron_type, test))
        plt.close('all')

    mean_ens = np.mean(rmses_ens)
    mean_ens2 = np.mean(rmses_ens2)
    CI_ens = sns.utils.ci(rmses_ens)
    CI_ens2 = sns.utils.ci(rmses_ens2)
    
    fig, ax = plt.subplots()
    sns.barplot(data=rmses_ens2)
    ax.set(ylabel='RMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens2, CI_ens2[0], CI_ens2[1]))
    plt.xticks()
    plt.savefig("plots/identity_%s_rmse.pdf"%neuron_type)

    print('rmses: ', rmses_ens, rmses_ens2)
    print('means: ', mean_ens, mean_ens2)
    print('confidence intervals: ', CI_ens, CI_ens2)
    np.savez('data/identity_%s_results.npz'%neuron_type, rmses_ens=rmses_ens, rmses_ens2=rmses_ens2)
    return rmses_ens2

# rmses_lif = run(neuron_type=LIF())
# rmses_alif = run(neuron_type=AdaptiveLIFT())
# rmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)
# rmses_durstewitz = run(neuron_type=DurstewitzNeuron())

rmses_lif = np.load("data/identity_LIF()_results.npz")['rmses_ens2']
rmses_alif = np.load("data/identity_AdaptiveLIFT()_results.npz")['rmses_ens2']
rmses_wilson = np.load("data/identity_WilsonEuler()_results.npz")['rmses_ens2']
rmses_durstewitz = np.load("data/identity_DurstewitzNeuron()_results.npz")['rmses_ens2']

rmses = np.vstack((rmses_lif, rmses_alif, rmses_wilson, rmses_durstewitz))
nt_names =  ['LIF', 'ALIF', 'Wilson', 'Durstewitz']
fig, ax = plt.subplots()
sns.barplot(data=rmses.T)
ax.set(ylabel='RMSE')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.savefig("figures/identity_all_rmses.pdf")
