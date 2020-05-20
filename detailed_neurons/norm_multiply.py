import numpy as np
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import d_opt, df_opt, LearningNode
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, WilsonRungeKutta, DurstewitzNeuron, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def multiply(x):
    return x[0]*x[1]

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 0.8), seed=1, dt=0.001,
        f=Lowpass(0.01), f_smooth=Lowpass(0.1), stim_func1=lambda t: np.sin(t), stim_func2=lambda t: np.sin(t),
        neuron_type=LIF(), w_ens=None, w_ens2=None, learn=False, learn2=False, tau_decay=20, half=False):

    neuron_type2 = LIF() if half else neuron_type

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func1)
        u2 = nengo.Node(stim_func2)

        # Ensembles
        pre = nengo.Ensemble(n_neurons, 2, radius=2, seed=seed)
        ens = nengo.Ensemble(n_neurons, 2, radius=2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(30, 1, max_rates=m, intercepts=i, neuron_type=neuron_type2, seed=seed+1)
        supv = nengo.Ensemble(n_neurons, 2, radius=2, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
        supv2 = nengo.Ensemble(30, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed+1)
        x = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        x2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        x2_supv = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre[0], synapse=None, seed=seed)
        nengo.Connection(u2, pre[1], synapse=None, seed=seed)
        nengo.Connection(u, x[0], synapse=f, seed=seed)
        nengo.Connection(u2, x[1], synapse=f, seed=seed)
        conn = nengo.Connection(pre, ens, synapse=f, seed=seed, label='pre_ens')
        nengo.Connection(x, supv, synapse=None, seed=seed)
        # nengo.Connection(x2, supv2, synapse=None, seed=seed)
        nengo.Connection(x2_supv, supv2, synapse=None, seed=seed)
        nengo.Connection(x, x2, synapse=f, function=multiply, seed=seed+1)
        nengo.Connection(x, x2_supv, synapse=f_ens, function=multiply, seed=seed+1)
        if isinstance(neuron_type, DurstewitzNeuron): # todo: fix nosolver bug
            conn2 = nengo.Connection(ens[0], ens2, synapse=f_ens, seed=seed+1, solver=NoSolver(d_ens), label='ens-ens2')
        else:
            conn2 = nengo.Connection(ens.neurons, ens2, synapse=f_ens, seed=seed+1, transform=d_ens.T, label='ens-ens2')

        if learn and isinstance(neuron_type, DurstewitzNeuron):
            node = LearningNode(n_neurons, pre.n_neurons, 2, conn, k=1e-5)
            nengo.Connection(pre.neurons, node[0:pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons:pre.n_neurons+n_neurons], synapse=f_smooth)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f_smooth)
            nengo.Connection(x, node[-2:], synapse=None)

        if learn2 and isinstance(neuron_type, DurstewitzNeuron):
            node2 = LearningNode(30, n_neurons, 1, conn2, k=1e-5)
            nengo.Connection(ens.neurons, node2[0:n_neurons], synapse=f_smooth) 
            nengo.Connection(ens2.neurons, node2[n_neurons:n_neurons+30], synapse=f_smooth)
            nengo.Connection(supv2.neurons, node2[n_neurons+30: n_neurons+60], synapse=f_smooth)
            nengo.Connection(x2, node2[-1], synapse=None)

        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_u2 = nengo.Probe(u2, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)
        p_supv2 = nengo.Probe(supv2.neurons, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
        p_x2 = nengo.Probe(x2, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ens):
            for pre in range(pre.n_neurons):
                for post in range(n_neurons):
                    conn.weights[pre, post] = w_ens[pre, post]
                    conn.netcons[pre, post].weight[0] = np.abs(w_ens[pre, post])
                    conn.netcons[pre, post].syn().e = 0 if w_ens[pre, post] > 0 else -70
        if np.any(w_ens2):
            for pre in range(n_neurons):
                for post in range(30):
                    conn2.weights[pre, post] = w_ens2[pre, post]
                    conn2.netcons[pre, post].weight[0] = np.abs(w_ens2[pre, post])
                    conn2.netcons[pre, post].syn().e = 0 if w_ens2[pre, post] > 0 else -70
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 

    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        u2=sim.data[p_u2],
        ens=sim.data[p_ens],
        ens2=sim.data[p_ens2],
        x=sim.data[p_x],
        x2=sim.data[p_x2],
        supv=sim.data[p_supv],
        supv2=sim.data[p_supv2],
        enc=sim.data[supv].encoders,
        enc2=sim.data[supv2].encoders,
        w_ens=conn.weights if isinstance(neuron_type, DurstewitzNeuron) and hasattr(conn, 'weights') else None,
        w_ens2=conn2.weights if isinstance(neuron_type, DurstewitzNeuron) and hasattr(conn2, 'weights') else None)


def run(n_neurons=60, t=10, t_test=30, n_trains=6, n_encodes=20,
        f=DoubleExp(1e-3, 3e-2), f_out=DoubleExp(1e-3, 1e-1), f_smooth=DoubleExp(1e-2, 2e-1),
        dt=0.001, n_tests=10, neuron_type=LIF(), reg=1e-2, penalty=0.5, load_w=None, load_df=None):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    w_ens = None
    w_ens2 = None
    print('\nNeuron Type: %s'%neuron_type)

    print('Creating input signal from concatenated, normalized, flipped white noise')
    value = 1.2
    u_length = int(t*(n_trains+2*n_encodes)/dt)+1 if isinstance(neuron_type, DurstewitzNeuron) else int(t*n_trains/dt)+1
    n_trains_total = n_trains+2*n_encodes if isinstance(neuron_type, DurstewitzNeuron) else n_trains
    u1_list = np.zeros((u_length, 1))
    u2_list = np.zeros((u_length, 1))
    for n in range(n_trains_total):
        stim_func1 = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.6, seed=0+n)
        stim_func2 = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.6, seed=100+n)
        with nengo.Network() as model:
            u1 = nengo.Node(stim_func1)
            u2 = nengo.Node(stim_func2)
            p1 = nengo.Probe(u1, synapse=None)
            p2 = nengo.Probe(u2, synapse=None)
        with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
            sim.run(t/2, progress_bar=False)
        x = f.filt(sim.data[p1] * sim.data[p2], dt=dt)
        norm = value / np.max(np.abs(x))
        u1n = sim.data[p1]*np.sqrt(norm)
        u2n = sim.data[p2]*np.sqrt(norm)
        u1_list[n*int(t/dt): (n+1)*int(t/dt)] = np.vstack((u1n, -u1n))
        u2_list[n*int(t/dt): (n+1)*int(t/dt)] = np.vstack((u2n, u2n))
    stim_func1 = lambda t: u1_list[int(t/dt)]
    stim_func2 = lambda t: u2_list[int(t/dt)]

    if isinstance(neuron_type, DurstewitzNeuron):
        if load_w:
            w_ens = np.load(load_w)['w_ens']
        else:
            print('Optimizing ens1 encoders')
            # stim_func = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=0)
            # stim_func2 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=1)
            data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_encodes, f=f, dt=0.001, stim_func1=stim_func1, stim_func2=stim_func2,
                f_smooth=f_smooth, neuron_type=neuron_type, w_ens=w_ens, w_ens2=w_ens2, learn=True, half=True)
            w_ens = data['w_ens']            
            fig, ax = plt.subplots()
            sns.distplot(w_ens.ravel(), ax=ax, bins=np.arange(-0.001, 0.001, 0.0001))
            ax.set(xlabel='weights', ylabel='frequency')
            plt.savefig("plots/multiply_%s_w_ens.pdf"%neuron_type)
            np.savez('data/multiply_w.npz', w_ens=w_ens)
            a_ens = f_smooth.filt(data['ens'], dt=0.001)
            a_supv = f_smooth.filt(data['supv'], dt=0.001)
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(2, 1)
                ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv')
                ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens')
                ax.set(ylim=((0, 40)))
                ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv')
                ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens')
                ax.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/multiply_ens1_activity_%s.pdf'%n)
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
        # stim_func = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=0)
        # stim_func2 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=1)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type,
            stim_func1=stim_func1, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2, half=True)
        d_ens, f_ens, taus_ens = df_opt(data['x'][:,0]*data['x'][:,1], data['ens'], f, dt=dt, penalty=penalty, reg=reg, name='multiply_%s'%neuron_type)
        d_ens = d_ens.reshape((n_neurons, 1))
        d_out1, f_out1, taus_out1 = df_opt(data['x'], data['ens'], f_out, dt=dt, name='multiply_%s'%neuron_type)
        np.savez('data/multiply_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out1=d_out1, taus_out1=taus_out1)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/multiply_%s_filters_ens.pdf"%neuron_type)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau=%.3f, \tau_2=%.3f$" %(-1./f_out.poles[0], -1./f_out.poles[1]))
        ax.plot(times, f_out1.impulse(len(times), dt=0.0001), label=r"$f^{out1}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_out1.poles[0], -1./f_out1.poles[1], np.count_nonzero(d_out1), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/multiply_%s_filters_out1.pdf"%neuron_type)

        fig, ax = plt.subplots()
        sns.distplot(d_ens.ravel())
        ax.set(xlabel='decoders', ylabel='frequency')
        plt.savefig("plots/multiply_%s_d_ens.pdf"%neuron_type)

        fig, ax = plt.subplots()
        sns.distplot(d_out1[:,0].ravel())
        sns.distplot(d_out1[:,1].ravel())
        ax.set(xlabel='decoders', ylabel='frequency')
        plt.savefig("plots/multiply_%s_d_out1.pdf"%neuron_type)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        x = f.filt(data['x'][:,0]*data['x'][:,1], dt=dt).ravel()
        xhat_ens = np.dot(a_ens, d_ens).ravel()
        nrmse_ens = nrmse(xhat_ens, target=x)
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_%s_ens1_train.pdf"%neuron_type)

        a_ens = f_out1.filt(data['ens'], dt=dt)
        x_out = f_out.filt(data['x'], dt=dt)
        xhat_ens_out = np.dot(a_ens, d_out1)
        nrmse_ens_out1 = nrmse(xhat_ens_out[:,0], target=x_out[:,0])
        nrmse_ens_out2 = nrmse(xhat_ens_out[:,1], target=x_out[:,1])
        fig, ax = plt.subplots()
        ax.plot(data['times'], x_out[:,0], linestyle="--", label='x_0')
        ax.plot(data['times'], x_out[:,1], linestyle="--", label='x_1')
        ax.plot(data['times'], xhat_ens_out[:,0], label='ens_0, nrmse=%.3f' %nrmse_ens_out1)
        ax.plot(data['times'], xhat_ens_out[:,1], label='ens_1, nrmse=%.3f' %nrmse_ens_out2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_%s_ens1_out_train.pdf"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron):
#         if load_w:
#             w_ens2 = np.load(load_w)['w_ens2']
#         else:
        print('Optimizing ens2 encoders')
        # stim_func = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=0)
        # stim_func2 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=1)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*2*n_encodes, f=f, f_smooth=f_smooth, neuron_type=neuron_type,
            stim_func1=stim_func1, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2, learn2=True)
        w_ens2 = data['w_ens2']
        fig, ax = plt.subplots()
        sns.distplot(w_ens2.ravel(), ax=ax, bins=np.arange(-0.001, 0.001, 0.0001))
        ax.set(xlabel='weights', ylabel='frequency')
        plt.savefig("plots/multiply_%s_w_ens2.pdf"%neuron_type)
        # np.savez('data/multiply_w.npz', w_ens=w_ens, w_ens2=w_ens2)
        a_ens = f_smooth.filt(data['ens2'], dt=0.001)
        a_supv = f_smooth.filt(data['supv2'], dt=0.001)
        for n in range(30):
            fig, (ax, ax2) = plt.subplots(2, 1)
            ax.plot(data['times'][:20000], a_supv[:,n][:20000], alpha=0.5, label='supv2')
            ax.plot(data['times'][:20000], a_ens[:,n][:20000], alpha=0.5, label='ens2')
            ax.set(ylim=((0, 40)))
            ax2.plot(data['times'][-20000:], a_supv[:,n][-20000:], alpha=0.5, label='supv2')
            ax2.plot(data['times'][-20000:], a_ens[:,n][-20000:], alpha=0.5, label='ens2')
            ax.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
            ax2.set(xlabel='time', ylabel='Firing Rate', ylim=((0, 40)))
            plt.legend()
            plt.savefig('plots/tuning/multiply_ens2_activity_%s.pdf'%n)
            plt.close('all')

#     if load_df:
#         load = np.load(load_df)
#         d_out2 = load['d_out2']
#         taus_out2 = load['taus_out2']
#         f_out2 = DoubleExp(taus_out2[0], taus_out2[1])
#     else:
    print('Optimizing ens2 filters and decoders')
    # stim_func = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=0)
    # stim_func2 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=1)
    data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type,
        stim_func1=stim_func1, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2)
    d_out2, f_out2, taus_out2  = df_opt(data['x2'], data['ens2'], f_out, dt=dt, name='multiply_%s'%neuron_type)
    np.savez('data/multiply_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out1=d_out1, taus_out1=taus_out1, d_out2=d_out2, taus_out2=taus_out2)

    times = np.arange(0, 1, 0.0001)
    fig, ax = plt.subplots()
    ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau=%.3f, \tau_2=%.3f$" %(-1./f_out.poles[0], -1./f_out.poles[1]))
    ax.plot(times, f_out2.impulse(len(times), dt=0.0001), label=r"$f^{out2}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
       %(-1./f_out2.poles[0], -1./f_out2.poles[1], np.count_nonzero(d_out2), 30))
    ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("plots/multiply_%s_filters_out2.pdf"%neuron_type)

    fig, ax = plt.subplots()
    sns.distplot(d_out2.ravel())
    ax.set(xlabel='decoders', ylabel='frequency')
    plt.savefig("plots/multiply_%s_d_out2.pdf"%neuron_type)

    a_ens2 = f_out2.filt(data['ens2'], dt=dt)
    x2 = f_out.filt(data['x2'], dt=dt)
    xhat_ens2 = np.dot(a_ens2, d_out2)
    nrmse_ens2 = nrmse(xhat_ens2, target=x2)

    fig, ax = plt.subplots()
    ax.plot(data['times'], x2, linestyle="--", label='x')
    ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
    ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train ens2")
    plt.legend(loc='upper right')
    plt.savefig("plots/multiply_%s_ens2_train.pdf"%neuron_type)


    nrmses_ens = np.zeros((n_tests))
    nrmses_ens_out = np.zeros((n_tests))
    nrmses_ens2 = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)

        print('Creating test signal from concatenated, normalized, flipped white noise')
        value = 1.0
        stim_func1 = nengo.processes.WhiteSignal(period=t_test/2, high=1, rms=0.6, seed=300+test)
        stim_func2 = nengo.processes.WhiteSignal(period=t_test/2, high=1, rms=0.6, seed=400+test)
        with nengo.Network() as model:
            u1 = nengo.Node(stim_func1)
            u2 = nengo.Node(stim_func2)
            p1 = nengo.Probe(u1, synapse=None)
            p2 = nengo.Probe(u2, synapse=None)
        with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
            sim.run(t_test/2, progress_bar=False)
        x = f.filt(sim.data[p1] * sim.data[p2], dt=dt)
        norm = value / np.max(np.abs(x))
        u1n = sim.data[p1]*np.sqrt(norm)
        u2n = sim.data[p2]*np.sqrt(norm)
        stim_func1 = lambda t: np.vstack((u1n, -u1n))[int(t/dt)-1]
        stim_func2 = lambda t: np.vstack((u2n, u2n))[int(t/dt)-1]

        # stim_func1 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=100+test)
        # stim_func2 = nengo.processes.WhiteSignal(period=1000, high=1, rms=0.6, seed=200+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type,
            stim_func1=stim_func1, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        x = f.filt(data['x'][:,0]*data['x'][:,1], dt=dt).ravel()
        xhat_ens = np.dot(a_ens, d_ens).ravel()
        nrmse_ens = nrmse(xhat_ens, target=x)

        a_ens_out = f_out1.filt(data['ens'], dt=dt)
        x_out = f_out.filt(data['x'], dt=dt)
        xhat_ens_out = np.dot(a_ens, d_out1)
        nrmse_ens_out = nrmse(xhat_ens_out, target=x_out)
        nrmse_ens_out1 = nrmse(xhat_ens_out[:,0], target=x_out[:,0])
        nrmse_ens_out2 = nrmse(xhat_ens_out[:,1], target=x_out[:,1])

        a_ens2 = f_out2.filt(data['ens2'], dt=dt)
        x2 = f_out.filt(data['x2'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_out2)
        nrmse_ens2 = nrmse(xhat_ens2, target=x2)
        nrmses_ens[test] = nrmse_ens
        nrmses_ens_out[test] = nrmse_ens_out
        nrmses_ens2[test] = nrmse_ens2        
    
        fig, ax = plt.subplots()
        ax.plot(data['times'], x_out[:,0], linestyle="--", label='x_0')
        ax.plot(data['times'], x_out[:,1], linestyle="--", label='x_1')
        ax.plot(data['times'], xhat_ens_out[:,0], label='ens_0, nrmse=%.3f' %nrmse_ens_out1)
        ax.plot(data['times'], xhat_ens_out[:,1], label='ens_1, nrmse=%.3f' %nrmse_ens_out2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test ens1 out")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_%s_ens1_out_test_%s.pdf"%(neuron_type, test))
        
        fig, ax = plt.subplots()
        ax.plot(data['times'], x, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_%s_ens1_test_%s.pdf"%(neuron_type, test))
        
        fig, ax = plt.subplots()
        ax.plot(data['times'], x2, linestyle="--", label='x')
        ax.plot(data['times'], xhat_ens2, label='ens2, nrmse=%.3f' %nrmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test ens2")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_%s_ens2_test_%s.pdf"%(neuron_type, test))
        plt.close('all')

    mean_ens = np.mean(nrmses_ens)
    mean_ens_out = np.mean(nrmses_ens_out)
    mean_ens2 = np.mean(nrmses_ens2)
    CI_ens = sns.utils.ci(nrmses_ens)
    CI_ens_out = sns.utils.ci(nrmses_ens_out)
    CI_ens2 = sns.utils.ci(nrmses_ens2)
    
    fig, ax = plt.subplots()
    sns.barplot(data=nrmses_ens2)
    ax.set(ylabel='NRMSE', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens2, CI_ens2[0], CI_ens2[1]))
    plt.xticks()
    plt.savefig("plots/multiply_%s_nrmse.pdf"%neuron_type)

    print('nrmses: ', nrmses_ens, nrmses_ens_out, nrmses_ens2)
    print('means: ', mean_ens, mean_ens_out, mean_ens2)
    print('confidence intervals: ', CI_ens, CI_ens_out, CI_ens2)
    np.savez('data/multiply_%s_results.npz'%neuron_type, nrmses_ens=nrmses_ens, nrmses_ens_out=nrmses_ens_out, nrmses_ens2=nrmses_ens2)
    return nrmses_ens2

# nrmses_lif = run(neuron_type=LIF())
# nrmses_alif = run(neuron_type=AdaptiveLIFT())
# nrmses_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)
# nrmses_durstewitz = run(neuron_type=DurstewitzNeuron())
# load_w="data/multiply_w.npz", load_df="data/multiply_DurstewitzNeuron()_df.npz")

nrmses_lif = np.load("data/multiply_LIF()_results.npz")['nrmses_ens2']
nrmses_alif = np.load("data/multiply_AdaptiveLIFT()_results.npz")['nrmses_ens2']
nrmses_wilson = np.load("data/multiply_WilsonEuler()_results.npz")['nrmses_ens2']
nrmses_durstewitz = np.load("data/multiply_DurstewitzNeuron()_results.npz")['nrmses_ens2']

nrmses = np.vstack((nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
nt_names =  ['LIF', 'ALIF', 'Wilson', 'Durstewitz']
fig, ax = plt.subplots()
sns.barplot(data=nrmses.T)
ax.set(ylabel='NRMSE')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.savefig("figures/multiply_all_nrmses.pdf")