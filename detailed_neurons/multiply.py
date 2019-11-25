import numpy as np

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem

from train2 import df_opt, LearningNode, tuning_curve
from neuron_models2 import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def go(d_ens, f_ens, n_neurons=100, t=10, m=Uniform(20, 40), i=Uniform(-1, 1), seed=0, dt=0.001, f=Lowpass(0.01), fs=Lowpass(0.1),
        neuron_type=nengo.LIF(), w_ens=None, w_ens2=None, learn=False, learn2=False, half=False, stim_func=lambda t: np.sin(t), stim_func2=lambda t: np.sin(t)):

    def multiply(x):
        return x[0]*x[1]

    neuron_type2 = nengo.LIF() if half else neuron_type

    with nengo.Network(seed=seed) as model:

        # Stimulus and Nodes
        u = nengo.Node(stim_func)
        u2 = nengo.Node(stim_func2)

        # Ensembles
        pre = nengo.Ensemble(300, 2, max_rates=m, seed=seed)
        ens = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=neuron_type2, seed=seed+1)
        tar = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tar_mult = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

        # Connections
        nengo.Connection(u, pre[0], synapse=None, seed=seed)
        nengo.Connection(u2, pre[1], synapse=None, seed=seed)
        nengo.Connection(u, tar[0], synapse=f, seed=seed)
        nengo.Connection(u2, tar[1], synapse=f, seed=seed)
        conn = nengo.Connection(pre, ens, synapse=f, seed=seed, label='pre_ens')
        nengo.Connection(tar, tar_mult, synapse=None, function=multiply, seed=seed+1)
        nengo.Connection(tar, tar2, synapse=f, function=multiply, seed=seed+1)
        if isinstance(neuron_type, DurstewitzNeuron): # todo: fix nosolver bug
            conn2 = nengo.Connection(ens[0], ens2, synapse=f_ens, seed=seed+1, solver=NoSolver(d_ens), label='ens-ens2')
        else:
            conn2 = nengo.Connection(ens.neurons, ens2, synapse=f_ens, seed=seed+1, transform=d_ens.T, label='ens-ens2')

        supv = nengo.Ensemble(n_neurons, 2, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(), seed=seed)
        nengo.Connection(tar, supv, synapse=None, seed=seed)
        p_supv = nengo.Probe(supv.neurons, synapse=None)

        if learn and isinstance(neuron_type, DurstewitzNeuron):
            node = LearningNode(n_neurons, pre.n_neurons, 2, conn, k=1e-2)
            nengo.Connection(pre.neurons, node[0:pre.n_neurons], synapse=f)
            nengo.Connection(ens.neurons, node[pre.n_neurons:pre.n_neurons+n_neurons], synapse=f)
            nengo.Connection(supv.neurons, node[pre.n_neurons+n_neurons: pre.n_neurons+2*n_neurons], synapse=f)
            nengo.Connection(tar, node[-2:], synapse=None)

        supv2 = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=AdaptiveLIFT(), seed=seed+1)
        nengo.Connection(tar_mult, supv2, synapse=f_ens, seed=seed)
        p_supv2 = nengo.Probe(supv2.neurons, synapse=None)

        if learn2 and isinstance(neuron_type, DurstewitzNeuron):
            # nengo.Connection(tar2, supv2, synapse=None, seed=seed)
            node2 = LearningNode(n_neurons, n_neurons, 1, conn2, k=1e-4)
            nengo.Connection(ens.neurons, node2[0:n_neurons], synapse=f_ens)
            nengo.Connection(ens2.neurons, node2[n_neurons:2*n_neurons], synapse=f)
            nengo.Connection(supv2.neurons, node2[2*n_neurons: 3*n_neurons], synapse=f)
            nengo.Connection(tar2, node2[-1], synapse=None)
            
        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_u2 = nengo.Probe(u2, synapse=None)
        p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_ens2 = nengo.Probe(ens2.neurons, synapse=None)
        p_tar = nengo.Probe(tar, synapse=None)
        p_tar2 = nengo.Probe(tar2, synapse=None)
        p_tar_mult = nengo.Probe(tar_mult, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        if np.any(w_ens):
            for pre in range(pre.n_neurons):
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
        neuron.h.init()
        sim.run(t)
        reset_neuron(sim, model) 
        
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        u2=sim.data[p_u2],
        ens=sim.data[p_ens],
        ens2=sim.data[p_ens2],
        tar=sim.data[p_tar],
        tar2=sim.data[p_tar2],
        tar_mult=sim.data[p_tar_mult],
        supv=sim.data[p_supv],
        enc=sim.data[supv].encoders,
        supv2=sim.data[p_supv2],
        enc2=sim.data[supv2].encoders,
        w_ens=conn.weights if isinstance(neuron_type, DurstewitzNeuron) and hasattr(conn, 'weights') else None,
        w_ens2=conn2.weights if isinstance(neuron_type, DurstewitzNeuron) and hasattr(conn2, 'weights') else None)


def run(n_neurons=100, t=10, t_test=10, t_enc=100, f=Lowpass(0.01), dt=0.001, n_tests=10, neuron_type=nengo.LIF(),
        load_w=None, load_fd=None, learn_w=False, learn_w2=False):

    d_ens = np.zeros((n_neurons, 1))
    f_ens = f
    fs = Lowpass(0.1)
    w_ens = np.load(load_w)['w_ens'] if load_w else None
    w_ens2 = np.load(load_w)['w_ens2'] if load_w else None
#     w_ens2 = None
    stim_func = nengo.processes.WhiteSignal(period=t_enc, high=1, rms=0.5, seed=0)
    stim_func2 = nengo.processes.WhiteSignal(period=t_enc, high=1, rms=0.5, seed=1)

    if isinstance(neuron_type, DurstewitzNeuron) and learn_w:
        print('optimizing encoders into DurstewitzNeuron ens')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_enc, f=f, dt=dt, stim_func=stim_func, stim_func2=stim_func2,
            neuron_type=neuron_type, w_ens=w_ens, w_ens2=w_ens2, learn=True, half=True)
        w_ens = data['w_ens']
        np.savez('data/w_multiply.npz', w_ens=w_ens)

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = Lowpass(taus_ens[0]) if len(taus_ens) == 1 else DoubleExp(taus_ens[0], taus_ens[1])
        d_ens_out = load['d_ens_out']
        taus_ens_out = load['taus_ens_out']
        f_ens_out = Lowpass(taus_ens_out[0]) if len(taus_ens_out) == 1 else DoubleExp(taus_ens_out[0], taus_ens_out[1])
    else:
        print('gathering filter/decoder training data for ens1')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type,
            stim_func=stim_func, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2, half=True)
        
        fs = Lowpass(0.1)
        a_durstewitz = fs.filt(data['ens'], dt=0.001)
        a_supv = fs.filt(data['supv'], dt=0.001)
        u = fs.filt(data['tar'], dt=0.001)
        enc = data['enc']
        xbins = 40
        xmin = -1
        xmax = 1
        for n in range(n_neurons):
            xdote = np.dot(u, enc[n])
            xdote_bins, a_bins_ens = tuning_curve(a_durstewitz[:,n], xdote, xbins, xmin, xmax)
            xdote_bins, a_bins_supv = tuning_curve(a_supv[:,n], xdote, xbins, xmin, xmax)
            CIs_ens = np.zeros((xbins, 2))
            CIs_supv = np.zeros((xbins, 2))
            for x in range(xbins):
                if len(a_bins_ens[x]) > 0:
                    CIs_ens[x] = sns.utils.ci(np.array(a_bins_ens[x]), which=95)
                    CIs_supv[x] = sns.utils.ci(np.array(a_bins_supv[x]), which=95)
            hz_mins_ens = np.zeros((xbins))
            hz_maxs_ens = np.zeros((xbins))
            hz_mins_supv = np.zeros((xbins))
            hz_maxs_supv = np.zeros((xbins))
            means_ens = np.zeros((xbins))
            means_supv = np.zeros((xbins))
            for x in range(xbins):
                hz_mins_ens[x] = CIs_ens[x, 0]  # np.min(a_bins_ens[x])
                hz_maxs_ens[x] = CIs_ens[x, 1]  # np.max(a_bins_ens[x])
                hz_mins_supv[x] = CIs_supv[x, 0]  # np.min(a_bins_supv[x])
                hz_maxs_supv[x] = CIs_supv[x, 1]  # np.max(a_bins_supv[x])
                means_ens[x] = np.mean(a_bins_ens[x])
                means_supv[x] = np.mean(a_bins_supv[x])
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.plot(xdote_bins, means_ens, label='ens')  # np.sign(enc[n])*
            ax.plot(xdote_bins, means_supv, label='supv')  # np.sign(enc[n])*
            ax.fill_between(xdote_bins, hz_mins_ens, hz_maxs_ens, alpha=0.25, label='ens')  # np.sign(enc[n])*
            ax.fill_between(xdote_bins, hz_mins_supv, hz_maxs_supv, alpha=0.25, label='supv')  # np.sign(enc[n])*
            ax.set(xlim=((xmin, xmax)), ylim=((0, 60)), xlabel=r"$\mathbf{x}$", ylabel='a (Hz)')
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/tuning/multiply_ens1_state_%s.png"%n)
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
            ax.plot(data['times'], a_durstewitz[:,n], alpha=0.5, label='ens')
            ax.set(xlabel='time', ylim=((0, 60)))
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/tuning/multiply_ens1_time_%s.png'%n)
            plt.close()
            
        if np.any(w_ens):
            fig, ax = plt.subplots()
            sns.distplot(w_ens.ravel())
            plt.savefig("plots/w_ens_multiply.png")
        
        print('optimizing filters and decoders')
        d_ens_out, f_ens_out, taus_ens_out = df_opt(data['tar'], data['ens'], f, order=2, dt=dt, name='multiply_readout_%s'%neuron_type)
        d_ens, f_ens, taus_ens = df_opt(data['tar_mult'], data['ens'], f, order=2, dt=dt, name='multiply_%s'%neuron_type)
        d_ens = d_ens.reshape((n_neurons, 1))
        np.savez('data/fd_multiply_%s.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_ens_out=d_ens_out, taus_ens_out=taus_ens_out)
        

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens_out.impulse(len(times), dt=0.0001), label="f_ens_out, nonzero d: %s/%s"%(np.count_nonzero(d_ens_out), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/multiply_filters_readout_%s.png"%neuron_type)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label="f_ens, nonzero d: %s/%s"%(np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/multiply_filters_%s.png"%neuron_type)

        a_ens_out = f_ens_out.filt(data['ens'], dt=dt)
        target_out = f.filt(data['tar'], dt=dt)
        xhat_ens_out = np.dot(a_ens_out, d_ens_out)
        nrmse_ens_out = nrmse(xhat_ens_out, target=target_out)

        a_ens = f_ens.filt(data['ens'], dt=dt)
        target = f.filt(data['tar_mult'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        nrmse_ens = nrmse(xhat_ens, target=target)

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(data['times'], target_out, alpha=0.5, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens_out, alpha=0.5, label='ens, nrmse=%.3f' %nrmse_ens_out)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_train_ens1_readout_%s.png"%neuron_type)

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(data['times'], target, alpha=0.5, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens, alpha=0.5, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_train_ens1_%s.png"%neuron_type)

    if isinstance(neuron_type, DurstewitzNeuron) and learn_w2:
        print('optimizing encoders into DurstewitzNeuron ens2')
        stim_func = nengo.processes.WhiteSignal(period=t_enc, high=1, rms=0.5, seed=0)
        stim_func2 = nengo.processes.WhiteSignal(period=t_enc, high=1, rms=0.5, seed=1)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_enc, f=f, dt=dt, neuron_type=neuron_type,
            stim_func=stim_func, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2, learn2=True)
        w_ens2 = data['w_ens2']
        np.savez('data/w_multiply.npz', w_ens=w_ens, w_ens2=w_ens2)

        fs = Lowpass(0.1)
        a_durstewitz = fs.filt(data['ens2'], dt=0.001)
        a_supv = fs.filt(data['supv2'], dt=0.001)
        u = fs.filt(data['tar2'], dt=0.001)
        enc = data['enc2']
        xbins = 40
        xmin = -1
        xmax = 1
        for n in range(n_neurons):
            xdote = np.dot(u, enc[n])
            xdote_bins, a_bins_ens = tuning_curve(a_durstewitz[:,n], xdote, xbins, xmin, xmax)
            xdote_bins, a_bins_supv = tuning_curve(a_supv[:,n], xdote, xbins, xmin, xmax)
            CIs_ens = np.zeros((xbins, 2))
            CIs_supv = np.zeros((xbins, 2))
            for x in range(xbins):
                if len(a_bins_ens[x]) > 0:
                    CIs_ens[x] = sns.utils.ci(np.array(a_bins_ens[x]), which=95)
                    CIs_supv[x] = sns.utils.ci(np.array(a_bins_supv[x]), which=95)
            hz_mins_ens = np.zeros((xbins))
            hz_maxs_ens = np.zeros((xbins))
            hz_mins_supv = np.zeros((xbins))
            hz_maxs_supv = np.zeros((xbins))
            means_ens = np.zeros((xbins))
            means_supv = np.zeros((xbins))
            for x in range(xbins):
                hz_mins_ens[x] = CIs_ens[x, 0]  # np.min(a_bins_ens[x])
                hz_maxs_ens[x] = CIs_ens[x, 1]  # np.max(a_bins_ens[x])
                hz_mins_supv[x] = CIs_supv[x, 0]  # np.min(a_bins_supv[x])
                hz_maxs_supv[x] = CIs_supv[x, 1]  # np.max(a_bins_supv[x])
                means_ens[x] = np.mean(a_bins_ens[x])
                means_supv[x] = np.mean(a_bins_supv[x])
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.plot(xdote_bins, means_ens, label='ens2')  # np.sign(enc[n])*
            ax.plot(xdote_bins, means_supv, label='supv2')  # np.sign(enc[n])*
            ax.fill_between(xdote_bins, hz_mins_ens, hz_maxs_ens, alpha=0.25, label='ens2')  # np.sign(enc[n])*
            ax.fill_between(xdote_bins, hz_mins_supv, hz_maxs_supv, alpha=0.25, label='supv2')  # np.sign(enc[n])*
            ax.set(xlim=((xmin, xmax)), ylim=((0, 60)), xlabel=r"$\mathbf{x}$", ylabel='a (Hz)')
            plt.legend()
            plt.tight_layout()
            plt.savefig("plots/tuning/multiply_ens2_state_%s.png"%n)
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.plot(data['times'], a_supv[:,n], alpha=0.5, label='supv')
            ax.plot(data['times'], a_durstewitz[:,n], alpha=0.5, label='ens')
            ax.set(xlabel='time', ylim=((0, 60)))
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/tuning/multiply_ens2_time_%s.png'%n)
            plt.close()

        if np.any(w_ens2):
            fig, ax = plt.subplots()
            sns.distplot(w_ens2.ravel())
            plt.savefig("plots/w_ens2_multiply.png")

    if load_fd:
        load = np.load(load_fd)
        d_ens2 = load['d_ens2']
        taus_ens2 = load['taus_ens2']
        f_ens2 = Lowpass(taus_ens2[0]) if len(taus_ens2) == 1 else DoubleExp(taus_ens2[0], taus_ens2[1])
    else:
        print('gathering filter/decoder training data for ens2')
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2)
        print('optimizing filters and decoders for ens2')
        d_ens2, f_ens2, taus_ens2  = df_opt(data['tar2'], data['ens2'], f, order=2, dt=dt, name='multiply_%s'%neuron_type)
        np.savez('data/fd_multiply_%s.npz'%neuron_type,
            d_ens=d_ens, taus_ens=taus_ens, d_ens_out=d_ens_out, taus_ens_out=taus_ens_out, d_ens2=d_ens2, taus_ens2=taus_ens2)

        times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(times, f.impulse(len(times), dt=0.0001), label="f")
        ax.plot(times, f_ens2.impulse(len(times), dt=0.0001), label="f_ens2, nonzero d2: %s/%s"%(np.count_nonzero(d_ens2), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/multiply_filters_ens2_%s.png"%neuron_type)

        a_ens2 = f_ens2.filt(data['ens2'], dt=dt)
        target2 = f.filt(data['tar2'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_ens2)
        nrmse_ens2 = nrmse(xhat_ens2, target=target2)

        fig, ax = plt.subplots(figsize=((6, 6)))
        ax.plot(data['times'], target2, alpha=0.5, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens2, alpha=0.5, label='ens2, nrmse=%.3f' %nrmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train2")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_train_ens2_%s.png"%neuron_type)

    nrmses_ens = np.zeros((n_tests))
    nrmses_ens_out = np.zeros((n_tests))
    nrmses_ens2 = np.zeros((n_tests))
    for test in range(n_tests):
        print('test %s' %test)
        stim_func = nengo.processes.WhiteSignal(period=t_test, high=1, rms=0.5, seed=100+test)
        stim_func2 = nengo.processes.WhiteSignal(period=t_test, high=1, rms=0.5, seed=200+test)
        data = go(d_ens, f_ens, n_neurons=n_neurons, t=t_test, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, stim_func2=stim_func2, w_ens=w_ens, w_ens2=w_ens2)

        a_ens_out = f_ens_out.filt(data['ens'], dt=dt)
        target_out = f.filt(data['tar'], dt=dt)
        xhat_ens_out = np.dot(a_ens_out, d_ens_out)
        nrmse_ens_out = nrmse(xhat_ens_out, target=target_out)
        a_ens = f_ens.filt(data['ens'], dt=dt)
        target = f.filt(data['tar_mult'], dt=dt)
        xhat_ens = np.dot(a_ens, d_ens)
        nrmse_ens = nrmse(xhat_ens, target=target)
        a_ens2 = f_ens2.filt(data['ens2'], dt=dt)
        target2 = f.filt(data['tar2'], dt=dt)
        xhat_ens2 = np.dot(a_ens2, d_ens2)
        nrmse_ens2 = nrmse(xhat_ens2, target=target2)
        nrmses_ens[test] = nrmse_ens
        nrmses_ens_out[test] = nrmse_ens_out
        nrmses_ens2[test] = nrmse_ens2    

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(data['times'], target_out, alpha=0.5, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens_out, alpha=0.5, label='ens, nrmse=%.3f' %nrmse_ens_out)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test_out")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_ens1_readout_test%s_%s.png"%(test, neuron_type))

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(data['times'], target, alpha=0.5, linestyle="--", label='test1')
        ax.plot(data['times'], xhat_ens, alpha=0.5, label='ens, nrmse=%.3f' %nrmse_ens)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ens1")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_ens1_test%s_%s.png"%(test, neuron_type))

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.plot(data['times'], target2, alpha=0.5, linestyle="--", label='target')
        ax.plot(data['times'], xhat_ens2, alpha=0.5, label='ens2, nrmse=%.3f' %nrmse_ens2)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ens2")
        plt.legend(loc='upper right')
        plt.savefig("plots/multiply_ens2_test%s_%s.png"%(test, neuron_type))
        plt.close()

    mean_ens_out = np.mean(nrmses_ens_out)
    mean_ens = np.mean(nrmses_ens)
    mean_ens2 = np.mean(nrmses_ens2)
    CI_ens_out = sns.utils.ci(nrmses_ens_out)
    CI_ens = sns.utils.ci(nrmses_ens)
    CI_ens2 = sns.utils.ci(nrmses_ens2)

    print('nrmses: ', nrmse_ens, nrmses_ens2)
    print('means: ', mean_ens, mean_ens2)
    print('confidence intervals: ', CI_ens, CI_ens2)
    np.savez('data/results_multiply_%s.npz'%neuron_type,
        nrmses_ens_out=nrmses_ens_out,
        nrmses_ens=nrmses_ens,
        nrmses_ens2=nrmses_ens2)
    return nrmses_ens2

nrmses_durstewitz = run(n_neurons=30, t=60, t_enc=300, n_tests=10, neuron_type=DurstewitzNeuron(), learn_w=True, learn_w2=True)
nrmses_lif = run(n_neurons=30, t=60, n_tests=10, neuron_type=nengo.LIF())
nrmses_alif = run(n_neurons=30, t=60, n_tests=10, neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1))
nrmses_wilson = run(n_neurons=30, t=60, n_tests=10, dt=0.000025, neuron_type=WilsonEuler())
#, load_fd="data/fd_multiply_DurstewitzNeuron().npz")

nrmses = np.vstack((nrmses_lif, nrmses_alif, nrmses_wilson, nrmses_durstewitz))
nt_names =  ['LIF', 'ALIF', 'Wilson', 'Durstewitz']
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.barplot(data=nrmses.T)
ax.set(ylabel='NRMSE')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.tight_layout()
plt.savefig("plots/multiply_nrmses.png")
