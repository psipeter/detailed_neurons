import numpy as np

# import h5py

from scipy.signal import argrelextrema
from scipy.linalg import norm
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter1d

import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver

from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from nengolib.synapses import ss2sim

from train import norms, downsample_spikes, DownsampleNode, df_opt, gb_opt, d_opt
from neuron_models import LIFNorm, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron

import neuron

import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(context='poster', style='white')

def go(d_supv, d_ens, f_ens, n_neurons=3000, t=30, t_supv=1, neuron_type=nengo.LIF(),
       m=Uniform(10, 20), i=Uniform(-0.7, 0.7), r=40, sigma=10, beta=8.0/3, rho=28, IC=[0,0,0],
       freq=1, seed=0, dt=0.000025, dt_sample=0.001, f=Lowpass(0.1)):

    solver_supv = NoSolver(d_supv)
    solver_ens = NoSolver(d_ens)

    def feedback(x):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta *x[2]
        return [dx, dy, dz]

    with nengo.Network(seed=seed) as model:
        # Ensembles
        u = nengo.Node(lambda t: IC*(t<=1.0))
        x = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        supv = nengo.Ensemble(n_neurons, 3, neuron_type=nengo.SpikingRectifiedLinear(), max_rates=m, intercepts=i, radius=r, seed=seed, label='supv')
        ens = nengo.Ensemble(n_neurons, 3, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=r, label='ens')
        dsn = DownsampleNode(dt=dt, dt_sample=dt_sample, size_in=ens.n_neurons, size_out=ens.n_neurons)
        spikes = nengo.Node(dsn, size_in=ens.n_neurons, size_out=ens.n_neurons)
        # Connections
        nengo.Connection(u, x, synapse=None)
        nengo.Connection(x, x, function=feedback, synapse=~s)
        nengo.Connection(x, supv, synapse=None)
        nengo.Connection(ens.neurons, spikes, synapse=None)
        ff = nengo.Connection(supv, ens, synapse=f, solver=solver_supv, label='ff', seed=seed)
        fb = nengo.Connection(ens, ens, synapse=f_ens, solver=solver_ens, label='fb', seed=seed)
        # probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
#         p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_ens = nengo.Probe(spikes, synapse=None, sample_every=dt_sample)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        sim.signals[sim.model.sig[ff]['weights']][:] = d_supv.T
        sim.signals[sim.model.sig[fb]['weights']][:] = 0
        sim.run(t_supv)
        sim.signals[sim.model.sig[ff]['weights']][:] = 0
        sim.signals[sim.model.sig[fb]['weights']][:] = d_ens.T
        if t > 0: sim.run(t)

    return dict(
        times=sim.trange(),
        x=sim.data[p_x],
        ens=sim.data[p_ens])


def run(n_neurons=2000, neuron_type=nengo.LIF(), t_train=50, t_supv=1, t=200, f=Lowpass(0.1), dt=0.001, seed=0,
        m=Uniform(20, 40), i=Uniform(-0.7, 0.7), freq=1, r=40, n_tests=1, dt_sample=0.001, smooth=60,
        df_evals=10, order=2, load_fd=False, NEF=False):

    g = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    f_ens = f
    d_ens = np.zeros((n_neurons, 3))

    print('gathering feedforward weights')
    def feedback(x):
        sigma=10
        beta=8.0/3
        rho=28
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta *x[2]
        return [dx, dy, dz]
    with nengo.Network(seed=seed) as model:
        supv = nengo.Ensemble(n_neurons, 3, neuron_type=nengo.SpikingRectifiedLinear(), max_rates=m, intercepts=i, radius=r, seed=seed)
        ens = nengo.Ensemble(n_neurons, 3, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed, radius=r)
        ff = nengo.Connection(supv, ens, synapse=f, seed=seed)
        fb = nengo.Connection(ens, ens, synapse=f, function=feedback, seed=seed)
    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
        d_supv = sim.data[ff].weights.T
        d_nef = sim.data[fb].weights.T

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        if len(load['taus_ens']) == 1:
            f_ens = Lowpass(load['taus_ens'][0])
        elif len(load['taus_ens']) == 2:
            f_ens = DoubleExp(load['taus_ens'][0], load['taus_ens'][1])
    elif NEF == True:
        d_ens = d_nef
        f_ens = f
    else:
        print("training")
        rng = np.random.RandomState(seed=0)
        IC = rng.uniform(-1, 1, size=3)
        IC /= norm(IC, 1)
        data = go(d_supv, d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC,
            t_supv=t_train, t=0, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)

        times = data['times'][::int(dt_sample/dt)]        
        target = data['x'][::int(dt_sample/dt)]
#         spk_file = h5py.File('data/lorenz_train_spk.h5', 'w')
#         spk_file.create_dataset('spk', data=data['ens'])  # , compression='gzip'
#         spk_file.close()
        A_ens = data['ens']
        del(data)
#         spk_file = h5py.File('data/lorenz_train_spk.h5', 'r')
#         A_ens = downsample_spikes(spk_file['spk'], dt=dt, dt_sample=dt_sample)
        # A_ens = downsample_spikes(data['ens'], dt=dt, dt_sample=dt_sample)
        if df_evals:
            print('optimizing filters and decoders')
            d_ens, f_ens, taus_ens = df_opt(target, A_ens, f, order=order, df_evals=df_evals, dt=dt_sample, name='lorenz_%s'%neuron_type)
        else:
            d_ens = d_opt(target, A_ens, f_ens, f, dt=dt_sample)
        np.savez('data/fd_lorenz_%s.npz'%neuron_type,
            d_ens=d_ens,
            taus_ens=taus_ens)
    
        ft = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(ft, f.impulse(len(ft), dt=0.0001), label="f")
        ax.plot(ft, f_ens.impulse(len(ft), dt=0.0001), label="f_ens, nonzero d: %s/%s"%(np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', title="%s"%neuron_type, ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/lorenz_filters_%s.png"%neuron_type)
        plt.close()
            
        target = f.filt(target, dt=dt_sample)
        xhat_ens = np.dot(f_ens.filt(A_ens, dt=dt_sample), d_ens)
        # np.savez("data/lorenz_states_train.npz", target=target, xhat_ens=xhat_ens)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, target, linestyle="--", label='target')
        ax.plot(times, xhat_ens, label='%s' %neuron_type)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train")
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_train_time_%s.png"%neuron_type)

        fig = plt.figure(figsize=(12, 8))    
        ax_ens = fig.add_subplot(221, projection='3d')
        ax_tar = fig.add_subplot(222, projection='3d')
        ax_ens.set(title="%s"%neuron_type)  # xlabel='$x$', ylabel="$y$", zlabel='$z$', 
        ax_tar.set(title="tar")  # xlabel='$x$', ylabel="$y$", zlabel='$z$', 
        ax_ens.plot(*xhat_ens.T, linewidth=0.25)
        ax_tar.plot(*target.T, linewidth=0.25)
        plt.savefig("plots/lorenz_train_3D_%s.png"%neuron_type)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=((12, 4)))
        ax1.plot(target[:,0], target[:,1], linestyle="--", linewidth=0.25, label='target')
        ax2.plot(target[:,1], target[:,2], linestyle="--", linewidth=0.25, label='target')
        ax3.plot(target[:,0], target[:,2], linestyle="--", linewidth=0.25, label='target')
        ax1.plot(xhat_ens[:,0], xhat_ens[:,1], linewidth=0.25, label='ens')
        ax2.plot(xhat_ens[:,1], xhat_ens[:,2], linewidth=0.25, label='ens')
        ax3.plot(xhat_ens[:,0], xhat_ens[:,2], linewidth=0.25, label='ens')
        ax1.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{y}$')
        ax2.set(xlabel=r'$\mathbf{y}$', ylabel=r'$\mathbf{z}$')
        ax3.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{z}$')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/lorenz_train_pairwise_%s.png"%neuron_type)

        z_target = gaussian_filter1d(target[:, 2], sigma=smooth)
        z_ens = gaussian_filter1d(xhat_ens[:, 2], sigma=smooth)
        z_target_maxima = z_target[argrelextrema(z_target, np.greater)]
        z_ens_maxima = z_ens[argrelextrema(z_ens, np.greater)]
        z_target_ratios = z_target_maxima[1:] / z_target_maxima[:-1]
        z_ens_ratios = z_ens_maxima[1:] / z_ens_maxima[:-1]
        bins = np.linspace(0.8, 1.2, 20)
        z_target_bins = np.histogram(z_target_ratios, bins=bins, density=True)[0]
        z_ens_bins = np.histogram(z_ens_ratios, bins=bins, density=True)[0]
        error = nrmse(z_ens_bins, target=z_target_bins)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.hist(z_target_ratios, bins=bins, density=True, alpha=0.75, label="target")
        ax.hist(z_ens_ratios, bins=bins, density=True, alpha=0.75, label="ens")
        # min_len = np.min([len(z_ens_ratios), len(z_target_ratios)])
        # error = entropy(z_target_ratios[:min_len], z_ens_ratios[:min_len])
        # error = nrmse(z_target_ratios[:min_len], target=z_ens_ratios[:min_len])
        # sns.distplot(z_target_ratios, label='target', ax=ax)
        # sns.distplot(z_ens_ratios, label='ens', ax=ax)
        ax.set(xlabel=r'$\frac{\mathrm{max}_n (z)}{\mathrm{max}_{n+1} (z)}$', ylabel='freq', title='nrmse=%.3f'%error)
        plt.legend()
        plt.savefig("plots/lorenz_train_dist_%s.png"%neuron_type)

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.scatter(z_target_maxima[:-1], z_target_maxima[1:], label='target')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='nrmse = %.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_train_tent_%s.png"%neuron_type)

    nrmses = np.zeros((n_tests))
    for n in range(n_tests):
        print("test #%s"%n)
        rng = np.random.RandomState(seed=n)
        IC = rng.uniform(-1, 1, size=3)
        IC /= norm(IC, 1)
        data = go(d_supv, d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC,
            t_supv=t_supv, t=t, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)
        times = data['times'][::int(dt_sample/dt)]        
        target = f.filt(data['x'][::int(dt_sample/dt)], dt=dt_sample)
#         spk_file = h5py.File('data/lorenz_test_spk.h5', 'w')
#         spk_file.create_dataset('spk', data=data['ens'])  # , compression='gzip'
#         spk_file.close()
        A_ens = f_ens.filt(data['ens'], dt=dt_sample)
        del(data)
#         spk_file = h5py.File('data/lorenz_test_spk.h5', 'r')
#         A_ens = f_ens.filt(downsample_spikes(spk_file['spk'], dt=dt, dt_sample=dt_sample), dt=dt_sample)
        xhat_ens = np.dot(A_ens, d_ens)
#         spk_file.close()

        z_target = gaussian_filter1d(target[:, 2], sigma=smooth)
        z_ens = gaussian_filter1d(xhat_ens[:, 2], sigma=smooth)
        z_target_maxima = z_target[argrelextrema(z_target, np.greater)]
        z_ens_maxima = z_ens[argrelextrema(z_ens, np.greater)]
        z_target_ratios = z_target_maxima[1:] / z_target_maxima[:-1]
        z_ens_ratios = z_ens_maxima[1:] / z_ens_maxima[:-1]
        bins = np.linspace(0.8, 1.2, 20)
        z_target_bins = np.histogram(z_target_ratios, bins=bins, density=True)[0]
        z_ens_bins = np.histogram(z_ens_ratios, bins=bins, density=True)[0]
        error = nrmse(z_ens_bins, target=z_target_bins)
        nrmses[n] = error

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.hist(z_target_ratios, bins=bins, density=True, alpha=0.75, label="target")
        ax.hist(z_ens_ratios, bins=bins, density=True, alpha=0.75, label="ens")
        ax.set(xlabel=r'$\frac{\mathrm{max}_n (z)}{\mathrm{max}_{n+1} (z)}$', ylabel='freq', title='nrmse = %.5f'%error)
        plt.legend()
        plt.savefig("plots/lorenz_test_%s_dist_%s.png"%(n, neuron_type))

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.scatter(z_target_maxima[:-1], z_target_maxima[1:], label='target')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='nrmse = %.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_test_%s_tent_%s.png"%(n, neuron_type))

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.plot(times, target, linestyle="--", label='target')
        ax.plot(times, xhat_ens, label='%s' %neuron_type)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train")
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_test_%s_time_%s.png"%(n, neuron_type))
        plt.close()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=((12, 4)))
        ax1.plot(target[:,0], target[:,1], linestyle="--", linewidth=0.25, label='target')
        ax2.plot(target[:,1], target[:,2], linestyle="--", linewidth=0.25, label='target')
        ax3.plot(target[:,0], target[:,2], linestyle="--", linewidth=0.25, label='target')
        ax1.plot(xhat_ens[:,0], xhat_ens[:,1], linewidth=0.25, label='ens')
        ax2.plot(xhat_ens[:,1], xhat_ens[:,2], linewidth=0.25, label='ens')
        ax3.plot(xhat_ens[:,0], xhat_ens[:,2], linewidth=0.25, label='ens')
        ax1.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{y}$')
        ax2.set(xlabel=r'$\mathbf{y}$', ylabel=r'$\mathbf{z}$')
        ax3.set(xlabel=r'$\mathbf{x}$', ylabel=r'$\mathbf{z}$')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/lorenz_test_%s_pairwise_%s.png"%(n, neuron_type))
        plt.close()

        fig = plt.figure(figsize=(12, 8))    
        ax_ens = fig.add_subplot(221, projection='3d')
        ax_tar = fig.add_subplot(222, projection='3d')
        ax_ens.set(title="%s"%neuron_type)  # xlabel='$x$', ylabel="$y$", zlabel='$z$', 
        ax_tar.set(title="tar")  # xlabel='$x$', ylabel="$y$", zlabel='$z$', 
        ax_ens.plot(*xhat_ens.T, linewidth=0.25)
        ax_tar.plot(*target.T, linewidth=0.25)
        plt.savefig("plots/lorenz_test_%s_3D_%s.png"%(n, neuron_type))
        plt.close()
        
        del(A_ens)
        del(xhat_ens)
        del(target)
        del(times)

    print('nrmses: ', nrmses)
    print('mean: ', np.mean(nrmses))
    print('confidence intervals: ', sns.utils.ci(nrmses))
    np.savez('data/nrmses_lorenz_%s.npz'%neuron_type, nrmses=nrmses, mean=np.mean(nrmses), CI=sns.utils.ci(nrmses))

run(neuron_type=nengo.LIF(), n_neurons=5000, n_tests=1, dt_sample=0.001, dt=0.0001, t_train=100, t=100)

# run(neuron_type=nengo.LIF(), n_neurons=6000, n_tests=1, load_fd="data/fd_lorenz_LIF().npz")

# run(neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1), n_neurons=5000, n_tests=1)


