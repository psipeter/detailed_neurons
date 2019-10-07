import numpy as np

# import h5py

from scipy.signal import argrelextrema
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

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

def go(d_ens, f_ens, n_neurons=5000, t=30, t_supv=1, neuron_type=nengo.LIF(),
       m=Uniform(10, 20), i=Uniform(-0.7, 0.7), r=40, IC=[0,0,0],
       seed=0, dt=0.000025, dt_sample=0.001, f=Lowpass(0.1)):

    solver_ens = NoSolver(d_ens)

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
        ff = nengo.Connection(supv, ens, synapse=f, label='ff', seed=seed)
        fb = nengo.Connection(ens, ens, synapse=f_ens, solver=solver_ens, label='fb', seed=seed)
        # probes
        p_u = nengo.Probe(u, synapse=None)
        p_x = nengo.Probe(x, synapse=None)
#         p_ens = nengo.Probe(ens.neurons, synapse=None)
        p_ens = nengo.Probe(spikes, synapse=None, sample_every=dt_sample)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        sim.signals[sim.model.sig[fb]['weights']][:] = 0
        sim.run(t_supv)
        sim.signals[sim.model.sig[ff]['weights']][:] = 0
        sim.signals[sim.model.sig[fb]['weights']][:] = d_ens.T
        if t > 0: sim.run(t)

    return dict(
        times=sim.trange(),
        x=sim.data[p_x],
        ens=sim.data[p_ens])

def feedback(x):
    sigma = 10
    beta = 8.0/3
    rho = 28
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta *x[2]
    return [dx, dy, dz]

def feedback_NEF(x):
    tau = 0.1
    sigma = 10
    beta = 8.0/3
    rho = 28
    dx0 = sigma * (x[1] - x[0])
    dx1 = x[0] * (rho - x[2]) - x[1]
    dx2 = x[0] * x[1] - beta *x[2]
    return [dx0 * tau + x[0], dx1 * tau + x[1], dx2 * tau + x[2]]

def mountain(x, a, b, c):
    return (a*x + b)*(x<=c) + (-a*x + 2*a*c + b)*(x>c)

def run(n_neurons=6000, neuron_type=nengo.LIF(), t_train=50, t_supv=1, t=200, f=Lowpass(0.1), dt=0.001, seed=0,
        m=Uniform(20, 40), i=Uniform(-0.7, 0.7), r=40, n_tests=1, dt_sample=0.001, smooth=70,
        df_evals=10, order=2, load_fd=False, NEF=False):

    g = 2e-3 * np.ones((n_neurons, 1))
    b = np.zeros((n_neurons, 1))
    f_ens = f
    d_ens = np.zeros((n_neurons, 3))

    print('gathering feedforward weights')
    with nengo.Network(seed=seed) as model:
        ens = nengo.Ensemble(n_neurons, 3, max_rates=m, intercepts=i, neuron_type=nengo.SpikingRectifiedLinear(), seed=seed, radius=r)
        fb = nengo.Connection(ens, ens, synapse=f, function=feedback_NEF, seed=seed)
    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
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
        data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
            t_supv=t_train, t=0, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)

        times = data['times'][::int(dt_sample/dt)]
        target = data['x'][::int(dt_sample/dt)]
        A_ens = data['ens']
        del(data)
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
        p0 = [1, 0, 31]
        bounds = ((0, -30, 25), (2, 30, 45))
        p_best_target, _ = curve_fit(mountain, z_target_maxima[:-1], z_target_maxima[1:], p0=p0, bounds=bounds)
        p_best_ens, _ = curve_fit(mountain, z_ens_maxima[:-1], z_ens_maxima[1:], p0=p0, bounds=bounds)
        error = np.abs(p_best_target[0] - p_best_ens[0])+np.abs(p_best_target[2] - p_best_ens[2])/10

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.scatter(z_target_maxima[:-1], z_target_maxima[1:], label='target')
        ax.scatter(z_target_maxima[:-1], mountain(z_target_maxima[:-1], *p_best_target), label='target fit')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.scatter(z_ens_maxima[:-1], mountain(z_ens_maxima[:-1], *p_best_ens), label='ens fit')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='error = %.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_train_tent_%s.png"%neuron_type)

        # fig, ax = plt.subplots(figsize=((12, 8)))
        # ax.bar(bins, means_target, alpha=0.75, width=bins[1]-bins[0], label="target")
        # ax.bar(bins, means_ens, alpha=0.75, width=bins[1]-bins[0], label="ens")
        # ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='nrmse = %.5f'%error)
        # ax.legend()
        # plt.savefig("plots/lorenz_train_dist_%s.png"%neuron_type)

    nrmses = np.zeros(n_tests)
    for n in range(n_tests):
        print("test #%s"%n)
        rng = np.random.RandomState(seed=n)
        IC = rng.uniform(-1, 1, size=3)
        IC /= norm(IC, 1)
        data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
            t_supv=t_supv, t=t, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)
        times = data['times'][::int(dt_sample/dt)]        
        target = f.filt(data['x'][::int(dt_sample/dt)], dt=dt_sample)
        A_ens = f_ens.filt(data['ens'], dt=dt_sample)
        del(data)
        xhat_ens = np.dot(A_ens, d_ens)

        z_target = gaussian_filter1d(target[:, 2], sigma=smooth)
        z_ens = gaussian_filter1d(xhat_ens[:, 2], sigma=smooth)
        z_target_maxima = z_target[argrelextrema(z_target, np.greater)]
        z_ens_maxima = z_ens[argrelextrema(z_ens, np.greater)]
        p0 = [1, 0, 31]
        bounds = ((0, -30, 25), (2, 30, 45))
        p_best_target, _ = curve_fit(mountain, z_target_maxima[:-1], z_target_maxima[1:], p0=p0, bounds=bounds)
        p_best_ens, _ = curve_fit(mountain, z_ens_maxima[:-1], z_ens_maxima[1:], p0=p0, bounds=bounds)
        error = np.abs(p_best_target[0] - p_best_ens[0])+np.abs(p_best_target[2] - p_best_ens[2])/10
        nrmses[n] = error

        fig, ax = plt.subplots(figsize=((12, 8)))
        ax.scatter(z_target_maxima[:-1], z_target_maxima[1:], label='target')
        ax.scatter(z_target_maxima[:-1], mountain(z_target_maxima[:-1], *p_best_target), label='target fit')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.scatter(z_ens_maxima[:-1], mountain(z_ens_maxima[:-1], *p_best_ens), label='ens fit')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='error = %.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_test_tent_%s.png"%neuron_type)

        # fig, ax = plt.subplots(figsize=((12, 8)))
        # ax.bar(bins, means_target, alpha=0.75, width=bins[1]-bins[0], label="target")
        # ax.bar(bins, means_ens, alpha=0.75, width=bins[1]-bins[0], label="ens")
        # ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='nrmse = %.5f'%error)
        # ax.legend()
        # plt.savefig("plots/lorenz_test_dist_%s.png"%neuron_type)

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
        
        np.savez("data/xhat_lorenz_%s.npz"%neuron_type, xhat_ens=xhat_ens, target=target)
        del(A_ens)
        del(xhat_ens)
        del(target)
        del(times)

    print('nrmses: ', nrmses)
    print('mean: ', np.mean(nrmses))
    print('confidence intervals: ', sns.utils.ci(nrmses))
    np.savez('data/nrmses_lorenz_%s.npz'%neuron_type, nrmses=nrmses, mean=np.mean(nrmses), CI=sns.utils.ci(nrmses))

run(neuron_type=nengo.SpikingRectifiedLinear(), r=40, smooth=90, NEF=True)

# run(neuron_type=nengo.LIF())#, load_fd="data/fd_lorenz_LIF().npz")

# run(neuron_type=AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1))#, load_fd="data/fd_lorenz_AdaptiveLIFT().npz")

# run(neuron_type=WilsonEuler(), dt=0.001)#, load_fd="data/fd_lorenz_WilsonEuler().npz")
