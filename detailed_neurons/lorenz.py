import numpy as np
from scipy.signal import argrelextrema
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import nengo
from nengo import SpikingRectifiedLinear
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from nengolib.synapses import ss2sim
from train import norms, df_opt
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron
import neuron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(context='paper', style='white')


class DownsampleNode(object):
    def __init__(self, size_in, size_out, dt, dt_sample):
        self.dt = dt
        self.dt_sample = dt_sample
        self.size_in = size_in
        self.size_out = size_out
        self.ratio = int(self.dt_sample/self.dt)
        self.output = np.zeros((int(self.size_in)))
        self.count = 0

    def __call__(self, t, x):
        if self.count == self.ratio:
            self.output *= 0
            self.count = 0
        self.output += x / self.ratio
        self.count += 1
        return self.output
    
def go(d_ens, f_ens, n_neurons=5000, t=50, learn=False, neuron_type=LIF(),
       m=Uniform(20, 40), i=Uniform(-1, 0.8), r=40, IC=[0,0,0],
       seed=0, dt=0.001, dt_sample=0.001, f=Lowpass(0.1)):

    with nengo.Network(seed=seed) as model:
        # Ensembles
        u = nengo.Node(lambda t: IC*(t<=1.0))
        x = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ens = nengo.Ensemble(n_neurons, 3, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=r)
        dss = nengo.Node(DownsampleNode(size_in=n_neurons, size_out=n_neurons, dt=dt, dt_sample=dt_sample), size_in=n_neurons, size_out=n_neurons)

        # Connections
        nengo.Connection(u, x, synapse=None)
        nengo.Connection(x, x, function=feedback, synapse=~s)
        if learn:
            supv = nengo.Ensemble(n_neurons, 3, neuron_type=SpikingRectifiedLinear(), max_rates=m, intercepts=i, radius=r, seed=seed)
            nengo.Connection(x, supv, synapse=None)
            nengo.Connection(supv, ens, synapse=f, seed=seed)
        else:
            nengo.Connection(ens, ens, synapse=f_ens, solver=NoSolver(d_ens), seed=seed)

        # Probes
        nengo.Connection(ens.neurons, dss, synapse=None)
        p_x = nengo.Probe(x, synapse=None, sample_every=dt_sample)
        p_ens = nengo.Probe(dss, synapse=None, sample_every=dt_sample)

    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        sim.run(t)

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

def mountain(x, a, b, c):
    return (a*x + b)*(x<=c) + (-a*x + 2*a*c + b)*(x>c)

def run(n_neurons=5000, neuron_type=LIF(), t_train=50, t=50, f=DoubleExp(1e-3, 1e-1), dt=0.001, dt_sample=0.001, seed=0,
        m=Uniform(20, 40), i=Uniform(-1, 0.8), r=40, n_tests=3, smooth=100, reg=1e-1, penalty=0, df_evals=20, load_fd=False):

    d_ens = np.zeros((n_neurons, 3))
    f_ens = f

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
    else:
        print('Optimizing ens filters and decoders')
        rng = np.random.RandomState(seed=0)
        IC = rng.uniform(-1, 1, size=3)
        IC /= norm(IC, 1)
        data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
            learn=True, t=t_train, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)

        d_ens, f_ens, taus_ens = df_opt(data['x'], data['ens'], f, df_evals=df_evals, reg=reg, penalty=penalty, dt=dt_sample,
            name='lorenz_%s'%neuron_type)
        np.savez('data/lorenz_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)
    
        f_times = np.arange(0, 1, 0.0001)
        fig, ax = plt.subplots()
        ax.plot(f_times, f.impulse(len(f_times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$"
            %(-1./f.poles[0], -1./f.poles[1]))
        ax.plot(f_times, f_ens.impulse(len(f_times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
           %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
        ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("plots/lorenz_%s_filters_ens.pdf"%neuron_type)

        x = f.filt(data['x'], dt=dt_sample)
        a_ens = f_ens.filt(data['ens'], dt=dt_sample)
        xhat_ens = np.dot(a_ens, d_ens)
        fig = plt.figure()    
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*xhat_ens.T, linewidth=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(False)
        plt.savefig("plots/lorenz_%s_train_3D.pdf"%neuron_type)

        # fig = plt.figure()    
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(*x.T, linewidth=0.25)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # ax.xaxis.pane.set_edgecolor('w')
        # ax.yaxis.pane.set_edgecolor('w')
        # ax.zaxis.pane.set_edgecolor('w')
        # ax.grid(False)
        # plt.savefig("plots/lorenz_%s_train_3D_target.pdf"%neuron_type)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(x[:,0], x[:,1], linestyle="--", linewidth=0.25)
        ax2.plot(x[:,1], x[:,2], linestyle="--", linewidth=0.25)
        ax3.plot(x[:,0], x[:,2], linestyle="--", linewidth=0.25)
        ax1.plot(xhat_ens[:,0], xhat_ens[:,1], linewidth=0.25)
        ax2.plot(xhat_ens[:,1], xhat_ens[:,2], linewidth=0.25)
        ax3.plot(xhat_ens[:,0], xhat_ens[:,2], linewidth=0.25)
        ax1.set(xlabel='x', ylabel='y')
        ax2.set(xlabel='y', ylabel='z')
        ax3.set(xlabel='x', ylabel='z')
        plt.tight_layout()
        plt.savefig("plots/lorenz_%s_train_pairwise.pdf"%neuron_type)

        z_x = gaussian_filter1d(x[:, 2], sigma=smooth)
        z_ens = gaussian_filter1d(xhat_ens[:, 2], sigma=smooth)
        z_x_maxima = z_x[argrelextrema(z_x, np.greater)]
        z_ens_maxima = z_ens[argrelextrema(z_ens, np.greater)]
        p0 = [1, 0, 31]
        bounds = ((0, -30, 25), (2, 30, 45))
        param_x, _ = curve_fit(mountain, z_x_maxima[:-1], z_x_maxima[1:], p0=p0, bounds=bounds)
        param_ens, _ = curve_fit(mountain, z_ens_maxima[:-1], z_ens_maxima[1:], p0=p0, bounds=bounds)
        error = nrmse(np.array([param_ens[0], param_ens[2]]), target=np.array([param_x[0], param_x[2]]))
        # error = np.abs(p_best_x[0] - p_best_ens[0])+np.abs(p_best_x[2] - p_best_ens[2])/10

        fig, ax = plt.subplots()
        ax.scatter(z_x_maxima[:-1], z_x_maxima[1:], label='target')
        ax.plot(np.sort(z_x_maxima[:-1]), mountain(np.sort(z_x_maxima[:-1]), *param_x), label='target fit')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.plot(np.sort(z_ens_maxima[:-1]), mountain(np.sort(z_ens_maxima[:-1]), *param_ens), label='ens fit')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='tent_error=%.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_%s_train_tent.pdf"%neuron_type)

    tent_errors = np.zeros(n_tests)
    for n in range(n_tests):
        print("test #%s"%n)
        rng = np.random.RandomState(seed=n)
        IC = rng.uniform(-1, 1, size=3)
        IC /= norm(IC, 1)
        data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
            learn=False, t=t, f=f, m=m, i=i, dt=dt, dt_sample=dt_sample, seed=seed)

        # reduce data arrays
        x = f.filt(data['x'], dt=dt_sample)
        a_ens = f_ens.filt(data['ens'], dt=dt_sample)
        xhat_ens = np.dot(a_ens, d_ens)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(x[:,0], x[:,1], linestyle="--", linewidth=0.25)
        ax2.plot(x[:,1], x[:,2], linestyle="--", linewidth=0.25)
        ax3.plot(x[:,0], x[:,2], linestyle="--", linewidth=0.25)
        ax1.plot(xhat_ens[:,0], xhat_ens[:,1], linewidth=0.25)
        ax2.plot(xhat_ens[:,1], xhat_ens[:,2], linewidth=0.25)
        ax3.plot(xhat_ens[:,0], xhat_ens[:,2], linewidth=0.25)
        ax1.set(xlabel='x', ylabel='y')
        ax2.set(xlabel='y', ylabel='z')
        ax3.set(xlabel='x', ylabel='z')
        plt.tight_layout()
        plt.savefig("plots/lorenz_%s_pairwise_test_%s.pdf"%(neuron_type, n))
        plt.close()

        a_ens = f_ens.filt(data['ens'], dt=dt_sample)
        xhat_ens = np.dot(a_ens, d_ens)
        fig = plt.figure()    
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*xhat_ens.T, linewidth=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(False)
        plt.savefig("plots/lorenz_%s_test_%s_3D.pdf"%(n, neuron_type))

        z_x = gaussian_filter1d(x[:, 2], sigma=smooth)
        z_ens = gaussian_filter1d(xhat_ens[:, 2], sigma=smooth)
        z_x_maxima = z_x[argrelextrema(z_x, np.greater)]
        z_ens_maxima = z_ens[argrelextrema(z_ens, np.greater)]
        p0 = [1, 0, 31]
        bounds = ((0, -30, 25), (2, 30, 45))
        param_x, _ = curve_fit(mountain, z_x_maxima[:-1], z_x_maxima[1:], p0=p0, bounds=bounds)
        param_ens, _ = curve_fit(mountain, z_ens_maxima[:-1], z_ens_maxima[1:], p0=p0, bounds=bounds)
        error = nrmse(np.array([param_ens[0], param_ens[2]]), target=np.array([param_x[0], param_x[2]]))
        # error = np.abs(p_best_x[0] - p_best_ens[0])+np.abs(p_best_x[2] - p_best_ens[2])/10
        tent_errors[n] = error

        fig, ax = plt.subplots()
        ax.scatter(z_x_maxima[:-1], z_x_maxima[1:], label='target')
        ax.plot(np.sort(z_x_maxima[:-1]), mountain(np.sort(z_x_maxima[:-1]), *param_x), label='target fit')
        ax.scatter(z_ens_maxima[:-1], z_ens_maxima[1:], label='ens')
        ax.plot(np.sort(z_ens_maxima[:-1]), mountain(np.sort(z_ens_maxima[:-1]), *param_ens), label='ens fit')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='tent_error = %.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_%s_tent_test_%s.pdf"%(neuron_type, n))
        np.savez("data/lorenz_%s_test.npz"%neuron_type, xhat_ens=xhat_ens, x=x)

    mean_ens = np.mean(tent_errors)
    CI_ens = sns.utils.ci(tent_errors)
    print('tent_errors: ', tent_errors)
    print('mean: ', mean_ens)
    print('confidence intervals: ', CI_ens)
    fig, ax = plt.subplots()
    sns.barplot(data=tent_errors)
    ax.set(ylabel='Tent Errors', title="mean=%.3f, CI=%.3f-%.3f"%(mean_ens, CI_ens[0], CI_ens[1]))
    plt.xticks()
    plt.savefig("plots/lorenz_%s_tent_errors.pdf"%neuron_type)
    np.savez('data/lorenz_%s_results.npz'%neuron_type, tent_errors=tent_errors)
    return tent_errors


tent_error_lif = run(neuron_type=LIF())
tent_error_alif = run(neuron_type=AdaptiveLIFT())
tent_error_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)

tent_errors = np.vstack((nrmses_lif, nrmses_alif, nrmses_wilson))
nt_names =  ['LIF', 'ALIF', 'Wilson']
fig, ax = plt.subplots(1, 1)
sns.barplot(data=tent_errors.T)
ax.set(ylabel='Tent Map Error')
plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
plt.tight_layout()
plt.savefig("plots/lorenz_all_errors.pdf")



# def compare_x(n_neurons=100, neuron_type=LIF(), t_train=30, t=50, f=DoubleExp(1e-3, 1e-1), seed=0,m=Uniform(20, 40), i=Uniform(-1, 0.8), r=40):

#     d_ens = np.zeros((n_neurons, 3))
#     f_ens = f
#     rng = np.random.RandomState(seed=0)
#     IC = rng.uniform(-1, 1, size=3)
#     IC /= norm(IC, 1)

#     data1 = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
#         learn=True, t=t_train, f=f, m=m, i=i, dt=0.001, dt_sample=0.001, seed=seed)
#     data2 = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, IC=IC, r=r,
#         learn=True, t=t_train, f=f, m=m, i=i, dt=0.0001, dt_sample=0.001, seed=seed)
#     x1 = data1['x']
#     x2 = data2['x']

#     fig = plt.figure()    
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(*x1.T, linewidth=0.25)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     ax.xaxis.pane.set_edgecolor('w')
#     ax.yaxis.pane.set_edgecolor('w')
#     ax.zaxis.pane.set_edgecolor('w')
#     ax.grid(False)
#     plt.savefig("plots/lorenz_%s_train_3D_x1.pdf"%neuron_type)

#     fig = plt.figure()    
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(*x2.T, linewidth=0.25)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False
#     ax.xaxis.pane.set_edgecolor('w')
#     ax.yaxis.pane.set_edgecolor('w')
#     ax.zaxis.pane.set_edgecolor('w')
#     ax.grid(False)
#     plt.savefig("plots/lorenz_%s_train_3D_x2.pdf"%neuron_type)

#     print(nrmse(x1, target=x2))