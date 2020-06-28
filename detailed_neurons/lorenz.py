import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import norm
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import entropy
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
    
def go(d_ens, f_ens, n_neurons=3000, t=100, L=False, neuron_type=LIF(),
       m=Uniform(30, 40), i=Uniform(-1, 1), r=40, IC=np.array([1,1,1]),
       seed=0, dt=0.001, dt_sample=0.001, f=DoubleExp(1e-3, 1e-1)):

    with nengo.Network(seed=seed) as model:
        # Ensembles
        u = nengo.Node(lambda t: IC*(t<=1.0))
        x = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ens = nengo.Ensemble(n_neurons, 3, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=r)
        dss = nengo.Node(DownsampleNode(size_in=n_neurons, size_out=n_neurons, dt=dt, dt_sample=dt_sample), size_in=n_neurons, size_out=n_neurons)

        # Connections
        nengo.Connection(u, x, synapse=None)
        nengo.Connection(x, x, function=feedback, synapse=~s)
        if L:
            supv = nengo.Ensemble(n_neurons, 3, neuron_type=SpikingRectifiedLinear(), radius=r, seed=seed)
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

def run(n_neurons=10000, neuron_type=LIF(), t_train=200, t=200, f=DoubleExp(1e-3, 1e-1), dt=0.001, dt_sample=0.003, tt=1.0, seed=0, smooth=30, reg=1e-1, penalty=0, df_evals=20, load_fd=False):

    d_ens = np.zeros((n_neurons, 3))
    f_ens = f

    if load_fd:
        load = np.load(load_fd)
        d_ens = load['d_ens']
        taus_ens = load['taus_ens']
        f_ens = DoubleExp(taus_ens[0], taus_ens[1])
        d_ens_gauss = load['d_ens_gauss']
    else:
        print('Optimizing ens filters and decoders')
        data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, L=True, t=t_train, f=f, dt=dt, dt_sample=dt_sample, seed=seed)

        d_ens, f_ens, taus_ens = df_opt(data['x'], data['ens'], f, df_evals=df_evals, reg=reg, penalty=penalty, dt=dt_sample, dt_sample=dt_sample, name='lorenz_%s'%neuron_type)
        all_targets_gauss = gaussian_filter1d(data['x'], sigma=smooth, axis=0)
        all_spikes_gauss = gaussian_filter1d(data['ens'], sigma=smooth, axis=0)
        d_ens_gauss = nengo.solvers.LstsqL2(reg=reg)(all_spikes_gauss, all_targets_gauss)[0]
        np.savez('data/lorenz_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_ens_gauss=d_ens_gauss)
    
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

        tar = f.filt(data['x'], dt=dt_sample)
        a_ens = f_ens.filt(data['ens'], dt=dt_sample)
        ens = np.dot(a_ens, d_ens)
        z_tar_peaks, _ = find_peaks(tar[:,2], height=0)  # gives time indices of z-component-peaks
        z_ens_peaks, _ = find_peaks(ens[:,2], height=0)

        fig = plt.figure()    
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        ax.plot(*tar.T, linewidth=0.25)
#             ax.scatter(*tar[z_tar_peaks].T, color='r', s=1)
        ax2.plot(*ens.T, linewidth=0.25)
#             ax2.scatter(*ens[z_ens_peaks].T, color='r', s=1, marker='v')
        ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-10, 30)), zlim=((0, 40)))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(False)
        ax2.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-10, 30)), zlim=((0, 40)))
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor('w')
        ax2.yaxis.pane.set_edgecolor('w')
        ax2.zaxis.pane.set_edgecolor('w')
        ax2.grid(False)
        plt.savefig("plots/lorenz_%s_train_3D.pdf"%neuron_type)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(tar[:,0], tar[:,1], linestyle="--", linewidth=0.25)
        ax2.plot(tar[:,1], tar[:,2], linestyle="--", linewidth=0.25)
        ax3.plot(tar[:,0], tar[:,2], linestyle="--", linewidth=0.25)
#             ax2.scatter(tar[z_tar_peaks, 1], tar[z_tar_peaks, 2], s=3, color='r')
#             ax3.scatter(tar[z_tar_peaks, 0], tar[z_tar_peaks, 2], s=3, color='g')
        ax1.plot(ens[:,0], ens[:,1], linewidth=0.25)
        ax2.plot(ens[:,1], ens[:,2], linewidth=0.25)
        ax3.plot(ens[:,0], ens[:,2], linewidth=0.25)
#             ax2.scatter(ens[z_ens_peaks, 1], ens[z_ens_peaks, 2], s=3, color='r', marker='v')
#             ax3.scatter(ens[z_ens_peaks, 0], ens[z_ens_peaks, 2], s=3, color='g', marker='v')
        ax1.set(xlabel='x', ylabel='y')
        ax2.set(xlabel='y', ylabel='z')
        ax3.set(xlabel='x', ylabel='z')
        plt.tight_layout()
        plt.savefig("plots/lorenz_%s_train_pairwise.pdf"%neuron_type)
        plt.close('all')

        # Plot tent map and fit the data to a gaussian
        print('Plotting tent map')
        trans = int(tt/dt)
        tar_gauss = gaussian_filter1d(data['x'][trans:], sigma=smooth, axis=0)
        a_ens_gauss = gaussian_filter1d(data['ens'][trans:], sigma=smooth, axis=0)
        ens_gauss = np.dot(a_ens_gauss, d_ens_gauss)
        z_tar_peaks = find_peaks(tar_gauss[:,2], height=0)[0][1:]
        z_tar_values_horz = np.ravel(tar_gauss[z_tar_peaks, 2][:-1])
        z_tar_values_vert = np.ravel(tar_gauss[z_tar_peaks, 2][1:])
        z_ens_peaks = find_peaks(ens_gauss[:,2], height=0)[0][1:]
        z_ens_values_horz = np.ravel(ens_gauss[z_ens_peaks, 2][:-1])
        z_ens_values_vert = np.ravel(ens_gauss[z_ens_peaks, 2][1:])
#         def gaussian(x, mu, sigma, mag):
#             return mag * np.exp(-0.5*(np.square((x-mu)/sigma)))
#         p0 = [36, 2, 40]
#         param_ens, _ = curve_fit(gaussian, z_ens_values_horz, z_ens_values_vert, p0=p0)
#         param_tar, _ = curve_fit(gaussian, z_tar_values_horz, z_tar_values_vert, p0=p0)
#         horzs_tar = np.linspace(np.min(z_tar_values_horz), np.max(z_tar_values_horz), 100)
#         gauss_tar = gaussian(horzs_tar, param_tar[0], param_tar[1], param_tar[2])
#         horzs_ens = np.linspace(np.min(z_ens_values_horz), np.max(z_ens_values_horz), 100)
#         gauss_ens = gaussian(horzs_ens, param_ens[0], param_ens[1], param_ens[2])
#         error = entropy(gauss_ens, gauss_tar)
        fig, ax = plt.subplots()
        ax.scatter(z_tar_values_horz, z_tar_values_vert, alpha=0.5, color='r', label='target')
#         ax.plot(horzs_tar, gauss_tar, color='r', linestyle='--', label='target fit')
        ax.scatter(z_ens_values_horz, z_ens_values_vert, alpha=0.5, color='b', label='ens')
#         ax.plot(horzs_ens, gauss_ens, color='b', linestyle='--', label='ens fit')
        ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$')#, title='error=%.5f'%error)
        plt.legend(loc='upper right')
        plt.savefig("plots/lorenz_%s_train_tent.pdf"%(neuron_type))        
        
    print("testing")
    data = go(d_ens, f_ens, neuron_type=neuron_type, n_neurons=n_neurons, L=False, t=t, f=f, dt=dt, dt_sample=dt_sample, seed=seed)

    tar = f.filt(data['x'], dt=dt_sample)
    a_ens = f_ens.filt(data['ens'], dt=dt_sample)
    ens = np.dot(a_ens, d_ens)
    z_tar_peaks, _ = find_peaks(tar[:,2], height=0)  # gives time indices of z-component-peaks
    z_ens_peaks, _ = find_peaks(ens[:,2], height=0)

    fig = plt.figure()    
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax.plot(*tar.T, linewidth=0.25)
#             ax.scatter(*tar[z_tar_peaks].T, color='r', s=1)
    ax2.plot(*ens.T, linewidth=0.25)
#             ax2.scatter(*ens[z_ens_peaks].T, color='r', s=1, marker='v')
    ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-10, 30)), zlim=((0, 40)))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax2.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-10, 30)), zlim=((0, 40)))
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor('w')
    ax2.yaxis.pane.set_edgecolor('w')
    ax2.zaxis.pane.set_edgecolor('w')
    ax2.grid(False)
    plt.savefig("plots/lorenz_%s_test_3D.pdf"%neuron_type)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(tar[:,0], tar[:,1], linestyle="--", linewidth=0.25)
    ax2.plot(tar[:,1], tar[:,2], linestyle="--", linewidth=0.25)
    ax3.plot(tar[:,0], tar[:,2], linestyle="--", linewidth=0.25)
#             ax2.scatter(tar[z_tar_peaks, 1], tar[z_tar_peaks, 2], s=3, color='r')
#             ax3.scatter(tar[z_tar_peaks, 0], tar[z_tar_peaks, 2], s=3, color='g')
    ax1.plot(ens[:,0], ens[:,1], linewidth=0.25)
    ax2.plot(ens[:,1], ens[:,2], linewidth=0.25)
    ax3.plot(ens[:,0], ens[:,2], linewidth=0.25)
#             ax2.scatter(ens[z_ens_peaks, 1], ens[z_ens_peaks, 2], s=3, color='r', marker='v')
#             ax3.scatter(ens[z_ens_peaks, 0], ens[z_ens_peaks, 2], s=3, color='g', marker='v')
    ax1.set(xlabel='x', ylabel='y')
    ax2.set(xlabel='y', ylabel='z')
    ax3.set(xlabel='x', ylabel='z')
    plt.tight_layout()
    plt.savefig("plots/lorenz_%s_test_pairwise.pdf"%neuron_type)
    plt.close('all')

    # Plot tent map and fit the data to a gaussian
    print('Plotting tent map')
    trans = int(tt/dt)
    tar_gauss = gaussian_filter1d(data['x'][trans:], sigma=smooth, axis=0)
    a_ens_gauss = gaussian_filter1d(data['ens'][trans:], sigma=smooth, axis=0)
    ens_gauss = np.dot(a_ens_gauss, d_ens_gauss)
    z_tar_peaks = find_peaks(tar_gauss[:,2], height=0)[0][1:]
    z_tar_values_horz = np.ravel(tar_gauss[z_tar_peaks, 2][:-1])
    z_tar_values_vert = np.ravel(tar_gauss[z_tar_peaks, 2][1:])
    z_ens_peaks = find_peaks(ens_gauss[:,2], height=0)[0][1:]
    z_ens_values_horz = np.ravel(ens_gauss[z_ens_peaks, 2][:-1])
    z_ens_values_vert = np.ravel(ens_gauss[z_ens_peaks, 2][1:])
#     def gaussian(x, mu, sigma, mag):
#         return mag * np.exp(-0.5*(np.square((x-mu)/sigma)))
#     p0 = [36, 2, 40]
#     param_ens, _ = curve_fit(gaussian, z_ens_values_horz, z_ens_values_vert, p0=p0)
#     param_tar, _ = curve_fit(gaussian, z_tar_values_horz, z_tar_values_vert, p0=p0)
#     horzs_tar = np.linspace(np.min(z_tar_values_horz), np.max(z_tar_values_horz), 100)
#     gauss_tar = gaussian(horzs_tar, param_tar[0], param_tar[1], param_tar[2])
#     horzs_ens = np.linspace(np.min(z_ens_values_horz), np.max(z_ens_values_horz), 100)
#     gauss_ens = gaussian(horzs_ens, param_ens[0], param_ens[1], param_ens[2])
#     error = entropy(gauss_ens, gauss_tar)
    fig, ax = plt.subplots()
    ax.scatter(z_tar_values_horz, z_tar_values_vert, alpha=0.5, color='r', label='target')
#     ax.plot(horzs_tar, gauss_tar, color='r', linestyle='--', label='target fit')
    ax.scatter(z_ens_values_horz, z_ens_values_vert, alpha=0.5, color='b', label='ens')
#     ax.plot(horzs_ens, gauss_ens, color='b', linestyle='--', label='ens fit')
    ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$')#, title='error=%.5f'%error)
    plt.legend(loc='upper right')
    plt.savefig("plots/lorenz_%s_test_tent.pdf"%(neuron_type))        

#     print('error: ', error)
#     return error

run(neuron_type=LIF())
# tent_error_alif = run(neuron_type=AdaptiveLIFT())
# tent_error_wilson = run(neuron_type=WilsonEuler(), dt=0.00005)

# tent_error_lif = run(neuron_type=LIF(), load_fd="data/lorenz_LIF()_fd.npz")
# tent_error_alif = run(neuron_type=AdaptiveLIFT(), load_fd="data/lorenz_AdaptiveLIFT()_fd.npz")
# tent_error_wilson = run(neuron_type=WilsonEuler(), dt=0.00005, load_fd="data/lorenz_WilsonEuler()_fd.npz")

# tent_errors = np.vstack((tent_error_lif, tent_error_alif, tent_error_wilson))
# nt_names =  ['LIF', 'ALIF', 'Wilson']
# fig, ax = plt.subplots(1, 1)
# sns.barplot(data=tent_errors.T)
# ax.set(ylabel='Tent Map Error')
# plt.xticks(np.arange(len(nt_names)), tuple(nt_names), rotation=0)
# plt.tight_layout()
# plt.savefig("plots/lorenz_all_errors.pdf")
