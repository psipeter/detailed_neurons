import numpy as np

from scipy.linalg import block_diag
from scipy.integrate import ode
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm

import nengo
from nengo.utils.matplotlib import rasterplot
from nengo.params import Default, NumberParam
from nengo.dists import Uniform
from nengo.neurons import *
from nengo.builder.neurons import *
from nengo.dists import Uniform
from nengo.solvers import LstsqL2, NoSolver
from nengo.base import ObjView
from nengo.builder import Builder, Operator, Signal
from nengo.exceptions import BuildError
from nengo.builder.connection import build_decoders, BuiltConnection
from nengo.utils.builder import full_transform

from nengolib.signal import s, z, nrmse, LinearSystem
from nengolib import Lowpass, DoubleExp
from nengolib.synapses import ss2sim

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import os
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

from neuron_models import AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron
# from utils import bin_activities_values_single

import neuron
neuron.h.load_file('NEURON/durstewitz.hoc')
neuron.h.load_file('stdrun.hoc')


__all__ = ['norms', 'df_opt', 'gb_opt', 'gb_opt2', 'd_opt']

def norms(t, dt=0.001, stim_func=lambda t: np.cos(t), f=None, value=1.0):
    with nengo.Network() as model:
        u = nengo.Node(stim_func)
        p = nengo.Probe(u, synapse=f)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    norm = value / np.max(np.abs(sim.data[p]))
    return norm

def d_opt(target, spikes, h, h_tar, reg=1e-1, dt=0.001):
    target = h_tar.filt(target, dt=dt)
    A = h.filt(spikes, dt=dt)
    d_new = LstsqL2(reg=reg)(A, target)[0]
    return d_new

def df_opt(target, spikes, f, name='default', order=1, df_evals=100, seed=0, dt=0.001,
        tau_mins=[1e-3, 1e-4], tau_maxs=[5e-1, 1e-3], reg=1e-1):

    target = f.filt(target, dt=dt)
    np.savez_compressed('data/%s_spk.npz'%name, spikes=spikes)
    np.savez_compressed('data/%s_tar.npz'%name, target=target)
    del(spikes)
    del(target)
    
    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['order'] = order
    hyperparams['dt'] = dt
    for o in range(order):
        hyperparams[str(o)] = hp.uniform(str(o), tau_mins[o], tau_maxs[0])
        hyperparams[str(o)] = hp.loguniform(str(o), np.log10(tau_mins[o]), np.log10(tau_maxs[0]))
    hyperparams['reg'] = reg

    def objective(hyperparams):
        if hyperparams['order'] == 1:
            taus = [hyperparams['0']]
            h = Lowpass(taus[0])
        elif hyperparams['order'] == 2:
            taus = [hyperparams['0'], hyperparams['1']]
            h = DoubleExp(taus[0], taus[1])
        reg = hyperparams['reg']
        spk = np.load('data/%s_spk.npz'%hyperparams['name'])['spikes']
        act = h.filt(spk, dt=hyperparams['dt'])
        tar = np.load('data/%s_tar.npz'%hyperparams['name'])['target']
        d = LstsqL2(reg=reg)(act, tar)[0]
        xhat = np.dot(act, d)
        loss = nrmse(xhat, target=tar)
        return {'loss': loss, 'taus': taus, 'reg': reg, 'status': STATUS_OK }
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=df_evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]

    if order == 1:
        h_new = Lowpass(best['result']['taus'][0])
    elif order == 2:
        h_new = DoubleExp(best['result']['taus'][0], best['result']['taus'][1])
    d_new = LstsqL2(reg=best['result']['reg'])(
        h_new.filt(np.load('data/%s_spk.npz'%name)['spikes'], dt=dt),
        np.load('data/%s_tar.npz'%name)['target'])[0]
        
    return d_new, h_new


def tuning_curve(a_ens, xdote, xbins, xmin=-1, xmax=1):
    # xdote_bins = np.linspace(np.min(xdote), np.max(xdote), num=xbins)
    xdote_bins = np.linspace(xmin, xmax, num=xbins)
    a_bins = [[] for _ in range(xbins)]
    def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return idx
    for t in range(a_ens.shape[0]):
        x = find_nearest(xdote_bins, xdote[t])
        a_bins[x].append(a_ens[t])
    return xdote_bins, a_bins


def gb_opt(ens, tar, u, enc, g, b, f=Lowpass(0.1), t_transient=0.1, dt=0.001,
        pt=True, xbins=40, ybins=10, ymax=60, CI=95, name="default", xmin=-1.2, xmax=1.2,
        delta_g=2e-5, delta_b=2e-5, tol=0.2, thr=3, loss_scale_x=2, loss_scale_y=0.05):

    n_neurons = ens.shape[1]
    a_ens = f.filt(ens, dt=dt)[int(t_transient/dt):]
    a_tar = f.filt(tar, dt=dt)[int(t_transient/dt):]
    u = u[int(t_transient/dt):]
    losses = np.zeros((n_neurons))
    for n in range(n_neurons):
        xdote = np.dot(u, enc[n])
        xdote_bins, a_bins_ens = tuning_curve(a_ens[:,n], xdote, xbins, xmin, xmax)
        xdote_bins, a_bins_tar = tuning_curve(a_tar[:,n], xdote, xbins, xmin, xmax)
        CIs_ens = np.zeros((xbins, 2))
        CIs_tar = np.zeros((xbins, 2))
        for x in range(xbins):
            if len(a_bins_ens[x]) > 0:
                CIs_ens[x] = sns.utils.ci(np.array(a_bins_ens[x]), which=CI)
                CIs_tar[x] = sns.utils.ci(np.array(a_bins_tar[x]), which=CI)
        a_total = 0
        for a in a_bins_tar:
            for a2 in a:
                a_total += a2
        if a_total < 10000:
            print('warning: a_tar[%s] is very small' %n)
            continue

        x_int_ens = 1.0
        x_int_tar = 1.0
        for x in range(xbins):
            # if np.max(a_bins_ens[x]) > thr:
            if CIs_ens[x, 1] > thr:
                x_int_ens = xdote_bins[x]
                break
        for x in range(xbins):
            # if np.max(a_bins_tar[x]) > thr:
            if CIs_tar[x, 1] > thr:
                x_int_tar = xdote_bins[x]
                break
        y_int_ens = np.mean(a_bins_ens[-1])
        y_int_tar = np.mean(a_bins_tar[-1])
        loss_x = (x_int_ens - x_int_tar) * loss_scale_x
        loss_y = (y_int_ens - y_int_tar) * loss_scale_y

        if pt:
            fig, ax = plt.subplots(figsize=(8, 8))
            hz_mins_ens = np.zeros((xbins))
            hz_maxs_ens = np.zeros((xbins))
            hz_mins_tar = np.zeros((xbins))
            hz_maxs_tar = np.zeros((xbins))
            means_ens = np.zeros((xbins))
            means_tar = np.zeros((xbins))
            for x in range(xbins):
                hz_mins_ens[x] = CIs_ens[x, 0]  # np.min(a_bins_ens[x])
                hz_maxs_ens[x] = CIs_ens[x, 1]  # np.max(a_bins_ens[x])
                hz_mins_tar[x] = CIs_tar[x, 0]  # np.min(a_bins_tar[x])
                hz_maxs_tar[x] = CIs_tar[x, 1]  # np.max(a_bins_tar[x])
                means_ens[x] = np.mean(a_bins_ens[x])
                means_tar[x] = np.mean(a_bins_tar[x])
            ax.plot(np.sign(enc[n])*xdote_bins, means_ens, label='ens')
            ax.plot(np.sign(enc[n])*xdote_bins, means_tar, label='tar')
            ax.fill_between(np.sign(enc[n])*xdote_bins, hz_mins_ens, hz_maxs_ens, alpha=0.25, label='ens')
            ax.fill_between(np.sign(enc[n])*xdote_bins, hz_mins_tar, hz_maxs_tar, alpha=0.25, label='tar')
            ax.set(xlim=((xmin, xmax)), ylim=((0, ymax)), xlabel=r"$\mathbf{x}$", ylabel='a (Hz)',
                title='neuron %s, loss_x=%.3f, loss_y=%.3f'%(n, loss_x, loss_y))
            plt.legend()
            plt.tight_layout()
            plt.savefig(name+"neuron%s"%n)
            plt.close()

        # print('x_int_ens', x_int_ens)
        # print('x_int_tar', x_int_tar)
        # print('y_int_ens', y_int_ens)
        # print('y_int_tar', y_int_tar)
        # print('loss_x', loss_x, 'loss_y', loss_y)
        losses[n] = np.abs(loss_x) + np.abs(loss_y)
        if np.abs(loss_x) > tol:
            b[n] += (x_int_ens - x_int_tar) * delta_b
        if np.abs(loss_y) > tol:
            g[n] += -(y_int_ens - y_int_tar) * delta_g

    return g, b, losses


def gb_opt2(ens, tar, u, enc, g, b, f=Lowpass(0.05), t_transient=0.2, dt=0.001,
        pt=True, ymax=60, name="default", xmin=-1, xmax=1,
        delta_g=3e-5, delta_b=2e-5, tol=0.2, thr=1, loss_scale_x=2, loss_scale_y=0.05):

    n_neurons = ens.shape[1]
    a_ens = f.filt(ens, dt=dt)[int(t_transient/dt):]
    a_tar = f.filt(tar, dt=dt)[int(t_transient/dt):]
    u = f.filt(u, dt=dt)[int(t_transient/dt):]
    losses = np.zeros((n_neurons))
    for n in range(n_neurons):
        xdote = np.dot(u, enc[n])
        y_int_ens = np.max(a_ens[:, n])
        y_int_tar = np.max(a_tar[:, n])
        x_int_ens = 1.0
        x_int_tar = 1.0        
        for t in range(a_ens.shape[0]):
            if a_ens[t, n] > thr and xdote[t] < x_int_ens and xmin < xdote[t] < xmax:
                x_int_ens = xdote[t]
            if a_tar[t, n] > thr and xdote[t] < x_int_tar and xmin < xdote[t] < xmax:
                x_int_tar = xdote[t]
        loss_x = (x_int_ens - x_int_tar) * loss_scale_x
        loss_y = (y_int_ens - y_int_tar) * loss_scale_y
        if pt:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(xdote, a_ens[:, n], s=10, alpha=0.5, label='ens')
            ax.scatter(xdote, a_tar[:, n], s=10, alpha=0.5, label='tar')
            ax.set(xlim=((-1, 1)), ylim=((0, ymax)), xlabel=r"$\mathbf{x}$", ylabel='a (Hz)',
                title='neuron %s, loss_x=%.3f, loss_y=%.3f \n x_int_ens=%.3f, x_int_tar=%.3f'
                %(n, loss_x, loss_y, x_int_ens, x_int_tar))
            plt.legend()
            plt.tight_layout()                   
            plt.savefig(name+"neuron%s"%n)
            plt.close()
        losses[n] = np.abs(loss_x) + np.abs(loss_y)
        if np.abs(loss_x) > tol:
            b[n] += (x_int_ens - x_int_tar) * delta_b
        if np.abs(loss_y) > tol:
            g[n] += -(y_int_ens - y_int_tar) * delta_g
    return g, b, losses



def dh_lstsq(stim_data, target_data, spk_data,
        lambda_c=1e-1, lambda_d=1e-1, order=1, n_samples=10000,
        min_d=-1e-2, max_d=1e-2, dt=0.001, h_tar=Lowpass(0.1), 
        mean_taus=[1e-1, 1e-2], std_taus=[1e-2, 1e-3], max_tau=1e0, lstsq_iter=100):
    
    """Courtesy of Aaron Voelker"""
    mean_taus = np.array(mean_taus)[:order]
    std_taus = np.array(std_taus)[:order]

    def sample_prior(n_samples, order, mean_taus, std_taus, min_tau=1e-5, rng=np.random.RandomState(seed=0)):
        """Return n samples (taus) from the prior of a k'th-order synapse."""
        taus = np.zeros((n_samples, order))
        for o in range(order):
            taus[:, o] = rng.normal(mean_taus[o], std_taus[o], size=(n_samples, )).clip(min_tau)
        return taus
    
    for att in range(lstsq_iter):  # attempts
        assert len(mean_taus) == order
        assert len(std_taus) == order
        taus = sample_prior(n_samples, order, mean_taus, std_taus)

        poles = -1. / taus
        n_steps = spk_data.shape[0]
        n_neurons = spk_data.shape[1]
        assert poles.shape == (n_samples, order)

        tf_params = np.zeros((n_samples, order))
        for i in range(n_samples):
            sys = LinearSystem(([], poles[i, :], 1 / np.prod(taus[i, :])))   # (zeros, poles, gain)
            assert len(sys) == order
            assert np.allclose(sys.dcgain, 1)
            den_normalized = np.asarray(sys.den / sys.num[0])
            assert len(den_normalized) == order + 1
            assert np.allclose(den_normalized[-1], 1)  # since normalized
            # tf_params ordered from lowest to highest, ignoring c_0 = 1, i.e., [c_1, ..., c_k]
            tf_params[i, :] = den_normalized[:-1][::-1]

        # We assume c_i are independent by setting the off-diagonals to zero
        C = np.cov(tf_params, rowvar=False)
        if order == 1:
            C = C*np.eye(1)
        Q = np.abs(np.linalg.inv(C))
        c0 = np.mean(tf_params, axis=0)
        d0 = np.ones((n_neurons, ))
        cd0 = np.hstack((c0, d0))
        assert Q.shape == (order, order)
        assert cd0.shape == (order+n_neurons,)

        diff = (1. - ~z) / dt
        A = np.zeros((n_steps, order + n_neurons))
        deriv_n = target_data
        for i in range(order):
            deriv_n = diff.filt(deriv_n, dt=dt)
            A[:, i] = deriv_n.ravel()  # todo: D>1
        for n in range(n_neurons):
            A[:, order+n] = spk_data[:, n]
        b = h_tar.tau  # set on pre_u ==> supv connection in network
        Y = (b*stim_data - target_data)
        A = h_tar.filt(A, dt=dt, axis=0)
        Y = h_tar.filt(Y, dt=dt)

        # construct block diagonal matrix with different regularizations for filter coefficients and decoders
        L = block_diag(lambda_c*Q, lambda_d*np.eye(n_neurons))
        gamma = A.T.dot(A) + L
        upsilon = A.T.dot(Y) + L.dot(cd0).reshape((order+n_neurons, 1))  # optional term with tikhonov regularization

        cd = np.linalg.inv(gamma).dot(upsilon).ravel()
        c_new = cd[:order]
        d_new = -1.*cd[-n_neurons:]
        assert c_new.shape==(order,)
        assert d_new.shape==(n_neurons,)
        print('taus attempt %s, nonzero d %s, tau=%s: '%(att, np.count_nonzero(d_new+1), c_new))
        for n in range(n_neurons):
            if d_new[n] > max_d or d_new[n] < min_d:
                d_new[n] = 0
        d_new = d_new.reshape((n_neurons, 1))
        if order == 1:
            h_new = Lowpass(c_new[0])
        elif order == 2:
            h_new = DoubleExp(c_new[0], c_new[1])
#         h_new = 1. / (1 + sum(c_new[i] * s**(i+1) for i in range(order)))
        assert np.allclose(h_new.dcgain, 1)
        if np.all(c_new > 0):
            break
        else:
            mean_taus[np.argmin(mean_taus)] *= 1.25
            lambda_c *= 1.25
            lambda_d *= 1.25

    return d_new, h_new

# def gb_opt(conn, fx=lambda x: x, h_tar=Lowpass(0.1), h_smooth=Lowpass(0.1), h_pre=Lowpass(0.1),
#         t=10, seed=0, gb_iter=1, n_trials=3, dt=0.001, pt=False, gain_pre=None, bias_pre=None, bins=20, 
#         neuron_type_pre=nengo.LIF(), delta_gain=3e-4, delta_bias=1e-5, tol=2.0, intercept_thr=5):

#     print('optimizing', conn)
#     n_neurons = conn.post_obj.n_neurons
#     nrn_idx = list(np.arange(n_neurons))
#     y_max = conn.post_obj.max_rates.high
#     dims = conn.post_obj.dimensions
#     gain = 1e-2 * np.ones((n_neurons, conn.post_obj.dimensions))
#     bias = np.zeros((n_neurons))
#     trimmed_gain = gain
#     trimmed_bias = bias
#     trimmed_max_rates = conn.post_obj.max_rates
#     trimmed_intercepts = conn.post_obj.intercepts
#     trimmed_encoders = conn.post_obj.encoders
#     solver = LstsqL2(reg=1e-1) if (isinstance(conn.solver, NoSolver) and np.count_nonzero(conn.solver.values) == 0) else conn.solver
#     for i in range(gb_iter):
#         if len(nrn_idx) == 0:
#             print('finished gain_bias optimization')
#             break
#         print('iteration %s/%s, remaining neurons %s' %(i, gb_iter, len(nrn_idx)))
#         timess = np.zeros((n_trials, int(t/dt), 1))
#         stims = np.zeros((n_trials, int(t/dt), dims))
#         a_enss = np.zeros((n_trials, int(t/dt), len(nrn_idx)))
#         a_lifs = np.zeros((n_trials, int(t/dt), len(nrn_idx)))
#         for trial in range(n_trials):
#             with nengo.Network(seed=seed) as net:
#                 uraws = []
#                 net.T = t
#                 def flip(t, x):
#                     if t<net.T/2:
#                         return x
#                     elif t>=net.T/2:
#                         return -1.0*x
#                 stim = nengo.Node(output=flip, size_in=dims)
#                 for dim in range(dims):
#                     uraws.append(nengo.Node(nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.6, seed=trial)))
#                     nengo.Connection(uraws[dim], stim[dim], synapse=None)
#                 spk = nengo.Ensemble(100, conn.pre_obj.dimensions, neuron_type=nengo.LIF())
#                 pre = nengo.Ensemble(conn.pre_obj.n_neurons, conn.pre_obj.dimensions, neuron_type=neuron_type_pre,
#                     max_rates=conn.pre_obj.max_rates, intercepts=conn.pre_obj.intercepts,
#                     encoders=conn.pre_obj.encoders, radius=conn.pre_obj.radius, seed=conn.pre_obj.seed)
#                 ens = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=conn.post_obj.neuron_type,
#                     max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
#                     radius=conn.post_obj.radius, seed=conn.post_obj.seed)
#                 lif = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=nengo.LIF(),
#                     max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
#                     radius=conn.post_obj.radius, seed=conn.post_obj.seed)
#                 nengo.Connection(stim, spk, synapse=None)
#                 conn_pre = nengo.Connection(spk, pre, synapse=h_pre)
#                 conn_ens = nengo.Connection(pre, ens, solver=solver, synapse=conn.synapse, seed=conn.seed)
#                 conn_lif = nengo.Connection(pre, lif, solver=solver, synapse=h_tar, function=fx, seed=conn.seed)
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     if np.any(gain_pre):
#                         conn_pre.gain = gain_pre
#                         conn_pre.bias = bias_pre
#                     conn_ens.gain = trimmed_gain
#                     conn_ens.bias = trimmed_bias
#                 p_stim = nengo.Probe(stim, synapse=h_pre)
#                 p_ens = nengo.Probe(ens.neurons, synapse=h_smooth)
#                 p_lif = nengo.Probe(lif.neurons, synapse=h_smooth)
#             with nengo.Simulator(net, dt=dt, seed=seed, progress_bar=False) as optsim:
#                 neuron.h.init()
#                 optsim.run(t, progress_bar=True)
#                 reset_neuron(optsim)
#             timess[trial] = optsim.trange().reshape(int(t/dt), 1)
#             stims[trial] = optsim.data[p_stim].reshape(int(t/dt), dims)
#             a_enss[trial] = optsim.data[p_ens]
#             a_lifs[trial] = optsim.data[p_lif]
#             enc = optsim.data[ens].encoders
#             if i == 0:
#                 trimmed_max_rates = optsim.data[ens].max_rates
#                 trimmed_intercepts = optsim.data[ens].intercepts
#                 trimmed_encoders = optsim.data[ens].encoders
# #             plt.plot(optsim.trange(), optsim.data[p_stim])
# #             plt.show()
#             del(net)
#             del(optsim)

#         times = timess.reshape((n_trials*int(t/dt), 1))
#         stim = stims.reshape((n_trials*int(t/dt), dims))
#         a_ens = a_enss.reshape((n_trials*int(t/dt), len(nrn_idx)))
#         a_lif = a_lifs.reshape((n_trials*int(t/dt), len(nrn_idx)))

#         losses = np.full((len(nrn_idx), dims), np.inf)
#         for dim in range(dims):
#             tar = h_smooth.filt(stim[:,dim], dt=dt)
#             for n, nrn in enumerate(np.copy(nrn_idx)):
#                 ens_bins, ens_means, ens_stds = bin_activities_values_single(tar, a_ens[:, n], bins=bins)
#                 lif_bins, lif_means, lif_stds = bin_activities_values_single(tar, a_lif[:, n], bins=bins)
#                 # update heuristics
#                 delta_all = ens_means - lif_means
#                 delta_y = np.max(lif_means) - np.max(ens_means)
#                 # tuning curve direction
#                 if np.sign(enc[n, dim]) == 1:
#                     first = 0
#                     last = -1
#                 elif np.sign(enc[n, dim]) == -1:
#                     first = -1
#                     last = 0
#                 if np.all(delta_all < -intercept_thr) or np.all(delta_all > intercept_thr):
#                     # if ens tuning stricly less (greater) than lif tuning, increase (decrease) bias
#                     delta_x = np.min(ens_means) - np.min(lif_means)
#                     trimmed_bias[n] -= (1+i/3)*delta_bias * delta_x/3
#                 else:
#                     # find intercepts of ens and lif, adjust bias accordingly
#                     ens_where = np.where(ens_means <= intercept_thr)[0]
#                     lif_where = np.where(lif_means <= intercept_thr)[0]
#                     if len(ens_where) > 0:
#                         x_int_ens_idx = ens_where[last]
#                         x_int_ens = ens_bins[x_int_ens_idx]
#                     else:
#                         x_int_ens = ens_bins[first]
#                     if len(lif_where) > 0:
#                         x_int_lif_idx = lif_where[last]
#                         x_int_lif = lif_bins[x_int_lif_idx]
#                     else:
#                         x_int_lif = lif_bins[first]
#                     delta_x = x_int_lif - x_int_ens  # positive ==> x-intercept is too far left
#                     trimmed_bias[n] -= (1+i/3)*delta_bias * delta_x * np.sign(enc[n, dim])
#                 # if ens max_rate is below (above) lif max_rate, increase (decrease) gain
#                 gain_change = delta_gain * delta_y
#                 if np.abs(gain_change) > trimmed_gain[n, dim] and gain_change < 0:
#                     trimmed_gain[n, dim] /= 2
#                 else:
#                     trimmed_gain[n, dim] += gain_change
#                 losses[n, dim] = 10*np.abs(delta_x) + np.abs(delta_y)/10
#                 if pt:
#                     cmap = sns.color_palette()
#                     fig, ax = plt.subplots(figsize=(8, 8))
#                     ax.plot(ens_bins, ens_means)
#                     ax.fill_between(ens_bins, ens_means+ens_stds, ens_means-ens_stds, alpha=0.25)
#                     ax.plot(lif_bins, lif_means, linestyle='--', c=cmap[0])
#                     ax.set(xlim=((-1, 1)), ylim=((0, y_max)),
#                         xlabel='$\mathbf{x}$', ylabel='a (Hz)',
#                            title='neuron %s, dim %s\n gain %.6f, bias %.6f\n delta_x %.3f, delta_y %.3f\n loss %.3f'%(nrn, dim, trimmed_gain[n, dim], trimmed_bias[n], delta_x, delta_y, losses[n, dim]))
#                     plt.tight_layout()
#                     plt.show()                    
#         to_delete = []
#         for n, nrn in enumerate(np.copy(nrn_idx)):
#             if np.sum(np.abs(losses[n])) < tol*dims:
#                 to_delete.append(n)
#                 bias[nrn] = trimmed_bias[n]
#                 gain[nrn] = trimmed_gain[n]                    
#         trimmed_gain = np.delete(trimmed_gain, to_delete, axis=0)
#         trimmed_bias = np.delete(trimmed_bias, to_delete, axis=0)
#         trimmed_max_rates = np.delete(trimmed_max_rates, to_delete)
#         trimmed_intercepts = np.delete(trimmed_intercepts, to_delete)
#         trimmed_encoders = np.delete(trimmed_encoders, to_delete, axis=0)
#         nrn_idx = np.delete(np.array(nrn_idx), to_delete)

#     for n, nrn in enumerate(np.copy(nrn_idx)):
#         bias[nrn] = trimmed_bias[n]
#         gain[nrn] = trimmed_gain[n]

#     return gain, bias

# def gbopt_fb(conn, h_tar=Lowpass(0.1), t=10, seed=0, dt=0.001,
#         gb_iter=1, fb_iter=1, n_trials=3, pt=False, bins=20, order=1,
#         gain_pre=None, bias_pre=None, reg=1e-1, delta_gain=3e-4, delta_bias=1e-5, tol=2.0, intercept_thr=5):

#     print('optimizing', conn)
#     n_neurons = conn.post_obj.n_neurons
#     y_max = conn.post_obj.max_rates.high
#     dims = conn.post_obj.dimensions
#     if not np.all(gain_pre):
#         gain = 1e-2 * np.ones((n_neurons, conn.post_obj.dimensions))
#     else:
#         gain = gain_pre
#     if not np.all(bias_pre):
#         bias = np.zeros((n_neurons))
#     else:
#         bias = bias_pre
#     h_ens = conn.synapse
#     solver_ens = conn.solver
    
#     for a in range(fb_iter):
#         print('gbh iteration %s/%s' %(a, fb_iter))
#         nrn_idx = list(np.arange(n_neurons))
#         trimmed_gain = gain
#         trimmed_bias = bias
#         trimmed_max_rates = conn.post_obj.max_rates
#         trimmed_intercepts = conn.post_obj.intercepts
#         trimmed_encoders = conn.post_obj.encoders
#         for i in range(gb_iter):
#             if len(nrn_idx) == 0:
#                 print('finished gain_bias optimization')
#                 break
#             print('iteration %s/%s, remaining neurons %s' %(i, gb_iter, len(nrn_idx)))
#             timess = np.zeros((n_trials, int(t/dt), dims))
#             stims = np.zeros((n_trials, int(t/dt), dims))
#             a_enss = np.zeros((n_trials, int(t/dt), len(nrn_idx)))
#             a_lifs = np.zeros((n_trials, int(t/dt), len(nrn_idx)))
#             spk_ens = np.zeros((n_trials, int(t/dt), len(nrn_idx)))
#             for trial in range(n_trials):
#                 with nengo.Network(seed=seed) as net:
#                     uraws = []
#                     net.T = t
#                     def flip(t, x):
#                         if t<net.T/2:
#                             return x
#                         elif t>=net.T/2:
#                             return -1.0*x
#                     stim = nengo.Node(output=flip, size_in=dims)
#                     for dim in range(dims):
#                         uraws.append(nengo.Node(nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.6, seed=trial)))
#                         nengo.Connection(uraws[dim], stim[dim], synapse=None)
#                     spk = nengo.Ensemble(100, conn.pre_obj.dimensions, neuron_type=nengo.LIF())
#                     pre_ens = nengo.Ensemble(conn.pre_obj.n_neurons, conn.pre_obj.dimensions,
#                         neuron_type=conn.pre_obj.neuron_type,
#                         max_rates=conn.pre_obj.max_rates, intercepts=conn.pre_obj.intercepts,
#                         encoders=conn.pre_obj.encoders, radius=conn.pre_obj.radius, seed=conn.pre_obj.seed)
#                     pre_lif = nengo.Ensemble(conn.pre_obj.n_neurons, conn.pre_obj.dimensions, neuron_type=nengo.LIF(),
#                         max_rates=conn.pre_obj.max_rates, intercepts=conn.pre_obj.intercepts,
#                         encoders=conn.pre_obj.encoders, radius=conn.pre_obj.radius, seed=conn.pre_obj.seed)
#                     ens = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=conn.post_obj.neuron_type,
#                         max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
#                         radius=conn.post_obj.radius, seed=conn.post_obj.seed)
#                     lif = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=nengo.LIF(),
#                         max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
#                         radius=conn.post_obj.radius, seed=conn.post_obj.seed)
#                     nengo.Connection(stim, spk, synapse=None)
#                     conn_pre_ens = nengo.Connection(spk, pre_ens, synapse=h_tar)
#                     conn_pre_lif = nengo.Connection(spk, pre_lif, synapse=h_tar)
#                     conn_ens = nengo.Connection(pre_ens, ens, solver=solver_ens, synapse=h_ens, seed=conn.seed)
#                     conn_lif = nengo.Connection(pre_lif, lif, solver=LstsqL2(reg=reg), synapse=h_tar, seed=conn.seed)
#                     with warnings.catch_warnings():
#                         warnings.simplefilter("ignore")
#                         conn_pre_ens.gain = gain_pre
#                         conn_pre_ens.bias = bias_pre
#                         conn_ens.gain = trimmed_gain
#                         conn_ens.bias = trimmed_bias
#                     p_stim = nengo.Probe(stim, synapse=h_tar)
#                     p_ens = nengo.Probe(ens.neurons, synapse=h_ens)
#                     p_lif = nengo.Probe(lif.neurons, synapse=h_tar)
#                     p_spk = nengo.Probe(ens.neurons, synapse=None)
#                     p_v = nengo.Probe(ens.neurons, 'voltage', synapse=None)
#                 with nengo.Simulator(net, dt=dt, seed=seed, progress_bar=False) as optsim:
#                     neuron.h.init()
#                     optsim.run(t, progress_bar=True)
#                     reset_neuron(optsim) 
#                 timess[trial] = optsim.trange().reshape(-1, dims)
#                 stims[trial] = optsim.data[p_stim].reshape(-1, dims)
#                 a_enss[trial] = optsim.data[p_ens]
#                 a_lifs[trial] = optsim.data[p_lif]
#                 spk_ens[trial] = optsim.data[p_spk]
#                 enc = optsim.data[ens].encoders
#                 if i == 0:
#                     trimmed_max_rates = optsim.data[ens].max_rates
#                     trimmed_intercepts = optsim.data[ens].intercepts
#                     trimmed_encoders = optsim.data[ens].encoders
#                 del(net)
#                 del(optsim)

#             times = timess.reshape((n_trials*int(t/dt), dims))
#             stim = stims.reshape((n_trials*int(t/dt), dims))
#             a_ens = a_enss.reshape((n_trials*int(t/dt), len(nrn_idx)))
#             a_lif = a_lifs.reshape((n_trials*int(t/dt), len(nrn_idx)))
#             spk_ens = spk_ens.reshape((n_trials*int(t/dt), len(nrn_idx)))

#             losses = np.full((len(nrn_idx), dims), np.inf)
#             for dim in range(dims):
#                 tar = h_tar.filt(stim[:,dim], dt=dt)
#                 for n, nrn in enumerate(np.copy(nrn_idx)):
#                     ens_bins, ens_means, ens_stds = bin_activities_values_single(tar, a_ens[:, n], bins=bins)
#                     lif_bins, lif_means, lif_stds = bin_activities_values_single(tar, a_lif[:, n], bins=bins)
#                     # update heuristics
#                     delta_all = ens_means - lif_means
#                     delta_y = np.max(lif_means) - np.max(ens_means)
#                     # tuning curve direction
#                     if np.sign(enc[n, dim]) == 1:
#                         first = 0
#                         last = -1
#                     elif np.sign(enc[n, dim]) == -1:
#                         first = -1
#                         last = 0
#                     if np.all(delta_all < -intercept_thr) or np.all(delta_all > intercept_thr):
#                         # if ens tuning stricly less (greater) than lif tuning, increase (decrease) bias
#                         delta_x = np.min(ens_means) - np.min(lif_means)
#                         trimmed_bias[n] -= (1+i/3)*delta_bias * delta_x/3
#                     else:
#                         # find intercepts of ens and lif, adjust bias accordingly
#                         ens_where = np.where(ens_means <= intercept_thr)[0]
#                         lif_where = np.where(lif_means <= intercept_thr)[0]
#                         if len(ens_where) > 0:
#                             x_int_ens_idx = ens_where[last]
#                             x_int_ens = ens_bins[x_int_ens_idx]
#                         else:
#                             x_int_ens = ens_bins[first]
#                         if len(lif_where) > 0:
#                             x_int_lif_idx = lif_where[last]
#                             x_int_lif = lif_bins[x_int_lif_idx]
#                         else:
#                             x_int_lif = lif_bins[first]
#                         delta_x = x_int_lif - x_int_ens  # positive ==> x-intercept is too far left
#                         trimmed_bias[n] -= (1+i/3)*delta_bias * delta_x * np.sign(enc[n, dim])  # multiply by delta_y at this value, to prevent flat?
#                     # if ens max_rate is below (above) lif max_rate, increase (decrease) gain
#                     gain_change = delta_gain * delta_y
#                     if np.abs(gain_change) > trimmed_gain[n, dim] and gain_change < 0:
#                         trimmed_gain[n, dim] /= 2
#                     else:
#                         trimmed_gain[n, dim] += gain_change
#                     losses[n, dim] = 10*np.abs(delta_x) + np.abs(delta_y)/10
#                     if np.sum(ens_means) == 0:
#                         losses[n, dim] += tol
#                     if pt:
#                         cmap = sns.color_palette()
#                         fig, ax = plt.subplots(figsize=(8, 8))
#                         ax.plot(ens_bins, ens_means)
#                         ax.fill_between(ens_bins, ens_means+ens_stds, ens_means-ens_stds, alpha=0.25)
#                         ax.plot(lif_bins, lif_means, linestyle='--', c=cmap[0])
#                         ax.set(xlim=((-1, 1)), ylim=((0, y_max)),
#                             xlabel='$\mathbf{x}$', ylabel='a (Hz)',
#                                 title='neuron %s, dim %s\n gain %.6f, bias %.6f\n delta_x %.3f, delta_y %.3f\n loss %.3f'%\
#                                 (nrn, dim, trimmed_gain[n, dim], trimmed_bias[n], delta_x, delta_y, losses[n, dim]))
#                         plt.tight_layout()
#                         plt.show()

#             to_delete = []
#             for n, nrn in enumerate(np.copy(nrn_idx)):
#                 if np.sum(np.abs(losses[n])) < tol*dims:
#                     to_delete.append(n)
#                     bias[nrn] = trimmed_bias[n]
#                     gain[nrn] = trimmed_gain[n]
#             trimmed_gain = np.delete(trimmed_gain, to_delete, axis=0)
#             trimmed_bias = np.delete(trimmed_bias, to_delete, axis=0)
#             trimmed_max_rates = np.delete(trimmed_max_rates, to_delete)
#             trimmed_intercepts = np.delete(trimmed_intercepts, to_delete)
#             trimmed_encoders = np.delete(trimmed_encoders, to_delete, axis=0)
#             nrn_idx = np.delete(np.array(nrn_idx), to_delete)

#         for n, nrn in enumerate(np.copy(nrn_idx)):
#             bias[nrn] = trimmed_bias[n]
#             gain[nrn] = trimmed_gain[n]

#         gain_pre = gain
#         bias_pre = bias

#     return gain, bias


# dt = 0.0001
# t = np.arange(0, 20, dt)
# f = np.cumsum(1+0.5*np.sin(t))*dt
# y = np.sin(2*np.pi * f)
# fig, ax = plt.subplots(figsize=(16, 8))
# ax.plot(t, y)
# ax.plot(t, np.sin(2*np.pi*t))
# ax.plot(t, f)
# plt.show()


    # print('gathering training data for filter/decoder optimization')
    # lifs = np.zeros((n_trains, int(t_train/dt), n_neurons))
    # alifs = np.zeros((n_trains, int(t_train/dt), n_neurons))
    # wilsonss = np.zeros((n_trains, int(t_train/dt), n_neurons))
    # durstewitzs = np.zeros((n_trains, int(t_train/dt), n_neurons))
    # times = np.zeros((n_trains, int(t_train/dt)))
    # nefs = np.zeros((n_trains, int(t_train/dt), 1))
    # us = np.zeros((n_trains, int(t_train/dt), 1))
    # tars = np.zeros((n_trains, int(t_train/dt), 1))

    # for n in range(n_trains):
    #     stim_func=nengo.processes.WhiteSignal(period=t_train/2, high=1, rms=0.5, seed=n)
    #     data = go(d_lif, d_alif, d_wilson, d_durstewitz, h_lif, h_alif, h_wilson, h_durstewitz,
    #         n_neurons=n_neurons, t=t_train, h_tar=h_tar, dt=dt,
    #         gain=gain, bias=databias, stim_func=stim_func, supv=1)
    #     lifs[n] = data['lif']
    #     alifs[n] = data['alif']
    #     wilsonss[n] = data['wilson']
    #     durstewitzs[n] = data['durstewitz']
    #     times[n] =  data['times']
    #     nefs[n] = data['nef']
    #     us[n] = data['u']
    #     tars[n] = data['tar']
    # lifs = lifs.reshape((n_trains*int(t_train/dt), n_neurons))
    # alifs = alifs.reshape((n_trains*int(t_train/dt), n_neurons))
    # wilsonss = wilsonss.reshape((n_trains*int(t_train/dt), n_neurons))
    # durstewitzs = durstewitzs.reshape((n_trains*int(t_train/dt), n_neurons))
    # times = times.reshape((n_trains*int(t_train/dt), 1))
    # nefs = nefs.reshape((n_trains*int(t_train/dt), 1))
    # us = us.reshape((n_trains*int(t_train/dt), 1))
    # tars = tars.reshape((n_trains*int(t_train/dt), 1))
        
    # print('optimizing filters and decoders')
    # if h_iter:
    #     d_lif, h_lif  = dh_hyperopt(h_tar.filt(tars, dt=dt), lifs, order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_lif')
    #     d_alif, h_alif  = dh_hyperopt(h_tar.filt(tars, dt=dt), alifs, order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_alif')
    #     d_wilson, h_wilson  = dh_hyperopt(h_tar.filt(tars, dt=dt), wilsonss, order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_wilson')
    #     d_durstewitz, h_durstewitz  = dh_hyperopt(h_tar.filt(tars, dt=dt), durstewitzs, order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_durstewitz')
    # else:
    #     h_lif, h_alif, h_wilson, h_durstewitz = h_tar, h_tar, h_tar, h_tar
    #     d_lif = d_lstsq(tars, lifs, h_lif, h_tar, reg=reg, dt=dt)
    #     d_alif = d_lstsq(tars, alifs, h_alif, h_tar, reg=reg, dt=dt)
    #     d_wilson = d_lstsq(tars, wilsonss, h_wilson, h_tar, reg=reg, dt=dt)
    #     d_durstewitz = d_lstsq(tars, durstewitzs, h_durstewitz, h_tar, reg=reg, dt=dt)


    #     print('gathering training data for filter/decoder optimization')
    # for n in range(n_trains):
    #     stim_func = nengo.processes.WhiteSignal(period=t_train/2, high=1, rms=0.5, seed=n)
    #     fd_data = go(d_lif, d_alif, d_wilson, d_durstewitz, h_lif, h_alif, h_wilson, h_durstewitz,
    #         n_neurons=n_neurons, t=t_train, h_tar=h_tar, dt=dt,
    #         gain=gain, bias=databias, stim_func=stim_func, supv=1)
        
    # print('optimizing filters and decoders')
    # if h_iter:
    #     d_lif, h_lif  = dh_hyperopt(h_tar.filt(fd_data['tar'], dt=dt), fd_data['lif'], order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_lif')
    #     d_alif, h_alif  = dh_hyperopt(h_tar.filt(fd_data['tar'], dt=dt), fd_data['alif'], order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_alif')
    #     d_wilson, h_wilson  = dh_hyperopt(h_tar.filt(fd_data['tar'], dt=dt), fd_data['wilson'], order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_wilson')
    #     d_durstewitz, h_durstewitz  = dh_hyperopt(h_tar.filt(fd_data['tar'], dt=dt), fd_data['durstewitz'], order=order, h_iter=h_iter, dt=dt, reg_min=reg, name='integrator_durstewitz')
    # else:
    #     h_lif, h_alif, h_wilson, h_durstewitz = h_tar, h_tar, h_tar, h_tar
    #     d_lif = d_lstsq(fd_data['tar'], fd_data['lif'], h_lif, h_tar, reg=reg, dt=dt)
    #     d_alif = d_lstsq(fd_data['tar'], fd_data['alif'], h_alif, h_tar, reg=reg, dt=dt)
    #     d_wilson = d_lstsq(fd_data['tar'], fd_data['wilson'], h_wilson, h_tar, reg=reg, dt=dt)
    #     d_durstewitz = d_lstsq(fd_data['tar'], fd_data['durstewitz'], h_durstewitz, h_tar, reg=reg, dt=dt)



    # elif metric == 'probability':
    # # measure probabilistic overlap between the two activity ranges for each x value
    # # currently returns nonsense values
    # for x in range(xbins):
    #     mu_ens = np.mean(a_bins_ens[x])
    #     sigma_ens = np.var(a_bins_ens[x])
    #     mu_tar = np.mean(a_bins_tar[x])
    #     sigma_tar = np.var(a_bins_tar[x])
    #     if sigma_ens == 0 and sigma_tar == 0:
    #         loss += 0
    #         continue
    #     elif sigma_ens == 0:
    #         sigma_ens = 0.01
    #     elif sigma_tar == 0:
    #         sigma_tar = 0.01
    #     mu_joint = (sigma_ens**(-2)*mu_ens + sigma_tar**(-2)*mu_tar) / (sigma_ens**(-2) + sigma_tar**(-2))
    #     sigma_joint = sigma_ens*sigma_tar / (sigma_ens + sigma_tar)
    #     def normal(hz):
    #         return norm.pdf(hz, mu_joint, sigma_joint)
    #     overlap, err = quad(normal, 0, 60)
    #     print(xdote_bins[x], overlap)
    #     loss += 1.0 - overlap

            # loss = 0
        # if metric == 'geometric':
        #     def geo_overlap(min1, max1, min2, max2):
        #         if min1<thr and min2<thr and max1<thr and max2<thr:
        #             return 0.0
        #         overlap = np.max([0, min(max1, max2) - max(min1, min2)])
        #         hz_range = np.abs(np.min([min1, min2]) - np.max([max1, max2]))
        #         return hz_range - overlap
        #     for x in range(xbins):
        #         error = geo_overlap(np.min(a_bins_ens[x]), np.max(a_bins_ens[x]),
        #             np.min(a_bins_tar[x]), np.max(a_bins_tar[x]))
        #         loss += error
        #     loss /= xbins


#         def find_nearest(array, value):
#     idx = (np.abs(array-value)).argmin()
#     return idx

# def bin_activities_values_single(xhat_pre, act_bio, bins=20):
#     x_bins = np.linspace(np.max([np.min(xhat_pre), -1]), np.min([1, np.max(xhat_pre)]), num=bins)
#     hz_means = np.zeros((bins))
#     hz_stds = np.zeros((bins))
#     bin_act = [[] for _ in range(x_bins.shape[0])]
#     for t in range(act_bio.shape[0]):
#         idx = find_nearest(x_bins, xhat_pre[t])
#         bin_act[idx].append(act_bio[t])
#     for x in range(len(bin_act)):
#         hz_means[x] = np.average(bin_act[x]) if len(bin_act[x]) > 0 else 0
#         hz_stds[x] = np.std(bin_act[x]) if len(bin_act[x]) > 1 else 0
#     return x_bins, hz_means, hz_stds
            
# def isi(all_spikes, dt=0.000025):
#     nz = []
#     for n in range(all_spikes.shape[1]):
#         sts = np.nonzero(all_spikes[:,n])
#         nz.append((np.diff(sts)*dt).ravel())
#     return nz

# def nrmse_vs_n_neurons():
#     neurons = [3, 10, 30, 100]
#     n_trials = 3
#     nrmses = np.zeros((len(neurons), 5, n_trials))
#     for nn, n_neurons in enumerate(neurons):
#         nrmses[nn] = trials(n_neurons=n_neurons, n_trials=n_trials, gb_iter=3, h_iter=200)

#     nts =  ['LIF (static)', 'LIF (temporal)', 'ALIF', 'Wilson', 'Durstewitz']
#     columns=['nrmse', 'n_neurons', 'neuron_type', 'trial']
#     df = pd.DataFrame(columns=columns)
#     for nn in range(len(neurons)):
#         for nt in range(len(nts)):
#             for trial in range(n_trials):
#                 df_new = pd.DataFrame([[nrmses[nn, nt, trial], neurons[nn], nts[nt], trial]], columns=columns)
#                 df = df.append(df_new, ignore_index=True)

#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax = sns.lineplot(x='n_neurons', y='nrmse', hue='neuron_type', data=df)
#     plt.show()