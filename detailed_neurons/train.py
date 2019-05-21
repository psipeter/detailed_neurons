import numpy as np

from scipy.linalg import block_diag
from scipy.integrate import ode
from scipy.optimize import minimize

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
from utils import bin_activities_values_single

import neuron
neuron.h.load_file('/home/pduggins/detailed_neurons/detailed_neurons/NEURON/durstewitz/durstewitz.hoc')
neuron.h.load_file('stdrun.hoc')


__all__ = ['norms', 'dh_hyperopt', 'dh_lstsq', 'gbopt', 'd_lstsq']

def norms(t, dt=0.001, stim_func=lambda t: np.cos(t)):
    with nengo.Network() as model:
        stim = nengo.Node(stim_func)
        p_stimulus = nengo.Probe(stim, synapse=None)
        p_integral = nengo.Probe(stim, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    norm_stim = np.max(np.abs(sim.data[p_stimulus]))
    norm_int = np.max(np.abs(sim.data[p_integral]))
    return norm_stim, norm_int

def norms_fx(fx, t, dt=0.001, stim_func=lambda t: np.cos(t)):
    with nengo.Network() as model:
        stim = nengo.Node(stim_func)
        fx = nengo.Ensemble(1, stim.size_out, neuron_type=nengo.Direct())
        nengo.Connection(stim, fx, synapse=None, function=fx)
        p_stim = nengo.Probe(stim, synapse=None)
        p_fx = nengo.Probe(fx, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    norm_stim = np.max(np.abs(sim.data[p_stim]))
    norm_fx = np.max(np.abs(sim.data[p_fx]))
    return norm_stim, norm_fx

def d_lstsq(target, spikes, h, h_tar, reg=1e-1, dt=0.001):
    target = h_tar.filt(target, dt=dt)
    A = h.filt(spikes, dt=dt)
    d_new = LstsqL2(reg=reg)(A, target)[0]
    return d_new

def dh_hyperopt(target, spikes, name='default', order=1, h_iter=10, seed=0, dt=0.001,
        tau_mins=[0.05, 0.001], tau_maxs=[0.2, 0.01], reg_min=1e-1, reg_max=1e-1):

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
    hyperparams['reg'] = reg_min

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
        max_evals=h_iter,
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


def dh_lstsq(stim_data, target_data, spk_data,
        lambda_c=1e-1, lambda_d=1e-1, order=1, n_samples=10000,
        min_d=-1e-2, max_d=1e-2, dt=0.001, h_tar=Lowpass(0.1), 
        mean_taus=[1e-1, 1e-2], std_taus=[1e-2, 1e-3], max_tau=1e0, lstsq_iter=100):
    
    """Courtesy of Aaron Voelker"""
    mean_taus = np.array(mean_taus)[:order]
    std_taus = np.array(std_taus)[:order]
#     for o in range(order):
#         mean_taus.append(np.exp(-(o+1)))
#         std_taus.append(np.exp(-(o+2)))

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

def gbopt(conn, fx=lambda x: x, h_tar=Lowpass(0.1), h_smooth=Lowpass(0.1), h_pre=Lowpass(0.1), stim_func=lambda t: np.sin(t),
        t=4*np.pi, seed=0, gb_iter=1, dt=0.001, t_trans=0.25, pt=False, gain_pre=None, bias_pre=None, bins=20,
        do_norm=False, do_flip=False, delta_gain=1e-3, delta_bias=1e-5, tol=0.5, gain_thr=2, bias_thr=0.75):
        # delta_gain=1e-3, delta_bias=1e-5, tol=0.03, gain_thr=2):

    print('optimizing', conn)
    n_neurons = conn.post_obj.n_neurons
    nrn_idx = list(np.arange(n_neurons))
    y_max = conn.post_obj.max_rates.high
    dims = conn.post_obj.dimensions
    norm_stims = []
    for dim in range(dims):
        norm_stims.append(norms(t, dt=dt, stim_func=stim_func)[0])
    gain = 1e-2 * np.ones((n_neurons, conn.post_obj.dimensions))
    bias = np.zeros((n_neurons))
    trimmed_gain = gain
    trimmed_bias = bias
    trimmed_max_rates = conn.post_obj.max_rates
    trimmed_intercepts = conn.post_obj.intercepts
    trimmed_encoders = conn.post_obj.encoders
    solver = LstsqL2(reg=1e-1) if (isinstance(conn.solver, NoSolver) and np.count_nonzero(conn.solver.values) == 0) else conn.solver
    for i in range(gb_iter):
        if len(nrn_idx) == 0:
            print('finished gain_bias optimization')
            break
        print('iteration %s/%s, remaining neurons %s' %(i, gb_iter, len(nrn_idx)))
        with nengo.Network(seed=seed) as net:
            uraws = []
            net.T = t
            if do_flip:
                def flip(t, x):
                    if t<net.T/2:
                        return x
                    elif t>=net.T/2:
                        return -1.0*x
                stim = nengo.Node(output=flip, size_in=dims)
            else:
                stim = nengo.Node(size_in=dims)
            for dim in range(dims):
                uraws.append(nengo.Node(stim_func))
                nengo.Connection(uraws[dim], stim[dim], synapse=None, transform=1.0/norm_stims[dim] if do_norm else 1)
            spk = nengo.Ensemble(100, conn.pre_obj.dimensions, neuron_type=nengo.LIF())
            pre = nengo.Ensemble(conn.pre_obj.n_neurons, conn.pre_obj.dimensions, neuron_type=conn.pre_obj.neuron_type,
                max_rates=conn.pre_obj.max_rates, intercepts=conn.pre_obj.intercepts,
                encoders=conn.pre_obj.encoders, radius=conn.pre_obj.radius, seed=conn.pre_obj.seed)
            ens = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=conn.post_obj.neuron_type,
                max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
                radius=conn.post_obj.radius, seed=conn.post_obj.seed)
            lif = nengo.Ensemble(len(nrn_idx), conn.post_obj.dimensions, neuron_type=nengo.LIF(),
                max_rates=trimmed_max_rates, intercepts=trimmed_intercepts, encoders=trimmed_encoders,
                radius=conn.post_obj.radius, seed=conn.post_obj.seed)
            nengo.Connection(stim, spk, synapse=None)
            conn_pre = nengo.Connection(spk, pre, synapse=h_pre)
            conn_ens = nengo.Connection(pre, ens, solver=solver, synapse=conn.synapse, seed=conn.seed)
            conn_lif = nengo.Connection(pre, lif, solver=solver, synapse=conn.synapse, function=fx, seed=conn.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.any(gain_pre):
                    conn_pre.gain = gain_pre
                    conn_pre.bias = bias_pre
                conn_ens.gain = trimmed_gain
                conn_ens.bias = trimmed_bias
            p_stim = nengo.Probe(stim, synapse=h_pre)
            p_ens = nengo.Probe(ens.neurons, synapse=h_smooth)
            p_lif = nengo.Probe(lif.neurons, synapse=h_smooth)
        with nengo.Simulator(net, dt=dt, seed=seed, progress_bar=False) as optsim:
            neuron.h.init()
            optsim.run(t, progress_bar=True)
            reset_neuron(optsim) 
        times = optsim.trange()
        stim = optsim.data[p_stim][int(t_trans/dt):]
        a_ens = optsim.data[p_ens][int(t_trans/dt):]
        a_lif = optsim.data[p_lif][int(t_trans/dt):]
        enc = optsim.data[ens].encoders
        if i == 0:
            trimmed_max_rates = optsim.data[ens].max_rates
            trimmed_intercepts = optsim.data[ens].intercepts
            trimmed_encoders = optsim.data[ens].encoders
        del(net)
        del(optsim)

        losses = np.full((n_neurons, dims), np.inf)
        for dim in range(dims):
#             tar = h_smooth.filt(h_tar.filt(fx(stim[:,dim]), dt=dt), dt=dt)
            # tar = h_smooth.filt(h_tar.filt(stim[:,dim], dt=dt), dt=dt)
            tar = h_smooth.filt(stim[:,dim], dt=dt)
            for n, nrn in enumerate(np.copy(nrn_idx)):
#                 print('ens sum', np.sum(a_ens[:, n]))
#                 print('lif sum', np.sum(a_lif[:, n]))
#                 print('trimmed_max_rates', trimmed_max_rates[n])
#                 print('trimmed_intercepts', trimmed_intercepts[n])
#                 print('trimmed_encoders', trimmed_encoders[n])
                ens_bins, ens_means, ens_stds = bin_activities_values_single(tar, a_ens[:, n], bins=bins)
                lif_bins, lif_means, lif_stds = bin_activities_values_single(tar, a_lif[:, n], bins=bins)

                if pt:
                    cmap = sns.color_palette()
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.plot(ens_bins, ens_means)
                    ax.fill_between(ens_bins, ens_means+ens_stds, ens_means-ens_stds, alpha=0.25)
                    ax.plot(lif_bins, lif_means, linestyle='--', c=cmap[0])
                    ax.set(xlim=((-1, 1)), ylim=((0, y_max)),
                        xlabel='$\mathbf{x}$', ylabel='a (Hz)', title='neuron %s, dim %s'%(nrn, dim))
                    plt.tight_layout()
                    plt.show()

#                 if np.all(ens_means < gain_thr):
#                     trimmed_bias[n] += 0.3 * delta_bias
#                     continue
#                 if np.all(ens_means > y_max):
#                     trimmed_bias[n] -= 0.3 * delta_bias
#                     continue
#                 if np.sign(enc[n, dim]) == 1:
#                     first = 0
#                     last = -1
#                 elif np.sign(enc[n, dim]) == -1:
#                     first = -1
#                     last = 0
#                 else:
#                     raise RuntimeError("encountered encoder == 0")
#                 ens_where = np.where(ens_means <= gain_thr)[0]
#                 lif_where = np.where(lif_means <= gain_thr)[0]
#                 if len(ens_where) > 0:
#                     x_int_ens_idx = ens_where[last]
#                     x_int_ens = ens_bins[x_int_ens_idx]
#                 else:
#                     x_int_ens = ens_bins[first]
#                 if len(lif_where) > 0:
#                     x_int_lif_idx = lif_where[last]
#                     x_int_lif = lif_bins[x_int_lif_idx]
#                 else:
#                     x_int_lif = lif_bins[first]
#                 y_int_ens = ens_means[last]
#                 y_int_lif = lif_means[last]
#                 delta_x = x_int_lif - x_int_ens  # positive ==> x-intercept is too far left
#                 delta_y = y_int_lif - y_int_ens  # positive ==> max firing rate is too low
#                 loss = np.square(delta_x) + np.square(delta_y / y_max)
#                 losses[n, dim] = loss
#                 if loss < tol:
#                     continue
#                 if delta_x >= 0:
#                     trimmed_bias[n] -= delta_bias * np.abs(delta_x) * np.sign(enc[n, dim])
#                 else:
#                     trimmed_bias[n] += delta_bias * np.abs(delta_x) * np.sign(enc[n, dim])
#                 if delta_y >= 0:
#                     trimmed_gain[n, dim] += delta_gain * np.abs(delta_y)
#                 else:
#                     trimmed_gain[n, dim] -= delta_gain * np.abs(delta_y)
#                 trimmed_gain[n, dim] = np.abs(trimmed_gain[n, dim])

                # update heuristics
                delta = ens_means - lif_means
                delta_max = np.max(ens_means) - np.max(lif_means)
                losses[n, dim] = np.abs(np.average(delta))
                # tuning curve direction
                if np.sign(enc[n, dim]) == 1:
                    first = 0
                    last = -1
                elif np.sign(enc[n, dim]) == -1:
                    first = -1
                    last = 0
                if np.abs(np.average(delta)) < tol:
                    continue
                if np.all(delta < 0):
                    # if ens tuning stricly less than lif tuning, increase bias
                    trimmed_bias[n] += 0.1* delta_bias * np.abs(np.average(delta))
                elif np.all(delta > 0):
                    # if ens tuning stricly greater than lif tuning, decrease bias
                    trimmed_bias[n] -= 0.1* delta_bias * np.abs(np.average(delta))
                # if len(np.where(delta < 0)[0])/len(delta) > bias_thr:
                #     # if bias_thr percent of ens tuning points lie below lif tuning points, increase bias
                #     trimmed_bias[n] += 0.2 * delta_bias
                # elif len(np.where(delta > 0)[0])/len(delta) > bias_thr:
                #     # if bias_thr percent of ens tuning points lie above lif tuning points, decrease bias
                #     trimmed_bias[n] -= 0.2 * delta_bias
                else:
                    # find intercepts of ens and lif, adjust bias accordingly
                    ens_where = np.where(ens_means <= gain_thr)[0]
                    lif_where = np.where(lif_means <= gain_thr)[0]
                    if len(ens_where) > 0:
                        x_int_ens_idx = ens_where[last]
                        x_int_ens = ens_bins[x_int_ens_idx]
                    else:
                        x_int_ens = ens_bins[first]
                    if len(lif_where) > 0:
                        x_int_lif_idx = lif_where[last]
                        x_int_lif = lif_bins[x_int_lif_idx]
                    else:
                        x_int_lif = lif_bins[first]
                    y_int_ens = ens_means[last]
                    y_int_lif = lif_means[last]
                    delta_x = x_int_lif - x_int_ens  # positive ==> x-intercept is too far left
                    if delta_x >= 0:
                        trimmed_bias[n] -= delta_bias * np.abs(delta_x) * np.sign(enc[n, dim])
                    else:
                        trimmed_bias[n] += delta_bias * np.abs(delta_x) * np.sign(enc[n, dim])
                if delta_max < 0:
                    # if ens max_rate is below lif max_rate, increase gain
                    trimmed_gain[n, dim] += delta_gain * np.abs(delta_max)
                elif delta_max > 0:
                    # if ens max_rate is above lif max_rate, decrease gain
                    trimmed_gain[n, dim] -= delta_gain * np.abs(delta_max)
                trimmed_gain[n, dim] = np.abs(trimmed_gain[n, dim])     
                    
        to_delete = []
        for n, nrn in enumerate(np.copy(nrn_idx)):
            if np.sum(losses[n]) < tol*dims:
                to_delete.append(n)
                bias[nrn] = trimmed_bias[n]
                gain[nrn] = trimmed_gain[n]                    
        trimmed_gain = np.delete(trimmed_gain, to_delete, axis=0)
        trimmed_bias = np.delete(trimmed_bias, to_delete, axis=0)
        trimmed_max_rates = np.delete(trimmed_max_rates, to_delete)
        trimmed_intercepts = np.delete(trimmed_intercepts, to_delete)
        trimmed_encoders = np.delete(trimmed_encoders, to_delete, axis=0)
        nrn_idx = np.delete(np.array(nrn_idx), to_delete)

    for n, nrn in enumerate(np.copy(nrn_idx)):
        bias[nrn] = trimmed_bias[n]
        gain[nrn] = trimmed_gain[n]

    return gain, bias

# dt = 0.0001
# t = np.arange(0, 20, dt)
# f = np.cumsum(1+0.5*np.sin(t))*dt
# y = np.sin(2*np.pi * f)
# fig, ax = plt.subplots(figsize=(16, 8))
# ax.plot(t, y)
# ax.plot(t, np.sin(2*np.pi*t))
# ax.plot(t, f)
# plt.show()