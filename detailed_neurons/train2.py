import numpy as np

from scipy.linalg import block_diag
from scipy.integrate import ode
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm

import h5py

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

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
base.have_bson = False

import os
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

from neuron_models2 import AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron
# from utils import bin_activities_values_single

import neuron
neuron.h.load_file('NEURON/durstewitz.hoc')
neuron.h.load_file('stdrun.hoc')


__all__ = ['norms', 'downsample_spikes', 'DownsampleNode', 'df_opt', 'LearningNode']

def norms(t, dt=0.001, stim_func=lambda t: np.cos(t), f=None, value=1.0):
    with nengo.Network() as model:
        u = nengo.Node(stim_func)
        p = nengo.Probe(u, synapse=f)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    norm = value / np.max(np.abs(sim.data[p]))
    return norm

def downsample_spikes(array, dt, dt_sample):
    ratio = int(dt_sample/dt)
    timesteps = int(array.shape[0]/ratio)
    a = np.zeros((timesteps, array.shape[1]))
    for t in range(timesteps):
        # a[t] = (np.sum(array[t*ratio:(t+1)*ratio], axis=0) > 0) / dt_sample
        # a[t] = np.sum(array[t*ratio:(t+1)*ratio], axis=0)
        a[t] = np.mean(array[t*ratio:(t+1)*ratio], axis=0)
    return a

class DownsampleNode(object):
    def __init__(self, size_in, size_out, dt, dt_sample):
        self.dt = dt
        self.dt_sample = dt_sample
        self.size_in = size_in
        self.size_out = size_out
        self.spikes = np.zeros((int(self.dt_sample/self.dt), int(self.size_in)))
        self.output = np.zeros((int(self.size_in)))
        self.count = 0
        self.reset = int(self.dt_sample/self.dt)

    def __call__(self, t, x):
        self.spikes[self.count] = x
        self.output = np.mean(self.spikes, axis=0)
        self.count = (self.count + 1) % self.reset
        return self.output

def df_opt(target, spikes, f, name='default', order=1, df_evals=100, seed=0, dt=0.001, dt_sample=0.001,
        tau_mins=[1e-3, 1e-4], tau_maxs=[5e-1, 1e-3], reg=1e-1):

    target = f.filt(target, dt=dt)
#     spk_file = h5py.File('data/%s_spk.h5'%name, 'w')
#     tar_file = h5py.File('data/%s_tar.h5'%name, 'w')
#     spk_file.create_dataset('spk', data=spikes)
#     tar_file.create_dataset('tar', data=target)
#     spk_file.close()
#     tar_file.close()
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
#         spk_file = h5py.File('data/%s_spk.h5'%name, 'r')
#         spk = np.array(spk_file.get('spk'))
        spk = np.load('data/%s_spk.npz'%hyperparams['name'])['spikes']  # , mmap_mode='r'
        act = h.filt(spk, dt=hyperparams['dt'])
#         tar_file = h5py.File('data/%s_tar.h5'%name, 'r')
#         tar = np.array(tar_file.get('tar'))
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
    best_taus = best['result']['taus']

    if order == 1:
        h_new = Lowpass(best_taus[0])
    elif order == 2:
        h_new = DoubleExp(best_taus[0], best_taus[1])
    d_new = LstsqL2(reg=best['result']['reg'])(
        h_new.filt(np.load('data/%s_spk.npz'%name)['spikes'], dt=dt),
        np.load('data/%s_tar.npz'%name)['target'])[0]
        
    return d_new, h_new, best_taus


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


def fit_sinusoid(xhat, times, evals=2000, freq=0.5, dt=0.001, seed=0):
    hyperparams = {}
    hyperparams['a'] = hp.uniform('a', 0.5, 2.0)
    hyperparams['b'] = hp.uniform('b', freq*np.pi, 3*freq*np.pi)
    hyperparams['c'] = hp.uniform('c', 0, 2*np.pi)

    def objective(hyperparams):
        a = hyperparams['a']
        b = hyperparams['b']
        c = hyperparams['c']
        sinusoid = a*np.sin(b*times+c)
        loss = nrmse(xhat, target=sinusoid)
        return {'loss': loss, 'a': a, 'b': b, 'c': c, 'status': STATUS_OK }
        
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
    best_a = best['result']['a']
    best_b = best['result']['b']
    best_c = best['result']['c']

    freq_error = np.abs(best_b - 2*np.pi*freq) / (2*np.pi*freq) + np.abs(best_a - 1)
    return best_a, best_b, best_c, freq_error


class LearningNode(nengo.Node):
    def __init__(self, N, N_pre, dim, conn, k=5e-2, dt=0.001, learn=True, seed=0):
        self.N = N
        self.N_pre = N_pre
        self.dim = dim
        self.conn = conn
        self.size_in = 2*N+N_pre+dim
        self.size_out = 0
        self.dt = dt
        self.k = k
        self.learn = learn
        self.rng = np.random.RandomState(seed=seed)
        super(LearningNode, self).__init__(
            self.step, size_in=self.size_in, size_out=self.size_out)

    def step(self, t, x):
        a_pre = x[:self.N_pre]
        a_bio = x[self.N_pre: self.N_pre+self.N]
        a_lif = x[self.N_pre+self.N:]
        u = x[-self.dim:]
        pre = self.rng.randint(0, self.conn.weights.shape[0])
        for post in range(self.conn.weights.shape[1]):
            delta_a = a_bio[post] - a_lif[post]
            for dim in range(self.conn.d.shape[1]):
                dim_scale = 1 if np.sum(np.abs(u)) == 0 else np.abs(u[dim])/np.sum(np.abs(u))
                if self.conn.d[pre, dim] >= 0:
                    delta_e = -self.k * a_pre[pre] * dim_scale
                if self.conn.d[pre, dim] < 0:
                    delta_e = self.k * a_pre[pre] * dim_scale
                self.conn.e[pre, post, dim] += delta_a * delta_e
            self.conn.weights[pre, post] = np.dot(self.conn.d[pre], self.conn.e[pre, post])
#                 self.conn.weights[pre, post] += delta_a * -self.k * a_pre[pre]
            self.conn.netcons[pre, post].weight[0] = np.abs(self.conn.weights[pre, post])
            self.conn.netcons[pre, post].syn().e = 0.0 if self.conn.weights[pre, post] > 0 else -70.0
        return