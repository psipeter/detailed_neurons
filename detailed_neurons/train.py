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
from nengo.solvers import Lstsq, LstsqL2, NoSolver
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

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

from neuron_models import AdaptiveLIFT, WilsonEuler, DurstewitzNeuron, reset_neuron
# from utils import bin_activities_values_single

# import neuron
# neuron.h.load_file('NEURON/durstewitz.hoc')
# neuron.h.load_file('stdrun.hoc')


__all__ = ['norms', 'd_opt', 'df_opt', 'LearningNode']

    
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

def df_opt(x, ens, f, name='default', df_evals=100, seed=0, dt=0.001, dt_sample=0.001, reg=0,
        tau_rise=[1e-2, 1e-1], tau_fall=[1e-2, 3e-1], penalty=0, algo=tpe.suggest):  # rand.suggest):#

    x = f.filt(x, dt=dt)
    np.savez_compressed('data/%s_ens.npz'%name, ens=ens)
    np.savez_compressed('data/%s_x.npz'%name, x=x)
    del(ens)
    del(x)

    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['reg'] = reg
    hyperparams['dt'] = dt
    hyperparams['dt_sample'] = dt_sample
    hyperparams['tau_rise'] = hp.uniform('tau_rise', tau_rise[0], tau_rise[1])
    hyperparams['tau_fall'] = hp.uniform('tau_fall', tau_fall[0], tau_fall[1])

    def objective(hyperparams):
        taus_ens = [hyperparams['tau_rise'], hyperparams['tau_fall']]
        h_ens = DoubleExp(taus_ens[0], taus_ens[1])
        A = h_ens.filt(np.load('data/%s_ens.npz'%hyperparams['name'])['ens'], dt=hyperparams['dt'])
        x = np.load('data/%s_x.npz'%hyperparams['name'])['x']
        if dt != dt_sample:
            A = A[::int(dt_sample/dt)]
            x = x[::int(dt_sample/dt)]
        if hyperparams['reg']:
            d_ens = LstsqL2(reg=hyperparams['reg'])(A, x)[0]
        else:
            d_ens = Lstsq()(A, x)[0]
        xhat = np.dot(A, d_ens)
        loss = nrmse(xhat, target=x)
        loss += penalty * (10*taus_ens[0] + taus_ens[1])
        return {'loss': loss, 'taus_ens': taus_ens, 'd_ens': d_ens, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=algo,
        max_evals=df_evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
    taus_ens = best['result']['taus_ens']
    d_ens = best['result']['d_ens']
    h_ens = DoubleExp(taus_ens[0], taus_ens[1])
        
    return d_ens, h_ens, taus_ens

def df_opt_hx(x, ens, name='default', df_evals=100, seed=0, dt=0.001, dt_sample=0.001,
        tau_rise=1e-3, tau_fall=[3e-2, 3e-1], penalty=0.25, algo=tpe.suggest):  # rand.suggest):#

    np.savez_compressed('data/%s_ens.npz'%name, ens=ens)
    np.savez_compressed('data/%s_x.npz'%name, x=x)
    del(ens)
    del(x)
    
    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['dt'] = dt
    hyperparams['dt_sample'] = dt_sample
    hyperparams['tau_rise'] = tau_rise
    # hyperparams['ens'] = hp.loguniform('ens', np.log10(tau_fall[0]), np.log10(tau_fall[1]))
    # hyperparams['x'] = hp.loguniform('x', np.log10(tau_fall[0]), np.log10(tau_fall[1]))
    hyperparams['ens'] = hp.uniform('ens', tau_fall[0], tau_fall[1])
    hyperparams['x'] = hp.uniform('x', tau_fall[0], tau_fall[1])

    def objective(hyperparams):
        taus_ens = [hyperparams['tau_rise'], hyperparams['ens']]
        taus_x = [hyperparams['tau_rise'], hyperparams['x']]
        h_ens = DoubleExp(taus_ens[0], taus_ens[1])
        h_x = DoubleExp(taus_x[0], taus_x[1])
        A = h_ens.filt(np.load('data/%s_ens.npz'%hyperparams['name'])['ens'], dt=hyperparams['dt_sample'])
        x = h_x.filt(np.load('data/%s_x.npz'%hyperparams['name'])['x'], dt=hyperparams['dt_sample'])
        if dt != dt_sample:
            A = A[::int(dt_sample/dt)]
            x = x[::int(dt_sample/dt)]
        d_ens = Lstsq()(A, x)[0]
        xhat = np.dot(A, d_ens)
        loss = nrmse(xhat, target=x)
        loss += penalty * taus_ens[1]
        return {'loss': loss, 'taus_ens': taus_ens, 'taus_x': taus_x, 'd_ens': d_ens, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=algo,
        max_evals=df_evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
    taus_ens = best['result']['taus_ens']
    taus_x = best['result']['taus_x']
    d_ens = best['result']['d_ens']
    h_ens = DoubleExp(taus_ens[0], taus_ens[1])
    h_x = DoubleExp(taus_x[0], taus_x[1])
        
    return d_ens, h_ens, taus_ens, h_x, taus_x


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

class LearningNode(nengo.Node):
    def __init__(self, N, N_pre, dim, conn, k=1e-5, w_max=2e-4, decay=lambda t: 1, seed=0):
        self.N = N
        self.N_pre = N_pre
        self.dim = dim
        self.conn = conn
        self.w_max = w_max
        self.size_in = 2*N+N_pre+dim
        self.size_out = 0
        self.k = k
        self.decay = decay
        self.rng = np.random.RandomState(seed=seed)
        super(LearningNode, self).__init__(
            self.step, size_in=self.size_in, size_out=self.size_out)

    def step(self, t, x):
        a_pre = x[:self.N_pre]
        a_bio = x[self.N_pre: self.N_pre+self.N]
        a_supv = x[self.N_pre+self.N:]
        u = x[-self.dim:]
        pre = self.rng.randint(0, self.conn.weights.shape[0])
#         print(np.sum(self.conn.e))
        for post in range(self.conn.weights.shape[1]):
            delta_a = a_bio[post] - a_supv[post]
            # if a_bio[post] == 0 or a_supv[post] == 0:
            #     delta_a *= 2
            for dim in range(self.conn.d.shape[1]):
                dim_scale = 1 if np.sum(np.abs(u)) == 0 else np.abs(u[dim])/np.sum(np.abs(u))
                if self.conn.d[pre, dim] >= 0:
                    delta_e = -self.k * a_pre[pre] * dim_scale * self.decay(t)
                if self.conn.d[pre, dim] < 0:
                    delta_e = self.k * a_pre[pre] * dim_scale * self.decay(t)
                self.conn.e[pre, post, dim] += delta_a * delta_e
            self.conn.weights[pre, post] = np.dot(self.conn.d[pre], self.conn.e[pre, post])
            if self.conn.weights[pre, post] > self.w_max:
                self.conn.weights[pre, post] = self.w_max
                self.conn.e[pre, post] *= 0.8
            if self.conn.weights[pre, post] < -self.w_max:
                self.conn.weights[pre, post] = -self.w_max
                self.conn.e[pre, post] *= 0.8
#                 self.conn.weights[pre, post] += delta_a * -self.k * a_pre[pre]
            self.conn.netcons[pre, post].weight[0] = np.abs(self.conn.weights[pre, post])
            # print(np.abs(self.conn.weights[pre, post]))
            self.conn.netcons[pre, post].syn().e = 0.0 if self.conn.weights[pre, post] > 0 else -70.0
        return
    
    
class LearningNode2(nengo.Node):
    def __init__(self, N, N_pre, conn, k=1e-5, seed=0):
        self.N = N
        self.N_pre = N_pre
        self.conn = conn
        self.check = 10
        self.size_in = 2*N+N_pre
        self.size_out = 0
        self.k = k
        self.rng = np.random.RandomState(seed=seed)
        super(LearningNode2, self).__init__(
            self.step, size_in=self.size_in, size_out=self.size_out)
    def step(self, t, x):
        if t < 0.01: return
        a_pre = x[:self.N_pre]
        a_bio = x[self.N_pre: self.N_pre+self.N]
        a_supv = x[self.N_pre+self.N:]
        pre = self.rng.randint(0, self.conn.weights.shape[0])
        for post in range(self.conn.weights.shape[1]):
            volts = np.array([self.conn.v_recs[post][-n] for n in range(self.check)])
#             volts2 = np.array([self.conn.v2_recs[post][-n] for n in range(self.check)])
            if np.any(np.isnan(volts)):  # crash check
#                 print('volts', volts[-1])
                continue  # no weight update
            elif a_bio[post] > 40: # oversaturation condition 1
                for pp in range(self.conn.weights.shape[0]):
                    self.conn.e[pp, post] *= 0.9
            elif len(np.where((volts > -40) & (volts < 5))[0]) == self.check:  # oversaturation condition 2
                for pp in range(self.conn.weights.shape[0]):
                    self.conn.e[pp, post] *= 0.9
            else:  # encoder/weight update
                delta_a = a_bio[post] - a_supv[post]
                for dim in range(self.conn.d.shape[1]):
                    sign = -1 if self.conn.d[pre, dim] >= 0 else 1
                    delta_e = sign * self.k * a_pre[pre]
                    self.conn.e[pre, post, dim] += delta_a * delta_e
            w = np.dot(self.conn.d[pre], self.conn.e[pre, post])
            self.conn.weights[pre, post] = w 
            self.conn.netcons[pre, post].weight[0] = np.abs(w)
            self.conn.netcons[pre, post].syn().e = 0.0 if w > 0 else -70.0
        return
    
#     # optimize T_ff by trying sample values and seeing if any decoders d_out will match target=x=integral(u)
#     if isinstance(neuron_type, DurstewitzNeuron):
#         print("Optimizing T_ff and d_out")
#         hyperparams = {}
#         hyperparams['reg'] = reg
#         hyperparams['n_neurons'] = n_neurons
#         hyperparams['t'] = t
#         hyperparams['dt'] = dt
#         hyperparams['neuron_type'] = neuron_type
#         hyperparams['stim_func'] = stim_func
#         hyperparams['T_ff'] = hp.uniform('T_ff', 0.1, 0.3)
# #         hyperparams['T_ff'] = 0.175
#         np.savez("data/Toptimize.npz", d_ens=d_ens, taus_ens=taus_ens,
#             taus_f=np.array([-1./f.poles[0], -1./f.poles[1]]), w_pre=w_pre, w_ens=w_ens)

#         def objective(hyperparams):
#             T_ff = hyperparams['T_ff']
#             print("testing T_ff=%.3f, T_fb=%.3f"%(T_ff, T_fb))
#             d_ens = np.load("data/Toptimize.npz")['d_ens']
#             f_ens = DoubleExp(np.load("data/Toptimize.npz")['taus_ens'][0], np.load("data/Toptimize.npz")['taus_ens'][1])
#             f = DoubleExp(np.load("data/Toptimize.npz")['taus_f'][0], np.load("data/Toptimize.npz")['taus_f'][1])
#             w_pre = np.load("data/Toptimize.npz")['w_pre']
#             w_post = np.load("data/Toptimize.npz")['w_ens']
#             data = go(
#                 d_ens,
#                 f_ens,
#                 n_neurons=hyperparams['n_neurons'],
#                 t=hyperparams['t'],
#                 f=f,
#                 dt=hyperparams['dt'],
#                 neuron_type=hyperparams['neuron_type'],
#                 stim_func=hyperparams['stim_func'],
#                 T_ff=T_ff,
#                 T_fb=1.0,
#                 w_pre=w_pre,
#                 w_ens=w_ens,
#                 progress_bar=True)
#             a_ens = f_ens.filt(data['ens'], dt=hyperparams['dt'])
#             target = f.filt(data['x'], dt=hyperparams['dt'])
#             d_out = nengo.solvers.LstsqL2(reg=hyperparams['reg'])(a_ens, target)[0]
#             xhat_ens = np.dot(a_ens, d_out)
#             loss = nrmse(xhat_ens, target=target)
#             fig, ax = plt.subplots()
#             ax.plot(data['times'], target, linestyle="--", label='target')
#             ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %loss)
#             ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="unsupervised\n T_ff=%.3f"%(T_ff))
#             plt.legend(loc='upper right')
#             plt.savefig("plots/integrate_%s_ens_Tff_%.3f.pdf"%(neuron_type, T_ff))
#             return {'loss': loss, 'T_ff': T_ff, 'd_out': d_out, 'status': STATUS_OK}

#         trials = Trials()
#         fmin(objective,
#             rstate=np.random.RandomState(seed=0),
#             space=hyperparams,
#             algo=tpe.suggest,
#             max_evals=T_evals,
#             trials=trials)
#         best_idx = np.argmin(trials.losses())
#         best = trials.trials[best_idx]
#         T_ff = best['result']['T_ff']
#         d_out = best['result']['d_out']
#         print("Final T_ff:", T_ff)
#         print("Final d_out:", d_out)

    
#     # optimize T_ff by trying sample values and seeing if any decoders d_out will match target=x=integral(u)
#     if isinstance(neuron_type, DurstewitzNeuron):
#         print("Optimizing T_ff and d_out")
#         hyperparams = {}
#         hyperparams['reg'] = reg
#         hyperparams['n_neurons'] = n_neurons
#         hyperparams['t'] = t
#         hyperparams['dt'] = dt
#         hyperparams['neuron_type'] = neuron_type
#         hyperparams['stim_func'] = stim_func
#         hyperparams['T_ff'] = hp.uniform('T_ff', 0.1, 0.3)
# #         hyperparams['T_ff'] = 0.175
#         np.savez("data/Toptimize.npz", d_ens=d_ens, taus_ens=taus_ens,
#         np.savez('data/integrate_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens)

#         times = np.arange(0, 1, 0.0001)
#         fig, ax = plt.subplots()
#         ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
#         ax.plot(times, f_ens.impulse(len(times), dt=0.0001), label=r"$f^{ens}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
#            %(-1./f_ens.poles[0], -1./f_ens.poles[1], np.count_nonzero(d_ens), n_neurons))
#         ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
#         ax.legend(loc='upper right')
#         plt.tight_layout()
#         plt.savefig("plots/integrate_%s_filters_ens.pdf"%neuron_type)

#         a_ens = f_ens.filt(data['ens'], dt=dt)
#         xhat_ens = np.dot(a_ens, d_ens)
#         target = f.filt(f.filt(data['u'], dt=dt), dt=dt)
#         nrmse_ens = nrmse(xhat_ens, target=target)
#         fig, ax = plt.subplots()
#         ax.plot(data['times'], target, linestyle="--", label='target')
#         ax.plot(data['times'], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
#         ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="supervised")
#         plt.legend(loc='upper right')
#         plt.savefig("plots/integrate_%s_ens_train.pdf"%neuron_type)


#             target = np.zeros((n_trains*int(t/dt), 1))
#             spikes = np.zeros((n_trains*int(t/dt), n_neurons))
#             for n in range(n_trains):
#                 stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=1, f=f, normed='x', seed=n)
#                 data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=1.0, w_ff=w_ff_small, w_fb=w_fb)
#                 target[n*int(t/dt): (n+1)*int(t/dt)] = data['x']
#                 spikes[n*int(t/dt): (n+1)*int(t/dt)] = data['ens']
#             print('optimizing readout decoders')
#             d_out, f_out, taus_out = df_opt(target, spikes, f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)
#             np.savez('data/integrate_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out=d_out, taus_out=taus_out)

#     if readout:
#         if load_fd:
#             load = np.load(load_fd)
#             d_out = load['d_out']
#             taus_out = load['taus_out']
#             f_out = DoubleExp(taus_out[0], taus_out[1])
#         else:
#             print('optimizing readout decoders')
#             stim_func = make_normed_flipped(value=1.0, t=t, dt=dt, N=n_trains, f=f, normed='x', seed=0)
#             data = go(d_ens, f_ens, n_neurons=n_neurons, t=t*n_trains, f=f, dt=dt, neuron_type=neuron_type, stim_func=stim_func, T_ff=T_ff, T_fb=1.0, w_ff=w_ff_small, w_ff2=w_ff_large, w_fb=w_fb, supervised=True)
#             target = f.filt(data['x'], dt=dt)
#     #             target = f.filt(f.filt(data['x']))
#             spikes = data['ens']
#             d_out, f_out, taus_out = df_opt(target, spikes, f, dt=dt, penalty=penalty, reg=reg, name='integrate_%s'%neuron_type)
#             np.savez('data/integrate_%s_fd.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out=d_out, taus_out=taus_out)

#             times = np.arange(0, 1, 0.0001)
#             fig, ax = plt.subplots()
#             ax.plot(times, f.impulse(len(times), dt=0.0001), label=r"$f^x, \tau_1=%.3f, \tau_2=%.3f$" %(-1./f.poles[0], -1./f.poles[1]))
#             ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
#                %(-1./f_out.poles[0], -1./f_out.poles[1], np.count_nonzero(d_out), n_neurons))
#             ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
#             ax.legend(loc='upper right')
#             plt.tight_layout()
#             plt.savefig("plots/integrate_%s_filters_out.pdf"%neuron_type)

#             a_ens = f_out.filt(spikes, dt=dt)
#             target = f.filt(target, dt=dt)
#             xhat_ens = np.dot(a_ens, d_out)
#             nrmse_ens = nrmse(xhat_ens, target=target)
#             fig, ax = plt.subplots()
#             ax.plot(dt*np.arange(0, len(target), 1), target, linestyle="--", label='target')
#             ax.plot(dt*np.arange(0, len(target), 1), xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
#             ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train readout")
#             plt.legend(loc='upper right')
#             plt.savefig("plots/integrate_x_%s_ens_train_readout.pdf"%neuron_type)
#     else:
#         d_out = d_ens
#         f_out = f_ens




#     print('gathering training data for readout filters and decoders')
#     data = go(d_ens, f_ens, n_neurons=n_neurons, t=t, f=f, dt=dt, neuron_type=neuron_type,
#         f_smooth=f_smooth, w_pre=w_pre, w_ens=w_ens)
#     d_out, f_out1, taus_out = df_opt(data['u'][10000:], data['ens'][10000:], f_out, dt=dt, name='oscillate_out_%s'%neuron_type, reg=reg, penalty=0)
#     np.savez('data/oscillate_%s_df.npz'%neuron_type, d_ens=d_ens, taus_ens=taus_ens, d_out=d_out, taus_out=taus_out)

#     fig, ax = plt.subplots()
#     sns.distplot(d_out.ravel())
#     ax.set(xlabel='decoders', ylabel='frequency')
#     plt.savefig("plots/oscillate_%s_d_out.pdf"%neuron_type)

#     times = np.arange(0, 1, 0.0001)
#     fig, ax = plt.subplots()
#     ax.plot(times, f_out.impulse(len(times), dt=0.0001), label=r"$f^{out}, \tau_1=%.3f, \tau_2=%.3f$"
#         %(-1./f_out.poles[0], -1./f_out.poles[1]))
#     ax.plot(times, f_out1.impulse(len(times), dt=0.0001), label=r"$f^{out1}, \tau_1=%.3f, \tau_2=%.3f, d: %s/%s$"
#        %(-1./f_out1.poles[0], -1./f_out1.poles[1], np.count_nonzero(d_out), n_neurons))
#     ax.set(xlabel='time (seconds)', ylabel='impulse response', ylim=((0, 10)))
#     ax.legend(loc='upper right')
#     plt.tight_layout()
#     plt.savefig("plots/oscillate_%s_filters_out.pdf"%neuron_type)

#     x = f_out.filt(data['u'], dt=dt)[10000:]
#     a_ens = f_out1.filt(data['ens'], dt=dt)
#     xhat_ens = np.dot(a_ens, d_out)[10000:]
#     nrmse_ens = nrmse(xhat_ens, target=x)
#     fig, ax = plt.subplots()
#     ax.plot(data['times'][10000:], x, linestyle="--", label='x')
#     ax.plot(data['times'][10000:], xhat_ens, label='ens, nrmse=%.3f' %nrmse_ens)
#     ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="ens_ens")
#     plt.legend(loc='upper right')
#     plt.savefig("plots/oscillate_%s_train.pdf"%neuron_type)
    
