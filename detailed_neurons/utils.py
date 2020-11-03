import numpy as np
import nengo
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from nengo.neurons import *
from nengo.builder.neurons import *
from nengo.solvers import Lstsq, LstsqL2, NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
base.have_bson = False
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='white')

def decode(spikes, targets, nTrain, dt=0.001, dtSample=0.001, reg=1e-3, penalty=0, evals=100, name="default", tauRiseMax=3e-2, tauFallMax=3e-1):
    d, tauRise, tauFall = dfOpt(spikes, targets, nTrain, name=name,
        evals=evals, reg=reg, penalty=penalty, dt=dt, dtSample=dtSample,
        tauRiseMax=tauRiseMax, tauFallMax=tauFallMax)
    print("tauRise: %.3f, tauFall: %.3f"%(tauRise, tauFall))
    f = DoubleExp(tauRise, tauFall)
    A = np.zeros((0, spikes.shape[2]))
    Y = np.zeros((0, targets.shape[2]))
    for n in range(nTrain):
        A = np.append(A, f.filt(spikes[n], dt=dt), axis=0)
        Y = np.append(Y, targets[n], axis=0)
    X = np.dot(A, d)
    error = rmse(X, Y)
    d = d.reshape((-1, targets.shape[2]))
    return d, f, tauRise, tauFall, X, Y, error

def plotState(times, X, Y, error, prefix, suffix, t):
    fig, ax = plt.subplots()
    ax.plot(times, X, label="estimate")
    ax.plot(times, Y, label="target")
    ax.set(xlabel='time', ylabel='state', title='rmse=%.3f'%error, xlim=((0, t)), ylim=((-1, 1)))
    ax.legend(loc='upper left')
    sns.despine()
    fig.savefig("plots/%s_%s.pdf"%(prefix, suffix))
    plt.close('all')
    
def plotActivity(t, dt, fS, times, aEns, aTar, prefix, suffix):
    for n in range(aTar.shape[1]):
        fig, ax = plt.subplots()
        ax.plot(times, fS.filt(aTar[:,n], dt=dt), alpha=0.5, label='target')
        ax.plot(times, fS.filt(aEns[:,n], dt=dt), alpha=0.5, label='ensemble')
        ax.set(xlim=((0, t)), ylim=((0, 40)))
        plt.legend(loc='upper left')
        plt.savefig('tuning/%s_%s_%s.pdf'%(prefix, suffix, n))
        plt.close('all')

def dfOpt(spikes, target, nTrain, name='default', evals=100, seed=0, dt=0.001, dtSample=0.001, reg=1e-3,
        tauRiseMin=1e-3, tauRiseMax=1e-2, tauFallMin=1e-2, tauFallMax=2e-1, penalty=0):

    np.savez_compressed('data/%s_spikes.npz'%name, spikes=spikes)
    np.savez_compressed('data/%s_target.npz'%name, target=target)
    hyperparams = {}
    hyperparams['nTrain'] = nTrain
    hyperparams['name'] = name
    hyperparams['reg'] = reg
    hyperparams['dt'] = dt
    hyperparams['dtSample'] = dtSample
    hyperparams['tauRise'] = hp.uniform('tauRise', tauRiseMin, tauRiseMax)
    hyperparams['tauFall'] = hp.uniform('tauFall', tauFallMin, tauFallMax)

    def objective(hyperparams):
        tauRise = hyperparams['tauRise']
        tauFall = hyperparams['tauFall']
        dt = hyperparams['dt']
        dtSample = hyperparams['dtSample']
        f = DoubleExp(tauRise, tauFall)
        spikes = np.load('data/%s_spikes.npz'%hyperparams['name'])['spikes']
        targets = np.load('data/%s_target.npz'%hyperparams['name'])['target']
        A = np.zeros((0, spikes.shape[2]))
        Y = np.zeros((0, targets.shape[2]))
        for n in range(hyperparams['nTrain']):
            A = np.append(A, f.filt(spikes[n], dt=dt), axis=0)
            Y = np.append(Y, targets[n], axis=0)
        if dt != dtSample:
            A = A[::int(dtSample/dt)]
            Y = Y[::int(dtSample/dt)]
        d, _ = LstsqL2(reg=hyperparams['reg'])(A, Y)
        X = np.dot(A, d)
        loss = rmse(X, Y)
        loss += penalty * (10*tauRise + tauFall)
        return {'loss': loss, 'd': d, 'tauRise': tauRise, 'tauFall': tauFall, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)
    idx = np.argmin(trials.losses())
    best = trials.trials[idx]
    d = best['result']['d']
    tauRise = best['result']['tauRise']
    tauFall = best['result']['tauFall']
        
    return d, tauRise, tauFall


class WNode(nengo.Node):
    def __init__(self, conn, alpha, eMax, kA, tTrans=0.01, tStep=0.01, exc=False, inh=False, seed=0):
        self.conn = conn
        self.nPre = conn.pre.n_neurons
        self.nPost = conn.post.n_neurons
        self.check = 10
        self.size_in = self.nPre+2*self.nPost
        self.alpha = alpha
        self.eMax = eMax
        self.kA = kA
        self.tTrans = tTrans
        self.tStep = tStep
        self.exc = exc
        self.inh = inh
        self.rng = np.random.RandomState(seed=seed)
        self.preUpdate = np.arange(0, self.nPre)
        if self.exc and self.inh:
            raise "can't force excitatory and inhibitory weights"
        super().__init__(self.step, size_in=self.size_in, size_out=0)
    def step(self, t, x):
        if t < self.tTrans: return
        if t % self.tStep > 1e-6: return
        assert self.conn.weights.shape[0] == self.nPre
        assert self.conn.weights.shape[1] == self.nPost
        aPre = x[:self.nPre]
        aBio = x[self.nPre: self.nPre+self.nPost]
        aTar = x[self.nPre+self.nPost:]
        for post in range(self.nPost):
            dA = aBio[post] - aTar[post]
            if aTar[post] < 1e0:
                dA *= self.kA  # extra error if bio is active when target is silent
            volts = np.array([self.conn.v_recs[post][-n] for n in range(self.check)])
            if np.any(np.isnan(volts)): # no weight update if voltage is nan
                continue
            if len(np.where((volts > -40) & (volts < 5))[0]) == self.check:
                self.conn.e[:, post] *= 0.9  # reduce weights if voltage has numerical errors
                continue
            for pre in range(self.nPre):
                for dim in range(self.conn.d.shape[1]):
                    sign = -1.0 if self.conn.d[pre, dim] >= 0 else 1.0
                    dE = sign * self.alpha * aPre[pre]
                    e = self.conn.e[pre, post, dim] + dA * dE
                    if e < -self.eMax: e = -self.eMax
                    if e > self.eMax: e = self.eMax
                    self.conn.e[pre, post, dim] = e
                w = np.dot(self.conn.d[pre], self.conn.e[pre, post])
                if self.exc and w < 0: w = 0
                if self.inh and w > 0: w = 0
                self.conn.weights[pre, post] = w
                self.conn.netcons[pre, post].weight[0] = np.abs(w)
                self.conn.netcons[pre, post].syn().e = 0.0 if w > 0 else -70.0
        return
    
def learnEncoders(conn, tar, fS, alpha=1e-7, eMax=1e-1, kA=3, tStep=1e-2, exc=False, inh=False):
    node = WNode(conn, alpha=alpha, kA=kA, eMax=eMax, tStep=tStep, exc=exc, inh=inh)
    nengo.Connection(conn.pre.neurons, node[0:conn.pre.n_neurons], synapse=fS)
    nengo.Connection(conn.post.neurons, node[conn.pre.n_neurons: conn.pre.n_neurons+conn.post.n_neurons], synapse=fS)
    nengo.Connection(tar.neurons, node[conn.pre.n_neurons+conn.post.n_neurons:], synapse=fS)

def setWeights(conn, d, e):
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")  
    conn.d = d
    conn.e = e
    if np.any(d) and np.any(e):
        for pre in range(conn.pre.n_neurons):
            for post in range(conn.post.n_neurons):
                w = np.dot(d[pre], e[pre, post])
                conn.weights[pre, post] = w
                conn.netcons[pre, post].weight[0] = np.abs(w)
                conn.netcons[pre, post].syn().e = 0 if w > 0 else -70

def fitSinusoid(times, vals, freq, tTrans, muFreq=1, sigmaFreq=0.1, base=False, name="sinusoid", seed=0, evals=2000):
    np.savez_compressed('data/%s_times.npz'%name, times=times)
    np.savez_compressed('data/%s_vals.npz'%name, vals=vals)
    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['tTrans'] = tTrans
    hyperparams['freq'] = hp.normal('freq', muFreq, sigmaFreq)
    hyperparams['phase'] = hp.uniform('phase', 0, 2*np.pi)
#     hyperparams['mag'] = 0.75
#     hyperparams['base'] = 0
#     hyperparams['phase'] = hp.choice('phase', [0, 0.2, 0.4, 0.6, 0.8])
#     hyperparams['freq'] = hp.uniform('freq', fMin*freq, fMax*freq)
    hyperparams['mag'] = hp.choice('mag', [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2])
    hyperparams['base'] = hp.choice('base', [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]) if base else 0
#     hyperparams['mag'] = hp.uniform('mag', 0.7, 1.0)
#     hyperparams['base'] = hp.uniform('base', -0.2, 0.2)
#     hyperparams['mag'] = 0.8

    def objective(hyperparams):
        freq = hyperparams['freq']
        tTrans = hyperparams['tTrans']
        phase = hyperparams['phase']
        mag = hyperparams['mag']
        base = hyperparams['base']
        times = np.load('data/%s_times.npz'%hyperparams['name'])['times']
        sin = base + mag*np.sin(times*2*np.pi*freq+phase)
        vals = np.load('data/%s_vals.npz'%hyperparams['name'])['vals']
        loss = rmse(sin[tTrans:], vals[tTrans:])
        return {'loss': loss, 'freq': freq, 'phase': phase, 'mag': mag, 'base': base, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
#         algo=hyperopt.rand.suggest,
        max_evals=evals,
        trials=trials)
    idx = np.argmin(trials.losses())
    best = trials.trials[idx]
    freq = best['result']['freq']
    phase = best['result']['phase']
    mag = best['result']['mag']
    base = best['result']['base']
        
    return freq, phase, mag, base

class DownsampleNode(object):
    def __init__(self, size_in, size_out, dt, dtSample):
        self.dt = dt
        self.dtSample = dtSample
        self.size_in = size_in
        self.size_out = size_out
        self.ratio = int(self.dtSample/self.dt)
        self.output = np.zeros((int(self.size_in)))
        self.count = 0

    def __call__(self, t, x):
        if self.count == self.ratio:
            self.output *= 0
            self.count = 0
        self.output += x / self.ratio
        self.count += 1
        return self.output
    
def plot3D(X, Y, neuron_type, suffix):
    print("plotting 3D")
    fig = plt.figure()    
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax.plot(*Y.T, linewidth=0.25)
    ax2.plot(*X.T, linewidth=0.25)
    ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-20, 20)), zlim=((0, 40)))
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    ax2.set(xlabel="x", ylabel="y", zlabel="z", xlim=((-20, 20)), ylim=((-20, 20)), zlim=((0, 40)))
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor('w')
    ax2.yaxis.pane.set_edgecolor('w')
    ax2.zaxis.pane.set_edgecolor('w')
    ax2.grid(False)
    plt.savefig("plots/lorenzNew_%s_%s_3D.pdf"%(neuron_type, suffix))

def plotPairwise(X, Y, neuron_type, suffix):
    print("Plotting pairwise")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(Y[:,0], Y[:,1], linestyle="--", linewidth=0.25)
    ax2.plot(Y[:,1], Y[:,2], linestyle="--", linewidth=0.25)
    ax3.plot(Y[:,0], Y[:,2], linestyle="--", linewidth=0.25)
    ax1.plot(X[:,0], X[:,1], linewidth=0.25)
    ax2.plot(X[:,1], X[:,2], linewidth=0.25)
    ax3.plot(X[:,0], X[:,2], linewidth=0.25)
    ax1.set(xlabel='x', ylabel='y', xlim=((-20, 20)), ylim=((-20, 20)))
    ax2.set(xlabel='y', ylabel='z', xlim=((-20, 20)), ylim=((0, 40)))
    ax3.set(xlabel='x', ylabel='z', xlim=((-20, 20)), ylim=((0, 40)))
    plt.tight_layout()
    plt.savefig("plots/lorenzNew_%s_%s_Pairwise.pdf"%(neuron_type, suffix))
    plt.close('all')

def plotTent(A, Y, d2, neuron_type, suffix, trans=1000):
    print('Plotting tent map')
    X2 = np.dot(A, d2)[trans:]
    Y2 = Y[trans:]
    zTarPeaks = find_peaks(Y2[:,2], height=0)[0][1:]
    zTarValuesH = np.ravel(Y2[zTarPeaks, 2][:-1])
    zTarValuesV = np.ravel(Y2[zTarPeaks, 2][1:])
    zEnsPeaks = find_peaks(X2[:,2], height=0)[0][1:]
    zEnsValuesH = np.ravel(X2[zEnsPeaks, 2][:-1])
    zEnsValuesV = np.ravel(X2[zEnsPeaks, 2][1:])
    error = 0
    for i in range(len(zEnsPeaks)-1):
        minR = np.inf
        for j in range(len(zTarPeaks)-1):
            r = np.sqrt(
                np.square(zTarValuesH[j] - zEnsValuesH[i]) +
                np.square(zTarValuesV[j] - zEnsValuesV[i]))
            minR = np.min([minR, r])
        error += minR
    error /= len(zEnsPeaks)
    fig, ax = plt.subplots()
    ax.scatter(zTarValuesH, zTarValuesV, alpha=0.5, color='r', label='target')
    ax.scatter(zEnsValuesH, zEnsValuesV, alpha=0.5, color='b', label='ens')
    ax.set(xlabel=r'$\mathrm{max}_n (z)$', ylabel=r'$\mathrm{max}_{n+1} (z)$', title='error=%.5f'%error)
    plt.legend(loc='upper right')
    plt.savefig("plots/lorenzNew_%s_%s_Tent.pdf"%(neuron_type, suffix))
    return error