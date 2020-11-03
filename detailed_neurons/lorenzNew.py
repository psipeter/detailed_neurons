import numpy as np
import nengo
from scipy.ndimage import gaussian_filter1d
# from scipy.optimize import curve_fit
# from scipy.signal import find_peaks
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity, DownsampleNode, plot3D, plotPairwise, plotTent
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import neuron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set(context='paper', style='whitegrid')

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

def go(N=3000, d1=None, f1=None, t=100, l=False, neuron_type=LIF(),
       m=Uniform(30, 30), i=Uniform(-0.8, 0.8), r=30, IC=np.array([1,1,1]),
       seed=0, dt=0.001, dtSample=0.001, f=DoubleExp(1e-3, 1e-1)):

    if not f1: f1 = f
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(lambda t: IC*(t<=1.0))
        tar = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ens = nengo.Ensemble(N, 3, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=r)
        dss = nengo.Node(DownsampleNode(size_in=N, size_out=N, dt=dt, dtSample=dtSample), size_in=N, size_out=N)
        nengo.Connection(inpt, tar, synapse=None)
        nengo.Connection(tar, tar, function=feedback, synapse=~s)
        if l:
            nengo.Connection(tar, ens, synapse=f, seed=seed)
        else:
            nengo.Connection(inpt, ens, synapse=None, seed=seed)
        nengo.Connection(ens, ens, synapse=f1, solver=NoSolver(d1), seed=seed)
        nengo.Connection(ens.neurons, dss, synapse=None)
        pTar = nengo.Probe(tar, synapse=None, sample_every=dtSample)
        pEns = nengo.Probe(dss, synapse=None, sample_every=dtSample)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)

    return dict(
        times=sim.trange(),
        tar=sim.data[pTar],
        ens=sim.data[pEns])

def run(N=3000, neuron_type=LIF(), tTrain=100, tTest=200, tTransTrain=0, tTransTest=20, nTrain=5, nTest=5, f=DoubleExp(1e-3, 1e-1), dt=0.001, dtSampleTrain=0.003, dtSampleTest=0.01, seed=0, f2=10, reg=1e-2, tauRiseMax=1e-2, evals=20, r=30, load=False, file=None):

    print('\nNeuron Type: %s'%neuron_type)
    rng = np.random.RandomState(seed=seed)
    if load:
        d1 = np.load(file)['d1']
        d2 = np.load(file)['d2']
        tauRise1 = np.load(file)['tauRise1']
        tauFall1 = np.load(file)['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('decoders for ens')
        spikes = np.zeros((nTrain, int((tTrain-tTransTrain)/dtSampleTrain), N))
        spikes2 = np.zeros((nTrain, int((tTrain-tTransTrain)/dtSampleTrain), N))
        targets = np.zeros((nTrain, int((tTrain-tTransTrain)/dtSampleTrain), 3))
        targets2 = np.zeros((nTrain, int((tTrain-tTransTrain)/dtSampleTrain), 3))
        for n in range(nTrain):
            IC = np.array([rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(10, 35)])
            data = go(N=N, neuron_type=neuron_type, l=True, t=tTrain, f=f, r=r, dt=dt, dtSample=dtSampleTrain, seed=seed, IC=IC)
            spikes[n] = data['ens'][int(tTransTrain/dtSampleTrain):]
            spikes2[n] = gaussian_filter1d(data['ens'], sigma=f2, axis=0)[int(tTransTrain/dtSampleTrain):]
            targets[n] = f.filt(data['tar'], dt=dtSampleTrain)[int(tTransTrain/dtSampleTrain):]
            targets2[n] = gaussian_filter1d(data['tar'], sigma=f2, axis=0)[int(tTransTrain/dtSampleTrain):]
        d1, f1, tauRise1, tauFall1, X, Y, error = decode(spikes, targets, nTrain, dt=dtSampleTrain, dtSample=dtSampleTrain, name="lorenzNew", evals=evals, tauRiseMax=tauRiseMax)
        spikes2 = spikes2.reshape((int((tTrain-tTransTrain)/dtSampleTrain)*nTrain), N)
        targets2 = targets2.reshape((int((tTrain-tTransTrain)/dtSampleTrain)*nTrain, 3))
        d2, _ = nengo.solvers.LstsqL2(reg=reg)(spikes2, targets2)
        np.savez("data/lorenzNew_%s.npz"%neuron_type, d1=d1, d2=d2, tauRise1=tauRise1, tauFall1=tauFall1, f2=f2)
        
        plot3D(X, Y, neuron_type, suffix="train")
        plotPairwise(X, Y, neuron_type, suffix="train")
        error = plotTent(spikes2, targets2, d2, neuron_type, suffix="train")
    
    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        IC = np.array([rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(10, 35)])
        data = go(d1=d1, f1=f1, N=N, neuron_type=neuron_type, t=tTest, f=f, r=r, dt=dt, dtSample=dtSampleTest, seed=seed, IC=IC)
        A = f1.filt(data['ens'], dt=dtSampleTest)
        X = np.dot(A, d1)[int(tTransTest/dtSampleTest):]
        Y = f.filt(data['tar'], dt=dtSampleTest)[int(tTransTest/dtSampleTest):]
        X2 = gaussian_filter1d(data['ens'], sigma=f2, axis=0)[int(tTransTest/dtSampleTest):]
        Y2 = gaussian_filter1d(data['tar'], sigma=f2, axis=0)[int(tTransTest/dtSampleTest):]
        plot3D(X, Y, neuron_type, suffix="test%s"%test)
        plotPairwise(X, Y, neuron_type, suffix="test%s"%test)
        error = plotTent(X2, Y2, d2, neuron_type, suffix="test%s"%test)
        errors[test] = error
    print('errors: ', errors)
    np.savez("data/lorenzNew_%s.npz"%neuron_type, d1=d1, d2=d2, tauRise1=tauRise1, tauFall1=tauFall1, f2=f2, errors=errors)
    return errors

errorsLIF = run(neuron_type=LIF(), N=2000, tTrain=300, nTrain=3, nTest=10, load=False, file="data/lorenzNew_LIF().npz")
# errorsALIF = run(neuron_type=ALIF(), load=True, file="data/lorenzNew_ALIF().npz")
#errorsWilson = run(neuron_type=Wilson(), dt=1e-4, r=25, load=False, file="data/lorenzNew_Wilson().npz")
