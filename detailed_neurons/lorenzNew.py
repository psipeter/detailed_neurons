import numpy as np
import nengo
from scipy.ndimage import gaussian_filter1d
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity, DownsampleNode, plot3D, plotPairwise, plotTent, plotLorenz
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

def go(N=2000, d=None, f=None, t=100, l=False, neuron_type=LIF(),
       m=Uniform(30, 30), i=Uniform(-1, 1), r=30, IC=np.array([0,0,0]),
       seed=0, dt=0.001, dtSample=0.001):

    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(lambda t: IC*(t<=1.0))
        tar = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ens = nengo.Ensemble(N, 3, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=r)
        dss = nengo.Node(DownsampleNode(size_in=N, size_out=N, dt=dt, dtSample=dtSample), size_in=N, size_out=N)
        nengo.Connection(inpt, tar, synapse=None)
        nengo.Connection(tar, tar, function=feedback, synapse=~s)
        if l:
            nengo.Connection(tar, ens, synapse=None, seed=seed)
        else:
            nengo.Connection(inpt, ens, synapse=None, seed=seed)
            nengo.Connection(ens, ens, synapse=f, solver=NoSolver(d), seed=seed)
        nengo.Connection(ens.neurons, dss, synapse=None)
        pTar = nengo.Probe(tar, synapse=None, sample_every=dtSample)
        pEns = nengo.Probe(dss, synapse=None, sample_every=dtSample)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        sim.run(t+dt, progress_bar=True)

    return dict(
        times=sim.trange(),
        tar=sim.data[pTar],
        ens=sim.data[pEns])

def run(N=3000, neuron_type=LIF(), tTrain=200, tTest=100, tTransTrain=20, tTransTest=20,
    nTrain=1, nTest=10, dt=0.001, dtSampleTrain=0.003, dtSampleTest=0.01, seed=0,
    f2=10, reg=1e-3, reg2=1e-3, evals=30, r=30, load=False, file=None,
    tauRiseMin=1e-2, tauRiseMax=3e-2, tauFallMin=1e-1, tauFallMax=3e-1):

    print('\nNeuron Type: %s'%neuron_type)
    rng = np.random.RandomState(seed=seed)
    timeStepsTrain = int((tTrain-tTransTrain)/dtSampleTrain)
    tStart = int(tTransTrain/dtSampleTrain)
    tFlat = int((tTrain-tTransTrain)/dtSampleTrain)*nTrain
    if load:
        d = np.load(file)['d']
        d2 = np.load(file)['d2']
        tauRise = np.load(file)['tauRise']
        tauFall = np.load(file)['tauFall']
        f = DoubleExp(tauRise, tauFall)
    else:
        print('decoders for ens')
        spikes = np.zeros((nTrain, timeStepsTrain, N))
        spikes2 = np.zeros((nTrain, timeStepsTrain, N))
        targets = np.zeros((nTrain, timeStepsTrain, 3))
        targets2 = np.zeros((nTrain, timeStepsTrain, 3))
        for n in range(nTrain):
            IC = np.array([rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(10, 35)])
            data = go(N=N, neuron_type=neuron_type, l=True, t=tTrain, r=r, dt=dt, dtSample=dtSampleTrain, seed=seed, IC=IC)
            spikes[n] = data['ens'][-timeStepsTrain:]
            spikes2[n] = gaussian_filter1d(data['ens'], sigma=f2, axis=0)[-timeStepsTrain:]
            targets[n] = data['tar'][-timeStepsTrain:]
            targets2[n] = gaussian_filter1d(data['tar'], sigma=f2, axis=0)[-timeStepsTrain:]
        d, f, tauRise, tauFall, X, Y, error = decode(
            spikes, targets, nTrain, dt=dtSampleTrain, dtSample=dtSampleTrain, name="lorenzNew", evals=evals, reg=reg,
            tauRiseMin=tauRiseMin, tauRiseMax=tauRiseMax, tauFallMin=tauFallMin, tauFallMax=tauFallMax)
        spikes2 = spikes2.reshape((tFlat, N))
        targets2 = targets2.reshape((tFlat, 3))
        A2 = gaussian_filter1d(spikes2, sigma=f2, axis=0)[-timeStepsTrain:]
        Y2 = gaussian_filter1d(targets2, sigma=f2, axis=0)[-timeStepsTrain:]
        d2, _ = nengo.solvers.LstsqL2(reg=reg2)(spikes2, targets2)
        np.savez("data/lorenzNew_%s.npz"%neuron_type, d=d, d2=d2, tauRise=tauRise, tauFall=tauFall, f2=f2)
        X2 = np.dot(A2, d2)[-timeStepsTrain:]
        error = plotLorenz(X, X2, Y2, neuron_type, "train")

    print("testing")
    tStart = int(tTransTest/dtSampleTest)
    errors = np.zeros((nTest))
    rng = np.random.RandomState(seed=100+seed)
    for test in range(nTest):
        IC = np.array([rng.uniform(-15, 15), rng.uniform(-20, 20), rng.uniform(10, 35)])
        data = go(d=d, f=f, N=N, neuron_type=neuron_type, t=tTest, r=r, dt=dt, dtSample=dtSampleTest, seed=seed, IC=IC)
        A = f.filt(data['ens'], dt=dtSampleTest)
        A2 = gaussian_filter1d(data['ens'], sigma=f2, axis=0)[tStart:]
        X = np.dot(A, d)[tStart:]
        X2 = np.dot(A2, d2)[tStart:]
        Y = data['tar'][tStart:]
        Y2 = gaussian_filter1d(data['tar'], sigma=f2, axis=0)[tStart:]
        error = plotLorenz(X, X2, Y2, neuron_type, test)
        errors[test] = error
    _ = plotLorenz(Y, Y2, Y2, 'target', '')
    print('errors: ', errors)
    np.savez("data/lorenzNew_%s.npz"%neuron_type, d=d, d2=d2, tauRise=tauRise, tauFall=tauFall, f2=f2, errors=errors)
    return errors

errorsLIF = run(neuron_type=LIF(), load=False, file="data/lorenzNew_LIF().npz")
errorsALIF = run(neuron_type=ALIF(), load=False, file="data/lorenzNew_ALIF().npz")
errorsWilson = run(neuron_type=Wilson(), dt=1e-4, load=False, file="data/lorenzNew_Wilson().npz")
