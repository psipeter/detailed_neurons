import numpy as np
import nengo
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, f, dt=0.001, value=1.0, seed=0):
    stim = nengo.processes.WhiteSignal(period=t/2, high=0.5, rms=0.5, seed=seed)
    stim2 = nengo.processes.WhiteSignal(period=t/2, high=0.5, rms=0.5, seed=100+seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        u2 = nengo.Node(stim2)
        both = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tar = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        nengo.Connection(u, both[0], synapse=None)
        nengo.Connection(u2, both[1], synapse=None)
        nengo.Connection(both, tar, synapse=f)
        nengo.Connection(tar, tar2, synapse=f, function=multiply)
        pBoth = nengo.Probe(both, synapse=None)
        pTar = nengo.Probe(tar, synapse=None)
        pTar2 = nengo.Probe(tar2, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    tar2 = f.filt(sim.data[pTar2], dt=dt)
    norm = value / np.max(np.abs(tar2))
    positive = np.sqrt(norm)*np.vstack((sim.data[pBoth][:,0], sim.data[pBoth][:,1])).T 
    negative = np.sqrt(norm)*np.vstack((sim.data[pBoth][:,0], -sim.data[pBoth][:,1])).T
    mirrored = np.concatenate(([[0, 0]], positive, negative))
    return lambda t: mirrored[int(t/dt)]

def multiply(x):
    return x[0]*x[1]

def go(NPre=100, N=100, N2=30, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), neuron_type=LIF(), d1=None, d2=None, f1=None, f2=None, e1=None, e2=None, l1=False, l2=False, stim=lambda t: [0, 0]):

    if not f1: f1 = f
    if not f2: f2 = f
    if not np.any(d2): d2 = np.zeros((N, 1))
    with nengo.Network(seed=seed) as model:
        # Stimulus and Nodes
        inpt = nengo.Node(stim)
        tar = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(NPre, 2, radius=2, max_rates=m, seed=seed, neuron_type=LIF())
        ens = nengo.Ensemble(N, 2, radius=2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(N2, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed+1)
        nengo.Connection(inpt, pre, synapse=None, seed=seed)
        nengo.Connection(inpt, tar, synapse=f, seed=seed)
        nengo.Connection(tar, tar2, synapse=f, function=multiply, seed=seed+1)
        c1 = nengo.Connection(pre, ens, synapse=f1, seed=seed, solver=NoSolver(d1))
        if isinstance(neuron_type, Bio):
            c2 = nengo.Connection(ens, ens2, synapse=f2, seed=seed+1, function=multiply)
        else:
            c2 = nengo.Connection(ens.neurons, ens2, synapse=f2, seed=seed+1, transform=d2.T)
        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTar = nengo.Probe(tar, synapse=None)
        pTar2 = nengo.Probe(tar2, synapse=None)
        # Encoder Learning (Bio)
        if l1:
            tarEns = nengo.Ensemble(N, 2, radius=2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(tar, tarEns, synapse=None, seed=seed)
            learnEncoders(c1, tarEns, fS, alpha=3e-8)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if l2:
            tarEns2 = nengo.Ensemble(N2, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed+1)
            nengo.Connection(tar2, tarEns2, synapse=None)
#             nengo.Connection(ens.neurons, tarEns2, synapse=f2, transform=d2.T, seed=seed+1)
            learnEncoders(c2, tarEns2, fS, alpha=1e-7)
            pTarEns2 = nengo.Probe(tarEns2.neurons, synapse=None)
            pTarState = nengo.Probe(tarEns2, synapse=f)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Bio):
            setWeights(c1, d1, e1)
            setWeights(c2, d2, e2)
            neuron.h.init()
            sim.run(t, progress_bar=True)
            reset_neuron(sim, model) 
        else:
            sim.run(t, progress_bar=True)
      
    e1 = c1.e if l1 else e1
    e2 = c2.e if l2 else e2

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        ens2=sim.data[pEns2],
        tar=sim.data[pTar],
        tar2=sim.data[pTar2],
        tarEns=sim.data[pTarEns] if l1 else None,
        tarEns2=sim.data[pTarEns2] if l2 else None,
        tarState=sim.data[pTarState] if l2 else None,
        e1=e1,
        e2=e2,
    )


def run(NPre=200, N=100, N2=100, t=10, nTrain=10, nEnc=20, nTest=10, neuron_type=LIF(),
        dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), tauRiseMax=1e-2, load=False, file=None):

    print('\nNeuron Type: %s'%neuron_type)
    if load:
        d1 = np.load(file)['d1']
        tauRise1 = np.load(file)['tauRise1']
        tauFall1 = np.load(file)['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('readout decoders for pre')
        spikes = np.zeros((nTrain, int(t/0.001), NPre))
        targets = np.zeros((nTrain, int(t/0.001), 2))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, N2=N2, t=t, dt=0.001, f=f, fS=fS, neuron_type=LIF(), stim=stim)
            spikes[n] = data['pre']
            targets[n] = f.filt(data['inpt'], dt=0.001)
        d1, f1, tauRise1, tauFall1, X, Y, error = decode(spikes, targets, nTrain, dt=0.001, tauRiseMax=tauRiseMax, name="multiplyNew")
        np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X, Y, error, "multiplyNew", "%s_pre"%neuron_type, t*nTrain)

    if load:
        e1 = np.load(file)['e1']
    elif isinstance(neuron_type, Bio):
        print("ens1 encoders")
        e1 = np.zeros((NPre, N, 2))
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(d1=d1, e1=e1, f1=f1, NPre=NPre, N=N, N2=N2, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l1=True)
            e1 = data['e1']
            np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1)
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "multiplyNew", "ens")
    else:
        e1 = np.zeros((NPre, N, 2))

    if load:
        d2 = np.load(file)['d2']
        tauRise2 = np.load(file)['tauRise2']
        tauFall2 = np.load(file)['tauFall2']
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(d1=d1, e1=e1, f1=f1, NPre=NPre, N=N, N2=N2, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
            spikes[n] = data['ens']
            targets[n] = f.filt(data['tar'][:,0]*data['tar'][:,1], dt=dt).reshape(-1, 1)
        d2, f2, tauRise2, tauFall2, X, Y, error = decode(spikes, targets, nTrain, dt=dt, tauRiseMax=tauRiseMax, name="multiplyNew")
        np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "multiplyNew", "%s_ens"%neuron_type, t*nTrain)

    if load:
        e2 = np.load(file)['e2']
    elif isinstance(neuron_type, Bio):
        print("ens2 encoders")
        e2 = np.zeros((N, N2, 1))
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, NPre=NPre, N=N, N2=N2, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l2=True)
            e2 = data['e2']
            np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2)
            plotActivity(t, dt, fS, data['times'], data['ens2'], data['tarEns2'], "multiplyNew", "ens2")
    else:
        e2 = np.zeros((N, N2, 1))
        
    if load:
        d3 = np.load(file)['d3']
        tauRise3 = np.load(file)['tauRise3']
        tauFall3 = np.load(file)['tauFall3']
        f3 = DoubleExp(tauRise3, tauFall3)
    else:
        print('readout decoders for ens2')
        spikes = np.zeros((nTrain, int(t/dt), N2))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, NPre=NPre, N=N, N2=N2, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
            spikes[n] = data['ens2']
            targets[n] = f.filt(data['tar2'], dt=dt)
        d3, f3, tauRise3, tauFall3, X, Y, error = decode(spikes, targets, nTrain, dt=dt, name="multiplyNew")
        np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "multiplyNew", "%s_ens2"%neuron_type, t*nTrain)
        
    errors = np.zeros((nTest))
    print("testing")
    for test in range(nTest):
        stim = makeSignal(t, f, dt=dt, seed=100+test)
        data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, NPre=NPre, N=N, N2=N2, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
        A = f3.filt(data['ens2'], dt=dt)
        X = np.dot(A, d3)
        Y = f.filt(data['tar2'], dt=dt)
        error = rmse(X, Y)
        errors[test] = error
        plotState(data['times'], X, Y, error, "multiplyNew", "%s_test%s"%(neuron_type, test), t)
        A = f2.filt(data['ens'], dt=dt)
        X = np.dot(A, d2)
        Y = f.filt(data['tar'][:,0]*data['tar'][:,1], dt=dt).reshape(-1, 1)
        plotState(data['times'], X, Y, rmse(X, Y), "multiplyNew", "%s_pretest%s"%(neuron_type, test), t)

    print('%s errors:'%neuron_type, errors)
    np.savez("data/multiplyNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3, errors=errors)
    return errors

# errorsLIF = run(neuron_type=LIF(), load=False, file="data/multiplyNew_LIF().npz")
# errorsALIF = run(neuron_type=ALIF(), load=False, file="data/multiplyNew_ALIF().npz")
# errorsWilson = run(neuron_type=Wilson(), dt=1e-4, load=False, file="data/multiplyNew_Wilson().npz")
errorsBio = run(neuron_type=Bio("Pyramidal"), load=False, file="data/multiplyNew_Bio().npz")
