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

def makeSignal(t, f, dt=0.001, value=1.0, nFilts=3, seed=0):
    stim = nengo.processes.WhiteSignal(period=t/2, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    u = sim.data[pU]
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    norm = value / np.max(np.abs(u))
    mirrored = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    return lambda t: mirrored[int(t/dt)]

def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), neuron_type=LIF(), d1=None, d2=None, f1=None, f2=None, e1=None, e2=None, l1=False, l2=False, stim=lambda t: np.sin(t)):

    if not f1: f1=f
    if not f2: f2=f
    with nengo.Network(seed=seed) as model:
        # Stimulus and Nodes
        inpt = nengo.Node(stim)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        tar2 = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(NPre, 1, radius=2, max_rates=m, seed=seed, neuron_type=LIF())
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed+1)
        nengo.Connection(inpt, pre, synapse=None, seed=seed)
        nengo.Connection(inpt, tar, synapse=f, seed=seed)
        nengo.Connection(tar, tar2, synapse=f, seed=seed+1)
        c1 = nengo.Connection(pre, ens, synapse=f1, seed=seed, solver=NoSolver(d1))
        c2 = nengo.Connection(ens, ens2, synapse=f2, seed=seed+1, solver=NoSolver(d2))
        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        pTar = nengo.Probe(tar, synapse=None)
        pTar2 = nengo.Probe(tar2, synapse=None)
        # Encoder Learning (Bio)
        if l1:
            tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
            nengo.Connection(tar, tarEns, synapse=None, seed=seed)
            learnEncoders(c1, tarEns, fS)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if l2:
            tarEns2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed+1)
            nengo.Connection(tar2, tarEns2, synapse=None, seed=seed)
            learnEncoders(c2, tarEns2, fS)
            pTarEns2 = nengo.Probe(tarEns2.neurons, synapse=None)

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
        e1=e1,
        e2=e2,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nEnc=10, nTest=10, neuron_type=LIF(),
        dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), tauRiseMax=1e-2, load=False, file=None):

    print('\nNeuron Type: %s'%neuron_type)
    if load:
        d1 = np.load(file)['d1']
        tauRise1 = np.load(file)['tauRise1']
        tauFall1 = np.load(file)['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        spikes = np.zeros((nTrain, int(t/0.001), NPre))
        targets = np.zeros((nTrain, int(t/0.001), 1))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=0.001, f=f, fS=fS, neuron_type=LIF(), stim=stim)
            spikes[n] = data['pre']
            targets[n] = f.filt(data['inpt'], dt=0.001)
        d1, f1, tauRise1, tauFall1, X, Y, error = decode(spikes, targets, nTrain, dt=0.001, name="identityNew", tauRiseMax=tauRiseMax)
        np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X, Y, error, "identityNew", "%s_pre"%neuron_type, t*nTrain)

    if load:
        e1 = np.load(file)['e1']
    elif isinstance(neuron_type, Bio):
        print("ens1 encoders")
        e1 = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(NPre=NPre, d1=d1, e1=e1, f1=f1, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l1=True)
            e1 = data['e1']
            np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1)
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "identityNew", "ens")
    else:
        e1 = np.zeros((NPre, N, 1))

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
            stim = makeSignal(t, f, dt=0.001, seed=n)
            data = go(NPre=NPre, d1=d1, e1=e1, f1=f1, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
            spikes[n] = data['ens']
            targets[n] = f.filt(data['tar'], dt=dt)
        d2, f2, tauRise2, tauFall2, X, Y, error = decode(spikes, targets, nTrain, dt=dt, name="identityNew", tauRiseMax=tauRiseMax)
        np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "identityNew", "%s_ens"%neuron_type, t*nTrain)


    if load:
        e2 = np.load(file)['e2']
    elif isinstance(neuron_type, Bio):
        print("ens2 encoders")
        e2 = np.zeros((N, N, 1))
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(NPre=NPre, d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l2=True)
            e2 = data['e2']
            np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2)
            plotActivity(t, dt, fS, data['times'], data['ens2'], data['tarEns2'], "identityNew", "ens2")
    else:
        e2 = np.zeros((N, N, 1))
        
    if load:
        d3 = np.load(file)['d3']
        tauRise3 = np.load(file)['tauRise3']
        tauFall3 = np.load(file)['tauFall3']
        f3 = DoubleExp(tauRise3, tauFall3)
    else:
        print('readout decoders for ens2')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=0.001, seed=n)
            data = go(NPre=NPre, d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
            spikes[n] = data['ens2']
            targets[n] = f.filt(data['tar2'], dt=dt)
        d3, f3, tauRise3, tauFall3, X, Y, error = decode(spikes, targets, nTrain, dt=dt, name="identityNew")
        np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "identityNew", "%s_ens2"%neuron_type, t*nTrain)

    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        stim = makeSignal(t, f, dt=dt, seed=100+test)
        data = go(NPre=NPre, d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim)
        A = f3.filt(data['ens2'], dt=dt)
        X = np.dot(A, d3)
        Y = f.filt(data['tar2'], dt=dt)
        error = rmse(X, Y)
        errors[test] = error
        plotState(data['times'], X, Y, error, "identityNew", "%s_test%s"%(neuron_type, test), t)
    
    print('%s errors:'%neuron_type, errors)
    np.savez("data/identityNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3, errors=errors)
    return errors

errorsLIF = run(neuron_type=LIF(), load=False, file="data/identityNew_LIF().npz")
errorsALIF = run(neuron_type=ALIF(), load=False, file="data/identityNew_ALIF().npz")
errorsWilson = run(neuron_type=Wilson(), dt=1e-4, load=False, file="data/identityNew_Wilson().npz")
errorsBio = run(neuron_type=Bio("Pyramidal"), load=False, file="data/identityNew_Bio().npz")