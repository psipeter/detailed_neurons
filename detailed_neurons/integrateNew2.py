import numpy as np
import nengo
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, f, dt=0.001, value=1.0, nFilts=1, norm='x', freq=1.0, seed=0):
    stim = nengo.processes.WhiteSignal(period=t/2, high=freq, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
        pX = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    u = sim.data[pU]
    x = sim.data[pX]
    if norm == 'x':
        for n in range(nFilts):
            x = f.filt(x, dt=dt)
        norm = value / np.max(np.abs(x))
    elif norm == 'u':
        for n in range(nFilts):
            u = f.filt(u, dt=dt)
        norm = value / np.max(np.abs(u))
    mirrored = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    return lambda t: mirrored[int(t/dt)]

def makeSin(t, f, dt=0.001, seed=0):
    rng = np.random.RandomState(seed=seed)
    mag = rng.uniform(0.2, 0.8)
    phase = rng.uniform(0, 2*np.pi)
    freq = rng.uniform(0.5*np.pi/t, 2*np.pi/t)
    sin = lambda t: mag*np.sin(freq*t + phase)
    return sin

def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=DoubleExp(1e-3, 1e-2), fS=DoubleExp(1e-3, 1e-1), neuron_type=LIF(), d1a=None, d1b=None, d2=None, f1a=None, f1b=None, f2=None, e1a=None, e1b=None, e2=None, l1a=False, l1b=False, l2=False, l3=False, test=False, stim=lambda t: np.sin(t), stim2=lambda t: 0):

    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        preInpt = nengo.Ensemble(NPre, 1, radius=3, max_rates=m, seed=seed)
        preIntg = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s, seed=seed)
        c0a = nengo.Connection(inpt, preInpt, synapse=None, seed=seed)
        c0b = nengo.Connection(intg, preIntg, synapse=None, seed=seed)
        c1a = nengo.Connection(preInpt, ens, synapse=f1a, solver=NoSolver(d1a), seed=seed)
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pPreInpt = nengo.Probe(preInpt.neurons, synapse=None)
        pPreIntg = nengo.Probe(preIntg.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        if l1b:  # preIntg-to-ens
            tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(preIntg, tarEns, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
            c1b = nengo.Connection(preIntg, ens, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
            learnEncoders(c1b, tarEns, fS)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if l1a:  # preInpt-to-ens, given preIntg-to-ens
            inpt2 = nengo.Node(stim2)
            tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(preInpt, tarEns, synapse=f1a, solver=NoSolver(d1a), seed=seed)
            c0b.transform = 0
            nengo.Connection(inpt2, preIntg, synapse=None, seed=seed)
            nengo.Connection(preIntg, tarEns, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
            c1b = nengo.Connection(preIntg, ens, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
            learnEncoders(c1a, tarEns, fS)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if l2: # ens readout, given preIntg and preInpt
            c1b = nengo.Connection(preIntg, ens, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
        if l3:  # ens2-to-ens, given preInpt-ens and preIntg-ens2
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            c0a.synapse = f
            c2a = nengo.Connection(preInpt, ens2, synapse=f1a, solver=NoSolver(d1a), seed=seed)
            c2b = nengo.Connection(preIntg, ens2, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
            c3 = nengo.Connection(ens2, ens, synapse=f2, solver=NoSolver(d2), seed=seed)
            learnEncoders(c3, ens2, fS)
            pTarEns2 = nengo.Probe(ens2.neurons, synapse=None)
        if test:
            c5 = nengo.Connection(ens, ens, synapse=f2, solver=NoSolver(d2), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Bio):
            if l1b:
                setWeights(c1b, d1b, e1b)
            if l1a:
                setWeights(c1a, d1a, e1a)
                setWeights(c1b, d1b, e1b)
            if l2:
                setWeights(c1a, d1a, e1a)
                setWeights(c1b, d1b, e1b)
            if l3:
                setWeights(c1a, d1a, e1a)
                setWeights(c2a, d1a, e1a)
                setWeights(c2b, d1b, e1b)
                setWeights(c3, d2, e2)
            if test:
                setWeights(c1a, d1a, e1a)
                setWeights(c5, d2, e2)
            neuron.h.init()
            sim.run(t, progress_bar=True)
            reset_neuron(sim, model) 
        else:
            sim.run(t, progress_bar=True)

    e1a = c1a.e if l1a else e1a
    e1b = c1b.e if l1b else e1b
    e2 = c3.e if l3 else e2

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        preInpt=sim.data[pPreInpt],
        preIntg=sim.data[pPreIntg],
        ens=sim.data[pEns],
        tarEns=sim.data[pTarEns] if l1a or l1b else None,
        tarEns2=sim.data[pTarEns2] if l3 else None,
        e1a=e1a,
        e1b=e1b,
        e2=e2,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=30, nEnc=10, neuron_type=LIF(),
        dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), Tff=0.1, Tfb=1.0, reg=1e-3, tauRiseMax=5e-2, tauFallMax=3e-1, load=False, file=None):

    print('\nNeuron Type: %s'%neuron_type)
    if load:
        d1a = np.load(file)['d1a']
        d1b = np.load(file)['d1b']
        tauRise1a = np.load(file)['tauRise1a']
        tauRise1b = np.load(file)['tauRise1b']
        tauFall1a = np.load(file)['tauFall1a']
        tauFall1b = np.load(file)['tauFall1b']
        f1a = DoubleExp(tauRise1a, tauFall1a)
        f1b = DoubleExp(tauRise1b, tauFall1b)
    else:
        print('readout decoders for preInpt and preIntg')
        spikesInpt = np.zeros((nTrain, int(t/0.001), NPre))
        spikesIntg = np.zeros((nTrain, int(t/0.001), NPre))
        targetsInpt = np.zeros((nTrain, int(t/0.001), 1))
        targetsIntg = np.zeros((nTrain, int(t/0.001), 1))
        for n in range(nTrain):
            stim = makeSignal(t, f, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=0.001, f=f, fS=fS, neuron_type=LIF(), stim=stim)
            spikesInpt[n] = data['preInpt']
            spikesIntg[n] = data['preIntg']
            targetsInpt[n] = f.filt(Tff*data['inpt'], dt=0.001)
            targetsIntg[n] = f.filt(data['intg'], dt=0.001)
        d1a, f1a, tauRise1a, tauFall1a, X1a, Y1a, error1a = decode(spikesInpt, targetsInpt, nTrain, dt=0.001, reg=reg, name="integrateNew", tauRiseMax=tauRiseMax, tauFallMax=tauFallMax)
        d1b, f1b, tauRise1b, tauFall1b, X1b, Y1b, error1b = decode(spikesIntg, targetsIntg, nTrain, dt=0.001, reg=reg, name="integrateNew", tauRiseMax=tauRiseMax, tauFallMax=tauFallMax)
        np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X1a, Y1a, error1a, "integrateNew", "%s_preInpt"%neuron_type, t*nTrain)
        plotState(times, X1b, Y1b, error1b, "integrateNew", "%s_preIntg"%neuron_type, t*nTrain)

    if load:
        e1a = np.load(file)['e1a']
        e1b = np.load(file)['e1b']
    elif isinstance(neuron_type, Bio):
        e1a = np.zeros((NPre, N, 1))
        e1b = np.zeros((NPre, N, 1))
        print("encoders for preIntg-to-ens")
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            data = go(d1a=d1a, d1b=d1b, e1a=e1a, e1b=e1b, f1a=f1a, f1b=f1b, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l1b=True)
            e1b = data['e1b']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNew", "preIntgToEns")
            np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b, e1a=e1a, e1b=e1b)
        print("encoders for preInpt-to-ens")
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            # stim2 = makeSignal(t, f, dt=dt, seed=n, value=0.5)
#             stim2 = makeSignal(t, f, dt=dt, norm='u', freq=0.25, value=0.8, seed=100+n)
            stim2 = makeSin(t, f, dt=dt, seed=n)
            data = go(d1a=d1a, d1b=d1b, e1a=e1a, e1b=e1b, f1a=f1a, f1b=f1b, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l1a=True, stim2=stim2)
            e1a = data['e1a']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNew", "preInptToEns")
            np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b, e1a=e1a, e1b=e1b)
    else:
        e1a = np.zeros((NPre, N, 1))
        e1b = np.zeros((NPre, N, 1))

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
            data = go(d1a=d1a, d1b=d1b, e1a=e1a, e1b=e1b, f1a=f1a, f1b=f1b, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l2=True)
            spikes[n] = data['ens']
            targets[n] = f.filt(Tfb*data['intg'], dt=dt)
        d2, f2, tauRise2, tauFall2, X, Y, error = decode(spikes, targets, nTrain, dt=dt, reg=reg, tauRiseMax=tauRiseMax, tauFallMax=tauFallMax, name="integrateNew")
        np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b, e1a=e1a, e1b=e1b, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNew", "%s_ens"%neuron_type, t*nTrain)

    if load:
        e2 = np.load(file)['e2']
    elif isinstance(neuron_type, Bio):
        print("encoders from ens2 to ens")
        e2 = np.zeros((N, N, 1))
        for n in range(nEnc):
            stim = makeSignal(t, f, dt=dt, seed=n)
            #stim = makeSin(t, f, dt=dt, seed=n)
            data = go(d1a=d1a, d1b=d1b, d2=d2, e1a=e1a, e1b=e1b, e2=e2, f1a=f1a, f1b=f1b, f2=f2, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, l3=True)
            e2 = data['e2']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns2'], "integrateNew", "Ens2Ens")
            np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b, e1a=e1a, e1b=e1b, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2)
    else:
        e2 = np.zeros((N, N, 1))

    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        stim = makeSignal(t, f, dt=dt, seed=200+test)
        data = go(d1a=d1a, d2=d2, e1a=e1a, e2=e2, f1a=f1a, f2=f2, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, stim=stim, test=True)
        A = f2.filt(data['ens'], dt=dt)
        X = np.dot(A, d2)
        Y = f.filt(data['intg'], dt=dt)
        U = f.filt(f.filt(Tff*data['inpt'], dt=dt))
        error = rmse(X, Y)
        errorU = rmse(X, U)
        errors[test] = error
        plotState(data['times'], X, Y, error, "integrateNew", "%s_test%s"%(neuron_type, test), t)
        #plotState(data['times'], X, U, errorU, "integrateNew", "%s_inpt%s"%(neuron_type, test), t)
    print('%s errors:'%neuron_type, errors)
    np.savez("data/integrateNew_%s.npz"%neuron_type, d1a=d1a, d1b=d1b, tauRise1a=tauRise1a, tauRise1b=tauRise1b, tauFall1a=tauFall1a, tauFall1b=tauFall1b, e1a=e1a, e1b=e1b, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, errors=errors)
    return errors

#errorsLIF = run(neuron_type=LIF(), load=False, file="data/integrateNew_LIF().npz")
#errorsALIF = run(neuron_type=ALIF(), load=False, file="data/integrateNew_ALIF().npz")
#errorsWilson = run(neuron_type=Wilson(), dt=1e-4, load=False, file="data/integrateNew_Wilson().npz")
errorsBio = run(neuron_type=Bio("Pyramidal"), N=30, Tff=0.3, reg=1e-1, nTrain=10, nEnc=10, nTest=10, fS=DoubleExp(1e-3, 1e-1), load=False, file="data/integrateNew_Bio().npz")
