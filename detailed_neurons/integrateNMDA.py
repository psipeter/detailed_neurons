import numpy as np
import nengo
from nengo.dists import Uniform, Gaussian
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import decodeNoF, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron, NMDA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, fPre, fNMDA, dt=0.001, value=1.0, freq=1.0, norm='x', seed=0, c=None):
    if not c: c = t
    stim = nengo.processes.WhiteSignal(period=t/2, high=freq, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
        pX = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    if norm == 'x':
        x = fNMDA.filt(fPre.filt(sim.data[pX], dt=dt), dt=dt)
        norm = value / np.max(np.abs(x))
    elif norm == 'u':
        u = fNMDA.filt(fPre.filt(sim.data[pU], dt=dt), dt=dt)
        norm = value / np.max(np.abs(u))
    mirroredU = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    mirroredX = np.concatenate(([[0]], sim.data[pX]*norm, -sim.data[pX]*norm))
    funcU = lambda t: mirroredU[int(t/dt)] if t<c else 0
    funcX = lambda t: mirroredX[int(t/dt)] if t<c else mirroredX[int(c/dt)]
    return funcU, funcX

def makeSin(t, dt=0.001, seed=0, value=1.0):
    rng = np.random.RandomState(seed=seed)
    # mag = rng.uniform(0.4, 0.8)
    # mag = 0.8
    phase = rng.uniform(0, 2*np.pi)
    # freq = rng.uniform(np.pi/t, 3*np.pi/t)
    mag = value
    freq = 2*np.pi/t
    sin = lambda t: mag*np.sin(freq*t + phase)
    return sin

def makeFlat(t, dt=0.001, seed=0):
    idx = seed % 3
    vals = [-0.8, 0, 0.8]
    flat = lambda t: vals[idx]
    return flat

def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, Tff=0.3, Tneg=-1, tTrans=0.01,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1),
        dPreA=None, dPreB=None, dFdfw=None, dBio=None, ePreA=None, ePreB=None, eFdfw=None, eBio=None, eNeg=None,
        stage=None, alpha=1e-6, eMax=1e-1, stimA=lambda t: np.sin(t), stimB=lambda t: 0, DA=lambda t: 0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preInptA = nengo.Ensemble(NPre, 1, radius=2, max_rates=m, seed=seed)
        preInptB = nengo.Ensemble(NPre, 1, radius=2, max_rates=m, seed=seed)
        preInptC = nengo.Ensemble(NPre, 1, radius=2, max_rates=m, seed=seed)
        fdfw = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        tarFdfw = nengo.Ensemble(N, 1, radius=2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed+1)
        cA = nengo.Connection(inptA, preInptA, synapse=None, seed=seed)
        cB = nengo.Connection(inptB, preInptB, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pPreInptA = nengo.Probe(preInptA.neurons, synapse=None)
        pPreInptB = nengo.Probe(preInptB.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if stage==1:
            nengo.Connection(preInptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(preInptB, tarEns, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c1, tarFdfw, fS, alpha=1e-6, eMax=1e-1, tTrans=tTrans)
            learnEncoders(c2, tarEns, fS, alpha=1e-6, eMax=1e0, tTrans=tTrans)
        if stage==2:
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        if stage==3:
            cB.synapse = fNMDA
            nengo.Connection(preInptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(preInptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(tarFdfw, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            learnEncoders(c3, tarEns, fS, alpha=3e-7, eMax=1e-1, tTrans=tTrans)
        # if stage==3:  # works well for ens-ens2 feedforward, bad for integration
        #     cB.synapse = fNMDA
        #     ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        #     c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        #     c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
        #     c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        #     nengo.Connection(inptA, preInptC, synapse=fNMDA, seed=seed)
        #     c4 = nengo.Connection(preInptC, ens2, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        #     c5 = nengo.Connection(preInptB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
        #     learnEncoders(c3, ens2, fS, alpha=1e-6, eMax=1e0, tTrans=tTrans)
        #     pTarEns = nengo.Probe(ens2.neurons, synapse=None)
        if stage==4:
            cB.synapse = fNMDA
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        if stage==5:
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            nengo.Connection(inptB, preInptC, synapse=fNMDA, seed=seed)
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(preInptB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c4 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c5 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c6 = nengo.Connection(preInptC, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c4, ens3, fS, alpha=3e-7, eMax=1e-1, tTrans=tTrans)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==6:
            fdfw2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            tarFdfw2 = nengo.Ensemble(N, 1, radius=2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(preInptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(preInptA, tarFdfw2, synapse=fPre, seed=seed)
            nengo.Connection(preInptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(tarFdfw, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            nengo.Connection(tarEns, tarFdfw2, synapse=fNMDA, transform=Tneg, seed=seed)
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c4 = nengo.Connection(ens, fdfw2, synapse=NMDA(), solver=NoSolver(-dBio), seed=seed)
            c5 = nengo.Connection(preInptA, fdfw2, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            learnEncoders(c4, tarFdfw2, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            pFdfw2 = nengo.Probe(fdfw2.neurons, synapse=None)
            pTarFdfw2 = nengo.Probe(tarFdfw2.neurons, synapse=None)
        # if stage==6:
        #     preInptC = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        #     fdfw2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        #     fdfw3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        #     nengo.Connection(inptB, preInptC, synapse=fNMDA, transform=-1, seed=seed)
        #     c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        #     c2 = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
        #     c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        #     c4 = nengo.Connection(ens, fdfw2, synapse=NMDA(), solver=NoSolver(-dBio), seed=seed)
        #     c5 = nengo.Connection(preInptA, fdfw2, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        #     c6 = nengo.Connection(preInptA, fdfw3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        #     c7 = nengo.Connection(preInptC, fdfw3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
        #     learnEncoders(c4, fdfw3, fS, alpha=3e-7, eMax=1e-1, tTrans=tTrans)
        #     pFdfw2 = nengo.Probe(fdfw2.neurons, synapse=None)
        #     pTarFdfw2 = nengo.Probe(fdfw3.neurons, synapse=None)
        if stage==7:
            c1 = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c4 = nengo.Connection(ens, fdfw, synapse=NMDA(), solver=NoSolver(-dBio), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==1:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
        if stage==2:
            setWeights(c1, dPreA, ePreA)
        if stage==3:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
            setWeights(c3, dFdfw, eFdfw)
            # setWeights(c4, dPreA, ePreA)
            # setWeights(c5, dPreB, ePreB)
        if stage==4:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
            setWeights(c3, dFdfw, eFdfw)            
        if stage==5:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dFdfw, eFdfw)
            setWeights(c3, dPreB, ePreB)
            setWeights(c4, dBio, eBio)
            setWeights(c5, dFdfw, eFdfw)
            setWeights(c6, dPreB, ePreB)
        if stage==6:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
            setWeights(c3, dFdfw, eFdfw)
            setWeights(c4, -dBio, eNeg)
            setWeights(c5, dPreA, ePreA)
            # setWeights(c6, dPreA, ePreA)
            # setWeights(c7, dPreB, ePreB)
        if stage==7:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dFdfw, eFdfw)
            setWeights(c3, dBio, eBio)
            # setWeights(c4, -dBio, eNeg)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreA = c1.e if stage==1 else ePreA
    ePreB = c2.e if stage==1 else ePreB
    eFdfw = c3.e if stage==3 else eFdfw
    eBio = c4.e if stage==5 else eBio
    eNeg = c4.e if stage==6 else eNeg

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        preInptA=sim.data[pPreInptA],
        preInptB=sim.data[pPreInptB],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        fdfw2=sim.data[pFdfw2] if stage==6 else None,
        tarFdfw2=sim.data[pTarFdfw2] if stage==6 else None,
        ePreA=ePreA,
        ePreB=ePreB,
        eFdfw=eFdfw,
        eBio=eBio,
        eNeg=eNeg,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nEnc=10, dt=0.001, c=None,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1),
        Tff=0.3, Tneg=-1, reg=1e-2, load=[], file=None):

    if not c: c = t
    if 0 in load:
        dPreA = np.load(file)['dPreA']
        dPreB = np.load(file)['dPreB']
    else:
        print('readout decoders for preInptA and preInptB')
        spikesInptA = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptB = np.zeros((nTrain, int(t/dt), NPre))
        targetsInptA = np.zeros((nTrain, int(t/dt), 1))
        targetsInptB = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, stage=0)
            spikesInptA[n] = data['preInptA']
            spikesInptB[n] = data['preInptB']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=dt)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=dt)
        dPreA, X1a, Y1a, error1a = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=dt, reg=reg)
        dPreB, X1b, Y1b, error1b = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1a, Y1a, error1a, "integrateNMDA", "preInptA", t*nTrain)
        plotState(times, X1b, Y1b, error1b, "integrateNMDA", "preInptB", t*nTrain)

    if 1 in load:
        ePreA = np.load(file)['ePreA']
        ePreB = np.load(file)['ePreB']
    else:
        print("encoders for preInptA-to-fdfw and preInptB-to-ens")
        ePreA = np.zeros((NPre, N, 1))
        ePreB = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
                stage=1)
            ePreA = data['ePreA']
            ePreB = data['ePreB']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDA", "preAFdfw")
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preBEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB)

    if 2 in load:
        dFdfw = np.load(file)['dFdfw']
    else:
        print('readout decoders for fdfw')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
                stage=2)
            spikes[n] = data['fdfw']
            targets[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
        dFdfw, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "fdfw", t*nTrain)
        
    if 3 in load:
        eFdfw = np.load(file)['eFdfw']
    else:
        print("encoders for fdfw-to-ens")
        eFdfw = np.zeros((N, N, 1))
        for n in range(nEnc):  # 3-4 if type 2
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            # stimA, _ = makeSignal(t, fPre, fNMDA, dt=dt, seed=n, norm='u', value=3)
            # stimB = makeSin(t, value=0.3)
            # stimB = makeFlat(t, seed=n)
            # stimB = lambda t: 0
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, Tff=Tff,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw,
                stage=3)
            eFdfw = data['eFdfw']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "fdfwToEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, eFdfw=eFdfw)

    if 4 in load:
        dBio = np.load(file)['dBio']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            # stimB = lambda t: 0
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw,
                stage=4)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
            # targets[n] = fNMDA.filt(fPre.filt(fNMDA.filt(data['inptB'])) + fNMDA.filt(Tff*fPre.filt(data['inptA'])))
        dBio, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
            dFdfw=dFdfw, eFdfw=eFdfw, dBio=dBio)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "ens", t*nTrain)

    if 5 in load:
        eBio = np.load(file)['eBio']
    else:
        print("encoders from ens2 to ens")
        eBio = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, dBio=dBio, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, eBio=eBio,
                stage=5)
            eBio = data['eBio']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "Ens2Ens")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
                dFdfw=dFdfw, eFdfw=eFdfw, dBio=dBio, eBio=eBio)
    # eBio = None

    # if 6 in load:
    #     eNeg = np.load(file)['eNeg']
    # else:
    #     print("encoders from ens to fdfw2")
    #     eNeg = np.zeros((N, N, 1))
    #     for n in range(nEnc):
    #         stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
    #         stimB = makeSin(t)
    #         # stimB = makeFlat(t, seed=n)
    #         data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, Tff=Tff, Tneg=Tneg,
    #             dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, dBio=dBio, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, eBio=eBio, eNeg=eNeg,
    #             stage=6, alpha=3e-7)
    #         eNeg = data['eNeg']
    #         plotActivity(t, dt, fS, data['times'], data['fdfw2'], data['tarFdfw2'], "integrateNMDA", "ensToFdfw2")
    #         np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
    #             dFdfw=dFdfw, eFdfw=eFdfw, dBio=dBio, eBio=eBio, eNeg=eNeg)
    eNeg = None

    print("testing")
    errors = np.zeros((nTest))
    rng = np.random.RandomState(seed=0)
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=200+test, c=c)
        data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
            dPreA=dPreA, dFdfw=dFdfw, dBio=dBio, ePreA=ePreA, eFdfw=eFdfw, eBio=eBio, eNeg=eNeg,
            stage=7)
        aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        aBio = fNMDA.filt(data['ens'], dt=dt)
        xhatFdfw = np.dot(aFdfw, dFdfw)
        xhatBio = np.dot(aBio, dBio)
        xFdfw = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt), dt=dt)
        xBio = fNMDA.filt(fPre.filt(data['inptB'], dt=dt), dt=dt)
        errorFdfw = rmse(xhatFdfw, xFdfw)
        errorBio = rmse(xhatBio, xBio)
        errors[test] = errorBio
        fig, ax = plt.subplots()
        ax.plot(data['times'], xFdfw, color='r', label="input")
        ax.plot(data['times'], fNMDA.filt(xFdfw, dt=dt), alpha=0.5, color='r', label="input (filtered)")
        ax.plot(data['times'], xhatFdfw, alpha=0.5, color='r', label="fdfw, rmse=%.3f"%errorFdfw)
        ax.plot(data['times'], xBio, color='b', label="integral")
        ax.plot(data['times'], xhatBio, alpha=0.5, color='b', label="ens, rmse=%.3f"%errorBio)
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_test%s.pdf"%test)
        plt.close('all')
    print('errors:', errors)
    np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
        dFdfw=dFdfw, eFdfw=eFdfw, dBio=dBio, eBio=eBio, eNeg=eNeg, errors=errors)
    return errors

errors = run(N=30, Tff=0.3, Tneg=-1, nTrain=3, nEnc=10, load=[0,1,2,], file="data/integrateNMDA.npz")
