import numpy as np
import nengo
from nengo.dists import Uniform
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

def makeSignal(t, fPre, fNMDA, dt=0.001, value=1.0, freq=1.0, seed=0, cutoff=None):
    stim = nengo.processes.WhiteSignal(period=t/2, high=freq, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
        pX = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    u = sim.data[pU]
    x = sim.data[pX]
    x = fNMDA.filt(fPre.filt(x, dt=dt), dt=dt)
    norm = value / np.max(np.abs(x))
    mirroredU = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    mirroredX = np.concatenate(([[0]], sim.data[pX]*norm, -sim.data[pX]*norm))
    if not cutoff:
        cutoff = t
    funcU = lambda t: mirroredU[int(t/dt)] if t<cutoff else 0
    funcX = lambda t: mirroredX[int(t/dt)] if t<cutoff else mirroredX[int(cutoff/dt)]
    return funcU, funcX

def makeSin(t, dt=0.001, seed=0):
    rng = np.random.RandomState(seed=seed)
    mag = rng.uniform(0.25, 1.25)
    phase = rng.uniform(0, 2*np.pi)
    freq = rng.uniform(np.pi/t, 3*np.pi/t)
    sin = lambda t: mag*np.sin(freq*t + phase)
    return sin

def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, fPre=None, fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1), dPreA=None, dPreB=None, dFdfw=None, dTarFdfw=None, dBio=None, ePreA=None, ePreB=None, eFdfw=None, eBio=None, stage=None, alpha=1e-7, eMax=1e-1, stimA=lambda t: np.sin(t), stimB=lambda t: 0, DA=lambda t: 1.0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preInptA = nengo.Ensemble(NPre, 1, radius=2, max_rates=m, seed=seed)
        preInptB = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        fdfw = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        tarFdfw = nengo.Ensemble(N, 1, radius=2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed+1)
        cpa = nengo.Connection(inptA, preInptA, synapse=None, seed=seed)
        cpb = nengo.Connection(inptB, preInptB, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pPreInptA = nengo.Probe(preInptA.neurons, synapse=None)
        pPreInptB = nengo.Probe(preInptB.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if stage==1:
            c0a = nengo.Connection(preInptA, tarFdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c0b = nengo.Connection(preInptB, tarEns, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c1a, tarFdfw, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
            learnEncoders(c1b, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==2:
            c0a = nengo.Connection(preInptA, tarFdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)            
        if stage==3:
            c0a = nengo.Connection(preInptA, tarFdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c0b = nengo.Connection(preInptB, tarEns, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c0c = nengo.Connection(tarFdfw, tarEns, synapse=fNMDA, solver=NoSolver(dTarFdfw), seed=seed)
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c2a = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            learnEncoders(c2a, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==4:
            cpb.synapse = fNMDA
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c2a = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        if stage==5:
            preInptC = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            cpc = nengo.Connection(inptB, preInptC, synapse=fNMDA, seed=seed)
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2a = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3b = nengo.Connection(preInptB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c4b = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c5a = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c5b = nengo.Connection(preInptC, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c4b, ens3, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==6:
            c1a = nengo.Connection(preInptA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2a = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c2b = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==1:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
        if stage==2:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
        if stage==3:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
            setWeights(c2a, dFdfw, eFdfw)
        if stage==4:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
            setWeights(c2a, dFdfw, eFdfw)
        if stage==5:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2a, dFdfw, eFdfw)
            setWeights(c3b, dPreB, ePreB)
            setWeights(c4b, dBio, eBio)
            setWeights(c5a, dFdfw, eFdfw)
            setWeights(c5b, dPreB, ePreB)
        if stage==6:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2a, dFdfw, eFdfw)
            setWeights(c2b, dBio, eBio)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreA = c1a.e if stage==1 else ePreA
    ePreB = c1b.e if stage==1 else ePreB
    eFdfw = c2a.e if stage==3 else eFdfw
    eBio = c4b.e if stage==5 else eBio

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
        ePreA=ePreA,
        ePreB=ePreB,
        eFdfw=eFdfw,
        eBio=eBio,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nEnc=10, dt=0.001, fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1), Tff=0.2, reg=1e-2, load=[], file=None):

    if 1 in load:
        dPreA = np.load(file)['dPreA'] # fix indexing of decoders
        dPreB = np.load(file)['dPreB']
    else:
        print('readout decoders for preInptA and preInptB')
        spikesInptA = np.zeros((nTrain, int(t/0.001), NPre))
        spikesInptB = np.zeros((nTrain, int(t/0.001), NPre))
        targetsInptA = np.zeros((nTrain, int(t/0.001), 1))
        targetsInptB = np.zeros((nTrain, int(t/0.001), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=0.001, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB)
            spikesInptA[n] = data['preInptA']
            spikesInptB[n] = data['preInptB']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=0.001)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=0.001)
        dPreA, X1a, Y1a, error1a = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=0.001, reg=reg)
        dPreB, X1b, Y1b, error1b = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=0.001, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X1a, Y1a, error1a, "integrateNMDA", "preInptA", t*nTrain)
        plotState(times, X1b, Y1b, error1b, "integrateNMDA", "preInptB", t*nTrain)

    if 2 in load:
        ePreA = np.load(file)['ePreA']
        ePreB = np.load(file)['ePreB']
    else:
        print("encoders for preInptA-to-fdfw and preInptB-to-ens")
        ePreA = np.zeros((NPre, N, 1))
        ePreB = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=1, alpha=1e-6, eMax=1e0)
            ePreA = data['ePreA']
            ePreB = data['ePreB']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDA", "preAFdfw")
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preBEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB)

    if 3 in load:
        dFdfw = np.load(file)['dFdfw']
        dTarFdfw = np.load(file)['dTarFdfw']
    else:
        print('readout decoders for fdfw')
        spikes = np.zeros((nTrain, int(t/dt), N))
        spikesTar = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=2)
            spikes[n] = data['fdfw']
            spikesTar[n] = data['tarFdfw']
            targets[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
        dFdfw, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=0.001, reg=reg)
        dTarFdfw, _, _, _ = decodeNoF(spikesTar, targets, nTrain, fNMDA, dt=0.001, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "fdfw", t*nTrain)
        
    if 4 in load:
        eFdfw = np.load(file)['eFdfw']
    else:
        print("encoders for fdfw-to-ens")
        eFdfw = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            stimB = makeSin(t)
            data = go(dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, fPre=fPre, NPre=NPre, N=N, t=t, dt=dt, fS=fS, stimA=stimA, stimB=stimB, alpha=1e-6, eMax=1e0, stage=3)
            eFdfw = data['eFdfw']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "fdfwToEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw, eFdfw=eFdfw)

    if 5 in load:
        dBio = np.load(file)['dBio']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=4)
            spikes[n] = data['ens']
#             targets[n] = fPre.filt(data['inptB'])
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
        dBio, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=0.001, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw, eFdfw=eFdfw, dBio=dBio)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "ens", t*nTrain)

    if 6 in load:
        eBio = np.load(file)['eBio']
    else:
        print("encoders from ens2 to ens")
        eBio = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, dBio=dBio, ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, eBio=eBio, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=5, alpha=3e-7, eMax=1e0)
            eBio = data['eBio']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "Ens2Ens")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw, eFdfw=eFdfw, dBio=dBio, eBio=eBio)

    print("testing")
    errors = np.zeros((nTest))
    rng = np.random.RandomState(seed=0)
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=200+test, cutoff=rng.uniform(0, t))
        data = go(dPreA=dPreA, dFdfw=dFdfw, dBio=dBio, ePreA=ePreA, eFdfw=eFdfw, eBio=eBio, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=6)
        A = fNMDA.filt(data['ens'], dt=dt)
        X = np.dot(A, dBio)
        Y = fNMDA.filt(fPre.filt(data['inptB'], dt=dt))
        U = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt), dt=dt)
        error = rmse(X, Y)
        errorU = rmse(X, U)
        errors[test] = error
        fig, ax = plt.subplots()
        ax.plot(data['times'], U, alpha=0.5, label="input")
        ax.plot(data['times'], X, label="estimate")
        ax.plot(data['times'], Y, label="target")
        ax.set(xlabel='time', ylabel='state', title='rmse=%.3f'%error, xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_test%s.pdf"%test)
        plt.close('all')
    print('errors:', errors)
    np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dFdfw=dFdfw, dTarFdfw=dTarFdfw, eFdfw=eFdfw, dBio=dBio, eBio=eBio, errors=errors)
    return errors

errors = run(N=30, nTrain=10, nEnc=10, load=[1,2,3,4,5,6], nTest=1, file="data/integrateNMDA.npz")
