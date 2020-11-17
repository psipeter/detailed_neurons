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
    mirroredU = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    mirroredX = np.concatenate(([[0]], sim.data[pX]*norm, -sim.data[pX]*norm))
    return lambda t: mirroredU[int(t/dt)], lambda t: mirroredX[int(t/dt)] 

def makeSin(t, f, dt=0.001, seed=0):
    rng = np.random.RandomState(seed=seed)
    mag = rng.uniform(0.0, 1.0)
    phase = rng.uniform(0, 2*np.pi)
    freq = rng.uniform(np.pi/t, 3*np.pi/t)
    sin = lambda t: mag*np.sin(freq*t + phase)
    return sin

def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, fPre=None, fNMDA=None, fS=DoubleExp(1e-3, 1e-1), dPreA=None, dPreB=None, dBio=None, ePreA=None, ePreB=None, eBio=None, stage=None, alpha=1e-7, eMax=1e-1, stimA=lambda t: np.sin(t), stimB=lambda t: 0, DA=lambda t: 1.0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preInptA = nengo.Ensemble(NPre, 1, radius=3, max_rates=m, seed=seed)
        preInptB = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        cpa = nengo.Connection(inptA, preInptA, synapse=None, seed=seed)
        cpb = nengo.Connection(inptB, preInptB, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pPreInptA = nengo.Probe(preInptA.neurons, synapse=None)
        pPreInptB = nengo.Probe(preInptB.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if stage==1:
            c0b = nengo.Connection(preInptB, tarEns, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            learnEncoders(c1b, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==2:
            c0a = nengo.Connection(preInptA, tarEns, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c0b = nengo.Connection(preInptB, tarEns, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            learnEncoders(c1a, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==3:
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
        if stage==4:
            preInptC = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            nengo.Connection(inptB, preInptC, synapse=fNMDA, seed=seed)
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2b = nengo.Connection(preInptB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            c3 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c4a = nengo.Connection(preInptA, ens3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c4b = nengo.Connection(preInptC, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c3, ens3, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==5:
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c5 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==1:
            setWeights(c1b, dPreB, ePreB)
        if stage==2:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
        if stage==3:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c1b, dPreB, ePreB)
        if stage==4:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2b, dPreB, ePreB)
            setWeights(c4a, dPreA, ePreA)
            setWeights(c4b, dPreB, ePreB)
            setWeights(c3, dBio, eBio)
        if stage==5:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c5, dBio, eBio)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreB = c1b.e if stage==1 else ePreB
    ePreA = c1a.e if stage==2 else ePreA
    eBio = c3.e if stage==4 else eBio

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        preInptA=sim.data[pPreInptA],
        preInptB=sim.data[pPreInptB],
        ens=sim.data[pEns],
        tarEns=sim.data[pTarEns],
        ePreA=ePreA,
        ePreB=ePreB,
        eBio=eBio,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nEnc=10, dt=0.001, fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1), Tff=0.3, reg=1e-1, tauRiseMax=3e-3, tauFallMax=3e-1, load=[], file=None):

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
            stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=0.001, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB)
            spikesInptA[n] = data['preInptA']
            spikesInptB[n] = data['preInptB']
            targetsInptA[n] = fPre.filt(Tff*data['inptA'], dt=0.001)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=0.001)
        dPreA, X1a, Y1a, error1a = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=0.001, reg=reg)
        dPreB, X1b, Y1b, error1b = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=0.001, reg=reg)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X1a, Y1a, error1a, "integrateNMDA", "preInptA", t*nTrain)
        plotState(times, X1b, Y1b, error1b, "integrateNMDA", "preInptB", t*nTrain)

    if 2 in load:
        ePreB = np.load(file)['ePreB']
    else:
        print("encoders for preInptB-to-ens")
        ePreB = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreB=ePreB, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=1, alpha=1e-6, eMax=1e0)
            ePreB = data['ePreB']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preIntgToEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreB=ePreB)
    if 3 in load:
        ePreA = np.load(file)['ePreA']
    else:
        print("encoders for preInptA-to-ens")
        ePreA = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=n)
            stimB = makeSin(t, fPre)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, fPre=fPre, NPre=NPre, N=N, t=t, dt=dt, fS=fS, stimA=stimA, stimB=stimB, alpha=1e-6, eMax=1e0, stage=2)
            ePreA = data['ePreA']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preInptToEns")
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB)

    if 4 in load:
        dBio = np.load(file)['dBio']
#         tauRiseBio = np.load(file)['tauRiseBio']
#         tauFallBio = np.load(file)['tauFallBio']
#         fNMDA = DoubleExp(tauRiseBio, tauFallBio)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=3)
            spikes[n] = data['ens']
            targets[n] = fPre.filt(data['inptB'], dt=dt)
#         dBio, fNMDA, tauRiseBio, tauFallBio, X2, Y2, error2 = decode(spikes, targets, nTrain, dt=dt, reg=reg, tauRiseMax=tauRiseMax, tauFallMax=tauFallMax, name="integrateNMDA")
        dBio, X2, Y2, error2 = decodeNoF(spikes, targets, nTrain, fNMDA, dt=0.001, reg=reg)
#         np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio, tauRiseBio=tauRiseBio, tauFallBio=tauFallBio)
        np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X2, Y2, error2, "integrateNMDA", "ens", t*nTrain)

    if 5 in load:
        eBio = np.load(file)['eBio']
    else:
        print("encoders from ens2 to ens")
        eBio = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, dBio=dBio, ePreA=ePreA, ePreB=ePreB, eBio=eBio, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fNMDA=fNMDA, fS=fS, stimA=stimA, stimB=stimB, stage=4, alpha=1e-7)
            eBio = data['eBio']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "Ens2Ens")
#             np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio, tauRiseBio=tauRiseBio, tauFallBio=tauFallBio, eBio=eBio)
            np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio, eBio=eBio)

    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, dt=0.001, seed=200+test)
        data = go(dPreA=dPreA, dBio=dBio, ePreA=ePreA, eBio=eBio, NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fNMDA=fNMDA, fS=fS, stimA=stimA, stimB=stimB, stage=5)
        A = fNMDA.filt(data['ens'], dt=dt)
        X = np.dot(A, dBio)
        Y = fPre.filt(data['inptB'], dt=dt)
        U = fPre.filt(Tff*data['inptA'], dt=dt)
        error = rmse(X, Y)
        errorU = rmse(X, U)
        errors[test] = error
        fig, ax = plt.subplots()
        ax.plot(data['times'], U, label="input")
        ax.plot(data['times'], X, label="estimate")
        ax.plot(data['times'], Y, label="target")
        ax.set(xlabel='time', ylabel='state', title='rmse=%.3f'%error, xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_test%s.pdf"%test)
        plt.close('all')
    print('errors:', errors)
#     np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio, tauRiseBio=tauRiseBio, tauFallBio=tauFallBio, eBio=eBio, errors=errors)
    np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dBio=dBio, eBio=eBio, errors=errors)
    return errors

errors = run(N=30, nTrain=10, nEnc=10, load=[], nTest=10, file="data/integrateNMDA.npz")
