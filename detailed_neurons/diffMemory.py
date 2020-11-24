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
    u = fNMDA.filt(fPre.filt(u, dt=dt), dt=dt)
    norm = value / np.max(np.abs(u))
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



def ideal(t=10, c=7):
    stim = nengo.processes.WhiteSignal(period=t, high=1, rms=0.3, seed=0)
    cutoff = lambda t: 0 if t<c else 1
    with nengo.Network() as network:
        inpt = nengo.Node(stim)
        cut = nengo.Node(cutoff)
        diff = nengo.Ensemble(100, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=0)
        ens = nengo.Ensemble(100, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=0)
        nengo.Connection(inpt, diff, synapse=None)
        nengo.Connection(cut, diff.neurons, synapse=None, transform=-1e2*np.ones((100, 1)))
        nengo.Connection(diff, ens, synapse=0.1)
        nengo.Connection(ens, ens, synapse=0.1)
        nengo.Connection(ens, diff, synapse=0.1, transform=-1)
        pInpt = nengo.Probe(inpt, synapse=0.1)
        pDiff = nengo.Probe(diff, synapse=0.1)
        pEns = nengo.Probe(ens, synapse=0.1)
    with nengo.Simulator(network, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)
    fig, ax = plt.subplots()
    ax.plot(sim.trange(), sim.data[pInpt], label='inpt')
    ax.plot(sim.trange(), sim.data[pDiff], label='diff')
    ax.plot(sim.trange(), sim.data[pEns], label='ens')
    ax.legend()
    fig.savefig("plots/diffMemory_exp.pdf")


def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001,
    fPre=None, fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1),
    dPreA=None, dDiff=None, dBio=None, ePreA=None, eNeg=None, eDiff=None, eBio=None,
    stage=None, alpha=1e-7, eMax=1e-1, stimA=lambda t: np.sin(t), stimB=lambda t: 0, DA=lambda t: 1.0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preInptA = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        diff = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        tarDiff = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        cpa = nengo.Connection(inptA, preInptA, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pPreInptA = nengo.Probe(preInptA.neurons, synapse=None)
        pDiff = nengo.Probe(diff.neurons, synapse=None)
        pTarDiff = nengo.Probe(tarDiff.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if stage==1:
            c0a = nengo.Connection(preInptA, tarDiff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1a = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            learnEncoders(c1a, tarDiff, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==2:
            c1a = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
        if stage==3:
            preInptB = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            cpb = nengo.Connection(inptA, preInptB, synapse=fNMDA, seed=seed)
            c0b = nengo.Connection(preInptB, tarEns, synapse=fNMDA, seed=seed)
            c1a = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2a = nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(dDiff), seed=seed)
            learnEncoders(c2a, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
        if stage==4:
            c1a = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2a = nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(dDiff), seed=seed)
        if stage==5:
            preInptB = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            preInptC = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            diff2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            diff3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            c0 = nengo.Connection(inptA, preInptB, synapse=fNMDA, seed=seed)
            c1 = nengo.Connection(inptA, preInptC, synapse=fNMDA, seed=seed, transform=-1)
            c2 = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c3 = nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(dDiff), seed=seed)
            c4 = nengo.Connection(preInptB, ens3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c5 = nengo.Connection(ens, ens2, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c6 = nengo.Connection(preInptA, diff2, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c7 = nengo.Connection(ens, diff2, synapse=NMDA(), solver=NoSolver(-dBio), seed=seed)
            c8 = nengo.Connection(preInptA, diff3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c9 = nengo.Connection(preInptC, diff3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            learnEncoders(c5, ens3, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
            learnEncoders(c7, diff3, fS, alpha=alpha, eMax=eMax, tTrans=1.0)
            pDiff2 = nengo.Probe(diff2.neurons, synapse=None)
            pDiff3 = nengo.Probe(diff3.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
            pEns3 = nengo.Probe(ens3.neurons, synapse=None)
        if stage==6:
            c1a = nengo.Connection(preInptA, diff, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2a = nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(dDiff), seed=seed)
            c2b = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c3b = nengo.Connection(ens, diff, synapse=NMDA(), solver=NoSolver(-dBio), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==1:
            setWeights(c1a, dPreA, ePreA)
        if stage==2:
            setWeights(c1a, dPreA, ePreA)
        if stage==3:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2a, dDiff, eDiff)
        if stage==4:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2a, dDiff, eDiff)
        if stage==5:
            setWeights(c2, dPreA, ePreA)
            setWeights(c3, dDiff, eDiff)
            setWeights(c4, dPreA, ePreA)
            setWeights(c5, dBio, eBio)
            setWeights(c6, dPreA, ePreA)
            setWeights(c7, -dBio, eNeg)
            setWeights(c8, dPreA, ePreA)
            setWeights(c9, dPreA, ePreA)
        if stage==6:
            setWeights(c1a, dPreA, ePreA)
            setWeights(c2a, dDiff, eDiff)
            # setWeights(c2b, dBio, eBio)
            # setWeights(c3b, -dBio, eNeg)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreA = c1a.e if stage==1 else ePreA
    eDiff = c2a.e if stage==3 else eDiff
    eBio = c5.e if stage==5 else eBio
    eNeg = c7.e if stage==5 else eNeg

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        preInptA=sim.data[pPreInptA],
        diff=sim.data[pDiff],
        ens=sim.data[pEns],
        tarDiff=sim.data[pTarDiff],
        tarEns=sim.data[pTarEns],
        diff2=sim.data[pDiff2] if stage==5 else None,
        diff3=sim.data[pDiff3] if stage==5 else None,
        ens2=sim.data[pEns2] if stage==5 else None,
        ens3=sim.data[pEns3] if stage==5 else None,
        ePreA=ePreA,
        eDiff=eDiff,
        eBio=eBio,
        eNeg=eNeg,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nEnc=10, dt=0.001,
    fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1),
    reg=1e-2, load=[], file=None):

    if 0 in load:
        dPreA = np.load(file)['dPreA']
    else:
        print('readout decoders for preInptA')
        spikesInptA = np.zeros((nTrain, int(t/0.001), NPre))
        targetsInptA = np.zeros((nTrain, int(t/0.001), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(NPre=NPre, N=N, t=t, dt=0.001, fPre=fPre, fS=fS, stimA=stimA)
            spikesInptA[n] = data['preInptA']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=0.001)
        dPreA, X1a, Y1a, error1a = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=0.001, reg=reg)
        np.savez("data/diffMemory.npz", dPreA=dPreA)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X1a, Y1a, error1a, "diffMemory", "preInptA", t*nTrain)

    if 1 in load:
        ePreA = np.load(file)['ePreA']
    else:
        print("encoders for preInptA-to-diff")
        ePreA = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, ePreA=ePreA,
                NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stage=1, alpha=1e-6, eMax=1e0)
            ePreA = data['ePreA']
            plotActivity(t, dt, fS, data['times'], data['diff'], data['tarDiff'], "diffMemory", "preADiff")
            np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA)

    if 2 in load:
        dDiff = np.load(file)['dDiff']
    else:
        print('readout decoders for diff')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, ePreA=ePreA,
                NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stage=2)
            spikes[n] = data['diff']
            targets[n] = fNMDA.filt(fPre.filt(data['inptA'], dt=dt))
        dDiff, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=0.001, reg=reg)
        np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA, dDiff=dDiff)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "diffMemory", "diff", t*nTrain)
        
    if 3 in load:
        eDiff = np.load(file)['eDiff']
    else:
        print("encoders for diff-to-ens")
        eDiff = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            stimB = makeSin(t)
            data = go(dPreA=dPreA, dDiff=dDiff, ePreA=ePreA, eDiff=eDiff,
                fPre=fPre, NPre=NPre, N=N, t=t, dt=dt, fS=fS, stimA=stimA, alpha=1e-6, eMax=1e0, stage=3)
            eDiff = data['eDiff']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "diffMemory", "diffToEns")
            np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA, dDiff=dDiff, eDiff=eDiff)

    if 4 in load:
        dBio = np.load(file)['dBio']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dDiff=dDiff, ePreA=ePreA, eDiff=eDiff,
                NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stage=4)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fNMDA.filt(fPre.filt(data['inptA'])))
        dBio, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=0.001, reg=reg)
        np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA, dDiff=dDiff, dBio=dBio)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "diffMemory", "ens", t*nTrain)

    eBio = np.array(eDiff)

    if 5 in load:
        eBio = np.load(file)['eBio']
        eNeg = np.load(file)['eNeg']
    else:
        print("encoders from ens-to-ens and ens-to-diff")
        eBio = np.zeros((N, N, 1))
        eNeg = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=n)
            data = go(dPreA=dPreA, dDiff=dDiff, dBio=dBio, ePreA=ePreA, eDiff=eDiff, eBio=eBio, eNeg=eNeg,
                NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stage=5, alpha=3e-7, eMax=1e0)
            eBio = data['eBio']
            eNeg = data['eNeg']
            plotActivity(t, dt, fS, data['times'], data['ens2'], data['ens3'], "diffMemory", "EnsEns2")
            plotActivity(t, dt, fS, data['times'], data['diff2'], data['diff3'], "diffMemory", "EnsDiff2")
            np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA, dDiff=dDiff, eDiff=eDiff, dBio=dBio, eBio=eBio, eNeg=eNeg)

    print("testing")
    errors = np.zeros((nTest))
    rng = np.random.RandomState(seed=0)
    for test in range(nTest):
        # stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=200+test, cutoff=rng.uniform(0, t))
        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=0.001, seed=200+test)
        data = go(dPreA=dPreA, dDiff=dDiff, dBio=dBio, ePreA=ePreA, eDiff=eDiff, eBio=eBio, eNeg=eNeg,
            NPre=NPre, N=N, t=t, dt=dt, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB, stage=6)
        A = fNMDA.filt(fPre.filt(data['diff']))
        A2 = fNMDA.filt(fNMDA.filt(fPre.filt(data['ens'])))
        X = np.dot(A, dDiff)
        X2 = np.dot(A2, dBio)
        U = fNMDA.filt(fPre.filt(data['inptA']))
        Y = fNMDA.filt(fPre.filt(data['inptB']))
        error = rmse(X2, Y)
        # errorU = rmse(X, U)
        errors[test] = error
        fig, ax = plt.subplots()
        ax.plot(data['times'], U, alpha=0.5, label="input")
        ax.plot(data['times'], Y, label="integral")
        ax.plot(data['times'], X, label="diff")
        ax.plot(data['times'], X2, label="ens")
        ax.set(xlabel='time', ylabel='state', title="rmse=%.3f"%error, xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/diffMemory_test%s.pdf"%test)
        plt.close('all')
    print('errors:', errors)
    np.savez("data/diffMemory.npz", dPreA=dPreA, ePreA=ePreA, dDiff=dDiff, eDiff=eDiff, dBio=dBio, eBio=eBio, errors=errors)
    return errors

# errors = run(N=30, nTrain=5, nEnc=5, load=[0,1,2,3,4,5], nTest=10, file="data/diffMemory.npz")

# ideal()

def test(NPre=100, N=30, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, file="data/diffMemory.npz"):
    stim = nengo.processes.WhiteSignal(period=t, high=1, rms=0.5, seed=seed)
    fNMDA=DoubleExp(10.6e-3, 285e-3)
    fPre=DoubleExp(1e-3, 1e-1)
    dPreA = np.load(file)['dPreA']
    ePreA = np.load(file)['ePreA']
    dDiff = np.load(file)['dDiff']
    eDiff = np.load(file)['eDiff']
    dBio = np.load(file)['dBio']

    with nengo.Network(seed=0) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(NPre, 1, max_rates=m, seed=0)
        diff = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=lambda t: 1), seed=0)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=Bio("Pyramidal", DA=lambda t: 1), seed=0)
        nengo.Connection(inpt, pre, synapse=None, seed=0)
        c1 = nengo.Connection(pre, diff, synapse=fPre, solver=NoSolver(dPreA), seed=0)
        c2 = nengo.Connection(diff, ens, synapse=NMDA(), solver=NoSolver(dDiff), seed=0)
        nengo.Connection(inpt, intg, synapse=1/s)
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pDiff = nengo.Probe(diff.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=0, progress_bar=False) as sim:
        setWeights(c1, dPreA, ePreA)
        setWeights(c2, dDiff, eDiff)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    aDiff = fNMDA.filt(fPre.filt(sim.data[pDiff]))
    aEns = fNMDA.filt(fNMDA.filt(fPre.filt(sim.data[pEns])))
    xhatDiff = np.dot(aDiff, dDiff)
    xhatEns = np.dot(aEns, dBio)
    u = fNMDA.filt(fPre.filt(sim.data[pInpt]))
    x = fNMDA.filt(fPre.filt(sim.data[pIntg]))
    error = rmse(xhatEns, x)
    fig, ax = plt.subplots()
    ax.plot(sim.trange(), u, alpha=0.5, label="input")
    ax.plot(sim.trange(), x, label="integral")
    ax.plot(sim.trange(), xhatDiff, label="diff")
    ax.plot(sim.trange(), xhatEns, label="ens")
    ax.set(xlabel='time', ylabel='state', title="rmse=%.3f"%error, xlim=((0, t)), ylim=((-1, 1)))
    ax.legend(loc='upper left')
    sns.despine()
    fig.savefig("plots/diffMemoryTest_seed%s.pdf"%seed)

for seed in range(10):
    test(seed=seed)