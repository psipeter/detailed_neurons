import numpy as np
import nengo
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import decodeNoF, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
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
        x = f.filt(x, dt=dt)
        norm = value / np.max(np.abs(x))
    elif norm == 'u':
        u = f.filt(u, dt=dt)
        norm = value / np.max(np.abs(u))
    mirroredU = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    mirroredX = np.concatenate(([[0]], sim.data[pX]*norm, -sim.data[pX]*norm))
    return lambda t: mirroredU[int(t/dt)], lambda t: mirroredX[int(t/dt)]


def go(NPre=100, N=100, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), neuron_type=LIF(),
        seed=0, dt=0.001, fPre=None, fEns=None, fS=DoubleExp(2e-2, 2e-1),
        dPreA=None, dPreB=None, dEns=None, ePreA=None, ePreB=None, eBio=None,
        stage=None, alpha=1e-6, eMax=1e0, stimA=lambda t: np.sin(t), stimB=lambda t: 0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preInptA = nengo.Ensemble(NPre, 1, radius=3, max_rates=m, seed=seed)
        preInptB = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
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
            learnEncoders(c1b, tarEns, fS, alpha=alpha, eMax=eMax)
        if stage==2:
            c0a = nengo.Connection(preInptA, tarEns, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c0b = nengo.Connection(preInptB, tarEns, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            learnEncoders(c1a, tarEns, fS, alpha=alpha, eMax=eMax)
        if stage==3:
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preInptB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
        if stage==4:
            preInptC = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
            ens2 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
            nengo.Connection(inptB, preInptC, synapse=fEns, seed=seed)
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2b = nengo.Connection(preInptB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed+1)
            c3 = nengo.Connection(ens2, ens, synapse=fEns, solver=NoSolver(dEns), seed=seed)
            c4a = nengo.Connection(preInptA, ens3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c4b = nengo.Connection(preInptC, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c3, ens3, fS, alpha=alpha/10, eMax=eMax)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==5:
            c1a = nengo.Connection(preInptA, ens, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c5 = nengo.Connection(ens, ens, synapse=fEns, solver=NoSolver(dEns), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Bio):
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
                setWeights(c3, dEns, eBio)
            if stage==5:
                setWeights(c1a, dPreA, ePreA)
                setWeights(c5, dEns, eBio)
            neuron.h.init()
            sim.run(t, progress_bar=True)
            reset_neuron(sim, model)
        else:
            sim.run(t, progress_bar=True)
            

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


def run(NPre=100, N=30, t=10, nTrain=5, nEnc=5, nTest=10, dt=0.001, neuron_type=LIF(),
        fPre=DoubleExp(1e-3, 1e-1), fS=DoubleExp(2e-2, 2e-1),
        Tff=0.3, reg=1e-1, tauRiseMax=1e-1, tauFallMax=3e-1, load=[], file="data/integrate"):

    print('\nNeuron Type: %s'%neuron_type)
    file = file+f"{neuron_type}.npz"

    if 0 in load:
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
            data = go(NPre=NPre, N=N, t=t, dt=0.001, neuron_type=neuron_type, fPre=fPre, fS=fS, stimA=stimA, stimB=stimB)
            spikesInptA[n] = data['preInptA']
            spikesInptB[n] = data['preInptB']
            targetsInptA[n] = fPre.filt(Tff*data['inptA'], dt=0.001)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=0.001)
        dPreA, X1a, Y1a, error1a = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=0.001, reg=reg)
        dPreB, X1b, Y1b, error1b = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=0.001, reg=reg)
        np.savez(file, dPreA=dPreA, dPreB=dPreB)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X1a, Y1a, error1a, "integrate", "%s_preInptA"%neuron_type, t*nTrain)
        plotState(times, X1b, Y1b, error1b, "integrate", "%s_preInptB"%neuron_type, t*nTrain)

    if 1 in load:
        ePreB = np.load(file)['ePreB']
    elif isinstance(neuron_type, Bio):
        print("encoders for preInptB-to-ens")
        ePreB = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=dt, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreB=ePreB,
                NPre=NPre, N=N, t=t, dt=dt, neuron_type=neuron_type,
                fPre=fPre, fS=fS,
                stimA=stimA, stimB=stimB, stage=1)
            ePreB = data['ePreB']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrate", "preIntgToEns")
            np.savez(file, dPreA=dPreA, dPreB=dPreB, ePreB=ePreB)
    else:
        ePreB = np.zeros((NPre, N, 1))

    if 2 in load:
        ePreA = np.load(file)['ePreA']
    elif isinstance(neuron_type, Bio):
        print("encoders for preInptA-to-ens")
        ePreA = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=dt, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
                fPre=fPre, NPre=NPre, N=N, t=t, dt=dt, neuron_type=neuron_type, fS=fS,
                stimA=stimA, stimB=stimB, stage=2)
            ePreA = data['ePreA']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrate", "preInptToEns")
            np.savez(file, dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB)
    else:
        ePreA = np.zeros((NPre, N, 1))

    if 3 in load:
        dEns = np.load(file)['dEns']
        tauRiseEns = np.load(file)['tauRiseEns']
        tauFallEns = np.load(file)['tauFallEns']
        fEns = DoubleExp(tauRiseEns, tauFallEns)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, dt=dt, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
                NPre=NPre, N=N, t=t, dt=dt, neuron_type=neuron_type, fPre=fPre, fS=fS,
                stimA=stimA, stimB=stimB, stage=3)
            spikes[n] = data['ens']
            targets[n] = fPre.filt(data['inptB'], dt=dt)
        dEns, fEns, tauRiseEns, tauFallEns, X2, Y2, error2 = decode(spikes, targets, nTrain, dt=dt, reg=reg, tauRiseMax=tauRiseMax, tauFallMax=tauFallMax, name="integrate")
        np.savez(file, dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dEns=dEns, tauRiseEns=tauRiseEns, tauFallEns=tauFallEns)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X2, Y2, error2, "integrate", "%s_ens"%neuron_type, t*nTrain)

    if 4 in load:
        eBio = np.load(file)['eBio']
    elif isinstance(neuron_type, Bio):
        print("encoders from ens2 to ens")
        eBio = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, dt=dt, seed=n)
            data = go(dPreA=dPreA, dPreB=dPreB, dEns=dEns, ePreA=ePreA, ePreB=ePreB, eBio=eBio,
                NPre=NPre, N=N, t=t, dt=dt, neuron_type=neuron_type, fPre=fPre, fEns=fEns, fS=fS,
                stimA=stimA, stimB=stimB, stage=4)
            eBio = data['eBio']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrate", "Ens2Ens")
            np.savez(file, dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dEns=dEns, tauRiseEns=tauRiseEns, tauFallEns=tauFallEns, eBio=eBio)
    else:
        eBio = np.zeros((N, N, 1))

    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, dt=dt, seed=200+test)
        data = go(dPreA=dPreA, dEns=dEns, ePreA=ePreA, eBio=eBio,
            NPre=NPre, N=N, t=t, dt=dt, neuron_type=neuron_type, fPre=fPre, fEns=fEns, fS=fS,
            stimA=stimA, stimB=stimB, stage=5)
        A = fEns.filt(data['ens'], dt=dt)
        X = np.dot(A, dEns)
        Y = fPre.filt(data['inptB'], dt=dt)
        U = fPre.filt(Tff*data['inptA'], dt=dt)
        error = rmse(X, Y)
        errorU = rmse(X, U)
        errors[test] = error
        fig, ax = plt.subplots()
#         ax.plot(data['times'], U, label="input")
        ax.plot(data['times'], X, label="estimate")
        ax.plot(data['times'], Y, label="target")
        ax.set(xlabel='time', ylabel='state', title='rmse=%.3f'%error, xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrate_%s_test%s.pdf"%(neuron_type, test))
        plt.close('all')
    # print('errors:', errors)
    # np.savez(file, errors=errors, dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB, dEns=dEns, tauRiseEns=tauRiseEns, tauFallEns=tauFallEns, eBio=eBio)

    return data['times'], X, Y


times, XLIF, Y = run(neuron_type=LIF(), dt=1e-4, load=[0,1,2,3,4], nTest=1)
times, XALIF, Y = run(neuron_type=ALIF(), dt=1e-4, load=[0,1,2,3,4], nTest=1)
times, XWilson, Y = run(neuron_type=Wilson(), dt=1e-4, load=[0,1,2,3,4], nTest=1)
times, XBio, Y = run(neuron_type=Bio("Pyramidal"), dt=1e-4, load=[0,1,2,3,4], nTest=1)
eLIF = rmse(XLIF, Y)
eALIF = rmse(XALIF, Y)
eWilson = rmse(XWilson, Y)
eBio = rmse(XBio, Y)

yticks = np.array([-1, 0, 1])
xticks = np.array([0, 2, 4, 6, 8, 10])
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=((6, 4)), sharex=True, sharey=True)
ax0.plot(times, Y, color='k')
ax0.plot(times, XLIF, color='tab:blue')
ax1.plot(times, Y, color='k')
ax1.plot(times, XALIF, color='tab:orange')
ax2.plot(times, Y, color='k')
ax2.plot(times, XWilson, color='tab:green')
ax3.plot(times, Y, color='k')
ax3.plot(times, XBio, color='tab:red')
ax0.set(xlim=((0, 10)), ylim=((-1, 1)), xticks=xticks, yticks=yticks, title=f"LIF, RMSE={eLIF:.3f}", ylabel=r"$\mathbf{x}(t)$")
ax1.set(xlim=((0, 10)), ylim=((-1, 1)), xticks=xticks, yticks=yticks, title=f"ALIF, RMSE={eALIF:.3f}")
ax2.set(xlim=((0, 10)), ylim=((-1, 1)), xticks=xticks, yticks=yticks, title=f"Wilson, RMSE={eWilson:.3f}", xlabel="time (s)", ylabel=r"$\mathbf{x}(t)$")
ax3.set(xlim=((0, 10)), ylim=((-1, 1)), xticks=xticks, yticks=yticks, title=f"Durstewitz, RMSE={eBio:.3f}", xlabel="time (s)")
fig.savefig("plots/integrate_all.pdf")