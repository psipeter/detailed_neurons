import numpy as np
import nengo
from nengo.dists import Uniform, Gaussian, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from utils import decodeNoF, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron, NMDA, GABA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, fPre, fNMDA, dt=0.001, value=1.0, norm='x', seed=0):
    stim = nengo.processes.WhiteSignal(period=t/2, high=1, rms=0.5, seed=seed)
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
    funcU = lambda t: mirroredU[int(t/dt)]
    funcX = lambda t: mirroredX[int(t/dt)]
    return funcU, funcX


def go(NPre=300, NBias=30, N=30, t=10, seed=1, dt=0.001, Tff=0.3, tTrans=0.01,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(2e-2, 2e-1),
        alpha=1e-6, eMax=3e-1,
        dPre=None, dFdfw=None, dEns=None, dBias=None,
        ePreFdfw=None, ePreBias=None, ePreEns=None, eFdfwEns=None, eEnsEns=None, eBiasEns=None,
        stage=None, stimA=lambda t: 0, stimB=lambda t: 0, DA=lambda t: 0):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        preA = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preB = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        fdfw = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        tarFdfw = nengo.Ensemble(N, 1, neuron_type=ALIF(), max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed)
        tarEns = nengo.Ensemble(N, 1, neuron_type=ALIF(), max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed+1)
        cA = nengo.Connection(inptA, preA, synapse=None, seed=seed)
        cB = nengo.Connection(inptB, preB, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pPreA = nengo.Probe(preA.neurons, synapse=None)
        pPreB = nengo.Probe(preB.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if stage==0:  # readout decoders for [preA, preB, preC]
            pass
        if stage==1:  # encoders for [preA, preC] to [fdfw]
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            learnEncoders(c1, tarFdfw, fS, alpha=alpha, eMax=eMax, tTrans=tTrans, fPre=fPre)
        if stage==2:  # readout decoders for fdfw
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
        if stage==3:  # encoders for [preB] to [ens]
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed+1)
            learnEncoders(c1, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=tTrans, fPre=fPre)
        if stage==4:  # encoders for [fdfw] to ens
            cB.synapse = fNMDA
            tarPreA = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            tarPreB = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(inptA, tarPreA, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarPreB, synapse=fNMDA, seed=seed)
            nengo.Connection(tarPreA, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            nengo.Connection(tarPreB, tarEns, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed+1)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed+2)
            learnEncoders(c3, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=tTrans, fPre=fNMDA)
        if stage==5:  # readout decoders for ens
            cB.synapse = fNMDA
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed+1)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed+2)
        if stage==6: # encoders from ens to ens
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            ens2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)  # acts as preB input to ens
            ens3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)  # acts as tarEns
            nengo.Connection(inptB, preB2, synapse=fNMDA, seed=seed)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens2, synapse=fPre, solver=NoSolver(dPre), seed=seed+1)
            c3 = nengo.Connection(preB2, ens3, synapse=fPre, solver=NoSolver(dPre), seed=seed+1)
            c4 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed+2)
            c5 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed+2)
            c6 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed+3)
            learnEncoders(c6, ens3, fS, alpha=alpha/3, eMax=eMax/3, tTrans=tTrans, fPre=fNMDA)
        if stage==7:  # test
            inptC = nengo.Node(lambda t: 1)
            preC = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            bias = nengo.Ensemble(NBias, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+2)
            nengo.Connection(inptC, preC, synapse=None)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed+1)
            c3 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed+3)
            c4 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed+4)
            c5 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed+5)
            # pBias = nengo.Probe(bias.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==0:
            pass
        if stage==1:
            setWeights(c1, dPre, ePreFdfw)
        if stage==2:
            setWeights(c1, dPre, ePreFdfw)
        if stage==3:
            setWeights(c1, dPre, ePreEns)
        if stage==4:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==5:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==6:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dPre, ePreEns)
            setWeights(c4, dFdfw, eFdfwEns)
            setWeights(c5, dFdfw, eFdfwEns)
            setWeights(c6, dEns, eEnsEns)
        if stage==7:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c3, dEns, eEnsEns)
            setWeights(c4, dPre, ePreBias)
            setWeights(c5, dBias, eBiasEns)

        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreFdfw = c1.e if stage==1 else ePreFdfw
    ePreEns = c1.e if stage==3 else ePreEns
    eFdfwEns = c3.e if stage==5 else eFdfwEns
    eEnsEns = c6.e if stage==6 else eEnsEns

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        preA=sim.data[pPreA],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        # bias=sim.data[pBias] if stage==7 else None,
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        ePreFdfw=ePreFdfw,
        ePreEns=ePreEns,
        eFdfwEns=eFdfwEns,
        eEnsEns=eEnsEns,
    )


def run(NPre=300, NBias=30, N=30, t=10, nTrain=10, nTest=5, nEnc=10, dt=0.001,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(2e-2, 2e-1),
        DATrain=lambda t: 0, DATest=lambda t: 0,
        Tff=0.3, reg=1e-2, load=[], file=None):

    if 0 in load:
        dPre = np.load(file)['dPre']
    else:
        print('readout decoders for [preA, preB]')
        spikesInpt = np.zeros((nTrain, int(t/dt), NPre))
        targetsInpt = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                stage=0)
            spikesInpt[n] = data['preA']
            targetsInpt[n] = fPre.filt(data['inptA'], dt=dt)
        dPre, X, Y, error = decodeNoF(spikesInpt, targetsInpt, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
        )
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDAbias", "pre", t*nTrain)

    if 1 in load:
        ePreFdfw = np.load(file)['ePreFdfw']
    else:
        print("encoders for [preA, preC] to [fdfw]")
        ePreFdfw = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre,
                ePreFdfw=ePreFdfw,
                stage=1)
            ePreFdfw = data['ePreFdfw']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDAbias", "pre_fdfw")
            np.savez("data/integrateNMDAbias.npz",
                dPre=dPre,
                ePreFdfw=ePreFdfw)

    if 2 in load:
        dFdfw = np.load(file)['dFdfw']
    else:
        print('readout decoders for fdfw ')
        spikesFdfw = np.zeros((nTrain, int(t/dt), N))
        targetsFdfw = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA,
                dPre=dPre,
                ePreFdfw=ePreFdfw,
                stage=2)
            spikesFdfw[n] = data['fdfw']
            targetsFdfw[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
        dFdfw, X1, Y1, error1 = decodeNoF(spikesFdfw, targetsFdfw, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            dFdfw=dFdfw)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1, Y1, error1, "integrateNMDAbias", "fdfw", t*nTrain)


    if 3 in load:
        ePreEns = np.load(file)['ePreEns']
    else:
        print("encoders for [preB] to [ens]")
        ePreEns = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre,
                ePreEns=ePreEns,
                stage=3)
            ePreEns = data['ePreEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias", "pre_ens")
            np.savez("data/integrateNMDAbias.npz",
                dPre=dPre,
                ePreFdfw=ePreFdfw,
                dFdfw=dFdfw,
                ePreEns=ePreEns,)

    if 4 in load:
        eFdfwEns = np.load(file)['eFdfwEns']
    else:
        print("encoders for [fdfw] to ens")
        eFdfwEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, Tff=Tff,
                dPre=dPre, dFdfw=dFdfw,
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns,
                stage=4)
            eFdfwEns = data['eFdfwEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias", "fdfw_ens")
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            dFdfw=dFdfw,
            eFdfwEns=eFdfwEns)

    if 5 in load:
        dEns = np.load(file)['dEns']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre, dFdfw=dFdfw,
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns,
                stage=5)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
        dEns, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            dFdfw=dFdfw,
            eFdfwEns=eFdfwEns,
            dEns=dEns)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDAbias", "ens", t*nTrain)

    if 6 in load:
        eEnsEns = np.load(file)['eEnsEns']
    else:
        print("encoders from ens to ens")
        eEnsEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre, dFdfw=dFdfw, dEns=dEns,
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns,
                stage=6)
            eEnsEns = data['eEnsEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias", "ens_ens")
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            dFdfw=dFdfw,
            eFdfwEns=eFdfwEns,
            dEns=dEns,
            eEnsEns=eEnsEns)

    if 7 in load:
        dBias = np.load(file)['dBias']
        ePreBias = np.load(file)['ePreBias']
        eBiasEns = np.load(file)['eBiasEns']
    else:
        dBias = -5e-5 * np.ones((NBias, 1))
        ePreBias = 1e-2 * np.ones((NPre, NBias, 1))
        eBiasEns = 1e0 * np.ones((NBias, N, 1))
        np.savez("data/integrateNMDAbias.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            dFdfw=dFdfw,
            eFdfwEns=eFdfwEns,
            dEns=dEns,
            eEnsEns=eEnsEns,
            dBias=dBias,
            ePreBias=ePreBias,
            eBiasEns=eBiasEns)

    print("testing")
    vals = np.linspace(-1, 1, nTest)
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=200+test)
        data = go(
            NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATest,
            stimA=stimA, stimB=stimB, Tff=Tff,
            dPre=dPre, dFdfw=dFdfw, dEns=dEns, dBias=dBias,
            ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, ePreBias=ePreBias, eBiasEns=eBiasEns,
            stage=7)
        aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        aBio = fNMDA.filt(data['ens'], dt=dt)
        xhatFdfw = np.dot(aFdfw, dFdfw)
        xhatEns = np.dot(aBio, dEns)
        xFdfw = fNMDA.filt(fPre.filt(data['inptA'], dt=dt), dt=dt)
        xEns = fNMDA.filt(fPre.filt(data['inptB'], dt=dt), dt=dt)
        errorBio = rmse(xhatEns, xEns)
        fig, ax = plt.subplots()
        ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        ax.plot(data['times'], Tff*xFdfw, alpha=0.5, label="input")
        ax.plot(data['times'], xEns,  label="target")
        ax.plot(data['times'], xhatEns, alpha=0.5, label="ens, rmse=%.3f"%errorBio)
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDAbias_testWhite%s.pdf"%test)
        plt.close('all')

run(NBias=50, N=50, t=10, nTrain=10, nEnc=10, nTest=10, DATrain=lambda t: 0, DATest=lambda t: 0.2,
    load=[0,1,2,3,4,5,6,7], file="data/integrateNMDAbias.npz")