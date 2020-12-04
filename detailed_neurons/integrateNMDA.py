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
    phase = rng.uniform(0, 2*np.pi)
    mag = value
    freq = 2*np.pi/t
    sin = lambda t: mag*np.sin(freq*t + phase)
    return sin

def makeFlat(t, dt=0.001, seed=0):
    idx = seed % 3
    vals = [-0.8, 0, 0.8]
    flat = lambda t: vals[idx]
    return flat

def go(NPre=100, N=100, t=10, c=None, seed=0, dt=0.001, Tff=0.3, tTrans=0.01,
        stage=None, alpha=3e-7, eMax=1e-1,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fS=DoubleExp(1e-3, 1e-1),
        dPreA=None, dPreB=None, dPreC=None, dFdfw=None, dBio=None, dNeg=None, dInh=None,
        ePreA=None, ePreB=None, ePreC=None, eFdfw=None, eBio=None, eNeg=None, eInh=None,
        stimA=lambda t: 0, stimB=lambda t: 0, stimC=lambda t: 0, DA=lambda t: 0):

    if not c: c = t
    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        inptC = nengo.Node(stimC)
        preA = nengo.Ensemble(NPre, 1, max_rates=Uniform(20, 40), radius=2, seed=seed)
        preB = nengo.Ensemble(NPre, 1, max_rates=Uniform(20, 40), seed=seed)
        preC = nengo.Ensemble(NPre, 1, max_rates=Uniform(20, 40), seed=seed)
        fdfw = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        inh = nengo.Ensemble(N, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+2)
        tarFdfw = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), radius=2, neuron_type=nengo.LIF(), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), neuron_type=nengo.LIF(), seed=seed+1)
        tarInh = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(0.2, 0.8), encoders=Choice([[1]]), neuron_type=nengo.LIF(), seed=seed+2)
        cA = nengo.Connection(inptA, preA, synapse=None, seed=seed)
        cB = nengo.Connection(inptB, preB, synapse=None, seed=seed)
        cC = nengo.Connection(inptC, preC, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pInptC = nengo.Probe(inptC, synapse=None)
        pPreA = nengo.Probe(preA.neurons, synapse=None)
        pPreB = nengo.Probe(preB.neurons, synapse=None)
        pPreC = nengo.Probe(preC.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pTarInh = nengo.Probe(tarInh.neurons, synapse=None)
        if stage==1:
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarInh, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(preC, inh, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            learnEncoders(c1, tarFdfw, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            learnEncoders(c2, tarEns, fS, alpha=3*alpha, eMax=10*eMax, tTrans=tTrans)
            learnEncoders(c3, tarInh, fS, alpha=alpha/3, eMax=eMax, tTrans=tTrans)
        if stage==2:
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preC, inh, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
        if stage==3:
            cB.synapse = fNMDA
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(tarFdfw, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            learnEncoders(c3, tarEns, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
        if stage==4:
            cB.synapse = fNMDA
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        if stage==5:
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(20, 40), seed=seed)
            ens2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            ens3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            nengo.Connection(inptB, preB2, synapse=fNMDA, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(preB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c4 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c5 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c6 = nengo.Connection(preB2, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c4, ens3, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==6:
            preA2 = nengo.Ensemble(NPre, 1, radius=2, max_rates=Uniform(20, 40), seed=seed)
            fdfw2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            fdfw3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            fdfw4 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            tarFdfw4 = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), radius=2, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(inptA, tarFdfw4, synapse=fPre, seed=seed)
            nengo.Connection(inptB, preA2, synapse=fNMDA, seed=seed)
            nengo.Connection(inptC, tarInh, synapse=fPre, seed=seed)
            nengo.Connection(tarInh, tarFdfw4.neurons, synapse=None, transform=-1e2*np.ones((N, 1)), seed=seed)
            c1 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c2 = nengo.Connection(ens, fdfw2, synapse=NMDA(), solver=NoSolver(dNeg), seed=seed)
            c3 = nengo.Connection(preA2, fdfw3, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c4 = nengo.Connection(preA, fdfw4, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c5 = nengo.Connection(preC, inh, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            c6 = nengo.Connection(inh, fdfw4, synapse=GABA(), solver=NoSolver(dInh), seed=seed)
            learnEncoders(c2, fdfw3, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            learnEncoders(c6, tarFdfw4, fS, alpha=1e3*alpha, eMax=1e3*eMax, tTrans=tTrans, inh=True)
            pFdfw2 = nengo.Probe(fdfw2.neurons, synapse=None)
            pFdfw4 = nengo.Probe(fdfw4.neurons, synapse=None)
            pTarFdfw2 = nengo.Probe(fdfw3.neurons, synapse=None)
            pTarFdfw4 = nengo.Probe(tarFdfw4.neurons, synapse=None)
        if stage==7:
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dBio), seed=seed)
            c4 = nengo.Connection(ens, fdfw, synapse=NMDA(), solver=NoSolver(dNeg), seed=seed)
            c5 = nengo.Connection(preC, inh, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            c6 = nengo.Connection(inh, fdfw, synapse=GABA(), solver=NoSolver(dInh), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==1:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
            setWeights(c3, dPreC, ePreC)
        if stage==2:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreC, ePreC)
        if stage==3:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dPreB, ePreB)
            setWeights(c3, dFdfw, eFdfw)
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
            setWeights(c1, dPreB, ePreB)
            setWeights(c2, dNeg, eNeg)
            setWeights(c3, dPreA, ePreA)
            setWeights(c4, dPreA, ePreA)
            setWeights(c5, dPreC, ePreC)
            setWeights(c6, dInh, eInh)
        if stage==7:
            setWeights(c1, dPreA, ePreA)
            setWeights(c2, dFdfw, eFdfw)
            setWeights(c3, dBio, eBio)
            setWeights(c4, dNeg, eNeg)
            setWeights(c5, dPreC, ePreC)
            setWeights(c6, dInh, eInh)
            # setWeights(c6, -np.ones((N, 1)), np.ones((N, N)))
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreA = c1.e if stage==1 else ePreA
    ePreB = c2.e if stage==1 else ePreB
    ePreC = c3.e if stage==1 else ePreC
    eFdfw = c3.e if stage==3 else eFdfw
    eBio = c4.e if stage==5 else eBio
    eNeg = c2.e if stage==6 else eNeg
    eInh = c6.e if stage==6 else eInh

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        inptC=sim.data[pInptC],
        preA=sim.data[pPreA],
        preB=sim.data[pPreB],
        preC=sim.data[pPreC],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        inh=sim.data[pInh],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        tarInh=sim.data[pTarInh],
        fdfw2=sim.data[pFdfw2] if stage==6 else None,
        fdfw4=sim.data[pFdfw4] if stage==6 else None,
        tarFdfw2=sim.data[pTarFdfw2] if stage==6 else None,
        tarFdfw4=sim.data[pTarFdfw4] if stage==6 else None,
        ePreA=ePreA,
        ePreB=ePreB,
        ePreC=ePreC,
        eFdfw=eFdfw,
        eBio=eBio,
        eNeg=eNeg,
        eInh=eInh,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nEnc=10, dt=0.001, c=None,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(1e-3, 1e-1),
        Tff=0.3, Tneg=-1, reg=1e-2, load=[], file=None, DATest=lambda t: 0):

    if not c: c = t
    if 0 in load:
        dPreA = np.load(file)['dPreA']
        dPreB = np.load(file)['dPreB']
        dPreC = np.load(file)['dPreC']
    else:
        print('readout decoders for pre[A,B,C]')
        spikesInptA = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptB = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptC = np.zeros((nTrain, int(t/dt), NPre))
        targetsInptA = np.zeros((nTrain, int(t/dt), 1))
        targetsInptB = np.zeros((nTrain, int(t/dt), 1))
        targetsInptC = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimB,
                stage=0)
            spikesInptA[n] = data['preA']
            spikesInptB[n] = data['preB']
            spikesInptC[n] = data['preC']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=dt)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=dt)
            targetsInptC[n] = fPre.filt(data['inptC'], dt=dt)
        dPreA, XA, YA, errorA = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=dt, reg=reg)
        dPreB, XB, YB, errorB = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=dt, reg=reg)
        dPreC, XC, YC, errorC = decodeNoF(spikesInptC, targetsInptC, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, XA, YA, errorA, "integrateNMDA", "preA", t*nTrain)
        plotState(times, XB, YB, errorB, "integrateNMDA", "preB", t*nTrain)
        plotState(times, XC, YC, errorC, "integrateNMDA", "preC", t*nTrain)

    if 1 in load:
        ePreA = np.load(file)['ePreA']
        ePreB = np.load(file)['ePreB']
        ePreC = np.load(file)['ePreC']
    else:
        print("encoders for pre[A,B,C] to [fdfw, ens, inh]")
        ePreA = np.zeros((NPre, N, 1))
        ePreB = np.zeros((NPre, N, 1))
        ePreC = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC,
                stage=1)
            ePreA = data['ePreA']
            ePreB = data['ePreB']
            ePreC = data['ePreC']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDA", "preAFdfw")
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preBEns")
            plotActivity(t, dt, fS, data['times'], data['inh'], data['tarInh'], "integrateNMDA", "preCInh")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC)

    if 2 in load:
        dFdfw = np.load(file)['dFdfw']
        dInh = np.load(file)['dInh']
    else:
        print('readout decoders for fdfw and inh')
        spikesFdfw = np.zeros((nTrain, int(t/dt), N))
        spikesInh = np.zeros((nTrain, int(t/dt), N))
        targetsFdfw = np.zeros((nTrain, int(t/dt), 1))
        targetsInh = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimC = lambda x: 0.5+0.5*np.sin(2*np.pi*x/t)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimC, 
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC,
                stage=2)
            spikesFdfw[n] = data['fdfw']
            spikesInh[n] = data['inh']
            targetsFdfw[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
            targetsInh[n] = fGABA.filt(fPre.filt(data['inptC'], dt=dt))
        dFdfw, X1, Y1, error1 = decodeNoF(spikesFdfw, targetsFdfw, nTrain, fNMDA, dt=dt, reg=reg)
        dInh, X2, Y2, error2 = decodeNoF(spikesInh, targetsInh, nTrain, fGABA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dInh=dInh,
            ePreA=ePreA, ePreB=ePreB, ePreC=ePreC)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1, Y1, error1, "integrateNMDA", "fdfw", t*nTrain)
        plotState(times, X2, Y2, error2, "integrateNMDA", "inh", t*nTrain)
        
    if 3 in load:
        eFdfw = np.load(file)['eFdfw']
    else:
        print("encoders for fdfw-to-ens")
        eFdfw = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, Tff=Tff,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw,
                ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw,
                stage=3)
            eFdfw = data['eFdfw']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "fdfwEns")
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dInh=dInh, 
            ePreA=ePreA, ePreB=ePreB, ePreC=ePreC, eFdfw=eFdfw)

    if 4 in load:
        dBio = np.load(file)['dBio']
        dNeg = np.load(file)['dNeg']
        # dNeg = -np.array(dBio)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw,
                ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw,
                stage=4)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
            # targets[n] = fNMDA.filt(
            #     fPre.filt(fNMDA.filt(data['inptB'])) +
            #     fNMDA.filt(Tff*fPre.filt(data['inptA'])))
        dBio, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        dNeg = -np.array(dBio)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dInh=dInh, dBio=dBio, dNeg=dNeg,
            ePreA=ePreA, ePreB=ePreB, ePreC=ePreC, eFdfw=eFdfw)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "ens", t*nTrain)

    if 5 in load:
        eBio = np.load(file)['eBio']
    else:
        print("encoders from ens to ens")
        eBio = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dFdfw=dFdfw, dBio=dBio,
                ePreA=ePreA, ePreB=ePreB, eFdfw=eFdfw, eBio=eBio,
                stage=5)
            eBio = data['eBio']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "ensEns")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dBio=dBio, dInh=dInh, dNeg=dNeg,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC, eFdfw=eFdfw, eBio=eBio)

    if 6 in load:
        eNeg = np.load(file)['eNeg']
        eInh = np.load(file)['eInh']
    else:
        print("encoders from [ens, inh] to fdfw")
        eNeg = np.zeros((N, N, 1))
        eInh = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimC = lambda t: 0 if t<c else 1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, stimC=stimC,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dBio=dBio, dNeg=dNeg, dInh=dInh,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC, eFdfw=eFdfw, eBio=eBio, eNeg=eNeg, eInh=eInh,
                stage=6)
            eNeg = data['eNeg']
            eInh = data['eInh']
            plotActivity(t, dt, fS, data['times'], data['fdfw2'], data['tarFdfw2'], "integrateNMDA", "ensFdfw")
            plotActivity(t, dt, fS, data['times'], data['fdfw4'], data['tarFdfw4'], "integrateNMDA", "inhFdfw")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dFdfw=dFdfw, dBio=dBio, dNeg=dNeg, dInh=dInh,
                ePreA=ePreA, ePreB=ePreB, ePreC=ePreC, eFdfw=eFdfw, eBio=eBio, eNeg=eNeg, eInh=eInh)
    # eNeg = None
    # eInh = None

    print("testing")
    vals = np.linspace(-1, 1, nTest)
    for test in range(nTest):

        stimA = lambda t: vals[test] if t<c else 0
        stimC = lambda t: 0 if t<c else 1
        data = go(
            NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimC=stimC,
            dPreA=dPreA, dPreC=dPreC, dFdfw=dFdfw, dBio=dBio, dNeg=Tneg*dNeg, dInh=dInh,
            ePreA=ePreA, ePreC=ePreC, eFdfw=eFdfw, eBio=eBio, eNeg=eNeg, eInh=eInh,
            stage=7, c=c, DA=DATest)
        aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        aBio = fNMDA.filt(data['ens'], dt=dt)
        aInh = fGABA.filt(data['inh'], dt=dt)
        xhatFdfw = np.dot(aFdfw, dFdfw)/Tff
        xhatBio = np.dot(aBio, dBio)
        xhatInh = np.dot(aInh, dInh)
        xFdfw = fNMDA.filt(fNMDA.filt(fPre.filt(data['inptA'], dt=dt), dt=dt), dt=dt)
        xBio = xFdfw[int(c/dt)] * np.ones_like(data['times'])
        errorBio = rmse(xhatBio[int(c/dt):], xBio[int(c/dt):])
        fig, ax = plt.subplots()
        ax.plot(data['times'], xFdfw, alpha=0.5, label="input (filtered)")
        ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        ax.plot(data['times'], xBio, label="target")
        ax.plot(data['times'], xhatBio, alpha=0.5, label="ens, rmse=%.3f"%errorBio)
        ax.plot(data['times'], xhatInh, alpha=0.5, label="inh")
        ax.axvline(c, label="cutoff")
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_testFlat%s.pdf"%test)

        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=200+test)
        stimC = lambda t: 0
        data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, stimC=stimC, Tff=Tff,
            dPreA=dPreA, dPreC=dPreC, dFdfw=dFdfw, dBio=dBio, dNeg=None, dInh=dInh,
            ePreA=ePreA, ePreC=ePreC, eFdfw=eFdfw, eBio=eBio, eNeg=None, eInh=eInh,
            stage=7, DA=DATest)
        aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        aBio = fNMDA.filt(data['ens'], dt=dt)
        xhatFdfw = np.dot(aFdfw, dFdfw)
        xhatBio = np.dot(aBio, dBio)
        xhatNeg = np.dot(aBio, dNeg)
        xFdfw = fNMDA.filt(fPre.filt(data['inptA'], dt=dt), dt=dt)
        xBio = fNMDA.filt(fPre.filt(data['inptB'], dt=dt), dt=dt)
        errorBio = rmse(xhatBio, xBio)
        fig, ax = plt.subplots()
        ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        ax.plot(data['times'], fNMDA.filt(Tff*xFdfw, dt=dt), alpha=0.5, label="input (filtered)")
        ax.plot(data['times'], xBio,  label="target")
        ax.plot(data['times'], xhatBio, alpha=0.5, label="ens, rmse=%.3f"%errorBio)
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_testWhite%s.pdf"%test)
        plt.close('all')

    # np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreA=ePreA, ePreB=ePreB,
        # dFdfw=dFdfw, eFdfw=eFdfw, dBio=dBio, dNeg=dNeg, eBio=eBio, eNeg=eNeg, eInh=eInh)

run(N=30, c=5, nTrain=3, nEnc=10, Tneg=-0.5, 
    load=[0,1,2,3,4,5,6], file="data/integrateNMDA.npz")
