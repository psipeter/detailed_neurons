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

def closestAttractor(value, attractors):
    idx = (np.abs(attractors - value)).argmin()
    return attractors[idx]

def makeSignal(t, fPre, fNMDA, nAttr, dt=0.001, value=1.0, seed=0, c=None):
    if not c: c = t
    stim = nengo.processes.WhiteSignal(period=t/2, high=0.1, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/2, progress_bar=False)
    u = fNMDA.filt(fPre.filt(sim.data[pU], dt=dt), dt=dt)
    norm = value / np.max(np.abs(u))
    mirrored = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    # m = fNMDA.filt(fPre.filt(mirrored))
    attr = []
    attractors = np.linspace(-value, value, nAttr)
    for i in range(len(mirrored)):
        if i*dt < c:
            # attr.append(closestAttractor(m[i], attractors))
            attr.append(closestAttractor(mirrored[i], attractors))
        else:
            attr.append(attr[-1])
    # fig, ax = plt.subplots()
    # ax.plot(mirrored)
    # ax.plot(attr)
    # fig.savefig('plots/testAttr.pdf')
    stimA = lambda t: mirrored[int(t/dt)] if t<c else 0
    stimB = lambda t: attr[int(t/dt)]
    return stimA, stimB


def go(NPre=100, N=100, t=10, c=None, seed=1, dt=0.001, tTrans=0.01,
        stage=None, alpha=3e-7, eMax=1e-1, Tff=0.3,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(1e-3, 1e-1),
        dPreA=None, dPreB=None, dPreC=None, dPreD=None, dFdfw=None, dEns=None, dOff=None,
        ePreAFdfw=None, ePreBEns=None, ePreCOff=None, eFdfwEns=None, eEnsEns=None, ePreDEns=None, eOffFdfw=None,
        stimA=lambda t: 0, stimB=lambda t: 0, stimC=lambda t: 0, stimD=lambda t: 0, DA=lambda t: 0):

    if not c: c = t
    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        inptC = nengo.Node(stimC)
        inptD = nengo.Node(stimD)
        preA = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preB = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preC = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preD = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        fdfw = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        off = nengo.Ensemble(N, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+3)
        tarFdfw = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed+1)
        tarOff = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(0.2, 0.8), encoders=Choice([[1]]), seed=seed+3)
        cA = nengo.Connection(inptA, preA, synapse=None, seed=seed)
        cB = nengo.Connection(inptB, preB, synapse=None, seed=seed)
        cC = nengo.Connection(inptC, preC, synapse=None, seed=seed)
        cD = nengo.Connection(inptD, preD, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pInptC = nengo.Probe(inptC, synapse=None)
        pInptD = nengo.Probe(inptD, synapse=None)
        pPreA = nengo.Probe(preA.neurons, synapse=None)
        pPreB = nengo.Probe(preB.neurons, synapse=None)
        pPreC = nengo.Probe(preC.neurons, synapse=None)
        pPreD = nengo.Probe(preD.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        pOff = nengo.Probe(off.neurons, synapse=None)
        pTarOff = nengo.Probe(tarOff.neurons, synapse=None)
        if stage==0:
            nengo.Connection(preD, tarEns, synapse=fPre, seed=seed)
            c0 = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            learnEncoders(c0, tarEns, fS, alpha=10*alpha, eMax=10*eMax, tTrans=tTrans)
        if stage==1:
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarOff, synapse=fPre, seed=seed)
            c0 = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(preC, off, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            learnEncoders(c1, tarFdfw, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
            learnEncoders(c2, tarEns, fS, alpha=10*alpha, eMax=10*eMax, tTrans=tTrans)
            learnEncoders(c3, tarOff, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
        if stage==2:
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preC, off, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
        if stage==3:
            cB.synapse = fNMDA
            ff = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            fb = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(inptA, ff, synapse=fPre, seed=seed)
            nengo.Connection(inptB, fb, synapse=fNMDA, seed=seed)
            nengo.Connection(fb, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(ff, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            c0 = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            learnEncoders(c3, tarEns, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
        if stage==4:
            cB.synapse = fNMDA
            c0 = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        if stage==5:
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            ens2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            ens3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            nengo.Connection(inptB, preB2, synapse=fNMDA, seed=seed)
            c0a = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c0b = nengo.Connection(preD, ens2, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c0c = nengo.Connection(preD, ens3, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(preB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c4 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)
            c5 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c6 = nengo.Connection(preB2, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c4, ens3, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
        if stage==9:
            c0 = nengo.Connection(preD, ens, synapse=fPre, solver=NoSolver(dPreD), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)
            c6 = nengo.Connection(preC, off, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            c7 = nengo.Connection(off, fdfw, synapse=GABA(), solver=NoSolver(dOff), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==0:
            setWeights(c0, dPreD, ePreDEns)
        if stage==1:
            setWeights(c0, dPreD, ePreDEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dPreC, ePreCOff)
        if stage==2:
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreC, ePreCOff)
        if stage==3:
            setWeights(c0, dPreD, ePreDEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==4:
            setWeights(c0, dPreD, ePreDEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==5:
            setWeights(c0a, dPreD, ePreDEns)
            setWeights(c0b, dPreD, ePreDEns)
            setWeights(c0c, dPreD, ePreDEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c3, dPreB, ePreBEns)
            setWeights(c4, dEns, eEnsEns)
            setWeights(c5, dFdfw, eFdfwEns)
            setWeights(c6, dPreB, ePreBEns)
        if stage==9:
            setWeights(c0, dPreD, ePreDEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c3, dEns, eEnsEns)
            setWeights(c6, dPreC, ePreCOff)
            setWeights(c7, dOff, eOffFdfw)

        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreDEns = c0.e if stage==0 else ePreDEns
    ePreAFdfw = c1.e if stage==1 else ePreAFdfw
    ePreBEns = c2.e if stage==1 else ePreBEns
    ePreCOff = c3.e if stage==1 else ePreCOff
    eFdfwEns = c3.e if stage==3 else eFdfwEns
    eEnsEns = c4.e if stage==5 else eEnsEns

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        inptC=sim.data[pInptC],
        inptD=sim.data[pInptD],
        preA=sim.data[pPreA],
        preB=sim.data[pPreB],
        preC=sim.data[pPreC],
        preD=sim.data[pPreD],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        off=sim.data[pOff],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        tarOff=sim.data[pTarOff],
        ePreAFdfw=ePreAFdfw,
        ePreBEns=ePreBEns,
        ePreCOff=ePreCOff,
        ePreDEns=ePreDEns,
        eFdfwEns=eFdfwEns,
        eEnsEns=eEnsEns,
        eOffFdfw=eOffFdfw,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=3, nAttr=5, nEnc=10, dt=0.001, c=None,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(2e-2, 1e-1),
        Tff=0.3, reg=1e-2, load=[], file=None):

    if not c: c = t
    if 0 in load:
        dPreA = np.load(file)['dPreA']
        dPreB = np.load(file)['dPreB']
        dPreC = np.load(file)['dPreC']
        dPreD = np.load(file)['dPreD']
    else:
        print('readout decoders for pre')
        spikesInptA = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptB = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptC = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptD = np.zeros((nTrain, int(t/dt), NPre))
        targetsInptA = np.zeros((nTrain, int(t/dt), 1))
        targetsInptB = np.zeros((nTrain, int(t/dt), 1))
        targetsInptC = np.zeros((nTrain, int(t/dt), 1))
        targetsInptD = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, _ = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            stimB = stimA
            stimC = lambda t: 0 if t<c else 1
            stimD = lambda t: 1e-1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimC, stimD=stimD,
                stage=None)
            spikesInptA[n] = data['preA']
            spikesInptB[n] = data['preB']
            spikesInptC[n] = data['preC']
            spikesInptD[n] = data['preD']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=dt)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=dt)
            targetsInptC[n] = fPre.filt(data['inptC'], dt=dt)
            targetsInptD[n] = fPre.filt(data['inptD'], dt=dt)
        dPreA, XA, YA, errorA = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=dt, reg=reg)
        dPreB, XB, YB, errorB = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=dt, reg=reg)
        dPreC, XC, YC, errorC = decodeNoF(spikesInptC, targetsInptC, nTrain, fPre, dt=dt, reg=reg)
        dPreD, XD, YD, errorD = decodeNoF(spikesInptD, targetsInptD, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateAttractors.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, XA, YA, errorA, "integrateAttractors", "preA", t*nTrain)
        plotState(times, XB, YB, errorB, "integrateAttractors", "preB", t*nTrain)
        plotState(times, XC, YC, errorC, "integrateAttractors", "preC", t*nTrain)
        plotState(times, XD, YD, errorD, "integrateAttractors", "preD", t*nTrain)

    if 0 in load:
        ePreDEns = np.load(file)['ePreDEns']
    else:
        print("encoders for preD to ens")
        ePreDEns = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                dPreD=dPreD,
                ePreDEns=ePreDEns,
                stage=0)
            ePreDEns = data['ePreDEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateAttractors", "preDEns")
            np.savez("data/integrateAttractors.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD,
                ePreDEns=ePreDEns)

    if 1 in load:
        ePreAFdfw = np.load(file)['ePreAFdfw']
        ePreBEns = np.load(file)['ePreBEns']
        ePreCOff = np.load(file)['ePreCOff']
    else:
        print("encoders for [preA, preB, preC] to [fdfw, ens, off]")
        ePreAFdfw = np.zeros((NPre, N, 1))
        ePreBEns = np.zeros((NPre, N, 1))
        ePreCOff = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, _ = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            stimB = stimA
            stimC = lambda t: 0 if t<c else 1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, ePreDEns=ePreDEns,
                stage=1)
            ePreAFdfw = data['ePreAFdfw']
            ePreBEns = data['ePreBEns']
            ePreCOff = data['ePreCOff']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateAttractors", "preAFdfw")
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateAttractors", "preBEns")
            plotActivity(t, dt, fS, data['times'], data['off'], data['tarOff'], "integrateAttractors", "preCOff")
            np.savez("data/integrateAttractors.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD,
                ePreDEns=ePreDEns, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff)

    if 2 in load:
        dFdfw = np.load(file)['dFdfw']
        dOff = np.load(file)['dOff']
    else:
        print('readout decoders for fdfw and off')
        spikesFdfw = np.zeros((nTrain, int(t/dt), N))
        spikesOff = np.zeros((nTrain, int(t/dt), N))
        targetsFdfw = np.zeros((nTrain, int(t/dt), 1))
        targetsOff = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            stimC = lambda t: 0 if t<c else 1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimC,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, ePreDEns=ePreDEns,
                stage=2)
            spikesFdfw[n] = data['fdfw']
            spikesOff[n] = data['off']
            targetsFdfw[n] = fNMDA.filt(Tff*fPre.filt(data['inptA']))
            # targetsFdfw[n] = fNMDA.filt(fPre.filt(data['inptB']))
            targetsOff[n] = fGABA.filt(fPre.filt(data['inptC']))
        dFdfw, X1, Y1, error1 = decodeNoF(spikesFdfw, targetsFdfw, nTrain, fNMDA, dt=dt, reg=reg)
        dOff, X2, Y2, error2 = decodeNoF(spikesOff, targetsOff, nTrain, fGABA, dt=dt, reg=reg)
        np.savez("data/integrateAttractors.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD, dFdfw=dFdfw, dOff=dOff,
            ePreDEns=ePreDEns, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1, Y1, error1, "integrateAttractors", "fdfw", t*nTrain)
        plotState(times, X2, Y2, error2, "integrateAttractors", "off", t*nTrain)
        
    if 3 in load:
        eFdfwEns = np.load(file)['eFdfwEns']
    else:
        print("encoders for fdfw-to-ens")
        eFdfwEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, Tff=Tff,
                stimA=stimA, stimB=stimB, 
                dPreA=dPreA, dPreB=dPreB, dPreD=dPreD, dFdfw=dFdfw,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreDEns=ePreDEns, eFdfwEns=eFdfwEns,
                stage=3)
            eFdfwEns = data['eFdfwEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateAttractors", "fdfwEns")
        np.savez("data/integrateAttractors.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD, dFdfw=dFdfw, dOff=dOff,
            ePreDEns=ePreDEns, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns)

    if 4 in load:
        dEns = np.load(file)['dEns']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreD=dPreD, dFdfw=dFdfw,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreDEns=ePreDEns, eFdfwEns=eFdfwEns,
                stage=4)
            spikes[n] = data['ens']
            # targets[n] = data['inptB']  # stimA rounded to nearest attractor
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))  # stimA rounded to nearest attractor
        dEns, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateAttractors.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD, dFdfw=dFdfw, dOff=dOff, dEns=dEns,
            ePreDEns=ePreDEns, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateAttractors", "ens", t*nTrain)

    if 5 in load:
        eEnsEns = np.load(file)['eEnsEns']
    else:
        print("encoders from ens to ens")
        eEnsEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, nAttr, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreD=dPreD, dFdfw=dFdfw, dEns=dEns,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreDEns=ePreDEns, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns,
                stage=5)
            eEnsEns = data['eEnsEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateAttractors", "ensEns")
            np.savez("data/integrateAttractors.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dPreD=dPreD, dFdfw=dFdfw, dOff=dOff, dEns=dEns,
                ePreDEns=ePreDEns, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns)

    print("testing")
    eOffFdfw = -1e2*np.ones((N, N, 1))
    vals = np.linspace(-1, 1, nTest)
    att = np.linspace(-1, 1, nAttr)
    for test in range(nTest):
        stimA = lambda t: vals[test] if t<c else 0
        stimB = lambda t: closestAttractor(vals[test], att)
        stimC = lambda t: 0 if t<c else 1
        DATest = lambda t: 0
        data = go(
            NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, stimC=stimC,
            dPreA=dPreA, dPreC=dPreC, dPreD=dPreD, dFdfw=dFdfw, dEns=dEns, dOff=dOff,
            ePreAFdfw=ePreAFdfw, ePreCOff=ePreCOff, ePreDEns=ePreDEns, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eOffFdfw=eOffFdfw,
            stage=9, c=c, DA=DATest)
        aFdfw = fNMDA.filt(data['fdfw'])
        aEns = fNMDA.filt(data['ens'])
        aOff = fGABA.filt(data['off'])
        xhatFdfw = np.dot(aFdfw, dFdfw)/Tff
        xhatEns = np.dot(aEns, dEns)
        xhatOff = np.dot(aOff, dOff)
        xFdfw = fNMDA.filt(fPre.filt(data['inptA']))
        # xEns = xFdfw[int(c/dt)] * np.ones_like(data['times'])
        xEns = fNMDA.filt(fPre.filt(data['inptB']))
        # xEns = data['inptB']
        errorEns = rmse(xhatEns[int(c/dt):], xEns[int(c/dt):])
        fig, ax = plt.subplots()
        ax.plot(data['times'], xFdfw, alpha=0.5, label="input (filtered)")
        ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        ax.plot(data['times'], xEns, label="target")
        ax.plot(data['times'], xhatEns, alpha=0.5, label="ens, rmse=%.3f"%errorEns)
        ax.plot(data['times'], xhatOff, alpha=0.5, label="off")
        ax.axvline(c, label="cutoff")
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateAttractors_testFlat%s.pdf"%test)

run(N=30, t=50, c=20, nTrain=2, nEnc=4, nAttr=4, nTest=7, load=[], file="data/integrateAttractors.npz")