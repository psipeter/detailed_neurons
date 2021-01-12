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

def makeSignal(t, fPre, fNMDA, dt=0.001, value=1.0, norm='x', seed=0, c=None):
    if not c: c = t
    stim = nengo.processes.WhiteSignal(period=t/2, high=0.5, rms=0.5, seed=seed)
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


def go(NPre=300, N=100, t=10, c=None, seed=1, dt=0.001, Tff=0.3, tTrans=0.01,
        stage=None, alpha=3e-7, eMax=1e-1,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(1e-3, 1e-1),
        dPreA=None, dPreB=None, dPreC=None, dFdfw=None, dEns=None, dInh=None, dBias=None, dOff=None,
        ePreAFdfw=None, ePreBEns=None, ePreCOff=None, eFdfwEns=None, eEnsEns=None, eEnsInh=None, eBiasEns=None, eBiasInh=None, eInhFdfw=None, eOffFdfw=None,
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
        bias = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        fdfw = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        inh = nengo.Ensemble(N, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+2)
        off = nengo.Ensemble(N, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+3)
        tarFdfw = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed+1)
        tarInh = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed+2)
        tarOff = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(0.2, 0.8), encoders=Choice([[1]]), seed=seed+3)
        cA = nengo.Connection(inptA, preA, synapse=None, seed=seed)
        cB = nengo.Connection(inptB, preB, synapse=None, seed=seed)
        cC = nengo.Connection(inptC, preC, synapse=None, seed=seed)
        cD = nengo.Connection(inptD, bias, synapse=None, seed=seed)
        pInptA = nengo.Probe(inptA, synapse=None)
        pInptB = nengo.Probe(inptB, synapse=None)
        pInptC = nengo.Probe(inptC, synapse=None)
        pInptD = nengo.Probe(inptD, synapse=None)
        pPreA = nengo.Probe(preA.neurons, synapse=None)
        pPreB = nengo.Probe(preB.neurons, synapse=None)
        pPreC = nengo.Probe(preC.neurons, synapse=None)
        pBias = nengo.Probe(bias.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pTarInh = nengo.Probe(tarInh.neurons, synapse=None)
        pOff = nengo.Probe(off.neurons, synapse=None)
        pTarOff = nengo.Probe(tarOff.neurons, synapse=None)
        if stage==0:
            nengo.Connection(bias, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(bias, tarInh, synapse=fPre, seed=seed)
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            learnEncoders(c0a, tarEns, fS, alpha=10*alpha, eMax=10*eMax, tTrans=tTrans)
            learnEncoders(c0b, tarInh, fS, alpha=alpha/10, eMax=eMax/10, tTrans=tTrans)
        if stage==1:
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarOff, synapse=fPre, seed=seed)
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
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
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            learnEncoders(c3, tarEns, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
        if stage==4:
            cB.synapse = fNMDA
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
        if stage==5:
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            ens2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            ens3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
            nengo.Connection(inptB, preB2, synapse=fNMDA, seed=seed)
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, ens2, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0c = nengo.Connection(bias, ens3, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(preB, ens2, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c4 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)
            c5 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c6 = nengo.Connection(preB2, ens3, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            learnEncoders(c4, ens3, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)

        if stage==6:  # encoders ens to inh
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(tarEns, tarInh, synapse=fNMDA, transform=-1, seed=seed)
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c2 = nengo.Connection(ens, inh, synapse=NMDA(), solver=NoSolver(-dEns), seed=seed)
            learnEncoders(c2, tarInh, fS, alpha=alpha/10, eMax=eMax/10, tTrans=tTrans)

        if stage==7:  # inh decoders
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c2 = nengo.Connection(ens, inh, synapse=NMDA(), solver=NoSolver(-dEns), seed=seed)

        if stage==8:  # encoders inh to fdfw
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            fdfw2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
            tarFdfw2 = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed)
            alt = nengo.Connection(inptA, preB2, synapse=fNMDA, seed=seed)
            
            t0a = nengo.Connection(bias, tarEns, synapse=fPre, seed=seed)  # solver=NoSolver(dBias),
            t0b = nengo.Connection(bias, tarInh, synapse=fPre, seed=seed)  # solver=NoSolver(dBias),
            t1a = nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            t1b = nengo.Connection(inptA, tarFdfw2, synapse=fPre, seed=seed)
            t2 = nengo.Connection(tarFdfw, tarEns, synapse=fPre, seed=seed)
            t3 = nengo.Connection(tarEns, tarInh, synapse=fNMDA, transform=-1, seed=seed)
            t4 = nengo.Connection(tarInh, tarFdfw2, synapse=fGABA, seed=seed)

            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1a = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c1b = nengo.Connection(preA, fdfw2, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            # c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            # must avoid Tff multiplication of fdfw to ens
            c2 = nengo.Connection(preB2, ens, synapse=fPre, solver=NoSolver(dPreB), seed=seed)
            c3 = nengo.Connection(ens, inh, synapse=NMDA(), solver=NoSolver(-dEns), seed=seed)
            c4 = nengo.Connection(inh, fdfw2, synapse=GABA(), solver=NoSolver(dInh), seed=seed)
            # learnEncoders(c4, tarFdfw2, fS, alpha=1e3*alpha, eMax=1e3*eMax, tTrans=tTrans)
            learnEncoders(c4, tarFdfw2, fS, alpha=1e3*alpha, eMax=1e3*eMax, tTrans=tTrans)
            pFdfw2 = nengo.Probe(fdfw2.neurons, synapse=None)
            pTarFdfw2 = nengo.Probe(tarFdfw2.neurons, synapse=None)
            pTarFdfwState = nengo.Probe(tarFdfw, synapse=fNMDA)
            pTarFdfw2State = nengo.Probe(tarFdfw2, synapse=fNMDA)

        if stage==9:
            c0a = nengo.Connection(bias, ens, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c0b = nengo.Connection(bias, inh, synapse=fPre, solver=NoSolver(dBias), seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPreA), seed=seed)
            c2 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c3 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)
            c4 = nengo.Connection(ens, inh, synapse=NMDA(), solver=NoSolver(-dEns), seed=seed)
            c5 = nengo.Connection(inh, fdfw, synapse=GABA(), solver=NoSolver(dInh), seed=seed)
            c6 = nengo.Connection(preC, off, synapse=fPre, solver=NoSolver(dPreC), seed=seed)
            c7 = nengo.Connection(off, fdfw, synapse=GABA(), solver=NoSolver(dOff), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==0:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
        if stage==1:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dPreC, ePreCOff)
        if stage==2:
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreC, ePreCOff)
        if stage==3:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==4:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, dFdfw, eFdfwEns)
        if stage==5:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasEns)
            setWeights(c0c, dBias, eBiasEns)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c3, dPreB, ePreBEns)
            setWeights(c4, dEns, eEnsEns)
            setWeights(c5, dFdfw, eFdfwEns)
            setWeights(c6, dPreB, ePreBEns)
        if stage==6:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
            setWeights(c1, dPreB, ePreBEns)
            setWeights(c2, -dEns, eEnsInh)
        if stage==7:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
            setWeights(c1, dPreB, ePreBEns)
            setWeights(c2, -dEns, eEnsInh)
        if stage==8:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
            setWeights(c1a, dPreA, ePreAFdfw)
            setWeights(c1b, dPreA, ePreAFdfw)
            # setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c2, dPreB, ePreBEns)
            setWeights(c3, -dEns, eEnsInh)
            setWeights(c4, dInh, eInhFdfw)
        if stage==9:
            setWeights(c0a, dBias, eBiasEns)
            setWeights(c0b, dBias, eBiasInh)
            setWeights(c1, dPreA, ePreAFdfw)
            setWeights(c2, dFdfw, eFdfwEns)
            setWeights(c3, dEns, eEnsEns)
            setWeights(c4, -dEns, eEnsInh)
            setWeights(c5, dInh, eInhFdfw)
            setWeights(c6, dPreC, ePreCOff)
            setWeights(c7, dOff, eOffFdfw)

        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    eBiasEns = c0a.e if stage==0 else eBiasEns
    eBiasInh = c0b.e if stage==0 else eBiasInh
    ePreAFdfw = c1.e if stage==1 else ePreAFdfw
    ePreBEns = c2.e if stage==1 else ePreBEns
    ePreCOff = c3.e if stage==1 else ePreCOff
    eFdfwEns = c3.e if stage==3 else eFdfwEns
    eEnsEns = c4.e if stage==5 else eEnsEns
    eEnsInh = c2.e if stage==6 else eEnsInh
    eInhFdfw = c4.e if stage==8 else eInhFdfw

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        inptC=sim.data[pInptC],
        inptD=sim.data[pInptD],
        preA=sim.data[pPreA],
        preB=sim.data[pPreB],
        preC=sim.data[pPreC],
        bias=sim.data[pBias],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        inh=sim.data[pInh],
        off=sim.data[pOff],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        tarInh=sim.data[pTarInh],
        tarOff=sim.data[pTarOff],
        ePreAFdfw=ePreAFdfw,
        ePreBEns=ePreBEns,
        ePreCOff=ePreCOff,
        eBiasEns=eBiasEns,
        eBiasInh=eBiasInh,
        eFdfwEns=eFdfwEns,
        eEnsEns=eEnsEns,
        eEnsInh=eEnsInh,
        eInhFdfw=eInhFdfw,
        eOffFdfw=eOffFdfw,
        fdfw2 = sim.data[pFdfw2] if stage==8 else None,
        tarFdfw2 = sim.data[pTarFdfw2] if stage==8 else None,
        tarFdfwState = sim.data[pTarFdfwState] if stage==8 else None,
        tarFdfw2State = sim.data[pTarFdfw2State] if stage==8 else None,
    )


def run(NPre=100, N=100, t=10, nTrain=10, nTest=5, nEnc=10, dt=0.001, c=None,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(2e-2, 1e-1),
        Tff=0.3, reg=1e-2, load=[], file=None):

    if not c: c = t
    if 0 in load:
        dPreA = np.load(file)['dPreA']
        dPreB = np.load(file)['dPreB']
        dPreC = np.load(file)['dPreC']
        dBias = np.load(file)['dBias']
    else:
        print('readout decoders for [preA, preB, off, bias]')
        spikesInptA = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptB = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptC = np.zeros((nTrain, int(t/dt), NPre))
        spikesInptD = np.zeros((nTrain, int(t/dt), NPre))
        targetsInptA = np.zeros((nTrain, int(t/dt), 1))
        targetsInptB = np.zeros((nTrain, int(t/dt), 1))
        targetsInptC = np.zeros((nTrain, int(t/dt), 1))
        targetsInptD = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimC = lambda t: 0 if t<c else 1
            stimD = lambda t: 1e-1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimC, stimD=stimD,
                stage=None)
            spikesInptA[n] = data['preA']
            spikesInptB[n] = data['preB']
            spikesInptC[n] = data['preC']
            spikesInptD[n] = data['bias']
            targetsInptA[n] = fPre.filt(data['inptA'], dt=dt)
            targetsInptB[n] = fPre.filt(data['inptB'], dt=dt)
            targetsInptC[n] = fPre.filt(data['inptC'], dt=dt)
            targetsInptD[n] = fPre.filt(data['inptD'], dt=dt)
        dPreA, XA, YA, errorA = decodeNoF(spikesInptA, targetsInptA, nTrain, fPre, dt=dt, reg=reg)
        dPreB, XB, YB, errorB = decodeNoF(spikesInptB, targetsInptB, nTrain, fPre, dt=dt, reg=reg)
        dPreC, XC, YC, errorC = decodeNoF(spikesInptC, targetsInptC, nTrain, fPre, dt=dt, reg=reg)
        dBias, XD, YD, errorD = decodeNoF(spikesInptD, targetsInptD, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, XA, YA, errorA, "integrateNMDA", "preA", t*nTrain)
        plotState(times, XB, YB, errorB, "integrateNMDA", "preB", t*nTrain)
        plotState(times, XC, YC, errorC, "integrateNMDA", "preC", t*nTrain)
        plotState(times, XD, YD, errorD, "integrateNMDA", "bias", t*nTrain)

    if 0 in load:
        eBiasEns = np.load(file)['eBiasEns']
        eBiasInh = np.load(file)['eBiasInh']
    else:
        print("encoders for bias to [ens, inh]")
        eBiasEns = np.zeros((NPre, N, 1))
        eBiasInh = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            # signals beyond zero?
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                dBias=dBias,
                eBiasEns=eBiasEns, eBiasInh=eBiasInh,
                stage=0)
            eBiasEns = data['eBiasEns']
            eBiasInh = data['eBiasInh']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "biasEns")
            plotActivity(t, dt, fS, data['times'], data['inh'], data['tarInh'], "integrateNMDA", "biasInh")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias,
                eBiasEns=eBiasEns, eBiasInh=eBiasInh)

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
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimC = lambda t: 0 if t<c else 1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, 
                stage=1)
            ePreAFdfw = data['ePreAFdfw']
            ePreBEns = data['ePreBEns']
            ePreCOff = data['ePreCOff']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDA", "preAFdfw")
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "preBEns")
            plotActivity(t, dt, fS, data['times'], data['off'], data['tarOff'], "integrateNMDA", "preCOff")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias,
                eBiasEns=eBiasEns, eBiasInh=eBiasInh, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff)

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
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimC = lambda t: 0 if t<c else 1
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, stimC=stimC,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eBiasEns=eBiasEns,
                stage=2)
            spikesFdfw[n] = data['fdfw']
            spikesOff[n] = data['off']
            targetsFdfw[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
            targetsOff[n] = fGABA.filt(fPre.filt(data['inptC']))
        dFdfw, X1, Y1, error1 = decodeNoF(spikesFdfw, targetsFdfw, nTrain, fNMDA, dt=dt, reg=reg)
        dOff, X2, Y2, error2 = decodeNoF(spikesOff, targetsOff, nTrain, fGABA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff,
            eBiasEns=eBiasEns, eBiasInh=eBiasInh, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1, Y1, error1, "integrateNMDA", "fdfw", t*nTrain)
        plotState(times, X2, Y2, error2, "integrateNMDA", "off", t*nTrain)
        
    if 3 in load:
        eFdfwEns = np.load(file)['eFdfwEns']
    else:
        print("encoders for fdfw-to-ens")
        eFdfwEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB, Tff=Tff,
                dPreA=dPreA, dPreB=dPreB, dBias=dBias, dFdfw=dFdfw,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eFdfwEns=eFdfwEns,
                stage=3)
            eFdfwEns = data['eFdfwEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "fdfwEns")
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff,
            eBiasEns=eBiasEns, eBiasInh=eBiasInh, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns)

    if 4 in load:
        dEns = np.load(file)['dEns']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dBias=dBias, dFdfw=dFdfw,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eFdfwEns=eFdfwEns,
                stage=4)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
        dEns, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff, dEns=dEns,
            eBiasEns=eBiasEns, eBiasInh=eBiasInh, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDA", "ens", t*nTrain)

    if 5 in load:
        eEnsEns = np.load(file)['eEnsEns']
    else:
        print("encoders from ens to ens")
        eEnsEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dBias=dBias, dFdfw=dFdfw, dEns=dEns,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns,
                stage=5)
            eEnsEns = data['eEnsEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "ensEns")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff, dEns=dEns,
                eBiasEns=eBiasEns, eBiasInh=eBiasInh, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns)

    if 6 in load:
        eEnsInh = np.load(file)['eEnsInh']
    else:
        print("encoders from ens to inh")
        eEnsInh = np.zeros((N, N, 1))
        for n in range(nEnc):
            _, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dEns=dEns,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh,
                stage=6)
            eEnsInh = data['eEnsInh']
            plotActivity(t, dt, fS, data['times'], data['inh'], data['tarInh'], "integrateNMDA", "ensInh")
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff, dEns=dEns, 
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh)

    if 7 in load:
        dInh = np.load(file)['dInh']
    else:
        print('readout decoders for inh')
        spikesInh = np.zeros((nTrain, int(t/dt), N))
        targetsInh = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt,
                stimA=stimA, stimB=stimB,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dEns=dEns,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh,
                stage=7)
            spikesInh[n] = data['inh']
            targetsInh[n] = -fGABA.filt(fNMDA.filt(fPre.filt(data['inptB'])))
        dInh, X2, Y2, error2 = decodeNoF(spikesInh, targetsInh, nTrain, fGABA, dt=dt, reg=reg)
        np.savez("data/integrateNMDA.npz",
            dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff, dEns=dEns, dInh=dInh,
            ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X2, Y2, error2, "integrateNMDA", "inh", t*nTrain)


    if 8 in load:
        eInhFdfw = np.load(file)['eInhFdfw']
        eOffFdfw = np.load(file)['eOffFdfw']
    else:
        print("encoders from inh to fdfw")
        eInhFdfw = np.zeros((N, N, 1))
        eOffFdfw = -1e2*np.ones((N, N, 1))
        for n in range(nEnc):
            # use -1 to 1 signal here
            _, stimA = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, N=N, t=t, dt=dt, stimA=stimA,
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dEns=dEns, dInh=dInh,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh, eInhFdfw=eInhFdfw,
                stage=8)
            eInhFdfw = data['eInhFdfw']
            # plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDA", "ensTest")                
            # plotActivity(t, dt, fS, data['times'], data['inh'], data['tarInh'], "integrateNMDA", "inhTest")                
            plotActivity(t, dt, fS, data['times'], data['fdfw2'], data['tarFdfw2'], "integrateNMDA", "inhFdfw")                
            # plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDA", "inhFdfwBaseline")                
            np.savez("data/integrateNMDA.npz",
                dPreA=dPreA, dPreB=dPreB, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dOff=dOff, dEns=dEns, dInh=dInh,
                ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns, ePreCOff=ePreCOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh, eInhFdfw=eInhFdfw, eOffFdfw=eOffFdfw)


    print("testing")
    vals = np.linspace(-1, 1, nTest)
    for test in range(nTest):
        stimA = lambda t: vals[test] if t<c else 0
        stimC = lambda t: 0 if t<c else 1
        DATest = lambda t: 0
        data = go(
            NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimC=stimC,
            dPreA=dPreA, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dEns=dEns, dInh=dInh, dOff=dOff,
            ePreAFdfw=ePreAFdfw, ePreCOff=ePreCOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh, eInhFdfw=eInhFdfw, eOffFdfw=eOffFdfw,
            stage=9, c=c, DA=DATest)
        aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        aBio = fNMDA.filt(data['ens'], dt=dt)
        aInh = fGABA.filt(data['inh'], dt=dt)
        aOff = fGABA.filt(data['off'], dt=dt)
        xhatFdfw = np.dot(aFdfw, dFdfw)/Tff
        xhatEns = np.dot(aBio, dEns)
        xhatInh = np.dot(aInh, dInh)
        xhatOff = np.dot(aOff, dOff)
        xFdfw = fNMDA.filt(fNMDA.filt(fPre.filt(data['inptA'], dt=dt), dt=dt), dt=dt)
        xEns = xFdfw[int(c/dt)] * np.ones_like(data['times'])
        errorBio = rmse(xhatEns[int(c/dt):], xEns[int(c/dt):])
        fig, ax = plt.subplots()
        ax.plot(data['times'], xFdfw, alpha=0.5, label="input (filtered)")
        ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        ax.plot(data['times'], xEns, label="target")
        ax.plot(data['times'], xhatEns, alpha=0.5, label="ens, rmse=%.3f"%errorBio)
        ax.plot(data['times'], xhatInh, alpha=0.5, label="inh")
        ax.plot(data['times'], xhatOff, alpha=0.5, label="off")
        ax.axvline(c, label="cutoff")
        ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/integrateNMDA_testFlat%s.pdf"%test)

        # stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=200+test)
        # stimC = lambda t: 0
        # data = go(NPre=NPre, N=N, t=t, dt=dt, stimA=stimA, stimB=stimB, stimC=stimC, Tff=Tff,
        #     dPreA=dPreA, dPreC=dPreC, dBias=dBias, dFdfw=dFdfw, dEns=dEns, dInh=dInh,
        #     ePreAFdfw=ePreAFdfw, eOff=eOff, eBiasEns=eBiasEns, eBiasInh=eBiasInh, eFdfwEns=eFdfwEns, eEnsEns=eEnsEns, eEnsInh=eEnsInh, eInhFdfw=None,
        #     stage=9, DA=DATest)
        # aFdfw = fNMDA.filt(data['fdfw'], dt=dt)
        # aBio = fNMDA.filt(data['ens'], dt=dt)
        # xhatFdfw = np.dot(aFdfw, dFdfw)
        # xhatEns = np.dot(aBio, dEns)
        # xFdfw = fNMDA.filt(fPre.filt(data['inptA'], dt=dt), dt=dt)
        # xEns = fNMDA.filt(fPre.filt(data['inptB'], dt=dt), dt=dt)
        # errorBio = rmse(xhatEns, xEns)
        # fig, ax = plt.subplots()
        # ax.plot(data['times'], xhatFdfw, alpha=0.5, label="fdfw")
        # ax.plot(data['times'], Tff*xFdfw, alpha=0.5, label="input")
        # ax.plot(data['times'], xEns,  label="target")
        # ax.plot(data['times'], xhatEns, alpha=0.5, label="ens, rmse=%.3f"%errorBio)
        # ax.set(xlabel='time', ylabel='state', xlim=((0, t)), ylim=((-1.5, 1.5)))
        # ax.legend(loc='upper left')
        # sns.despine()
        # fig.savefig("plots/integrateNMDA_testWhite%s.pdf"%test)
        # plt.close('all')

    # np.savez("data/integrateNMDA.npz", dPreA=dPreA, dPreB=dPreB, ePreAFdfw=ePreAFdfw, ePreBEns=ePreBEns,
        # dFdfw=dFdfw, eFdfwEns=eFdfwEns, dEns=dEns, dNeg=dNeg, eEnsEns=eEnsEns, eNeg=eNeg, eEnsInh=eEnsInh)

run(N=30, t=20, c=10, nTrain=3, nEnc=10, nTest=5, load=[0,1,2,3,4,5,6,7], file="data/integrateNMDA.npz")