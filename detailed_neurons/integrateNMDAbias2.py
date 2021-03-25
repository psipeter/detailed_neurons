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

def makeCarrier(t, fPre, fNMDA, dt=0.001, seed=0):
    stimWhite = nengo.processes.WhiteSignal(period=t/4, high=10, rms=2, seed=seed)
    stimSin = lambda x: 2*np.sin(4*2*np.pi*x*4/t)
    if seed % 2 == 0:
        stimRamp = lambda x: 4/t if (t/4)<x<(3*t/4) else -4/t
    else:        
        stimRamp = lambda x: -4/t if (t/4)<x<(3*t/4) else 4/t
    with nengo.Network() as model:
        uSignal = nengo.Node(stimWhite)
        # uSignal = nengo.Node(stimSin)
        uCarrier = nengo.Node(stimRamp)
        u = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        nengo.Connection(uSignal, u, synapse=None)
        nengo.Connection(uCarrier, u, synapse=None)
        pU = nengo.Probe(u, synapse=None)
        pX = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    plt.plot(fNMDA.filt(sim.data[pX]))
    plt.savefig('plots/testCarrier.pdf')
    funcU = lambda t: sim.data[pU][int(t/dt)]
    funcX = lambda t: sim.data[pX][int(t/dt)]
    return funcU, funcX


def go(NPre=300, NBias=100, N=30, t=10, seed=1, dt=0.001, Tff=0.3, tTrans=0.01,
        stage=None, alpha=3e-7, eMax=1e-1,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(1e-3, 1e-1),
        dPre=None, dFdfw=None, dEns=None, dBias=None,
        ePreFdfw=None, ePreEns=None, ePreBias=None, eFdfwEns=None, eBiasEns=None, eEnsEns=None, 
        stimA=lambda t: 0, stimB=lambda t: 0, stimC=lambda t: 0.1, DA=lambda t: 1):

    with nengo.Network(seed=seed) as model:
        inptA = nengo.Node(stimA)
        inptB = nengo.Node(stimB)
        inptC = nengo.Node(stimC)
        preA = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preB = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        preC = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
        fdfw = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed)
        ens = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)
        bias = nengo.Ensemble(NBias, 1, neuron_type=Bio("Interneuron", DA=DA), seed=seed+2)
        tarFdfw = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed)
        tarEns = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=seed+1)
        tarBias = nengo.Ensemble(NBias, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0), encoders=Choice([[1]]), seed=seed+2)
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
        pBias = nengo.Probe(bias.neurons, synapse=None)
        pTarBias = nengo.Probe(tarBias.neurons, synapse=None)
        if stage==0:  # readout decoders for [preA, preB, preC]
            pass
        if stage==1:  # encoders for [preA, preC] to [fdfw, bias]
            nengo.Connection(inptA, tarFdfw, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarBias, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            learnEncoders(c1, tarFdfw, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
            learnEncoders(c2, tarBias, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
        if stage==2:  # readout decoders for fdfw and bias
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
        if stage==3:  # encoders for [bias] to ens
            # nengo.Connection(inptC, tarBias, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarEns, synapse=fGABA, seed=seed)
            c1 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            learnEncoders(c2, tarEns, fS, alpha=1e3*alpha, eMax=1e3*eMax, tTrans=tTrans)
        if stage==4:  # encoders for [preB] to [ens]
            # nengo.Connection(inptC, tarBias, synapse=fPre, seed=seed)
            nengo.Connection(inptC, tarEns, synapse=fGABA, seed=seed)
            nengo.Connection(inptB, tarEns, synapse=fPre, seed=seed)
            c1 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            c3 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            learnEncoders(c3, tarEns, fS, alpha=10*alpha, eMax=10*eMax, tTrans=tTrans)
        if stage==5:  # encoders for [fdfw] to ens
            cB.synapse = fNMDA
            tarPreA = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            tarPreB = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            tarPreC = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(inptA, tarPreA, synapse=fPre, seed=seed)
            nengo.Connection(inptB, tarPreB, synapse=fNMDA, seed=seed)
            nengo.Connection(inptC, tarPreC, synapse=fPre, seed=seed)
            nengo.Connection(tarPreA, tarEns, synapse=fNMDA, transform=Tff, seed=seed)
            nengo.Connection(tarPreB, tarEns, synapse=fPre, seed=seed)
            nengo.Connection(tarPreC, tarEns, synapse=fGABA, seed=seed)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c3 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c4 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c5 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            learnEncoders(c4, tarEns, fS, alpha=3*alpha, eMax=3*eMax, tTrans=tTrans)
        if stage==6:  # readout decoders for ens
            cB.synapse = fNMDA
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c3 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c4 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c5 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
        if stage==7: # encoders from ens to ens
            preB2 = nengo.Ensemble(NPre, 1, max_rates=Uniform(30, 30), seed=seed)
            ens2 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)  # acts as preB input to ens
            ens3 = nengo.Ensemble(N, 1, neuron_type=Bio("Pyramidal", DA=DA), seed=seed+1)  # acts as tarEns
            nengo.Connection(inptB, preB2, synapse=fNMDA, seed=seed)
            pTarEns = nengo.Probe(ens3.neurons, synapse=None)
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preB, ens2, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c3 = nengo.Connection(preB2, ens3, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c4 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c5 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c6 = nengo.Connection(fdfw, ens3, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c7 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            c8 = nengo.Connection(bias, ens2, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            c9 = nengo.Connection(bias, ens3, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            c10 = nengo.Connection(ens2, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)
            learnEncoders(c10, ens3, fS, alpha=alpha, eMax=eMax, tTrans=tTrans)
        if stage==8:  # test
            c1 = nengo.Connection(preA, fdfw, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c2 = nengo.Connection(preC, bias, synapse=fPre, solver=NoSolver(dPre), seed=seed)
            c3 = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfw), seed=seed)
            c4 = nengo.Connection(bias, ens, synapse=GABA(), solver=NoSolver(dBias), seed=seed)
            c5 = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dEns), seed=seed)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if stage==0:
            pass
        if stage==1:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreBias)
        if stage==2:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreBias)
        if stage==3:
            setWeights(c1, dPre, ePreBias)
            setWeights(c2, dBias, eBiasEns)
        if stage==4:
            setWeights(c1, dPre, ePreBias)
            setWeights(c2, dBias, eBiasEns)
            setWeights(c3, dPre, ePreEns)
        if stage==5:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dPre, ePreBias)
            setWeights(c4, dFdfw, eFdfwEns)
            setWeights(c5, dBias, eBiasEns)
        if stage==6:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dPre, ePreBias)
            setWeights(c4, dFdfw, eFdfwEns)
            setWeights(c5, dBias, eBiasEns)
        if stage==7:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreEns)
            setWeights(c3, dPre, ePreEns)
            setWeights(c4, dPre, ePreBias)
            setWeights(c5, dFdfw, eFdfwEns)
            setWeights(c6, dFdfw, eFdfwEns)
            setWeights(c7, dBias, eBiasEns)
            setWeights(c8, dBias, eBiasEns)
            setWeights(c9, dBias, eBiasEns)
            setWeights(c10, dEns, eEnsEns)
        if stage==8:
            setWeights(c1, dPre, ePreFdfw)
            setWeights(c2, dPre, ePreBias)
            setWeights(c3, dFdfw, eFdfwEns)
            setWeights(c4, dBias, eBiasEns)
            setWeights(c5, dEns, eEnsEns)

        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 

    ePreFdfw = c1.e if stage==1 else ePreFdfw
    ePreBias = c2.e if stage==1 else ePreBias
    eBiasEns = c2.e if stage==3 else eBiasEns
    ePreEns = c3.e if stage==4 else ePreEns
    eFdfwEns = c4.e if stage==5 else eFdfwEns
    eEnsEns = c10.e if stage==7 else eEnsEns

    return dict(
        times=sim.trange(),
        inptA=sim.data[pInptA],
        inptB=sim.data[pInptB],
        inptC=sim.data[pInptC],
        preA=sim.data[pPreA],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        bias=sim.data[pBias],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        tarBias=sim.data[pTarBias],
        ePreFdfw=ePreFdfw,
        ePreEns=ePreEns,
        ePreBias=ePreBias,
        eFdfwEns=eFdfwEns,
        eBiasEns=eBiasEns,
        eEnsEns=eEnsEns,
    )


def run(NPre=100, NBias=100, N=30, t=10, nTrain=10, nTest=5, nEnc=10, dt=0.001,
        fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), fS=DoubleExp(2e-2, 1e-1),
        DATrain=lambda t: 0, DATest=lambda t: 0,
        Tff=0.3, reg=1e-2, load=[], file=None):

    if 0 in load:
        dPre = np.load(file)['dPre']
    else:
        print('readout decoders for [preA, preB, preC]')
        spikesInpt = np.zeros((nTrain, int(t/dt), NPre))
        targetsInpt = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                stage=0)
            spikesInpt[n] = data['preA']
            targetsInpt[n] = fPre.filt(data['inptA'], dt=dt)
        dPre, X, Y, error = decodeNoF(spikesInpt, targetsInpt, nTrain, fPre, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
        )
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDAbias2", "pre", t*nTrain)

    if 1 in load:
        ePreFdfw = np.load(file)['ePreFdfw']
        ePreBias = np.load(file)['ePreBias']
    else:
        print("encoders for [preA, preC] to [fdfw, bias]")
        ePreFdfw = np.zeros((NPre, N, 1))
        ePreBias = np.zeros((NPre, NBias, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre,
                ePreFdfw=ePreFdfw, ePreBias=ePreBias,
                stage=1)
            ePreFdfw = data['ePreFdfw']
            ePreBias = data['ePreBias']
            plotActivity(t, dt, fS, data['times'], data['fdfw'], data['tarFdfw'], "integrateNMDAbias2", "pre_fdfw")
            plotActivity(t, dt, fS, data['times'], data['bias'], data['tarBias'], "integrateNMDAbias2", "pre_bias")
            np.savez("data/integrateNMDAbias2.npz",
                dPre=dPre,
                ePreFdfw=ePreFdfw,
                ePreBias=ePreBias)

    if 2 in load:
        dFdfw = np.load(file)['dFdfw']
        dBias = np.load(file)['dBias']
    else:
        print('readout decoders for fdfw and bias')
        spikesFdfw = np.zeros((nTrain, int(t/dt), N))
        spikesBias = np.zeros((nTrain, int(t/dt), NBias))
        targetsFdfw = np.zeros((nTrain, int(t/dt), 1))
        targetsBias = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stimA, _ = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA,
                dPre=dPre,
                ePreFdfw=ePreFdfw, ePreBias=ePreBias,
                stage=2)
            spikesFdfw[n] = data['fdfw']
            spikesBias[n] = data['bias']
            targetsFdfw[n] = fNMDA.filt(Tff*fPre.filt(data['inptA'], dt=dt))
            targetsBias[n] = fGABA.filt(fPre.filt(data['inptC'], dt=dt))
        dFdfw, X1, Y1, error1 = decodeNoF(spikesFdfw, targetsFdfw, nTrain, fNMDA, dt=dt, reg=reg)
        dBias, X2, Y2, error2 = decodeNoF(spikesBias, targetsBias, nTrain, fGABA, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreBias=ePreBias,
            dFdfw=dFdfw,
            dBias=dBias)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X1, Y1, error1, "integrateNMDAbias2", "fdfw", t*nTrain)
        plotState(times, X2, Y2, error2, "integrateNMDAbias2", "bias", t*nTrain)


    if 3 in load:
        eBiasEns = np.load(file)['eBiasEns']
    else:
        print("encoders for [bias] to ens")
        eBiasEns = np.zeros((NBias, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA,
                dPre=dPre, dBias=dBias, 
                ePreBias=ePreBias, eBiasEns=eBiasEns,
                stage=3)
            eBiasEns = data['eBiasEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias2", "bias_ens")
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreBias=ePreBias,
            dFdfw=dFdfw,
            dBias=dBias,
            eBiasEns=eBiasEns,)

    if 4 in load:
        ePreEns = np.load(file)['ePreEns']
    else:
        print("encoders for [preB] to [ens]")
        ePreEns = np.zeros((NPre, N, 1))
        for n in range(nEnc):
            stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre, dBias=dBias,
                ePreBias=ePreBias, eBiasEns=eBiasEns, ePreEns=ePreEns,
                stage=4)
            ePreEns = data['ePreEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias2", "pre_ens")
            np.savez("data/integrateNMDAbias2.npz",
                dPre=dPre,
                ePreFdfw=ePreFdfw,
                ePreBias=ePreBias,
                dFdfw=dFdfw,
                dBias=dBias,
                eBiasEns=eBiasEns,
                ePreEns=ePreEns,)

    if 5 in load:
        eFdfwEns = np.load(file)['eFdfwEns']
    else:
        print("encoders for [fdfw] to ens")
        eFdfwEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            # stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimA, stimB = makeCarrier(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, Tff=Tff,
                dPre=dPre, dFdfw=dFdfw, dBias=dBias, 
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eBiasEns=eBiasEns,
                stage=5)
            eFdfwEns = data['eFdfwEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias2", "fdfw_ens")
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            ePreBias=ePreBias,
            dFdfw=dFdfw,
            dBias=dBias,
            eFdfwEns=eFdfwEns,
            eBiasEns=eBiasEns,)

    if 6 in load:
        dEns = np.load(file)['dEns']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            # stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimA, stimB = makeCarrier(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre, dFdfw=dFdfw, dBias=dBias, 
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eBiasEns=eBiasEns,
                stage=6)
            spikes[n] = data['ens']
            targets[n] = fNMDA.filt(fPre.filt(data['inptB']))
        dEns, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, dt=dt, reg=reg)
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            ePreBias=ePreBias,
            dFdfw=dFdfw,
            dBias=dBias,
            eFdfwEns=eFdfwEns,
            eBiasEns=eBiasEns,
            dEns=dEns)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "integrateNMDAbias2", "ens", t*nTrain)

    if 7 in load:
        eEnsEns = np.load(file)['eEnsEns']
    else:
        print("encoders from ens to ens")
        eEnsEns = np.zeros((N, N, 1))
        for n in range(nEnc):
            # stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=n)
            stimA, stimB = makeCarrier(t, fPre, fNMDA, dt=dt, seed=n)
            data = go(
                NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATrain,
                stimA=stimA, stimB=stimB,
                dPre=dPre, dFdfw=dFdfw, dBias=dBias, dEns=dEns,
                ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eBiasEns=eBiasEns, eEnsEns=eEnsEns,
                stage=7)
            eEnsEns = data['eEnsEns']
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "integrateNMDAbias2", "ens_ens")
        np.savez("data/integrateNMDAbias2.npz",
            dPre=dPre,
            ePreFdfw=ePreFdfw,
            ePreEns=ePreEns,
            ePreBias=ePreBias,
            dFdfw=dFdfw,
            dBias=dBias,
            eFdfwEns=eFdfwEns,
            eBiasEns=eBiasEns,
            dEns=dEns,
            eEnsEns=eEnsEns)


    print("testing")
    vals = np.linspace(-1, 1, nTest)
    for test in range(nTest):
        stimA, stimB = makeSignal(t, fPre, fNMDA, dt=dt, seed=200+test)
        data = go(
            NPre=NPre, NBias=NBias, N=N, t=t, dt=dt, DA=DATest,
            stimA=stimA, stimB=stimB, Tff=Tff,
            dPre=dPre,  dFdfw=dFdfw, dBias=dBias, dEns=dEns,
            ePreFdfw=ePreFdfw, ePreEns=ePreEns, eFdfwEns=eFdfwEns, eBiasEns=eBiasEns, eEnsEns=eEnsEns,
            stage=8)
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
        fig.savefig("plots/integrateNMDAbias2_testWhite%s.pdf"%test)
        plt.close('all')

run(NBias=100, N=30, t=10, nTrain=5, nEnc=5, nTest=5, DATrain=lambda t: 0, DATest=lambda t: 0,
    load=[0,1,2,3,4], file="data/integrateNMDAbias2.npz")