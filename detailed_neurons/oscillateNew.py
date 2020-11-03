import numpy as np
from scipy.optimize import curve_fit
import nengo
from nengo import SpikingRectifiedLinear
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import LinearSystem
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity, fitSinusoid
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def go(NPre=100, N=30, t=10, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), neuron_type=LIF(), d1=None, d2=None, f1=None, f2=None, e1=None, e2=None, l1=False, l2=False, test=False, freq=1, phase=0, tDrive=0.2):

    A = [[1, 1e-1*2*np.pi*freq], [-1e-1*2*np.pi*freq, 1]]  # tau*A + I
    if isinstance(neuron_type, Bio) and not f1: f1=DoubleExp(1e-3, 1e-1)
    if isinstance(neuron_type, Bio) and not f2: f2=DoubleExp(1e-3, 1e-1)
    stim = lambda t: [np.sin(2*np.pi*freq*t+phase), np.cos(2*np.pi*freq*t+phase)]

    with nengo.Network(seed=seed) as model:          
        inpt = nengo.Node(stim)
        tar = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(NPre, 2, max_rates=m, neuron_type=nengo.SpikingRectifiedLinear(), radius=2, seed=seed)
        ens = nengo.Ensemble(N, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, radius=2, seed=seed)
        nengo.Connection(inpt, tar, synapse=None, transform=A, seed=seed)
        nengo.Connection(inpt, pre, synapse=None, seed=seed)
        c1 = nengo.Connection(pre, ens, synapse=f1, seed=seed, solver=NoSolver(d1))
        pInpt = nengo.Probe(inpt, synapse=None)
        pTar = nengo.Probe(tar, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        # Encoder Learning (Bio)
        if l1:
            tarEns = nengo.Ensemble(N, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            nengo.Connection(inpt, tarEns, synapse=None, seed=seed)
            learnEncoders(c1, tarEns, fS)
            pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        if l2:
            pre2 = nengo.Ensemble(NPre, 2, max_rates=m, neuron_type=nengo.LIF(), seed=seed, radius=2)
            tarEns2 = nengo.Ensemble(N, 2, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
            ens2 = nengo.Ensemble(N, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2)
            
#             ens3 = nengo.Ensemble(N, 2, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed, radius=2)
#             nengo.Connection(tar, pre2, synapse=f)
#             c3 = nengo.Connection(ens, ens2, synapse=f2, seed=seed)
#             c4 = nengo.Connection(pre2, ens3, synapse=f1, seed=seed)
#             learnEncoders(c3, ens3, fS)
#             pTarEns2 = nengo.Probe(ens3.neurons, synapse=None)
#             pEns2 = nengo.Probe(ens2.neurons, synapse=None)

            nengo.Connection(inpt, pre2, synapse=f)
            nengo.Connection(pre2, tarEns2, synapse=f, seed=seed)
            c3 = nengo.Connection(ens, ens2, synapse=f2, seed=seed)
            learnEncoders(c3, tarEns2, fS, alpha=3e-7)
            pTarEns2 = nengo.Probe(tarEns2.neurons, synapse=None)
            pEns2 = nengo.Probe(ens2.neurons, synapse=None)
        if test:
            c2 = nengo.Connection(ens, ens, synapse=f2, seed=seed, solver=NoSolver(d2))
            off = nengo.Node(lambda t: 1 if t>tDrive else 0)
            nengo.Connection(off, pre.neurons, synapse=None, transform=-1e4*np.ones((NPre, 1)))

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        if isinstance(neuron_type, Bio):
            setWeights(c1, d1, e1)
            if l2: setWeights(c3, d2, e2)
#             if l2: setWeights(c4, d1, e1)
            if test: setWeights(c2, d2, e2)
            neuron.h.init()
            sim.run(t, progress_bar=True)
            reset_neuron(sim, model) 
        else:
            sim.run(t, progress_bar=True)
      
    e1 = c1.e if l1 else e1
    e2 = c3.e if l2 else e2

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        tar=sim.data[pTar],
        pre=sim.data[pPre],
        ens=sim.data[pEns],
        tarEns=sim.data[pTarEns] if l1 else None,
        tarEns2=sim.data[pTarEns2] if l2 else None,
        ens2=sim.data[pEns2] if l2 else None,
        e1=e1,
        e2=e2,
    )


def run(NPre=300, N=100, t=20, tTrans=2, nTrain=1, nEnc=10, nTest=10, neuron_type=LIF(),
        dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), freq=1, muFreq=1.0, sigmaFreq=0.1, reg=1e-1, tauRiseMax=5e-2, tDrive=0.2, base=False, load=False, file=None):

    print('\nNeuron Type: %s'%neuron_type)
    rng = np.random.RandomState(seed=0)
    if load:
        d1 = np.load(file)['d1']
        tauRise1 = np.load(file)['tauRise1']
        tauFall1 = np.load(file)['tauFall1']
        f1 = DoubleExp(tauRise1, tauFall1)
    else:
        print('readout decoders for pre')  
        spikes = np.zeros((nTrain, int(t/0.001), NPre))
        targets = np.zeros((nTrain, int(t/0.001), 2))
        for n in range(nTrain):
            data = go(NPre=NPre, N=N, t=t, dt=0.001, f=f, fS=fS, neuron_type=LIF(), freq=freq, phase=2*np.pi*(n/nTrain))
            spikes[n] = data['pre']
            targets[n] = f.filt(data['inpt'], dt=0.001)
        d1, f1, tauRise1, tauFall1, X, Y, error = decode(spikes, targets, nTrain, dt=0.001, tauRiseMax=tauRiseMax, name="oscillateNew")
        np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X, Y, error, "oscillateNew", "%s_pre"%neuron_type, t*nTrain)

    if load:
        e1 = np.load(file)['e1']
    elif isinstance(neuron_type, Bio):
        print("ens1 encoders")
        e1 = np.zeros((NPre, N, 2))
        for n in range(nEnc):
            data = go(d1=d1, e1=e1, f1=f1, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, freq=freq, phase=2*np.pi*(n/nEnc), l1=True)
            e1 = data['e1']
            np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1)
            plotActivity(t, dt, fS, data['times'], data['ens'], data['tarEns'], "oscillateNew", "ens")
    else:
        e1 = np.zeros((NPre, N, 2))
        
    if load:
        d2 = np.load(file)['d2']
        tauRise2 = np.load(file)['tauRise2']
        tauFall2 = np.load(file)['tauFall2']
        f2 = DoubleExp(tauRise2, tauFall2)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int((t-tTrans)/dt), N))
        targets = np.zeros((nTrain, int((t-tTrans)/dt), 2))
        for n in range(nTrain):
            data = go(d1=d1, e1=e1, f1=f1, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, freq=freq, phase=2*np.pi*(n/nTrain))
            spikes[n] = data['ens'][int(tTrans/dt):]
            targets[n] = f.filt(f.filt(data['tar'], dt=dt), dt=dt)[int(tTrans/dt):]
        d2, f2, tauRise2, tauFall2, X, Y, error = decode(spikes, targets, nTrain, dt=dt, name="oscillateNew", reg=reg, tauRiseMax=tauRiseMax)
        np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2)
        times = np.arange(0, t*nTrain, dt)[:len(X)]
        plotState(times, X, Y, error, "oscillateNew", "%s_ens"%neuron_type, (t-tTrans)*nTrain)

    if load:
        e2 = np.load(file)['e2']
    #elif isinstance(neuron_type, Bio):
        print("ens2 encoders")
        #e2 = np.zeros((N, N, 2))
        for n in range(nEnc):
            data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, NPre=NPre, N=N, t=t, dt=dt, f=f, fS=fS, neuron_type=neuron_type, freq=freq, phase=2*np.pi*(n/nEnc), l2=True)
            e2 = data['e2']
            np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2)
            plotActivity(t, dt, fS, data['times'], data['ens2'], data['tarEns2'], "oscillateNew", "ens2")
    else:
        e2 = np.zeros((N, N, 2))
        np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2)

    print("testing")
    errors = np.zeros((nTest))
    for test in range(nTest):
        data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, NPre=NPre, N=N, t=t+tTrans, dt=dt, f=f, fS=fS, neuron_type=neuron_type, freq=freq, phase=2*np.pi*(test/nTest), tDrive=tDrive, test=True)
        # curve fit to a sinusoid of arbitrary frequency, phase, magnitude
        times = data['times']
        A = f2.filt(data['ens'], dt=dt)
        X = np.dot(A, d2)
        freq0, phase0, mag0, base0 = fitSinusoid(times, X[:,0], freq, int(tTrans/dt), muFreq=muFreq, sigmaFreq=sigmaFreq, base=base)
        freq1, phase1, mag1, base1 = fitSinusoid(times, X[:,1], freq, int(tTrans/dt), muFreq=muFreq, sigmaFreq=sigmaFreq, base=base)
        s0 = base0+mag0*np.sin(times*2*np.pi*freq0+phase0)
        s1 = base1+mag1*np.sin(times*2*np.pi*freq1+phase1)
#         freqError0 = np.abs(freq-freq0)
#         freqError1 = np.abs(freq-freq1)
        rmseError0 = rmse(X[int(tTrans/dt):,0], s0[int(tTrans/dt):])
        rmseError1 = rmse(X[int(tTrans/dt):,1], s1[int(tTrans/dt):])
#         error0 = (1+freqError0) * rmseError0
#         error1 = (1+freqError1) * rmseError1
        error0 = rmseError0
        error1 = rmseError1
        errors[test] = (error0 + error1)/2
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax.plot(times, X[:,0], label="estimate (dim 0)")
        ax.plot(times, s0, label="target (dim 0)")
        ax.set(ylabel='state', title='freq=%.3f, rmse=%.3f'%(freq0, error0), xlim=((0, t)), ylim=((-1.2, 1.2)))
        ax.legend(loc='upper left')
        ax2.plot(times, X[:,1], label="estimate (dim 1)")
        ax2.plot(times, s1, label="target (dim 1)")
        ax2.set(xlabel='time', ylabel='state', title='freq=%.3f, rmse=%.3f'%(freq1, error1), xlim=((0, t)), ylim=((-1.2, 1.2)))
        ax2.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/oscillateNew_%s_test%s.pdf"%(neuron_type, test))
    plt.close('all')
    print('%s errors:'%neuron_type, errors)
    np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, errors=errors)
    return errors

# errorsLIF = run(neuron_type=LIF(), load=False, file="data/oscillateNew_LIF().npz")
# errorsALIF = run(neuron_type=ALIF(), load=False, file="data/oscillateNew_ALIF().npz")
# errorsWilson = run(neuron_type=Wilson(), dt=1e-4, load=False, file="data/oscillateNew_Wilson().npz")
errorsBio = run(neuron_type=Bio("Pyramidal"), load=True, muFreq=4, base=True, tTrans=5.0, file="data/oscillateNew_Bio().npz")
