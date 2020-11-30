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

def makeSignal(t, fPre, fNMDA, dt=0.001, value=1.0, freq=1, seed=0, c=None):
    if not c: c = t
    stim = nengo.processes.WhiteSignal(period=t/2, high=freq, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
        pX = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(t/2, progress_bar=False)
    u = fNMDA.filt(fPre.filt(sim.data[pU]))
    x = fNMDA.filt(fPre.filt(sim.data[pX]))
    norm = value / np.max(np.abs(u))
    # norm = value / np.max(np.abs(x))
    mirrored = np.concatenate(([[0]], sim.data[pU]*norm, -sim.data[pU]*norm))
    func = lambda t: mirrored[int(t/dt)] if t<c else 0
    return func

# def ideal(N=100, t=10, c=7, f=DoubleExp(10e-3, 2.85e-1)):
#     stim = nengo.processes.WhiteSignal(period=t, high=0.5, rms=0.3, seed=0)
#     cutoff = lambda t: 0 if t<c else 1
#     with nengo.Network() as network:
#         inpt = nengo.Node(stim)
#         cut = nengo.Node(cutoff)
#         diff = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=0)
#         ens = nengo.Ensemble(N, 1, max_rates=Uniform(30, 30), intercepts=Uniform(-0.8, 0.8), seed=0)
#         nengo.Connection(inpt, diff, synapse=None)
#         nengo.Connection(cut, diff.neurons, synapse=None, transform=-1e2*np.ones((N, 1)))
#         nengo.Connection(diff, ens, synapse=f)
#         nengo.Connection(ens, ens, synapse=f)
#         nengo.Connection(ens, diff, synapse=f, transform=-1)
#         pInpt = nengo.Probe(inpt, synapse=f)
#         pDiff = nengo.Probe(diff, synapse=f)
#         pEns = nengo.Probe(ens, synapse=f)
#     with nengo.Simulator(network, progress_bar=False) as sim:
#         sim.run(t, progress_bar=True)
#     fig, ax = plt.subplots()
#     ax.plot(sim.trange(), sim.data[pInpt], label='inpt')
#     ax.plot(sim.trange(), sim.data[pDiff], label='diff')
#     ax.plot(sim.trange(), sim.data[pEns], label='ens')
#     ax.legend()
#     fig.savefig("plots/diffMemory_ideal.pdf")

def go(NPre=100, N=100, t=10, c=None, dt=0.001, m=Uniform(30, 30), i=Uniform(-0.8, 0.8), seed=0, neuron_type=LIF(),
    fPre=None, fNMDA=None, dPre=None, dDiff=None, dEns=None, dNeg=None, Tff=1.0, stim=lambda t: 0, train=False):

    if not c: c = t
    dFF = dDiff*Tff if np.any(dDiff) else None
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        cut = nengo.Node(lambda t: 0 if t<c else 1)
        pre = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        pre2 = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
        diff = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s)
        nengo.Connection(inpt, pre, synapse=None, seed=seed)
        nengo.Connection(intg, pre2, synapse=fNMDA, seed=seed)
        c1 = nengo.Connection(pre, diff, synapse=fPre, solver=NoSolver(dPre), seed=seed)
        c2 = nengo.Connection(diff, ens, synapse=fNMDA, solver=NoSolver(dFF), seed=seed)
        c3 = nengo.Connection(ens, ens, synapse=fNMDA, solver=NoSolver(dEns), seed=seed)
        c4 = nengo.Connection(ens, diff, synapse=fNMDA, solver=NoSolver(dNeg), seed=seed)
        c5 = nengo.Connection(cut, diff.neurons, synapse=None, transform=-1e2*np.ones((N, 1)))
        if train: nengo.Connection(pre2, ens, synapse=fPre, solver=NoSolver(dPre), seed=seed)
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pDiff = nengo.Probe(diff.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        pre=sim.data[pPre],
        diff=sim.data[pDiff],
        ens=sim.data[pEns],
    )


def run(NPre=100, N=100, t=10, c=None, nTrain=10, nTest=5, dt=0.001, neuron_type=LIF(),
    fPre=DoubleExp(1e-3, 1e-1), fNMDA=DoubleExp(10.6e-3, 285e-3),
    reg=1e-2, load=[], file=None, neg=True):

    if not c: c = t
    if 0 in load:
        dPre = np.load(file)['dPre']
    else:
        print('readout decoders for pre')
        spikesInpt = np.zeros((nTrain, int(t/dt), NPre))
        targetsInpt = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, fPre, fNMDA, seed=n)
            data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type, stim=stim)
            spikesInpt[n] = data['pre']
            targetsInpt[n] = fPre.filt(data['inpt'], dt=dt)
        dPre, X, Y, error = decodeNoF(spikesInpt, targetsInpt, nTrain, fPre, reg=reg)
        np.savez("data/diffMemory_%s.npz"%neuron_type, dPre=dPre)
        times = np.arange(0, t*nTrain, 0.001)
        plotState(times, X, Y, error, "diffMemory_%s"%neuron_type, "pre", t*nTrain)

    if 1 in load:
        dDiff = np.load(file)['dDiff']
    else:
        print('readout decoders for diff')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, fPre, fNMDA, seed=n)
            data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type, stim=stim, dPre=dPre)
            spikes[n] = data['diff']
            targets[n] = fNMDA.filt(fPre.filt(data['inpt'], dt=dt), dt=dt)
        dDiff, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, reg=reg)
        np.savez("data/diffMemory_%s.npz"%neuron_type, dPre=dPre, dDiff=dDiff)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "diffMemory_%s"%neuron_type, "diff", t*nTrain)

    if 2 in load:
        dEns = np.load(file)['dEns']
        dNeg = np.load(file)['dNeg']
    else:
        print('readout decoders for ens')
        spikes = np.zeros((nTrain, int(t/dt), N))
        targets = np.zeros((nTrain, int(t/dt), 1))
        for n in range(nTrain):
            stim = makeSignal(t, fPre, fNMDA, seed=n)
            # data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type, stim=stim, dPre=dPre, dDiff=dDiff)
            data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type, stim=stim, dPre=dPre, dDiff=dDiff, train=True, Tff=0.3)
            spikes[n] = data['ens']
            # targets[n] = fNMDA.filt(fNMDA.filt(fPre.filt(data['inpt'], dt=dt), dt=dt), dt=dt)
            targets[n] = fNMDA.filt(fPre.filt(data['intg'], dt=dt), dt=dt)
        dEns, X, Y, error = decodeNoF(spikes, targets, nTrain, fNMDA, reg=reg)
        dNeg = -np.array(dEns)
        np.savez("data/diffMemory_%s.npz"%neuron_type, dPre=dPre, dDiff=dDiff, dEns=dEns, dNeg=dNeg)
        times = np.arange(0, t*nTrain, dt)
        plotState(times, X, Y, error, "diffMemory_%s"%neuron_type, "ens", t*nTrain)

    print('testing')
    vals = np.linspace(-1, 1, nTest)
    for test in range(nTest):
        if neg:
            stim = lambda t: vals[test] if t<c else 0
            data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type,
                stim=stim, dPre=dPre, dDiff=dDiff, dEns=dEns, dNeg=dNeg, c=c)        
        else:
            stim = makeSignal(t, fPre, fNMDA, seed=test)
            data = go(NPre=NPre, N=N, t=t, fPre=fPre, fNMDA=fNMDA, neuron_type=neuron_type,
                stim=stim, dPre=dPre, dDiff=dDiff, dEns=dEns, dNeg=None, c=None, Tff=0.3)        
        aDiff = fNMDA.filt(fPre.filt(data['diff'], dt=dt), dt=dt)
        aEns = fNMDA.filt(fNMDA.filt(fPre.filt(data['ens'], dt=dt), dt=dt), dt=dt)
        xhatDiff = np.dot(aDiff, dDiff)
        xhatEns = np.dot(aEns, dEns)
        u = fNMDA.filt(fPre.filt(data['inpt'], dt=dt), dt=dt)
        u2 = fNMDA.filt(fNMDA.filt(fPre.filt(data['inpt'], dt=dt), dt=dt), dt=dt)
        x = fNMDA.filt(fNMDA.filt(fPre.filt(data['intg'], dt=dt), dt=dt), dt=dt)
        error = rmse(xhatEns, x)
        fig, ax = plt.subplots()
        if neg:
            ax.plot(data['times'], u2, alpha=0.5, label="input (delayed)")
            ax.axvline(c, label="cutoff")
        else:
            # ax.plot(data['times'], 0.3*u2, alpha=0.5, label="input")
            ax.plot(data['times'], x, alpha=0.5, label="integral")
        # ax.plot(data['times'], xhatDiff, label="diff")
        ax.plot(data['times'], xhatEns, label="ens")
        ax.set(xlabel='time', ylabel='state', title="rmse=%.3f"%error, xlim=((0, t)), ylim=((-1, 1)))
        ax.legend(loc='upper left')
        sns.despine()
        fig.savefig("plots/diffMemory_%s_test%s.pdf"%(neuron_type, test))

# ideal()
# run(N=100, t=10, c=5, nTrain=3, load=[0,1,2], neg=True, neuron_type=LIF(), file="data/diffMemory_LIF().npz")
run(N=30, t=10, c=5, nTrain=3, load=[0,1,2], neg=False, neuron_type=ALIF(), file="data/diffMemory_ALIF().npz")