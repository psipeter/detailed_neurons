import numpy as np
import nengo
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, f, dt=0.001, value=1.0, nFilts=2, seed=0):
    stim = nengo.processes.WhiteSignal(period=t/4, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t/4, progress_bar=False)
    u = sim.data[pU]
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    norm = value / np.max(np.abs(u))
    mirrored = np.concatenate(([[0]],
        sim.data[pU]*norm,
        -sim.data[pU]*norm,
        sim.data[pU]*norm,
        -sim.data[pU]*norm))
    return lambda t: mirrored[int(t/dt)]

def go(t=10, m=Uniform(30, 30), i=Uniform(0, 0), seed=0, dt=0.001, f=DoubleExp(1e-3, 1e-1), fS=DoubleExp(1e-3, 1e-1), d1=None, f1=None, e1=None, l1=False, stim=lambda t: np.sin(t)):

    if not f1: f1=f
    with nengo.Network(seed=seed) as model:
        # Stimulus and Nodes
        inpt = nengo.Node(stim)
        tar = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        pre = nengo.Ensemble(100, 1, max_rates=m, seed=seed, neuron_type=LIF())
        lif = nengo.Ensemble(1, 1, max_rates=m, intercepts=i, encoders=Choice([[1]]), neuron_type=LIF(), seed=seed)
        wilson = nengo.Ensemble(1, 1, max_rates=m, intercepts=i, encoders=Choice([[1]]), neuron_type=Wilson(), seed=seed)
        bio = nengo.Ensemble(1, 1, max_rates=m, intercepts=i, encoders=Choice([[1]]), neuron_type=Bio("Pyramidal"), seed=seed)
        nengo.Connection(inpt, pre, synapse=None, seed=seed)
        cLif = nengo.Connection(pre, lif, synapse=f1, seed=seed, solver=NoSolver(d1))
        cWilson = nengo.Connection(pre, wilson, synapse=f1, seed=seed, solver=NoSolver(d1))
        cBio = nengo.Connection(pre, bio, synapse=f1, seed=seed, solver=NoSolver(d1))
        pInpt = nengo.Probe(inpt, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pLif = nengo.Probe(lif.neurons, synapse=None)
        pWilson = nengo.Probe(wilson.neurons, synapse=None)
        pBio = nengo.Probe(bio.neurons, synapse=None)
        if l1: learnEncoders(cBio, lif, fS, alpha=3e-7) # Encoder Learning (Bio)

    with nengo.Simulator(model, seed=seed, dt=dt, progress_bar=False) as sim:
        setWeights(cBio, d1, e1)
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 
      
    e1 = cBio.e if l1 else e1

    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        pre=sim.data[pPre],
        lif=sim.data[pLif],
        wilson=sim.data[pWilson],
        bio=sim.data[pBio],
        e1=e1,
    )

def run(t=60, tTrans=30, dt=1e-4, maxRate=30, intercept=-0.25, f=DoubleExp(1e-3, 3e-2), fS=DoubleExp(1e-2, 1e-1), nBins=21):

    print('training readout decoders for pre')
    stim = makeSignal(t, f, dt=0.001, seed=0)
    data = go(t=t, m=Uniform(maxRate, maxRate), i=Uniform(intercept, intercept), dt=0.001, f=f, fS=fS, stim=stim)
    spikes = np.array([data['pre']])
    targets = np.array([f.filt(data['inpt'], dt=0.001)])
    d1, f1, tauRise1, tauFall1, X, Y, error = decode(spikes, targets, 1, dt=0.001, name="tuneNew")

    print("training encoders")
    e1 = np.zeros((100, 1, 1))
    stim = makeSignal(t, f, dt=dt, seed=0)
    data = go(d1=d1, e1=e1, f1=f1, t=t, dt=dt, f=f, fS=fS, stim=stim, l1=True)
    x = fS.filt(data['inpt'], dt=dt)
    times = data['times']
    aLif = fS.filt(data['lif'], dt=dt)
    aWilson = fS.filt(data['wilson'], dt=dt)
    aBio = fS.filt(data['bio'], dt=dt)
    
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax.plot(times, x, label="input")
    ax.axhline(intercept, color='k', alpha=0.5, label="intercept")
    ax2.plot(times, aLif, alpha=0.5, label='LIF')
    ax2.plot(times, aWilson, alpha=0.5, label='Wilson')
    ax2.plot(times, aBio, alpha=0.5, label='bio')
    ax.set(xlim=((0, tTrans)), ylim=((-1, 1)), ylabel=r"$\mathbf{x}$(t)")
    ax2.set(xlim=((0, tTrans)), ylim=((0, 40)), xlabel='time (s)', ylabel=r"$a(t)$")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.savefig('plots/tuneNew.pdf')
    plt.close('all')

    x = fS.filt(data['inpt'], dt=dt)[int(tTrans/dt):]
    times = data['times'][int(tTrans/dt):]
    aLif = fS.filt(data['lif'], dt=dt)[int(tTrans/dt):]
    aWilson = fS.filt(data['wilson'], dt=dt)[int(tTrans/dt):]
    aBio = fS.filt(data['bio'], dt=dt)[int(tTrans/dt):]
    bins = np.linspace(-1, 1, nBins)
    bLif, bWilson, bBio = [], [], []
    for b in range(len(bins)):
        bLif.append([])
        bWilson.append([])
        bBio.append([])
    for t in range(len(times)):
        idx = (np.abs(bins - x[t])).argmin()
        bLif[idx].append(aLif[t][0])
        bWilson[idx].append(aWilson[t][0])
        bBio[idx].append(aBio[t][0])
    mLif, mBio, mWilson = np.zeros_like(bins), np.zeros_like(bins), np.zeros_like(bins)
    ci1Lif, ci1Bio, ci1Wilson = np.zeros_like(bins), np.zeros_like(bins), np.zeros_like(bins)
    ci2Lif, ci2Bio, ci2Wilson = np.zeros_like(bins), np.zeros_like(bins), np.zeros_like(bins)
    for b in range(len(bins)):
        mLif[b] = np.mean(bLif[b])
        mWilson[b] = np.mean(bWilson[b])
        mBio[b] = np.mean(bBio[b])
        ci1Lif[b] = sns.utils.ci(np.array(bLif[b]), which=95)[0]
        ci1Wilson[b] = sns.utils.ci(np.array(bWilson[b]), which=95)[0]
        ci1Bio[b] = sns.utils.ci(np.array(bBio[b]), which=95)[0]
        ci2Lif[b] = sns.utils.ci(np.array(bLif[b]), which=95)[1]
        ci2Wilson[b] = sns.utils.ci(np.array(bWilson[b]), which=95)[1]
        ci2Bio[b] = sns.utils.ci(np.array(bBio[b]), which=95)[1]

    fig, ax = plt.subplots()
    ax.plot(bins, mLif, label="LIF")
    ax.fill_between(bins, ci1Lif, ci2Lif, alpha=0.1)
    ax.plot(bins, mWilson, label="Wilson")
    ax.fill_between(bins, ci1Wilson, ci2Wilson, alpha=0.1)
    ax.plot(bins, mBio, label="Bio")
    ax.fill_between(bins, ci1Bio, ci2Bio, alpha=0.1)
    ax.axhline(maxRate, color='k', linestyle="--", label="target max rate")
    ax.axvline(intercept, color='k', label="target intercept")
    ax.set(xlim=((-1, 1)), ylim=((0, maxRate+1)), xlabel=r"$\mathbf{x}$", ylabel="firing rate (Hz)")
    ax.legend()
    fig.savefig("plots/tuneNew2.pdf")

run()