import numpy as np
import nengo
from scipy.optimize import nnls
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver, LstsqL2
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s
from nengolib.neurons import init_lif
from utils import dfOpt, WNode, learnEncoders, setWeights, decode, plotState, plotActivity
from neurons import LIF, ALIF, Wilson, Bio, reset_neuron, AMPA, GABA, NMDA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def makeSignal(t, f, dt=0.001, value=1.0, nFilts=3, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[pU][::2]
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    norm = value / np.max(np.abs(u))
    stim = sim.data[pU][::2] * norm
    mirrored = np.concatenate((stim, -stim))
    return lambda t: mirrored[int(t/dt)]

def goTarget(f1=Lowpass(0.01), f2=Lowpass(0.1), stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 30), i=Uniform(-1, 0.6), kInh=-1.5, seed=0):
    wInh = kInh*np.ones((N, 1))
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        gate = nengo.Node(gating)
        fdfw = nengo.Ensemble(N, 1, seed=seed)
        fdbk = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        nengo.Connection(inpt, fdfw, synapse=None)
        nengo.Connection(fdfw, ens, synapse=f1)
        nengo.Connection(ens, fdbk, synapse=f1)
        nengo.Connection(fdbk, ens, synapse=f2)
        nengo.Connection(gate, fdfw.neurons, transform=wInh, function=lambda x: x)
        nengo.Connection(gate, fdbk.neurons, transform=wInh, function=lambda x: 1-x)
        pInpt = nengo.Probe(inpt, synapse=f2)
        pGate = nengo.Probe(gate, synapse=None)
        pFdfw = nengo.Probe(fdfw, synapse=f2)
        pFdbk = nengo.Probe(fdbk, synapse=f2)
        pEns = nengo.Probe(ens, synapse=f2)
    with nengo.Simulator(model, seed=seed) as sim:
        sim.run(t)
    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        gate=sim.data[pGate],
        fdfw=sim.data[pFdfw],
        fdbk=sim.data[pFdbk],
        ens=sim.data[pEns]   
    )

def easy(t=10, dt=0.001, f1=Lowpass(1e-2), f2=Lowpass(1e-1)):
    stim = makeSignal(t=t, dt=dt, f=f2)
    gating = lambda t: 0 if (0<t<1 or 8<t<9) else 1
    data = goTarget(t=t, f1=f1, f2=f2, stim=stim, gating=gating)

    fig, ax = plt.subplots()
    ax.plot(data['times'], data['inpt'], linestyle="--", label='inpt')
    ax.plot(data['times'], data['gate'], linestyle="--", label='gate')
    ax.plot(data['times'], data['fdfw'], alpha=0.5, label='fdfw')
#     ax.plot(data['times'], data['fdbk'], alpha=0.5, label='fdbk')
    ax.plot(data['times'], data['ens'], alpha=0.5, label='ens')
    ax.legend(loc="upper right")
    ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goTarget.pdf")
    
# easy()
# raise


def makeSignalEven(T, dt=0.001, value=1.0):
    return lambda t: value/2 - value/2*np.cos(2*np.pi*t/T)

def makeSignalPos(t, f, dt=0.001, value=1.0, nFilts=2, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = np.abs(np.array(sim.data[pU]))
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    norm = value / np.max(u)
    stim = np.abs(sim.data[pU]) * norm
    return lambda t: stim[int(t/dt)]

def makeSignalIntg(t, f, dt=0.001, value=1.0, nFilts=2, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = np.array(sim.data[pU])
    u[u<0] = 0
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    x = np.cumsum(u)
    norm = value / np.max(x)
    stim = x * norm
    return lambda t: stim[int(t/dt)]

def makeSignalSquare(t, dt=0.001, nSquare=4, seed=0):
    rng = np.random.RandomState(seed=seed)
    lengths = rng.uniform(t/20, t/5, size=(nSquare))
    heights = rng.uniform(0.2, 0.3, size=(nSquare))
#     heights = 1.0 / (nSquare*lengths)
    starts = np.arange(0, t, t/nSquare)
    stim = np.zeros((int(t/dt)+1))
    for ti in range(len(stim)):
        for n in range(nSquare):
            if starts[n] < ti*dt < starts[n]+lengths[n]:
                stim[ti] = heights[n]
    return lambda t: stim[int(t/dt)]

def goLIF(N=100, t=10, dt=0.001, stim=lambda t: 0, m=Uniform(30, 30), i=Uniform(0, 0.8), e=Choice([[1]]), i2=Uniform(0.4, 1), dFdfwEnsAMPA=None, dFdfwEnsNMDA=None, dEnsEnsAMPA=None, dEnsEnsNMDA=None, dEnsInhAMPA=None, dEnsInhNMDA=None, dInhFdfwGABA=None, dInhEnsGABA=None, fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(2.3e-3, 95.0e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), x0=0, seed=0):

    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        fdfw = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, seed=seed)
        inh = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i2, neuron_type=LIF(), seed=seed)
        ens = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=LIF(), seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s)
        nengo.Connection(inpt, fdfw, synapse=None)
        nengo.Connection(fdfw, ens, synapse=fAMPA, solver=NoSolver(dFdfwEnsAMPA))
        nengo.Connection(fdfw, ens, synapse=fNMDA, solver=NoSolver(dFdfwEnsNMDA))
        nengo.Connection(ens, ens, synapse=fAMPA, solver=NoSolver(dEnsEnsAMPA))
        nengo.Connection(ens, ens, synapse=fNMDA, solver=NoSolver(dEnsEnsNMDA))
        nengo.Connection(ens, inh, synapse=fAMPA, solver=NoSolver(dEnsInhAMPA))
        nengo.Connection(ens, inh, synapse=fNMDA, solver=NoSolver(dEnsInhNMDA))
        nengo.Connection(inh, fdfw, synapse=fGABA, solver=NoSolver(dInhFdfwGABA))
        nengo.Connection(inh, ens, synapse=fGABA, solver=NoSolver(dInhEnsGABA))
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
#         init_lif(sim, ens, x0=x0)
        sim.run(t, progress_bar=True)
    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        fdfw=sim.data[pFdfw],
        inh=sim.data[pInh],
        ens=sim.data[pEns]   
    )

def medium(N=300, t=10, dt=0.001, nTrain=5, nTest=5, fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(2.3e-3, 95.0e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), daAMPA=0.8, daNMDA=0.7, daGABA=0.8, kEnsEnsAMPA=0.2, kEnsEnsNMDA=0.85, kEnsInhAMPA=0.1, kEnsInhNMDA=1.0, kGABAFdfw=-4.0, kGABAEns=-1e-3, T=0.05):

    print("readout decoders for fdfw")
    targetsAMPA = np.zeros((1, 1))
    targetsNMDA = np.zeros((1, 1))
    asAMPA = np.zeros((1, N))
    asNMDA = np.zeros((1, N))
    for n in range(nTrain):
        stim = makeSignalEven(t, dt=dt, value=(n+1)/nTrain)
#         stim = makeSignalPos(t, fNMDA, dt=dt, seed=n)
        data = goLIF(N=N, stim=stim, t=t, dt=dt,
            fAMPA=fAMPA, fNMDA=fNMDA, fGABA=fGABA)
        asAMPA = np.append(asAMPA, fAMPA.filt(data['fdfw']), axis=0)
        asNMDA = np.append(asNMDA, fNMDA.filt(data['fdfw']), axis=0)
        targetsAMPA = np.append(targetsAMPA, fAMPA.filt(data['inpt']), axis=0)
        targetsNMDA = np.append(targetsNMDA, fNMDA.filt(data['inpt']), axis=0)
#     dFdfwEnsAMPA, _ = LstsqL2(reg=1e-1)(asAMPA, np.ravel(targetsAMPA))
#     dFdfwEnsNMDA, _ = LstsqL2(reg=1e-1)(asNMDA, np.ravel(targetsNMDA))
    dFdfwEnsAMPA, _ = nnls(asAMPA, np.ravel(targetsAMPA))
    dFdfwEnsNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
    dFdfwEnsAMPA = dFdfwEnsAMPA.reshape((N, 1))
    dFdfwEnsNMDA = dFdfwEnsNMDA.reshape((N, 1))
    xhatFFAMPA = np.dot(asAMPA, dFdfwEnsAMPA)
    xhatFFNMDA = np.dot(asNMDA, dFdfwEnsNMDA)
    fig, ax = plt.subplots()
    ax.plot(targetsAMPA, linestyle="--", label='target (AMPA)')
    ax.plot(targetsNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(xhatFFAMPA, alpha=0.5, label='fdfw (AMPA)')
    ax.plot(xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.legend(loc="upper right")
    ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_fdfw.pdf")

    print("readout decoders for ens, given AMPA and NMDA from fdfw in high DA condition")
    targetsAMPA = np.zeros((1, 1))
    targetsNMDA = np.zeros((1, 1))
    asAMPA = np.zeros((1, N))
    asNMDA = np.zeros((1, N))
    for n in range(nTrain):
        stim = makeSignalEven(t, dt=dt, value=(n+1)/nTrain)
#         stim = makeSignalPos(t, fNMDA, dt=dt, seed=n)
        data = goLIF(N=N, stim=stim, t=t, dt=dt,
            fAMPA=fAMPA, fNMDA=fNMDA, fGABA=fGABA,
            dFdfwEnsAMPA=0*dFdfwEnsAMPA, dFdfwEnsNMDA=dFdfwEnsNMDA)
        asAMPA = np.append(asAMPA, fAMPA.filt(data['ens']), axis=0)
        asNMDA = np.append(asNMDA, fNMDA.filt(data['ens']), axis=0)
        targetsAMPA = np.append(targetsAMPA, fAMPA.filt(fNMDA.filt(data['inpt'])), axis=0)
        targetsNMDA = np.append(targetsNMDA, fNMDA.filt(fNMDA.filt(data['inpt'])), axis=0)
#     dEnsEnsAMPA, _ = LstsqL2(reg=1e-1)(asAMPA, kAMPA*np.ravel(targetsAMPA))
#     dEnsEnsNMDA, _ = LstsqL2(reg=1e-1)(asNMDA, kNMDA*np.ravel(targetsNMDA))
    dEnsEnsAMPA, _ = nnls(asAMPA, kEnsEnsAMPA*np.ravel(targetsAMPA))
    dEnsEnsNMDA, _ = nnls(asNMDA, kEnsEnsNMDA*np.ravel(targetsNMDA))
    dEnsEnsAMPA = dEnsEnsAMPA.reshape((N, 1))
    dEnsEnsNMDA = dEnsEnsNMDA.reshape((N, 1))
    dEnsInhAMPA, _ = nnls(asAMPA, kEnsInhAMPA*np.ravel(targetsAMPA))
    dEnsInhNMDA, _ = nnls(asNMDA, kEnsInhNMDA*np.ravel(targetsNMDA))
    dEnsInhAMPA = dEnsInhAMPA.reshape((N, 1))
    dEnsInhNMDA = dEnsInhNMDA.reshape((N, 1))
    xhatFBAMPA = np.dot(asAMPA, dEnsEnsAMPA)
    xhatFBNMDA = np.dot(asNMDA, dEnsEnsNMDA)
    xhatConstAMPA = np.dot(asAMPA, dEnsInhAMPA)
    xhatConstNMDA = np.dot(asNMDA, dEnsInhNMDA)
    fig, ax = plt.subplots()
    ax.plot(kEnsEnsAMPA*targetsAMPA, linestyle="--", label='target (AMPA)')
    ax.plot(kEnsEnsNMDA*targetsNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(xhatFBAMPA, alpha=0.5, label='ens (AMPA)')
    ax.plot(xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
    ax.legend(loc="upper right")
    ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_ens.pdf")

    print("readout decoders for inh, given ff and fb AMPA and NMDA and high DA")
    targetsGABA = np.zeros((1, 1))
    asGABA = np.zeros((1, N))
    for n in range(nTrain):
        stim = makeSignalEven(t, dt=dt, value=(n+1)/nTrain)
#         stim = makeSignalPos(t, fNMDA, dt=dt, seed=n)
        data = goLIF(N=N, stim=stim, t=t, dt=dt,
            fAMPA=fAMPA, fNMDA=fNMDA, fGABA=fGABA,
            dFdfwEnsAMPA=daAMPA*dFdfwEnsAMPA, dFdfwEnsNMDA=dFdfwEnsNMDA,
            dEnsInhAMPA=daAMPA*dEnsInhAMPA, dEnsInhNMDA=dEnsInhNMDA)
        asGABA = np.append(asGABA, fGABA.filt(data['inh']), axis=0)
        targetsGABA = np.append(targetsGABA, fGABA.filt(fNMDA.filt(data['inpt'])), axis=0)
    dInhEnsGABA, _ = nnls(-asGABA, kGABAEns*np.ravel(targetsGABA))
    dInhFdfwGABA, _ = nnls(-asGABA, kGABAFdfw*np.ravel(targetsGABA))
    dInhEnsGABA = -dInhEnsGABA.reshape((N, 1))
    dInhFdfwGABA = -dInhFdfwGABA.reshape((N, 1))
    xhatInhEns = np.dot(asGABA, dInhEnsGABA)
    xhatInhFdfw = np.dot(asGABA, dInhFdfwGABA)
    fig, ax = plt.subplots()
    ax.plot(kGABAEns*targetsGABA, linestyle="--", label='target (ens)')
    ax.plot(kGABAFdfw*targetsGABA, linestyle="--", label='target (fdfw)')
    ax.plot(xhatInhEns, alpha=0.5, label='inh (ens)')
    ax.plot(xhatInhFdfw, alpha=0.5, label='inh (fdfw)')
    ax.legend(loc="upper right")
    ax.set(ylim=((-1, 0)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_inh.pdf")
        
    # Test in high and low DA
    print('testing with high and low DA')
    for n in range(nTest):
        stim = makeSignalSquare(t, dt=dt, seed=100+n)
#         data = goLIF(N=N, stim=lambda t: 0, x0=n/nTest, t=t, dt=dt,
        data = goLIF(N=N, stim=stim, t=t, dt=dt,
            fAMPA=fAMPA, fNMDA=fNMDA, fGABA=fGABA,
            dFdfwEnsAMPA=daAMPA*T*dFdfwEnsAMPA, dFdfwEnsNMDA=T*dFdfwEnsNMDA,
            dEnsEnsAMPA=daAMPA*dEnsEnsAMPA, dEnsEnsNMDA=dEnsEnsNMDA,
            dEnsInhAMPA=daAMPA*dEnsInhAMPA, dEnsInhNMDA=dEnsInhNMDA,
            dInhEnsGABA=dInhEnsGABA, dInhFdfwGABA=dInhFdfwGABA)
        aFdfwNMDA = fNMDA.filt(data['fdfw'])
        targetFdfwNMDA = fNMDA.filt(data['inpt'])
        xhatFdfwNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
        aEnsNMDA = fNMDA.filt(data['ens'])
        targetIntgNMDA = fNMDA.filt(data['intg'])
#         targetFlatNMDA = n/nTest*np.ones((aEnsNMDA.shape[0]))
        xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
        fig, (ax, ax2) = plt.subplots(ncols=1, nrows=2, sharex=True)
        ax.plot(data['times'], targetFdfwNMDA, linestyle="--", label='input (NMDA)')
        ax.plot(data['times'], xhatFdfwNMDA, alpha=0.5, label='fdfw (NMDA)')
        ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
#         ax.plot(data['times'], targetFlatNMDA, linestyle="--", label='flat (NMDA)')
        ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
        ax.legend(loc="upper right")
        ax.set(ylim=((0, 1)), ylabel=r"$\mathbf{\hat{x}}(t)$", title="high DA")
        stim = makeSignalSquare(t, dt=dt, seed=100+n)
        data = goLIF(N=N, stim=stim, t=t, dt=dt,
            fAMPA=fAMPA, fNMDA=fNMDA, fGABA=fGABA,
            dFdfwEnsAMPA=T*dFdfwEnsAMPA, dFdfwEnsNMDA=daNMDA*T*dFdfwEnsNMDA,
            dEnsEnsAMPA=dEnsEnsAMPA, dEnsEnsNMDA=daNMDA*dEnsEnsNMDA,
            dEnsInhAMPA=dEnsInhAMPA, dEnsInhNMDA=daNMDA*dEnsInhNMDA,
            dInhEnsGABA=daGABA*dInhEnsGABA, dInhFdfwGABA=daGABA*dInhFdfwGABA)
        aFdfwNMDA = fNMDA.filt(data['fdfw'])
        targetFdfwNMDA = fNMDA.filt(data['inpt'])
        xhatFdfwNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
        aEnsNMDA = fNMDA.filt(data['ens'])
        targetIntgNMDA = fNMDA.filt(data['intg'])
#         targetFlatNMDA = n/nTest*np.ones((aEnsNMDA.shape[0]))
        xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
#         fig, ax = plt.subplots()
        ax2.plot(data['times'], targetFdfwNMDA, linestyle="--", label='input (NMDA)')
        ax2.plot(data['times'], xhatFdfwNMDA, alpha=0.5, label='fdfw (NMDA)')
        ax2.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
#         ax2.plot(data['times'], targetFlatNMDA, linestyle="--", label='flat (NMDA)')
        ax2.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
        ax2.legend(loc="upper right")
        ax2.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$", title="low DA")
        fig.savefig("plots/gatedMemory_goLIF_test%s.pdf"%n)

medium()