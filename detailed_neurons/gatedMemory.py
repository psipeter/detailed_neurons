import numpy as np
from scipy.optimize import nnls
import nengo
from nengo.params import Default
from nengo.dists import Uniform, Choice
from nengo.solvers import NoSolver, LstsqL2
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import WNode
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, BioNeuron, reset_neuron, AMPA, GABA, NMDA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def makeSignal(t=10.0, dt=0.001, period=10, f=Lowpass(0.05), seed=0):
    stim = nengo.processes.WhiteSignal(period=period, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    u = f.filt(sim.data[pU], dt=dt)
    norm = 1.0 / np.max(np.abs(u))
    stim = np.ravel(u) * norm
    mirrored = np.concatenate(([0], stim, -stim))[::2]
    return mirrored


def goTarget(fFF=Lowpass(0.01), fFB=Lowpass(0.1), stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(-1, 0.8), kInh=-1.5, seed=0):
    wInh = kInh*np.ones((N, 1))
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        gate = nengo.Node(gating)
        fdfw = nengo.Ensemble(N, 1, seed=seed)
        fdbk = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        nengo.Connection(inpt, fdfw, synapse=None)
        nengo.Connection(fdfw, ens, synapse=fFF)
        nengo.Connection(ens, fdbk, synapse=fFF)
        nengo.Connection(fdbk, ens, synapse=fFB)
        nengo.Connection(gate, fdfw.neurons, transform=wInh, function=lambda x: x)
        nengo.Connection(gate, fdbk.neurons, transform=wInh, function=lambda x: 1-x)
        pInpt = nengo.Probe(inpt, synapse=fFB)
        pGate = nengo.Probe(gate, synapse=None)
        pFdfw = nengo.Probe(fdfw, synapse=fFB)
        pFdbk = nengo.Probe(fdbk, synapse=fFB)
        pEns = nengo.Probe(ens, synapse=fFB)
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

def ideal():
    t = 10
    dt = 0.001
    s = makeSignal(t=t, dt=dt, period=t)
    stim = lambda t: s[int(t/dt)]
    gating = lambda t: 0 if (0<t<2.5 or 5<t<7.5) else 1
    data = goTarget(t=t, stim=stim, gating=gating)

    fig, ax = plt.subplots()
    ax.plot(data['times'], data['inpt'], linestyle="--", label='inpt')
    ax.plot(data['times'], data['gate'], linestyle="--", label='gate')
    ax.plot(data['times'], data['fdfw'], alpha=0.5, label='fdfw')
    ax.plot(data['times'], data['fdbk'], alpha=0.5, label='fcbk')
    ax.plot(data['times'], data['ens'], alpha=0.5, label='ens')
    ax.legend()
    ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goTarget.pdf")
    
# ideal()


def goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(-1, 0.8), seed=0):

    absv = lambda x: np.abs(x)
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        fdfw = nengo.Ensemble(N, 1, seed=seed)
        inh = nengo.Ensemble(N, 1, encoders=Choice([[1]]), intercepts=Uniform(0, 1), neuron_type=nengo.LIF(), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s)
        nengo.Connection(inpt, fdfw, synapse=None)
        nengo.Connection(fdfw, ens, synapse=fAMPA, solver=NoSolver(dFFAMPA))
        nengo.Connection(fdfw, ens, synapse=fNMDA, solver=NoSolver(dFFNMDA))
        nengo.Connection(ens, inh, synapse=fAMPA, transform=wEnsInhAMPA, function=absv)
        nengo.Connection(ens, inh, synapse=fNMDA, transform=wEnsInhNMDA, function=absv)
        nengo.Connection(ens, ens, synapse=fAMPA, solver=NoSolver(dFBAMPA))
        nengo.Connection(ens, ens, synapse=fNMDA, solver=NoSolver(dFBNMDA))
        nengo.Connection(inh.neurons, fdfw.neurons, synapse=fGABA, transform=wInhFdfw)
        nengo.Connection(inh.neurons, ens.neurons, synapse=fGABA, transform=wInhEns)
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pInhState = nengo.Probe(inh, synapse=fGABA)
        pEns = nengo.Probe(ens.neurons, synapse=None)
    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)
    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        fdfw=sim.data[pFdfw],
        inh=sim.data[pInh],
        inhState=sim.data[pInhState],
        ens=sim.data[pEns]   
    )

def easy():
    N = 100
    dFFAMPA = np.zeros((N, 1))
    dFFNMDA = np.zeros((N, 1))
    dFBAMPA = np.zeros((N, 1))
    dFBNMDA = np.zeros((N, 1))
    fAMPA = DoubleExp(5.5e-4, 2.2e-3)
    fNMDA = DoubleExp(1e-2, 2.85e-1)
    fGABA = DoubleExp(5e-4, 1.5e-3)
    rng = np.random.RandomState(seed=0)
    kAMPA = 0.1
    kNMDA = 0.7
    kGABA = 0.8
    wEnsInhAMPA = 0.2
    wEnsInhNMDA = 2
    wInhFdfw = rng.uniform(-1.3e-3, 0, size=(N, N))
    wInhEns = rng.uniform(-1e-4, 0, size=(N, N))
    t = 10
    dt = 0.001
    u = makeSignal(t=t, dt=dt, period=t, f=fNMDA)
    stim = lambda t: u[int(t/dt)]

    # Stage 1 - feedforward decoders from fdfw to ens
    data = goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, 0*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwAMPA = fAMPA.filt(data['fdfw'])
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetAMPA = fAMPA.filt(data['inpt'])
    targetNMDA = fNMDA.filt(data['inpt'])
    dFFAMPA, _ = LstsqL2(reg=1e-2)(aFdfwAMPA, targetAMPA)
    dFFNMDA, _ = LstsqL2(reg=1e-2)(aFdfwNMDA, targetNMDA)
    xhatFFAMPA = np.dot(aFdfwAMPA, dFFAMPA)
    xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], targetAMPA, linestyle="--", label='target (AMPA)')
    ax.plot(data['times'], targetNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(data['times'], xhatFFAMPA, alpha=0.5, label='fdfw (AMPA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_fdfw.pdf")

    # Stage 2 - readout decoders for ens; assume high DA condition and inh-ens, and only get NMDA decoders
    data = goLIF(kAMPA*dFFAMPA, dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetNMDA = fNMDA.filt(fNMDA.filt(data['inpt']))
    dFBNMDA, _ = LstsqL2(reg=1e-2)(aEnsNMDA, targetNMDA)
    dFBAMPA = rng.uniform(0, 1e-4, size=(N, 1))  # negligable feedback
    xhatFBNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], targetNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(data['times'], xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
    ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_fdbk.pdf")

    # Stage 3 - test integration in high vs low DA; assume inh-ens
    # high DA
    data = goLIF(kAMPA*dFFAMPA, 0.285*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
#     ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_intg_highDA.pdf")
    # low DA
    data = goLIF(dFFAMPA, kNMDA*0.285*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, wEnsInhAMPA, kNMDA*wEnsInhNMDA, 0*wInhFdfw, kGABA*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
#     ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_intg_lowDA.pdf")

    # Stage 4 - test integration with inh-fdfw in high vs low DA
    data = goLIF(kAMPA*dFFAMPA, 0.285*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfw = fNMDA.filt(data['fdfw'])
    target = fNMDA.filt(data['inpt'])
    xhatFF = np.dot(aFdfw, dFFNMDA)
    aEns = fNMDA.filt(data['ens'])
    targetIntg = fNMDA.filt(data['intg'])
    xhatIntg = np.dot(aEns, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], target, linestyle="--", label='input')
    ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
#     ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_inh_highDA.pdf")
    # Low DA
    data = goLIF(dFFAMPA, kNMDA*0.285*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, wEnsInhAMPA, kNMDA*wEnsInhNMDA, kGABA*wInhFdfw, kGABA*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfw = fNMDA.filt(data['fdfw'])
    target = fNMDA.filt(data['inpt'])
    xhatFF = np.dot(aFdfw, dFFNMDA)
    aEns = fNMDA.filt(data['ens'])
    targetIntg = fNMDA.filt(data['intg'])
    xhatIntg = np.dot(aEns, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], target, linestyle="--", label='input')
    ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
#     ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_inh_lowDA.pdf")

# easy()


def makeSignalCutoff(t=10.0, dt=0.001, period=10, f=Lowpass(0.05), value=1.0, nFilts=2, seed=0):
    stim = nengo.processes.WhiteSignal(period=period, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[pU]
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
    norm = value / np.max(np.abs(u))
    stim = np.ravel(sim.data[pU]) * norm
    pos = stim
    pos[pos<0] = 0
    return pos

def makeSignalShift(t=10.0, dt=0.001, period=10, f=Lowpass(0.05), seed=0):
    stim = nengo.processes.WhiteSignal(period=period, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    u = f.filt(sim.data[pU], dt=dt)
    norm = 1.0 / np.max(np.abs(u))
    stim = np.ravel(u) * norm
    mirrored = np.concatenate(([0], stim, -stim))[::2]
    shifted = 0.5 + 0.5*mirrored
    return shifted

def makeSignalSquare(t=10.0, dt=0.001, period=10, f=Lowpass(0.05), seed=0):
    stim = nengo.processes.WhiteSignal(period=period, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        pU = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[pU]
    u2 = f.filt(np.square(u))
    norm = 1.0 / np.max(np.abs(u2))
    stim = np.ravel(u2) * norm
    return stim

def goDale(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, dEnsInhAMPA, dEnsInhNMDA, dInhFdfw, dInhEns, fAMPA, fNMDA, fGABA, stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(0, 0.8), e=Choice([[1]]), lif=nengo.LIF(), seed=0):

    absv = lambda x: np.abs(x)
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        fdfw = nengo.Ensemble(N, 1, radius=2, seed=seed)
        inh = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        ens = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s)
        nengo.Connection(inpt, fdfw, synapse=None)
        nengo.Connection(fdfw, ens, synapse=fAMPA, solver=NoSolver(dFFAMPA))
        nengo.Connection(fdfw, ens, synapse=fNMDA, solver=NoSolver(dFFNMDA))
        nengo.Connection(ens, inh, synapse=fAMPA, solver=NoSolver(dEnsInhAMPA))
        nengo.Connection(ens, inh, synapse=fNMDA, solver=NoSolver(dEnsInhNMDA))
        nengo.Connection(ens, ens, synapse=fAMPA, solver=NoSolver(dFBAMPA))
        nengo.Connection(ens, ens, synapse=fNMDA, solver=NoSolver(dFBNMDA))
        nengo.Connection(inh, fdfw.neurons, synapse=fGABA, solver=NoSolver(dInhFdfw), transform=np.ones((N, 1)))
        nengo.Connection(inh, ens.neurons, synapse=fGABA, solver=NoSolver(dInhEns), transform=np.ones((N, 1)))
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        sim.run(t, progress_bar=True)
    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        fdfw=sim.data[pFdfw],
        inh=sim.data[pInh],
        ens=sim.data[pEns]   
    )

def medium():
    N = 50
    nTrains = 3
    dFFAMPA = np.zeros((N, 1))
    dFFNMDA = np.zeros((N, 1))
    dFBAMPA = np.zeros((N, 1))
    dFBNMDA = np.zeros((N, 1))
    fAMPA = DoubleExp(5.5e-4, 2.2e-3)
    fNMDA = DoubleExp(1e-2, 2.85e-1)
    fGABA = DoubleExp(5e-4, 1.5e-3)
    kAMPA = 0.2
    kNMDA = 0.7
    kGABA = 0.8
    tNMDA = 0.285
    rng = np.random.RandomState(seed=0)
    dEnsInhAMPA = rng.uniform(0, 5e-4, size=(N, 1))
    dEnsInhNMDA = rng.uniform(0, 5e-4, size=(N, 1))
    dInhFdfw = rng.uniform(-6e-4, 0, size=(N, 1))
    dInhEns = rng.uniform(-1e-4, 0, size=(N, 1))
    t = 10
    dt = 0.001

    # Stage 1 - feedforward decoders from fdfw to ens
    targetsAMPA = np.zeros((1, 1))
    targetsNMDA = np.zeros((1, 1))
    asAMPA = np.zeros((1, N))
    asNMDA = np.zeros((1, N))
    for n in range(nTrains):
        u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
        stim = lambda t: u[int(t/dt)]
        data = goDale(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, dEnsInhAMPA, dEnsInhNMDA, 0*dInhFdfw, 0*dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
        asAMPA = np.append(asAMPA, fAMPA.filt(data['fdfw']), axis=0)
        asNMDA = np.append(asNMDA, fNMDA.filt(data['fdfw']), axis=0)
        targetsAMPA = np.append(targetsAMPA, fAMPA.filt(data['inpt']), axis=0)
        targetsNMDA = np.append(targetsNMDA, fNMDA.filt(data['inpt']), axis=0)
#     dFFAMPA, _ = LstsqL2(reg=1e-2)(asAMPA, targetsAMPA)
#     dFFNMDA, _ = LstsqL2(reg=1e-2)(asNMDA, targetsNMDA)
    dFFAMPA, _ = nnls(asAMPA, np.ravel(targetsAMPA))
    dFFNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
    dFFAMPA = dFFAMPA.reshape((N, 1))
    dFFNMDA = dFFNMDA.reshape((N, 1))
    xhatFFAMPA = np.dot(asAMPA, dFFAMPA)
    xhatFFNMDA = np.dot(asNMDA, dFFNMDA)
    fig, ax = plt.subplots()
#     ax.plot(targetsAMPA, linestyle="--", label='target (AMPA)')
    ax.plot(targetsNMDA, linestyle="--", label='target (NMDA)')
#     ax.plot(xhatFFAMPA, alpha=0.5, label='fdfw (AMPA)')
    ax.plot(xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.legend()
    ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_fdfw.pdf")

    # Stage 2 - readout decoders for ens; assume high DA condition and inh-ens, and only get NMDA decoders
    targetsNMDA = np.zeros((1, 1))
    asNMDA = np.zeros((1, N))
    for n in range(nTrains):
        u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
        stim = lambda t: u[int(t/dt)]
        data = goDale(0*dFFAMPA, dFFNMDA, 0*dFBAMPA, dFBNMDA, 0*dEnsInhAMPA, dEnsInhNMDA, 0*dInhFdfw, dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
        asNMDA = np.append(asNMDA, fNMDA.filt(data['ens']), axis=0)
        targetsNMDA = np.append(targetsNMDA, fNMDA.filt(fNMDA.filt(data['inpt'])), axis=0)
#     dFBNMDA, _ = LstsqL2(reg=1e-2)(asNMDA, targetsNMDA)
    dFBNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
    dFBNMDA = dFBNMDA.reshape((N, 1))
    dFBAMPA = rng.uniform(0, 1e-5, size=(N, 1))  # negligable feedback
    xhatFBNMDA = np.dot(asNMDA, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.plot(targetsNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
    ax.legend()
    ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_fdbk.pdf")

    # Stage 3 - test integration in high vs low DA; assume inh-ens
    # high DA
    u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=0)
    stim = lambda t: u[int(t/dt)]
    data = goDale(kAMPA*dFFAMPA, tNMDA*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*dEnsInhAMPA, dEnsInhNMDA, 0*dInhFdfw, dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_intg_highDA.pdf")
    # low DA
    data = goDale(dFFAMPA, kNMDA*tNMDA*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, dEnsInhAMPA, kNMDA*dEnsInhNMDA, 0*dInhFdfw, kGABA*dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_intg_lowDA.pdf")

    # Stage 4 - test integration with inh-fdfw in high vs low DA
    data = goDale(kAMPA*dFFAMPA, tNMDA*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*dEnsInhAMPA, dEnsInhNMDA, dInhFdfw, dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfw = fNMDA.filt(data['fdfw'])
    target = fNMDA.filt(data['inpt'])
    xhatFF = np.dot(aFdfw, dFFNMDA)
    aEns = fNMDA.filt(data['ens'])
    targetIntg = fNMDA.filt(data['intg'])
    xhatIntg = np.dot(aEns, dFBNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_inh_highDA.pdf")
    # Low DA
    data = goDale(dFFAMPA, kNMDA*tNMDA*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, dEnsInhAMPA, kNMDA*dEnsInhNMDA, kGABA*dInhFdfw, kGABA*dInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfw = fNMDA.filt(data['fdfw'])
    target = fNMDA.filt(data['inpt'])
    xhatFF = np.dot(aFdfw, dFFNMDA)
    aEns = fNMDA.filt(data['ens'])
    targetIntg = fNMDA.filt(data['intg'])
    xhatIntg = np.dot(aEns, dFBNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goDale_inh_lowDA.pdf")

    
# medium()


def goBio(
    dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
    fAMPA, fNMDA, fGABA, fS,
    ePreFdfw=None, wPreFdfw=None,
    eFdfwEnsAMPA=None, wFdfwEnsAMPA=None, eFdfwEnsNMDA=None, wFdfwEnsNMDA=None, 
    eEnsEnsAMPA=None, wEnsEnsAMPA=None, eEnsEnsNMDA=None, wEnsEnsNMDA=None, 
    eEnsInhAMPA=None, wEnsInhAMPA=None, eEnsInhNMDA=None, wEnsInhNMDA=None, 
    wInhFdfw=None, wInhEns=None,
    stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(0, 0.8), e=Choice([[1]]), lif=nengo.LIF(), DA=lambda t: 0, stage=0, seed=0):

    absv = lambda x: np.abs(x)
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        pre = nengo.Ensemble(N, 1, radius=2, seed=seed)
        # bioneuron network
        fdfw = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=lambda t: 0), seed=seed)
        ens = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron( "Pyramidal", DA=DA), seed=seed)
        inh = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron( "Interneuron", DA=DA), seed=seed)
        nengo.Connection(inpt, intg, synapse=1/s)
        inptPre = nengo.Connection(inpt, pre, synapse=None)
        preFdfw = nengo.Connection(pre, fdfw, synapse=fAMPA)
        fdfwEnsAMPA = nengo.Connection(fdfw, ens, synapse=AMPA(), solver=NoSolver(dFdfwEnsAMPA))
        fdfwEnsNMDA = nengo.Connection(fdfw, ens, synapse=NMDA(), solver=NoSolver(dFdfwEnsNMDA))
        ensEnsAMPA = nengo.Connection(ens, ens, synapse=AMPA(), solver=NoSolver(dEnsEnsAMPA))
        ensEnsNMDA = nengo.Connection(ens, ens, synapse=NMDA(), solver=NoSolver(dEnsEnsNMDA))
        ensInhAMPA = nengo.Connection(ens, inh, synapse=AMPA(), solver=NoSolver(None))
        ensInhNMDA = nengo.Connection(ens, inh, synapse=NMDA(), solver=NoSolver(None))
        inhFdfw = nengo.Connection(inh, fdfw, synapse=GABA(), solver=NoSolver(None))
        inhEns = nengo.Connection(inh, ens, synapse=GABA(), solver=NoSolver(None))
        pInpt = nengo.Probe(inpt, synapse=None)
        pIntg = nengo.Probe(intg, synapse=None)
        pPre = nengo.Probe(pre.neurons, synapse=None)
        pFdfw = nengo.Probe(fdfw.neurons, synapse=None)
        pInh = nengo.Probe(inh.neurons, synapse=None)
        pEns = nengo.Probe(ens.neurons, synapse=None)
        # target network
        delayed = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        tarFdfw = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        tarEns = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        tarInh = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)
        pTarInh = nengo.Probe(tarInh.neurons, synapse=None)

        # Training
        if stage==0:
            # decoder training
            pass
        if stage==1:
            nengo.Connection(inpt, tarFdfw, synapse=fAMPA)
            node = WNode(preFdfw, alpha=3e-5)
            nengo.Connection(pre.neurons, node[0:N], synapse=fNMDA)
            nengo.Connection(fdfw.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(tarFdfw.neurons, node[2*N: 3*N], synapse=fS)
        if stage==2:
            nengo.Connection(inpt, tarEns, synapse=fAMPA)
            node = WNode(fdfwEnsAMPA, alpha=1e-3, exc=True)
            nengo.Connection(fdfw.neurons, node[0:N], synapse=fAMPA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(tarEns.neurons, node[2*N: 3*N], synapse=fS)
        if stage==3:
            nengo.Connection(inpt, tarEns, synapse=fNMDA)
            node = WNode(fdfwEnsNMDA, alpha=1e-6, exc=True)
            nengo.Connection(fdfw.neurons, node[0:N], synapse=fNMDA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(tarEns.neurons, node[2*N: 3*N], synapse=fS)
        if stage==4:
            pre2 = nengo.Ensemble(N, 1, radius=2, seed=seed)
            fdbk = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=DA), seed=seed)
            pFdbk = nengo.Probe(fdbk.neurons, synapse=None)
            nengo.Connection(intg, pre2, synapse=None)
            pre2Fdbk = nengo.Connection(pre2, fdbk, synapse=fAMPA)
            inptPre.synapse = fNMDA
#             nengo.Connection(inpt, delayed, synapse=fNMDA)
#             nengo.Connection(delayed, tarEns, synapse=fNMDA)
#             nengo.Connection(delayed, tarInh, synapse=fGABA)
            fdbkEnsAMPA = nengo.Connection(fdbk, ens, synapse=AMPA(), solver=NoSolver(dEnsEnsAMPA))
            fdbkEnsNMDA = nengo.Connection(fdbk, ens, synapse=NMDA(), solver=NoSolver(dEnsEnsNMDA))
            node = WNode(fdbkEnsAMPA, alpha=1e-9, exc=True)
            nengo.Connection(fdbk.neurons, node[0:N], synapse=fAMPA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(fdbk.neurons, node[2*N: 3*N], synapse=fS)
            node2 = WNode(fdbkEnsNMDA, alpha=3e-8, exc=True)
            nengo.Connection(fdbk.neurons, node2[0:N], synapse=fNMDA)
            nengo.Connection(ens.neurons, node2[N:2*N], synapse=fS)
            nengo.Connection(fdbk.neurons, node2[2*N: 3*N], synapse=fS)
#             node5 = WNode(ensInhAMPA, alpha=1e-5, exc=True)
#             nengo.Connection(ens.neurons, node5[0:N], synapse=fAMPA)
#             nengo.Connection(inh.neurons, node5[N:2*N], synapse=fS)
#             nengo.Connection(tarInh.neurons, node5[2*N: 3*N], synapse=fS)
#             node6 = WNode(ensInhNMDA, alpha=1e-7, exc=True)
#             nengo.Connection(ens.neurons, node6[0:N], synapse=fNMDA)
#             nengo.Connection(inh.neurons, node6[N:2*N], synapse=fS)
#             nengo.Connection(tarInh.neurons, node6[2*N: 3*N], synapse=fS)

    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        # Weight setting
        for pre in range(N):
            for post in range(N):
                if np.any(wPreFdfw):
                    preFdfw.weights[pre, post] = wPreFdfw[pre, post]
                    preFdfw.netcons[pre, post].weight[0] = np.abs(wPreFdfw[pre, post])
                    preFdfw.netcons[pre, post].syn().e = 0 if wPreFdfw[pre, post] > 0 else -70
                    if stage==4:
                        pre2Fdbk.weights[pre, post] = wPreFdfw[pre, post]
                        pre2Fdbk.netcons[pre, post].weight[0] = np.abs(wPreFdfw[pre, post])
                        pre2Fdbk.netcons[pre, post].syn().e = 0 if wPreFdfw[pre, post] > 0 else -70
                if np.any(wFdfwEnsAMPA):
                    fdfwEnsAMPA.weights[pre, post] = wFdfwEnsAMPA[pre, post]
                    fdfwEnsAMPA.netcons[pre, post].weight[0] = np.abs(wFdfwEnsAMPA[pre, post])
                    fdfwEnsAMPA.netcons[pre, post].syn().e = 0 if wFdfwEnsAMPA[pre, post] > 0 else -70
                    fdfwEnsNMDA.weights[pre, post] = wFdfwEnsNMDA[pre, post]
                    fdfwEnsNMDA.netcons[pre, post].weight[0] = np.abs(wFdfwEnsNMDA[pre, post])
                    fdfwEnsNMDA.netcons[pre, post].syn().e = 0 if wFdfwEnsNMDA[pre, post] > 0 else -70
                if np.any(wEnsEnsAMPA):
                    if stage==4:
                        fdbkEnsAMPA.weights[pre, post] = wEnsEnsAMPA[pre, post]
                        fdbkEnsAMPA.netcons[pre, post].weight[0] = np.abs(wEnsEnsAMPA[pre, post])
                        fdbkEnsAMPA.netcons[pre, post].syn().e = 0 if wEnsEnsAMPA[pre, post] > 0 else -70
                        fdbkEnsNMDA.weights[pre, post] = wEnsEnsNMDA[pre, post]
                        fdbkEnsNMDA.netcons[pre, post].weight[0] = np.abs(wEnsEnsNMDA[pre, post])
                        fdbkEnsNMDA.netcons[pre, post].syn().e = 0 if wEnsEnsNMDA[pre, post] > 0 else -70
                    else:
                        ensEnsAMPA.weights[pre, post] = wEnsEnsAMPA[pre, post]
                        ensEnsAMPA.netcons[pre, post].weight[0] = np.abs(wEnsEnsAMPA[pre, post])
                        ensEnsAMPA.netcons[pre, post].syn().e = 0 if wEnsEnsAMPA[pre, post] > 0 else -70
                        ensEnsNMDA.weights[pre, post] = wEnsEnsNMDA[pre, post]
                        ensEnsNMDA.netcons[pre, post].weight[0] = np.abs(wEnsEnsNMDA[pre, post])
                        ensEnsNMDA.netcons[pre, post].syn().e = 0 if wEnsEnsNMDA[pre, post] > 0 else -70
                if np.any(wEnsInhAMPA):
                    ensInhAMPA.weights[pre, post] = wEnsInhAMPA[pre, post]
                    ensInhAMPA.netcons[pre, post].weight[0] = np.abs(wEnsInhAMPA[pre, post])
                    ensInhAMPA.netcons[pre, post].syn().e = 0 if wEnsInhAMPA[pre, post] > 0 else -70
                    ensInhNMDA.weights[pre, post] = wEnsInhNMDA[pre, post]
                    ensInhNMDA.netcons[pre, post].weight[0] = np.abs(wEnsInhNMDA[pre, post])
                    ensInhNMDA.netcons[pre, post].syn().e = 0 if wEnsInhNMDA[pre, post] > 0 else -70
                if np.any(wInhFdfw):
                    inhFdfw.weights[pre, post] = wInhFdfw[pre, post]
                    inhFdfw.netcons[pre, post].weight[0] = np.abs(wInhFdfw[pre, post])
                    inhFdfw.netcons[pre, post].syn().e = 0 if wInhFdfw[pre, post] > 0 else -70
                if np.any(wInhEns):
                    inhEns.weights[pre, post] = wInhEns[pre, post]
                    inhEns.netcons[pre, post].weight[0] = np.abs(wInhEns[pre, post])
                    inhEns.netcons[pre, post].syn().e = 0 if wInhEns[pre, post] > 0 else -70
                   
        # Encoder setting
        if stage==1:
            if np.any(ePreFdfw): preFdfw.e = ePreFdfw
        if stage==2:
            if np.any(eFdfwEnsAMPA): fdfwEnsAMPA.e = eFdfwEnsAMPA
        if stage==3:
            if np.any(eFdfwEnsNMDA): fdfwEnsNMDA.e = eFdfwEnsNMDA
        if stage==4:
            if np.any(eEnsEnsAMPA): fdbkEnsAMPA.e = eEnsEnsAMPA
            if np.any(eEnsEnsNMDA): fdbkEnsNMDA.e = eEnsEnsNMDA
#             if np.any(eEnsInhAMPA): ensInhAMPA.e = eEnsInhAMPA
#             if np.any(eEnsInhNMDA): ensInhNMDA.e = eEnsInhNMDA
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 
    if stage==1:
        ePreFdfw = preFdfw.e
        wPreFdfw = preFdfw.weights
    if stage==2:
        eFdfwEnsAMPA = fdfwEnsAMPA.e
        wFdfwEnsAMPA = fdfwEnsAMPA.weights
    if stage==3:
        eFdfwEnsNMDA = fdfwEnsNMDA.e
        wFdfwEnsNMDA = fdfwEnsNMDA.weights
    if stage==4:
        eEnsEnsAMPA = fdbkEnsAMPA.e
        wEnsEnsAMPA = fdbkEnsAMPA.weights
        eEnsEnsNMDA = fdbkEnsNMDA.e
        wEnsEnsNMDA = fdbkEnsNMDA.weights
#         eEnsInhAMPA = ensInhAMPA.e
#         wEnsInhAMPA = ensInhAMPA.weights
#         eEnsInhNMDA = ensInhNMDA.e
#         wEnsInhNMDA = ensInhNMDA.weights
    return dict(
        times=sim.trange(),
        inpt=sim.data[pInpt],
        intg=sim.data[pIntg],
        fdfw=sim.data[pFdfw],
        ens=sim.data[pEns],
        fdbk=sim.data[pFdbk] if stage==4 else None,
        inh=sim.data[pInh],
        tarFdfw=sim.data[pTarFdfw],
        tarEns=sim.data[pTarEns],
        tarInh=sim.data[pTarInh],
        ePreFdfw=ePreFdfw,
        wPreFdfw=wPreFdfw,
        eFdfwEnsAMPA=eFdfwEnsAMPA,
        wFdfwEnsAMPA=wFdfwEnsAMPA,
        eFdfwEnsNMDA=eFdfwEnsNMDA,
        wFdfwEnsNMDA=wFdfwEnsNMDA,
        eEnsEnsAMPA=eEnsEnsAMPA,
        wEnsEnsAMPA=wEnsEnsAMPA,
        eEnsEnsNMDA=eEnsEnsNMDA,
        wEnsEnsNMDA=wEnsEnsNMDA,
        eEnsInhAMPA=eEnsInhAMPA,
        wEnsInhAMPA=wEnsInhAMPA,
        eEnsInhNMDA=eEnsInhNMDA,
        wEnsInhNMDA=wEnsInhNMDA,
    )

def hard(load=False, dataFile="data/gatedMemory.npz"):
    N = 30
    nTrains = 10
    dFdfwEnsAMPA = np.zeros((N, 1))
    dFdfwEnsNMDA = np.zeros((N, 1))
    dEnsEnsAMPA = np.zeros((N, 1))
    dEnsEnsNMDA = np.zeros((N, 1))
#     dEnsInhAMPA = np.zeros((N, 1))
#     dEnsInhNMDA = np.zeros((N, 1))
    fAMPA = DoubleExp(5.5e-4, 2.2e-3)
    fNMDA = DoubleExp(1e-2, 2.85e-1)
    fGABA = DoubleExp(5e-4, 1.5e-3)
    fS = Lowpass(2e-1)
    rng = np.random.RandomState(seed=0)
    wEnsInhAMPA = rng.uniform(0, 3e-7, size=(N, N))
    wEnsInhNMDA = rng.uniform(0, 6e-6, size=(N, N))
    wInhFdfw = rng.uniform(-1e-2, 0, size=(N, N))
    wInhEns = rng.uniform(-1e-4, 0, size=(N, N))
    tNMDA = 0.285
    t = 10
    dt = 0.001
    
    # Mixed encoders from pre to fdfw
    if load:
        ePreFdfw = np.load(dataFile)['ePreFdfw']
        wPreFdfw = np.load(dataFile)['wPreFdfw']
    else:
        ePreFdfw = None
        for n in range(nTrains):
            print("pre-fdfw encoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                ePreFdfw=ePreFdfw,
                N=N, stim=stim, t=t, dt=dt, stage=1)
            ePreFdfw = data['ePreFdfw']
            wPreFdfw = data['wPreFdfw']
            np.savez('data/gatedMemory.npz',
                ePreFdfw=ePreFdfw,
                wPreFdfw=wPreFdfw)
            aTar = fS.filt(data['tarFdfw'])
            aFdfw = fS.filt(data['fdfw'])
            for n in range(N):
                fig, ax = plt.subplots(1, 1)
                ax.plot(data['times'], aTar[:,n], alpha=0.5, label='target')
                ax.plot(data['times'], aFdfw[:,n], alpha=0.5, label='fdfw')
                ax.set(ylim=((0, 60)))
                plt.legend()
                plt.savefig('plots/tuning/gatedMemroy_ePreFdfw_%s.pdf'%n)
                plt.close('all')

    # Positive decoders from fdfw to ens
    if load:
        dFdfwEnsAMPA = np.load(dataFile)['dFdfwEnsAMPA']
        dFdfwEnsNMDA = np.load(dataFile)['dFdfwEnsNMDA']
    else:
        targetsAMPA = np.zeros((1, 1))
        targetsNMDA = np.zeros((1, 1))
        asAMPA = np.zeros((1, N))
        asNMDA = np.zeros((1, N))
        for n in range(3):
            print("fdfw decoder trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw,
                N=N, stim=stim, t=t, dt=dt, stage=0)
            asAMPA = np.append(asAMPA, fAMPA.filt(data['fdfw']), axis=0)
            asNMDA = np.append(asNMDA, fNMDA.filt(data['fdfw']), axis=0)
            targetsAMPA = np.append(targetsAMPA, fAMPA.filt(data['inpt']), axis=0)
            targetsNMDA = np.append(targetsNMDA, fNMDA.filt(data['inpt']), axis=0)
        dFdfwEnsAMPA, _ = nnls(asAMPA, np.ravel(targetsAMPA))
        dFdfwEnsNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
        dFdfwEnsAMPA = dFdfwEnsAMPA.reshape((N, 1))
        dFdfwEnsNMDA = dFdfwEnsNMDA.reshape((N, 1))
        np.savez('data/gatedMemory.npz',
            ePreFdfw=ePreFdfw,
            wPreFdfw=wPreFdfw,
            dFdfwEnsAMPA=dFdfwEnsAMPA,
            dFdfwEnsNMDA=dFdfwEnsNMDA)
        xhatFFAMPA = np.dot(asAMPA, dFdfwEnsAMPA)
        xhatFFNMDA = np.dot(asNMDA, dFdfwEnsNMDA)
        fig, ax = plt.subplots()
        ax.plot(targetsNMDA, linestyle="--", label='target (NMDA)')
        ax.plot(xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
        ax.legend()
        ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
        fig.savefig("plots/gatedMemory_goBio_fdfw.pdf")

    # Optimize encoders from fdfw to ens with low DA
    if load:
        eFdfwEnsAMPA = np.load(dataFile)['eFdfwEnsAMPA']
        wFdfwEnsAMPA = np.load(dataFile)['wFdfwEnsAMPA']
    else:
        eFdfwEnsAMPA = None
        for n in range(nTrains):
            print("fdfw-ens AMPA encoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, eFdfwEnsAMPA=eFdfwEnsAMPA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, stim=stim, t=t, dt=dt, stage=2, DA=lambda t: 0)
            eFdfwEnsAMPA = data['eFdfwEnsAMPA']
            wFdfwEnsAMPA = data['wFdfwEnsAMPA']
            np.savez('data/gatedMemory.npz',
                ePreFdfw=ePreFdfw,
                wPreFdfw=wPreFdfw,
                dFdfwEnsAMPA=dFdfwEnsAMPA,
                dFdfwEnsNMDA=dFdfwEnsNMDA,
                eFdfwEnsAMPA=eFdfwEnsAMPA,
                wFdfwEnsAMPA=wFdfwEnsAMPA)
            aTar = fS.filt(data['tarEns'])
            aEns = fS.filt(data['ens'])
            for n in range(N):
                fig, ax = plt.subplots(1, 1)
                ax.plot(data['times'], aTar[:,n], alpha=0.5, label='target')
                ax.plot(data['times'], aEns[:,n], alpha=0.5, label='ens')
                ax.set(ylim=((0, 60)))
                plt.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdfwEnsAMPA_%s.pdf'%n)
                plt.close('all')

    # Optimize encoders from fdfw to ens for high DA
    if load:
        eFdfwEnsNMDA = np.load(dataFile)['eFdfwEnsNMDA']
        wFdfwEnsNMDA = np.load(dataFile)['wFdfwEnsNMDA']
    else:
        eFdfwEnsNMDA = None
        for n in range(nTrains):
            print("fdfw-ens encoding NMDA trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, eFdfwEnsNMDA=eFdfwEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, stim=stim, t=t, dt=dt, stage=3, DA=lambda t: 1)
            eFdfwEnsNMDA = data['eFdfwEnsNMDA']
            wFdfwEnsNMDA = data['wFdfwEnsNMDA']
            np.savez('data/gatedMemory.npz',
                ePreFdfw=ePreFdfw,
                wPreFdfw=wPreFdfw,
                dFdfwEnsAMPA=dFdfwEnsAMPA,
                dFdfwEnsNMDA=dFdfwEnsNMDA,
                eFdfwEnsAMPA=eFdfwEnsAMPA,
                wFdfwEnsAMPA=wFdfwEnsAMPA,
                eFdfwEnsNMDA=eFdfwEnsNMDA,
                wFdfwEnsNMDA=wFdfwEnsNMDA)
            aTar = fS.filt(data['tarEns'])
            aEns = fS.filt(data['ens'])
            for n in range(N):
                fig, ax = plt.subplots(1, 1)
                ax.plot(data['times'], aTar[:,n], alpha=0.5, label='target')
                ax.plot(data['times'], aEns[:,n], alpha=0.5, label='ens')
                ax.set(ylim=((0, 60)))
                plt.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdfwEnsNMDA_%s.pdf'%n)
                plt.close('all')

    # Readout decoders for ens; assume high DA condition and inh-ens, and only get NMDA decoders
    if load:
        dEnsEnsAMPA = rng.uniform(0, 1e-7, size=(N, 1))
        dEnsEnsNMDA = np.load(dataFile)['dEnsEnsNMDA']
        dEnsInhAMPA = np.array(dEnsEnsAMPA)
        dEnsInhNMDA = np.array(dEnsEnsNMDA)
    else:
        targetsNMDA = np.zeros((1, 1))
        asNMDA = np.zeros((1, N))
        for n in range(3):
            print("ens decoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=wFdfwEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
            asNMDA = np.append(asNMDA, fNMDA.filt(data['ens']), axis=0)
            targetsNMDA = np.append(targetsNMDA, fNMDA.filt(fNMDA.filt(data['inpt'])), axis=0)
        dEnsEnsAMPA = rng.uniform(0, 1e-6, size=(N, 1))  # negligable feedback
#         dEnsInhAMPA = np.array(dEnsEnsAMPA)
        dEnsEnsNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
        dEnsEnsNMDA = dEnsEnsNMDA.reshape((N, 1))
#         dEnsInhNMDA = np.array(dEnsEnsNMDA)
        np.savez('data/gatedMemory.npz',
            ePreFdfw=ePreFdfw,
            wPreFdfw=wPreFdfw,
            dFdfwEnsAMPA=dFdfwEnsAMPA,
            dFdfwEnsNMDA=dFdfwEnsNMDA,
            eFdfwEnsAMPA=eFdfwEnsAMPA,
            wFdfwEnsAMPA=wFdfwEnsAMPA,
            eFdfwEnsNMDA=eFdfwEnsNMDA,
            wFdfwEnsNMDA=wFdfwEnsNMDA,
            dEnsEnsNMDA=dEnsEnsNMDA,
            dEnsEnsAMPA=dEnsEnsAMPA)
#             dEnsInhAMPA=dEnsInhAMPA,
#             dEnsInhNMDA=dEnsInhNMDA)
        xhatFBNMDA = np.dot(asNMDA, dEnsEnsNMDA)
        fig, ax = plt.subplots()
        ax.plot(targetsNMDA, linestyle="--", label='target (NMDA)')
        ax.plot(xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
        ax.legend()
        ax.set(ylim=((0, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
        fig.savefig("plots/gatedMemory_goBio_fdbk.pdf")
    
    # Encoders for fdbk-ens
    if load:
#         eEnsEnsAMPA = np.load(dataFile)['eEnsEnsAMPA']
#         wEnsEnsAMPA = np.load(dataFile)['wEnsEnsAMPA']
#         eEnsEnsNMDA = np.load(dataFile)['eEnsEnsNMDA']
#         wEnsEnsNMDA = np.load(dataFile)['wEnsEnsNMDA']

#         eEnsInhAMPA = np.load(dataFile)['eEnsInhAMPA']
#         wEnsInhAMPA = np.load(dataFile)['wEnsInhAMPA']
#         eEnsInhNMDA = np.load(dataFile)['eEnsInhNMDA']
#         wEnsInhNMDA = np.load(dataFile)['wEnsInhNMDA']
#     else:
        eEnsEnsAMPA = None
        wEnsEnsAMPA = None
        eEnsEnsNMDA = None
        wEnsEnsNMDA = None
#         eEnsInhAMPA = None
#         wEnsInhAMPA = None
#         eEnsInhNMDA = None
#         wEnsInhNMDA = None
        for n in range(nTrains):
            print("ens-ens2 and ens-inh encoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n, nFilts=1)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, eEnsEnsAMPA=eEnsEnsAMPA, eEnsEnsNMDA=eEnsEnsNMDA, # eEnsInhAMPA=eEnsInhAMPA, eEnsInhNMDA=eEnsInhNMDA, 
                N=N, stim=stim, t=t, dt=dt, stage=4, DA=lambda t: 1)
            eEnsEnsAMPA = data['eEnsEnsAMPA']
            wEnsEnsAMPA = data['wEnsEnsAMPA']
            eEnsEnsNMDA = data['eEnsEnsNMDA']
            wEnsEnsNMDA = data['wEnsEnsNMDA']
#             eEnsInhAMPA = data['eEnsInhAMPA']
#             wEnsInhAMPA = data['wEnsInhAMPA']
#             eEnsInhNMDA = data['eEnsInhNMDA']
#             wEnsInhNMDA = data['wEnsInhNMDA']
            np.savez('data/gatedMemory.npz',
                ePreFdfw=ePreFdfw,
                wPreFdfw=wPreFdfw,
                dFdfwEnsAMPA=dFdfwEnsAMPA,
                dFdfwEnsNMDA=dFdfwEnsNMDA,
                eFdfwEnsAMPA=eFdfwEnsAMPA,
                wFdfwEnsAMPA=wFdfwEnsAMPA,
                eFdfwEnsNMDA=eFdfwEnsNMDA,
                wFdfwEnsNMDA=wFdfwEnsNMDA,
                dEnsEnsNMDA=dEnsEnsNMDA,
                dEnsEnsAMPA=dEnsEnsAMPA,
#                 dEnsInhAMPA=dEnsInhAMPA,
#                 dEnsInhNMDA=dEnsInhNMDA,
                eEnsEnsAMPA=eEnsEnsAMPA,
                wEnsEnsAMPA=wEnsEnsAMPA,
                eEnsEnsNMDA=eEnsEnsNMDA,
                wEnsEnsNMDA=wEnsEnsNMDA,
#                 eEnsInhAMPA=eEnsInhAMPA,
                wEnsInhAMPA=wEnsInhAMPA,
#                 eEnsInhNMDA=eEnsInhNMDA,
                wEnsInhNMDA=wEnsInhNMDA)
            aTarEns = fS.filt(data['fdbk'])
            aEns = fS.filt(data['ens'])
            aInh = fS.filt(data['inh'])
            for n in range(N):
                fig, ax = plt.subplots(ncols=1, nrows=1)
                ax.plot(data['times'], aTarEns[:,n], alpha=0.5, label='target')
                ax.plot(data['times'], aEns[:,n], alpha=0.5, label='ens')
                ax.plot(data['times'], aInh[:,n], alpha=0.25, label='inh')
                ax.set(ylim=((0, 60)))
                ax.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdbkEns_%s.pdf'%n)
                plt.close('all')
    
    # Test integration in high vs low DA; assume inh-ens and no inh-fdfw
    print("integration test, high DA")
    u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=0)
    stim = lambda t: u[int(t/dt)]
    data = goBio(
        dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
        fAMPA, fNMDA, fGABA, fS,
        wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, #wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA,
        N=N, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goBio_intg_highDA.pdf")
    
    print("integration test, low DA")
    u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=0)
    stim = lambda t: u[int(t/dt)]
    data = goBio(
        dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
        fAMPA, fNMDA, fGABA, fS,
        wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA,wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, # wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA,
        N=N, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 0)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
    fig, ax = plt.subplots()
#     ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goBio_intg_lowDA.pdf")
    
    # Test integration in with inh-fdfw and high vs low DA
    print("inhibition test, high DA")
    u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=0)
    stim = lambda t: u[int(t/dt)]
    data = goBio(
        # todo: make fdfw bio so that GABA() can be applied to inh-fdfw
        dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA, 
        fAMPA, fNMDA, fGABA, fS,
        wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, wInhFdfw=wInhFdfw,
        N=N, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], targetNMDA, linestyle="--", label='inpt')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goBio_inh_highDA.pdf")
    
    print("inhibition test, low DA")
    u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=0)
    stim = lambda t: u[int(t/dt)]
    data = goBio(
        dFdfwEnsAMPA, dFdfwEnsNMDA, dEnsEnsAMPA, dEnsEnsNMDA,# dEnsInhAMPA, dEnsInhNMDA,
        fAMPA, fNMDA, fGABA, fS,
        wPreFdfw=wPreFdfw, wFdfwEnsAMPA=wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, wInhFdfw=wInhFdfw,
        N=N, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 0)
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetNMDA = fNMDA.filt(data['inpt'])
    xhatFFNMDA = np.dot(aFdfwNMDA, dFdfwEnsNMDA)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetIntgNMDA = fNMDA.filt(data['intg'])
    xhatIntgNMDA = np.dot(aEnsNMDA, dEnsEnsNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], targetNMDA, linestyle="--", label='inpt')
    ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw')
    ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral')
    ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens')
    ax.legend()
    ax.set(ylim=((0, 2)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goBio_inh_lowDA.pdf")

hard(load=True)
