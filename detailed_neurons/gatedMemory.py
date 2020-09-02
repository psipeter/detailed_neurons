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

# t = 10
# dt = 0.001
# s = makeSignal(t=t, dt=dt, period=t)
# stim = lambda t: s[int(t/dt)]
# gating = lambda t: 0 if (0<t<2.5 or 5<t<7.5) else 1
# data = goTarget(t=t, stim=stim, gating=gating)

# fig, ax = plt.subplots()
# ax.plot(data['times'], data['inpt'], linestyle="--", label='inpt')
# ax.plot(data['times'], data['gate'], linestyle="--", label='gate')
# ax.plot(data['times'], data['fdfw'], alpha=0.5, label='fdfw')
# ax.plot(data['times'], data['fdbk'], alpha=0.5, label='fcbk')
# ax.plot(data['times'], data['ens'], alpha=0.5, label='ens')
# ax.legend()
# ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
# fig.savefig("plots/gatedMemory_goTarget.pdf")


def goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(-1, 0.8), seed=0):

    absv = lambda x: np.abs(x)
    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        fdfw = nengo.Ensemble(N, 1, seed=seed)
        inh = nengo.Ensemble(N, 1, encoders=Choice([[1]]), intercepts=Uniform(0.2, 1), neuron_type=nengo.LIF(), seed=seed)
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

N = 100
dFFAMPA = np.zeros((N, 1))
dFFNMDA = np.zeros((N, 1))
dFBAMPA = np.zeros((N, 1))
dFBNMDA = np.zeros((N, 1))
fAMPA = DoubleExp(5.5e-4, 2.2e-3)
fNMDA = DoubleExp(1e-2, 2.85e-1)
fGABA = DoubleExp(5e-4, 1.5e-3)
rng = np.random.RandomState(seed=0)
wEnsInhAMPA = 1.5
wEnsInhNMDA = 1.5
wInhFdfw = rng.uniform(-1e-4, 0, size=(N, N))
wInhEns = rng.uniform(-4e-5, 0, size=(N, N))
t = 10
dt = 0.001
u = makeSignal(t=t, dt=dt, period=t, f=fNMDA)
stim = lambda t: u[int(t/dt)]

# Stage 1 - feedforward decoders from fdfw to ens

data = goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)

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
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_fdfw.pdf")

# Stage 2 - readout decoders for ens
# assume ens excited inh and inh inhibits ens, but that inh does not inhibit fdfw
# produce separate FB decoders for AMPA and NMDA, assuming just those connections are fdfw

data = goLIF(dFFAMPA, 0*dFFNMDA, dFBAMPA, 0*dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aEnsAMPA = fAMPA.filt(data['ens'])
targetAMPA = fAMPA.filt(fAMPA.filt(data['inpt']))
dFBAMPA, _ = LstsqL2(reg=1e-2)(aEnsAMPA, targetAMPA)
xhatFBAMPA = np.dot(aEnsAMPA, dFBAMPA)
fig, ax = plt.subplots()
ax.plot(data['times'], targetAMPA, linestyle="--", label='target (AMPA)')
ax.plot(data['times'], xhatFBAMPA, alpha=0.5, label='ens (AMPA)')
# ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIFAMPA_ens.pdf")

data = goLIF(0*dFFAMPA, dFFNMDA, 0*dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aEnsNMDA = fNMDA.filt(data['ens'])
targetNMDA = fNMDA.filt(fNMDA.filt(data['inpt']))
dFBNMDA, _ = LstsqL2(reg=1e-2)(aEnsNMDA, targetNMDA)
xhatFBNMDA = np.dot(aEnsNMDA, dFBNMDA)
fig, ax = plt.subplots()
ax.plot(data['times'], targetNMDA, linestyle="--", label='target (NMDA)')
ax.plot(data['times'], xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
# ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIFNMDA_ens.pdf")

# dFFAMPA *= 0.0022
dFFNMDA *= 0.285

# Stage 3 - test integration without inh inhibiting fdfw
# assume high DA condition: (1) normal NMDA weights (2) normal GABA weights (3) reduced AMPA weights
data = goLIF(0*dFFAMPA, dFFNMDA, 0*dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aFdfwNMDA = fNMDA.filt(data['fdfw'])
targetNMDA = fNMDA.filt(data['inpt'])
xhatFFNMDA = np.dot(aFdfwNMDA, dFFNMDA)
aEnsNMDA = fNMDA.filt(data['ens'])
targetIntgNMDA = fNMDA.filt(data['intg'])
xhatIntgNMDA = np.dot(aEnsNMDA, dFBNMDA)
fig, ax = plt.subplots()
ax.plot(data['times'], targetNMDA, linestyle="--", label='input (NMDA)')
ax.plot(data['times'], xhatFFNMDA, alpha=0.5, label='fdfw (NMDA)')
ax.plot(data['times'], targetIntgNMDA, linestyle="--", label='integral (NMDA)')
ax.plot(data['times'], xhatIntgNMDA, alpha=0.5, label='ens (NMDA)')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_intgNMDA.pdf")

# dFFAMPA *= 0.0022
data = goLIF(dFFAMPA, 0*dFFNMDA, dFBAMPA, 0*dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aFdfwAMPA = fAMPA.filt(data['fdfw'])
targetAMPA = fAMPA.filt(data['inpt'])
xhatFFAMPA = np.dot(aFdfwAMPA, dFFAMPA)
aEnsAMPA = fAMPA.filt(data['ens'])
targetIntgAMPA = fAMPA.filt(data['intg'])
xhatIntgAMPA = np.dot(aEnsAMPA, dFBAMPA)
fig, ax = plt.subplots()
ax.plot(data['times'], targetAMPA, linestyle="--", label='input (AMPA)')
ax.plot(data['times'], xhatFFAMPA, alpha=0.5, label='fdfw (AMPA)')
ax.plot(data['times'], targetIntgAMPA, linestyle="--", label='integral (AMPA)')
ax.plot(data['times'], xhatIntgAMPA, alpha=0.5, label='ens (AMPA)')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_intgAMPA.pdf")


# Stage 4 - test integration with inh inhibiting fdfw
# High DA
data = goLIF(0.5*dFFAMPA, dFFNMDA, 0.5*dFBAMPA, dFBNMDA, 0.5*wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aFdfw = fNMDA.filt(data['fdfw'])
target = fNMDA.filt(data['inpt'])
xhatFF = np.dot(aFdfw, dFFNMDA)
aEns = fNMDA.filt(data['ens'])
targetIntg = fNMDA.filt(data['intg'])
xhatIntg = np.dot(aEns, dFBNMDA)
fig, ax = plt.subplots()
ax.plot(data['times'], target, linestyle="--", label='input')
ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
# ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_highDA.pdf")

# Low DA
data = goLIF(dFFAMPA, 0.5*dFFNMDA, dFBAMPA, 0.5*dFBNMDA, wEnsInhAMPA, 0.5*wEnsInhNMDA, 0.5*wInhFdfw, 0.5*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
# aFdfwAMPA = fAMPA.filt(data['fdfw'])
# targetAMPA = fAMPA.filt(data['inpt'])
# xhatFFAMPA = np.dot(aFdfwAMPA, dFFAMPA)
# aEnsAMPA = fAMPA.filt(data['ens'])
# targetIntgAMPA = fAMPA.filt(data['intg'])
# xhatIntgAMPA = np.dot(aEnsAMPA, dFBAMPA)
aFdfw = fNMDA.filt(data['fdfw'])
target = fNMDA.filt(data['inpt'])
xhatFF = np.dot(aFdfw, dFFNMDA)
aEns = fNMDA.filt(data['ens'])
targetIntg = fNMDA.filt(data['intg'])
xhatIntg = np.dot(aEns, dFBNMDA)
fig, ax = plt.subplots()
ax.plot(data['times'], target, linestyle="--", label='input')
ax.plot(data['times'], xhatFF, alpha=0.5, label='fdfw')
ax.plot(data['times'], targetIntg, linestyle="--", label='integral')
ax.plot(data['times'], xhatIntg, alpha=0.5, label='ens')
# ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_lowDA.pdf")