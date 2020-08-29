import numpy as np
from scipy.optimize import nnls
import nengo
from nengo.params import Default
from nengo.dists import Uniform
from nengo.solvers import NoSolver
from nengo.utils.numpy import rmse
from nengolib import Lowpass, DoubleExp
from nengolib.signal import s, z, nrmse, LinearSystem
from train import WNode
from neuron_models import LIF, AdaptiveLIFT, WilsonEuler, BioNeuron, reset_neuron, AMPA, GABA, NMDA
import neuron
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white')

def make_signal(t=10.0, dt=0.001, f=Lowpass(0.01), seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        p_u = nengo.Probe(u, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    u = f.filt(sim.data[p_u], dt=dt)
    norm = 1.0 / np.max(np.abs(u))
    stim = np.ravel(u) * norm
    return np.concatenate(([0], stim, -stim))[::2]

def integrate(stim, dt=0.001):
    t = len(stim) * dt - dt
    with nengo.Network() as model:
        u = nengo.Node(lambda t: stim[int(t/dt)])
        p_x = nengo.Probe(u, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    x = np.concatenate(([0], np.ravel(sim.data[p_x])))
#     x = np.ravel(sim.data[p_x])
    return x

def partialIntegrate(stim, funcFF, funcFB, dt=0.001):
    result = np.zeros_like(stim)
    result[0] = stim[0]
    for t in range(stim.shape[0]-1):
        result[t+1] = funcFB(t*dt)*result[t] + funcFF(t*dt)*stim[t]
    return result

def gatedIntegrate(t, stim, DA, dt=0.001, tauFF=0.01, tauFB=0.25):
    with nengo.Network() as network:
        u = nengo.Node(stim)
        dopamine = nengo.Node(DA)
        gate = nengo.Ensemble(100, 1, seed=0)
        buffer = nengo.Ensemble(100, 1, seed=0)
        feedback = nengo.Ensemble(100, 1, seed=0)
        nengo.Connection(u, gate, synapse=None)
        nengo.Connection(gate, buffer, synapse=tauFF)  # , transform=tauFB
        nengo.Connection(buffer, feedback, synapse=tauFB)
        nengo.Connection(feedback, buffer, synapse=tauFF)
        nengo.Connection(dopamine, gate.neurons, transform=-2*np.ones((100, 1)), function=lambda x: x)
        nengo.Connection(dopamine, feedback.neurons, transform=-2*np.ones((100, 1)), function=lambda x: 1-x)
        p_da = nengo.Probe(dopamine, synapse=None)
        p_gate = nengo.Probe(gate, synapse=tauFF)
        p_buffer = nengo.Probe(buffer, synapse=tauFB)
        p_feedback = nengo.Probe(feedback, synapse=tauFB)
    with nengo.Simulator(network) as sim:
        sim.run(t)
    fig, ax = plt.subplots()
    ax.plot(sim.trange(), sim.data[p_da], label="dopamine")
    ax.plot(sim.trange(), sim.data[p_gate], label="gate")
    ax.plot(sim.trange(), sim.data[p_buffer], label="buffer")
    ax.plot(sim.trange(), sim.data[p_feedback], label="feedback")
    ax.legend()
    fig.savefig("plots/gatedIntegrator.pdf")

# tDA = 10
# DA = lambda t: t/tDA
# gatedIntegrate(t=10, stim=lambda t: np.sin(2*t), DA=DA)
# raise

# seed = 0
# tmax = 3
# dt = 0.001
# funcFF = lambda t: 0.5 + 0*t
# funcFB = lambda t: 0.75 + 0*t
# u = make_signal(t=tmax, f=Lowpass(0.1), seed=seed)
# x = partialIntegrate(u, funcFF, funcFB)
# times = np.arange(0, tmax+dt, dt)
# plt.plot(times, u, label='u')
# plt.plot(times, x, label='x')
# plt.plot(times, funcFB(times), label="DA")
# plt.legend()
# plt.savefig("plots/partialIntegrate%s.pdf"%seed)
# raise

def checkWeights(wPExc, wIExc, wPInh, wIInh, wPP, wPI, wIP, wII, dAMPA, dGABA):
    assert np.all(wPExc >= 0)
    assert np.all(wIExc >= 0)
    assert np.all(wPInh <= 0)
    assert np.all(wIInh <= 0)
    assert np.all(wPP >= 0)
    assert np.all(wPI >= 0)
    assert np.all(wIP <= 0)
    assert np.all(wII <= 0)
    assert np.all(dAMPA >= 0)
    assert np.all(dGABA <= 0)

def go(ePExc=None, eIExc=None, ePInh=None, eIInh=None, ePP=None, ePI=None, eIP=None, eII=None, wPExc=None, wIExc=None, wPInh=None, wIInh=None, wPP=None, wPI=None, wIP=None, wII=None, dNMDA=None, dAMPA=None, dGABA=None, f_NMDA=None, f_AMPA=None, f_GABA=None, f_s=None, stim=lambda t: 0, DA=lambda t: 0, n_pre=200, n_neurons=30, t=10, dt=0.001, m=Uniform(30, 40), i=Uniform(-1, 0.8), kFF=-2, kFB=-2, seed=0, stage=0):
    
    wDaInpt = kFF*np.ones((n_pre, 1))
    wDaFdbk = kFB*np.ones((n_neurons, 1))
    with nengo.Network(seed=seed) as model:
        # Stimulus and Nodes
        u = nengo.Node(stim)
        uDA = nengo.Node(DA)
        # Ensembles
        pre = nengo.Ensemble(n_pre, 1, seed=seed, label="pre")
        P = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=DA), seed=seed, label="P")
        I = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Interneuron", DA=DA), seed=seed, label="I")
        supv = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        gate = nengo.Ensemble(n_pre, 1, seed=seed, label="gate")
        buffer = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        fdbk = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=nengo.LIF(), seed=seed)
        # Connections
        uPre = nengo.Connection(u, pre, synapse=None, seed=seed)
        prePExc = nengo.Connection(pre, P, synapse=AMPA(), seed=seed)
        preIExc = nengo.Connection(pre, I, synapse=AMPA(), seed=seed)
        prePInh = nengo.Connection(pre, P, synapse=GABA(), seed=seed)
        preIInh = nengo.Connection(pre, I, synapse=GABA(), seed=seed)
        nengo.Connection(u, gate, synapse=None)
        nengo.Connection(gate, buffer, synapse=f_AMPA)
        nengo.Connection(buffer, fdbk, synapse=f_NMDA)
        nengo.Connection(fdbk, buffer, synapse=f_AMPA)
        nengo.Connection(uDA, gate.neurons, transform=wDaInpt, function=lambda x: x)
        nengo.Connection(uDA, fdbk.neurons, transform=wDaFdbk, function=lambda x: 1-x)
        # Probes
        p_u = nengo.Probe(u, synapse=None)
        p_DA = nengo.Probe(uDA, synapse=None)
        p_P = nengo.Probe(P.neurons, synapse=None)
        p_I = nengo.Probe(I.neurons, synapse=None)
        p_supv = nengo.Probe(supv.neurons, synapse=None)
        p_supv_x = nengo.Probe(supv, synapse=f_NMDA)
        p_gate = nengo.Probe(gate.neurons, synapse=None)
        p_gate_x = nengo.Probe(gate, synapse=f_AMPA)
        p_buffer = nengo.Probe(buffer.neurons, synapse=None)
        p_buffer_x = nengo.Probe(buffer, synapse=f_NMDA)
        p_fdbk = nengo.Probe(fdbk.neurons, synapse=None)
        p_fdbk_x = nengo.Probe(fdbk, synapse=f_NMDA)
        # Training
        if stage == 1:
            nengo.Connection(u, supv, synapse=f_AMPA, seed=seed)
            node = WNode(prePExc, alpha=1e-4, exc=True)
            nengo.Connection(pre.neurons, node[0:n_pre], synapse=f_AMPA)
            nengo.Connection(P.neurons, node[n_pre:n_pre+n_neurons], synapse=f_s)
            nengo.Connection(supv.neurons, node[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_s)
            node2 = WNode(preIExc, alpha=1e-5, exc=True)
            nengo.Connection(pre.neurons, node2[0:n_pre], synapse=f_AMPA)
            nengo.Connection(I.neurons, node2[n_pre:n_pre+n_neurons], synapse=f_s)
            nengo.Connection(supv.neurons, node2[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_s)
            node3 = WNode(prePInh, alpha=1e-4, inh=True)
            nengo.Connection(pre.neurons, node3[0:n_pre], synapse=f_GABA)
            nengo.Connection(P.neurons, node3[n_pre:n_pre+n_neurons], synapse=f_s)
            nengo.Connection(supv.neurons, node3[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_s)
            node4 = WNode(preIInh, alpha=1e-5, inh=True)
            nengo.Connection(pre.neurons, node4[0:n_pre], synapse=f_GABA)
            nengo.Connection(I.neurons, node4[n_pre:n_pre+n_neurons], synapse=f_s)
            nengo.Connection(supv.neurons, node4[n_pre+n_neurons: n_pre+2*n_neurons], synapse=f_s)
        if stage == 2:
            nengo.Connection(u, supv, synapse=f_AMPA, seed=seed)
        if stage == 3:
            #PTar, ITar have target activities from a gated integrator
            #PDrive, IDrive drive P, I with input from "feedback" pop
#             PTar = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=lambda t: 0), seed=seed, label="PTar")
#             ITar = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Interneuron", DA=lambda t: 0), seed=seed, label="ITar")
#             PDrive = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=DA), seed=seed, label="PDrive")  # DA?
#             IDrive = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Interneuron", DA=DA), seed=seed, label="IDrive")  # DA?
            # gated integrator populations drive PDrive/IDrive and PTar/ITar
#             bufferPPTar = nengo.Connection(buffer, PTar, synapse=f_AMPA, seed=seed)
#             bufferPITar = nengo.Connection(buffer, ITar, synapse=f_AMPA, seed=seed)
#             bufferIPTar = nengo.Connection(buffer, PTar, synapse=f_GABA, seed=seed)
#             bufferIITar = nengo.Connection(buffer, ITar, synapse=f_GABA, seed=seed)
#             fdbkPPDrive = nengo.Connection(fdbk, PDrive, synapse=f_AMPA, seed=seed) 
#             fdbkPIDrive = nengo.Connection(fdbk, IDrive, synapse=f_AMPA, seed=seed)
#             fdbkIPDrive = nengo.Connection(fdbk, PDrive, synapse=f_GABA, seed=seed) 
#             fdbkIIDrive = nengo.Connection(fdbk, IDrive, synapse=f_GABA, seed=seed)
#             # PDrive, IDrive drive P and I with ideal feedback activities given current gate state
#             PDriveP = nengo.Connection(PDrive, P, synapse=NMDA(), solver=NoSolver(dNMDA), seed=seed) 
#             PDriveI = nengo.Connection(PDrive, I, synapse=NMDA(), solver=NoSolver(dNMDA), seed=seed) 
#             IDriveP = nengo.Connection(IDrive, P, synapse=GABA(), solver=NoSolver(dGABA), seed=seed) 
#             IDriveI = nengo.Connection(IDrive, I, synapse=GABA(), solver=NoSolver(dGABA), seed=seed) 
            # P and I are driven with ideal feedback from fdbk
            fdbkPExc = nengo.Connection(fdbk, P, synapse=NMDA(), seed=seed) 
            fdbkIExc = nengo.Connection(fdbk, I, synapse=NMDA(), seed=seed) 
            fdbkPInh = nengo.Connection(fdbk, P, synapse=GABA(), seed=seed) 
            fdbkIInh = nengo.Connection(fdbk, I, synapse=GABA(), seed=seed) 
            # P and I have target activities from buffer
            node = WNode(fdbkPExc, alpha=1e-5, exc=True)
            nengo.Connection(fdbk.neurons, node[0:n_neurons], synapse=f_NMDA)
            nengo.Connection(P.neurons, node[n_neurons:2*n_neurons], synapse=f_s)
            nengo.Connection(buffer.neurons, node[2*n_neurons: 3*n_neurons], synapse=f_s)
            node2 = WNode(fdbkIExc, alpha=1e-6, exc=True)
            nengo.Connection(fdbk.neurons, node2[0:n_neurons], synapse=f_NMDA)
            nengo.Connection(I.neurons, node2[n_neurons:2*n_neurons], synapse=f_s)
            nengo.Connection(buffer.neurons, node2[2*n_neurons: 3*n_neurons], synapse=f_s)
            node3 = WNode(fdbkPInh, alpha=1e-5, inh=True)
            nengo.Connection(fdbk.neurons, node3[0:n_neurons], synapse=f_GABA)
            nengo.Connection(P.neurons, node3[n_neurons:2*n_neurons], synapse=f_s)
            nengo.Connection(buffer.neurons, node3[2*n_neurons: 3*n_neurons], synapse=f_s)
            node4 = WNode(fdbkIInh, alpha=1e-6, inh=True)
            nengo.Connection(fdbk.neurons, node4[0:n_neurons], synapse=f_GABA)
            nengo.Connection(I.neurons, node4[n_neurons:2*n_neurons], synapse=f_s)
            nengo.Connection(buffer.neurons, node4[2*n_neurons: 3*n_neurons], synapse=f_s)
#             p_PDrive = nengo.Probe(PDrive.neurons, synapse=None)
#             p_IDrive = nengo.Probe(IDrive.neurons, synapse=None)
#             p_PTar = nengo.Probe(PTar.neurons, synapse=None)
#             p_ITar = nengo.Probe(ITar.neurons, synapse=None)
        if stage == 4:
            PP = nengo.Connection(P, P, synapse=f_NMDA, seed=seed, solver=NoSolver(dNMDA))
            PI = nengo.Connection(P, I, synapse=f_NMDA, seed=seed, solver=NoSolver(dNMDA))
            IP = nengo.Connection(I, P, synapse=f_GABA, seed=seed, solver=NoSolver(dGABA))
            II = nengo.Connection(I, I, synapse=f_GABA, seed=seed, solver=NoSolver(dGABA))

    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        if stage > 1:
            for pre in range(n_pre):
                for post in range(n_neurons):
                    prePExc.weights[pre, post] = wPExc[pre, post]
                    prePExc.netcons[pre, post].weight[0] = np.abs(wPExc[pre, post])
                    prePExc.netcons[pre, post].syn().e = 0 if wPExc[pre, post] > 0 else -70
                    preIExc.weights[pre, post] = wIExc[pre, post]
                    preIExc.netcons[pre, post].weight[0] = np.abs(wIExc[pre, post])
                    preIExc.netcons[pre, post].syn().e = 0 if wIExc[pre, post] > 0 else -70
                    prePInh.weights[pre, post] = wPInh[pre, post]
                    prePInh.netcons[pre, post].weight[0] = np.abs(wPInh[pre, post])
                    prePInh.netcons[pre, post].syn().e = 0 if wPInh[pre, post] > 0 else -70
                    preIInh.weights[pre, post] = wIInh[pre, post]
                    preIInh.netcons[pre, post].weight[0] = np.abs(wIInh[pre, post])
                    preIInh.netcons[pre, post].syn().e = 0 if wIInh[pre, post] > 0 else -70
#         if stage==3:
#             for pre in range(n_pre):
#                 for post in range(n_neurons):
#                     bufferPPTar.weights[pre, post] = wpPP[pre, post]
#                     bufferPPTar.netcons[pre, post].weight[0] = np.abs(wpPP[pre, post])
#                     bufferPPTar.netcons[pre, post].syn().e = 0 if wpPP[pre, post] > 0 else -70
#                     bufferPITar.weights[pre, post] = wpPI[pre, post]
#                     bufferPITar.netcons[pre, post].weight[0] = np.abs(wpPI[pre, post])
#                     bufferPITar.netcons[pre, post].syn().e = 0 if wpPI[pre, post] > 0 else -70
#                     bufferIPTar.weights[pre, post] = wpIP[pre, post]
#                     bufferIPTar.netcons[pre, post].weight[0] = np.abs(wpIP[pre, post])
#                     bufferIPTar.netcons[pre, post].syn().e = 0 if wpIP[pre, post] > 0 else -70
#                     bufferIITar.weights[pre, post] = wpII[pre, post]
#                     bufferIITar.netcons[pre, post].weight[0] = np.abs(wpII[pre, post])
#                     bufferIITar.netcons[pre, post].syn().e = 0 if wpII[pre, post] > 0 else -70
#                     fdbkPPDrive.weights[pre, post] = wpPP[pre, post]
#                     fdbkPPDrive.netcons[pre, post].weight[0] = np.abs(wpPP[pre, post])
#                     fdbkPPDrive.netcons[pre, post].syn().e = 0 if wpPP[pre, post] > 0 else -70
#                     fdbkPIDrive.weights[pre, post] = wpPI[pre, post]
#                     fdbkPIDrive.netcons[pre, post].weight[0] = np.abs(wpPI[pre, post])
#                     fdbkPIDrive.netcons[pre, post].syn().e = 0 if wpPI[pre, post] > 0 else -70
#                     fdbkIPDrive.weights[pre, post] = wpIP[pre, post]
#                     fdbkIPDrive.netcons[pre, post].weight[0] = np.abs(wpIP[pre, post])
#                     fdbkIPDrive.netcons[pre, post].syn().e = 0 if wpIP[pre, post] > 0 else -70
#                     fdbkIIDrive.weights[pre, post] = wpII[pre, post]
#                     fdbkIIDrive.netcons[pre, post].weight[0] = np.abs(wpII[pre, post])
#                     fdbkIIDrive.netcons[pre, post].syn().e = 0 if wpII[pre, post] > 0 else -70
        if stage==4:
            for pre in range(n_neurons):
                for post in range(n_neurons):
                    PP.weights[pre, post] = wPP[pre, post]
                    PP.netcons[pre, post].weight[0] = np.abs(wPP[pre, post])
                    PP.netcons[pre, post].syn().e = 0 if wPP[pre, post] > 0 else -70
                    PI.weights[pre, post] = wPI[pre, post]
                    PI.netcons[pre, post].weight[0] = np.abs(wPI[pre, post])
                    PI.netcons[pre, post].syn().e = 0 if wPI[pre, post] > 0 else -70
                    IP.weights[pre, post] = wIP[pre, post]
                    IP.netcons[pre, post].weight[0] = np.abs(wIP[pre, post])
                    IP.netcons[pre, post].syn().e = 0 if wIP[pre, post] > 0 else -70
                    II.weights[pre, post] = wII[pre, post]
                    II.netcons[pre, post].weight[0] = np.abs(wII[pre, post])
                    II.netcons[pre, post].syn().e = 0 if wII[pre, post] > 0 else -70 
        if stage==1:
            if np.any(ePExc): prePExc.e = ePExc
            if np.any(eIExc): preIExc.e = eIExc
            if np.any(ePInh): prePInh.e = ePInh
            if np.any(eIInh): preIInh.e = eIInh
        if stage==3:
            if np.any(ePP): fdbkPExc.e = ePP
            if np.any(ePI): fdbkIExc.e = ePI
            if np.any(eIP): fdbkPInh.e = eIP
            if np.any(eII): fdbkIInh.e = eII
        neuron.h.init()
        sim.run(t, progress_bar=True)
        reset_neuron(sim, model) 
        
    if stage == 1:
        ePExc = prePExc.e
        wPExc = prePExc.weights
        eIExc = preIExc.e
        wIExc = preIExc.weights
        ePInh = prePInh.e
        wPInh = prePInh.weights
        eIInh = preIInh.e
        wIInh = preIInh.weights
    if stage == 3:
        ePP = fdbkPExc.e
        wPP = fdbkPExc.weights
        ePI = fdbkIExc.e
        wPI = fdbkIExc.weights
        eIP = fdbkPInh.e
        wIP = fdbkPInh.weights
        eII = fdbkIInh.e
        wII = fdbkIInh.weights
    return dict(
        times=sim.trange(),
        u=sim.data[p_u],
        uDA=sim.data[p_DA],
        P=sim.data[p_P],
        I=sim.data[p_I],
        supv=sim.data[p_supv],
        supv_x=sim.data[p_supv_x],
        gate=sim.data[p_gate],
        gate_x=sim.data[p_gate_x],
        buffer=sim.data[p_buffer],
        buffer_x=sim.data[p_buffer_x],
        fdbk=sim.data[p_fdbk],
        fdbk_x=sim.data[p_fdbk_x],
        ePExc=ePExc,
        wPExc=wPExc,
        eIExc=eIExc,
        wIExc=wIExc,
        ePInh=ePInh,
        wPInh=wPInh,
        eIInh=eIInh,
        wIInh=wIInh,
        ePP=ePP,
        wPP=wPP,
        ePI=ePI,
        wPI=wPI,
        eIP=eIP,
        wIP=wIP,
        eII=eII,
        wII=wII,
#         PDrive=sim.data[p_PDrive] if stage==3 else None,
#         IDrive=sim.data[p_IDrive] if stage==3 else None,
#         PTar=sim.data[p_PTar] if stage==3 else None,
#         ITar=sim.data[p_ITar] if stage==3 else None,
    )


def run(n_neurons=30, t=10, t_test=10, tDA=5, dt=0.001, n_encodes=10, n_tests=1, reg=0, f_NMDA=DoubleExp(1.06e-2, 2.85e-1), f_GABA=DoubleExp(5e-4, 1.5e-3), f_AMPA=DoubleExp(5.5e-4, 2.2e-3), f_s=DoubleExp(1e-2, 2e-1), load_w=None, load_fd=None, data_file="data/dale2.npz"):


    # Stage 1
    DA = lambda t: 0
    if load_w:
        ePExc = np.load(data_file)['ePExc']
        wPExc = np.load(data_file)['wPExc']
        eIExc = np.load(data_file)['eIExc']
        wIExc = np.load(data_file)['wIExc']
        ePInh = np.load(data_file)['ePInh']
        wPInh = np.load(data_file)['wPInh']
        eIInh = np.load(data_file)['eIInh']
        wIInh = np.load(data_file)['wIInh']
    else:
        ePExc = None
        eIExc = None
        ePInh = None
        eIInh = None
        print('Optimizing pre-ens encoders')
        for nenc in range(n_encodes):
            print("encoding trial %s"%nenc)
            s = make_signal(t=t, dt=dt, f=f_AMPA, seed=nenc)
            stim = lambda t: s[int(t/dt)]
            data = go(ePExc=ePExc, eIExc=eIExc, ePInh=ePInh, eIInh=eIInh,
                f_NMDA=f_NMDA, f_AMPA=f_AMPA, f_GABA=f_GABA, f_s=f_s,
                n_neurons=n_neurons, t=t, stim=stim, stage=1)
            ePExc = data['ePExc']
            wPExc = data['wPExc']
            eIExc = data['eIExc']
            wIExc = data['wIExc']
            ePInh = data['ePInh']
            wPInh = data['wPInh']
            eIInh = data['eIInh']
            wIInh = data['wIInh']
            np.savez('data/dale2.npz', ePExc=ePExc, wPExc=wPExc, eIExc=eIExc, wIExc=wIExc, ePInh=ePInh, wPInh=wPInh, eIInh=eIInh, wIInh=wIInh)
            aSupv = f_s.filt(data['supv'])
            aP = f_s.filt(data['P'])
            aI = f_s.filt(data['I'])
            for n in range(n_neurons):
                fig, ax = plt.subplots(1, 1)
                ax.plot(data['times'], aSupv[:,n], alpha=0.5, label='supv')
                ax.plot(data['times'], aP[:,n], alpha=0.5, label='P')
                ax.plot(data['times'], aI[:,n], alpha=0.5, label='I')
                ax.set(ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/dale_eFF_%s.pdf'%n)
                plt.close('all')

    # Stage 2
    if load_fd:
        dNMDA = np.load(data_file)['dNMDA']
        dAMPA = np.load(data_file)['dAMPA']
        dGABA = np.load(data_file)['dGABA']
    else:
        print("Optimizing decoders")
        s = make_signal(t=t, dt=dt, f=f_AMPA, seed=0)
        stim = lambda t: s[int(t/dt)]
        data = go(
            wPExc=wPExc, wIExc=wIExc, wPInh=wPInh, wIInh=wIInh,
            f_NMDA=f_NMDA, f_AMPA=f_AMPA, f_GABA=f_GABA, f_s=f_s,
            n_neurons=n_neurons, t=t, stim=stim, stage=2)
        aNMDA = f_NMDA.filt(data['P'])
        xFB = f_NMDA.filt(data['u'])
        dNMDA, _ = nengo.solvers.LstsqL2(reg=1e-2)(aNMDA, xFB)
        # positive w fron mixed dFB enforced by learning node in stage 3
        aAMPA = f_AMPA.filt(data['P'])
        aGABA = f_GABA.filt(data['I'])
        xOut = data['u']
        aOut = np.hstack((aAMPA, -aGABA))
        dBoth, _ = nnls(aOut, np.ravel(xOut))
        dAMPA = dBoth[:n_neurons].reshape((n_neurons, 1))
        dGABA = -dBoth[n_neurons:].reshape((n_neurons, 1))
        np.savez('data/dale2.npz', ePExc=ePExc, wPExc=wPExc, eIExc=eIExc, wIExc=wIExc, ePInh=ePInh, wPInh=wPInh, eIInh=eIInh, wIInh=wIInh, dNMDA=dNMDA, dAMPA=dAMPA, dGABA=dGABA)
        xhatFB = np.dot(aNMDA, dNMDA)
        xhatOut = np.dot(aAMPA, dAMPA) + np.dot(aGABA, dGABA)
        xhatFB_rmse = rmse(np.ravel(xhatFB), np.ravel(xFB))
        xhatOut_rmse = rmse(np.ravel(xhatOut), np.ravel(xOut))
        fig, ax = plt.subplots()
        ax.plot(data['times'], xFB, alpha=0.5, linestyle="--", label='xFB')
        ax.plot(data['times'], xhatFB, label='xhatFB, rmse=%.3f' %xhatFB_rmse)
#         ax.plot(data['times'], xOut, alpha=0.5, linestyle="--", label='xOut')
#         ax.plot(data['times'], xhatOut, label='xhatOut, rmse=%.3f' %xhatOut_rmse)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="train decoders")
        plt.legend(loc='upper right')
        plt.savefig("plots/dale2_train_decoders.pdf")

    # Stage 3
#     dGABAFb = np.random.RandomState(seed=0).uniform(-1e-4, 0, size=(n_neurons, 1))
    dGABAFb = -3e-4*np.ones((n_neurons, 1))
    if load_w:
        ePP = np.load(data_file)['ePP']
        wPP = np.load(data_file)['wPP']
        ePI = np.load(data_file)['ePI']
        wPI = np.load(data_file)['wPI']
        eIP = np.load(data_file)['eIP']
        wIP = np.load(data_file)['wIP']
        eII = np.load(data_file)['eII']
        wII = np.load(data_file)['wII']
    else:
        print('Optimizing ePP, ePI, eIP, eII')
        ePP = None
        ePI = None
        eIP = None
        eII = None
        DA = lambda t: 0 if t<tDA else 0.5
        for nenc in range(n_encodes):
            print("encoding trial %s"%nenc)
            s = make_signal(t=t, dt=dt, f=f_NMDA, seed=nenc)
            stim = lambda t: s[int(t/dt)]
            data = go(
                wPExc=wPExc, wIExc=wIExc, wPInh=wPInh, wIInh=wIInh,
                ePP=ePP, ePI=ePI, eIP=eIP, eII=eII,
                dNMDA=dNMDA, dAMPA=dAMPA, dGABA=dGABAFb,
                f_NMDA=f_NMDA, f_AMPA=f_AMPA, f_GABA=f_GABA, f_s=f_s,
                n_neurons=n_neurons, t=t, stim=stim, DA=DA, stage=3)
            ePP = data['ePP']
            wPP = data['wPP']
            ePI = data['ePI']
            wPI = data['wPI']
            eIP = data['eIP']
            wIP = data['wIP']
            eII = data['eII']
            wII = data['wII'] 
            np.savez('data/dale2.npz', ePExc=ePExc, wPExc=wPExc, eIExc=eIExc, wIExc=wIExc, ePInh=ePInh, wPInh=wPInh, eIInh=eIInh, wIInh=wIInh, dNMDA=dNMDA, dAMPA=dAMPA, dGABA=dGABA, ePP=ePP, wPP=wPP, ePI=ePI, wPI=wPI, eIP=eIP, wIP=wIP, eII=eII, wII=wII)
            aP = f_s.filt(data['P'])
            aI = f_s.filt(data['I'])
            aBuffer = f_s.filt(data['buffer'])
            for n in range(n_neurons):
                fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
                ax.plot(data['times'], aBuffer[:,n], alpha=0.5, label='buffer')
                ax.plot(data['times'], aP[:,n], alpha=0.5, label='P')
                ax2.plot(data['times'], aBuffer[:,n], alpha=0.5, label='buffer')
                ax2.plot(data['times'], aI[:,n], alpha=0.5, label='I')
                ax.set(ylim=((0, 40)))
                ax.legend()
                ax2.legend()
                plt.savefig('plots/tuning/dale_eFB_%s.pdf'%n)
                plt.close('all')
            # confirm gated integrator (which generates targets) is working in state-space
            fig, ax = plt.subplots()
            ax.plot(data['times'], data['uDA'], label="DA")
            ax.plot(data['times'], data['gate_x'], label="gate")
            ax.plot(data['times'], data['buffer_x'], label="buffer")
            ax.plot(data['times'], data['fdbk_x'], label="fdbk")
            ax.legend()
            fig.savefig("plots/dale2_train_gatedIntegrator.pdf")

            
    # Stage 4
    checkWeights(wPExc, wIExc, wPInh, wIInh, wPP, wPI, wIP, wII, dAMPA, dGABA)
    DA = lambda t: 0 if t < tDA else 1.0
    dGABAFb = np.random.RandomState(seed=0).uniform(-1e-4, 0, size=(n_neurons, 1))
    for test in range(n_tests):
        print("Test %s"%test)
        s = make_signal(t=t, dt=dt, f=f_NMDA, seed=test)
        stim = lambda t: s[int(t/dt)]
        data = go(
            wPExc=wPExc, wIExc=wIExc, wPInh=wPInh, wIInh=wIInh,
            wPP=wPP, wPI=wPI, wIP=wIP, wII=wII,
            dNMDA=dNMDA, dAMPA=dAMPA, dGABA=dGABAFb,
            f_NMDA=f_NMDA, f_AMPA=f_AMPA, f_GABA=f_GABA, f_s=f_s,
            n_neurons=n_neurons, t=t, stim=stim, DA=DA, stage=4)
        aNMDA = f_NMDA.filt(data['P'])
        aAMPA = f_AMPA.filt(data['P'])
        aGABA = f_GABA.filt(data['I'])
        u = data['u']
        da = data['uDA']
        gate = data['gate_x']
        buffer = data['buffer_x']
        fdbk = data['fdbk_x']
        xhat = np.dot(aNMDA, dNMDA)
        xhat_rmse = rmse(xhat, buffer)
        fig, ax = plt.subplots()
        ax.plot(data['times'], u, linestyle="--", label='u')
        ax.plot(data['times'], da, linestyle="--", label='DA')
        ax.plot(data['times'], gate, label="gate")
        ax.plot(data['times'], buffer, label="buffer")
        ax.plot(data['times'], fdbk, label="fdbk")
        ax.plot(data['times'], xhat, label='xhat, rmse=%.3f' %xhat_rmse)
        ax.set(xlabel='time (s)', ylabel=r'$\mathbf{x}$', title="test")
        plt.legend(loc='upper right')
        plt.savefig("plots/dale2_test_DA1_%s.pdf"%test)
      
        
run(n_neurons=60, t=10, tDA=3, n_encodes=10, n_tests=3, load_w=True, load_fd=True)
