
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

    
def mixedLstSq(aExc, aInh, target, dt=0.001, dMin=-1e-3, dMax=1e-3, evals=100, seed=0):
    def objective(hyperparams):
        aExc = np.load('data/mixedLstSq.npz')['aExc']
        aInh = np.load('data/mixedLstSq.npz')['aInh']
        target = np.load('data/mixedLstSq.npz')['target']
        dExc = [hyperparams["exc"+str(n)] for n in range(aExc.shape[1])]
        dInh = [hyperparams["inh"+str(n)] for n in range(aInh.shape[1])]
        d = np.hstack((dExc, dInh)).T
        A = np.hstack((aExc, aInh))
#         print(d.shape)
#         print(A.shape)
        xhat = np.dot(A, d)
        loss = rmse(xhat, target)
        result = {'loss': loss, 'status': STATUS_OK}
        for i in range(d.shape[0]):
            result[i] = d[i]
        return result
    np.savez_compressed('data/mixedLstSq.npz', aExc=aExc, aInh=aInh, target=target)
    hyperparams = {}
    hyperparams['dt'] = dt
    for n in range(aExc.shape[1]):
        hyperparams["exc"+str(n)] = hp.uniform("exc"+str(n), 0, dMax)
    for n in range(aInh.shape[1]):
        hyperparams["inh"+str(n)] = hp.uniform("inh"+str(n), dMin, 0)
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
#     taus_ens = best['result']['taus_ens']
#     d_ens = best['result']['d_ens']
#     h_ens = DoubleExp(taus_ens[0], taus_ens[1])
    dExc = np.array([best['result'][i] for i in range(aExc.shape[1])])
    dInh = np.array([best['result'][aExc.shape[1]+i] for i in range(aExc.shape[1])])
    return dExc, dInh
  
    
# # Calculate Lyapunov Exponent
# start = int(t_lyaps[0]/dt_sample)
# end = int(t_lyaps[1]/dt_sample)
# one_time = np.arange(0, t, dt)[start:end]
# times = np.zeros((n_trains, one_time.shape[0]))
# delta_tars = np.zeros((n_trains, one_time.shape[0]))
# delta_enss = np.zeros((n_trains, one_time.shape[0]))
# for ntrn in range(n_trains):
#     tar1 = f.filt(targets[ntrn-1], dt=dt_sample)[start:end]
#     tar2 = f.filt(targets[ntrn], dt=dt_sample)[start:end]
# #             tar1 = gaussian_filter1d(targets[ntrn-1], sigma=smooth, axis=0)[start:end]
# #             tar2 = gaussian_filter1d(targets[ntrn], sigma=smooth, axis=0)[start:end]
#     a_ens1 = f_ens.filt(spikes[ntrn-1], dt=dt_sample)[start:end]
#     a_ens2 = f_ens.filt(spikes[ntrn], dt=dt_sample)[start:end]
#     ens1 = np.dot(a_ens1, d_ens)
#     ens2 = np.dot(a_ens2, d_ens)
#     times[ntrn] = one_time
#     delta_tars[ntrn] = np.sqrt(
#         np.square(np.abs(tar1[:,0]-tar2[:,0])) + 
#         np.square(np.abs(tar1[:,1]-tar2[:,1])) + 
#         np.square(np.abs(tar1[:,2]-tar2[:,2])))
#     delta_enss[ntrn] = np.sqrt(
#         np.square(np.abs(ens1[:,0]-ens2[:,0])) + 
#         np.square(np.abs(ens1[:,1]-ens2[:,1])) + 
#         np.square(np.abs(ens1[:,2]-ens2[:,2])))
# all_times = times.reshape(n_trains*one_time.shape[0])
# delta_tar = delta_tars.reshape(n_trains*one_time.shape[0])
# delta_ens = delta_enss.reshape(n_trains*one_time.shape[0])
# slope_tar, intercept_tar, _, _, _ = linregress(all_times, np.log(delta_tar))
# if np.all(delta_ens > 0):
#     slope_ens, intercept_ens, _, _, _ = linregress(all_times, np.log(delta_ens))
#     error = np.abs(slope_ens - slope_tar) / slope_tar
# else:
#     print('trajectory pair %s, %s have identical points'%(ntrn-1, ntrn))
#     error = np.inf
# fig, ax = plt.subplots()
# ax.scatter(all_times, np.log(delta_tar), s=0.3, color='r', label='target')
# ax.plot(one_time, slope_tar*one_time+intercept_tar, color='r', linestyle="--", label='target fit, slope=%.4f'%slope_tar)
# if np.all(delta_ens > 0):
#     ax.scatter(all_times, np.log(delta_ens), s=0.3, color='b', label='ens')
#     ax.plot(one_time, slope_ens*one_time+intercept_ens, color='b', linestyle="--", label='ens fit, slope=%.4f'%slope_ens)
# ax.legend()
# ax.set(xlabel='time', ylabel='log euclidian distance between trajectory pair', title='error=%.3f'%error)
# fig.savefig("plots/lorenz_%s_train_lyapunov.pdf"%neuron_type)


class LearningNode(nengo.Node):
    def __init__(self, N, N_pre, dim, conn, k=1e-5, w_max=2e-4, decay=lambda t: 1, seed=0):
        self.N = N
        self.N_pre = N_pre
        self.dim = dim
        self.conn = conn
        self.w_max = w_max
        self.size_in = 2*N+N_pre+dim
        self.size_out = 0
        self.k = k
        self.decay = decay
        self.rng = np.random.RandomState(seed=seed)
        super(LearningNode, self).__init__(
            self.step, size_in=self.size_in, size_out=self.size_out)

    def step(self, t, x):
        a_pre = x[:self.N_pre]
        a_bio = x[self.N_pre: self.N_pre+self.N]
        a_supv = x[self.N_pre+self.N:]
        u = x[-self.dim:]
        pre = self.rng.randint(0, self.conn.weights.shape[0])
#         print(np.sum(self.conn.e))
        for post in range(self.conn.weights.shape[1]):
            delta_a = a_bio[post] - a_supv[post]
            # if a_bio[post] == 0 or a_supv[post] == 0:
            #     delta_a *= 2
            for dim in range(self.conn.d.shape[1]):
                dim_scale = 1 if np.sum(np.abs(u)) == 0 else np.abs(u[dim])/np.sum(np.abs(u))
                if self.conn.d[pre, dim] >= 0:
                    delta_e = -self.k * a_pre[pre] * dim_scale * self.decay(t)
                if self.conn.d[pre, dim] < 0:
                    delta_e = self.k * a_pre[pre] * dim_scale * self.decay(t)
                self.conn.e[pre, post, dim] += delta_a * delta_e
            self.conn.weights[pre, post] = np.dot(self.conn.d[pre], self.conn.e[pre, post])
            if self.conn.weights[pre, post] > self.w_max:
                self.conn.weights[pre, post] = self.w_max
                self.conn.e[pre, post] *= 0.8
            if self.conn.weights[pre, post] < -self.w_max:
                self.conn.weights[pre, post] = -self.w_max
                self.conn.e[pre, post] *= 0.8
#                 self.conn.weights[pre, post] += delta_a * -self.k * a_pre[pre]
            self.conn.netcons[pre, post].weight[0] = np.abs(self.conn.weights[pre, post])
            # print(np.abs(self.conn.weights[pre, post]))
            self.conn.netcons[pre, post].syn().e = 0.0 if self.conn.weights[pre, post] > 0 else -70.0
        return

            #PTar, ITar have target activities from a gated integrator
            #PDrive, IDrive drive P, I with input from "feedback" pop
            PTar = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=lambda t: 0), seed=seed, label="PTar")
            ITar = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Interneuron", DA=lambda t: 0), seed=seed, label="ITar")
            PDrive = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=DA), seed=seed, label="PDrive")  # DA?
            IDrive = nengo.Ensemble(n_neurons, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Interneuron", DA=DA), seed=seed, label="IDrive")  # DA?
            gated integrator populations drive PDrive/IDrive and PTar/ITar
            bufferPPTar = nengo.Connection(buffer, PTar, synapse=f_AMPA, seed=seed)
            bufferPITar = nengo.Connection(buffer, ITar, synapse=f_AMPA, seed=seed)
            bufferIPTar = nengo.Connection(buffer, PTar, synapse=f_GABA, seed=seed)
            bufferIITar = nengo.Connection(buffer, ITar, synapse=f_GABA, seed=seed)
            fdbkPPDrive = nengo.Connection(fdbk, PDrive, synapse=f_AMPA, seed=seed) 
            fdbkPIDrive = nengo.Connection(fdbk, IDrive, synapse=f_AMPA, seed=seed)
            fdbkIPDrive = nengo.Connection(fdbk, PDrive, synapse=f_GABA, seed=seed) 
            fdbkIIDrive = nengo.Connection(fdbk, IDrive, synapse=f_GABA, seed=seed)
            # PDrive, IDrive drive P and I with ideal feedback activities given current gate state
            PDriveP = nengo.Connection(PDrive, P, synapse=NMDA(), solver=NoSolver(dNMDA), seed=seed) 
            PDriveI = nengo.Connection(PDrive, I, synapse=NMDA(), solver=NoSolver(dNMDA), seed=seed) 
            IDriveP = nengo.Connection(IDrive, P, synapse=GABA(), solver=NoSolver(dGABA), seed=seed) 
            IDriveI = nengo.Connection(IDrive, I, synapse=GABA(), solver=NoSolver(dGABA), seed=seed) 
            p_PDrive = nengo.Probe(PDrive.neurons, synapse=None)
            p_IDrive = nengo.Probe(IDrive.neurons, synapse=None)
            p_PTar = nengo.Probe(PTar.neurons, synapse=None)
            p_ITar = nengo.Probe(ITar.neurons, synapse=None)
        if stage==3:
            for pre in range(n_pre):
                for post in range(n_neurons):
                    bufferPPTar.weights[pre, post] = wpPP[pre, post]
                    bufferPPTar.netcons[pre, post].weight[0] = np.abs(wpPP[pre, post])
                    bufferPPTar.netcons[pre, post].syn().e = 0 if wpPP[pre, post] > 0 else -70
                    bufferPITar.weights[pre, post] = wpPI[pre, post]
                    bufferPITar.netcons[pre, post].weight[0] = np.abs(wpPI[pre, post])
                    bufferPITar.netcons[pre, post].syn().e = 0 if wpPI[pre, post] > 0 else -70
                    bufferIPTar.weights[pre, post] = wpIP[pre, post]
                    bufferIPTar.netcons[pre, post].weight[0] = np.abs(wpIP[pre, post])
                    bufferIPTar.netcons[pre, post].syn().e = 0 if wpIP[pre, post] > 0 else -70
                    bufferIITar.weights[pre, post] = wpII[pre, post]
                    bufferIITar.netcons[pre, post].weight[0] = np.abs(wpII[pre, post])
                    bufferIITar.netcons[pre, post].syn().e = 0 if wpII[pre, post] > 0 else -70
                    fdbkPPDrive.weights[pre, post] = wpPP[pre, post]
                    fdbkPPDrive.netcons[pre, post].weight[0] = np.abs(wpPP[pre, post])
                    fdbkPPDrive.netcons[pre, post].syn().e = 0 if wpPP[pre, post] > 0 else -70
                    fdbkPIDrive.weights[pre, post] = wpPI[pre, post]
                    fdbkPIDrive.netcons[pre, post].weight[0] = np.abs(wpPI[pre, post])
                    fdbkPIDrive.netcons[pre, post].syn().e = 0 if wpPI[pre, post] > 0 else -70
                    fdbkIPDrive.weights[pre, post] = wpIP[pre, post]
                    fdbkIPDrive.netcons[pre, post].weight[0] = np.abs(wpIP[pre, post])
                    fdbkIPDrive.netcons[pre, post].syn().e = 0 if wpIP[pre, post] > 0 else -70
                    fdbkIIDrive.weights[pre, post] = wpII[pre, post]
                    fdbkIIDrive.netcons[pre, post].weight[0] = np.abs(wpII[pre, post])
                    fdbkIIDrive.netcons[pre, post].syn().e = 0 if wpII[pre, post] > 0 else -70
        PDrive=sim.data[p_PDrive] if stage==3 else None,
        IDrive=sim.data[p_IDrive] if stage==3 else None,
        PTar=sim.data[p_PTar] if stage==3 else None,
        ITar=sim.data[p_ITar] if stage==3 else None,

soma {
    g_pas = (10.00-1.0*DA)*1e-5  /* mho/cm2 */
}
dendrite {
    g_pas = (10.00-1.0*DA)*1e-5  /* mho/cm2 */
}

# Low DA
# dFFAMPA *= 0.0022
data = goLIF(dFFAMPA, 0.5*dFFNMDA, dFBAMPA, 0.5*dFBNMDA, wEnsInhAMPA, 0.5*wEnsInhNMDA, 0*wInhFdfw, 0.5*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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
ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_noFdfwInh_lowDA.pdf")

data = goLIF(dFFAMPA, 0*dFFNMDA, dFBAMPA, 0*dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, 0*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
aEnsAMPA = fAMPA.filt(data['ens'])
targetAMPA = fAMPA.filt(fAMPA.filt(data['inpt']))
dFBAMPA, _ = LstsqL2(reg=1e-2)(aEnsAMPA, targetAMPA)
xhatFBAMPA = np.dot(aEnsAMPA, dFBAMPA)
fig, ax = plt.subplots()
ax.plot(data['times'], targetAMPA, linestyle="--", label='target (AMPA)')
ax.plot(data['times'], xhatFBAMPA, alpha=0.5, label='ens (AMPA)')
ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIFAMPA_ens.pdf")

data = goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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
ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
ax.legend()
ax.set(xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
fig.savefig("plots/gatedMemory_goLIF_intg_InhEns.pdf")

aFdfwAMPA = fAMPA.filt(data['fdfw'])
targetAMPA = fAMPA.filt(data['inpt'])
xhatFFAMPA = np.dot(aFdfwAMPA, dFFAMPA)
aEnsAMPA = fAMPA.filt(data['ens'])
targetIntgAMPA = fAMPA.filt(data['intg'])
xhatIntgAMPA = np.dot(aEnsAMPA, dFBAMPA)


eEnsInhAMPA = np.load(dataFile)['eEnsInhAMPA']
wEnsInhAMPA = np.load(dataFile)['wEnsInhAMPA']
eEnsInhNMDA = np.load(dataFile)['eEnsInhNMDA']
wEnsInhNMDA = np.load(dataFile)['wEnsInhNMDA']
eEnsInhAMPA = None
wEnsInhAMPA = None
eEnsInhNMDA = None
wEnsInhNMDA = None

class WilsonRungeKutta(NeuronType):

    probeable = ('spikes', 'voltage', 'recovery', 'conductance')
    threshold = NumberParam('threshold')
    tau_V = NumberParam('tau_V')
    tau_R = NumberParam('tau_R')
    tau_H = NumberParam('tau_H')
    
    _v0 = -0.754  # initial voltage
    _r0 = 0.279  # initial recovery
    _maxJ = 2.0  # clip input current at this maximum to avoid catastrophic shutdown
    
    def __init__(self, threshold=-0.20, tau_V=0.00097, tau_R=0.0056, tau_H=0.0990):
        super(WilsonRungeKutta, self).__init__()
        self.threshold = threshold
        self.tau_V = tau_V
        self.tau_R = tau_R
        self.tau_H = tau_H
        
        # TODO(arvoelke): Try replacing this solver with something like
        # http://www2.gsu.edu/~matrhc/PyDSTool.htm
        # The main consideration is that we need a callback to count spikes
        from scipy.integrate import ode
        self.solver = ode(self._ode_fun).set_integrator(
            'dopri5', first_step=0.000025, nsteps=100,
            rtol=1e-2, atol=1e-3)  # runge-kutta method of order (4)5
        
    def rates(self, x, gain, bias):
        """Estimates steady-state firing rate given gain and bias."""
        J = self.current(x, gain, bias)
        voltage = self._v0*np.ones_like(J)
        recovery = self._r0*np.ones_like(J)
        conductance = np.zeros_like(J)

        return settled_firingrate(
            self.step_math, J, [voltage, recovery, conductance],
            dt=0.001, settle_time=0.1, sim_time=1.0)

    def _ode_fun(self, dummy_t, y, J):  # first argument to scipy.integrate.ode
        V, R, H = np.split(y, 3)
        dV = (-(17.81 + 47.58*V + 33.80*np.square(V))*(V - 0.48) -
              26*R*(V + 0.95) - 13*H*(V + 0.95) + J)
        dR = -R + 1.29*V + 0.79 + 3.30*np.square(V + 0.38)
        dH = -H + 11*(V + 0.754)*(V + 0.69)
        return np.concatenate((
            dV / self.tau_V, dR / self.tau_R, dH / self.tau_H))

    def step_math(self, dt, J, spiked, V, R, H):
        # It's a little silly to be reinitializing the solver on
        # every time-step, but any other ways that I could think of would 
        # violate the nengo builder's assumption that the neuron's state is
        # encapsulated by the signals in SimNeurons
        self.solver.set_initial_value(np.concatenate((V, R, H)))
        self.solver.set_f_params(J.clip(max=self._maxJ))
        
        spiked[:] = 0
        AP = V > self.threshold
        def spike_detector(dummy_t, y):  # callback for each sub-step
            V_t = y[:len(V)] > self.threshold
            spiked[:] += V_t & (~AP)  # note the "+="
            AP[:] = V_t
        self.solver.set_solout(spike_detector)

        V[:], R[:], H[:] = np.split(self.solver.integrate(self.solver.t + dt), 3)
        if not self.solver.successful():
            raise ValueError("ODE solver failed with status code: %d" % (
                self.solver.get_return_code()))
        spiked[:] /= dt

        return spiked, V, R, H

@Builder.register(WilsonRungeKutta)
def build_wilsonneuron(model, neuron_type, neurons):
    model.sig[neurons]['voltage'] = Signal(
        neuron_type._v0*np.ones(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['recovery'] = Signal(
        neuron_type._r0*np.ones(neurons.size_in), name="%s.recovery" % neurons)
    model.sig[neurons]['conductance'] = Signal(
        np.zeros(neurons.size_in), name="%s.conductance" % neurons)
    model.add_op(SimNeurons(
        neurons=neuron_type,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['recovery'],
                model.sig[neurons]['conductance']]))

    def df_opt_hx(x, ens, name='default', df_evals=100, seed=0, dt=0.001, dt_sample=0.001,
        tau_rise=1e-3, tau_fall=[3e-2, 3e-1], penalty=0.25, algo=tpe.suggest):  # rand.suggest):#

    np.savez_compressed('data/%s_ens.npz'%name, ens=ens)
    np.savez_compressed('data/%s_x.npz'%name, x=x)
    del(ens)
    del(x)
    
    hyperparams = {}
    hyperparams['name'] = name
    hyperparams['dt'] = dt
    hyperparams['dt_sample'] = dt_sample
    hyperparams['tau_rise'] = tau_rise
    # hyperparams['ens'] = hp.loguniform('ens', np.log10(tau_fall[0]), np.log10(tau_fall[1]))
    # hyperparams['x'] = hp.loguniform('x', np.log10(tau_fall[0]), np.log10(tau_fall[1]))
    hyperparams['ens'] = hp.uniform('ens', tau_fall[0], tau_fall[1])
    hyperparams['x'] = hp.uniform('x', tau_fall[0], tau_fall[1])

    def objective(hyperparams):
        taus_ens = [hyperparams['tau_rise'], hyperparams['ens']]
        taus_x = [hyperparams['tau_rise'], hyperparams['x']]
        h_ens = DoubleExp(taus_ens[0], taus_ens[1])
        h_x = DoubleExp(taus_x[0], taus_x[1])
        A = h_ens.filt(np.load('data/%s_ens.npz'%hyperparams['name'])['ens'], dt=hyperparams['dt_sample'])
        x = h_x.filt(np.load('data/%s_x.npz'%hyperparams['name'])['x'], dt=hyperparams['dt_sample'])
        if dt != dt_sample:
            A = A[::int(dt_sample/dt)]
            x = x[::int(dt_sample/dt)]
        d_ens = Lstsq()(A, x)[0]
        xhat = np.dot(A, d_ens)
        loss = rmse(xhat, x)
        loss += penalty * taus_ens[1]
        return {'loss': loss, 'taus_ens': taus_ens, 'taus_x': taus_x, 'd_ens': d_ens, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective,
        rstate=np.random.RandomState(seed=seed),
        space=hyperparams,
        algo=algo,
        max_evals=df_evals,
        trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
    taus_ens = best['result']['taus_ens']
    taus_x = best['result']['taus_x']
    d_ens = best['result']['d_ens']
    h_ens = DoubleExp(taus_ens[0], taus_ens[1])
    h_x = DoubleExp(taus_x[0], taus_x[1])
        
    return d_ens, h_ens, taus_ens, h_x, taus_x


def dh_lstsq(stim_data, target_data, spk_data,
        lambda_c=1e-1, lambda_d=1e-1, order=1, n_samples=10000,
        min_d=-1e-2, max_d=1e-2, dt=0.001, h_tar=Lowpass(0.1), 
        mean_taus=[1e-1, 1e-2], std_taus=[1e-2, 1e-3], max_tau=1e0, lstsq_iter=100):
    
    """Courtesy of Aaron Voelker"""
    mean_taus = np.array(mean_taus)[:order]
    std_taus = np.array(std_taus)[:order]

    def sample_prior(n_samples, order, mean_taus, std_taus, min_tau=1e-5, rng=np.random.RandomState(seed=0)):
        """Return n samples (taus) from the prior of a k'th-order synapse."""
        taus = np.zeros((n_samples, order))
        for o in range(order):
            taus[:, o] = rng.normal(mean_taus[o], std_taus[o], size=(n_samples, )).clip(min_tau)
        return taus
    
    for att in range(lstsq_iter):  # attempts
        assert len(mean_taus) == order
        assert len(std_taus) == order
        taus = sample_prior(n_samples, order, mean_taus, std_taus)

        poles = -1. / taus
        n_steps = spk_data.shape[0]
        n_neurons = spk_data.shape[1]
        assert poles.shape == (n_samples, order)

        tf_params = np.zeros((n_samples, order))
        for i in range(n_samples):
            sys = LinearSystem(([], poles[i, :], 1 / np.prod(taus[i, :])))   # (zeros, poles, gain)
            assert len(sys) == order
            assert np.allclose(sys.dcgain, 1)
            den_normalized = np.asarray(sys.den / sys.num[0])
            assert len(den_normalized) == order + 1
            assert np.allclose(den_normalized[-1], 1)  # since normalized
            # tf_params ordered from lowest to highest, ignoring c_0 = 1, i.e., [c_1, ..., c_k]
            tf_params[i, :] = den_normalized[:-1][::-1]

        # We assume c_i are independent by setting the off-diagonals to zero
        C = np.cov(tf_params, rowvar=False)
        if order == 1:
            C = C*np.eye(1)
        Q = np.abs(np.linalg.inv(C))
        c0 = np.mean(tf_params, axis=0)
        d0 = np.ones((n_neurons, ))
        cd0 = np.hstack((c0, d0))
        assert Q.shape == (order, order)
        assert cd0.shape == (order+n_neurons,)

        diff = (1. - ~z) / dt
        A = np.zeros((n_steps, order + n_neurons))
        deriv_n = target_data
        for i in range(order):
            deriv_n = diff.filt(deriv_n, dt=dt)
            A[:, i] = deriv_n.ravel()  # todo: D>1
        for n in range(n_neurons):
            A[:, order+n] = spk_data[:, n]
        b = h_tar.tau  # set on pre_u ==> supv connection in network
        Y = (b*stim_data - target_data)
        A = h_tar.filt(A, dt=dt, axis=0)
        Y = h_tar.filt(Y, dt=dt)

        # construct block diagonal matrix with different regularizations for filter coefficients and decoders
        L = block_diag(lambda_c*Q, lambda_d*np.eye(n_neurons))
        gamma = A.T.dot(A) + L
        upsilon = A.T.dot(Y) + L.dot(cd0).reshape((order+n_neurons, 1))  # optional term with tikhonov regularization

        cd = np.linalg.inv(gamma).dot(upsilon).ravel()
        c_new = cd[:order]
        d_new = -1.*cd[-n_neurons:]
        assert c_new.shape==(order,)
        assert d_new.shape==(n_neurons,)
        print('taus attempt %s, nonzero d %s, tau=%s: '%(att, np.count_nonzero(d_new+1), c_new))
        for n in range(n_neurons):
            if d_new[n] > max_d or d_new[n] < min_d:
                d_new[n] = 0
        d_new = d_new.reshape((n_neurons, 1))
        if order == 1:
            h_new = Lowpass(c_new[0])
        elif order == 2:
            h_new = DoubleExp(c_new[0], c_new[1])
#         h_new = 1. / (1 + sum(c_new[i] * s**(i+1) for i in range(order)))
        assert np.allclose(h_new.dcgain, 1)
        if np.all(c_new > 0):
            break
        else:
            mean_taus[np.argmin(mean_taus)] *= 1.25
            lambda_c *= 1.25
            lambda_d *= 1.25

    return d_new, h_new
    
    
class LearningNode2(nengo.Node):
    def __init__(self, N, N_pre, conn, conn_supv=None, k=1e-5, t_trans=0.01, exc=False, inh=False, seed=0):
        self.N = N
        self.N_pre = N_pre
        self.conn = conn
        self.conn_supv = conn_supv
        self.check = 10
        self.size_in = 2*N+N_pre
        self.size_out = 0
        self.k = k
        self.t_trans = t_trans
        self.exc = exc
        self.inh = inh
        if self.exc and self.inh:
            raise "can't force excitatory and inhibitory weights"
        self.rng = np.random.RandomState(seed=seed)               
        super(LearningNode2, self).__init__(
            self.step, size_in=self.size_in, size_out=self.size_out)
    def step(self, t, x):
        if t < self.t_trans:
            return
        a_pre = x[:self.N_pre]
        a_bio = x[self.N_pre: self.N_pre+self.N]
        a_supv = x[self.N_pre+self.N:]
        pre = self.rng.randint(0, self.conn.weights.shape[0])
        for post in range(self.conn.weights.shape[1]):
            if self.conn_supv:
                volts_supv = np.array([self.conn_supv.v_recs[post][-n] for n in range(self.check)])
#                 print(volts_supv)
                if np.any(np.isnan(volts_supv)):  # crash check
                    continue  # no weight update
            volts = np.array([self.conn.v_recs[post][-n] for n in range(self.check)])
            if np.any(np.isnan(volts)):  # crash check
                continue  # no weight update
            elif a_bio[post] > 40: # oversaturation condition 1
                for pp in range(self.conn.weights.shape[0]):
                    self.conn.e[pp, post] *= 0.9
            elif len(np.where((volts > -40) & (volts < 5))[0]) == self.check:  # oversaturation condition 2
                for pp in range(self.conn.weights.shape[0]):
                    self.conn.e[pp, post] *= 0.9
            else:  # encoder/weight update
                delta_a = a_bio[post] - a_supv[post]
                for dim in range(self.conn.d.shape[1]):
                    sign = -1 if self.conn.d[pre, dim] >= 0 else 1
                    delta_e = sign * self.k * a_pre[pre]
                    self.conn.e[pre, post, dim] += delta_a * delta_e
            w = np.dot(self.conn.d[pre], self.conn.e[pre, post])
            if self.exc and w < 0:
                w = 0
            if self.inh and w > 0:
                w = 0
            self.conn.weights[pre, post] = w 
            self.conn.netcons[pre, post].weight[0] = np.abs(w)
#             print(dir(self.conn.netcons[pre, post].syn()))
            self.conn.netcons[pre, post].syn().e = 0.0 if w > 0 else -70.0
        return
    

def d_opt(target, spikes, h, h_tar, reg=1e-1, dt=0.001):
    target = h_tar.filt(target, dt=dt)
    A = h.filt(spikes, dt=dt)
    d_new = LstsqL2(reg=reg)(A, target)[0]
    return d_new

#     is_neurons = isinstance(conn.post_obj, nengo.ensemble.Neurons)
    elif is_neurons:
        if isinstance(conn.post_obj.ensemble.neuron_type, Bio):
            is_NEURON = True

    if load:
        d3 = np.load(file)['d3']
        tauRise3 = np.load(file)['tauRise3']
        tauFall3 = np.load(file)['tauFall3']
        f3 = DoubleExp(tauRise3, tauFall3)
    else:
        print('readout decoders for ens')
        spikes = np.zeros((1, N))
        targets = np.zeros((1, 2))
        for n in range(nTrain):
            data = go(d1=d1, e1=e1, f1=f1, d2=d2, e2=e2, f2=f2, N=N, t=t, dt=dt, fS=fS, neuron_type=neuron_type, freq=freq, phase=rng.uniform(0, 1), tTrans=tTrans, test=True)
            spikes = np.append(spikes, data['ens'][int(tTrans/dt):], axis=0)
            tar = f.filt(data['inpt'], dt=dt)[int(tTrans/dt):]
            targets = np.append(targets, tar, axis=0)
        d3, f3, tauRise3, tauFall3, X, Y, error = decode(spikes, targets, dt=dt, name="oscillateNew")
        np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3)
        times = np.arange(0, (t-tTrans)*nTrain+dt, dt)
        plotState(times, X, Y, error, "oscillateNew", "%s_out"%neuron_type, t*nTrain)

        times = data['times'][int(2*tTrans/dt):]
        A = f3.filt(data['ens'], dt=dt)
        X = np.dot(A, d3)[int(2*tTrans/dt):]
        Y = f.filt(data['inpt'], dt=dt)[int(2*tTrans/dt):]
        plotState(times, X[:,0], Y[:,0], rmse(X[:,0], Y[:,0]), "oscillateNew", "%s_test%s_dim0"%(neuron_type, test), t)
        plotState(times, X[:,1], Y[:,1], rmse(X[:,1], Y[:,1]), "oscillateNew", "%s_test%s_dim1"%(neuron_type, test), t)
        errors[test] = rmse(X, Y)
    print('%s errors:'%neuron_type, errors)
    np.savez("data/oscillateNew_%s.npz"%neuron_type, d1=d1, tauRise1=tauRise1, tauFall1=tauFall1, e1=e1, d2=d2, tauRise2=tauRise2, tauFall2=tauFall2, e2=e2, d3=d3, tauRise3=tauRise3, tauFall3=tauFall3, errors=errors)

    def goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, stim=lambda t: 0, gating=lambda t: 0, N=100, t=10, dt=0.001, m=Uniform(30, 30), i=Uniform(-1, 0.6), seed=0):

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

def easy(N=200, t=10, dt=0.001, fAMPA=DoubleExp(0.55e-3, 2.2e-3), fNMDA=DoubleExp(2.3e-3, 95.0e-3), fGABA=DoubleExp(0.5e-3, 1.5e-3), kAMPA=0.1, kNMDA=0.7, kGABA=0.8, tNMDA=0.1, wEnsInhAMPA=0.1, wEnsInhNMDA=1, reg=1e-2):

    rng = np.random.RandomState(seed=0)
    dFFAMPA = np.zeros((N, 1))
    dFFNMDA = np.zeros((N, 1))
#     dFBAMPA = np.zeros((N, 1))
    dFBAMPA = rng.uniform(0, 1e-5, size=(N, 1))  # negligable feedback
    dFBNMDA = np.zeros((N, 1))
    wInhFdfw = rng.uniform(-1e-3, 0, size=(N, N))
    wInhEns = rng.uniform(-1e-4, 0, size=(N, N))
    stim = makeSignal(t=t, dt=dt, f=fNMDA)

    # Feedforward decoders from fdfw to ens
    data = goLIF(dFFAMPA, dFFNMDA, dFBAMPA, dFBNMDA, wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, 0*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aFdfwAMPA = fAMPA.filt(data['fdfw'])
    aFdfwNMDA = fNMDA.filt(data['fdfw'])
    targetAMPA = fAMPA.filt(data['inpt'])
    targetNMDA = fNMDA.filt(data['inpt'])
    dFFAMPA, _ = LstsqL2(reg=reg)(aFdfwAMPA, targetAMPA)
    dFFNMDA, _ = LstsqL2(reg=reg)(aFdfwNMDA, targetNMDA)
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

    # Readout decoders for ens; assume high DA condition and inh-ens
    data = goLIF(kAMPA*dFFAMPA, dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
    aEnsNMDA = fNMDA.filt(data['ens'])
    targetNMDA = fNMDA.filt(fNMDA.filt(data['inpt']))
    dFBNMDA, _ = LstsqL2(reg=reg)(aEnsNMDA, targetNMDA)
    xhatFBNMDA = np.dot(aEnsNMDA, dFBNMDA)
    fig, ax = plt.subplots()
    ax.plot(data['times'], targetNMDA, linestyle="--", label='target (NMDA)')
    ax.plot(data['times'], xhatFBNMDA, alpha=0.5, label='ens (NMDA)')
    ax.plot(data['times'], data['inhState'], alpha=0.5, label='inh')
    ax.legend()
    ax.set(ylim=((-1, 1)), xlabel="time (s)", ylabel=r"$\mathbf{\hat{x}}(t)$")
    fig.savefig("plots/gatedMemory_goLIF_fdbk.pdf")

    # Test integration in high vs low DA; assume inh-ens
    data = goLIF(kAMPA*dFFAMPA, tNMDA*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, 0*wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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
    data = goLIF(dFFAMPA, kNMDA*tNMDA*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, wEnsInhAMPA, kNMDA*wEnsInhNMDA, 0*wInhFdfw, kGABA*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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

    # Test integration with inh-fdfw in high vs low DA
    data = goLIF(kAMPA*dFFAMPA, tNMDA*dFFNMDA, kAMPA*dFBAMPA, dFBNMDA, kAMPA*wEnsInhAMPA, wEnsInhNMDA, wInhFdfw, wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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
    data = goLIF(dFFAMPA, kNMDA*tNMDA*dFFNMDA, dFBAMPA, kNMDA*dFBNMDA, wEnsInhAMPA, kNMDA*wEnsInhNMDA, kGABA*wInhFdfw, kGABA*wInhEns, fAMPA, fNMDA, fGABA, N=N, stim=stim, t=t, dt=dt)
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

easy()
raise



def goBio(
    fAMPA, fNMDA, fGABA, fS,
    dFdfwEnsAMPA=None, dFdfwEnsNMDA=None, dEnsEnsAMPA=None, dEnsEnsNMDA=None,
    ePreFdfw=None, wPreFdfw=None, 
    eFdfwEnsAMPA=None, wFdfwEnsAMPA=None, eFdfwEnsNMDA=None, wFdfwEnsNMDA=None, 
    eEnsEnsAMPA=None, wEnsEnsAMPA=None, eEnsEnsNMDA=None, wEnsEnsNMDA=None, 
    eEnsInhAMPA=None, wEnsInhAMPA=None, eEnsInhNMDA=None, wEnsInhNMDA=None, 
    wInhFdfw=None, wInhEns=None,
    stim=lambda t: 0, gating=lambda t: 0, N=100, NPre=200, t=10, dt=0.001,
    m=Uniform(20, 40), i=Uniform(0, 0.8), e=Choice([[1]]), lif=nengo.LIF(), DA=lambda t: 0, stage=0, seed=0):

    with nengo.Network(seed=seed) as model:
        inpt = nengo.Node(stim)
        intg = nengo.Ensemble(1, 1, neuron_type = nengo.Direct())
        pre = nengo.Ensemble(NPre, 1, radius=2, encoders=e, max_rates=m, intercepts=i, seed=seed)
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
        tarFdfw = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        tarEns = nengo.Ensemble(N, 1, encoders=e, max_rates=m, intercepts=i, neuron_type=lif, seed=seed)
        pTarFdfw = nengo.Probe(tarFdfw.neurons, synapse=None)
        pTarEns = nengo.Probe(tarEns.neurons, synapse=None)

        # Training
        if stage==0:
            # decoder training
            pass
        if stage==1:
            nengo.Connection(inpt, tarFdfw, synapse=fAMPA)
            node = WNode(preFdfw, alpha=3e-4, exc=True)
            nengo.Connection(pre.neurons, node[0:pre.n_neurons], synapse=fAMPA)
            nengo.Connection(fdfw.neurons, node[pre.n_neurons:pre.n_neurons+N], synapse=fS)
            nengo.Connection(tarFdfw.neurons, node[pre.n_neurons+N:], synapse=fS)
        if stage==2:
            nengo.Connection(inpt, tarEns, synapse=fAMPA)
            node = WNode(fdfwEnsAMPA, alpha=3e-4, exc=True)
            nengo.Connection(fdfw.neurons, node[0:N], synapse=fAMPA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(tarEns.neurons, node[2*N: 3*N], synapse=fS)
        if stage==3:
            nengo.Connection(inpt, tarEns, synapse=fNMDA)
            node = WNode(fdfwEnsNMDA, alpha=3e-6, exc=True)
            nengo.Connection(fdfw.neurons, node[0:N], synapse=fNMDA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(tarEns.neurons, node[2*N: 3*N], synapse=fS)
        if stage==4:
            pre2 = nengo.Ensemble(NPre, 1, radius=2, encoders=e, max_rates=m, intercepts=i, seed=seed)
            fdbk = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=BioNeuron("Pyramidal", DA=DA), seed=seed)
#             nengo.Connection(inpt, pre2, transform=2, synapse=None)
            nengo.Connection(intg, pre2, synapse=None)
            inptPre.synapse = fNMDA
            pre2Fdbk = nengo.Connection(pre2, fdbk, synapse=fAMPA)
            pFdbk = nengo.Probe(fdbk.neurons, synapse=None)
            fdbkEnsAMPA = nengo.Connection(fdbk, ens, synapse=AMPA(), solver=NoSolver(dEnsEnsAMPA))
            fdbkEnsNMDA = nengo.Connection(fdbk, ens, synapse=NMDA(), solver=NoSolver(dEnsEnsNMDA))
            node = WNode(fdbkEnsAMPA, alpha=3e-8, exc=True)
            nengo.Connection(fdbk.neurons, node[0:N], synapse=fAMPA)
            nengo.Connection(ens.neurons, node[N:2*N], synapse=fS)
            nengo.Connection(fdbk.neurons, node[2*N: 3*N], synapse=fS)
            node2 = WNode(fdbkEnsNMDA, alpha=3e-6, exc=True)
            nengo.Connection(fdbk.neurons, node2[0:N], synapse=fNMDA)
            nengo.Connection(ens.neurons, node2[N:2*N], synapse=fS)
            nengo.Connection(fdbk.neurons, node2[2*N: 3*N], synapse=fS)

    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        # Weight setting
        for pre in range(NPre):
            for post in range(N):
                if np.any(wPreFdfw):
                    preFdfw.weights[pre, post] = wPreFdfw[pre, post]
                    preFdfw.netcons[pre, post].weight[0] = np.abs(wPreFdfw[pre, post])
                    preFdfw.netcons[pre, post].syn().e = 0 if wPreFdfw[pre, post] > 0 else -70
                    if stage==4:
                        pre2Fdbk.weights[pre, post] = wPreFdfw[pre, post]
                        pre2Fdbk.netcons[pre, post].weight[0] = np.abs(wPreFdfw[pre, post])
                        pre2Fdbk.netcons[pre, post].syn().e = 0 if wPreFdfw[pre, post] > 0 else -70
        for pre in range(N):
            for post in range(N):
                if np.any(wFdfwEnsAMPA):
                    fdfwEnsAMPA.weights[pre, post] = wFdfwEnsAMPA[pre, post]
                    fdfwEnsAMPA.netcons[pre, post].weight[0] = np.abs(wFdfwEnsAMPA[pre, post])
                    fdfwEnsAMPA.netcons[pre, post].syn().e = 0 if wFdfwEnsAMPA[pre, post] > 0 else -70
                if np.any(wFdfwEnsNMDA):
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
    NPre = 50
    Ndec = 3
    nTrains = 5
    nTests = 1
    dFdfwEnsAMPA = np.zeros((N, 1))
    dFdfwEnsNMDA = np.zeros((N, 1))
    dEnsEnsAMPA = np.zeros((N, 1))
    dEnsEnsNMDA = np.zeros((N, 1))
#     fAMPA = DoubleExp(5.5e-4, 2.2e-3)
#     fNMDA = DoubleExp(1e-2, 2.85e-1)
#     fGABA = DoubleExp(5e-4, 1.5e-3)
    fAMPA = DoubleExp(0.55e-3, 2.2e-3)
    fNMDA = DoubleExp(2.3e-3, 95.0e-3)
    fGABA = DoubleExp(0.5e-3, 1.5e-3)
    fS = Lowpass(2e-1)
    rng = np.random.RandomState(seed=0)
    wEnsInhAMPA = rng.uniform(0, 5e-6, size=(N, N))
    wEnsInhNMDA = rng.uniform(0, 5e-4, size=(N, N))
    wInhFdfw = rng.uniform(-1e-2, 0, size=(N, N))
    wInhEns = rng.uniform(-1e-4, 0, size=(N, N))
    tNMDA = 1.0
    tAMPA = 1.0
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
                fAMPA, fNMDA, fGABA, fS,
                ePreFdfw=ePreFdfw,
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=1, DA=lambda t: 0)
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
                ax.set(ylim=((0, 40)))
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
        for n in range(Ndec):
            print("fdfw decoder trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, 
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0)
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
                fAMPA, fNMDA, fGABA, fS,
                dFdfwEnsAMPA=dFdfwEnsAMPA, dFdfwEnsNMDA=dFdfwEnsNMDA,
                wPreFdfw=wPreFdfw, eFdfwEnsAMPA=eFdfwEnsAMPA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=2, DA=lambda t: 0)
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
                ax.set(ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdfwEnsAMPA_%s.pdf'%n)
                plt.close('all')

    # Optimize encoders from fdfw to ens for high DA
    if load:
#         eFdfwEnsNMDA = np.load(dataFile)['eFdfwEnsNMDA']
#         wFdfwEnsNMDA = np.load(dataFile)['wFdfwEnsNMDA']
#     else:
        eFdfwEnsNMDA = None
        for n in range(nTrains):
            print("fdfw-ens encoding NMDA trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                fAMPA, fNMDA, fGABA, fS,
                dFdfwEnsAMPA=dFdfwEnsAMPA, dFdfwEnsNMDA=dFdfwEnsNMDA,
                wPreFdfw=wPreFdfw, eFdfwEnsNMDA=eFdfwEnsNMDA, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=3, DA=lambda t: 1)
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
                ax.set(ylim=((0, 40)))
                plt.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdfwEnsNMDA_%s.pdf'%n)
                plt.close('all')

    # Readout decoders for ens; assume high DA condition and inh-ens, and only compute NMDA decoders
    if load:
#         dEnsEnsAMPA = rng.uniform(0, 1e-7, size=(N, 1))
#         dEnsEnsNMDA = np.load(dataFile)['dEnsEnsNMDA']
#         dEnsInhAMPA = np.array(dEnsEnsAMPA)
#         dEnsInhNMDA = np.array(dEnsEnsNMDA)
#     else:
        targetsNMDA = np.zeros((1, 1))
        asNMDA = np.zeros((1, N))
        for n in range(Ndec):
            print("ens decoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                fAMPA, fNMDA, fGABA, fS,
                wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
            asNMDA = np.append(asNMDA, fNMDA.filt(data['ens']), axis=0)
            targetsNMDA = np.append(targetsNMDA, fNMDA.filt(fNMDA.filt(data['inpt'])), axis=0)
        dEnsEnsAMPA = rng.uniform(0, 1e-6, size=(N, 1))  # negligable feedback
        dEnsEnsNMDA, _ = nnls(asNMDA, np.ravel(targetsNMDA))
        dEnsEnsNMDA = dEnsEnsNMDA.reshape((N, 1))
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
#     else:
        eEnsEnsAMPA = None
        wEnsEnsAMPA = None
        eEnsEnsNMDA = None
        wEnsEnsNMDA = None
        for n in range(nTrains):
            print("ens-ens2 and ens-inh encoding trial %s"%n)
            u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=n)
            stim = lambda t: u[int(t/dt)]
            data = goBio(
                fAMPA, fNMDA, fGABA, fS,
                dEnsEnsAMPA=dEnsEnsAMPA, dEnsEnsNMDA=dEnsEnsNMDA,
                wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, eEnsEnsAMPA=eEnsEnsAMPA, eEnsEnsNMDA=eEnsEnsNMDA, 
                N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=4, DA=lambda t: 1)
            eEnsEnsAMPA = data['eEnsEnsAMPA']
            wEnsEnsAMPA = data['wEnsEnsAMPA']
            eEnsEnsNMDA = data['eEnsEnsNMDA']
            wEnsEnsNMDA = data['wEnsEnsNMDA']
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
                eEnsEnsAMPA=eEnsEnsAMPA,
                wEnsEnsAMPA=wEnsEnsAMPA,
                eEnsEnsNMDA=eEnsEnsNMDA,
                wEnsEnsNMDA=wEnsEnsNMDA,
                wEnsInhAMPA=wEnsInhAMPA,
                wEnsInhNMDA=wEnsInhNMDA)
            aTarEns = fS.filt(data['fdbk'])
            aEns = fS.filt(data['ens'])
            aInh = fS.filt(data['inh'])
            for n in range(N):
                fig, ax = plt.subplots(ncols=1, nrows=1)
                ax.plot(data['times'], aTarEns[:,n], alpha=0.5, label='target')
                ax.plot(data['times'], aEns[:,n], alpha=0.5, label='ens')
                ax.plot(data['times'], aInh[:,n], alpha=0.25, label='inh')
                ax.set(ylim=((0, 40)))
                ax.legend()
                plt.savefig('plots/tuning/gatedMemroy_eFdbkEns_%s.pdf'%n)
                plt.close('all')
    
    for test in range(nTests):
        # Test integration in high vs low DA; assume inh-ens and no inh-fdfw
        print("integration test %s, high DA"%test)
        u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=100+test)
        stim = lambda t: u[int(t/dt)]
        data = goBio(
            fAMPA, fNMDA, fGABA, fS,
            wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
            N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
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
        fig.savefig("plots/gatedMemory_goBio_intg_highDA_%s.pdf"%test)
        print("integration test %s, low DA"%test)
        data = goBio(
            fAMPA, fNMDA, fGABA, fS,
            wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA,wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns,
            N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 0)
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
        fig.savefig("plots/gatedMemory_goBio_intg_lowDA_%s.pdf"%test)

        # Test integration in with inh-fdfw and high vs low DA
        print("inhibition test %s, high DA"%test)
        u = makeSignalCutoff(t=t, dt=dt, period=t, f=fNMDA, seed=100+test)
        stim = lambda t: u[int(t/dt)]
        data = goBio(
            fAMPA, fNMDA, fGABA, fS,
            wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, wInhFdfw=wInhFdfw,
            N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 1)
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
        fig.savefig("plots/gatedMemory_goBio_inh_highDA_%s.pdf"%test)

        print("inhibition test %s, low DA"%test)
        data = goBio(
            fAMPA, fNMDA, fGABA, fS,
            wPreFdfw=wPreFdfw, wFdfwEnsAMPA=tAMPA*wFdfwEnsAMPA, wFdfwEnsNMDA=tNMDA*wFdfwEnsNMDA, wEnsEnsAMPA=wEnsEnsAMPA, wEnsEnsNMDA=wEnsEnsNMDA, wEnsInhAMPA=wEnsInhAMPA, wEnsInhNMDA=wEnsInhNMDA, wInhEns=wInhEns, wInhFdfw=wInhFdfw,
            N=N, NPre=NPre, stim=stim, t=t, dt=dt, stage=0, DA=lambda t: 0)
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
        fig.savefig("plots/gatedMemory_goBio_inh_lowDA_%s.pdf"%test)

hard(load=True)

def makeSignal(t, f, dt=0.001, value=1.0, nFilts=2, seed=0):
    stim = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=seed)
    stim2 = nengo.processes.WhiteSignal(period=t, high=1.0, rms=0.5, seed=100+seed)
    with nengo.Network() as model:
        u = nengo.Node(stim)
        u2 = nengo.Node(stim2)
        pU = nengo.Probe(u, synapse=None)
        pU2 = nengo.Probe(u2, synapse=None)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t+dt, progress_bar=False)
    u = sim.data[pU][::2]
    u2 = sim.data[pU2][::2]
    for n in range(nFilts):
        u = f.filt(u, dt=dt)
        u2 = f.filt(u2, dt=dt)
    norm = value / np.max(np.abs(u))
    norm2 = value / np.max(np.abs(u2))
    stim = sim.data[pU][::2] * norm
    stim2 = sim.data[pU2][::2] * norm2
    mirrored = np.concatenate((stim, -stim))
    mirrored2 = np.concatenate((stim2, -stim2))
    output = np.hstack((mirrored, mirrored2))
    return lambda t: output[int(t/dt)]

ax.plot(f.filt(data['tar2']), label='target')
ax.plot(np.dot(f2.filt(data['ens']), d2), alpha=0.5, label='ens')
ax.plot(data['tarState'], alpha=0.5, label="tarEns2")
ax.legend()
fig.savefig("plots/multiplyNew_tarState.pdf")
raise

preInpt2 = nengo.Ensemble(NPre, 1, radius=3, max_rates=m, seed=seed)
preIntg2 = nengo.Ensemble(NPre, 1, max_rates=m, seed=seed)
ens3 = nengo.Ensemble(N, 1, max_rates=m, intercepts=i, neuron_type=neuron_type, seed=seed)
nengo.Connection(inpt, preInpt2, synapse=f, seed=seed)
nengo.Connection(intg, preIntg2, synapse=f, seed=seed)
c4a = nengo.Connection(preInpt2, ens3, synapse=f1a, solver=NoSolver(d1a), seed=seed)
c4b = nengo.Connection(preIntg2, ens3, synapse=f1b, solver=NoSolver(d1b), seed=seed+1)
learnEncoders(c3, ens3, fS)
pTarEns2 = nengo.Probe(ens3.neurons, synapse=None)
setWeights(c4a, d1a, e1a)
setWeights(c4b, d1b, e1b)
