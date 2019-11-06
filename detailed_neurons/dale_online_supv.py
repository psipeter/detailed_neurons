import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengolib
from neuron_models import AdaptiveLIFT, WilsonEuler
sns.set(style='white', palette='dark')

class LearningNode(nengo.Node):
	def __init__(self, N_E, N_I, N_ens, w_E, w_I,
			dt=0.001, learn=True, k_E_pos=3e-9, k_E_neg=1e-8, k_I_pos=1e-7, k_I_neg=3e-9, discount=lambda t: 1, seed=0):
		self.N_E = N_E
		self.N_I = N_I
		self.N_ens = N_ens
		self.size_in = N_E + N_I + N_ens + N_ens
		self.size_out = N_ens
		self.k_E_pos = k_E_pos
		self.k_E_neg = k_E_neg
		self.k_I_pos = k_I_pos
		self.k_I_neg = k_I_neg
		self.w_E = w_E
		self.w_I = w_I
		self.dt = dt
		self.learn = learn
		self.discount = discount
		self.rng = np.random.RandomState(seed=seed)
		super(LearningNode, self).__init__(
			self.step, size_in=self.size_in, size_out=self.size_out)

	def step(self, t, x):
		a_E = x[:self.N_E]
		a_I = x[self.N_E: self.N_E+self.N_I]
		if self.learn:
			for idx_post in range(self.N_ens):
				a_post = x[self.N_E+self.N_I+idx_post]
				a_supv = x[self.N_E+self.N_I+self.N_ens+idx_post]
				delta_a = a_post - a_supv
				active_E = np.where(a_E > 0)[0]
				if len(active_E) > 0:
					idx_pre_E = self.rng.randint(0, len(active_E))
					if delta_a < 0:  # if underactive, increase weight from active E neuron (more positive)
						self.w_E[idx_pre_E, idx_post] += delta_a * -self.k_E_pos/(0.001/self.dt) * a_E[idx_pre_E]
					if delta_a > 0:  # if overactive, decrease weight from active E neuron (less positive)
						self.w_E[idx_pre_E, idx_post] += delta_a * -self.k_E_neg/(0.001/self.dt) * a_E[idx_pre_E]
					self.w_E[idx_pre_E, idx_post] = np.maximum(0, self.w_E[idx_pre_E, idx_post])
				active_I = np.where(a_I > 0)[0]
				if len(active_I) > 0:
					idx_pre_I = self.rng.randint(0, len(active_I))
					if delta_a > 0:  # if overactive, decrease weight from active I neuron (more negative)
						self.w_I[idx_pre_I, idx_post] += delta_a * -self.k_I_pos/(0.001/self.dt)
					if delta_a < 0:  # if underactive, increase weight from active I neuron (less negative)
						self.w_I[idx_pre_I, idx_post] += delta_a * -self.k_I_neg/(0.001/self.dt)
					self.w_I[idx_pre_I, idx_post] = np.minimum(0, self.w_I[idx_pre_I, idx_post])
		assert np.sum(self.w_E >= 0) == self.N_E * self.N_ens
		assert np.sum(self.w_I <= 0) == self.N_I * self.N_ens
		J_E = np.dot(a_E, self.w_E)
		J_I = np.dot(a_I, self.w_I)
		return self.discount(t) * (J_E + J_I)



def feedforward():
	'''train'''
	N_E = 100
	N_I = 400
	N_ens = 20
	seed = 1
	a = nengo.dists.Uniform(10, 20).sample(n=N_ens, rng=np.random.RandomState(seed=seed))
	b = np.zeros((N_ens))
	w_E = np.zeros((N_E, N_ens))
	w_I = np.zeros((N_I, N_ens))
	neuron_type = nengo.LIF()
	# neuron_type = AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1)
	# neuron_type = WilsonEuler()
	dt = 0.001
	# dt = 0.000025

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(nengo.processes.WhiteSignal(period=200, high=1, rms=0.4, seed=0))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
		node = LearningNode(N_E, N_I, N_ens, w_E, w_I, dt=dt, k_E_pos=1e-8, k_E_neg=1e-7, k_I_pos=1e-7, k_I_neg=1e-8)

		pre = nengo.Ensemble(N_E, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, neuron_type=neuron_type, seed=seed)
		# post = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed)
		nengo.Connection(x, supv, synapse=0.01, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, node[0:N_E], synapse=0.005)
		nengo.Connection(I.neurons, node[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(post.neurons, node[N_E+N_I: N_E+N_I+N_ens], synapse=0.1)
		nengo.Connection(supv.neurons, node[N_E+N_I+N_ens:N_E+N_I+N_ens+N_ens], synapse=0.1)
		nengo.Connection(node, post.neurons, synapse=None)
		# conn = nengo.Connection(post, error, synapse=0.05, function=lambda x: 0, learning_rule_type=nengo.PES(learning_rate=5e-5))
		conn = nengo.Connection(post, error, synapse=0.1, solver=nengo.solvers.NoSolver(np.zeros((N_ens, 1))),
			learning_rule_type=nengo.PES(learning_rate=6e-4))
		nengo.Connection(x, error, synapse=0.1, transform=-1)
		nengo.Connection(error, conn.learning_rule)

		# p_E = nengo.Probe(E, synapse=0.05)
		# p_I = nengo.Probe(I, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.1, sample_every=0.001)
		# p_supv = nengo.Probe(supv, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.1, sample_every=0.001)
		p_x = nengo.Probe(x, synapse=0.1, sample_every=0.001)
		p_error = nengo.Probe(error, synapse=0.1, sample_every=0.001)
		p_weights = nengo.Probe(conn, 'weights', synapse=None, sample_every=0.001)

	with nengo.Simulator(network, seed=seed, dt=dt) as sim:
		sim.run(200)

	xhat_post = np.zeros((sim.trange()[::int(0.001/dt)].shape[0], 1))
	for t in range(xhat_post.shape[0]):
		xhat_post[t] = np.dot(sim.data[p_post_neurons][t], sim.data[p_weights][t].T)
	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange()[::int(0.001/dt)], xhat_post, alpha=0.5, label='post')
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_x], alpha=0.5, label='target')
	# ax.plot(sim.trange(), sim.data[p_error], alpha=0.5, label='error')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange()[::int(0.001/dt)], sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='train')
	ax2.set(ylabel='Firing Rate', ylim=((0, 200)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 200)), title='post')

	'''test'''
	w_E = node.w_E
	w_I = node.w_I
	d_post = sim.data[p_weights][-1].T
	np.savez("data/dale_online_supv_feedforward_withbias_%s.npz"%neuron_type, w_E=w_E, w_I=w_I, d_post=d_post)

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(nengo.processes.WhiteSignal(period=200, high=1, rms=0.4, seed=1))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

		pre = nengo.Ensemble(N_E, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, neuron_type=neuron_type, seed=seed)
		# post = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed)
		nengo.Connection(x, supv, synapse=0.01, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, post.neurons, synapse=0.005, transform=w_E.T, seed=seed)
		nengo.Connection(I.neurons, post.neurons, synapse=0.01, transform=w_I.T, seed=seed)

		# p_E = nengo.Probe(E, synapse=0.05)
		# p_I = nengo.Probe(I, synapse=0.05)
		p_post = nengo.Probe(post, synapse=0.1, solver=nengo.solvers.NoSolver(d_post), sample_every=0.001)
		# p_supv = nengo.Probe(supv, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.1, sample_every=0.001)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.1, sample_every=0.001)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.1, sample_every=0.001)

	with nengo.Simulator(network, seed=seed, dt=dt) as sim:
		sim.run(20)

	nrmse_post = nengolib.signal.nrmse(sim.data[p_post], target=sim.data[p_x])
	# nrmse_supv = nengolib.signal.nrmse(sim.data[p_supv], target=sim.data[p_x])
	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post], alpha=0.5, label='post, NRMSE=%.3f'%nrmse_post)
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv, NRMSE=%.3f'%nrmse_supv)
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_x], alpha=0.5, label='target')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange()[::int(0.001/dt)], sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='test')
	ax2.set(ylabel='Firing Rate', ylim=((0, 200)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 200)), title='post')
	plt.savefig("plots/dale_online_supv_feedforward_withbias_%s.png"%neuron_type)
	plt.show()



def multiply():
	'''train'''
	N_E = 100
	N_I = 400
	N_ens = 20
	seed = 1
	a = nengo.dists.Uniform(10, 20).sample(n=N_ens, rng=np.random.RandomState(seed=seed))
	b = np.zeros((N_ens))
	w_E = np.zeros((N_E, N_ens))
	w_I = np.zeros((N_I, N_ens))
	# neuron_type = nengo.LIF()
	# neuron_type = AdaptiveLIFT(tau_adapt=0.1, inc_adapt=0.1)
	neuron_type = WilsonEuler()
	# dt = 0.001
	dt = 0.000025

	with nengo.Network(seed=seed) as network:
		u1 = nengo.Node(nengo.processes.WhiteSignal(period=500, high=1, rms=0.8, seed=0))
		u2 = nengo.Node(nengo.processes.WhiteSignal(period=500, high=1, rms=0.8, seed=1))
		u = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
		node = LearningNode(N_E, N_I, N_ens, w_E, w_I, dt=dt, k_E_pos=1e-8, k_E_neg=1e-7, k_I_pos=1e-7, k_I_neg=1e-8)

		pre = nengo.Ensemble(N_E, 2, seed=seed)
		E = nengo.Ensemble(N_E, 2, max_rates=nengo.dists.Uniform(20, 40), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 2, max_rates=nengo.dists.Uniform(10, 30), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, neuron_type=neuron_type, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8),
			neuron_type=neuron_type, seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		nengo.Connection(u1, u[0], synapse=None, seed=seed)
		nengo.Connection(u2, u[1], synapse=None, seed=seed)
		nengo.Connection(u, x, synapse=None, seed=seed, function=lambda x: x[0]*x[1])
		nengo.Connection(x, supv, synapse=0.01, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, node[0:N_E], synapse=0.005)
		nengo.Connection(I.neurons, node[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(post.neurons, node[N_E+N_I: N_E+N_I+N_ens], synapse=0.1)
		nengo.Connection(supv.neurons, node[N_E+N_I+N_ens:N_E+N_I+N_ens+N_ens], synapse=0.1)
		nengo.Connection(node, post.neurons, synapse=None)
		conn = nengo.Connection(post, error, synapse=0.1, solver=nengo.solvers.NoSolver(np.zeros((N_ens, 1))),
			learning_rule_type=nengo.PES(learning_rate=5e-4))
		nengo.Connection(x, error, synapse=0.1, transform=-1)
		nengo.Connection(error, conn.learning_rule)

		# p_E = nengo.Probe(E, synapse=0.05)
		# p_I = nengo.Probe(I, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.1, sample_every=0.001)
		# p_supv = nengo.Probe(supv, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.1, sample_every=0.001)
		p_x = nengo.Probe(x, synapse=0.1, sample_every=0.001)
		p_error = nengo.Probe(error, synapse=0.1, sample_every=0.001)
		p_weights = nengo.Probe(conn, 'weights', synapse=None, sample_every=0.001)

	with nengo.Simulator(network, seed=seed, dt=dt) as sim:
		sim.run(3)

	xhat_post = np.zeros((sim.trange()[::int(0.001/dt)].shape[0], 1))
	for t in range(xhat_post.shape[0]):
		xhat_post[t] = np.dot(sim.data[p_post_neurons][t], sim.data[p_weights][t].T)
	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange()[::int(0.001/dt)], xhat_post, alpha=0.5, label='post')
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_x], alpha=0.5, label='target')
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	# ax.plot(sim.trange(), sim.data[p_error], alpha=0.5, label='error')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange()[::int(0.001/dt)], sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='train')
	ax2.set(ylabel='Firing Rate', ylim=((0, 200)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 200)), title='post')

	'''test'''
	w_E = node.w_E
	w_I = node.w_I
	d_post = sim.data[p_weights][-1].T
	np.savez("data/dale_online_supv_multiply%s.npz"%neuron_type, w_E=w_E, w_I=w_I, d_post=d_post)

	with nengo.Network(seed=seed) as network:
		u1 = nengo.Node(nengo.processes.WhiteSignal(period=500, high=1, rms=0.8, seed=2))
		u2 = nengo.Node(nengo.processes.WhiteSignal(period=500, high=1, rms=0.8, seed=3))
		u = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

		pre = nengo.Ensemble(N_E, 2, seed=seed)
		E = nengo.Ensemble(N_E, 2, max_rates=nengo.dists.Uniform(20, 40), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 2, max_rates=nengo.dists.Uniform(10, 30), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, neuron_type=neuron_type, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		nengo.Connection(u1, u[0], synapse=None, seed=seed)
		nengo.Connection(u2, u[1], synapse=None, seed=seed)
		nengo.Connection(u, x, synapse=None, seed=seed, function=lambda x: x[0]*x[1])
		nengo.Connection(x, supv, synapse=0.01, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, post.neurons, synapse=0.005, transform=w_E.T, seed=seed)
		nengo.Connection(I.neurons, post.neurons, synapse=0.01, transform=w_I.T, seed=seed)

		# p_E = nengo.Probe(E, synapse=0.05)
		# p_I = nengo.Probe(I, synapse=0.05)
		p_post = nengo.Probe(post, synapse=0.1, solver=nengo.solvers.NoSolver(d_post), sample_every=0.001)
		# p_supv = nengo.Probe(supv, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.1, sample_every=0.001)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.1, sample_every=0.001)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.1, sample_every=0.001)

	with nengo.Simulator(network, seed=seed, dt=dt) as sim:
		sim.run(2)

	nrmse_post = nengolib.signal.nrmse(sim.data[p_post], target=sim.data[p_x])
	# nrmse_supv = nengolib.signal.nrmse(sim.data[p_supv], target=sim.data[p_x])
	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post], alpha=0.5, label='post, NRMSE=%.3f'%nrmse_post)
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv, NRMSE=%.3f'%nrmse_supv)
	ax.plot(sim.trange()[::int(0.001/dt)], sim.data[p_x], alpha=0.5, label='target')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange()[::int(0.001/dt)], sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange()[::int(0.001/dt)], sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='test')
	ax2.set(ylabel='Firing Rate', ylim=((0, 400)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 400)), title='post')
	plt.savefig("plots/dale_online_supv_multiply_%s.png"%neuron_type)
	plt.show()



def integrate():
	'''train initial guess at feedback weights using feedforward supervision'''
	N_E = 100
	N_I = 100
	seed = 1
	a_E = nengo.dists.Uniform(10, 20).sample(n=N_E, rng=np.random.RandomState(seed=seed))
	a_I = nengo.dists.Uniform(10, 20).sample(n=N_I, rng=np.random.RandomState(seed=seed))
	b_E = np.zeros((N_E))
	b_I = np.zeros((N_I))
	w_EE = np.zeros((N_E, N_E))
	w_EI = np.zeros((N_E, N_I))
	w_IE = np.zeros((N_I, N_E))
	w_II = np.zeros((N_I, N_I))
	# w_EE = np.load("data/dale_online_supv_integrate.npz")['w_EE']
	# w_EI = np.load("data/dale_online_supv_integrate.npz")['w_EI']
	# w_IE = np.load("data/dale_online_supv_integrate.npz")['w_IE']
	# w_II = np.load("data/dale_online_supv_integrate.npz")['w_II']
	neuron_type = nengo.LIF()
	dt = 0.001

	with nengo.Network(seed=seed) as network:
		# nodes
		u = nengo.Node(nengo.processes.WhiteSignal(period=50, high=1, rms=0.5, seed=3))
		node_E = LearningNode(N_E, N_I, N_E, w_EE, w_IE, dt=dt, k_E_pos=1e-8, k_E_neg=1e-7, k_I_pos=1e-7, k_I_neg=1e-8)
		node_I = LearningNode(N_E, N_I, N_I, w_EI, w_II, dt=dt, k_E_pos=1e-8, k_E_neg=1e-7, k_I_pos=1e-7, k_I_neg=1e-8)

		# ensembles
		pre_u = nengo.Ensemble(100, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post_E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post_I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)

		# input and supervised connections
		nengo.Connection(u, pre_u, synapse=None, seed=seed)
		nengo.Connection(pre_u, E, synapse=0.1, seed=seed)
		nengo.Connection(pre_u, I, synapse=0.1, seed=seed)
		nengo.Connection(pre_u, supv_E, synapse=0.1, seed=seed)
		nengo.Connection(pre_u, supv_I, synapse=0.1, seed=seed)

		# connections into post_E
		nengo.Connection(E.neurons, node_E[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node_E[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(post_E.neurons, node_E[N_E+N_I: N_E+N_I+N_E], synapse=0.1)
		nengo.Connection(supv_E.neurons, node_E[N_E+N_I+N_E: N_E+N_I+N_E+N_E], synapse=0.1)
		nengo.Connection(node_E, post_E.neurons, synapse=None)

		# connections into post_I
		nengo.Connection(E.neurons, node_I[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node_I[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(post_I.neurons, node_I[N_E+N_I: N_E+N_I+N_I], synapse=0.1)
		nengo.Connection(supv_I.neurons, node_I[N_E+N_I+N_I: N_E+N_I+N_I+N_I], synapse=0.1)
		nengo.Connection(node_I, post_I.neurons, synapse=None)

		# probes
		p_post_E_neurons = nengo.Probe(post_E.neurons, synapse=0.1, sample_every=0.001)
		p_post_I_neurons = nengo.Probe(post_I.neurons, synapse=0.1, sample_every=0.001)
		p_supv_E_neurons = nengo.Probe(supv_E.neurons, synapse=0.1, sample_every=0.001)
		p_supv_I_neurons = nengo.Probe(supv_I.neurons, synapse=0.1, sample_every=0.001)

	with nengo.Simulator(network, seed=seed, dt=dt) as sim:
		sim.run(300)

	w_EE = node_E.w_E
	w_EI = node_I.w_E
	w_IE = node_E.w_I
	w_II = node_I.w_I
	np.savez("data/dale_online_supv_integrate.npz", w_EE=w_EE, w_EI=w_EI, w_IE=w_IE, w_II=w_II)

	T = 50000
	for chunk in range(int(sim.trange().shape[0]/T)):
		fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=((12, 12)), sharex=True)
		for n in range(10):
			ax.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_supv_E_neurons][chunk*T: (chunk+1)*T:,n], alpha=0.5)
			ax2.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_post_E_neurons][chunk*T: (chunk+1)*T:,n], alpha=0.5)
			ax3.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_supv_I_neurons][chunk*T: (chunk+1)*T,n], alpha=0.5)
			ax4.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_post_I_neurons][chunk*T: (chunk+1)*T,n], alpha=0.5)
		ax.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_E')
		ax2.set(ylabel='Firing Rate', ylim=((0, 40)), title='post_E')
		ax3.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_I')
		ax4.set(ylabel='Firing Rate', ylim=((0, 40)), title='post_I')
		plt.savefig('plots/dale_online_supv_integrate_train_rates_%s'%chunk)
		plt.close()


	'''train fine-tuned feedback weights using feedback supervision'''
	# N_E = 100
	# N_I = 100
	N_post = 20
	# seed = 1
	# a_E = nengo.dists.Uniform(10, 20).sample(n=N_E, rng=np.random.RandomState(seed=seed))
	# a_I = nengo.dists.Uniform(10, 20).sample(n=N_I, rng=np.random.RandomState(seed=seed))
	a_post = nengo.dists.Uniform(10, 20).sample(n=N_post, rng=np.random.RandomState(seed=seed))
	# b_E = np.zeros((N_E))
	# b_I = np.zeros((N_I))
	b_post = np.zeros((N_post))
	w_E = np.zeros((N_E, N_post))
	w_I = np.zeros((N_I, N_post))
	# w_EE = np.load("data/dale_online_supv_integrate.npz")['w_EE']
	# w_EI = np.load("data/dale_online_supv_integrate.npz")['w_EI']
	# w_IE = np.load("data/dale_online_supv_integrate.npz")['w_IE']
	# w_II = np.load("data/dale_online_supv_integrate.npz")['w_II']
	neuron_type = nengo.LIF()
	dt = 0.001
	T = 400
	discount_fb = lambda t: 0.9+0.1*(t/T)
	discount_supv = lambda t: 0.1-0.1*(t/T)

	with nengo.Network(seed=seed) as network:
		# nodes
		u = nengo.Node(nengo.processes.WhiteSignal(period=50, high=1, rms=0.5, seed=3))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
		node_post = LearningNode(N_E, N_I, N_post, w_E, w_I, dt=dt, k_E_pos=1e-8, k_E_neg=1e-7, k_I_pos=1e-7, k_I_neg=1e-8)
		node_E = LearningNode(N_E, N_I, N_E, w_EE, w_IE, dt=dt, k_E_pos=3e-7, k_E_neg=1e-7, k_I_pos=3e-7, k_I_neg=1e-7, discount=discount_fb)
		node_I = LearningNode(N_E, N_I, N_I, w_EI, w_II, dt=dt, k_E_pos=3e-7, k_E_neg=1e-7, k_I_pos=3e-7, k_I_neg=1e-7, discount=discount_fb)
		node_E_supv = LearningNode(N_E, N_I, N_E, w_EE, w_IE, dt=dt, learn=False, discount=discount_supv)
		node_I_supv = LearningNode(N_E, N_I, N_I, w_EI, w_II, dt=dt, learn=False, discount=discount_supv)

		# ensembles
		pre_u = nengo.Ensemble(100, 1, seed=seed)
		pre_x = nengo.Ensemble(100, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_post, 1, gain=a_post, bias=b_post, neuron_type=neuron_type, seed=seed)
		supv_post = nengo.Ensemble(N_post, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		# input and supervised connections
		nengo.Connection(u, x, synapse=1/nengolib.signal.s, seed=seed)
		nengo.Connection(u, pre_u, synapse=None, seed=seed)
		nengo.Connection(x, pre_x, synapse=None, seed=seed)
		nengo.Connection(pre_u, E, synapse=0.1, transform=0.1, seed=seed)
		nengo.Connection(pre_u, I, synapse=0.1, transform=0.1, seed=seed)
		nengo.Connection(pre_x, supv_E, synapse=0.1, seed=seed)
		nengo.Connection(pre_x, supv_I, synapse=0.1, seed=seed)
		nengo.Connection(pre_x, supv_post, synapse=0.1, seed=seed)

		# connections into E
		nengo.Connection(E.neurons, node_E[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node_E[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(E.neurons, node_E[N_E+N_I: N_E+N_I+N_E], synapse=0.1)
		nengo.Connection(supv_E.neurons, node_E[N_E+N_I+N_E: N_E+N_I+N_E+N_E], synapse=0.1)
		nengo.Connection(node_E, E.neurons, synapse=None)
		nengo.Connection(supv_E.neurons, node_E_supv[0:N_E], synapse=0.1)
		nengo.Connection(supv_I.neurons, node_E_supv[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(E.neurons, node_E_supv[N_E+N_I: N_E+N_I+N_E], synapse=0.1)
		nengo.Connection(supv_E.neurons, node_E_supv[N_E+N_I+N_E: N_E+N_I+N_E+N_E], synapse=0.1)
		nengo.Connection(node_E_supv, E.neurons, synapse=None)

		# connections into I
		nengo.Connection(E.neurons, node_I[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node_I[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(I.neurons, node_I[N_E+N_I: N_E+N_I+N_I], synapse=0.1)
		nengo.Connection(supv_I.neurons, node_I[N_E+N_I+N_I: N_E+N_I+N_I+N_I], synapse=0.1)
		nengo.Connection(node_I, I.neurons, synapse=None)
		nengo.Connection(supv_E.neurons, node_I_supv[0:N_E], synapse=0.1)
		nengo.Connection(supv_I.neurons, node_I_supv[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(I.neurons, node_I_supv[N_E+N_I: N_E+N_I+N_I], synapse=0.1)
		nengo.Connection(supv_I.neurons, node_I_supv[N_E+N_I+N_I: N_E+N_I+N_I+N_I], synapse=0.1)
		nengo.Connection(node_I_supv, I.neurons, synapse=None)

		# connections to post and output
		nengo.Connection(E.neurons, node_post[0:N_E], synapse=0.005)
		nengo.Connection(I.neurons, node_post[N_E: N_E+N_I], synapse=0.01)
		nengo.Connection(post.neurons, node_post[N_E+N_I: N_E+N_I+N_post], synapse=0.1)
		nengo.Connection(supv_post.neurons, node_post[N_E+N_I+N_post:N_E+N_I+N_post+N_post], synapse=0.1)
		nengo.Connection(node_post, post.neurons, synapse=None)
		conn = nengo.Connection(post, error, synapse=0.1, solver=nengo.solvers.NoSolver(np.zeros((N_post, 1))), learning_rule_type=nengo.PES(learning_rate=1e-3))
		nengo.Connection(x, error, synapse=0.2, transform=-1)
		nengo.Connection(error, conn.learning_rule)

		# probes
		p_E_neurons = nengo.Probe(E.neurons, synapse=0.1, sample_every=0.001)
		p_I_neurons = nengo.Probe(I.neurons, synapse=0.1, sample_every=0.001)
		p_supv_E_neurons = nengo.Probe(supv_E.neurons, synapse=0.1, sample_every=0.001)
		p_supv_I_neurons = nengo.Probe(supv_I.neurons, synapse=0.1, sample_every=0.001)
		# p_post = nengo.Probe(post, synapse=0.1, solver=nengo.solvers.NoSolver(d_post), sample_every=0.001)
		p_supv_post = nengo.Probe(supv_post, synapse=0.1, sample_every=0.001)
		p_x = nengo.Probe(x, synapse=0.2, sample_every=0.001)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.1, sample_every=0.001)
		p_supv_post_neurons = nengo.Probe(supv_post.neurons, synapse=0.1, sample_every=0.001)
		p_weights = nengo.Probe(conn, 'weights', synapse=None, sample_every=0.001)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(T)

	xhat_post = np.zeros((sim.trange()[::int(0.001/dt)].shape[0], 1))
	for t in range(xhat_post.shape[0]):
		xhat_post[t] = np.dot(sim.data[p_post_neurons][t], sim.data[p_weights][t].T)

	T = 50000
	for chunk in range(int(sim.trange().shape[0]/T)):
		fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=((12, 12)), sharex=True)
		for n in range(10):
			ax.plot(sim.trange()[chunk*T: (chunk+1)*T], sim.data[p_supv_E_neurons][chunk*T: (chunk+1)*T:,n], alpha=0.5)
			ax2.plot(sim.trange()[chunk*T: (chunk+1)*T], sim.data[p_E_neurons][chunk*T: (chunk+1)*T:,n], alpha=0.5)
			ax3.plot(sim.trange()[chunk*T: (chunk+1)*T], sim.data[p_supv_I_neurons][chunk*T: (chunk+1)*T,n], alpha=0.5)
			ax4.plot(sim.trange()[chunk*T: (chunk+1)*T], sim.data[p_I_neurons][chunk*T: (chunk+1)*T,n], alpha=0.5)
		ax.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_E')
		ax2.set(ylabel='Firing Rate', ylim=((0, 40)), title='E')
		ax3.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_I')
		ax4.set(ylabel='Firing Rate', ylim=((0, 40)), title='I')
		plt.savefig('plots/dale_online_supv_integrate_train2_rates_%s'%chunk)
		plt.close()

		fig, ax = plt.subplots(1, 1, figsize=((12, 12)))
		ax.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], xhat_post[chunk*T: (chunk+1)*T], alpha=0.5, label='post')
		ax.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_supv_post][chunk*T: (chunk+1)*T], alpha=0.5, label='supv')
		ax.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_x][chunk*T: (chunk+1)*T], alpha=0.5, label='target')
		plt.savefig('plots/dale_online_supv_integrate_train2_values_%s'%chunk)
		plt.legend()
		plt.close()

	w_EE = node_E.w_E
	w_EI = node_I.w_E
	w_IE = node_E.w_I
	w_II = node_I.w_I
	w_E_post = node_post.w_E
	w_I_post = node_post.w_I
	d_post = sim.data[p_weights][-1].T
	np.savez("data/dale_online_supv_integrate_train2.npz", w_EE=w_EE, w_EI=w_EI, w_IE=w_IE, w_II=w_II, w_E_post=w_E_post, w_I_post=w_I_post)

	'''test'''
	# N_E = 50
	# N_I = 50
	# seed = 1
	# a_E = nengo.dists.Uniform(10, 20).sample(n=N_E, rng=np.random.RandomState(seed=seed))
	# a_I = nengo.dists.Uniform(10, 20).sample(n=N_I, rng=np.random.RandomState(seed=seed))
	# a_post = nengo.dists.Uniform(10, 20).sample(n=N_post, rng=np.random.RandomState(seed=seed))
	# b_E = np.zeros((N_E))
	# b_I = np.zeros((N_I))
	# b_post = np.zeros((N_post))
	# w_EE = np.load("data/dale_online_supv_integrate.npz")['w_EE']
	# w_EI = np.load("data/dale_online_supv_integrate.npz")['w_EI']
	# w_IE = np.load("data/dale_online_supv_integrate.npz")['w_IE']
	# w_II = np.load("data/dale_online_supv_integrate.npz")['w_II']
	# neuron_type = nengo.LIF()
	# dt = 0.001

	with nengo.Network(seed=seed) as network:
		# nodes
		u = nengo.Node(nengo.processes.WhiteSignal(period=50, high=1, rms=0.5, seed=3))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

		# ensembles
		pre_u = nengo.Ensemble(100, 1, seed=seed)
		pre_x = nengo.Ensemble(100, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		supv_I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(10, 30), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)
		post = nengo.Ensemble(N_post, 1, gain=a_post, bias=b_post, neuron_type=neuron_type, seed=seed)
		supv_post = nengo.Ensemble(N_post, 1, max_rates=nengo.dists.Uniform(20, 40), intercepts=nengo.dists.Uniform(-0.8, 0.8), neuron_type=neuron_type, seed=seed)

		# input and supervised connections
		nengo.Connection(u, x, synapse=1/nengolib.signal.s, seed=seed)
		nengo.Connection(u, pre_u, synapse=None, seed=seed)
		nengo.Connection(x, pre_x, synapse=None, seed=seed)
		nengo.Connection(pre_u, E, synapse=0.1, transform=0.1, seed=seed)
		nengo.Connection(pre_u, I, synapse=0.1, transform=0.1, seed=seed)
		nengo.Connection(pre_x, supv_E, synapse=0.1, seed=seed)
		nengo.Connection(pre_x, supv_I, synapse=0.1, seed=seed)
		nengo.Connection(pre_x, supv_post, synapse=0.1, seed=seed)

		# connections into E
		nengo.Connection(E.neurons, E.neurons, synapse=0.2, transform=w_EE.T, seed=seed)
		nengo.Connection(I.neurons, E.neurons, synapse=0.01, transform=w_IE.T, seed=seed)
		# connections into I
		nengo.Connection(E.neurons, I.neurons, synapse=0.2, transform=w_EI.T, seed=seed)
		nengo.Connection(I.neurons, I.neurons, synapse=0.01, transform=w_II.T, seed=seed)
		# connections into post
		nengo.Connection(E.neurons, post.neurons, synapse=0.005, transform=w_E_post.T, seed=seed)
		nengo.Connection(I.neurons, post.neurons, synapse=0.01, transform=w_I_post.T, seed=seed)

		# probes
		p_E_neurons = nengo.Probe(E.neurons, synapse=0.1, sample_every=0.001)
		p_I_neurons = nengo.Probe(I.neurons, synapse=0.1, sample_every=0.001)
		p_supv_E_neurons = nengo.Probe(supv_E.neurons, synapse=0.1, sample_every=0.001)
		p_supv_I_neurons = nengo.Probe(supv_I.neurons, synapse=0.1, sample_every=0.001)
		p_post = nengo.Probe(post, synapse=0.1, solver=nengo.solvers.NoSolver(d_post), sample_every=0.001)
		p_x = nengo.Probe(x, synapse=0.2, sample_every=0.001)
		p_supv_post_neurons = nengo.Probe(supv_post.neurons, synapse=0.1, sample_every=0.001)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(50)

	fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=((12, 12)), sharex=True)
	for n in range(10):
		ax.plot(sim.trange(), sim.data[p_supv_E_neurons][::,n], alpha=0.5)
		ax2.plot(sim.trange(), sim.data[p_E_neurons][::,n], alpha=0.5)
		ax3.plot(sim.trange(), sim.data[p_supv_I_neurons][:,n], alpha=0.5)
		ax4.plot(sim.trange(), sim.data[p_I_neurons][:,n], alpha=0.5)
	ax.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_E')
	ax2.set(ylabel='Firing Rate', ylim=((0, 40)), title='E')
	ax3.set(ylabel='Firing Rate', ylim=((0, 40)), title='supv_I')
	ax4.set(ylabel='Firing Rate', ylim=((0, 40)), title='I')
	plt.savefig('plots/dale_online_supv_integrate_test_rates.png')
	plt.close()

	nrmse_post = nengolib.signal.nrmse(sim.data[p_post], target=sim.data[p_x])
	# nrmse_supv = nengolib.signal.nrmse(sim.data[p_supv][chunk*T: (chunk+1)*T], target=sim.data[p_x][chunk*T: (chunk+1)*T])
	fig, ax = plt.subplots(1, 1, figsize=((12, 12)))
	ax.plot(sim.trange(), sim.data[p_post], alpha=0.5, label='post, NRMSE=%.3f'%nrmse_post)
	# ax.plot(sim.trange()[chunk*T: (chunk+1)*T: int(0.001/dt)], sim.data[p_supv_post][chunk*T: (chunk+1)*T], alpha=0.5, label='supv, NRMSE=%.3f'%nrmse_supv)
	ax.plot(sim.trange(), sim.data[p_x], alpha=0.5, label='target')
	plt.savefig('plots/dale_online_supv_integrate_test_values.png')
	plt.legend()
	plt.close()


# feedforward()
# multiply()
integrate()