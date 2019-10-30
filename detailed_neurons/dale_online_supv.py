import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='dark')

class LearningNode(nengo.Node):
	def __init__(self, N_E, N_I, N_ens, w_E, w_I, k_E_pos=1e-9, k_E_neg=3e-9, k_I_pos=3e-8, k_I_neg=1e-9, seed=0):
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
		self.rng = np.random.RandomState(seed=seed)
		super(LearningNode, self).__init__(
			self.step, size_in=self.size_in, size_out=self.size_out)

	def step(self, t, x):
		a_E = x[:self.N_E]
		a_I = x[self.N_E: self.N_E+self.N_I]
		idx_post = self.rng.randint(0, self.N_ens)
		a_post = x[self.N_E+self.N_I+idx_post]
		a_supv = x[self.N_E+self.N_I+self.N_ens+idx_post]
		delta_a = a_post - a_supv
		active_E = np.where(a_E > 0)[0]
		if len(active_E) > 0:
			idx_pre_E = self.rng.randint(0, len(active_E))
			if delta_a < 0:  # if underactive, increase weight from active E neuron (more positive)
				self.w_E[idx_pre_E, idx_post] += delta_a * -self.k_E_pos * a_E[idx_pre_E]
			if delta_a > 0:  # if overactive, decrease weight from active E neuron (less positive)
				self.w_E[idx_pre_E, idx_post] += delta_a * -self.k_E_neg * a_E[idx_pre_E]
			self.w_E[idx_pre_E, idx_post] = np.maximum(0, self.w_E[idx_pre_E, idx_post])
		active_I = np.where(a_I > 0)[0]
		if len(active_I) > 0:
			idx_pre_I = self.rng.randint(0, len(active_I))
			if delta_a > 0:  # if overactive, decrease weight from active I neuron (more negative)
				self.w_I[idx_pre_I, idx_post] += delta_a * -self.k_I_pos
			if delta_a < 0:  # if underactive, increase weight from active I neuron (less negative)
				self.w_I[idx_pre_I, idx_post] += delta_a * -self.k_I_neg
			self.w_I[idx_pre_I, idx_post] = np.minimum(0, self.w_I[idx_pre_I, idx_post])
		assert np.sum(self.w_E >= 0) == self.N_E * self.N_ens
		assert np.sum(self.w_I <= 0) == self.N_I * self.N_ens
		J_E = np.dot(a_E, self.w_E)
		J_I = np.dot(a_I, self.w_I)
		return J_E + J_I

def feedforward():
	'''train'''
	N_E = 100
	N_I = 200
	N_ens = 10
	seed = 1
	a = nengo.dists.Uniform(1, 10).sample(n=N_ens, rng=np.random.RandomState(seed=seed))
	b = np.zeros((N_ens))
	w_E = np.zeros((N_E, N_ens))
	w_I = np.zeros((N_I, N_ens))

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(lambda t: np.sin(t))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
		node = LearningNode(N_E, N_I, N_ens, w_E, w_I)

		pre = nengo.Ensemble(N_E, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(100, 200), seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(200, 400), seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(100, 200), intercepts=nengo.dists.Uniform(-0.8, 0.8), seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed)
		nengo.Connection(x, supv, synapse=0.1, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, node[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node[N_E: N_E+N_I], synapse=0.001)
		nengo.Connection(post.neurons, node[N_E+N_I: N_E+N_I+N_ens], synapse=0.05)
		nengo.Connection(supv.neurons, node[N_E+N_I+N_ens:N_E+N_I+N_ens+N_ens], synapse=0.05)
		nengo.Connection(node, post.neurons, synapse=None)
		conn = nengo.Connection(post, error, synapse=0.05, function=lambda x: 0, learning_rule_type=nengo.PES(learning_rate=1e-5))
		nengo.Connection(supv, error, synapse=0.05, transform=-1)
		nengo.Connection(error, conn.learning_rule)

		p_E = nengo.Probe(E, synapse=0.05)
		p_I = nengo.Probe(I, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.05)
		p_supv = nengo.Probe(supv, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.05)
		p_error = nengo.Probe(error, synapse=0.05)
		p_weights = nengo.Probe(conn, 'weights', synapse=None)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(200)


	# xhat_post = np.zeros((sim.trange().shape[0], 1))
	# for t in range(xhat_post.shape[0]):
	# 	xhat_post[t] = np.dot(sim.data[p_post_neurons][t], sim.data[p_weights][t].T)
	# fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	# ax.plot(sim.trange(), xhat_post, alpha=0.5, label='post')
	# ax.plot(sim.trange(), sim.data[p_x], alpha=0.5, label='target')
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	# ax.plot(sim.trange(), sim.data[p_error], alpha=0.5, label='error')
	# ax.legend(loc='upper right')
	# for n in range(N_ens):
	# 	ax3.plot(sim.trange(), sim.data[p_post_neurons][:,n], alpha=0.5)
	# 	ax2.plot(sim.trange(), sim.data[p_supv_neurons][:,n], alpha=0.5)
	# ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='train')
	# ax2.set(ylabel='Firing Rate', ylim=((0, 400)), title='supervisor')
	# ax3.set(ylabel='Firing Rate', ylim=((0, 400)), title='post')

	'''test'''
	w_E = node.w_E
	w_I = node.w_I
	d_post = sim.data[p_weights][-1].T

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(lambda t: np.sin(t))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

		pre = nengo.Ensemble(N_E, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(100, 200), seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(200, 400), seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(100, 200), intercepts=nengo.dists.Uniform(-0.8, 0.8), seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed)
		nengo.Connection(x, supv, synapse=0.1, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, post.neurons, synapse=0.1, transform=w_E.T, seed=seed)
		nengo.Connection(I.neurons, post.neurons, synapse=0.001, transform=w_I.T, seed=seed)

		p_E = nengo.Probe(E, synapse=0.05)
		p_I = nengo.Probe(I, synapse=0.05)
		p_post = nengo.Probe(post, synapse=0.05, solver=nengo.solvers.NoSolver(d_post))
		p_supv = nengo.Probe(supv, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.05)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(10)

	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange(), sim.data[p_post], alpha=0.5, label='post')
	ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	ax.plot(sim.trange(), sim.data[p_x], alpha=0.5, label='target')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange(), sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange(), sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='test')
	ax2.set(ylabel='Firing Rate', ylim=((0, 400)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 400)), title='post')
	plt.show()

def multiply():
	'''train'''
	N_E = 200
	N_I = 400
	N_ens = 10
	seed = 1
	a = nengo.dists.Uniform(1, 10).sample(n=N_ens, rng=np.random.RandomState(seed=seed))
	b = np.zeros((N_ens))
	w_E = np.zeros((N_E, N_ens))
	w_I = np.zeros((N_I, N_ens))

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(lambda t: [np.sin(t), np.cos(3*t)])
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
		node = LearningNode(N_E, N_I, N_ens, w_E, w_I)

		pre = nengo.Ensemble(N_E, 2, seed=seed)
		E = nengo.Ensemble(N_E, 2, max_rates=nengo.dists.Uniform(100, 200), seed=seed)
		I = nengo.Ensemble(N_I, 2, max_rates=nengo.dists.Uniform(200, 400), seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(100, 200), seed=seed)
		error = nengo.Ensemble(N_E, 1, seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed, function=lambda x: x[0]*x[1])
		nengo.Connection(x, supv, synapse=0.1, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, node[0:N_E], synapse=0.1)
		nengo.Connection(I.neurons, node[N_E: N_E+N_I], synapse=0.001)
		nengo.Connection(post.neurons, node[N_E+N_I: N_E+N_I+N_ens], synapse=0.05)
		nengo.Connection(supv.neurons, node[N_E+N_I+N_ens:N_E+N_I+N_ens+N_ens], synapse=0.05)
		nengo.Connection(node, post.neurons, synapse=None)
		conn = nengo.Connection(post, error, synapse=0.05, function=lambda x: 0, learning_rule_type=nengo.PES(learning_rate=1e-5))
		nengo.Connection(supv, error, synapse=0.05, transform=-1)
		nengo.Connection(error, conn.learning_rule)

		p_E = nengo.Probe(E, synapse=0.05)
		p_I = nengo.Probe(I, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.05)
		p_supv = nengo.Probe(supv, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.05)
		p_error = nengo.Probe(error, synapse=0.05)
		p_weights = nengo.Probe(conn, 'weights', synapse=None)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(200)


	xhat_post = np.zeros((sim.trange().shape[0], 1))
	for t in range(xhat_post.shape[0]):
		xhat_post[t] = np.dot(sim.data[p_post_neurons][t], sim.data[p_weights][t].T)
	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange(), xhat_post, alpha=0.5, label='post')
	ax.plot(sim.trange(), sim.data[p_x], alpha=0.5, label='target')
	# ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	# ax.plot(sim.trange(), sim.data[p_error], alpha=0.5, label='error')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange(), sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange(), sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='train')
	ax2.set(ylabel='Firing Rate', ylim=((0, 400)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 400)), title='post')

	'''test'''
	w_E = node.w_E
	w_I = node.w_I
	d_post = sim.data[p_weights][-1].T

	with nengo.Network(seed=seed) as network:
		u = nengo.Node(lambda t: np.sin(t))
		x = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())

		pre = nengo.Ensemble(N_E, 1, seed=seed)
		E = nengo.Ensemble(N_E, 1, max_rates=nengo.dists.Uniform(100, 200), seed=seed)
		I = nengo.Ensemble(N_I, 1, max_rates=nengo.dists.Uniform(200, 400), seed=seed)
		post = nengo.Ensemble(N_ens, 1, gain=a, bias=b, seed=seed)
		supv = nengo.Ensemble(N_ens, 1, max_rates=nengo.dists.Uniform(100, 200), seed=seed)

		nengo.Connection(u, x, synapse=None, seed=seed)
		nengo.Connection(x, supv, synapse=0.1, seed=seed)
		nengo.Connection(u, pre, synapse=None, seed=seed)
		nengo.Connection(pre, E, synapse=0.01, seed=seed)
		nengo.Connection(pre, I, synapse=0.01, seed=seed)
		nengo.Connection(E.neurons, post.neurons, synapse=0.1, transform=w_E.T, seed=seed)
		nengo.Connection(I.neurons, post.neurons, synapse=0.001, transform=w_I.T, seed=seed)

		p_E = nengo.Probe(E, synapse=0.05)
		p_I = nengo.Probe(I, synapse=0.05)
		p_post = nengo.Probe(post, synapse=0.05, solver=nengo.solvers.NoSolver(d_post))
		p_supv = nengo.Probe(supv, synapse=0.05)
		p_x = nengo.Probe(x, synapse=0.05)
		p_post_neurons = nengo.Probe(post.neurons, synapse=0.05)
		p_supv_neurons = nengo.Probe(supv.neurons, synapse=0.05)

	with nengo.Simulator(network, seed=seed) as sim:
		sim.run(10)

	fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=((12, 12)), sharex=True)
	# ax.plot(sim.trange(), sim.data[p_E], alpha=0.5, label='E')
	# ax.plot(sim.trange(), sim.data[p_I], alpha=0.5, label='I')
	ax.plot(sim.trange(), sim.data[p_post], alpha=0.5, label='post')
	ax.plot(sim.trange(), sim.data[p_supv], alpha=0.5, label='supv')
	ax.plot(sim.trange(), sim.data[p_x], alpha=0.5, label='target')
	ax.legend(loc='upper right')
	for n in range(N_ens):
		ax3.plot(sim.trange(), sim.data[p_post_neurons][:,n], alpha=0.5)
		ax2.plot(sim.trange(), sim.data[p_supv_neurons][:,n], alpha=0.5)
	ax.set(ylim=((-1, 1)), xlabel=r"$\mathbf{x}$", title='test')
	ax2.set(ylabel='Firing Rate', ylim=((0, 400)), title='supervisor')
	ax3.set(ylabel='Firing Rate', ylim=((0, 400)), title='post')
	plt.show()

feedforward()