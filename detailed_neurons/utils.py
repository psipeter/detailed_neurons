import numpy as np
import nengo

__all__ = ['bin_activities_values_single', 'isi']

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def bin_activities_values_single(xhat_pre, act_bio, bins=20):
    x_bins = np.linspace(np.min(xhat_pre), np.max(xhat_pre), num=bins)
    hz_means = np.zeros((bins))
    hz_stds = np.zeros((bins))
    bin_act = [[] for _ in range(x_bins.shape[0])]
    for t in range(act_bio.shape[0]):
        idx = find_nearest(x_bins, xhat_pre[t])
        bin_act[idx].append(act_bio[t])
    for x in range(len(bin_act)):
        hz_means[x] = np.average(bin_act[x]) if len(bin_act[x]) > 0 else 0
        hz_stds[x] = np.std(bin_act[x]) if len(bin_act[x]) > 1 else 0
    return x_bins, hz_means, hz_stds
            
def isi(all_spikes, dt=0.000025):
    nz = []
    for n in range(all_spikes.shape[1]):
        sts = np.nonzero(all_spikes[:,n])
        nz.append((np.diff(sts)*dt).ravel())
    return nz

def nrmse_vs_n_neurons():
    neurons = [3, 10, 30, 100]
    n_trials = 3
    nrmses = np.zeros((len(neurons), 5, n_trials))
    for nn, n_neurons in enumerate(neurons):
        nrmses[nn] = trials(n_neurons=n_neurons, n_trials=n_trials, gb_iter=3, h_iter=200)

    nts =  ['LIF (static)', 'LIF (temporal)', 'ALIF', 'Wilson', 'Durstewitz']
    columns=['nrmse', 'n_neurons', 'neuron_type', 'trial']
    df = pd.DataFrame(columns=columns)
    for nn in range(len(neurons)):
        for nt in range(len(nts)):
            for trial in range(n_trials):
                df_new = pd.DataFrame([[nrmses[nn, nt, trial], neurons[nn], nts[nt], trial]], columns=columns)
                df = df.append(df_new, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.lineplot(x='n_neurons', y='nrmse', hue='neuron_type', data=df)
    plt.show()