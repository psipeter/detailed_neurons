import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='whitegrid')

def barplots(networks=["identityNew", "multiplyNew"], neurons=["LIF", "ALIF", "Wilson", "Bio"]):
    columns = ("network", "neuron", "test", "error")
    dfs = []
    for network in networks:
        for neuron in neurons:
            print(network, neuron)
            errors = np.load("data/%s_%s.npz"%(network, neuron+"()"))['errors']
            for e in range(len(errors)):
                dfs.append(pd.DataFrame([[network, neuron, e, errors[e]]], columns=columns))
    df = pd.concat([df for df in dfs], ignore_index=True)
    
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="network", y="error", hue="neuron", ax=ax)
    fig.savefig("figures/barplots.pdf")
    plt.close("all")
    
barplots(networks=["identityNew", "multiplyNew", "oscillateNew", "integrateNew"])
