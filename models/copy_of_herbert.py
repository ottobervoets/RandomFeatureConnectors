import numpy as np
from scipy.sparse import random as sprandn
from scipy.sparse.linalg import eigs
import pandas as pd
import matplotlib.pyplot as plt

# Functions for generating sequences should be replaced with CSV readers
def generate_internal_weights(n_internal_units, connectivity):
    success = False
    while not success:
        try:
            internal_weights = sprandn(n_internal_units, n_internal_units, density=connectivity).toarray()
            spec_rad = abs(eigs(internal_weights, k=1, which='LM', return_eigenvectors=False)[0])
            internal_weights = internal_weights / spec_rad
            success = True
        except Exception as e:
            pass  # Retry on failure
    return internal_weights

def nrmse(output, target):
    combined_var = 0.5 * (np.var(target, axis=1, ddof=0) + np.var(output, axis=1, ddof=0))
    error_signal = output - target
    return np.sqrt(np.mean(error_signal**2, axis=1) / combined_var)

# Experiment control
rand_state = 1
new_nets = True
new_system_scalings = True
new_chaos_data = True

# Setting system params
netsize = 500  # Network size
net_sr = 0.6  # Spectral radius
net_inp_scaling = 1.2  # Scaling of input weights
bias_scaling = 0.4  # Size of bias

# Weight learning
tych_alpha_equi = 0.0001  # Regularizer for equi weight training
washout_length = 500
learn_length = 2000  # For learning weights
tych_alpha_readout = 0.01

np.random.seed(rand_state)

# Create raw weights
if new_nets:
    net_connectivity = 1 if netsize <= 20 else 10 / netsize
    win_raw = np.random.randn(netsize, 2)
    wstar_raw = generate_internal_weights(netsize, net_connectivity)
    wbias_raw = np.random.randn(netsize, 1)

# Scale raw weights and initialize weights
if new_system_scalings:
    wstar = net_sr * wstar_raw
    win = net_inp_scaling * win_raw
    wbias = bias_scaling * wbias_raw

# Placeholder for loading sequences from CSV
def load_sequence_from_csv(filename):
    return np.genfromtxt(filename, delimiter=',')

if new_chaos_data:
    patts = [None] * 4
    L = washout_length + learn_length
    # Replace the following lines with actual CSV data loading
    patts[0] = pd.read_csv("../data/RoesslerSeq.csv", header=None).to_numpy().T
    patts[1] = pd.read_csv("../data/lorenz.csv", header=None).to_numpy().T
    patts[2] = pd.read_csv("../data/MGSeq.csv", header=None).to_numpy().T
    patts[3] = pd.read_csv("../data/HenonSeq.csv", header=None).to_numpy().T
    no_pads = 4

# Collect data from the network
all_train_args = np.zeros((netsize, no_pads * learn_length))
all_train_old_args = np.zeros((netsize, no_pads * learn_length))
all_train_outs = np.zeros((2, no_pads * learn_length))
patternRs = {}
startXs = {}

for p in range(no_pads):
    patt = patts[p]
    x_collector = np.zeros((netsize, learn_length))
    x_old_collector = np.zeros((netsize, learn_length))
    p_collector = np.zeros((2, learn_length))
    x = np.zeros((netsize, 1))

    for n in range(washout_length + learn_length):
        u = patt[n]
        # print(f"shape u {u.shape}")
        x_old = x
        x = np.tanh(wstar @ x + win @ u + wbias)
        if n >= washout_length:
            x_collector[:, n - washout_length] = x[:, 0]
            x_old_collector[:, n - washout_length] = x_old[:, 0]
            p_collector[0, n - washout_length] = u[0]

    x_collector_centered = x_collector - np.mean(x_collector, axis=1, keepdims=True)
    all_train_args[:, p * learn_length:(p + 1) * learn_length] = x_collector
    all_train_old_args[:, p * learn_length:(p + 1) * learn_length] = x_old_collector
    all_train_outs[:, p * learn_length:(p + 1) * learn_length] = p_collector
    startXs[p] = x
    u, s, vh = np.linalg.svd(x_collector @ x_collector.T / learn_length)
    diag_s = np.diag(s)
    r = u @ diag_s @ u.T
    patternRs[p] = r


print(f"all training args: {np.shape(all_train_args)}, outs: {np.shape(all_train_outs)}")
# Compute readout weights
wout = (np.linalg.inv(all_train_args @ all_train_args.T + tych_alpha_readout * np.eye(netsize)) @ \
       all_train_args @ all_train_outs.T).T
nrmse_readout = nrmse(wout @ all_train_args, all_train_outs)
print(f"NRMSE readout: {nrmse_readout}")

# Compute W
w_targets = np.arctanh(all_train_args) - wbias
w = (np.linalg.inv(all_train_old_args @ all_train_old_args.T + tych_alpha_equi * np.eye(netsize)) @ \
    all_train_old_args @ w_targets.T).T
nrmse_w = nrmse(w @ all_train_old_args, w_targets)
print(f"Mean NRMSE W: {np.mean(nrmse_w)}")

conceptors = {}
aperture = 500
for id, value in patternRs.items():
    conceptors[id] = r @ np.linalg.inv(r + (aperture ** -2) * np.identity(netsize))

prediction_length = 84
for id, conceptor in conceptors.items():
    recording = []

    x = startXs[id]
    for i in range(prediction_length):
        x = conceptor @ np.tanh(w @ x + wbias)
        recording.append(wout @ x)
    recording = np.array(recording)

    plt.plot(recording[:,0], recording[:,1])
    plt.show()