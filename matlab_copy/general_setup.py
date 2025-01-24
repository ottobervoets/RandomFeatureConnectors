import numpy as np
from scipy.sparse import random as sprandn
from scipy.sparse.linalg import eigs
from matlab_copy.helper_functions import *
import matplotlib.pyplot as plt

# Initialize random state
# randstate = 2
# np.random.seed(randstate)

# Experiment control
newNets = True
newSystemScalings = True
newChaosData = True

# System parameters
Netsize = 500
NetSR = 0.6
Netinum_pScaling = 1.2
BiasScaling = 0.4

# Weight learning parameters
TychonovAlphaEqui = 0.0001
washoutLength = 500
learnLength = 2000
TychonovAlphaReadout = 0.01

# Generate raw weights
Netconnectivity = 1 if Netsize <= 20 else 10 / Netsize
WinRaw = np.random.randn(Netsize, 2)
WstarRaw = generate_internal_weights(Netsize, Netconnectivity)  # Placeholder for internal weights function
WbiasRaw = np.random.randn(Netsize)

# Scale raw weights and initialize weights
Wstar = NetSR * WstarRaw
Win = Netinum_pScaling * WinRaw
Wbias = BiasScaling * WbiasRaw

# Win = np.loadtxt("win.csv", delimiter=",", dtype=float)
# Wbias = np.loadtxt("wbias.csv", delimiter=",", dtype=float)
# Wstar = np.loadtxt("wstar.csv", delimiter=",", dtype=float).T



patts = []
# Set pattern handles
if newChaosData:
    L = washoutLength + learnLength
    LorenzSeq = lorenz_attractor_2d(200, 15, L, 5000)
    patts.append(lambda n: 2 * LorenzSeq[:, n] - 1)
    RoesslerSeq = rossler_attractor_2d(200, 150, L, 5000)
    patts.append(lambda n: RoesslerSeq[:, n])
    MGSeq = mackey_glass_2d(17, 10, 3, L, 5000)
    patts.append(lambda n: 2 * MGSeq[:, n] - 1)
    HenonSeq = henon_attractor_2d(L, 1000)
    patts.append(lambda n: HenonSeq[:, n])

# LorenzSeq = np.loadtxt('1.csv', delimiter=",")
# patts.append(LorenzSeq.T)
# patts.append(np.loadtxt('2.csv', delimiter=",").T)
# patts.append(np.loadtxt('3.csv', delimiter=",").T)
# patts.append(np.loadtxt('4.csv', delimiter=",").T)


num_p = len(patts)

# Data collection for training
allTrainArgs = np.zeros((Netsize, num_p * learnLength))
allTrainOldArgs = np.zeros((Netsize, num_p * learnLength))
allTrainOuts = np.zeros((2, num_p * learnLength))

patternCollectors = []
xCollectorsCentered = []
xCollectors = []
patternRs = []
startXs = np.zeros((Netsize, num_p))

# Collect data from driving native reservoir with different drivers
for p in range(num_p):
    patt = patts[p]
    xCollector = np.zeros((Netsize, learnLength))
    xOldCollector = np.zeros((Netsize, learnLength))
    pCollector = np.zeros((2, learnLength))
    x = np.zeros((Netsize))
    plot_list = []

    for n in range(washoutLength + learnLength):
        # print(patt.shape)
        u = patt(n)
        xOld = x
        x = np.tanh(Wstar @ x + Win @ u + Wbias)
        # print(x.shape)
        # break
        if n >= washoutLength:
            idx = n - washoutLength
            xCollector[:, idx] = x[:]
            xOldCollector[:, idx] = xOld[:]
            if p in [0, 2]:
                pCollector[:, idx] = 0.5 * (u[:,] + 1)
            else:
                pCollector[:, idx] = u[:,]

    xCollectorCentered = xCollector - np.mean(xCollector, axis=1, keepdims=True)
    xCollectorsCentered.append(xCollectorCentered)
    xCollectors.append(xCollector)
    Ux, Sx, Vx = np.linalg.svd(xCollector @ xCollector.T / learnLength)
    startXs[:, p] = x[:]
    R = Ux @ np.diag(Sx) @ Ux.T
    patternRs.append(R)
    plot_list = np.array(plot_list)
    # plt.plot(pCollector.T[:,0], pCollector.T[:,1], linewidth=1)
    # plt.show()
    patternCollectors.append(pCollector)
    allTrainArgs[:, p * learnLength:(p + 1) * learnLength] = xCollector
    allTrainOldArgs[:, p * learnLength:(p + 1) * learnLength] = xOldCollector
    allTrainOuts[:, p * learnLength:(p + 1) * learnLength] = pCollector

# Compute readout weights

Wout = (np.linalg.inv(allTrainArgs @ allTrainArgs.T +
                     TychonovAlphaReadout * np.eye(Netsize)) @ allTrainArgs @ allTrainOuts.T).T
NRMSE_readout = nrmse(Wout @ allTrainArgs, allTrainOuts)
print(f'NRMSE readout: {np.mean(NRMSE_readout)}')
print(f'train outs shape:{np.shape(allTrainOuts)}, Wout shape: {np.shape(Wout)}')
# Compute W
print(num_p, learnLength)
Wtargets = np.arctanh(allTrainArgs) - Wbias.reshape(Netsize,1)
W = (np.linalg.inv(allTrainOldArgs @ allTrainOldArgs.T +
                  TychonovAlphaEqui * np.eye(Netsize)) @ allTrainOldArgs @ Wtargets.T).T
NRMSE_W = nrmse(W @ allTrainOldArgs, Wtargets)
print(f'mean NRMSE W: {np.mean(NRMSE_W)}')

# Compute conceptors
Cs = []
aperture = [10**3, 10**2.6, 10**3.1, 10**2.8]
# aperture = [1000, 460, 400, 700]
for p in range(num_p):
    R = patternRs[p]
    C = R @ np.linalg.inv(R + (aperture[p] ** -2) * np.identity(Netsize))
    Cs.append(C)


for p in range(4):
    x = np.array(startXs[:,p])
    record = []
    Wbias = np.array(Wbias).reshape(Netsize)
    for t in range(500):
        x = Cs[p] @ np.tanh(W @ x + Wbias)
        record.append(Wout @ x)
    record = np.array(record)
    plt.plot(record[:,0], record[:,1], linewidth=1)
    plt.show()

