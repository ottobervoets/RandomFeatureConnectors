import numpy as np
from scipy.sparse import random as sprandn
from scipy.sparse.linalg import eigs
from matlab_copy.helper_functions import *
import matplotlib.pyplot as plt


class MatriXConceptorWorking:
    def __init__(self,
                 N: int = 500,
                 W_sr: float = 0.6,
                 W_in_std: float = 1.2,
                 b_std: float = 0.4,
                 **kwargs):
        randstate = 2
        # np.random.seed(randstate)
        self.N = N
        Netconnectivity = 1 if self.N <= 20 else 10 / self.N
        WinRaw = np.random.randn(self.N, 2)
        WstarRaw = generate_internal_weights(self.N, Netconnectivity)  # Placeholder for internal weights function
        WbiasRaw = np.random.randn(self.N)

        self.Win = W_in_std * WinRaw
        self.Wstar = W_sr * WstarRaw
        self.Wbias = b_std * WbiasRaw
        self.startXs = {}
        self.Cs = {}

        self.Wout = None
        self.W = None


    def store_patterns(self, training_patterns: dict = None, signal_noise: float = None, washout: int = None,
                       n_adapt: int = None,
                       beta_W_out: float = 0.01, beta_W: float = 0.0001, noise_std: float = None, **kwargs):

        # Weight learning parameters
        TychonovAlphaEqui = beta_W
        washoutLength = washout
        learnLength = n_adapt
        TychonovAlphaReadout = beta_W_out

        # patts = []
        # patts.append(training_patterns['1'])
        # patts.append(training_patterns['2'])
        # patts.append(training_patterns['3'])
        # patts.append(training_patterns['4'])
        # Set pattern handles
        # L = washoutLength + learnLength
        # LorenzSeq = generate_lorenz_sequence_2d(200, 15, L, 5000)
        # patts.append(lambda n: 2 * LorenzSeq[:, n] - 1)
        # RoesslerSeq = generate_roessler_sequence_2d(200, 150, L, 5000)
        # patts.append(lambda n: RoesslerSeq[:, n])
        # MGSeq = generate_mg_sequence_2d(17, 10, 3, L, 5000)
        # patts.append(lambda n: 2 * MGSeq[:, n] - 1)
        # HenonSeq = generate_henon_sequence_2d(L, 1000)
        # patts.append(lambda n: HenonSeq[:, n])

        # LorenzSeq = np.loadtxt('1.csv', delimiter=",")
        # patts.append(LorenzSeq.T)
        # patts.append(np.loadtxt('2.csv', delimiter=",").T)
        # patts.append(np.loadtxt('3.csv', delimiter=",").T)
        # patts.append(np.loadtxt('4.csv', delimiter=",").T)


        num_p = len(training_patterns.keys())

        # Data collection for training
        allTrainArgs = np.zeros((self.N, num_p * learnLength))
        allTrainOldArgs = np.zeros((self.N, num_p * learnLength))
        allTrainOuts = np.zeros((2, num_p * learnLength))

        patternCollectors = []
        xCollectorsCentered = []
        xCollectors = []
        patternRs = []

        # Collect data from driving native reservoir with different drivers
        for p, name in zip(range(num_p), training_patterns.keys()):
            patt = training_patterns[name]
            xCollector = np.zeros((self.N, learnLength))
            xOldCollector = np.zeros((self.N, learnLength))
            pCollector = np.zeros((2, learnLength))
            x = np.zeros((self.N))
            plot_list = []

            for n in range(washoutLength + learnLength):
                # print(patt.shape)
                u = patt[n]
                u_in = u
                # if name == "lorenz_attractor" or name == "mackey_glass":
                #     print("adapted", name)
                #     u_in = (u*2) - 1
                xOld = x
                x = np.tanh(self.Wstar @ x + self.Win @ u_in + self.Wbias)
                # print(x.shape)
                # break
                if n >= washoutLength:
                    idx = n - washoutLength
                    xCollector[:, idx] = x[:]
                    xOldCollector[:, idx] = xOld[:]
                    # if p in [0, 2]:
                    #     pCollector[:, idx] = 0.5 * (u[:,] + 1)
                    # else:
                    pCollector[:, idx] = u[:,]

            xCollectorCentered = xCollector - np.mean(xCollector, axis=1, keepdims=True)
            xCollectorsCentered.append(xCollectorCentered)
            xCollectors.append(xCollector)
            Ux, Sx, Vx = np.linalg.svd(xCollector @ xCollector.T / learnLength)
            self.startXs[name] = x[:]
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

        self.Wout = (np.linalg.inv(allTrainArgs @ allTrainArgs.T +
                             TychonovAlphaReadout * np.eye(self.N)) @ allTrainArgs @ allTrainOuts.T).T
        NRMSE_readout = nrmse(self.Wout @ allTrainArgs, allTrainOuts)
        print(f'NRMSE readout: {np.mean(NRMSE_readout)}')
        print(f'train outs shape:{np.shape(allTrainOuts)}, Wout shape: {np.shape(self.Wout)}')
        # Compute W
        print(num_p, learnLength)
        Wtargets = np.arctanh(allTrainArgs) - self.Wbias.reshape(self.N,1)
        self.W = (np.linalg.inv(allTrainOldArgs @ allTrainOldArgs.T +
                          TychonovAlphaEqui * np.eye(self.N)) @ allTrainOldArgs @ Wtargets.T).T
        NRMSE_W = nrmse(self.W @ allTrainOldArgs, Wtargets)
        print(f'mean NRMSE W: {np.mean(NRMSE_W)}')

        # Compute conceptors
        Cs = []
        aperture = [10**3, 10**2.6, 10**3.1, 10**2.8]
        # aperture = [1000, 460, 400, 700]
        for p, name in zip(range(num_p), training_patterns.keys()):
            aperture = kwargs[f"aperture_{name}"]
            print(f"aperture {name}, {aperture}")
            R = patternRs[p]
            C = R @ np.linalg.inv(R + (aperture ** -2) * np.identity(self.N))
            self.Cs[name] = C

    def record_chaotic(self, length, pattern_name):
        x = np.array(self.startXs[pattern_name])
        record = []
        for t in range(length):
            # print(self.Wbias.shape)
            x = self.Cs[pattern_name] @ np.tanh(self.W @ x + self.Wbias)
            record.append(self.Wout @ x)
        test = []
        test.extend(record)
        record = np.array(record)
        for t in range(500):
            x = self.Cs[pattern_name] @ np.tanh(self.W @ x + self.Wbias)
            test.append(self.Wout @ x)
        # plt.plot(record[:,0], record[:,1], linewidth=1)
        # plt.show()
        return np.array(record)

# patts = {}
# LorenzSeq = np.loadtxt('1.csv', delimiter=",")
# patts["1"] = LorenzSeq.T
# patts["2"] = np.loadtxt('2.csv', delimiter=",").T
# patts["3"] = np.loadtxt('3.csv', delimiter=",").T
# patts["4"] = np.loadtxt('4.csv', delimiter=",").T
#
# test_unit = MatriXConceptorWorking()
#
#
# test_unit.store_patterns(training_patterns=patts, n_adapt=2000, washout=500)
# for idx in range(1,5):
#     record = test_unit.record_chaotic(500, f'{idx}')
#     plt.plot(record[:,0], record[:,1], linewidth=1)
#     plt.show()

# test_unit.record_chaotic(500, '2')
# test_unit.record_chaotic(500, '3')
# test_unit.record_chaotic(500, '4')

