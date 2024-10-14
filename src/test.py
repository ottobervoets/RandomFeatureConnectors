import numpy as np
import matplotlib.pyplot as plt
from signal_generators import *


res = rossler_attractor(1000)

plt.plot(res[:,0], res[:,1])
plt.show()