import numpy as np
import matplotlib.pyplot as plt
from signal_generators import *

ls = [[1,2,3],[4,6]]

for i in range(len(ls)):
    ls[i] += np.ones(len(ls[i]))

print(ls)