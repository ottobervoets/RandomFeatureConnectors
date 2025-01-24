import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("../data/lorenz.csv", header=None).to_numpy()

data = data.to_numpy()
plt.plot(data[0], data[1])
plt.show()