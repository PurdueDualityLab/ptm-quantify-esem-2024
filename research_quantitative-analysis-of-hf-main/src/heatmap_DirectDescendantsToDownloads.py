import pickle

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

# Read data from pickle file
with open("pickle/data.pickle", "rb") as pf:
    metric: DataFrame = pickle.load(file=pf)
    pf.close()

df: DataFrame = metric[["downloads", "count"]]


# Assuming your data is stored in a 2D numpy array called 'data'
# Example data
# Clamp values between 0 and 255
clamped_data = np.clip(df.to_numpy(), 0, 255)

print(df.to_numpy())

# Plot the heatmap
plt.imshow(clamped_data, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.savefig("test.pdf")
