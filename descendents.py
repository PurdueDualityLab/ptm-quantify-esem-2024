import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Set text size for plots
plt.rcParams.update({'font.size': 20})

# Load the CSV data into a DataFrame
df = pd.read_csv('number_of_direct_descendant_models_per_download.csv')

# Sort by count if needed
df = df.sort_values(by='count', ascending=False)


# # Plot the scatter plot
# plt.figure(figsize=(10, 10))
# plt.scatter(df['downloads'], df['count'])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Downloads')
# plt.ylabel('Number of Direct Descendant Models')
# plt.title('Direct Descendant Models per Download', pad=20)


# Extract the clean columns
downloads = df['downloads']
count = df['count']

# Create logarithmic bins
downloads_bins = np.logspace(np.log10(1), 8, 30)
count_bins = np.logspace(np.log10(1), 3, 30)

# Plot the 2D histogram
plt.figure(figsize=(10, 10))
plt.hist2d(downloads, count, bins=[downloads_bins, count_bins], cmap='viridis', norm=LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Downloads')
plt.ylabel('Number of Direct Descendant Models')
plt.colorbar(label='Number of models')
# elevate title a lot more above the plot
plt.title('Direct Descendant Models per Download', pad=20)


# Ensure data is sorted or unique
df = df.drop_duplicates(subset=['downloads', 'count'])

# Add a small constant to avoid zero values
epsilon = 1e-5

# Log-transform the columns (add a small constant to avoid zero issues)
log_downloads = np.log10(df['downloads'] + epsilon)
log_count = np.log10(df['count'] + epsilon)

# Fit a single line of best fit using the log-transformed data
m, b = np.polyfit(log_downloads, log_count, 1)

# Plot the fitted line (converting back to linear scale)
downloads = np.logspace(0, 8, 100)
best_fit_line = 10 ** (m * np.log10(downloads) + b)
# Make this line thiccc
# Put the label and legend in the top left corner, and then also shift it down a bit
plt.plot(downloads, best_fit_line, color='red', linewidth=3, label=f'Best fit line: y = {m:.2f}x + {b:.2f}')
plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.9), borderaxespad=0.)
plt.show()
