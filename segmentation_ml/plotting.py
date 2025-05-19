import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_gaussian_models(means, std_devs, weights=None, x_range=(0, 255), num_points=1000, save_path=None):
    x = np.linspace(x_range[0], x_range[1], num_points)
    total_pdf = np.zeros_like(x)

    # If weights are not provided, assume equal weights
    if weights is None:
        weights = np.ones(len(means)) / len(means)

    # Plot each Gaussian component
    for mean, std_dev, weight in zip(means, std_devs, weights):
        pdf = weight * norm.pdf(x, mean, std_dev)
        plt.plot(x, pdf, label=f'Gaussian(mean={mean:.2f}, std={std_dev:.2f})')
        total_pdf += pdf

    # Plot the sum of the Gaussian components
    plt.plot(x, total_pdf, label='Sum of Gaussians', color='black', linewidth=1)

    plt.legend()
    plt.xlabel('Intensity')
    plt.ylabel('Probability Density')
    plt.title('Gaussian Models')
    if save_path:
        plt.savefig(save_path)
    else:
      plt.show()

    plt.close()