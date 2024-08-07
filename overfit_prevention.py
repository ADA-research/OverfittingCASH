import numpy as np


class Thresholdout:
    def __init__(self, labels, noise_rate):

        self.labels = labels
        self.noise_rate = noise_rate / np.sqrt(len(labels))
        self.threshold = 2 * noise_rate / np.sqrt(len(labels))
        self.noisy_t = self.threshold + self.sample_laplace_noise(self.noise_rate * 2)

    @staticmethod
    def sample_laplace_noise(scale):
        """Sample a Laplace noise with specified scale."""
        return np.random.laplace(loc=0, scale=scale)

    def score(self, train_acc, val_acc):
        """
        Process a single query function and update the threshold accordingly.

        Returns:
        float: The result of the query
        """
        eta = self.sample_laplace_noise(4 * self.noise_rate)

        if abs(val_acc - train_acc) > self.noisy_t + eta:
            xi = self.sample_laplace_noise(self.noise_rate)
            gamma = self.sample_laplace_noise(2 * self.noise_rate)
            self.noisy_t = self.threshold + gamma

            return 1 - (val_acc + xi)
        else:
            return 1 - train_acc
