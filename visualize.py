import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm


class ExperimentResults:
    def __init__(self, dataset_id=None, exp=None):
        """
        Initializes the ExperimentResults object by loading result files and parsing data.

        Args:
            dataset_id (int or str): Identifier for the dataset.
            exp (str): Experiment identifier.
        """
        self.exp = exp

        # Load result files from the specified directory
        self.files = [f'results/{self.exp}/{dataset_id}/{x}' for x in os.listdir(f'results/{self.exp}/{dataset_id}/')]
        self.results = [pickle.load(open(f, 'rb')) for f in self.files]

        # Number of iterations and runs
        self.iterations = len(self.results[0]['rs_public'])
        self.n_runs = len(self.results)

        # Extract public and private results for Random Search (RS) and Bayesian Optimization (BO)
        self.public_rs = np.array([x['rs_public'] for x in self.results])
        self.private_rs = np.array([x['rs_private'] for x in self.results])

        self.public_bo = np.array([x['bo_public'] for x in self.results])
        self.private_bo = np.array([x['bo_private'] for x in self.results])

    @staticmethod
    def accumulate_best_scores(val_scores, test_scores):
        """
        Accumulates the best validation and corresponding test scores at each iteration.

        Args:
            val_scores (list): Validation scores.
            test_scores (list): Test scores.

        Returns:
            tuple: Lists of accumulated best validation and test scores.
        """
        scores = [(x, y) for x, y in zip(val_scores, test_scores)]
        val_max = [scores[0]]

        # Accumulate the best scores over iterations
        for score in scores[1:]:
            if score[0] > val_max[-1][0]:
                val_max.append(score)
            else:
                val_max.append(val_max[-1])

        test_max = [s[1] for s in val_max]
        val_max = [s[0] for s in val_max]

        return val_max, test_max

    @staticmethod
    def get_selected_private(val_scores, test_scores, max_iter=250):
        """
        Selects the test score corresponding to the best validation score within max_iter.

        Args:
            val_scores (list): Validation scores.
            test_scores (list): Test scores.
            max_iter (int): Maximum iteration to consider.

        Returns:
            tuple: Best validation and corresponding test score.
        """
        if max_iter == 0:
            return val_scores[0], test_scores[0]

        scores = [(x, y) for x, y in zip(val_scores, test_scores)][:max_iter]
        best = max(scores, key=lambda l: l[0])
        return best

    def plot_selected_moe(self):
        """
        Plots the Mean Over Expected (MOE) for Random Search (RS) and Bayesian Optimization (BO).
        """
        rs_moe_iterations = []
        bo_moe_iterations = []

        # Calculate MOE over iterations
        for i in tqdm(range(0, 250)):
            rs_scores = [self.get_selected_private(a, b, max_iter=i) for a, b in zip(self.public_rs, self.private_rs)]
            bo_scores = [self.get_selected_private(a, b, max_iter=i) for a, b in zip(self.public_bo, self.private_bo)]

            # Calculate the mean MOE
            rs_moe_iterations.append(np.mean([x[0] - x[1] for x in rs_scores]))
            bo_moe_iterations.append(np.mean([x[0] - x[1] for x in bo_scores]))

        # Convert to numpy arrays for plotting
        rs_moe_iterations = np.array(rs_moe_iterations)
        bo_moe_iterations = np.array(bo_moe_iterations)

        # Plot MOE for RS and BO
        plt.plot(range(len(rs_moe_iterations)), rs_moe_iterations, label='RS')
        plt.plot(range(len(bo_moe_iterations)), bo_moe_iterations, color='red', label='BO')

        plt.legend()
        plt.ylabel('MOE')
        plt.xlabel('Iterations')
        plt.show()

    def plot_accuracy(self):
        """
        Plots the accuracy trends for Random Search (RS) and Bayesian Optimization (BO) over iterations.
        """
        public_rs_max = []
        private_rs_max = []
        public_bo_max = []
        private_bo_max = []

        # Accumulate best scores for RS
        for x, y in zip(self.public_rs, self.private_rs):
            a, b = self.accumulate_best_scores(x, y)
            public_rs_max.append(a)
            private_rs_max.append(b)

        # Accumulate best scores for BO
        for x, y in zip(self.public_bo, self.private_bo):
            a, b = self.accumulate_best_scores(x, y)
            public_bo_max.append(a)
            private_bo_max.append(b)

        # Convert to numpy arrays for plotting
        public_rs_max = np.array(public_rs_max)
        private_rs_max = np.array(private_rs_max)
        public_bo_max = np.array(public_bo_max)
        private_bo_max = np.array(private_bo_max)

        # Plot accuracy for RS
        plt.plot(range(self.iterations), np.mean(public_rs_max, axis=0), label='RS val', linestyle='--',
                 color='tab:blue')
        plt.plot(range(self.iterations), np.mean(private_rs_max, axis=0), color='tab:blue', label='RS selected test')
        plt.plot(range(self.iterations), np.mean(np.maximum.accumulate(self.private_rs, axis=1), axis=0),
                 label='RS best test', linestyle=':', color='tab:blue')

        # Plot accuracy for BO
        plt.plot(range(self.iterations), np.mean(public_bo_max, axis=0), color='red', label='BO val', linestyle='--')
        plt.plot(range(self.iterations), np.mean(private_bo_max, axis=0), color='red', label='BO selected test')
        plt.plot(range(self.iterations), np.mean(np.maximum.accumulate(self.private_bo, axis=1), axis=0), color='red',
                 label='BO best test', linestyle=':')

        plt.ylabel('Accuracy')
        plt.xlabel('Iterations')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage with a specific dataset ID
    ds = 1590
    results = ExperimentResults(1590, "openml-classification")
    results.plot_accuracy()
    results.plot_selected_moe()
