import os

from random_search import RandomSearch
from bo import BayesianOptimization
import random
import numpy as np
import pickle
import shutil
import openml
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('results'):
    os.mkdir('results')


class Config:
    def __init__(self):
        self.out_dir = 'results/openml_validation_sizes'
        self.iterations = 250
        self.debugging = False

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if os.path.exists('smac3_output'):
            shutil.rmtree('smac3_output')


config = Config()


def seed_everything(seed: int) -> None:
    """
    Function to seed random and numpy
    :param seed: Seed to be used
    """
    np.random.seed(seed)
    random.seed(seed)


def draw_seed() -> int:
    """
    Function to draw valid random seed
    :return: Seed
    """
    return random.randint(1000000, 2 ** 32 - 1)


def reset_smac() -> None:
    """
    Function to remove smac3 history to ensure separate runs
    """
    if os.path.exists('smac3_output'):
        shutil.rmtree('smac3_output')


def run_experiment_on_dataset(X_train, y_train, X_val, y_val, X_test, y_test,
                              cv=None, seed=0, dataset_id=1590, noise_rate=0.25):
    # Define HPO algorithms
    rs = RandomSearch(iterations=config.iterations, verbose=config.debugging, problem_type='binary', cv=cv)

    bo = BayesianOptimization(iterations=config.iterations, problem_type='binary', verbose=config.debugging,
                              prevention=None, cv=cv, seed=seed)

    bo_thresholdout = BayesianOptimization(iterations=config.iterations, verbose=config.debugging,
                                           prevention='thresholdout', tho_noise=noise_rate,
                                           seed=seed, problem_type='binary', cv=cv)

    # Fit HPO algorithms
    rs.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    bo.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    bo_thresholdout.fit(X_train, y_train, X_val, y_val, X_test, y_test)

    # Check if experiment fully succeeded
    if len(rs.public_scores) == config.iterations and \
            len(bo.public_scores) == config.iterations and \
            len(bo_thresholdout.public_scores) == config.iterations and \
            len(rs.private_scores) == config.iterations and \
            len(bo.private_scores) == config.iterations and \
            len(bo_thresholdout.private_scores) == config.iterations:
        # Save results
        results = {'dataset_id': dataset_id,
                   'seed': seed,
                   'iterations': config.iterations,
                   'rs_public': rs.public_scores,
                   'rs_private': rs.private_scores,
                   'rs_configs': rs.configs,
                   'bo_public': bo.public_scores,
                   'bo_private': bo.private_scores,
                   'bo_configs': bo.configs,
                   'tho_public': bo_thresholdout.public_scores,
                   'tho_private': bo_thresholdout.private_scores,
                   'tho_configs': bo_thresholdout.configs,
                   'train_val_test': (len(X_train), len(X_val), len(X_test)),
                   'n_features': X_train.shape[1]}

        with open(f'{config.out_dir}/{dataset_id}_{seed}_val-{X_val.shape[0]}_nr-{noise_rate}_cv-{cv}.pkl', 'wb') as f:
            pickle.dump(results, f)

        reset_smac()

        # Experiment successful -> return True
        return True

    # Experiment not successful -> return False
    return False


def run_samples_experiment(dataset_id, val_sizes=None, seed=0) -> None:
    """
    Driver function to perform one experiment using different validation sizes
    :param dataset_id: OpenML ID of dataset
    :param val_sizes: Sizes to use. Default is 100, 200, 500, 1000, 2000, 5000, and 10000
    :param seed: Random seed to use
    """
    if val_sizes is None:
        val_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]

        # Retrieve a dataset by its ID (e.g., Iris dataset with ID 61)
        dataset = openml.datasets.get_dataset(dataset_id)

        # Get the dataset's data (features and labels)
        X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        y = [0 if x == y[0] else 1 for x in y]
        X['label'] = y

        # Drop all empty columns
        df = X.sample(frac=1.0, random_state=seed)
        df.reset_index()
        cols_to_drop = [c for c in X.columns if list(set(X[c])) == [None]]
        X = df.drop(columns=cols_to_drop, inplace=False)

        # Determine fixed train and test sizes
        train_size = 1000
        test_size = min(X.shape[0] - train_size - max(val_sizes), 100000)

        # Sample fixed train set
        X_train = X.sample(n=train_size, random_state=seed)
        X = X.drop(X_train.index)

        # Sample fixed test set
        X_test = X.sample(n=test_size, random_state=seed)
        X = X.drop(X_test.index)

        # Split labels
        y_train = X_train['label']
        y_test = X_test['label']
        X_train = X_train.drop(columns=['label'])
        X_test = X_test.drop(columns=['label'])

        for val_size in val_sizes:
            done = False

            while not done:
                # Sampling with same seeds leads to same sequence
                # Change seeds each iteration for randomness, but use seed + val_size for reproducibility
                X_val = X.sample(n=val_size, random_state=seed + val_size)
                y_val = X_val['label']
                X_val = X_val.drop(columns=['label'])

                done = run_experiment_on_dataset(X_train, y_train,
                                                 X_val, y_val,
                                                 X_test, y_test,
                                                 seed=seed,
                                                 cv=None,
                                                 dataset_id=dataset_id)


def run():
    """
    Method to perform one repetition (7 experiments) on adult dataset
    """
    seed = draw_seed()
    seed_everything(seed)
    run_samples_experiment(1590, seed=seed)
