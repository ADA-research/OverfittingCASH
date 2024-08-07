import os

from random_search import RandomSearch
from bo import BayesianOptimization
import random
import numpy as np
import pickle
import shutil
import openml
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

if not os.path.exists('results'):
    os.mkdir('results')


class Config:
    """
    Class to store configurations
    """

    def __init__(self):
        self.out_dir = 'results/openml'
        self.validation_samples = 500
        self.iterations = 250
        self.debugging = False
        self.noise_rate = 0.25

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if os.path.exists('smac3_output'):
            shutil.rmtree('smac3_output')


config = Config()


def custom_split(X, y, seed=0):
    """
    Function to shuffle and split any dataset
    """

    # Negative regression labels cause errors, so add minimum to all labels
    if min(y) < 0:
        min_label = min(y)
        y = [x - min_label for x in y]

    # Add label to dataframe and shuffle
    X['label'] = y
    df = X.sample(frac=1.0)
    df.reset_index(drop=True)

    # Separate label and drop empty columns
    y = df['label']
    cols_to_drop = [c for c in X.columns if list(set(X[c])) == [None]] + ['label']
    X = df.drop(columns=cols_to_drop, inplace=False)

    # When the dataset is small, we split .4/.2/.4
    if X.shape[0] < 5 * config.validation_samples:
        # If the dataset is small enough, proceed with a simple three-way split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=seed, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2 / 3, random_state=seed,
                                                        shuffle=True)
    # When the dataset is large, we split validation_samples * 2/validation_samples/rest
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=2 * config.validation_samples,
                                                            random_state=seed, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=config.validation_samples,
                                                        random_state=seed,
                                                        shuffle=True)
    # In case of extremely large datasets, we only sample the first 100,000 instances for the test set
    if X_test.shape[0] > 100000:
        X_test, y_test = X_test[:100000], y_test[:100000]

    return X_train, X_val, X_test, y_train, y_val, y_test


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


def run_experiment_on_dataset(dataset_id, cv=None, problem_type='binary'):
    """
    Main function that performs the experiment on a dataset
    :param dataset_id: Id of openml dataset to be used
    :param cv: Number of folds in cross-validation. If None, the holdout method is used
    :param problem_type: binary/multiclass/regression
    """
    seed = draw_seed()
    seed_everything(seed)

    # Retrieve a dataset by its ID
    dataset = openml.datasets.get_dataset(dataset_id)

    # Get the dataset's data (features and labels)
    X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

    # Map classes to integers if required
    if problem_type != 'regression':
        classes = list(set(y))
        class_map = dict(zip(classes, range(len(classes))))
        y = [class_map[x] for x in y]

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = custom_split(X, y, seed=seed)
    print(f"\nRunning seed {seed} on ds {dataset_id} with shapes ({len(X_train)}, {len(X_val)}, {len(X_test)}) "
          f"and {X_train.shape[1]} features for cv={cv}")

    # Initialize HPO algorithms
    rs = RandomSearch(iterations=config.iterations, verbose=config.debugging, problem_type=problem_type, cv=cv)

    bo = BayesianOptimization(iterations=config.iterations, problem_type=problem_type, verbose=config.debugging,
                              prevention=None, cv=cv, seed=seed)

    bo_thresholdout = BayesianOptimization(iterations=config.iterations, verbose=config.debugging,
                                           prevention='thresholdout', tho_noise=config.noise_rate,
                                           seed=seed, problem_type=problem_type, cv=cv)

    # Fit all HPO algorithms
    rs.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    bo.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    bo_thresholdout.fit(X_train, y_train, X_val, y_val, X_test, y_test)

    # Store results in dictionary
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

    # Store results as pickle
    with open(f'{config.out_dir}/{dataset_id}_{seed}_val-{X_val.shape[0]}_cv-{cv}.pkl', 'wb') as f:
        pickle.dump(results, f)

    reset_smac()


if __name__ == '__main__':
    # Example usage to perform one repetition of the experiment on adult dataset
    run_experiment_on_dataset(1590, cv=None, problem_type='binary')
