from tqdm import tqdm

from configspace import get_configspace
from pipeline import initialize_pipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from smac import HyperparameterOptimizationFacade, Scenario
from overfit_prevention import Thresholdout
from pathlib import Path


class BayesianOptimization:
    def __init__(self, problem_type='classification', iterations=1000, verbose=False, prevention=None,
                 tho_noise=None, cv=None, seed=0):

        self.iterations = iterations

        # Define configuration space and best configuration so far
        self.config_space = get_configspace(problem_type)
        self.best_config = self.config_space.sample_configuration()
        self.verbose = verbose

        self.problem_type = problem_type
        self.prevention = prevention
        self.thresholdout_noise = tho_noise
        self.seed = seed

        # Define metric to score configurations
        self.score = self.accuracy if problem_type != 'regression' else self.rmse

        # Classification
        self.best_score = 0.0 if self.problem_type != 'regression' else -np.inf

        # Define lists to store configurations and their evaluations
        self.public_scores = []
        self.private_scores = []
        self.configs = []

        self.cv = cv

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        pbar = tqdm(total=self.iterations, desc='BO')

        # Define thresholdout object if selected
        if self.prevention == 'thresholdout':
            thresholdout = Thresholdout(y_val, noise_rate=self.thresholdout_noise)

        # Define nested target function to work with SMAC3
        def target_function(config, seed=self.seed) -> float:
            # Initialize pipeline from configuration
            model = initialize_pipeline(config, problem_type=self.problem_type, verbose=self.verbose)

            pbar.set_postfix(
                {"Fitting Model": str(config.get('model')).split('.')[-1],
                 "Best Score": self.best_score,
                 "Best Model": str(self.best_config.get('model')).split('.')[-1]})

            # Fit model and predict and evaluate on validation and test sets
            if self.cv is None:
                model.fit(X_train, y_train)

                val_pred = model.predict(X_val)
                val_score = self.score(y_val, val_pred)

                test_pred = model.predict(X_test)
                test_score = self.score(y_test, test_pred)

            # Cross-validation procedure
            else:
                # Define folds
                kf = KFold(n_splits=self.cv, shuffle=False)

                # Concatenate train and validation data to be split in folds
                X = pd.concat((X_train, X_val), axis=0)
                y = pd.concat((y_train, y_val), axis=0)

                val_scores_cv = []
                test_scores_cv = []

                # Loop over folds
                for train_index, val_index in kf.split(X):
                    model = initialize_pipeline(config, problem_type=self.problem_type,
                                                verbose=self.verbose)

                    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
                    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

                    model.fit(X_train_fold, y_train_fold)

                    val_pred = model.predict(X_val_fold)
                    val_acc = self.score(y_val_fold, val_pred)
                    val_scores_cv.append(val_acc)

                    test_pred = model.predict(X_test)
                    test_acc = self.score(y_test, test_pred)
                    test_scores_cv.append(test_acc)

                # Average the evaluation of the folds
                val_score = np.mean(val_scores_cv)
                test_score = np.mean(test_scores_cv)

            # Append scores of configuration
            self.public_scores.append(val_score)
            self.private_scores.append(test_score)

            # Update best configuration and score
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_config = config

            # Add config
            self.configs.append(config.get_dictionary())

            pbar.update(1)

            # When thresholdout is used, return validation score from thresholdout
            # The returned values in this function are used to guide BO
            if self.prevention == 'thresholdout':
                train_score = self.score(y_train, model.predict(X_train))
                val_score = thresholdout.score(train_score, val_score)
                return val_score

            if self.problem_type != 'regression':
                return 1 - val_score
            else:
                return -val_score

        # Scenario object specifying the optimization environment
        scenario = Scenario(self.config_space, output_directory=Path(f"smac3_output/{self.prevention}/{self.seed}"),
                            deterministic=True, n_trials=self.iterations, seed=self.seed)

        # Use SMAC to find the best configuration/hyperparameters
        smac = HyperparameterOptimizationFacade(scenario, target_function, logging_level=False, overwrite=True)
        smac.optimize()
        smac.runhistory.reset()

        pbar.close()

    def accuracy(self, y_true, y_pred, threshold=0.5):
        if y_pred.dtype != int:
            if self.problem_type == 'multiclass':
                raise ValueError('Wrong dtype in predictions')
            y_pred = (y_pred > threshold).astype(int)
        return np.where(y_true == y_pred, 1, 0).sum() / len(y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return -(np.sqrt(np.square(y_true - y_pred)).mean())
