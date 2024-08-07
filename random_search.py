import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

from configspace import get_configspace
from pipeline import initialize_pipeline

from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


class RandomSearch:
    def __init__(self, problem_type='binary', iterations=1000, cv=None, verbose=False):
        self.iterations = iterations

        # Initialize configuration space and best configuration so far
        self.config_space = get_configspace(problem_type)
        self.best_config = self.config_space.sample_configuration()

        self.verbose = verbose
        self.problem_type = problem_type
        self.cv = cv

        # Define scoring function based on problem type
        self.score = self.accuracy if problem_type != 'regression' else self.rmse

        # Determine worst score depending on problem type
        self.best_score = 0.0 if self.problem_type != 'regression' else -np.inf

        # Initialize lists to store configurations and its validation and test scores
        self.public_scores = []
        self.private_scores = []
        self.configs = []

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Fit random search
        """
        with tqdm(total=self.iterations, desc='Random Search') as pbar:
            for i in range(self.iterations):
                config = self.config_space.sample_configuration()

                model = initialize_pipeline(config, problem_type=self.problem_type,
                                            verbose=self.verbose)

                pbar.set_postfix(
                    {"Fitting Model": str(config.get('model')).split('.')[-1],
                     "Best Score": self.best_score,
                     "Best Model": str(self.best_config.get('model')).split('.')[-1]})

                # Use holdout method
                if self.cv is None:
                    model.fit(X_train, y_train)

                    val_pred = model.predict(X_val)
                    val_acc = self.score(y_val, val_pred)
                    self.public_scores.append(val_acc)

                    test_pred = model.predict(X_test)
                    test_acc = self.score(y_test, test_pred)
                    self.private_scores.append(test_acc)

                # Use cross-validation
                else:
                    kf = KFold(n_splits=self.cv, shuffle=False)
                    X = pd.concat((X_train, X_val), axis=0)
                    y = pd.concat((y_train, y_val), axis=0)

                    val_scores_cv = []
                    test_scores_cv = []

                    for train_index, val_index in kf.split(X):
                        model = initialize_pipeline(config, X_train.shape[1], problem_type=self.problem_type,
                                                    verbose=self.verbose)

                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                        model.fit(X_train, y_train)

                        val_pred = model.predict(X_val)
                        val_acc = self.score(y_val, val_pred)
                        val_scores_cv.append(val_acc)

                        test_pred = model.predict(X_test)
                        test_acc = self.score(y_test, test_pred)
                        test_scores_cv.append(test_acc)

                    # Average scores over all folds
                    val_acc = np.mean(val_scores_cv)
                    test_acc = np.mean(test_scores_cv)

                    self.public_scores.append(val_acc)
                    self.private_scores.append(test_acc)

                self.configs.append(config.get_dictionary())

                # Update best score
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_config = config

                pbar.update(1)

    def accuracy(self, y_true, y_pred, threshold=0.5):
        if y_pred.dtype != int:
            if self.problem_type == 'multiclass':
                raise ValueError('Wrong dtype in predictions')
            y_pred = (y_pred > threshold).astype(int)
        return np.where(y_true == y_pred, 1, 0).sum() / len(y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        return -(np.sqrt(np.square(y_true - y_pred)).mean())
