from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.experimental import enable_iterative_imputer

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, \
    QuantileTransformer, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, ARDRegression, \
    PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, ExtraTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, \
    ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import scipy


class PrintOutput(BaseEstimator, TransformerMixin):
    """
    Class for debugging purposes
    """

    def __init__(self, label=''):
        self.label = label  # Optional: a label to identify the output

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        print(X.shape, self.label)
        print(np.isinf(X).any())
        return X  # Return the data unchanged


class ToDense(BaseEstimator, TransformerMixin):
    """
    Pipeline component to transform sparse matrices to dense
    """

    def __init__(self, label=''):
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, scipy.sparse._csr.csr_matrix):
            return np.asarray(X.todense())
        return X


class DynamicDimensionReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction component that takes into account the maximum number of dimensions to be reduced to
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        # Determine the appropriate n_components based on the number of features in X
        n_features = X.shape[1]
        self.estimator.n_components = max(1, min(n_features - 1, self.estimator.n_components))
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.estimator.transform(X)


def initialize_classifier(config):
    # Model initialization
    if config['model'] == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=config.get('n_estimators'),
            max_depth=config.get('max_depth'),
            min_samples_split=config.get('rf_min_samples_split'),
            min_samples_leaf=config.get('rf_min_samples_leaf'),
            criterion=config.get('rf_criterion'),
            bootstrap=config.get('rf_bootstrap'),
            n_jobs=-1
        )

    elif config['model'] == 'AdaBoostClassifier':
        model = AdaBoostClassifier(
            n_estimators=config.get('ab_n_estimators'),
            learning_rate=config.get('ab_learning_rate')
        )

    elif config['model'] == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=config.get('gb_n_estimators'),
            learning_rate=config.get('gb_learning_rate'),
            max_depth=config.get('gb_max_depth'),
            min_samples_split=config.get('gb_min_samples_split'),
            min_samples_leaf=config.get('gb_min_samples_leaf'),
            loss=config.get('gb_loss'),
            subsample=config.get('gb_subsample')
        )

    elif config['model'] == 'BaggingClassifier':
        model = BaggingClassifier(
            n_estimators=config.get('bagging_n_estimators'),
            max_samples=config.get('bagging_max_samples'),
            max_features=config.get('bagging_max_features'),
            bootstrap=config.get('bagging_bootstrap'),
            n_jobs=-1
        )

    elif config['model'] == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(
            max_depth=config.get('dt_max_depth'),
            min_samples_split=config.get('dt_min_samples_split'),
            min_samples_leaf=config.get('dt_min_samples_leaf')
        )

    elif config['model'] == 'ExtraTreeClassifier':
        model = ExtraTreeClassifier(
            max_depth=config.get('et_max_depth'),
            min_samples_split=config.get('et_min_samples_split'),
            min_samples_leaf=config.get('et_min_samples_leaf'),
            splitter=config.get('et_splitter'),
            criterion=config.get('et_criterion')
        )

    elif config['model'] == 'ExtraTreesClassifier':
        max_features = config['ets_max_features'] if 'ets_max_features' in config.keys() else None

        model = ExtraTreesClassifier(
            n_estimators=config['ets_n_estimators'],
            criterion=config['ets_criterion'],
            max_features=max_features,
            min_samples_split=config['ets_min_samples_split'],
            min_samples_leaf=config['ets_min_samples_leaf'],
            min_weight_fraction_leaf=config['ets_min_weight_fraction_leaf'],
            max_leaf_nodes=config['ets_max_leaf_nodes'],
            min_impurity_decrease=config['ets_min_impurity_decrease'],
            bootstrap=config['ets_bootstrap'],
            n_jobs=-1
        )

    elif config['model'] == 'LogisticRegression':
        model = LogisticRegression(
            C=config.get('lr_C'),
            solver=config.get('lr_solver'),
            n_jobs=-1
        )

    elif config['model'] == 'SGDClassifier':
        model = SGDClassifier(
            loss=config.get('sgd_loss'),
            penalty=config.get('sgd_penalty'),
            alpha=config.get('sgd_alpha'),
            learning_rate=config.get('sgd_learning_rate'),
            eta0=config.get('sgd_eta0'),
            l1_ratio=config.get('sgd_l1_ratio'),
            power_t=config.get('sgd_power_t'),
            n_jobs=-1
        )

    elif config['model'] == 'PassiveAggressiveClassifier':
        model = PassiveAggressiveClassifier(
            C=config.get('pac_C'),
            max_iter=config.get('pac_max_iter'),
            tol=config.get('pac_tol'),
            n_jobs=-1
        )

    elif config['model'] == 'RidgeClassifier':
        model = RidgeClassifier(
            alpha=config.get('ridge_alpha'),
            solver=config.get('ridge_solver')
        )

    elif config['model'] == 'Lasso':
        model = Lasso(
            alpha=config.get('lasso_alpha')
        )

    elif config['model'] == 'ElasticNet':
        model = ElasticNet(
            alpha=config.get('en_alpha'),
            l1_ratio=config.get('en_l1_ratio')
        )

    elif config['model'] == 'KNeighborsClassifier':
        model = KNeighborsClassifier(
            n_neighbors=config.get('knc_n_neighbors'),
            weights=config.get('knc_weights'),
            algorithm=config.get('knc_algorithm'),
            leaf_size=config.get('knc_leaf_size'),
            p=config.get('knc_p'),
            n_jobs=-1
        )

    elif config['model'] == 'GaussianNB':
        model = GaussianNB(
            var_smoothing=config.get('gnb_var_smoothing')
        )

    elif config['model'] == 'MLPClassifier':
        model = MLPClassifier(
            hidden_layer_sizes=(config.get('mlp_hidden_layer_size'),),
            activation=config.get('mlp_activation'),
            solver=config.get('mlp_solver'),
            learning_rate_init=config['mlp_lr_init'] if config['mlp_solver'] in ['adam', 'sgd'] else 0.001,
            max_iter=config.get('mlp_max_iter'),
            early_stopping=config.get('mlp_early_stop')
        )

    elif config['model'] == 'LinearDiscriminantAnalysis':
        model = LinearDiscriminantAnalysis(
            solver=config.get('lda_solver'),
            shrinkage=config.get('lda_shrinkage')
        )

    else:
        raise ValueError(f'Unknown model encountered: {config["model"]}')

    return model


def initialize_regressor(config):
    if config['model'] == 'LinearRegression':
        model = LinearRegression()

    elif config['model'] == 'LogisticRegression':
        model = LogisticRegression(
            C=config.get('lr_C'),
            solver=config.get('lr_solver')
        )

    elif config['model'] == 'Ridge':
        model = Ridge(
            alpha=config['ridge_alpha'],
            solver=config['ridge_solver']
        )

    elif config['model'] == 'Lasso':
        model = Lasso(
            alpha=config['lasso_alpha']
        )

    elif config['model'] == 'ElasticNet':
        model = ElasticNet(
            alpha=config['en_alpha'],
            l1_ratio=config['en_l1_ratio']
        )

    elif config['model'] == 'SGDRegressor':
        model = SGDRegressor(
            loss=config['sgd_loss'],
            penalty=config['sgd_penalty'],
            learning_rate=config['sgd_learning_rate'],
            l1_ratio=config['sgd_l1_ratio'],
            power_t=config['sgd_power_t'],
            eta0=config['sgd_eta0'],
            alpha=config['sgd_alpha']
        )

    elif config['model'] == 'ARDRegression':
        model = ARDRegression(
            alpha_1=config['ard_alpha_1'],
            alpha_2=config['ard_alpha_2'],
            lambda_1=config['ard_lambda_1'],
            lambda_2=config['ard_lambda_2'],
            threshold_lambda=config['ard_threshold_lambda']
        )

    elif config['model'] == 'PassiveAggressiveRegressor':
        model = PassiveAggressiveRegressor(
            C=config['pac_c'],
            max_iter=config['pac_max_iter'],
            tol=config['pac_tolerance']
        )

    elif config['model'] == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(
            max_depth=config['dt_max_depth'],
            min_samples_split=config['dt_min_samples_split'],
            min_samples_leaf=config['dt_min_samples_leaf']
        )

    elif config['model'] == 'RandomForestRegressor':
        model = RandomForestRegressor(
            n_estimators=config['rf_n_estimators'],
            max_depth=config['rf_max_depth'],
            min_samples_split=config['rf_min_samples_split'],
            min_samples_leaf=config['rf_min_samples_leaf'],
            criterion=config['rf_criterion'],
            bootstrap=config['rf_bootstrap']
        )

    elif config['model'] == 'AdaBoostRegressor':
        model = AdaBoostRegressor(
            n_estimators=config['ada_n_estimators'],
            learning_rate=config['ada_learning_rate']
        )

    elif config['model'] == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(
            loss=config['gbr_loss'],
            n_estimators=config['gbr_n_estimators'],
            learning_rate=config['gbr_learning_rate'],
            max_depth=config['gbr_max_depth'],
            subsample=config['gbr_subsample'],
            min_samples_split=config['gbr_min_samples_split'],
            min_samples_leaf=config['gbr_min_samples_leaf']
        )

    elif config['model'] == 'BaggingRegressor':
        model = BaggingRegressor(
            n_estimators=config['bagging_n_estimators'],
            max_samples=config['bagging_max_samples'],
            max_features=config['bagging_max_features'],
            bootstrap=config['bagging_bootstrap']
        )

    elif config['model'] == 'ExtraTreeRegressor':
        model = ExtraTreeRegressor(
            max_depth=config['etr_max_depth'],
            criterion=config['etr_criterion'],
            min_samples_split=config['etr_min_samples_split'],
            min_samples_leaf=config['etr_min_samples_leaf'],
            splitter=config['etr_splitter']
        )

    elif config['model'] == 'ExtraTreesRegressor':
        max_features = config['etrs_max_features'] if 'etrs_max_features' in config.keys() else None

        model = ExtraTreesRegressor(
            n_estimators=config['etrs_n_estimators'],
            criterion=config['etrs_criterion'],
            max_features=max_features,
            min_samples_split=config['etrs_min_samples_split'],
            min_samples_leaf=config['etrs_min_samples_leaf'],
            min_weight_fraction_leaf=config['etrs_min_weight_fraction_leaf'],
            max_leaf_nodes=config['etrs_max_leaf_nodes'],
            min_impurity_decrease=config['etrs_min_impurity_decrease'],
            bootstrap=config['etrs_bootstrap'],
        )

    elif config['model'] == 'KNeighborsRegressor':
        model = KNeighborsRegressor(
            n_neighbors=config['knr_n_neighbors'],
            weights=config['knr_weights'],
            algorithm=config['knr_algorithm'],
            leaf_size=config['knr_leaf_size'],
            p=config['knr_p']
        )

    elif config['model'] == 'MLPRegressor':
        model = MLPRegressor(
            hidden_layer_sizes=(config['mlp_hidden_layer_size'],),
            activation=config['mlp_activation'],
            solver=config['mlp_solver'],
            alpha=config['mlp_alpha'],
            learning_rate_init=config['mlp_lr_init'] if config['mlp_solver'] in ['adam', 'sgd'] else 0.001,
            max_iter=config['mlp_max_iter'],
            early_stopping=config['mlp_early_stop']
        )

    else:
        raise ValueError(f'Unknown model: {config.get("model")}')

    return model


def initialize_pipeline(config, problem_type='binary', verbose=False) -> Pipeline:
    """
    Create sklearn pipeline object from one configuration
    :param config: Configuration to use to create pipeline
    :param problem_type: binary/multiclass/regression
    :param verbose: When True, print output and shapes of each pipeline component when fitting
    :return:
    """
    if problem_type != 'regression':
        model = initialize_classifier(config)
    else:
        model = initialize_regressor(config)

    scaler = 'passthrough'

    # Scaler initialization based on ConfigSpace
    scaler_choice = config.get('scaler')
    if scaler_choice == 'StandardScaler':
        scaler = StandardScaler()

    elif scaler_choice == 'MinMaxScaler':
        scaler = MinMaxScaler()

    elif scaler_choice == 'MaxAbsScaler':
        scaler = MaxAbsScaler()

    elif scaler_choice == 'RobustScaler':
        quantile_range = config.get('robust_scaler_quantile_range', 25.0)

        quantile = 100 - quantile_range if quantile_range > 50 else quantile_range
        scaler = RobustScaler(quantile_range=(quantile, 100 - quantile))

    elif scaler_choice == 'Normalizer':
        scaler = Normalizer()

    elif scaler_choice == 'QuantileTransformer':
        n_quantiles = config.get('quantile_transformer_n_quantiles', 1000)
        output_distribution = config.get('quantile_transformer_output_distribution', 'uniform')
        scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)

    elif scaler_choice == 'PowerTransformer':
        scaler = PowerTransformer()

    # DimRed initialization
    dimred_choice = config.get('dim_reduction')

    dimred = 'passthrough'

    if dimred_choice == 'PCA':
        dimred = PCA(n_components=config.get('num_components'),
                     whiten=config.get('pca_whiten'))

    elif dimred_choice == 'FastICA':
        dimred = FastICA(n_components=config.get('num_components'),
                         algorithm=config.get('fastica_algorithm'),
                         fun=config.get('fastica_fun'),
                         max_iter=config.get('fastica_max_iter'))

    num_imputer = None
    num_imputer_choice = config.get('imputer')

    if num_imputer_choice == 'SimpleImputer':
        num_imputer = SimpleImputer(
            strategy=config.get('simple_strategy')
        )
    elif num_imputer_choice == 'IterativeImputer':
        num_imputer = IterativeImputer(
            max_iter=config.get('iterative_max_iter'),
            imputation_order=config.get('iterative_imputation_order'),
            skip_complete=True
        )
    elif num_imputer_choice == 'KNNImputer':
        num_imputer = KNNImputer(
            n_neighbors=config.get('knn_n_neighbors'),
            weights=config.get('knn_weights'),
        )
    cat_imputer = SimpleImputer(strategy=config.get('cat_imputer'), fill_value='missing')

    if config.get('encoder') == 'OrdinalEncoder':
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        encoder = OneHotEncoder(handle_unknown='infrequent_if_exist')

    # Create transformers for categorical and numerical data
    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', cat_imputer),
        ('encoder', encoder)
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_imputer, selector(dtype_exclude="category")),
            ('cat', categorical_transformer, selector(dtype_include="category"))
        ])

    # Add the feature selector initialization based on ConfigSpace
    feature_selector_choice = config.get('feature_selector')

    feature_selector = 'passthrough'  # Default to no selection

    if feature_selector_choice == 'VarianceThreshold':
        feature_selector = VarianceThreshold(threshold=config.get('variance_threshold', 0.0))

    elif feature_selector_choice == 'SelectKBest':
        if config.get('score_func') == 'f_classif':
            score_func = f_classif
        elif config.get('score_func') == 'mutual_info_classif':
            score_func = mutual_info_classif
        elif config.get('score_func') == 'f_regression':
            score_func = f_regression
        elif config.get('score_func') == 'mutual_info_regression':
            score_func = mutual_info_regression
        else:
            raise ValueError('Unknown score func: ', config.get('score_func'))

        feature_selector = SelectKBest(score_func=score_func, k=config.get('k_best'))

    elif feature_selector_choice == 'SelectPercentile':
        if config.get('score_func_per') == 'f_classif':
            score_func = f_classif
        elif config.get('score_func_per') == 'mutual_info_classif':
            score_func = mutual_info_classif
        elif config.get('score_func_per') == 'f_regression':
            score_func = f_regression
        elif config.get('score_func_per') == 'mutual_info_regression':
            score_func = mutual_info_regression
        else:
            raise ValueError('Unknown score func: ', config.get('score_func_per'))

        percentile = config.get('percentile')
        feature_selector = SelectPercentile(score_func=score_func, percentile=percentile)

    dimred_transformer = DynamicDimensionReducer(dimred) if not isinstance(dimred, str) else 'passthrough'

    if verbose:
        steps = [('print0', PrintOutput('start')),
                 ('preprocess', preprocessor),
                 ('print2', PrintOutput('2')),
                 ('todense', ToDense()),
                 ('print', PrintOutput('after dense')),
                 ('feature_selection', feature_selector),
                 ('print3', PrintOutput('3')),
                 ('scaler', scaler),
                 ('print4', PrintOutput('4')),
                 ('dimred', dimred_transformer),
                 ('print5', PrintOutput('5')),
                 ('model', model)]
    else:
        steps = [('preprocess', preprocessor),
                 ('todense', ToDense()),
                 ('feature_selection', feature_selector),
                 ('scaler', scaler),
                 ('dimred', dimred_transformer),
                 ('model', model)]

    return Pipeline(steps)
