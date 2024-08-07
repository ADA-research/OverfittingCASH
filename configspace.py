from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition, InCondition, AndConjunction, OrConjunction
from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction


def get_configspace(problem_type):
    # Define models included in classification tasks
    if problem_type != 'regression':
        models = [
            'RandomForestClassifier',
            'AdaBoostClassifier',
            'DecisionTreeClassifier',
            'GradientBoostingClassifier',
            'BaggingClassifier',
            'ExtraTreeClassifier',
            'ExtraTreesClassifier',
            'LogisticRegression',
            'SGDClassifier',
            'PassiveAggressiveClassifier',
            'RidgeClassifier',
            'Lasso',
            'ElasticNet',
            'GaussianNB',
            'MLPClassifier',
            'KNeighborsClassifier',
            'LinearDiscriminantAnalysis',
        ]

        cs = ConfigurationSpace()

        model_hp = CategoricalHyperparameter('model', choices=models)
        cs.add_hyperparameter(model_hp)

        # RandomForestClassifier Hyperparameters
        n_estimators = UniformIntegerHyperparameter('n_estimators', 10, 200)
        max_depth = UniformIntegerHyperparameter('max_depth', 1, 25)
        rf_min_samples_split = UniformIntegerHyperparameter('rf_min_samples_split', 2, 20)
        rf_min_samples_leaf = UniformIntegerHyperparameter('rf_min_samples_leaf', 1, 20)
        rf_criterion = CategoricalHyperparameter('rf_criterion', choices=['gini', 'entropy'])
        rf_bootstrap = CategoricalHyperparameter('rf_bootstrap', choices=[True, False])

        cs.add_hyperparameters(
            [n_estimators, max_depth, rf_min_samples_split, rf_min_samples_leaf, rf_bootstrap, rf_criterion])

        for x in [n_estimators, max_depth, rf_min_samples_split, rf_min_samples_leaf, rf_bootstrap, rf_criterion]:
            cs.add_condition(EqualsCondition(x, model_hp, 'RandomForestClassifier'))

        # LogisticRegression Hyperparameters
        lr_C = UniformFloatHyperparameter('lr_C', 1e-4, 50, log=True)
        lr_solver = CategoricalHyperparameter('lr_solver', choices=['newton-cg', 'liblinear', 'lbfgs', 'saga', 'sag',
                                                                    'newton-cholesky'])

        cs.add_hyperparameters([lr_C, lr_solver])

        # Conditions
        cs.add_conditions([EqualsCondition(lr_C, model_hp, 'LogisticRegression'),
                           EqualsCondition(lr_solver, model_hp, 'LogisticRegression')])

        # MLPClassifier Hyperparameters
        mlp_hidden_layer_size = UniformIntegerHyperparameter('mlp_hidden_layer_size', 10, 1000)
        mlp_activation = CategoricalHyperparameter('mlp_activation', choices=['identity', 'logistic', 'tanh', 'relu'])
        mlp_solver = CategoricalHyperparameter('mlp_solver', choices=['lbfgs', 'sgd', 'adam'])
        mlp_lr_init = UniformFloatHyperparameter('mlp_lr_init', 0.0001, 1.0, log=True)
        mlp_max_iter = UniformIntegerHyperparameter('mlp_max_iter', 10, 1000)
        mlp_early_stop = CategoricalHyperparameter('mlp_early_stop', choices=[True, False])
        mlp_alpha = UniformFloatHyperparameter('mlp_alpha', 1e-5, 1.0, log=True)

        cs.add_hyperparameters(
            [mlp_hidden_layer_size, mlp_activation, mlp_solver, mlp_alpha, mlp_lr_init, mlp_max_iter, mlp_early_stop])

        # Conditions
        cs.add_conditions([EqualsCondition(mlp_hidden_layer_size, model_hp, 'MLPClassifier'),
                           EqualsCondition(mlp_activation, model_hp, 'MLPClassifier'),
                           EqualsCondition(mlp_solver, model_hp, 'MLPClassifier'),
                           EqualsCondition(mlp_max_iter, model_hp, 'MLPClassifier'),
                           EqualsCondition(mlp_early_stop, model_hp, 'MLPClassifier'),
                           EqualsCondition(mlp_alpha, model_hp, 'MLPClassifier')])

        mlp_init_condition = InCondition(mlp_lr_init, mlp_solver, ['adam', 'sgd'])
        cs.add_condition(mlp_init_condition)

        # GradientBoostingClassifier Hyperparameters
        gb_loss = CategoricalHyperparameter('gb_loss', choices=['log_loss', 'exponential'])
        gb_n_estimators = UniformIntegerHyperparameter('gb_n_estimators', 10, 200)
        gb_learning_rate = UniformFloatHyperparameter('gb_learning_rate', 0.001, 1.0, log=True)
        gb_max_depth = UniformIntegerHyperparameter('gb_max_depth', 1, 15)
        gb_subsample = UniformFloatHyperparameter('gb_subsample', 0.05, 1.0)
        gb_min_samples_split = UniformIntegerHyperparameter('gb_min_samples_split', 2, 20)
        gb_min_samples_leaf = UniformIntegerHyperparameter('gb_min_samples_leaf', 1, 20)
        cs.add_hyperparameters(
            [gb_n_estimators, gb_learning_rate, gb_max_depth, gb_min_samples_split, gb_min_samples_leaf, gb_subsample,
             gb_loss])

        # Conditions
        cs.add_conditions([EqualsCondition(gb_n_estimators, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_learning_rate, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_max_depth, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_loss, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_subsample, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_min_samples_split, model_hp, 'GradientBoostingClassifier'),
                           EqualsCondition(gb_min_samples_leaf, model_hp, 'GradientBoostingClassifier')])

        # AdaBoostClassifier Hyperparameters
        ab_n_estimators = UniformIntegerHyperparameter('ab_n_estimators', 10, 250)
        ab_learning_rate = UniformFloatHyperparameter('ab_learning_rate', 0.001, 2.0, log=True)
        cs.add_hyperparameters([ab_n_estimators, ab_learning_rate])

        # Conditions
        cs.add_conditions([EqualsCondition(ab_n_estimators, model_hp, 'AdaBoostClassifier'),
                           EqualsCondition(ab_learning_rate, model_hp, 'AdaBoostClassifier')])

        # DecisionTreeClassifier Hyperparameters
        dt_max_depth = UniformIntegerHyperparameter('dt_max_depth', 1, 25)
        dt_min_samples_split = UniformIntegerHyperparameter('dt_min_samples_split', 2, 20)
        dt_min_samples_leaf = UniformIntegerHyperparameter('dt_min_samples_leaf', 1, 20)
        cs.add_hyperparameters([dt_max_depth, dt_min_samples_split, dt_min_samples_leaf])

        # Conditions
        cs.add_conditions([EqualsCondition(dt_max_depth, model_hp, 'DecisionTreeClassifier'),
                           EqualsCondition(dt_min_samples_split, model_hp, 'DecisionTreeClassifier'),
                           EqualsCondition(dt_min_samples_leaf, model_hp, 'DecisionTreeClassifier')])

        # BaggingClassifier Hyperparameters
        bagging_n_estimators = UniformIntegerHyperparameter('bagging_n_estimators', 10, 50)
        bagging_max_samples = UniformFloatHyperparameter('bagging_max_samples', 0.01, 1.0)
        bagging_max_features = UniformFloatHyperparameter('bagging_max_features', 0.01, 1.0)
        bagging_bootstrap = CategoricalHyperparameter('bagging_bootstrap', choices=[True, False])
        cs.add_hyperparameters([bagging_n_estimators, bagging_max_samples, bagging_max_features, bagging_bootstrap])

        # Conditions
        cs.add_conditions([EqualsCondition(bagging_n_estimators, model_hp, 'BaggingClassifier'),
                           EqualsCondition(bagging_max_samples, model_hp, 'BaggingClassifier'),
                           EqualsCondition(bagging_max_features, model_hp, 'BaggingClassifier'),
                           EqualsCondition(bagging_bootstrap, model_hp, 'BaggingClassifier')])

        # ExtraTreeClassifier Hyperparameters
        et_max_depth = UniformIntegerHyperparameter('et_max_depth', 1, 150)
        et_criterion = CategoricalHyperparameter('et_criterion', choices=['gini', 'entropy'])
        et_splitter = CategoricalHyperparameter('et_splitter', choices=['random', 'best'])
        et_min_samples_split = UniformIntegerHyperparameter('et_min_samples_split', 2, 20)
        et_min_samples_leaf = UniformIntegerHyperparameter('et_min_samples_leaf', 1, 20)
        cs.add_hyperparameters([et_max_depth, et_min_samples_split, et_min_samples_leaf, et_splitter, et_criterion])

        # Conditions
        cs.add_conditions([EqualsCondition(et_max_depth, model_hp, 'ExtraTreeClassifier'),
                           EqualsCondition(et_min_samples_split, model_hp, 'ExtraTreeClassifier'),
                           EqualsCondition(et_splitter, model_hp, 'ExtraTreeClassifier'),
                           EqualsCondition(et_criterion, model_hp, 'ExtraTreeClassifier'),
                           EqualsCondition(et_min_samples_leaf, model_hp, 'ExtraTreeClassifier')])

        # PassiveAggressiveClassifier Hyperparameters
        pac_C = UniformFloatHyperparameter('pac_C', 0.01, 10.0, log=True)
        pac_max_iter = UniformIntegerHyperparameter('pac_max_iter', 1, 3000, log=True)
        pac_tol = UniformFloatHyperparameter('pac_tol', 1e-4, 1e-1, log=True)
        cs.add_hyperparameters([pac_C, pac_max_iter, pac_tol])

        # Conditions
        cs.add_conditions([EqualsCondition(pac_C, model_hp, 'PassiveAggressiveClassifier'),
                           EqualsCondition(pac_max_iter, model_hp, 'PassiveAggressiveClassifier'),
                           EqualsCondition(pac_tol, model_hp, 'PassiveAggressiveClassifier')])

        # RidgeClassifier Hyperparameters
        ridge_alpha = UniformFloatHyperparameter('ridge_alpha', 0.01, 10.0, log=True)
        ridge_solver = CategoricalHyperparameter('ridge_solver',
                                                 choices=['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'])
        cs.add_hyperparameters([ridge_alpha, ridge_solver])

        # Conditions
        cs.add_conditions([EqualsCondition(ridge_alpha, model_hp, 'RidgeClassifier'),
                           EqualsCondition(ridge_solver, model_hp, 'RidgeClassifier')])

        # Lasso Hyperparameters
        lasso_alpha = UniformFloatHyperparameter('lasso_alpha', 0.0001, 1.0, log=True)
        cs.add_hyperparameters([lasso_alpha])

        # Conditions
        cs.add_conditions([EqualsCondition(lasso_alpha, model_hp, 'Lasso')])

        # ElasticNet Hyperparameters
        en_alpha = UniformFloatHyperparameter('en_alpha', 0.0001, 1.0, log=True)
        en_l1_ratio = UniformFloatHyperparameter('en_l1_ratio', 0.0, 1.0)
        cs.add_hyperparameters([en_alpha, en_l1_ratio])

        # Conditions
        cs.add_conditions([EqualsCondition(en_alpha, model_hp, 'ElasticNet'),
                           EqualsCondition(en_l1_ratio, model_hp, 'ElasticNet')])

        # KNeighborsClassifier Hyperparameters
        knc_n_neighbors = UniformIntegerHyperparameter('knc_n_neighbors', 1, 100)
        knc_weights = CategoricalHyperparameter('knc_weights', choices=['uniform', 'distance'])
        knc_algorithm = CategoricalHyperparameter('knc_algorithm', choices=['ball_tree', 'kd_tree', 'auto'])
        knc_leaf_size = UniformIntegerHyperparameter('knc_leaf_size', 10, 50)
        knc_p = UniformIntegerHyperparameter('knc_p', 1, 3)
        cs.add_hyperparameters([knc_n_neighbors, knc_weights, knc_algorithm, knc_leaf_size, knc_p])

        # Conditions
        cs.add_conditions([EqualsCondition(knc_n_neighbors, model_hp, 'KNeighborsClassifier'),
                           EqualsCondition(knc_weights, model_hp, 'KNeighborsClassifier'),
                           EqualsCondition(knc_algorithm, model_hp, 'KNeighborsClassifier'),
                           EqualsCondition(knc_leaf_size, model_hp, 'KNeighborsClassifier'),
                           EqualsCondition(knc_p, model_hp, 'KNeighborsClassifier')])

        # GaussianNB Hyperparameters
        gnb_var_smoothing = UniformFloatHyperparameter('gnb_var_smoothing', 1e-12, 1e-6, log=True)
        cs.add_hyperparameters([gnb_var_smoothing])

        # Conditions
        cs.add_conditions([EqualsCondition(gnb_var_smoothing, model_hp, 'GaussianNB')])

        # LinearDiscriminantAnalysis Hyperparameters
        lda_solver = CategoricalHyperparameter('lda_solver', choices=['svd', 'lsqr', 'eigen'])
        lda_shrinkage = UniformFloatHyperparameter('lda_shrinkage', 0.0, 1.0)
        cs.add_hyperparameters([lda_solver, lda_shrinkage])

        # Conditions
        lda_model_condition = EqualsCondition(lda_shrinkage, model_hp, 'LinearDiscriminantAnalysis')
        lda_hpar_condition = InCondition(lda_shrinkage, cs['lda_solver'], ['eigen', 'lsqr'])

        cs.add_condition(EqualsCondition(lda_solver, model_hp, 'LinearDiscriminantAnalysis'))
        cs.add_condition(AndConjunction(lda_model_condition, lda_hpar_condition))

        # SGDClassifier Hyperparameters
        sgd_loss = CategoricalHyperparameter('sgd_loss',
                                             choices=['huber', 'squared_epsilon_insensitive', 'squared_error',
                                                      'squared_hinge',
                                                      'perceptron', 'hinge', 'log_loss', 'modified_huber',
                                                      'epsilon_insensitive'])

        sgd_penalty = CategoricalHyperparameter('sgd_penalty', choices=['l2', 'l1', 'elasticnet'])
        sgd_alpha = UniformFloatHyperparameter('sgd_alpha', 1e-6, 1e-1, log=True)
        sgd_learning_rate = CategoricalHyperparameter('sgd_learning_rate',
                                                      choices=['constant', 'optimal', 'invscaling', 'adaptive'])
        sgd_l1_ratio = UniformFloatHyperparameter('sgd_l1_ratio', 0.0, 1.0)
        sgd_power_t = UniformFloatHyperparameter('sgd_power_t', 0.0, 50)
        sgd_eta0 = UniformFloatHyperparameter('sgd_eta0', 1e-7, 1e-2, log=True)

        cs.add_hyperparameters(
            [sgd_loss, sgd_penalty, sgd_alpha, sgd_learning_rate, sgd_eta0, sgd_power_t, sgd_l1_ratio])

        # Conditions
        cs.add_conditions([
            EqualsCondition(sgd_loss, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_penalty, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_alpha, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_learning_rate, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_power_t, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_l1_ratio, model_hp, 'SGDClassifier'),
            EqualsCondition(sgd_eta0, model_hp, 'SGDClassifier')
        ])

        # ExtraTreesClassifier Hyperparameters
        ets_n_estimators = UniformIntegerHyperparameter('ets_n_estimators', 10, 200)
        ets_criterion = CategoricalHyperparameter('ets_criterion', choices=['gini', 'entropy'])
        ets_max_features = CategoricalHyperparameter('ets_max_features', choices=['sqrt', 'log2'])
        ets_min_samples_split = UniformIntegerHyperparameter('ets_min_samples_split', 2, 20)
        ets_min_samples_leaf = UniformIntegerHyperparameter('ets_min_samples_leaf', 1, 20)
        ets_min_weight_fraction_leaf = UniformFloatHyperparameter('ets_min_weight_fraction_leaf', 0.0, 0.5)
        ets_max_leaf_nodes = UniformIntegerHyperparameter('ets_max_leaf_nodes', 10, 1000, default_value=None)
        ets_min_impurity_decrease = UniformFloatHyperparameter('ets_min_impurity_decrease', 0.0, 0.5)
        ets_bootstrap = CategoricalHyperparameter('ets_bootstrap', choices=[True, False])

        # Adding hyperparameters to the configuration space
        cs.add_hyperparameters([
            ets_n_estimators, ets_criterion, ets_max_features, ets_min_samples_split,
            ets_min_samples_leaf, ets_min_weight_fraction_leaf, ets_max_leaf_nodes,
            ets_min_impurity_decrease, ets_bootstrap
        ])

        # Conditions
        cs.add_conditions([
            EqualsCondition(ets_n_estimators, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_criterion, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_min_samples_split, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_min_samples_leaf, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_min_weight_fraction_leaf, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_max_leaf_nodes, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_min_impurity_decrease, model_hp, 'ExtraTreesClassifier'),
            EqualsCondition(ets_bootstrap, model_hp, 'ExtraTreesClassifier')
        ])

        ets_bootstrap_condition = AndConjunction(
            EqualsCondition(ets_max_features, ets_bootstrap, True),
            EqualsCondition(ets_max_features, model_hp, 'ExtraTreesClassifier')
        )

        cs.add_condition(ets_bootstrap_condition)

        # Multiclass restrictions
        if problem_type == 'multiclass':
            forbidden_lasso = ForbiddenEqualsClause(model_hp, 'Lasso')
            forbidden_elasticnet = ForbiddenEqualsClause(model_hp, 'ElasticNet')
            cs.add_forbidden_clauses([forbidden_lasso, forbidden_elasticnet])

            forbidden_gb_exp_mc = ForbiddenAndConjunction(
                ForbiddenEqualsClause(model_hp, 'GradientBoostingClassifier'),
                ForbiddenEqualsClause(gb_loss, 'exponential')
            )

            cs.add_forbidden_clause(forbidden_gb_exp_mc)

    # Add regression models and hyperparameters
    else:
        models = [
            'RandomForestRegressor',
            'AdaBoostRegressor',
            'DecisionTreeRegressor',
            'GradientBoostingRegressor',
            'BaggingRegressor',
            'ExtraTreeRegressor',
            'ExtraTreesRegressor',
            'SGDRegressor',
            'PassiveAggressiveRegressor',
            'Ridge',
            'Lasso',
            'ElasticNet',
            'MLPRegressor',
            'KNeighborsRegressor',
            'ARDRegression',
        ]

        cs = ConfigurationSpace()

        model_hp = CategoricalHyperparameter('model', choices=models)
        cs.add_hyperparameter(model_hp)

        # RidgeClassifier Hyperparameters
        ridge_alpha = UniformFloatHyperparameter('ridge_alpha', 0.01, 10.0, log=True)
        ridge_solver = CategoricalHyperparameter('ridge_solver',
                                                 choices=['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'])
        cs.add_hyperparameters([ridge_alpha, ridge_solver])

        cs.add_conditions([EqualsCondition(ridge_alpha, model_hp, 'Ridge')])
        cs.add_conditions([EqualsCondition(ridge_solver, model_hp, 'Ridge')])

        # Lasso Hyperparameters
        lasso_alpha = UniformFloatHyperparameter('lasso_alpha', 0.0001, 1.0, log=True)
        cs.add_hyperparameters([lasso_alpha])

        # Conditions
        cs.add_conditions([EqualsCondition(lasso_alpha, model_hp, 'Lasso')])

        # ElasticNet Hyperparameters
        en_alpha = UniformFloatHyperparameter('en_alpha', 0.0001, 1.0, log=True)
        en_l1_ratio = UniformFloatHyperparameter('en_l1_ratio', 0.0, 1.0)
        cs.add_hyperparameters([en_alpha, en_l1_ratio])

        # Conditions
        cs.add_conditions([EqualsCondition(en_alpha, model_hp, 'ElasticNet'),
                           EqualsCondition(en_l1_ratio, model_hp, 'ElasticNet')])

        # SGDRegressor Hyperparameters
        sgd_loss = CategoricalHyperparameter('sgd_loss',
                                             choices=['huber', 'squared_epsilon_insensitive', 'squared_error',
                                                      'epsilon_insensitive'])

        sgd_penalty = CategoricalHyperparameter('sgd_penalty', choices=['l2', 'l1', 'elasticnet'])
        sgd_alpha = UniformFloatHyperparameter('sgd_alpha', 1e-6, 1e-1, log=True)
        sgd_learning_rate = CategoricalHyperparameter('sgd_learning_rate',
                                                      choices=['constant', 'optimal', 'invscaling', 'adaptive'])
        sgd_l1_ratio = UniformFloatHyperparameter('sgd_l1_ratio', 0.0, 1.0)
        sgd_power_t = UniformFloatHyperparameter('sgd_power_t', 0.0, 50)
        sgd_eta0 = UniformFloatHyperparameter('sgd_eta0', 1e-7, 1e-2, log=True)
        cs.add_hyperparameters(
            [sgd_loss, sgd_penalty, sgd_alpha, sgd_learning_rate, sgd_eta0, sgd_power_t, sgd_l1_ratio])

        # Conditions
        cs.add_conditions([
            EqualsCondition(sgd_loss, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_penalty, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_alpha, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_learning_rate, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_power_t, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_l1_ratio, model_hp, 'SGDRegressor'),
            EqualsCondition(sgd_eta0, model_hp, 'SGDRegressor')
        ])

        # ARDRegression Hyperparameters
        ard_alpha_1 = UniformFloatHyperparameter('ard_alpha_1', 1e-6, 1e-3, log=True)
        ard_alpha_2 = UniformFloatHyperparameter('ard_alpha_2', 1e-6, 1e-3, log=True)
        ard_lambda_1 = UniformFloatHyperparameter('ard_lambda_1', 1e-6, 1e-3, log=True)
        ard_lambda_2 = UniformFloatHyperparameter('ard_lambda_2', 1e-6, 1e-3, log=True)
        ard_threshold_lambda = UniformFloatHyperparameter('ard_threshold_lambda', 1e4, 1e5)
        cs.add_hyperparameters([ard_alpha_1, ard_alpha_2, ard_lambda_1, ard_lambda_2, ard_threshold_lambda])
        cs.add_condition(EqualsCondition(ard_alpha_1, model_hp, 'ARDRegression'))
        cs.add_condition(EqualsCondition(ard_alpha_2, model_hp, 'ARDRegression'))
        cs.add_condition(EqualsCondition(ard_lambda_1, model_hp, 'ARDRegression'))
        cs.add_condition(EqualsCondition(ard_lambda_2, model_hp, 'ARDRegression'))
        cs.add_condition(EqualsCondition(ard_threshold_lambda, model_hp, 'ARDRegression'))

        # PassiveAggressiveRegressor Hyperparameters
        pac_C = UniformFloatHyperparameter('pac_c', 0.01, 10.0, log=True)
        pac_max_iter = UniformIntegerHyperparameter('pac_max_iter', 1, 3000, log=True)
        pac_tolerance = UniformFloatHyperparameter('pac_tolerance', 1e-4, 1e-1, log=True)
        cs.add_hyperparameters([pac_C, pac_max_iter, pac_tolerance])

        # Conditions
        cs.add_conditions([EqualsCondition(pac_C, model_hp, 'PassiveAggressiveRegressor'),
                           EqualsCondition(pac_max_iter, model_hp, 'PassiveAggressiveRegressor'),
                           EqualsCondition(pac_tolerance, model_hp, 'PassiveAggressiveRegressor')])

        # DecisionTreeRegressor Hyperparameters
        dt_max_depth = UniformIntegerHyperparameter('dt_max_depth', 1, 25)
        dt_min_samples_split = UniformIntegerHyperparameter('dt_min_samples_split', 2, 20)
        dt_min_samples_leaf = UniformIntegerHyperparameter('dt_min_samples_leaf', 1, 20)
        cs.add_hyperparameters([dt_max_depth, dt_min_samples_split, dt_min_samples_leaf])

        cs.add_condition(EqualsCondition(dt_max_depth, model_hp, 'DecisionTreeRegressor'))
        cs.add_condition(EqualsCondition(dt_min_samples_split, model_hp, 'DecisionTreeRegressor'))
        cs.add_condition(EqualsCondition(dt_min_samples_leaf, model_hp, 'DecisionTreeRegressor'))

        # RandomForestRegressor Hyperparameters
        rf_n_estimators = UniformIntegerHyperparameter('rf_n_estimators', 10, 200)
        rf_max_depth = UniformIntegerHyperparameter('rf_max_depth', 1, 25)
        rf_min_samples_split = UniformIntegerHyperparameter('rf_min_samples_split', 2, 20)
        rf_min_samples_leaf = UniformIntegerHyperparameter('rf_min_samples_leaf', 1, 20)
        rf_criterion = CategoricalHyperparameter('rf_criterion',
                                                 choices=['friedman_mse', 'absolute_error', 'poisson', 'squared_error'])
        rf_bootstrap = CategoricalHyperparameter('rf_bootstrap', choices=[True, False])

        cs.add_hyperparameters(
            [rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf, rf_criterion, rf_bootstrap])
        cs.add_condition(EqualsCondition(rf_n_estimators, model_hp, 'RandomForestRegressor'))
        cs.add_condition(EqualsCondition(rf_max_depth, model_hp, 'RandomForestRegressor'))
        cs.add_condition(EqualsCondition(rf_min_samples_split, model_hp, 'RandomForestRegressor'))
        cs.add_condition(EqualsCondition(rf_min_samples_leaf, model_hp, 'RandomForestRegressor'))
        cs.add_condition(EqualsCondition(rf_criterion, model_hp, 'RandomForestRegressor'))
        cs.add_condition(EqualsCondition(rf_bootstrap, model_hp, 'RandomForestRegressor'))

        # AdaBoostRegressor Hyperparameters
        ada_n_estimators = UniformIntegerHyperparameter('ada_n_estimators', 10, 250)
        ada_learning_rate = UniformFloatHyperparameter('ada_learning_rate', 0.001, 2.0, log=True)

        cs.add_hyperparameters([ada_n_estimators, ada_learning_rate])
        cs.add_condition(EqualsCondition(ada_n_estimators, model_hp, 'AdaBoostRegressor'))
        cs.add_condition(EqualsCondition(ada_learning_rate, model_hp, 'AdaBoostRegressor'))

        # GradientBoostingRegressor Hyperparameters
        gb_loss = CategoricalHyperparameter('gbr_loss',
                                            choices=['squared_error', 'absolute_error', 'huber', 'quantile'])
        gbr_n_estimators = UniformIntegerHyperparameter('gbr_n_estimators', 10, 200)
        gbr_learning_rate = UniformFloatHyperparameter('gbr_learning_rate', 0.001, 1.0, log=True)
        gbr_max_depth = UniformIntegerHyperparameter('gbr_max_depth', 1, 15)
        gb_subsample = UniformFloatHyperparameter('gbr_subsample', 0.05, 1.0)

        gbr_min_samples_split = UniformIntegerHyperparameter('gbr_min_samples_split', 2, 20)
        gbr_min_samples_leaf = UniformIntegerHyperparameter('gbr_min_samples_leaf', 1, 20)
        cs.add_hyperparameters(
            [gb_loss, gbr_n_estimators, gbr_learning_rate, gbr_max_depth, gbr_min_samples_split, gbr_min_samples_leaf,
             gb_subsample])

        cs.add_condition(EqualsCondition(gb_loss, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gbr_n_estimators, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gbr_learning_rate, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gbr_max_depth, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gbr_min_samples_split, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gbr_min_samples_leaf, model_hp, 'GradientBoostingRegressor'))
        cs.add_condition(EqualsCondition(gb_subsample, model_hp, 'GradientBoostingRegressor'))

        # BaggingRegressor Hyperparameters
        bagging_n_estimators = UniformIntegerHyperparameter('bagging_n_estimators', 10, 50)
        bagging_max_samples = UniformFloatHyperparameter('bagging_max_samples', 0.01, 1.0)
        bagging_max_features = UniformFloatHyperparameter('bagging_max_features', 0.01, 1.0)
        bagging_bootstrap = CategoricalHyperparameter('bagging_bootstrap', [True, False])
        cs.add_hyperparameters([bagging_n_estimators, bagging_max_samples, bagging_max_features, bagging_bootstrap])
        cs.add_condition(EqualsCondition(bagging_n_estimators, model_hp, 'BaggingRegressor'))
        cs.add_condition(EqualsCondition(bagging_max_samples, model_hp, 'BaggingRegressor'))
        cs.add_condition(EqualsCondition(bagging_max_features, model_hp, 'BaggingRegressor'))
        cs.add_condition(EqualsCondition(bagging_bootstrap, model_hp, 'BaggingRegressor'))

        # ExtraTreeRegressor Hyperparameters
        etr_max_depth = UniformIntegerHyperparameter('etr_max_depth', 1, 150)
        etr_criterion = CategoricalHyperparameter('etr_criterion',
                                                  choices=['squared_error', 'poisson', 'absolute_error',
                                                           'friedman_mse'])
        etr_min_samples_split = UniformIntegerHyperparameter('etr_min_samples_split', 2, 20)
        etr_min_samples_leaf = UniformIntegerHyperparameter('etr_min_samples_leaf', 1, 20)
        etr_splitter = CategoricalHyperparameter('etr_splitter', choices=['best', 'random'])

        cs.add_hyperparameters(
            [etr_max_depth, etr_min_samples_split, etr_min_samples_leaf, etr_criterion, etr_splitter])

        cs.add_condition(EqualsCondition(etr_max_depth, model_hp, 'ExtraTreeRegressor'))
        cs.add_condition(EqualsCondition(etr_criterion, model_hp, 'ExtraTreeRegressor'))
        cs.add_condition(EqualsCondition(etr_min_samples_split, model_hp, 'ExtraTreeRegressor'))
        cs.add_condition(EqualsCondition(etr_min_samples_leaf, model_hp, 'ExtraTreeRegressor'))
        cs.add_condition(EqualsCondition(etr_splitter, model_hp, 'ExtraTreeRegressor'))

        # KNeighborsRegressor Hyperparameters
        knr_n_neighbors = UniformIntegerHyperparameter('knr_n_neighbors', 1, 100)
        knr_weights = CategoricalHyperparameter('knr_weights', ['uniform', 'distance'])
        knr_algorithm = CategoricalHyperparameter('knr_algorithm', ['auto', 'ball_tree', 'kd_tree'])
        knr_leaf_size = UniformIntegerHyperparameter('knr_leaf_size', 10, 50)
        knr_p = UniformIntegerHyperparameter('knr_p', 1, 3)
        cs.add_hyperparameters([knr_n_neighbors, knr_weights, knr_algorithm, knr_leaf_size, knr_p])
        cs.add_condition(EqualsCondition(knr_n_neighbors, model_hp, 'KNeighborsRegressor'))
        cs.add_condition(EqualsCondition(knr_weights, model_hp, 'KNeighborsRegressor'))
        cs.add_condition(EqualsCondition(knr_algorithm, model_hp, 'KNeighborsRegressor'))
        cs.add_condition(EqualsCondition(knr_leaf_size, model_hp, 'KNeighborsRegressor'))
        cs.add_condition(EqualsCondition(knr_p, model_hp, 'KNeighborsRegressor'))

        # MLPRegressor Hyperparameters
        mlp_hidden_layer_size = UniformIntegerHyperparameter('mlp_hidden_layer_size', 10, 1000)
        mlp_activation = CategoricalHyperparameter('mlp_activation', choices=['identity', 'logistic', 'tanh', 'relu'])
        mlp_solver = CategoricalHyperparameter('mlp_solver', choices=['lbfgs', 'sgd', 'adam'])
        mlp_lr_init = UniformFloatHyperparameter('mlp_lr_init', 0.0001, 1.0, log=True)
        mlp_max_iter = UniformIntegerHyperparameter('mlp_max_iter', 10, 1000)
        mlp_early_stop = CategoricalHyperparameter('mlp_early_stop', choices=[True, False])
        mlp_alpha = UniformFloatHyperparameter('mlp_alpha', 1e-5, 1.0, log=True)

        cs.add_hyperparameters(
            [mlp_hidden_layer_size, mlp_activation, mlp_solver, mlp_alpha, mlp_lr_init, mlp_max_iter, mlp_early_stop])

        # Conditions
        cs.add_conditions([EqualsCondition(mlp_hidden_layer_size, model_hp, 'MLPRegressor'),
                           EqualsCondition(mlp_activation, model_hp, 'MLPRegressor'),
                           EqualsCondition(mlp_solver, model_hp, 'MLPRegressor'),
                           EqualsCondition(mlp_max_iter, model_hp, 'MLPRegressor'),
                           EqualsCondition(mlp_early_stop, model_hp, 'MLPRegressor'),
                           EqualsCondition(mlp_alpha, model_hp, 'MLPRegressor')])

        mlp_init_condition = InCondition(mlp_lr_init, mlp_solver, ['adam', 'sgd'])
        cs.add_condition(mlp_init_condition)

        forbidden_inf = ForbiddenAndConjunction(
            ForbiddenEqualsClause(model_hp, 'MLPRegressor'),
            ForbiddenEqualsClause(mlp_solver, 'sgd'),
            ForbiddenEqualsClause(mlp_activation, 'relu')
        )

        forbidden_inf2 = ForbiddenAndConjunction(
            ForbiddenEqualsClause(model_hp, 'MLPRegressor'),
            ForbiddenEqualsClause(mlp_solver, 'sgd'),
            ForbiddenEqualsClause(mlp_activation, 'identity')
        )

        forbidden_nan = ForbiddenAndConjunction(
            ForbiddenEqualsClause(model_hp, 'MLPRegressor'),
            ForbiddenEqualsClause(mlp_solver, 'sgd'),
            ForbiddenEqualsClause(mlp_activation, 'tanh')
        )

        cs.add_forbidden_clauses([forbidden_inf, forbidden_inf2, forbidden_nan])

        # ExtraTreesRegressor Hyperparameters
        etrs_n_estimators = UniformIntegerHyperparameter('etrs_n_estimators', 10, 200)
        etrs_criterion = CategoricalHyperparameter('etrs_criterion',
                                                   choices=['squared_error', 'absolute_error', 'poisson'])
        etrs_max_features = CategoricalHyperparameter('etrs_max_features', choices=['sqrt', 'log2'])
        etrs_min_samples_split = UniformIntegerHyperparameter('etrs_min_samples_split', 2, 20)
        etrs_min_samples_leaf = UniformIntegerHyperparameter('etrs_min_samples_leaf', 1, 20)
        etrs_min_weight_fraction_leaf = UniformFloatHyperparameter('etrs_min_weight_fraction_leaf', 0.0, 0.5)
        etrs_max_leaf_nodes = UniformIntegerHyperparameter('etrs_max_leaf_nodes', 10, 1000, default_value=None)
        etrs_min_impurity_decrease = UniformFloatHyperparameter('etrs_min_impurity_decrease', 0.0, 0.5)
        etrs_bootstrap = CategoricalHyperparameter('etrs_bootstrap', choices=[True, False])

        # Adding hyperparameters to the configuration space
        cs.add_hyperparameters([
            etrs_n_estimators, etrs_criterion, etrs_max_features, etrs_min_samples_split,
            etrs_min_samples_leaf, etrs_min_weight_fraction_leaf, etrs_max_leaf_nodes,
            etrs_min_impurity_decrease, etrs_bootstrap
        ])

        # Conditions
        cs.add_conditions([
            EqualsCondition(etrs_n_estimators, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_criterion, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_min_samples_split, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_min_samples_leaf, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_min_weight_fraction_leaf, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_max_leaf_nodes, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_min_impurity_decrease, model_hp, 'ExtraTreesRegressor'),
            EqualsCondition(etrs_bootstrap, model_hp, 'ExtraTreesRegressor'),
        ])

        etrs_bootstrap_condition = AndConjunction(
            EqualsCondition(etrs_max_features, etrs_bootstrap, True),
            EqualsCondition(etrs_max_features, model_hp, 'ExtraTreesRegressor'))

        cs.add_condition(etrs_bootstrap_condition)

    # Add preprocessing hyperparameters
    # Scaler choice
    scaler_choices = ['None', 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'Normalizer',
                      'QuantileTransformer', 'PowerTransformer']

    scaler_hp = CategoricalHyperparameter('scaler', choices=scaler_choices)
    cs.add_hyperparameter(scaler_hp)

    mlp_forbidden_model = 'MLPClassifier' if problem_type != 'regression' else 'MLPRegressor'
    forbidden_norm = ForbiddenAndConjunction(
        ForbiddenEqualsClause(scaler_hp, 'Normalizer'),
        ForbiddenEqualsClause(model_hp, mlp_forbidden_model)
    )

    forbidden_norm2 = ForbiddenAndConjunction(
        ForbiddenEqualsClause(scaler_hp, 'None'),
        ForbiddenEqualsClause(model_hp, mlp_forbidden_model)
    )

    cs.add_forbidden_clauses([forbidden_norm, forbidden_norm2])

    # RobustScaler Hyperparameters
    robust_scaler_quantile_range = UniformFloatHyperparameter('robust_scaler_quantile_range', 0.0, 100.0)
    cs.add_hyperparameter(robust_scaler_quantile_range)
    cs.add_condition(EqualsCondition(robust_scaler_quantile_range, scaler_hp, 'RobustScaler'))

    # QuantileTransformer Hyperparameters
    quantile_transformer_n_quantiles = UniformIntegerHyperparameter('quantile_transformer_n_quantiles', 10, 1000)
    quantile_transformer_output_distribution = CategoricalHyperparameter('quantile_transformer_output_distribution',
                                                                         choices=['uniform', 'normal'])
    cs.add_hyperparameters([quantile_transformer_n_quantiles, quantile_transformer_output_distribution])
    cs.add_conditions([EqualsCondition(quantile_transformer_n_quantiles, scaler_hp, 'QuantileTransformer'),
                       EqualsCondition(quantile_transformer_output_distribution, scaler_hp, 'QuantileTransformer')])

    mlp_forbidden_model = 'MLPClassifier' if problem_type != 'regression' else 'MLPRegressor'
    infinite_weights = ForbiddenAndConjunction(
        ForbiddenEqualsClause(model_hp, mlp_forbidden_model),
        ForbiddenEqualsClause(scaler_hp, "None")
    )

    cs.add_forbidden_clause(infinite_weights)

    # Dimensionality Reduction choice
    dim_reduction_choices = ['None', 'PCA', 'FastICA']
    dim_reduction_hp = CategoricalHyperparameter('dim_reduction', choices=dim_reduction_choices)
    cs.add_hyperparameter(dim_reduction_hp)

    # Number of components/dimensions - General for methods that use it
    num_components = UniformIntegerHyperparameter('num_components', 5, 50)
    cs.add_hyperparameter(num_components)
    cs.add_condition(InCondition(num_components, dim_reduction_hp, ['PCA', 'FastICA']))

    # PCA Hyperparameters
    pca_whiten = CategoricalHyperparameter('pca_whiten', [True, False])
    cs.add_hyperparameter(pca_whiten)
    cs.add_condition(EqualsCondition(pca_whiten, dim_reduction_hp, 'PCA'))

    # FastICA Hyperparameters
    fastica_algorithm = CategoricalHyperparameter('fastica_algorithm', ['parallel', 'deflation'])
    fastica_max_iter = UniformIntegerHyperparameter('fastica_max_iter', 50, 100)
    fastica_fun = CategoricalHyperparameter('fastica_fun', ['logcosh', 'exp', 'cube'])

    cs.add_hyperparameters([fastica_algorithm, fastica_max_iter, fastica_fun])
    cs.add_conditions([
        EqualsCondition(fastica_algorithm, dim_reduction_hp, 'FastICA'),
        EqualsCondition(fastica_max_iter, dim_reduction_hp, 'FastICA'),
        EqualsCondition(fastica_fun, dim_reduction_hp, 'FastICA')
    ])

    infinite_weights2 = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dim_reduction_hp, 'FastICA'),
        ForbiddenEqualsClause(scaler_hp, "None")
    )
    cs.add_forbidden_clause(infinite_weights2)

    # Imputation choice
    imputer_choices = ['SimpleImputer', 'IterativeImputer', 'KNNImputer']
    imputer_hp = CategoricalHyperparameter('imputer', choices=imputer_choices)
    cs.add_hyperparameter(imputer_hp)

    # SimpleImputer Hyperparameters
    simple_strategy = CategoricalHyperparameter('simple_strategy', ['mean', 'median', 'constant'])
    cs.add_hyperparameter(simple_strategy)

    cs.add_conditions([
        EqualsCondition(simple_strategy, imputer_hp, 'SimpleImputer')
    ])

    # IterativeImputer Hyperparameters
    iterative_max_iter = UniformIntegerHyperparameter('iterative_max_iter', 10, 100)
    iterative_imputation_order = CategoricalHyperparameter('iterative_imputation_order',
                                                           ['ascending', 'descending', 'roman', 'arabic'])

    cs.add_hyperparameters([iterative_max_iter, iterative_imputation_order])
    cs.add_conditions([
        EqualsCondition(iterative_max_iter, imputer_hp, 'IterativeImputer'),
        EqualsCondition(iterative_imputation_order, imputer_hp, 'IterativeImputer')
    ])

    # KNNImputer Hyperparameters
    knn_n_neighbors = UniformIntegerHyperparameter('knn_n_neighbors', 1, 10)
    knn_weights = CategoricalHyperparameter('knn_weights', ['uniform', 'distance'])
    cs.add_hyperparameters([knn_n_neighbors, knn_weights])
    cs.add_conditions([
        EqualsCondition(knn_n_neighbors, imputer_hp, 'KNNImputer'),
        EqualsCondition(knn_weights, imputer_hp, 'KNNImputer'),
    ])

    cat_imputer_choices = ['constant', 'most_frequent']
    cat_imputer_hp = CategoricalHyperparameter('cat_imputer', choices=cat_imputer_choices)
    cs.add_hyperparameter(cat_imputer_hp)

    encoder_choices = ['OrdinalEncoder', 'OneHotEncoder']
    encoder_hp = CategoricalHyperparameter('encoder', choices=encoder_choices)
    cs.add_hyperparameter(encoder_hp)

    # Feature Selector choice
    feature_selector_choices = ['None', 'VarianceThreshold', 'SelectKBest', 'SelectPercentile']
    feature_selector = CategoricalHyperparameter('feature_selector', choices=feature_selector_choices)
    cs.add_hyperparameter(feature_selector)

    # VarianceThreshold Hyperparameters
    variance_threshold = UniformFloatHyperparameter('variance_threshold', 0.0, 0.05)
    cs.add_hyperparameter(variance_threshold)
    cs.add_condition(EqualsCondition(variance_threshold, feature_selector, 'VarianceThreshold'))

    # SelectKBest Hyperparameters
    k_best = UniformIntegerHyperparameter('k_best', 3, 50)
    if problem_type != "regression":
        score_func = CategoricalHyperparameter('score_func', ['f_classif', 'mutual_info_classif'])
    else:
        score_func = CategoricalHyperparameter('score_func', ['f_regression', 'mutual_info_regression'])

    cs.add_hyperparameters([k_best, score_func])
    cs.add_conditions([EqualsCondition(k_best, feature_selector, 'SelectKBest'),
                       EqualsCondition(score_func, feature_selector, 'SelectKBest')])

    # SelectPercentile Hyperparameters
    percentile = UniformIntegerHyperparameter('percentile', 10, 100)

    if problem_type != "regression":
        score_func_per = CategoricalHyperparameter('score_func_per', ['f_classif', 'mutual_info_classif'])
    else:
        score_func_per = CategoricalHyperparameter('score_func_per', ['f_regression', 'mutual_info_regression'])

    cs.add_hyperparameter(score_func_per)
    cs.add_condition(EqualsCondition(score_func_per, feature_selector, 'SelectPercentile'))

    cs.add_hyperparameter(percentile)
    cs.add_condition(EqualsCondition(percentile, feature_selector, 'SelectPercentile'))

    return cs