# Overfitting in Combined Algorithm Selection and Hyperparameter Optimization

This repository contains the implementation of the experiments conducted for our paper: "Overfitting in Combined
Algorithm Selection and Hyperparameter Optimization".

## Installation

To use the implementation, clone the repository, install the requirements and run the main experiment on the adult
dataset:

1. `git clone https://github.com/ADA-research/OverfittingCASH.git`
2. `cd OverfittingCASH`
3. `pip3 install -r requirements.txt`
4. `python3 experiments.py`

Installation of the requirements works on Linux or Mac. Due to the usage of the smac package, installation on Windows
can be done using WSL.

## Datasets

### Large Scale Holdout Experiments

The following table combines the classification and regression datasets used in the large-scale holdout experiments.
It includes the OpenML dataset ID, name, number of instances, number of features, and type of task (Classification or
Regression).

| **ID** | **Name**                               | **Instances** | **Features** | **Classes** | **Task Type**  |
|--------|----------------------------------------|---------------|--------------|-------------|----------------|
| 3      | kr-vs-kp                               | 3196          | 37           | 2           | Classification |
| 6      | letter                                 | 20000         | 17           | 26          | Classification |
| 11     | balance-scale                          | 625           | 5            | 3           | Classification |
| 15     | breast-w                               | 699           | 10           | 2           | Classification |
| 18     | mfeat-morphological                    | 2000          | 7            | 10          | Classification |
| 22     | mfeat-zernike                          | 2000          | 48           | 10          | Classification |
| 23     | cmc                                    | 1473          | 10           | 3           | Classification |
| 29     | credit-approval                        | 690           | 16           | 2           | Classification |
| 31     | credit-g                               | 1000          | 21           | 2           | Classification |
| 32     | pendigits                              | 10992         | 17           | 10          | Classification |
| 37     | diabetes                               | 768           | 9            | 2           | Classification |
| 38     | sick                                   | 3772          | 30           | 2           | Classification |
| 50     | tic-tac-toe                            | 958           | 10           | 2           | Classification |
| 54     | vehicle                                | 846           | 19           | 4           | Classification |
| 151    | electricity                            | 45312         | 9            | 2           | Classification |
| 182    | satimage                               | 6430          | 37           | 6           | Classification |
| 188    | eucalyptus                             | 736           | 20           | 5           | Classification |
| 307    | vowel                                  | 990           | 13           | 11          | Classification |
| 469    | analcatdata_dmft                       | 797           | 5            | 6           | Classification |
| 1049   | pc4                                    | 1458          | 38           | 2           | Classification |
| 1050   | pc3                                    | 1563          | 38           | 2           | Classification |
| 1053   | jm1                                    | 10885         | 22           | 2           | Classification |
| 1063   | kc2                                    | 522           | 22           | 2           | Classification |
| 1067   | kc1                                    | 2109          | 22           | 2           | Classification |
| 1068   | pc1                                    | 1109          | 22           | 2           | Classification |
| 1461   | bank-marketing                         | 45211         | 17           | 2           | Classification |
| 1462   | banknote-authentication                | 1372          | 5            | 2           | Classification |
| 1464   | blood-transfusion-service-center       | 748           | 5            | 2           | Classification |
| 1480   | ilpd                                   | 583           | 11           | 2           | Classification |
| 1489   | phoneme                                | 5404          | 6            | 2           | Classification |
| 1494   | qsar-biodeg                            | 1055          | 42           | 2           | Classification |
| 1497   | wall-robot-navigation                  | 5456          | 25           | 4           | Classification |
| 1510   | wdbc                                   | 569           | 31           | 2           | Classification |
| 1590   | adult                                  | 48842         | 15           | 2           | Classification |
| 23381  | dresses-sales                          | 500           | 13           | 2           | Classification |
| 23517  | numerai28.6                            | 96320         | 22           | 2           | Classification |
| 40499  | texture                                | 5500          | 41           | 11          | Classification |
| 40668  | connect-4                              | 67557         | 43           | 3           | Classification |
| 40701  | churn                                  | 5000          | 21           | 2           | Classification |
| 40975  | car                                    | 1728          | 7            | 4           | Classification |
| 40982  | steel-plates-fault                     | 1941          | 28           | 7           | Classification |
| 40983  | wilt                                   | 4839          | 6            | 2           | Classification |
| 40984  | segment                                | 2310          | 20           | 7           | Classification |
| 40994  | climate-model-simulation-crashes       | 540           | 21           | 2           | Classification |
| 41027  | jungle_chess_2pcs_raw_endgame_complete | 44819         | 7            | 3           | Classification |
| 4534   | PhishingWebsites                       | 11055         | 31           | 2           | Classification |
| 4538   | GesturePhaseSegmentationProcessed      | 9873          | 33           | 5           | Classification |
| 6332   | cylinder-bands                         | 540           | 40           | 2           | Classification |
| 201    | pol                                    | 15000         | 49           | -           | Regression     |
| 287    | wine_quality                           | 6497          | 12           | -           | Regression     |
| 507    | space_ga                               | 3107          | 7            | -           | Regression     |
| 531    | boston                                 | 506           | 14           | -           | Regression     |
| 541    | socmob                                 | 1156          | 6            | -           | Regression     |
| 546    | sensory                                | 576           | 12           | -           | Regression     |
| 550    | quake                                  | 2178          | 4            | -           | Regression     |
| 574    | house_16H                              | 22784         | 17           | -           | Regression     |
| 41021  | Moneyball                              | 1232          | 15           | -           | Regression     |
| 41540  | black_friday                           | 166821        | 10           | -           | Regression     |
| 42225  | diamonds                               | 53940         | 10           | -           | Regression     |
| 42688  | Brazilian_houses                       | 10692         | 13           | -           | Regression     |
| 42726  | abalone                                | 4177          | 9            | -           | Regression     |
| 42727  | colleges                               | 7063          | 48           | -           | Regression     |
| 42728  | Airlines_DepDelay_10M                  | 10000000      | 10           | -           | Regression     |
| 42729  | nyc-taxi-green-dec-2016                | 581835        | 19           | -           | Regression     |

### 10CV and Altering Validation Sizes

The following table lists the binary classification datasets used to investigate 10-fold cross-validation (10CV) and
varying validation sizes. These datasets were selected from OpenML-CC18 and include all binary datasets with more than
40,000 samples. These datasets are also included in the large-scale holdout experiments.

| **ID** | **Name**       | **Instances** | **Features** | **Classes** |
|--------|----------------|---------------|--------------|-------------|
| 151    | electricity    | 45312         | 9            | 2           |
| 1590   | adult          | 48842         | 15           | 2           |
| 1461   | bank-marketing | 45211         | 17           | 2           |
| 23517  | numerai28.6    | 96320         | 22           | 2           |

