## OVERVIEW

This repository contains our replication of the paper:
**"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"**.

We implemented and reproduced two key experiments from the paper:

---

## 1. MNIST Experiment 

Goal: Estimate model uncertainty on rotated MNIST digits using Monte Carlo (MC) Dropout.

* Train.py — Trains a LeNet-style CNN with dropout.
* Test.py — Applies MC Dropout to visualize prediction uncertainty under rotation.
* Figure.png — Example output showing input/output uncertainty trends.

---

## 2. UCI Regression Experiment 

Goal: Evaluate MC Dropout for uncertainty estimation on regression tasks.

* train.py — Performs grid search over dropout and tau, trains the model.
* test.py — Evaluates models using RMSE, MC RMSE, and log-likelihood.
* UCI_Datasets/ — Preprocessed datasets and index files.

Each model is trained on 20 splits, with MC Dropout applied during evaluation.




