import pandas as pd
import numpy as np
from scipy.optimize import minimize

file_path = "salmon-analysis-main/ground_truth/1.sim.genes.results"
ground_truth = pd.read_csv(file_path, delimiter="\t")[["TPM"]].to_numpy()


def MARD(weights, em_iter, vbem_iter):
    weighted_avg = weights[0] * em_iter + weights[1] * vbem_iter
    ARD_list = []

    for i in range(len(ground_truth)):

        if ground_truth[i] == weighted_avg[i] and ground_truth[i] == 0:
            ARD_list.append(0)
        else:
            ARD_list.append(
                abs(ground_truth[i] - weighted_avg[i])
                / (ground_truth[i] + weighted_avg[i])
            )

    return np.mean(ARD_list)


def minimization(em_iter, vbem_iter, initial_guess=[0.5, 0.5], MARD=MARD):
    result = minimize(
        MARD,
        x0=initial_guess,
        args=(em_iter, vbem_iter, ground_truth),
        bounds=[(0, 1), (0, 1)],
    )
    return result.x
