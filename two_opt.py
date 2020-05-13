import numpy as np

def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]


def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best

import solver 
import torch
import time
from tqdm.auto import tqdm
if __name__ == '__main__':
    np.random.seed(1234321)
    nodes = 1000
    tsp_problems = np.random.uniform(size=(100, nodes, 2))
    results = []
    times = []
    pbar = tqdm(total=100)
    pbar.update(0)
    for prob in tsp_problems:
        pbar.update(1)
        p = solver.TSPProblem(torch.tensor(prob))
        init_route = list(range(nodes))
        cost_mat = p.distances.numpy()
        cost_mat = list(cost_mat)
        start = time.time()
        best_route = two_opt(init_route, cost_mat)
        end = time.time()
        cycle = best_route + [best_route[0]]
        total_len = 0
        for i, j in zip(cycle[:-1], cycle[1:]):
            total_len += cost_mat[i][j]
        results.append(total_len)
        times.append(end - start)
    pbar.close()
    mean_time = np.array(times).mean().item()
    mean_len = np.array(results).mean().item()
    print(f'time: {mean_time}')
    print(f'length: {mean_len}')