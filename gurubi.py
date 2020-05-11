#!/usr/bin/env python3.7

# Copyright 2020, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch 
import time


# Given a tuplelist of edges, find the shortest subtour

def subtour(edges, n):
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


def solve_gs_subtour(coords, direction, start):

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                    if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected, num_nodes)
            if len(tour) < num_nodes:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(gp.quicksum(model._vars[i, j]
                                        for i, j in combinations(tour, 2))
                            <= len(tour)-1)


    # (torch.Tensor, torch.Tensor, int) -> ...
    direction = direction.unsqueeze(0)
    size = coords.size(0)

    # add coordinate for direction
    coords = torch.cat([coords, direction])

    g1 = coords.unsqueeze(0).float()
    g2 = coords.unsqueeze(1).float()
    distances = (g1 - g2).norm(p=2, dim=2)
    
    dist = {(i, j): float(distances[i, j].detach().numpy())
            for i in range(size + 1) for j in range(i)}
    
    # add dummy coordinate to ensure source and sink
    dummy = size + 1
    dist[size, dummy] = 0 
    dist[dummy, start] = 0
    num_nodes = dummy + 1
    
    m = gp.Model()

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction
    

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(size + 1))

    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = subtour(selected, size + 2)
    assert len(tour) == num_nodes 

    tour = tour[:size]  # subtour always starts with 0. 

    return tour, m.objVal


if __name__ == "__main__":
        
    # Parse argument

    if len(sys.argv) < 2:
        print('Usage: tsp.py npoints')
        sys.exit(1)
    n = int(sys.argv[1])

    # Create n random points

    #random.seed(1)
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]

    points = torch.tensor(points).long()
    direction = torch.tensor([100,100]).long()
    points[0] = torch.tensor([0,0]).long()
    start = 0
    start_time = time.time()
    tour, objVal = solve_gs_subtour(points, direction, start)
    total_time = time.time() - start_time

    print(f'tour: {tour}')
    print(f'distance: {objVal}')
    print(f'time: {total_time}')

