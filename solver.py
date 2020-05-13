#### Input is list of coordinates

import argparse
import numpy as np 
import torch
import time
import math 
import json 

from pygcn.models import GCN
from pygcn.multi_model import MultiModel
import dataprocess
import gurubi
from two_opt import two_opt

parser = argparse.ArgumentParser()
parser.add_argument('--dir_model', type=str,
                    help='Path to direction model checkpoint.')
parser.add_argument('--neighbor_model', type=str,
                    help='Path to neighborhood model checkpoint.')
parser.add_argument('--tsp_size', type=int, default=100,
                    help='What size tsp problems?')
parser.add_argument('--num_instances', type=int, default=1,
                    help='Number of instances to test on.')
parser.add_argument('--set_seed', type=int, default=0,
                    help='Set to positive integer if seed desired.')
parser.add_argument('--max_expansion', type=int, default=2)
parser.add_argument('--max_solutions', type=int, default=1024)
parser.add_argument('--solver_threshold', type=int, default=40,
                    help='Size at which external solver takes over.')
parser.add_argument('--no_cuda', help='Indicates not to use cuda.')



class TSPProblem:
    def __init__(self, coords, max_input_size=50000, solution_cap=1024, branching_factor=2):
        self.coords = coords
        self.size = coords.size(0)
        self.distances = self.distance_adj(coords)
        self.inverse_distances = self.invert_distances(self.distances)
        self.subproblems = []
        self.subsolutions = []
        self.subproblem_stack = []
        self.to_model_buffer = []
        self.max_input_size = max_input_size
        self.solution_cap = solution_cap
        self.num_leaves = 0
        self.branching_factor = branching_factor

    def print_progress(self):
        current_progress = self.subsolutions[-1].tour.size(0)
        progress_percent = current_progress / self.size
        progress_out_of_20 = math.floor(progress_percent * 20)
        pbar = '#'*progress_out_of_20 + ' '*(20 - progress_out_of_20)
        print(f'\r[{pbar}]', end='')

    @staticmethod
    def distance_adj(graph):
        size = graph.size(0)
        g1 = graph.unsqueeze(0).float()
        g2 = graph.unsqueeze(1).float()
        distances = (g1 - g2).norm(p=2, dim=2)
        return distances
    
    @staticmethod
    def invert_distances(d):
        inv_d = 1 / d
        inv_d[torch.isinf(inv_d)] = 0.0
        return inv_d

    @staticmethod
    def combine_subinstances(multi_feats, multi_sparse_adj):
        # Combines problems into one that can be fed into model in parallel
        feats = torch.cat(multi_feats, dim=0)
        partitions = []
        current_start = 0
        for indices, values, size in multi_sparse_adj:
            partitions.append((current_start, current_start + size))
            current_start += size
        full_adj = TSPProblem.combine_sparse_parts(multi_sparse_adj)
        return feats, full_adj, partitions

    @staticmethod
    def combine_sparse_parts(multi_sparse_parts):
        """combines adjacency matrices into one graph of multiple components"""
        cum_size = 0
        cum_indices = None
        cum_values = None
        for indices, values, size in multi_sparse_parts:
            indices = indices + cum_size
            cum_indices = indices if cum_indices is None else torch.cat([cum_indices, indices], dim=1)
            cum_values = values if cum_values is None else torch.cat([cum_values, values])
            cum_size += size

        t = torch.sparse.FloatTensor(cum_indices, cum_values, (cum_size, cum_size))
        return t

    def fill_to_model_buffer(self):
        self.to_model_buffer = []
        input_size = 0
        while (len(self.subproblem_stack) > 0 and 
               input_size + self.subproblem_stack[-1].subproblem_size <= self.max_input_size) or \
               len(self.to_model_buffer) == 0: 
            input_size += self.subproblem_stack[-1].subproblem_size
            self.to_model_buffer.append(self.subproblem_stack.pop(-1))

    def combine_subinstances_on_buffer(self):
        # Combines problems into one that can be fed into model in parallel
        multi_feats = []
        multi_sparse_adj = []
        for problem in self.to_model_buffer:
            problem.prepare()
            multi_feats.append(problem.feats)
            multi_sparse_adj.append(problem.sparse_adj)
        feats = torch.cat(multi_feats, dim=0)
        partitions = []
        current_start = 0
        for _, _, size in multi_sparse_adj:
            partitions.append((current_start, current_start + size))
            current_start += size
        full_adj = TSPProblem.combine_sparse_parts(multi_sparse_adj)
        return feats, full_adj, partitions

    def sample_directions(self, dir_output):
        dir_output = dir_output.mean(1)
        branching = min(self.branching_factor, dir_output.size(0), self.solution_cap - self.num_leaves + 1)
        self.num_leaves += branching - 1
        sample = np.random.choice(np.arange(dir_output.size(0)), size=branching, replace=False)
        sample = torch.tensor(sample).long()
        sampled_dirs = dir_output[sample]
        return sampled_dirs

    def split_outputs(self, output, partitions):
        outputs = []
        for start, end in partitions:
            outputs.append(output[:,start:end])
        return outputs

    def get_ngb_input(self, outputs):
        feats = []
        adjs = []
        parents = []
        directions = []
        
        for output, sp in zip(outputs, self.to_model_buffer):
            dirs = self.sample_directions(output)
            for d in dirs:
                dir_feats = sp.prepare_direction(d)
                feats.append(dir_feats)
                adjs.append(sp.sparse_adj)
                parents.append(sp)
                directions.append(d)

        full_feats, full_adj, partitions = self.combine_subinstances(feats, adjs)
        return full_feats, full_adj, partitions, parents, directions


    def create_solutions_and_add_new_problems(self, ngb_outputs, directions, parent_probs):
        for output, prob, direction in zip(ngb_outputs, parent_probs, directions):
            output = output[0].flatten()
            output[prob.unmasked_to_masked[prob.end]] = -1000000
            mask = (torch.sigmoid(output) > 0.5).long()
            if prob.unmasked_to_masked[prob.current] != -1:
                mask[prob.unmasked_to_masked[prob.current]] = 0  # don't include current in neighborhood (will be added by subsolution)
            if mask.sum() > 40:
                mask_indices = output.topk(k=40).indices
            elif mask.sum() < 10:
                mask_indices = output.topk(k=10).indices
            else:
                mask_indices = mask.nonzero().flatten()
            orig_prob_mask = prob.masked_to_unmasked[mask_indices].flatten()
            sub_solution = SubSolution(self, prob, orig_prob_mask, prob.current, direction)
            solved = sub_solution.solve()
            self.subsolutions.append(sub_solution)
            self.print_progress()
            if not solved and mask_indices.size(0) > 0:
                new_mask, new_cur, end = sub_solution.get_new_problem_params()
                if new_mask.sum() == 0:
                    breakpoint()
                new_problem = SubProblem(self, sub_solution, new_mask, new_cur, end)
                self.subproblems.append(new_problem)
                self.subproblem_stack.append(new_problem)            

    def solve(self):
        start = time.time()
        starter_solution = SubSolution(self, None, None, None, None)
        starter_solution.tour = torch.tensor([]).long()
        starter_problem = SubProblem(self, starter_solution, torch.ones(self.size), 0, 0)
        self.num_leaves = 1
        self.subproblems.append(starter_problem)
        self.subproblem_stack.append(starter_problem)
        while(len(self.subproblem_stack) > 0):
            self.fill_to_model_buffer()
            feats, full_adj, partitions = self.combine_subinstances_on_buffer()
            dir_output = dir_model(feats, full_adj)
            outputs = self.split_outputs(dir_output, partitions)
            feats, full_adj, partitions, parents, directions = self.get_ngb_input(outputs)
            ngb_output = ngb_model(feats, full_adj)
            outputs = self.split_outputs(ngb_output, partitions)
            self.create_solutions_and_add_new_problems(outputs, directions, parents)

        solutions = [s for s in self.subsolutions if s.solved]
        best_solution = torch.tensor([s.tour_len() for s in solutions]).argmin().numpy()
        best_solution_length = solutions[best_solution].tour_len().numpy().item()
        GS_end = time.time()
        solutions[best_solution].two_opt()
        local_opt_best_solution_length = solutions[best_solution].tour_len().numpy().item()
        GS_2OPT_end = time.time()
        return best_solution_length, local_opt_best_solution_length, GS_end-start, GS_2OPT_end-start



class SubProblem:
    def __init__(self, tsp_problem, parent_solution, mask, current, end):
        self.tsp_problem = tsp_problem
        self.parent_solution = parent_solution
        self.mask = mask
        self.mask_indices = self.mask.nonzero().flatten()
        self.subproblem_size = self.mask_indices.size(0)
        self.current = current
        self.end = end 
        self.feats = None 
        self.sparse_adj = None  # stored in sparse parts format (indices, values, size)
        self.scale_factor = None
        self.translation_factor = None
        self.unmasked_to_masked = None 
        self.masked_to_unmasked = None
        self.dirfeats = None

    def prepare(self):
        instance = self.tsp_problem.coords
        size = instance.size(0)

        mask_indices = self.mask_indices
        subsize = mask_indices.size(0)
        masked_to_unmasked = mask_indices  # original idxs of nodes prior to masking
        unmasked_to_masked = -torch.ones(size).long()
        unmasked_to_masked[mask_indices] = torch.arange(subsize)

        subinstance = instance[mask_indices]
        self.translation_factor = -instance[self.current]
        subinstance = subinstance + self.translation_factor
        self.scale_factor = 1 / subinstance.norm(p=2, dim=1).max()
        subinstance = subinstance * self.scale_factor

        weighted = self.tsp_problem.inverse_distances[mask_indices,:][:,mask_indices]
        normalized_adj = self.normalize_adj(weighted)

        sparse_adj = dataprocess.TSPDirectionDataloader.to_sparse_parts(normalized_adj)
        cur_one_hot = torch.eye(subsize)[unmasked_to_masked[self.current]].unsqueeze(1)
        final_dest_one_hot = torch.eye(subsize)[unmasked_to_masked[self.end]].unsqueeze(1)

        final_dest_coord = subinstance[unmasked_to_masked[self.end]]
        final_dest_coord = final_dest_coord.repeat(subsize).reshape(-1,2)

        feats = torch.cat([cur_one_hot, final_dest_one_hot, final_dest_coord, subinstance], dim=1)

        self.feats = feats 
        self.sparse_adj = sparse_adj 
        self.unmasked_to_masked = unmasked_to_masked 
        self.masked_to_unmasked = masked_to_unmasked

    @staticmethod
    def normalize_adj(adj):
        # Renormalize adj and add self-loops
        size = adj.size(0)
        adj = adj / adj.sum(0).expand_as(adj).T
        adj = adj / 2
        if True:  # self-loops?
            idx_range = torch.arange(size)
            adj[idx_range, idx_range] = torch.ones(size) / 2
        return adj 

    def prepare_direction(self, direction):
        self.dir_feats = torch.cat([self.feats, direction.repeat(self.subproblem_size).reshape(-1,2)], dim=1)
        return self.dir_feats


class SubSolution:
    def __init__(self, tsp_problem, parent_prob, neighborhood_mask, start, direction):
        self.tsp_problem = tsp_problem
        self.neighborhood_mask = neighborhood_mask  # indices in original problem from which to solve subtour
        self.start = start 
        self.direction = direction
        self.parent_prob = parent_prob
        self.tour = None
        self.solved = False

    def tour_len(self):
        total_dist = 0
        distances = self.tsp_problem.distances
        for node, next_node in zip(self.tour[:-1], self.tour[1:]):
            total_dist += distances[node, next_node]
        return total_dist


    def solve(self):
        coords = self.tsp_problem.coords[self.neighborhood_mask]
        coords = torch.cat([self.tsp_problem.coords[self.parent_prob.current].unsqueeze(0), coords], dim=0)
        coords = (coords + self.parent_prob.translation_factor) * self.parent_prob.scale_factor
        tour, _ = gurubi.solve_gs_subtour(coords, self.direction, 0)
        tour = torch.tensor(tour)[1:] - 1  # remove current node
        self.tour = self.neighborhood_mask[tour]
        self.tour = torch.cat([self.parent_prob.parent_solution.tour, self.tour])

        if self.tsp_problem.size - self.tour.size(0) <= args.solver_threshold:
            self.solve_to_completion()
            return True
        else:
            return False

    def solve_to_completion(self):
        new_mask, new_current, end = self.get_new_problem_params()
        new_mask[end] = 0
        new_neighborhood_mask = new_mask.nonzero().flatten()
        if new_neighborhood_mask.size(0) > 0:
            coords = self.tsp_problem.coords[new_neighborhood_mask]
            coords = torch.cat([self.tsp_problem.coords[new_current].unsqueeze(0), coords], dim=0)
            # Don't need to translate as final node is just coordinates (not direction in translated space)
            end_coord = self.tsp_problem.coords[end]
            tour, _ = gurubi.solve_gs_subtour(coords, end_coord, 0)
            tour = torch.tensor(tour)[1:] - 1  # remove current node
            self.tour = torch.cat([self.tour, new_neighborhood_mask[tour], torch.tensor([end])], dim=0)
        else:
            self.tour = torch.cat([self.tour, torch.tensor([end])])
        self.solved = True

    def get_new_problem_params(self):
        new_current = self.tour[-1]
        new_mask = self.parent_prob.mask.clone()
        new_mask[self.neighborhood_mask] = 0
        end = self.parent_prob.end
        return new_mask, new_current, end

    def two_opt(self):
        improved_tour = two_opt(list(self.tour.numpy()), self.tsp_problem.distances)
        self.tour = torch.tensor(improved_tour)
    







if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    DEVICE = 'cuda' if args.cuda else 'cpu'

    dir_cpk = torch.load(args.dir_model,  map_location=torch.device(DEVICE))
    ngb_cpk = torch.load(args.neighbor_model, map_location=torch.device(DEVICE))

    dir_model = MultiModel(GCN, dir_cpk['model_params']['n_models'], 
                        nfeat=dir_cpk['model_params']['nfeat'],
                        nhid=dir_cpk['model_params']['nhid'],
                        nlayers=dir_cpk['model_params']['nlayers'],
                        nclass=dir_cpk['model_params']['nclass'], 
                        dropout=dir_cpk['model_params']['dropout'])
                        
    ngb_model = MultiModel(GCN, ngb_cpk['model_params']['n_models'], 
                        nfeat=ngb_cpk['model_params']['nfeat'],
                        nhid=ngb_cpk['model_params']['nhid'],
                        nlayers=ngb_cpk['model_params']['nlayers'],
                        nclass=ngb_cpk['model_params']['nclass'], 
                        dropout=ngb_cpk['model_params']['dropout'])

    torch.no_grad()
    dir_model.eval()
    ngb_model.eval()

    if args.set_seed > 0:
        np.random.seed(args.set_seed)
    tsp_problems = np.random.uniform(size=(args.num_instances, args.tsp_size, 2))



    def _cuda(o):
        return o.cuda() if args.use_cuda else o

    gs_results = []
    gs_2opt_results = []
    gs_times = []
    gs_2opt_times = []

    for i, p in enumerate(tsp_problems):
        problem = TSPProblem(torch.tensor(p).float(), max_input_size=50000, solution_cap=args.max_solutions, 
                             branching_factor=args.max_expansion)
        gs, gs_2opt, gs_time, gs_2opt_time = problem.solve()
        gs_results.append(gs)
        gs_2opt_results.append(gs_2opt)
        gs_times.append(gs_time)
        gs_2opt_times.append(gs_2opt_time)
        with open('temp_results.json', 'w') as f:
            json.dump({
                'gs_results': gs_results,
                'gs_2opt_results': gs_2opt_results,
                'gs_times': gs_times,
                'gs_2opt_times': gs_2opt_times
            }, f)
        print(f'Finished instance {i+1} / {len(tsp_problems)}')
    gs_results = torch.tensor(gs_results)
    gs_2opt_results = torch.tensor(gs_2opt_results)
    gs_times = torch.tensor(gs_times)
    gs_2opt_times = torch.tensor(gs_2opt_times)

    print('\n')
    print(f'GS: {gs_results.mean()}')
    print(f'GS_2OPT: {gs_2opt_results.mean()}')
    print(f'Time GS: {gs_times.mean()}')
    print(f'Time GS_2Opt: {gs_2opt_times.mean()}')
                            




