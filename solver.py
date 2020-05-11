#### Input is list of coordinates

import argparse
import numpy as np 
import torch

from pygcn.models import GCN
from pygcn.multi_model import MultiModel
import dataprocess
import gurubi

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

dir_cpk = torch.load(args.dir_model)
ngb_cpk = torch.load(args.neighbor_model)

dir_model = MultiModel(GCN, dir_cpk.params.n_models, 
                       nfeat=dir_cpk.params.nfeat,
                       nhid=dir_cpk.params.nhid,
                       nlayers=dir_cpk.params.nlayers,
                       nclass=dir_cpk.params.nclass, 
                       dropout=dir_cpk.params.dropout)
                       
ngb_model = MultiModel(GCN, ngb_cpk.params.n_models, 
                       nfeat=ngb_cpk.params.nfeat,
                       nhid=ngb_cpk.params.nhid,
                       nlayers=ngb_cpk.params.nlayers,
                       nclass=ngb_cpk.params.nclass, 
                       dropout=ngb_cpk.params.dropout)


if args.set_seed > 0:
    np.random.seed(args.set_seed)
tsp_problems = np.random.uniform(size=(args.num_instances, args.tsp_size, 2))



def _cuda(o):
    return o.cuda() if args.use_cuda else o



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
        for adj in multi_sparse_adj:
            size = adj.size(0)
            partitions.append((current_start, current_start + size))
            current_start += size
        full_adj = dataprocess.TSPDirectionDataloader.combine_adjacencies(multi_sparse_adj)
        return feats, full_adj, partitions

    def fill_to_model_buffer(self):
        self.to_model_buffer = []
        input_size = 0
        while (len(self.subproblem_stack) > 0 and 
               input_size + self.subproblem_stack[-1].subproblem_size <= self.max_input_size):
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
        for adj in multi_sparse_adj:
            size = adj.size(0)
            partitions.append((current_start, current_start + size))
            current_start += size
        full_adj = dataprocess.TSPDirectionDataloader.combine_adjacencies(multi_sparse_adj)
        return feats, full_adj, partitions

    def sample_directions(self, dir_output):
        branching = min(self.branching_factor, dir_output.size(0), self.solution_cap - self.num_leaves + 1)
        sample = np.random.choice(np.arange(dir_output.size(0)), size=branching, replace=False)
        sample = torch.tensor(sample)

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
            mask = (output > 0.5).long()
            mask_indices = mask.nonzero()
            orig_prob_mask = prob.masked_to_unmasked[mask_indices]
            sub_solution = SubSolution(self, prob, orig_prob_mask, prob.current, direction)
            solved = sub_solution.solve()
            self.subsolutions.append(sub_solution)
            if not solved:
                new_mask, new_cur, end = sub_solution.get_new_problem_params()
                new_problem = SubProblem(self, sub_solution, new_mask, new_cur, end)
                self.subproblems.append(new_problem)
                self.subproblem_stack.append(new_problem)

            

    def solve(self):
        starter_problem = SubProblem(self, None, torch.ones(self.size), 0, 0)
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




class SubProblem:
    def __init__(self, tsp_problem, parent_solution, mask, current, end):
        self.tsp_problem = tsp_problem
        self.parent_solution = parent_solution
        self.mask = mask
        self.mask_indices = torch.nonzero(self.mask).flatten()
        self.subproblem_size = self.mask_indices.size(0)
        self.current = current
        self.end = end 
        self.feats = None 
        self.sparse_adj = None
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
        masked_to_unmasked = torch.arange(size)[mask_indices]  # original ids of nodes prior to masking
        unmasked_to_masked = mask_indices[:]
        unmasked_to_masked[mask_indices] = torch.arange(subsize)

        subinstance = instance[mask_indices]
        self.translation_factor = -instance[self.current]
        subinstance = subinstance + self.translation_factor
        self.scale_factor = 1 / subinstance.norm(p=2, dim=1).max()
        subinstance = subinstance * self.scale_factor

        weighted = self.tsp_problem.inverse_distances[mask_indices,:][:,mask_indices]
        normalized_adj = self.normalize_adj(weighted)

        sparse_adj = dataprocess.TSPDirectionDataloader.to_sparse_parts(normalized_adj)
        cur_one_hot = torch.eye(subsize)[unmasked_to_masked[self.current]]
        final_dest_one_hot = torch.eye(subsize)[unmasked_to_masked[self.end]]

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

    def solve(self):
        coords = self.tsp_problem.coords[self.neighborhood_mask]
        coords = torch.cat([self.tsp_problem.coords[self.parent_prob.current], coords], dim=0)
        coords = (coords + self.parent_prob.translation_factor) * self.parent_prob.scale_factor
        tour, _ = gurubi.solve_gs_subtour(coords, self.direction, 0)
        self.tour = self.neighborhood_mask[tour[1:]]
        return false  # TODO: return whether solved or not

    def get_new_problem_params(self):
        new_current = self.tour[-1]
        new_mask = self.parent_prob.mask
        new_mask[self.neighborhood_mask] = 0
        end = self.parent_prob.end
        return new_mask, new_current, end
    












