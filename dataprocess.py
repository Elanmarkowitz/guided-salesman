import os 
import tsplib95 as tsp 
import numpy as np 

import torch 
from torch.utils.data import Dataset, DataLoader

DATADIR = 'ALL_tsp'

class TSPDataset(Dataset):

    def __init__(self, datadir):
        self.datadir = datadir 
        self.problem_names, self.graphs, self.tours, self.cur_nodes = self._create_training_data()

    def __len__(self):
        return len(self.problem_names)

    def __getitem__(self, idx):
        return self.graphs[idx], self.cur_nodes[idx], self.tours[idx]

    def _create_training_data(self):
        datadir = self.datadir
        problems = []
        graphs = []  
        tours = []
        cur_nodes = []
        for filename in os.listdir(datadir):
            name, ext = os.path.splitext(filename)
            tourpath = f'{name}.opt.tour'

            if ext == '.tsp' and tourpath in os.listdir(datadir):
                problem = tsp.load(os.path.join(datadir, filename))
                
                if problem.type != 'TSP' or problem.edge_weight_type != 'EUC_2D' or problem.dimension < 20:
                    # Only using 2D symmetric graphs for TSP
                    # also no point in training if graph is so small (could always generate data elsewhere)
                    continue
                print(problem.name, problem.dimension)

                graph = []
                for i in range(1, problem.dimension + 1):
                    graph.append(problem.node_coords[i])
                
                tour = tsp.load(os.path.join(datadir, tourpath))
                if np.array(tour.tours).shape[0] > 1: 
                    print(tour.name)
                    breakpoint()

                opttour = np.array(tour.tours[0])
                for i in range(len(opttour)):
                    curtour = np.append(opttour[i:], opttour[:i])
                    curtour = curtour - 1  # adjust to 0 indexed numbering to match graph
                    curnode = curtour[0]
                    graphs.append(graph)
                    tours.append(curtour)
                    cur_nodes.append(curnode)
                    problems.append(problem.name)

                opttour = opttour[::-1]  # repeat for reversed tour
                for i in range(len(opttour)):
                    curtour = np.append(opttour[i:], opttour[:i])
                    curtour = curtour - 1  # adjust to 0 indexed numbering to match graph
                    curnode = curtour[0]
                    graphs.append(graph)
                    tours.append(curtour)
                    cur_nodes.append(curnode)
                    problems.append(problem.name)

        return np.array(problems), np.array(graphs), np.array(tours), np.array(cur_nodes)


class TSPDirectionDataloader(DataLoader):

    def __init__(self, *args, L_steps=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.L_steps = L_steps
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        adjacencies = []
        features = []
        ys = []
        for graph, cur, tour in batch:
            graph = torch.tensor(graph)
            tour = torch.tensor(tour)

            cur_one_hot = torch.eye(graph.size(0))[cur]
            x_cur = graph[cur]
            graph = graph - x_cur  # center graph on current node
            
            y = tour[self.L_steps]
            y_one_hot = torch.eye(graph.size(0))[y]

            tour_len = self.sample_tour_len(tour, self.L_steps)
            tour = tour[:tour_len].long()

            final_dest = tour[-1]
            final_dest_one_hot = torch.eye(graph.size(0))[final_dest]
            
            graph = graph[tour].float()
            graph = graph / graph.norm(p=2, dim=1).max()
            y_one_hot = y_one_hot[tour].float()
            cur_one_hot = cur_one_hot[tour].unsqueeze(-1).float()
            final_dest_one_hot = final_dest_one_hot[tour].unsqueeze(-1).float()
            feat = torch.cat([cur_one_hot, final_dest_one_hot, graph], dim=1)
            A = self.weighted_adj_from_pos(graph)
            adjacencies.append(A)
            features.append(feat)
            ys.append(y_one_hot)
        full_adjacency = self.combine_adjacencies(adjacencies)
        full_feats = torch.cat(features)
        ys = torch.cat(ys)
        return full_feats, full_adjacency, ys

    @staticmethod
    def to_sparse_parts(x):
        """ converts dense tensor x to sparse format 
        Assumes square matrix (adj)
        """
        indices = torch.nonzero(x).T
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return indices, values, x.size(0)

    @staticmethod
    def combine_adjacencies(adjacencies):
        """combines adjacency matrices into one graph of multiple components"""
        cum_size = 0
        cum_indices = None
        cum_values = None
        for adj in adjacencies:
            indices, values, size = TSPDirectionDataloader.to_sparse_parts(adj)
            indices = indices + cum_size
            cum_indices = indices if cum_indices is None else torch.cat([cum_indices, indices], dim=1)
            cum_values = values if cum_values is None else torch.cat([cum_values, values])
            cum_size += size

        return torch.sparse.FloatTensor(cum_indices, cum_values, (cum_size, cum_size))

    @staticmethod
    def sample_tour_len(tour, min_length):
        min_len = max(tour.size(0) / 2, min_length)
        max_len = tour.size(0)
        tour_len = np.random.triangular(min_len, max_len, max_len)
        tour_len = int(np.ceil(tour_len))
        return tour_len

    @staticmethod
    def weighted_adj_from_pos(graph):
        """
        Constucts a weighted, normalized adjacency matrix from the node coordinates
        graph: |G| x 2 tensor
        returns: |G| x |G| tensor
        """
        size = graph.size(0)
        g1 = graph.unsqueeze(0)
        g2 = graph.unsqueeze(1)
        distances = (g1 - g2).norm(p=2, dim=2)
        weighted = 1 / distances
        weighted[torch.arange(size), torch.arange(size)] = torch.zeros(size)
        # TODO: May want to experiment with non row-based normalization
        normalized = weighted / weighted.sum(0).expand_as(weighted).T
        normalized = normalized / 2
        if True:  # self-loops?
            normalized[torch.arange(size), torch.arange(size)] = torch.ones(size) / 2
        return normalized


class TSPNeighborhoodDataloader(TSPDirectionDataloader):

    def __init__(self, *args, L_steps=20, **kwargs):
        super().__init__(*args, L_steps=20, **kwargs)
        self.L_steps = L_steps
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        adjacencies = []
        features = []
        neighborhoods = []
        for graph, cur, tour in batch:
            graph = torch.tensor(graph)
            tour = torch.tensor(tour)

            cur_one_hot = torch.eye(graph.size(0))[cur]
            x_cur = graph[cur]
            graph = graph - x_cur  # center graph on current node
            
            neighborhood = tour[1:self.L_steps + 1].long()
            neighborhood_label = torch.zeros(size = (graph.size(0),))
            neighborhood_label[neighborhood] = 1.0

            tour_len = self.sample_tour_len(tour, self.L_steps)
            tour = tour[:tour_len].long()

            final_dest = tour[-1]
            final_dest_one_hot = torch.eye(graph.size(0))[final_dest]
            
            graph = graph[tour].float()
            graph = graph / graph.norm(p=2, dim=1).max()
            neighborhood_label = neighborhood_label[tour].float()
            cur_one_hot = cur_one_hot[tour].unsqueeze(-1).float()
            final_dest_one_hot = final_dest_one_hot[tour].unsqueeze(-1).float()
            feat = torch.cat([cur_one_hot, final_dest_one_hot, graph], dim=1)
            A = self.weighted_adj_from_pos(graph)
            adjacencies.append(A)
            features.append(feat)
            neighborhoods.append(neighborhood_label)
        full_adjacency = self.combine_adjacencies(adjacencies)
        full_feats = torch.cat(features)
        neighborhoods = torch.cat(neighborhoods)
        return full_feats, full_adjacency, neighborhoods
    

        

        
    
dataset = TSPDataset(DATADIR)
dir_dataloader = TSPDirectionDataloader(dataset, batch_size=3)

neighborhood_dataloader = TSPNeighborhoodDataloader(dataset, batch_size=3)

breakpoint()

for feats, full_A, y in dir_dataloader:
    print(feats.shape)
    break

for feats, full_A, neighborhoods in neighborhood_dataloader:
    print(neighborhoods.shape)
    break




