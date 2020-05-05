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

    def __init__(self, *args, L_steps=20, use_cuda=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.L_steps = L_steps
        self.use_cuda = use_cuda
        self.collate_fn = self._collate_fn

    def _cuda(self, t):
        return t.cuda() if self.use_cuda and torch.cuda.is_available() else t

    def _collate_fn(self, batch):
        adjacencies = []
        features = []
        ys = []
        graph_sizes = []
        for graph, cur, tour in batch:
            graph = self._cuda(torch.tensor(graph))
            tour = self._cuda(torch.tensor(tour))

            cur_one_hot = self._cuda(torch.eye(graph.size(0))[cur])
            x_cur = graph[cur]
            graph = graph - x_cur  # center graph on current node

            tour_len = self.sample_tour_len(tour, self.L_steps)
            tour = tour[:tour_len].long()

            final_dest = tour[-1]
            final_dest_one_hot = self._cuda(torch.eye(graph.size(0))[final_dest])
            
            graph = graph[tour].float()
            graph = self.normalize_graph_scale(graph)
            graph = self.rotate_graph(graph)

            # y_one_hot = y_one_hot[tour].float()
            cur_one_hot = cur_one_hot[tour].unsqueeze(-1).float()
            final_dest_one_hot = final_dest_one_hot[tour].unsqueeze(-1).float()
            final_dest_coord = graph[-1]
            final_dest_coord = final_dest_coord.repeat(graph.size(0)).reshape(-1,2)
            feat = torch.cat([cur_one_hot, final_dest_one_hot, final_dest_coord, graph], dim=1)
            A = self.weighted_adj_from_pos(graph)
            adjacencies.append(A)
            features.append(feat)
            direction = graph[self.L_steps]
            ys.append(direction) 
            graph_start = 0 if graph_sizes == [] else graph_sizes[-1][1]
            graph_stop = graph_start + graph.size(0)
            graph_sizes.append((graph_start, graph_stop))
        full_adjacency = self.combine_adjacencies(adjacencies)
        full_feats = torch.cat(features)

        return full_feats, full_adjacency, ys, graph_sizes

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
    def normalize_graph_scale(graph):
        return graph / graph.norm(p=2, dim=1).max()

    def rotate_graph(self, graph):
        radians = torch.rand(())*2*np.pi
        c, s = torch.cos(radians), torch.sin(radians)
        rotation_matrix = self._cuda(torch.tensor([[c, -s], [s, c]]))
        return torch.mm(graph, rotation_matrix)

    def weighted_adj_from_pos(self, graph):
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
        weighted[torch.isinf(weighted)] = 0.0  # replace Infs (duplicate points and self-distances)
        # TODO: May want to experiment with non row-based normalization
        normalized = weighted / weighted.sum(0).expand_as(weighted).T
        normalized = normalized / 2
        if True:  # self-loops?
            idx_range = self._cuda(torch.arange(size))
            normalized[idx_range, idx_range] = self._cuda(torch.ones(size)) / 2
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
        graph_sizes = []
        for graph, cur, tour in batch:
            graph = self._cuda(torch.tensor(graph))
            tour = self._cuda(torch.tensor(tour))

            cur_one_hot = self._cuda(torch.eye(graph.size(0))[cur])
            x_cur = graph[cur]
            graph = graph - x_cur  # center graph on current node

            dir_node, num_steps = self.sample_direction(tour)

            neighborhood = tour[1:self.L_steps + 1].long()
            neighborhood_label = self._cuda(torch.zeros(size = (graph.size(0),)))
            neighborhood_label[neighborhood] = 1.0

            tour_len = self.sample_tour_len(tour, self.L_steps)
            tour_len = max([tour_len, num_steps + 1])

            tour = tour[:tour_len].long()

            final_dest = tour[-1]
            final_dest_one_hot = self._cuda(torch.eye(graph.size(0))[final_dest])

            graph = graph[tour].float()
            graph = self.normalize_graph_scale(graph)
            graph = self.rotate_graph(graph)
            
            neighborhood_label = neighborhood_label[tour].float()
            cur_one_hot = cur_one_hot[tour].unsqueeze(-1).float()
            direction = graph[num_steps]
            direction = direction.float().repeat(graph.size(0)).reshape(-1,2)
            final_dest_one_hot = final_dest_one_hot[tour].unsqueeze(-1).float()
            final_dest_coord = graph[-1]
            final_dest_coord = final_dest_coord.repeat(graph.size(0)).reshape(-1,2)
            feat = torch.cat([cur_one_hot, final_dest_one_hot, final_dest_coord, graph, direction], dim=1)
            A = self.weighted_adj_from_pos(graph)
            adjacencies.append(A)
            features.append(feat)
            neighborhoods.append(neighborhood_label)
            graph_start = 0 if graph_sizes == [] else graph_sizes[-1][1]
            graph_stop = graph_start + graph.size(0)
            graph_sizes.append((graph_start, graph_stop))
        full_adjacency = self.combine_adjacencies(adjacencies)
        full_feats = torch.cat(features)

        return full_feats, full_adjacency, neighborhoods, graph_sizes

    def sample_direction(self, tour, lower_factor_bound=0.5, upper_factor_bound=2):
        step_count = np.random.triangular(self.L_steps * lower_factor_bound, 
                                          self.L_steps, 
                                          self.L_steps * upper_factor_bound)
        step_count = min([tour.size(0)-1, int(np.round(step_count))])
        node =  tour[step_count]
        return node, step_count
    

        

        
if __name__ == '__main__':
    dataset = TSPDataset(DATADIR)
    dir_dataloader = TSPDirectionDataloader(dataset, use_cuda=True, batch_size=8, shuffle=True)

    neighborhood_dataloader = TSPNeighborhoodDataloader(dataset, use_cuda=True, batch_size=8, shuffle=True)

    from tqdm.auto import tqdm 
    pbar = tqdm(total=len(dir_dataloader))
    for batch in dir_dataloader:
        pbar.update(1)
        continue
    pbar.close()
    from tqdm.auto import tqdm 
    pbar = tqdm(total=len(neighborhood_dataloader))
    for batch in neighborhood_dataloader:
        pbar.update(1)
        continue
    pbar.close()




