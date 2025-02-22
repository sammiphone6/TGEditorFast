from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
import warnings
# import config
import os
import torch
import torch.nn.functional as F
from random import SystemRandom
from torch.optim import SparseAdam
from torch.utils.data import DataLoader, RandomSampler
# from dataset_lab import *
from concurrent.futures import ProcessPoolExecutor
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import subprocess

faster = True

def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def auto_select_device(device, memory_max=8000, memory_bias=200, strategy='random'):
    r'''
    Auto select device for the experiment. Useful when having multiple GPUs.

    Args:
        memory_max (int): Threshold of existing GPU memory usage. GPUs with
        memory usage beyond this threshold will be deprioritized.
        memory_bias (int): A bias GPU memory usage added to all the GPUs.
        Avoild dvided by zero error.
        strategy (str, optional): 'random' (random select GPU) or 'greedy'
        (greedily select GPU)

    '''
    if device != 'cpu' and torch.cuda.is_available():
        if device == 'auto':
            memory_raw = get_gpu_memory_map()
            # memory_raw[2] = 1000000
            # memory_raw[3] = 1000000
            if strategy == 'greedy' or np.all(memory_raw > memory_max):
                cuda = np.argmin(memory_raw)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info(
                    'Greedy select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))
            elif strategy == 'random':
                memory = 1 / (memory_raw + memory_bias)
                memory[memory_raw > memory_max] = 0
                gpu_prob = memory / memory.sum()
                cuda = np.random.choice(len(gpu_prob), p=gpu_prob)
                logging.info('GPU Mem: {}'.format(memory_raw))
                logging.info('GPU Prob: {}'.format(gpu_prob.round(2)))
                logging.info(
                    'Random select GPU, select GPU {} with mem: {}'.format(
                        cuda, memory_raw[cuda]))

            device = 'cuda:{}'.format(cuda)
    else:
        device = 'cpu'
    return device


class temporal_data:
    def __init__(self, src_path=None, orig_data=None):
        self.src_path = src_path
        print(src_path)
        if orig_data is None:
            if src_path.split('.')[-1] == 'txt':
                orig_data = self._load_data()
                self.data = self._split_data_lines(orig_data)
                self.df_data = pd.DataFrame(orig_data, columns=['src', 'tar', 't', 'label'])
            elif src_path.split('.')[-1] == 'csv':
                self.df_data = pd.read_csv(src_path, sep=',')
            else:
                raise NotImplementedError
        else:
            self.df_data=orig_data
        # Filter out rows where source and target nodes are the same
        self.df_data = self.df_data[self.df_data['src'] != self.df_data['tar']]

        # Create a copy of the DataFrame with source and target nodes swapped
        data_copy = self.df_data.copy()
        data_copy['src'] = self.df_data['tar']
        data_copy['tar'] = self.df_data['src']
        data_copy['label'] = self.df_data['label']

        # Concatenate the original data with the swapped copy
        self.df_data = pd.concat([self.df_data, data_copy], axis=0)

        # Clean the data by removing duplicates
        self._clean_data()

        # Calculate the minimum and maximum values of 't'
        self.min_t = self.df_data['t'].min()
        self.max_t = self.df_data['t'].max()

        # Normalize 't' values to be between 0 and 1
        self.df_data['t'] = self.df_data['t'] - self.min_t
        self.df_data['t'] = self.df_data['t'] / (self.max_t - self.min_t)

        # Get unique 't' values and sort them
        self.unique_ts = sorted(self.df_data['t'].unique())

    def _clean_data(self):
        self.df_data.drop_duplicates(inplace=True)
        self.df_data.reset_index(drop=True, inplace=True)

    def _load_data(self):
        with open(self.src_path, 'r') as reader:
            lines = reader.readlines()
        return lines

    def _split_data_lines(self, orig_data):
        data = [list(map(int, line.split(' '))) for line in orig_data]
        line_len = len(data[0])
        for i, line in enumerate(data):
            assert line_len == len(line), f"Check the {i}-th line of the data."
            line_len = len(line)
        return data

    def visualize(self, data: pd.DataFrame=None):
        '''
        Visualize the data.
        Usage example:
        --------
        >>> data = CTDNE_data(src_path=config.DBLP_edges)
        >>> data.visualize()
        >>> walker = CTDNE_random_walker(data)
        >>> walker.data.visualize(walker.test_walks())
        '''
        if data is None:
            data = self.df_data
        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        xs = data['src']
        ys = data['tar']
        zs = data['t']
#         labels = data['label']
        ax.scatter3D(xs, ys, zs, c=zs, cmap='viridis', s=3, linewidth=0.2)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter3D(xs, ys, zs, c=zs, cmap='viridis', s=5, linewidth=0.2)
        ax.view_init(30, 120)
        plt.tight_layout()
        plt.show()

class CTDNE_data(temporal_data):
    '''
    Usage example:
    --------
    >>> data = CTDNE_data(src_path=config.DBLP_edges)
    >>> save_pickle(data, config.loaded_DBLP)
    >>> data = load_pickle(config.loaded_DBLP)
    --------
    '''
    def __init__(self, nodes=None, edge_list=None,src_path=None, vocal=True, node2idx=None):
        temporal_data.__init__(self, src_path=src_path, orig_data=edge_list)
        self.num_edges = len(self.df_data)
        if nodes is None:
            self.nodes = np.concatenate((self.df_data['src'].unique(), self.df_data['tar'].unique()))
        else:
            self.nodes = nodes
        self.nodes = set(self.nodes.tolist())
        self.nodes = sorted(list(self.nodes)) # By default sorted
        self.vocal = vocal
        self.num_nodes = len(self.nodes)# len(node2idx.items()) if node2idx else len(self.nodes)

        if vocal:
            print("Building index now")
        self.node2idx = node2idx if node2idx else self._build_index()
        self.df_data['src'] = self.df_data['src'].map(self.node2idx)
        self.df_data['tar'] = self.df_data['tar'].map(self.node2idx)

        if vocal:
            print("Finidng neighbors now")
        self.temp_nbrs = self._find_neighbors()
        if vocal:
            print("built successfully")
        # assert len(self.temp_nbrs) == self.num_nodes, "Error occuring while finding neighbors."

    def __len__(self):
        return self.num_edges

    def _build_index(self):
        node2idx = {}
        for i, node in enumerate(self.nodes):
            node2idx[node] = i
        
        return node2idx

    def _find_neighbors(self):
        
        temp_nbrs = defaultdict(list)
        neighbor_check = defaultdict(set)
        
        # Iterate through the DataFrame using itertuples for better performance
        for row in self.df_data.itertuples(index=False):
            
            node1, node2, t, label = int(row.src), int(row.tar), row.t, row.label
    
            # Add the neighbors without duplicates, using sets for efficient membership checks
            if (node1, t,label) not in neighbor_check[node2]:
                temp_nbrs[node2].append((node1, t, label))
                neighbor_check[node2].add((node1, t, label))
            if (node2, t, label) not in neighbor_check[node1]:
                temp_nbrs[node1].append((node2, t, label))
                neighbor_check[node1].add((node2, t, label))
    
        return dict(temp_nbrs)

    
if not faster:
    class HTNE_dataset(CTDNE_data):
        def __init__(self, nodes=None, edge_list=None, src_path=None, neg_size=10, hist_len=2, vocal=True, node2idx=None, device='cpu'):
            if src_path != None:
                CTDNE_data.__init__(self, src_path=src_path, node2idx=node2idx)
            else:
                print("loading edge list to CTDNE in utils.py")
                CTDNE_data.__init__(self, nodes=nodes, edge_list=edge_list, node2idx=node2idx)
            if vocal:
                print("Now building HTNE related data: Find hist")
                
            self.device = device
            self.node2hist = find_hist(self.df_data)
            self.degrees = {}
            self.neg_size = neg_size
            self.hist_len = hist_len
            if vocal:
                print("Now building HTNE related data: Iterate data")
            start = time.time()
            self.calculate_degrees()
            # for _, row in self.df_data.iterrows():
            #     src = int(row['src'])
            #     tar = int(row['tar'])
            #     lab = int(row['label'])
            #     if src not in self.degrees:
            #         self.degrees[src] = 0
            #     if tar not in self.degrees:
            #         self.degrees[tar] = 0
            #     self.degrees[src] += 1
            #     self.degrees[tar] += 1
            end = time.time()
            print("Elapsed time:", end - start, "seconds is used for computing degree")
            self.NEG_SAMPLING_POWER = 0.75
            self.neg_table_size = int(1e8)
            self.data_size = 0
            if vocal:
                print("Now building HTNE related data: Build hist")
            for s in self.node2hist:
                hist = self.node2hist[s]
                self.data_size += len(hist)
            
            self.idx2src_id = [0] * self.data_size
            self.idx2tar_id = [0] * self.data_size
            idx = 0
            if vocal:
                print("Now building HTNE related data: Node hist")
            # for src in self.node2hist:
            #     for tar_id in range(len(self.node2hist[src])):
            #         self.idx2src_id[idx] = src
            #         self.idx2tar_id[idx] = tar_id
            #         idx += 1
            self.idx2src_id = [src for src, targets in self.node2hist.items() for _ in targets]
            self.idx2tar_id = [tar_id for targets in self.node2hist.values() for tar_id in range(len(targets))]

            self.neg_table = np.zeros((self.neg_table_size,))
            self.init_neg_table()
            print("done initializing")
            
        def calculate_degrees(self):
            # Count occurrences of each value in 'src' and 'tar' columns
            src_degrees = self.df_data['src'].value_counts().to_dict()
            tar_degrees = self.df_data['tar'].value_counts().to_dict()
        
            # Combine the counts for 'src' and 'tar'
            self.degrees = defaultdict(int, src_degrees)
            for key, value in tar_degrees.items():
                self.degrees[key] += value
        
            # Convert keys to integers
            self.degrees = {int(k): v for k, v in self.degrees.items()}

        def init_neg_table(self):
            tot_sum, cur_sum, por = 0., 0., 0.
            n_id = 0
            for k in range(self.num_nodes):
                if k in self.degrees:
                    tot_sum += np.power(
                        self.degrees[k],
                        self.NEG_SAMPLING_POWER
                    )
            for k in range(self.neg_table_size):
                if (k + 1.) / self.neg_table_size > por:
                    if n_id in self.degrees:
                        cur_sum += np.power(
                            self.degrees[n_id],
                            self.NEG_SAMPLING_POWER
                        )
                    por = cur_sum / tot_sum
                    n_id += 1
                self.neg_table[k] = n_id - 1

        def negative_sampling(self):
            rand_idx = np.random.randint(
                0,
                self.neg_table_size, 
                (self.neg_size,) 
            )
            sampled_nodes = self.neg_table[rand_idx]
            return sampled_nodes

        def __getitem__(self, idx):
            # print(idx)
            # print(f"idx: {idx}")
            # Edge-based sampling
            src = self.idx2src_id[idx]
            tar_idx = self.idx2tar_id[idx]
            tar, dat, lab = self.node2hist[src][tar_idx]
            if tar_idx - self.hist_len < 0:
                hist = self.node2hist[src][0: tar_idx]
            else:
                # take the most recent samples
                hist = self.node2hist[src][(tar_idx - self.hist_len): tar_idx]
            hist_nodes = [h[0] for h in hist]
            hist_times = [h[1] for h in hist]
            np_h_nodes = np.zeros((self.hist_len,))
            np_h_nodes[:len(hist_nodes)] = hist_nodes
            np_h_times = np.zeros((self.hist_len,))
            np_h_times[:len(hist_times)] = hist_times
            np_h_masks = np.zeros((self.hist_len,))
            np_h_masks[:len(hist_nodes)] = 1.

            neg_nodes = self.negative_sampling()

            # print(f"src: {src}")
            # print(f"tar: {tar}")
            # print(f"dat: {dat}")
            # print(f"lab: {lab}")
            # print(f"np_h_nodes: {np_h_nodes}")
            # print(f"np_h_times: {np_h_times}")
            # print(f"np_h_masks: {np_h_masks}")
            # print(f"neg_nodes: {neg_nodes}")
            return {
                'source': torch.tensor([src], dtype=torch.long),
                'target': torch.tensor([tar], dtype=torch.long),
                'date': torch.tensor([dat], dtype=torch.float),
                'label': torch.tensor([int(lab)], dtype=torch.long),
                'hist_nodes': torch.tensor(np_h_nodes, dtype=torch.long),
                'hist_times': torch.tensor(np_h_times, dtype=torch.float),
                'hist_masks': torch.tensor(np_h_masks, dtype=torch.long),
                'negs': torch.tensor(neg_nodes, dtype=torch.long)
            }

    class CTDNE_random_walker:
        '''
        Usage example:
        --------
        >>> data = CTDNE_data(src_path=config.loaded_DBLP)
        >>> walker = CTDNE_random_walker(data)
        --------
        '''
        def __init__(self, data: CTDNE_data,
                rw_len=20, batch_size=128, label_informed=True,
                edge_bias='uniform', neighbor_bias='uniform'):
            self.data = data
            self.unique_ts = data.unique_ts
            self.rw_len = rw_len
            self.batch_size = batch_size
            self.edge_bias = edge_bias                                   # ['exp', 'order', None]
            self.neighbor_bias = neighbor_bias                           # ['exp', 'order', None]
            self.num_edges = self.data.num_edges
            self.edges = self.data.df_data[['src', 'tar', 'label']].values.tolist()
            self.label_informed = label_informed
            if label_informed:
                label_informed_data = self.data.df_data.loc[data.df_data['label']==1]
                self.edges_labeled = label_informed_data[['src', 'tar', 'label']].values.tolist()
                self.times_labeled = label_informed_data['t'].tolist()
                
            self.times = self.data.df_data['t'].tolist()
            self.temp_nbrs = self.data.temp_nbrs
            if self.edge_bias != 'uniform':
                self.data[f'{self.edge_bias}_prob'] = self._get_edge_prob()

        # def walk(self):
        #     rw = []
        #     ts = []
        #     labs = []
        #     for _ in range(self.batch_size):
        #         start_edge, start_t = self._edge_selection(self.label_informed)
        #         nodes, times, labels = self._temporal_walk(start_edge, start_t, self.rw_len)
        #         rw.append(nodes)
        #         ts.append(times)
        #         labs.append(labels)
        #     rw = np.array(rw) # (batch_size, rw_len)
        #     ts = np.array(ts) # (batch_size, rw_len)
        #     labs = np.array(labs) # (batch_size, rw_len)
        #     return rw, ts, labs

        def single_walk(self):
            start_edge, start_t = self._edge_selection(self.label_informed)
            print("start walking")
            return self._temporal_walk(start_edge, start_t, self.rw_len)
        
        def walk(self, fast=False):
            rw = []
            ts = []
            labs = []
        
            # Use a process pool to execute the walks in parallel
            if fast:
                with ProcessPoolExecutor(max_workers=16) as executor:
                    results = list(executor.map(self.single_walk, [self] * self.batch_size))

                for nodes, times, labels in results:
                    rw.append(nodes)
                    ts.append(times)
                    labs.append(labels)
            else:
                for _ in range(self.batch_size):
                    start_edge, start_t = self._edge_selection(self.label_informed)
                    # print('start_edge', start_edge)
                    # print('start_t', start_t)
                    # print('rw_len', self.rw_len)
                    nodes, times, labels = self._temporal_walk(start_edge, start_t, self.rw_len)
                    # print('nodes', nodes)
                    # print('times', times)
                    # print('labels', labels)
                    rw.append(nodes)
                    ts.append(times)
                    labs.append(labels)
            # Unpack the results
            
        
            rw = np.array(rw) # (batch_size, rw_len)
            ts = np.array(ts) # (batch_size, rw_len)
            labs = np.array(labs) # (batch_size, rw_len)
        
            return rw, ts, labs


        def test_walks(self, label_informed=True):
            rws = None
            times = None
            edges_per_batch = self.batch_size * (self.rw_len - 1)
            num_batches = int(int(self.num_edges / 2) / edges_per_batch) + 1
            for _ in range(num_batches):
                rw, ts = self.walk()
                if rws is None:
                    rws = rw
                else:
                    rws = np.concatenate([rws, rw], axis=0)
                if times is None:
                    times = ts
                else:
                    times = np.concatenate([times, ts], axis=0)
            rws = rws.astype(int)
            edges = temp_walk2edge(rws, times)
            edges = edges[:int(self.num_edges / 2), :]
            edges = pd.DataFrame(edges, columns=['src', 'tar', 't', 'label'])
            edges_copy = edges.copy()
            edges_copy['src'] = edges['tar']
            edges_copy['tar'] = edges['src']
            edges_copy['label'] = edges['label']
            edges = pd.concat([edges, edges_copy], axis=0)
            edges.drop_duplicates(inplace=True)
            edges.reset_index(drop=True, inplace=True)
            print(f"Sampled {len(edges)} edges using temporal random walks.")
            return edges

        def _temporal_walk(self, start_edge, start_t, rw_len, window_size=1.0):
            if start_edge is not None:
                one_walk = [(start_edge[0], start_edge[1], start_t, start_edge[2])]
            else:
                raise ValueError('start_edge is not valid.')
            cur_node = start_edge[1]
            cur_t = start_t
            for _ in range(1, rw_len):
                neighbors = self.data.temp_nbrs[cur_node]
                neighbors = [(n, t, label) for n, t, label in neighbors if t >= cur_t]
                neighbors = [(n, t, label) for n, t, label in neighbors if t <= min(cur_t + window_size, 1.0)]
                next, _ = self._neighbor_selection(neighbors)
                next_node, next_t, nlabel = next
                one_walk.append((cur_node, next_node, next_t, nlabel))
                cur_node = next_node
                cur_t = next_t
            assert len(one_walk) == rw_len, "Error occuring while generating a walk."
            nodes = [i[0] for i in one_walk]
            times = [i[2] for i in one_walk]
            labels = one_walk[0][3]
            # print("retrieved all edges")
            return nodes, times, labels

        def _get_edge_prob(self):
            if self.edge_bias == 'exp':
                probs = [i - self.data.min_t for i in self.times]
                probs = np.exp(probs)
                base = sum(probs)
                probs = probs / base
            elif self.edge_bias == 'linear':
                idxs = np.argsort(self.times)
                probs = [0] * len(idxs)
                for i, j in enumerate(idxs):
                    probs[j] = i
                probs = np.array([i + 1 for i in probs])
                base = np.sum(probs)
                probs = probs / base
            else:
                raise NotImplementedError
            return probs

        def _edge_selection(self, label_informed=True):
            if label_informed:
                num_edges = len(self.edges_labeled)
            else:
                num_edges = self.num_edges
            if self.edge_bias == 'uniform':
                choice = np.random.choice(num_edges)
            elif self.edge_bias == 'exp' or self.edge_bias == 'linear':
                probs = self.data[f'{self.edge_bias}_prob']
                choice = np.random.choice(num_edges, p=probs)
            else:
                raise NotImplementedError
            if label_informed:
                return self.edges_labeled[choice], self.times_labeled[choice]
            else:
                return self.edges[choice], self.times[choice]

        def _neighbor_selection(self, neighbors):
            times = [i[1] for i in neighbors]                            # [3, 1, 2]
            if self.neighbor_bias == 'exp':                              # heavily favor edges that appear later in time
                prob_list = np.exp([t - self.data.min_t for t in times])
                prob_base = np.sum(prob_list)
                prob_list = prob_list / prob_base
                next_idx = np.random.choice(a=len(neighbors), p=prob_list)
            elif self.neighbor_bias == 'linear':                         # favor edges that appear later in time
                prob_list = [0] * len(times)
                idxs = np.argsort(times)                                 # [1, 2, 0]
                for i, j in enumerate(idxs):
                    prob_list[j] = i + 1
                prob_base = np.sum(prob_list)
                prob_list = prob_list / prob_base
                next_idx = np.random.choice(a=len(neighbors), p=prob_list)
            elif self.neighbor_bias == 'uniform':
                next_idx = np.random.choice(len(neighbors))
                prob_list = [1 / len(neighbors) for _ in range(len(neighbors))]
            else:
                raise NotImplementedError
            return neighbors[next_idx], prob_list
            
    def find_hist(df_data):
        start_time = time.time()
        # Group by 'src' and apply a lambda function to create a list of tuples for each 'src'
        node2hist = df_data.sort_values('t').groupby('src').apply(
            lambda x: list(zip(x['tar'], x['t'], x['label']))
        ).to_dict()

        # Convert keys to integers
        node2hist = {int(k): v for k, v in node2hist.items()}
        end_time = time.time()
        logging.info(f"Elapsed time: {end_time - start_time} seconds")
        return node2hist

if faster:

    class HTNE_dataset(CTDNE_data):
        def __init__(self, nodes=None, edge_list=None, src_path=None, neg_size=10, hist_len=2, vocal=True, node2idx=None, device='cpu'):
            if src_path != None:
                CTDNE_data.__init__(self, src_path=src_path, node2idx=node2idx)
            else:
                print("loading edge list to CTDNE in utils.py")
                CTDNE_data.__init__(self, nodes=nodes, edge_list=edge_list, node2idx=node2idx)
            if vocal:
                print("Now building HTNE related data: Find hist")
                
            self.device = device
            self.node2hist = find_hist(self.df_data)#.to(self.device)
            self.degrees = {}
            self.neg_size = neg_size
            self.hist_len = hist_len
            if vocal:
                logging.info("Now building HTNE related data: Iterate data")
            start = time.time()
            self.calculate_degrees()
            # for _, row in self.df_data.iterrows():
            #     src = int(row['src'])
            #     tar = int(row['tar'])
            #     lab = int(row['label'])
            #     if src not in self.degrees:
            #         self.degrees[src] = 0
            #     if tar not in self.degrees:
            #         self.degrees[tar] = 0
            #     self.degrees[src] += 1
            #     self.degrees[tar] += 1
            end = time.time()
            logging.info(f"Elapsed time: {end - start} seconds is used for computing degree")
            self.NEG_SAMPLING_POWER = 0.75
            self.neg_table_size = int(1e8)
            self.data_size = 0
            if vocal:
                logging.info("Now building HTNE related data: Build hist")
            for s in self.node2hist:
                hist = self.node2hist[s]
                self.data_size += hist.shape[0] #len(hist)
            
            self.idx2src_id = [0] * self.data_size
            self.idx2tar_id = [0] * self.data_size
            idx = 0
            if vocal:
                logging.info("Now building HTNE related data: Node hist")
            # for src in self.node2hist:
            #     for tar_id in range(len(self.node2hist[src])):
            #         self.idx2src_id[idx] = src
            #         self.idx2tar_id[idx] = tar_id
            #         idx += 1
            self.idx2src_id = torch.tensor([src for src, targets in self.node2hist.items() for _ in targets], device=self.device)
            # self.idx2tar_id = torch.tensor([tar_id for targets in self.node2hist.values() for tar_id in range(len(targets))], device=self.device)

            # self.tar_dat_lab = torch.tensor([targets[tar_id] for src, targets in self.node2hist.items() for tar_id in range(len(targets))], device=self.device)
            # self.hist_nodes = torch.tensor([torch.cat(targets[max(0, tar_id-self.hist_len):tar_id, 0], [torch.zeros(max(0, self.hist_len - tar_id), device=self.device)]) for src, targets in self.node2hist.items() for tar_id in range(len(targets))], device=self.device)
            # self.hist_times = torch.tensor([torch.cat(targets[max(0, tar_id-self.hist_len):tar_id, 1], [torch.zeros(max(0, self.hist_len - tar_id), device=self.device)]) for src, targets in self.node2hist.items() for tar_id in range(len(targets))], device=self.device)
            # self.hist_masks = torch.ones((self.data_size, self.hist_len), device=self.device)

            self.hist = torch.stack([torch.cat((targets[tar_id:tar_id+1], self.make_third_col_ones(targets[max(0, tar_id-self.hist_len):tar_id]), torch.zeros((max(0, self.hist_len - tar_id), 3)))) for src, targets in self.node2hist.items() for tar_id in range(len(targets))], dim=0)
            self.hist = self.hist.to(self.device)

            # import multiprocessing

            # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            #     results = pool.map(self.compute_hist_nodes, self.node2hist.items())
            # pool.close()
            # pool.join()

            # self.hist_nodes = torch.stack([item for sublist in results for item in sublist])
            
            # self.hist_nodes = []
            
            # for src, targets in self.node2hist.items():
            #      for tar_id in range(len(targets)):
            #          print(src, tar_id)
            #          print(type(targets))
            #          print(type(torch.zeros((max(0, self.hist_len - tar_id), 3))))
            #          print(type(targets[max(0, tar_id-self.hist_len):tar_id]))
            #          a = targets[tar_id:tar_id+1].shape
            #          b = targets[max(0, tar_id-self.hist_len):tar_id].shape
            #          c = torch.zeros((max(0, self.hist_len - tar_id), 3)).shape

            #         #  assert a[0] + b[0] + c[0] == 5, 'erro'

            #          x = torch.cat((targets[tar_id:tar_id+1], self.make_third_col_ones(targets[max(0, tar_id-self.hist_len):tar_id]), torch.zeros((max(0, self.hist_len - tar_id), 3))))
            #          self.hist_nodes.append(x)
            
            # self.hist_nodes = torch.stack(self.hist_nodes)
            
            # self.neg_table = torch.zeros((self.neg_table_size,))
            if vocal:
                logging.info("Now building HTNE related data: Init neg table")
            # self.neg_table = np.zeros((self.neg_table_size,))
            # self.init_neg_table()
            # self.neg_table = torch.tensor(self.neg_table, device=self.device)

            # self.neg_table_size = int(1e8)
            self.neg_table = np.zeros((self.neg_table_size,))
            # self.init_neg_table()
            logging.info("done initializing")

        def compute_hist_nodes(self, data):
            src, targets = data
            result = []
            for tar_id in range(len(targets)):
                x = torch.cat((targets[tar_id:tar_id+1], 
                            torch.ones((max(0, tar_id-self.hist_len), 3)), # Placeholder for your third column
                            torch.zeros((max(0, self.hist_len - tar_id), 3))))
                result.append(x)
            return result

        def make_third_col_ones(self, tensor):
            tensor[:,2] = 1
            return tensor
            
        def calculate_degrees(self):
            # Count occurrences of each value in 'src' and 'tar' columns
            src_degrees = self.df_data['src'].value_counts().to_dict()
            tar_degrees = self.df_data['tar'].value_counts().to_dict()
        
            # Combine the counts for 'src' and 'tar'
            self.degrees = defaultdict(int, src_degrees)
            for key, value in tar_degrees.items():
                self.degrees[key] += value
        
            # Convert keys to integers
            self.degrees = {int(k): v for k, v in self.degrees.items()}

        def init_neg_table(self):
            # return torch.tensor([0 if k not in self.degrees else np.power(
            #             self.degrees[k],
            #             self.NEG_SAMPLING_POWER
            #         ) for k in range(max(self.node2idx.values()) + 1)], device=self.device)
            tot_sum, cur_sum, por = 0., 0., 0.
            n_id = 0
            for k in range(self.num_nodes):
                if k in self.degrees:
                    tot_sum += np.power(
                        self.degrees[k],
                        self.NEG_SAMPLING_POWER
                    )
            for k in range(self.neg_table_size):
                if (k + 1.) / self.neg_table_size > por:
                    if n_id in self.degrees:
                        cur_sum += np.power(
                            self.degrees[n_id],
                            self.NEG_SAMPLING_POWER
                        )
                    por = cur_sum / tot_sum
                    n_id += 1
                self.neg_table[k] = n_id - 1

        def negative_sampling(self, batch_size):  ##FIX
            # print(type(self.neg_table))
            # x = torch.multinomial(input=self.neg_table, num_samples=batch_size*self.neg_size, replacement=True).reshape(batch_size, self.neg_size)
            # # print(x)
            # return x
            rand_idx = torch.randint(
                0,
                self.neg_table_size, 
                (batch_size, self.neg_size),
                device=self.device
            )
            # print('rand_idx', rand_idx, rand_idx.shape)
            sampled_nodes = torch.tensor(self.neg_table[rand_idx.cpu().numpy()], device=self.device)
            # sampled_nodes = torch.tensor(self.neg_table[rand_idx], device=self.device)
            # print('sampled_nodes', sampled_nodes, sampled_nodes.shape)

            # sampled_nodes = sampled_nodes.to(self.device)
            # print('sampled_nodes', sampled_nodes, sampled_nodes.shape)
            return sampled_nodes

        def __getitem__(self, idx):
            # Edge-based sampling
            with torch.autograd.profiler.record_function("htne getitem: src"):
                src = self.idx2src_id[idx].item()
            with torch.autograd.profiler.record_function("htne getitem: tar_idx"):
                tar_idx = self.idx2tar_id[idx].item()
            with torch.autograd.profiler.record_function("htne getitem: tar, dat, lab"):
                tar, dat, lab = self.node2hist[src][tar_idx]
            with torch.autograd.profiler.record_function("htne getitem: hist"):
                if tar_idx - self.hist_len < 0:
                    hist = self.node2hist[src][0: tar_idx]
                else: # self.node2hist    [(tar1, dat1, lab1), ()]
                    # take the most recent samples
                    hist = self.node2hist[src][(tar_idx - self.hist_len): tar_idx]
            with torch.autograd.profiler.record_function("htne getitem: hist_nodes"):
                hist_nodes = [h[0] for h in hist]
            with torch.autograd.profiler.record_function("htne getitem: hist_times"):
                hist_times = [h[1] for h in hist]
            with torch.autograd.profiler.record_function("htne getitem: np_h_nodes"):
                np_h_nodes = np.zeros((self.hist_len,))
                np_h_nodes[:len(hist_nodes)] = hist_nodes
            with torch.autograd.profiler.record_function("htne getitem: np_h_times"):
                np_h_times = np.zeros((self.hist_len,))
                np_h_times[:len(hist_times)] = hist_times
            with torch.autograd.profiler.record_function("htne getitem: np_h_masks"):
                np_h_masks = np.zeros((self.hist_len,))
                np_h_masks[:len(hist_nodes)] = 1.
            with torch.autograd.profiler.record_function("htne getitem: neg_nodes"):
                neg_nodes = self.negative_sampling()

                #### Try to tensorize all of this somehow. ALso change from dataloader to just access it your own way. (maybe don't mess with the dataloader)

            # print(f"src: {src}")
            # print(f"tar: {tar}")
            # print(f"dat: {dat}")
            # print(f"lab: {lab}")
            # print(f"np_h_nodes: {np_h_nodes}")
            # print(f"np_h_times: {np_h_times}")
            # print(f"np_h_masks: {np_h_masks}")
            # print(f"neg_nodes: {neg_nodes}")
            with torch.autograd.profiler.record_function("htne getitem: return"):
                return {
                    'source': torch.tensor([src], dtype=torch.long),#.clone().detach().cpu(),
                    'target': torch.tensor([tar], dtype=torch.long),#.clone().detach().cpu(),
                    'date': torch.tensor([dat], dtype=torch.float),#.clone().detach().cpu(),
                    'label': torch.tensor([int(lab)], dtype=torch.long),#.clone().detach().cpu(),
                    'hist_nodes': torch.tensor(np_h_nodes, dtype=torch.long),#.clone().detach().cpu(),
                    'hist_times': torch.tensor(np_h_times, dtype=torch.float),#.clone().detach().cpu(),
                    'hist_masks': torch.tensor(np_h_masks, dtype=torch.long),#.clone().detach().cpu(),
                    'negs': neg_nodes#torch.tensor(neg_nodes, dtype=torch.long)#.clone().detach().cpu()
                }

    class CTDNE_random_walker:
        '''
        Usage example:
        --------
        >>> data = CTDNE_data(src_path=config.loaded_DBLP)
        >>> walker = CTDNE_random_walker(data)
        --------
        '''
        def __init__(self, data: CTDNE_data,
                rw_len=20, batch_size=128, label_informed=True,
                edge_bias='uniform', neighbor_bias='uniform'):
            self.data = data
            self.unique_ts = data.unique_ts
            self.rw_len = rw_len
            self.batch_size = batch_size
            self.edge_bias = edge_bias                                   # ['exp', 'order', None]
            self.neighbor_bias = neighbor_bias                           # ['exp', 'order', None]
            self.num_edges = self.data.num_edges
            self.edges = self.data.df_data[['src', 'tar', 'label']].values.tolist()
            self.label_informed = label_informed
            if label_informed:
                label_informed_data = self.data.df_data.loc[data.df_data['label']==1]
                self.edges_labeled = label_informed_data[['src', 'tar', 'label']].values.tolist()
                self.times_labeled = label_informed_data['t'].tolist()
                
            self.times = self.data.df_data['t'].tolist()
            self.temp_nbrs = self.data.temp_nbrs
            if self.edge_bias != 'uniform':
                self.data[f'{self.edge_bias}_prob'] = self._get_edge_prob()

        # def walk(self):
        #     rw = []
        #     ts = []
        #     labs = []
        #     for _ in range(self.batch_size):
        #         start_edge, start_t = self._edge_selection(self.label_informed)
        #         nodes, times, labels = self._temporal_walk(start_edge, start_t, self.rw_len)
        #         rw.append(nodes)
        #         ts.append(times)
        #         labs.append(labels)
        #     rw = np.array(rw) # (batch_size, rw_len)
        #     ts = np.array(ts) # (batch_size, rw_len)
        #     labs = np.array(labs) # (batch_size, rw_len)
        #     return rw, ts, labs

        def single_walk(self):
            start_edge, start_t = self._edge_selection(self.label_informed)
            print("start walking")
            return self._temporal_walk(start_edge, start_t, self.rw_len)
        
        def walk(self, fast=False):
            rw = []
            ts = []
            labs = []
        
            # Use a process pool to execute the walks in parallel
            if fast:
                with ProcessPoolExecutor(max_workers=16) as executor:
                    results = list(executor.map(self.single_walk, [self] * self.batch_size))

                for nodes, times, labels in results:
                    rw.append(nodes)
                    ts.append(times)
                    labs.append(labels)
            else:
                for _ in range(self.batch_size):
                    with torch.autograd.profiler.record_function("create graph: start_edge"):
                        start_edge, start_t = self._edge_selection(self.label_informed)
                    # print('start_edge', start_edge)
                    # print('start_t', start_t)
                    # print('rw_len', self.rw_len)
                    with torch.autograd.profiler.record_function("create graph: _temporal_walk"):
                        nodes, times, labels = self._temporal_walk(start_edge, start_t, self.rw_len)
                    # print('nodes', nodes)
                    # print('times', times)
                    # print('labels', labels)
                    rw.append(nodes)
                    ts.append(times)
                    labs.append(labels)
            # Unpack the results
            
        
            rw = np.array(rw) # (batch_size, rw_len)
            ts = np.array(ts) # (batch_size, rw_len)
            labs = np.array(labs) # (batch_size, rw_len)
        
            return rw, ts, labs


        def test_walks(self, label_informed=True):
            rws = None
            times = None
            edges_per_batch = self.batch_size * (self.rw_len - 1)
            num_batches = int(int(self.num_edges / 2) / edges_per_batch) + 1
            for _ in range(num_batches):
                rw, ts = self.walk()
                if rws is None:
                    rws = rw
                else:
                    rws = np.concatenate([rws, rw], axis=0)
                if times is None:
                    times = ts
                else:
                    times = np.concatenate([times, ts], axis=0)
            rws = rws.astype(int)
            edges = temp_walk2edge(rws, times)
            edges = edges[:int(self.num_edges / 2), :]
            edges = pd.DataFrame(edges, columns=['src', 'tar', 't', 'label'])
            edges_copy = edges.copy()
            edges_copy['src'] = edges['tar']
            edges_copy['tar'] = edges['src']
            edges_copy['label'] = edges['label']
            edges = pd.concat([edges, edges_copy], axis=0)
            edges.drop_duplicates(inplace=True)
            edges.reset_index(drop=True, inplace=True)
            print(f"Sampled {len(edges)} edges using temporal random walks.")
            return edges

        def _temporal_walk(self, start_edge, start_t, rw_len, window_size=1.0):
            if start_edge is not None:
                one_walk = [(start_edge[0], start_edge[1], start_t, start_edge[2])]
            else:
                raise ValueError('start_edge is not valid.')
            cur_node = start_edge[1]
            cur_t = start_t
            for _ in range(1, rw_len):
                neighbors = self.data.temp_nbrs[cur_node]
                neighbors = [(n, t, label) for n, t, label in neighbors if t >= cur_t]
                neighbors = [(n, t, label) for n, t, label in neighbors if t <= min(cur_t + window_size, 1.0)]
                next, _ = self._neighbor_selection(neighbors)
                next_node, next_t, nlabel = next
                one_walk.append((cur_node, next_node, next_t, nlabel))
                cur_node = next_node
                cur_t = next_t
            assert len(one_walk) == rw_len, "Error occuring while generating a walk."
            nodes = [i[0] for i in one_walk]
            times = [i[2] for i in one_walk]
            labels = one_walk[0][3]
            # print("retrieved all edges")
            return nodes, times, labels

        def _get_edge_prob(self):
            if self.edge_bias == 'exp':
                probs = [i - self.data.min_t for i in self.times]
                probs = np.exp(probs)
                base = sum(probs)
                probs = probs / base
            elif self.edge_bias == 'linear':
                idxs = np.argsort(self.times)
                probs = [0] * len(idxs)
                for i, j in enumerate(idxs):
                    probs[j] = i
                probs = np.array([i + 1 for i in probs])
                base = np.sum(probs)
                probs = probs / base
            else:
                raise NotImplementedError
            return probs

        def _edge_selection(self, label_informed=True):
            if label_informed:
                num_edges = len(self.edges_labeled)
            else:
                num_edges = self.num_edges
            if self.edge_bias == 'uniform':
                choice = np.random.choice(num_edges)
            elif self.edge_bias == 'exp' or self.edge_bias == 'linear':
                probs = self.data[f'{self.edge_bias}_prob']
                choice = np.random.choice(num_edges, p=probs)
            else:
                raise NotImplementedError
            if label_informed:
                return self.edges_labeled[choice], self.times_labeled[choice]
            else:
                return self.edges[choice], self.times[choice]

        def _neighbor_selection(self, neighbors):
            times = [i[1] for i in neighbors]                            # [3, 1, 2]
            if self.neighbor_bias == 'exp':                              # heavily favor edges that appear later in time
                prob_list = np.exp([t - self.data.min_t for t in times])
                prob_base = np.sum(prob_list)
                prob_list = prob_list / prob_base
                next_idx = np.random.choice(a=len(neighbors), p=prob_list)
            elif self.neighbor_bias == 'linear':                         # favor edges that appear later in time
                prob_list = [0] * len(times)
                idxs = np.argsort(times)                                 # [1, 2, 0]
                for i, j in enumerate(idxs):
                    prob_list[j] = i + 1
                prob_base = np.sum(prob_list)
                prob_list = prob_list / prob_base
                next_idx = np.random.choice(a=len(neighbors), p=prob_list)
            elif self.neighbor_bias == 'uniform':
                next_idx = np.random.choice(len(neighbors))
                prob_list = [1 / len(neighbors) for _ in range(len(neighbors))]
            else:
                raise NotImplementedError
            return neighbors[next_idx], prob_list
            
    def find_hist(df_data):
        start_time = time.time()
        # Group by 'src' and apply a lambda function to create a list of tuples for each 'src'
        node2hist = df_data.sort_values('t').groupby('src').apply(
            # lambda x: torch.tensor(list(zip(x['tar'], x['t'], x['label'])))
            lambda x: torch.stack(
            (
                torch.tensor(x['tar'].values, dtype=torch.long),   # Assuming 'tar' is integer type
                torch.tensor(x['t'].values, dtype=torch.float),    # Assuming 't' needs to be float
                torch.tensor(x['label'].values, dtype=torch.long)  # Assuming 'label' is integer type
            ),
            dim=1
        )
        ).to_dict()

        # Convert keys to integers
        node2hist = {int(k): v for k, v in node2hist.items()}
        end_time = time.time()
        logging.info(f"Elapsed time: {end_time - start_time} seconds")
        return node2hist



def load_data(dataset, batch_size):
    with torch.autograd.profiler.record_function("htne: load data"):
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            # num_workers=0,
            # pin_memory=True
        )
    return dataloader

class Htne(nn.Module):
    ''' For training the stand alone encoder
    '''
    def __init__(
        self,
        emb_size,
        node_dim, # the size of the dataset
        new = False
    ):
        super(Htne, self).__init__()
        with torch.autograd.profiler.record_function("htne: initialize embeddings"):
            self.node_dim = node_dim
            self.emb_size = emb_size
            self.new = new
            self.node_emb = nn.Embedding(self.node_dim, self.emb_size, sparse=True)
            if self.new: self.node_emb2 = nn.Embedding(self.node_dim, self.emb_size, sparse=True)
            nn.init.uniform_(
                self.node_emb.weight.data,
                - 1.0 / self.emb_size,
                1.0 / self.emb_size
            )
        with torch.autograd.profiler.record_function("htne: initialize delta"):
            self.delta = nn.Embedding(self.node_dim, 1, sparse=True)
            nn.init.constant_(self.delta.weight.data, 1)

    def HTNE_loss(self, p_lambda, n_lambda):
        with torch.autograd.profiler.record_function("htne: compute HTNE loss"):
            pos_loss = torch.log(p_lambda.sigmoid() + 1e-6).neg()
            neg_loss = torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
            loss =  pos_loss - neg_loss
        return loss

    def forward(
        self,
        s_nodes, # source nodes
        t_nodes, # target nodes
        t_times, # edge times
        h_nodes, # history nodes
        h_times, # history times
        h_time_mask, # only a small size of his are considered
        n_nodes, # negative sampling nodes
        ret_early = False,
    ):
        with torch.autograd.profiler.record_function("htne: compute embeddings"):
            src_emb = self.node_emb(s_nodes)
            h_emb = self.node_emb(h_nodes)
            delta = self.delta(s_nodes).squeeze(2) # [batch_size, 1]
            d_time = torch.abs(t_times - h_times) # [batch_size, hist_len]

        if not self.new:
            with torch.autograd.profiler.record_function("htne: compute old model embeddings"):
                tar_emb = self.node_emb(t_nodes)
                n_emb = self.node_emb(n_nodes)
                att = F.softmax(((src_emb - h_emb)**2).sum(dim=2).neg(), dim=1) # [batch_size, hist_len]
                p_mu = ((src_emb - tar_emb)**2).sum(dim=2).neg().squeeze(dim=1) # [batch_size, 1]
                p_alpha = ((h_emb - tar_emb)**2).sum(dim=2).neg()  # [batch_size, hist_len]
                n_mu = ((src_emb - n_emb)**2).sum(dim=2).neg() # [batch_size, neg_size]
                n_alpha = ((h_emb.unsqueeze(2) - n_emb.unsqueeze(1))**2).sum(dim=3).neg() # [batch_size, hist_len, neg_size]
        else:
            with torch.autograd.profiler.record_function("htne: compute new model embeddings"):
                tar_emb = self.node_emb2(t_nodes)
                n_emb = self.node_emb2(n_nodes)
                att = F.softmax((src_emb * h_emb).sum(dim=2), dim=1) # [batch_size, hist_len]
                p_mu = (src_emb * tar_emb).sum(dim=2).squeeze(dim=1) # [batch_size, 1]
                p_alpha = (h_emb * tar_emb).sum(dim=2)  # [batch_size, hist_len]
                n_mu = (src_emb * n_emb).sum(dim=2) # [batch_size, neg_size]
                n_alpha = (h_emb.unsqueeze(2) * n_emb.unsqueeze(1)).sum(dim=3) # [batch_size, hist_len, neg_size]

        with torch.autograd.profiler.record_function("htne: compute lambda values"):
            p_lambda = p_mu + (att * h_time_mask * p_alpha * torch.exp(delta * d_time)).sum(dim=1)
            n_lambda = n_mu + ((att * h_time_mask).unsqueeze(2) * n_alpha * torch.exp(delta * d_time).unsqueeze(2)).sum(dim=1)

        with torch.autograd.profiler.record_function("htne: compute loss"):
            loss = self.HTNE_loss(p_lambda, n_lambda)

        if ret_early:
            return src_emb, tar_emb, n_emb, h_emb, self.delta, self.HTNE_loss
        else:
            return loss

    def normalize(self):
        if not self.new:
            self.node_emb.weight.data = F.normalize(self.node_emb.weight.data, p=2, dim=0)
        if self.new:
            # TODO fix for both embeddings
            self.node_emb2.weight.data = F.normalize(self.node_emb2.weight.data, p=2, dim=0)


def without(df1, df2):
    # return df1.merge(df2, indicator=True, how='outer').loc[lambda x: x['_merge'] == 'left_only'].drop('_merge', axis=1)#.reset_index(drop=True, inplace=True)

    df2 = df2.reindex(columns=df1.columns)
    
    # Perform an outer join and use the indicator argument to add a column _merge that shows whether each row is from 'left_only', 'right_only', or 'both'
    merged_df = pd.merge(df1, df2, how='outer', indicator=True)
    
    # Filter rows that are only in the left DataFrame (df1 and not in df2)
    result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    return result_df