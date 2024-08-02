import argparse
import matplotlib.pyplot as plt
import warnings
import faiss
import os
from HTNE import *
import torch
from torch.optim import SparseAdam
# from torch.utils.data import DataLoader, RandomSampler
from concurrent.futures import ProcessPoolExecutor
import torch.nn as nn
import numpy as np
import pandas as pd
# import utils
import copy
import torch.optim as optim
from torch.autograd import grad#, profiler as autograd_profiler
from torch.nn.functional import one_hot
import torch.nn.functional as F
import time
from torch.profiler import profile, record_function, ProfilerActivity
debug = False
debug_generator = False#True

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.zeros_(self.l1.bias)
        self.l2 = nn.Linear(hidden_size, output_size).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.zeros_(self.l2.bias)

    def forward(self, h, sigmoid=False):
        h = self.l1(h)
        if sigmoid:
            h = self.l2(torch.sigmoid(h))
        else:
            h = self.l2(F.relu(h))
        return h

class BaseGenerator(nn.Module):
    def __init__(self, H_inputs, N, device='cpu'):
        super(BaseGenerator, self).__init__()
        self.device = device

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim), dtype=torch.float64, device=self.device)

    def sample(self, num_samples):
        noise = self.sample_latent(num_samples)
        labels = torch.ones((num_samples, 1), device=self.device)
        noise = torch.hstack((noise, labels))
        input_zeros = self.init_hidden(num_samples).contiguous()#.type(torch.float64).to(self.device)
        if debug_generator: print("ITERATION STARTED")
        rw, ts, _ = self(noise, input_zeros, self.device)                              # (n, rw_len, N), (n, rw_len, 1)
        if debug_generator: print("ITERATION COMPLETE")
        return rw, ts

    def sample_discrete(self, num_samples):
        with torch.no_grad():
            rw, ts = self.sample(num_samples)                          # (n, rw_len, N), (n, rw_len, 1)
        rw = np.argmax(rw.cpu().numpy(), axis=2)                               # (n, rw_len)
        ts = torch.squeeze(ts, dim=2).cpu().numpy()                            # (n, rw_len)
        return rw, ts
    
    def graph_edit(self, num_samples, walks):
    
        noise = self.sample_latent(num_samples)
        labels = torch.ones(noise.shape[0], 1).to(self.device)
        noise = torch.hstack((noise, labels))
        input_zeros = self.init_hidden(num_samples).contiguous().type(torch.float64).to(self.device)
        if debug_generator: 
            print("Noise shape:", noise.shape, "Noise:", noise)
            print("Input zeros shape:", input_zeros.shape, "Input zeros:", input_zeros)
        rm_rw, add_rw = self(noise, input_zeros, input_rws=walks)                           # (n, rw_len)
        # if debug_generator: 
        #     print("Removed RW shape:", rm_rw.shape, "Removed RW:", rm_rw)
        #     print("Added RW shape:", add_rw.shape, "Added RW:", add_rw)
        # import sys
        # sys.exit(0)
        return rm_rw, add_rw
    
    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand(logits.shape, dtype=torch.float64, device=self.device)   # gumbel_noise = uniform noise [0, 1]
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temp):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gumbel = 0#self.sample_gumbel(logits)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temp, dim=1)
        return y

    def init_hidden(self, num_samples):
        # weight = next(self.parameters()).data
        # return weight.new(batch_size, self.H_inputs).zero_().type(torch.float64)
        return torch.zeros(num_samples, self.H_inputs, dtype=torch.float64, device=self.device)

class faiss_gpu_index():
    def __init__(self, matrix, device, temp, nlist = 4000, nprobe = 200, k = 750): #nlist = 4000, nprobe = 400, k = 2000
        self.generator = BaseGenerator(1, 1)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.n, self.d = matrix.shape
        self.device = device
        self.nlist = nlist
        self.nprobe = nprobe
        self.matrix = matrix
        self.k = k
        self.temp = temp

        norm = torch.norm(matrix, dim=1, p=2)
        sorted_norms, self.sorted_indices = torch.sort(norm)
        self.norm_positions = torch.empty_like(self.sorted_indices)
        self.norm_positions[self.sorted_indices] = torch.arange(len(self.sorted_indices)).to(device)
        
        matrix_np = matrix.cpu().numpy().astype(np.float32)
        
        self.quantizer = faiss.IndexFlatL2(self.d)  # The coarse quantizer
        self.index_ivf = faiss.IndexIVFFlat(self.quantizer, self.d, nlist, faiss.METRIC_L2)
        self.index_ivf.train(matrix_np)
        self.index_ivf.add(matrix_np)
        self.index_ivf.nprobe = self.nprobe
        res = faiss.StandardGpuResources()
        self.index_ivf_gpu = faiss.index_cpu_to_gpu(res, 0, self.index_ivf)

    def get(self, queries):
        queries_np = queries.cpu().numpy().astype(np.float32)
        try:
            # self.start_event.record()
            distances_ivf_gpu, indices_ivf_gpu = self.index_ivf_gpu.search(queries_np, self.k)
            # torch.cuda.synchronize()
            # self.end_event.record()
            faiss_ivf_gpu_time = 0#self.start_event.elapsed_time(self.end_event)
        except Exception as e:
            raise Exception(f"Error transferring index to GPU: {e}")
        return distances_ivf_gpu, indices_ivf_gpu, faiss_ivf_gpu_time
    
    def up_project(self, H_graph, real_indices):
        num_samples = int(self.n ** 0.5)
        torch.cuda.synchronize()
        _ , indices, _ = self.get(H_graph)
        torch.cuda.synchronize()
        indices = torch.tensor(indices, device=H_graph.device) #[batch, k]

        # Get indices with larger norm than indices
        batch_size = indices.size(0)
        max_norm_indices = indices[torch.arange(batch_size), torch.argmax(torch.norm(self.matrix[indices], dim=2), dim=1)]
        larger_magnitude_indices = []
        max_len = 0
        for batch_idx in range(batch_size):
            tensor = self.sorted_indices[self.norm_positions[max_norm_indices[batch_idx]]+1:]
            larger_magnitude_indices.append(tensor)
            max_len = max(max_len, tensor.shape[0])
        for batch_idx in range(batch_size):
            tensor = larger_magnitude_indices[batch_idx]
            if tensor.shape[0] < max_len:
                tensor = torch.cat((tensor, torch.randint(0, self.n, (max_len - tensor.shape[0],)).to(self.device)))
            larger_magnitude_indices[batch_idx] = tensor
        larger_magnitude_indices = torch.stack(larger_magnitude_indices, dim=0)
        print('larger_magnitude_indices', larger_magnitude_indices.shape)

        # Concat and continue the computation
        indices = torch.concat((indices, larger_magnitude_indices, real_indices.unsqueeze(1)), dim=1)
        print('indices.shape', indices.shape)
        inner_products = torch.matmul(self.matrix[indices], H_graph.unsqueeze(2)).squeeze(2)
        inner_products_softmaxed = self.generator.gumbel_softmax_sample(inner_products, self.temp)
        top_candidates = indices[torch.arange(inner_products_softmaxed.shape[0]), torch.argmax(inner_products_softmaxed, dim=1)]
        v = torch.zeros((batch_size, self.n), device=H_graph.device)
        batch_indices = torch.arange(batch_size, device=H_graph.device).unsqueeze(1)
        v[batch_indices, indices] = inner_products_softmaxed.float()
        
        thresh_indices = torch.randint(0, self.n, (H_graph.shape[0], num_samples), device=H_graph.device)
        print('thresh_indices.shape', thresh_indices.shape)
        thresh_vectors = self.matrix[thresh_indices]
        print('thresh_vectors.shape', thresh_vectors.shape)
        sampled_values = self.generator.gumbel_softmax_sample(torch.matmul(thresh_vectors, H_graph.unsqueeze(2)), self.temp) # [H_graph.shape[0], self.n/100), 1]
        # print(int(num_samples * 0.05))
        # print(torch.sort(sampled_values.squeeze(2), dim=1).values.shape)
        thresholds = torch.sort(sampled_values.squeeze(2), dim=1).values[:, int(num_samples * 0.05)]

        return v, top_candidates, thresholds
        

class DyGANGenerator(BaseGenerator):                                                                    #k=100
    def __init__(self, H_inputs, batch_size, H, z_dim, H_t, N, rw_len, temp, disten=False, k=2, rm_th=0.001, add_th=0.95, device='cpu'):
        ''' The original DyGAN generator
        H_inputs: input dimension
        H:        hidden state dimension
        z_dim:    dimension of the latent code z
        H_t:      dimension of the time hidden embedding
        N:        Number of nodes in the graph to generate
        rw_len:   Length of random walks to generate
        add_th:   threshold for adding node
        rm_th:   threshold for removing node
        temp:     temperature for the gumbel softmax
        k:        number of verification done during evaluation
        '''
        BaseGenerator.__init__(self, H_inputs, N, device)
        self.device = device
        self.compressor =  nn.Linear(z_dim+1, z_dim, device=self.device).type(torch.float64)
        self.shared_intermediate = nn.Linear(z_dim, H, device=self.device).type(torch.float64)     # (z_dim, H)
        torch.nn.init.xavier_uniform_(self.shared_intermediate.weight)
        torch.nn.init.zeros_(self.shared_intermediate.bias)
        self.c_intermediate = nn.Linear(H, H, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.c_intermediate.weight)
        torch.nn.init.zeros_(self.c_intermediate.bias)
        self.h_intermediate = nn.Linear(H, H, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.h_intermediate.weight)
        torch.nn.init.zeros_(self.h_intermediate.bias)
        self.lstmcell = nn.LSTMCell(H_inputs, H, device=self.device).type(torch.float64)
        if disten:
            self.time_adapter = MLP(H, H, H, device=self.device).type(torch.float64)  
        self.time_decoder = TimeDecoder(H, H_t, device=self.device).type(torch.float64)
        self.Wt_up = nn.Linear(1, H_t, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_up.weight)
        torch.nn.init.zeros_(self.Wt_up.bias)
        self.W_up = nn.Linear(H, N, bias=False, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_up.weight)
        # torch.nn.init.zeros_(self.W_up.bias)

        self.W_down = nn.Linear(N, H_inputs, bias=False, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.W_vt = nn.Linear(H_inputs + H_t, H_inputs, bias=False, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.W_vt.weight)
        self.prob = nn.Linear(H, 1, bias=False, device=self.device).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.prob.weight)
        self.disten = disten
        self.rw_len = rw_len
        self.temp = temp
        self.H_inputs = H_inputs
        self.batch_size=batch_size,
        self.H = H
        self.latent_dim = z_dim
        self.H_t = H_t
        self.N = N
        self.k = k
        self.rm_th = rm_th
        self.add_th = add_th
        
        self.index = None

    def setup_index(self):
        nlist = int(self.N/42)
        logging.info(f"nlist: {nlist}")
        self.index = faiss_gpu_index(self.W_up.weight.data.detach(), self.device, self.temp, nlist = nlist, k = int((self.N ** 0.5) /2))
    
    '''
        Training:
            graph editor module works just like a generator by producing the best sequence infered
        Evaluating:
            graph editor will produce the best action based on the joint probability of v1, v2, t
    '''
    def forward(self, latent, inputs, induced=True, input_rws=None, index=None):
        start = time.time()
        with torch.autograd.profiler.record_function("gen forward: debug initial"):
            if debug_generator:
                print(f"latent shape: {latent.shape}, inputs shape: {inputs.shape}, induced: {induced}, input_rws: {input_rws}")
        with torch.autograd.profiler.record_function("gen forward: compressor"):
            if induced:
                latent = torch.tanh(self.compressor(latent))
                if debug_generator:
                    print(f"latent after compressor shape: {latent.shape}")
        with torch.autograd.profiler.record_function("gen forward: shared_intermediate"):
            shared_intermediate = torch.tanh(self.shared_intermediate(latent))
            if debug_generator:
                print(f"shared_intermediate shape: {shared_intermediate.shape}")
        with torch.autograd.profiler.record_function("gen forward: hc calculation"):
            hc = (torch.tanh(self.h_intermediate(shared_intermediate)),
                torch.tanh(self.c_intermediate(shared_intermediate)))              # stack h and c
            if debug_generator:
                print(f"hc[0] shape: {hc[0].shape}, hc[1] shape: {hc[1].shape}")
        with torch.autograd.profiler.record_function("gen forward: hc copy"):
            hc_copy = (torch.clone(hc[0]), torch.clone(hc[1]))
            if debug_generator:
                print(f"hc_copy[0] shape: {hc_copy[0].shape}, hc_copy[1] shape: {hc_copy[1].shape}")
        with torch.autograd.profiler.record_function("gen forward: last_t initialization"):
            last_t = torch.zeros(inputs.shape[0], 1).type(torch.float64).to(self.device)
            last_t_copy = torch.zeros(inputs.shape[0], 1).type(torch.float64).to(self.device)
            if debug_generator:
                print(f"last_t shape: {last_t.shape}, last_t_copy shape: {last_t_copy.shape}")
        if input_rws == None: ## For training
            with torch.autograd.profiler.record_function("gen forward: training path"):
                rw = []
                ts = []
                for _ in range(self.rw_len):
                    with torch.autograd.profiler.record_function("gen forward: lstmcell"):
                        hh, cc = self.lstmcell(inputs, hc)                                 # start with a fake input
                        hc = (hh, cc)                                                      # for the next cell step
                        if debug_generator:
                            print(f"hh shape: {hh.shape}, cc shape: {cc.shape}")
                    with torch.autograd.profiler.record_function("gen forward: disten check"):
                        if self.disten:
                            attn_map = torch.sigmoid(self.time_adapter(hh))
                            H_graph = attn_map * hh
                            H_time = (1 - attn_map) * hh
                            if debug_generator:
                                print(f"attn_map shape: {attn_map.shape}, H_graph shape: {H_graph.shape}, H_time shape: {H_time.shape}")
                        else:
                            H_graph, H_time = hh, hh
                    with torch.autograd.profiler.record_function("gen forward: W_up"):
                        p = self.W_up(H_graph)                                             # (n, N)
                    with torch.autograd.profiler.record_function("gen forward: gumbel_softmax_sample"):
                        v = self.gumbel_softmax_sample(p, self.temp)
                        if debug_generator:
                            print(f"v shape: {v.shape}")
                    with torch.autograd.profiler.record_function("gen forward: W_down"):
                        H_graph = self.W_down(v)                                           # (n, H)
                    with torch.autograd.profiler.record_function("gen forward: time_decoder"):
                        t = self.time_decoder(H_time)                                      # (n, 1)
                        t = self.time_decoder.constraints(t, last_t)                       # (n, 1)
                    with torch.autograd.profiler.record_function("gen forward: Wt_up"):
                        H_time = self.Wt_up(t)                                             # (n, H_t)
                    with torch.autograd.profiler.record_function("gen forward: vt concatenation"):
                        vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                    with torch.autograd.profiler.record_function("gen forward: W_vt"):
                        inputs = self.W_vt(vt)
                    with torch.autograd.profiler.record_function("gen forward: last_t update"):
                        last_t = t
                    with torch.autograd.profiler.record_function("gen forward: append rw and ts"):
                        rw.append(v)
                        ts.append(t)
                        if debug_generator:
                            print(f"t shape: {t.shape}, vt shape: {vt.shape}, inputs shape: {inputs.shape}")
                with torch.autograd.profiler.record_function("gen forward: final lstmcell"):
                    hh, cc = self.lstmcell(inputs, hc)
                    if debug_generator:
                        print(f"Final hh shape: {hh.shape}, cc shape: {cc.shape}")
                
                with torch.autograd.profiler.record_function("gen forward: return training tensors"):
                    end = time.time()
                    print('time for training generator: ', end-start)
                    return  torch.stack(rw, dim=1), torch.stack(ts, dim=1), self.prob(hh)                  # (n, rw_len, N), (n, rw_len, 1)
        else: ## For graph editing
            if self.index == None:
                self.setup_index()
            with torch.autograd.profiler.record_function("gen forward: graph editing path"):
                # logging.info(f'inference {time.time()}')
                '''
                    we perform k step verifications
                '''
                nodes, times = input_rws
                with torch.autograd.profiler.record_function("gen forward: selections"):
                    num_batches = min(self.k, self.batch_size[0])
                    selections = torch.multinomial(torch.arange(self.batch_size[0], dtype=torch.float64), num_batches).to(self.device)
                    #regio DEBBUG
                    # if debug_generator:
                    print(f"selections shape: {selections.shape}")
                    #endregio
                with torch.autograd.profiler.record_function("gen forward: index_select"):
                    cur_walk_a = torch.index_select(nodes, 0, selections)  #to compute the p(rw) agains p(rw')
                    print('cur_walk_a.shape', cur_walk_a.shape)
                    cur_time_a = torch.index_select(times, 0, selections)
                    print('cur_time_a.shape', cur_time_a.shape)
                '''
                    we compute the initial rw score
                '''
                rs = []
                ts = []
                inputs_copy = inputs
                for rw_step in torch.arange(self.rw_len):
                    with torch.autograd.profiler.record_function("gen forward: lstmcell in graph editing"):
                        hh, cc = self.lstmcell(inputs_copy, hc)                                 # start with a fake input
                        hc = (hh, cc)                                                      # for the next cell step
                        #regio DEBBUG
                        if debug_generator:
                            print(f"hh shape: {hh.shape}, cc shape: {cc.shape}")
                        #endregio
                    with torch.autograd.profiler.record_function("gen forward: disten check in graph editing"):
                        if self.disten:
                            attn_map = torch.sigmoid(self.time_adapter(hh))
                            H_graph = attn_map * hh
                            H_time = (1 - attn_map) * hh
                            #regio DEBBUG
                            if debug_generator:
                                print(f"attn_map shape: {attn_map.shape}, H_graph shape: {H_graph.shape}, H_time shape: {H_time.shape}")
                            #endregio
                        else:
                            H_graph, H_time = hh, hh
                    with torch.autograd.profiler.record_function("gen forward: nodes and times indexing"):
                        v = nodes[:,rw_step,:]
                        #regio DEBBUG
                        if debug_generator:
                            print(f"v shape: {v.shape}")
                        #endregio
                    with torch.autograd.profiler.record_function("gen forward: W_down in graph editing"):
                        H_graph = self.W_down(v)                                           # (n, H)
                    with torch.autograd.profiler.record_function("gen forward: times indexing and view"):
                        t = times[:,rw_step,:].view(-1,1)
                    with torch.autograd.profiler.record_function("gen forward: Wt_up in graph editing"):
                        H_time = self.Wt_up(t)                                             # (n, H_t)
                    with torch.autograd.profiler.record_function("gen forward: vt concatenation in graph editing"):
                        vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                    with torch.autograd.profiler.record_function("gen forward: W_vt in graph editing"):
                        inputs_copy = self.W_vt(vt)
                    with torch.autograd.profiler.record_function("gen forward: last_t update in graph editing"):
                        last_t = t
                    with torch.autograd.profiler.record_function("gen forward: append rs and ts in graph editing"):
                        rs.append(v)
                        ts.append(t)
        
                    #regio DEBBUG
                    if debug_generator:
                        print(f"t shape: {t.shape}, vt shape: {vt.shape}, inputs_copy shape: {inputs_copy.shape}")
                    #endregio
                hh, cc = self.lstmcell(inputs_copy, hc)
                rs = torch.stack(rs, dim=1)
                ts =  torch.stack(ts, dim=1)
                #regio DEBBUG
                if debug_generator:
                    print(f"Final rs shape: {rs.shape}, ts shape: {ts.shape}")
                #endregio
                real_rw_prob = torch.sigmoid(self.prob(hh))
                #regio DEBBUG
                if debug_generator:
                    print(f"real_rw_prob value: {real_rw_prob}")
                #endregio
                
                overall_rm_rw = []
                overall_add_rw = []
                
                # overall_rm_ts = []
                # overall_add_ts = []
                # for step in torch.arange(num_batches, device=self.device):
                rm_rw = []
                add_rw = []

                rm_ts = []
                add_ts = []
                
                indices = torch.argmax(cur_walk_a, dim=-1)
                print('indices.shape', indices.shape)
                indices_cpu = indices.clone().detach().cpu().numpy()
                time_indices = cur_time_a
                print('time_indices.shape', time_indices.shape)
                time_indices_cpu = time_indices.clone().detach().cpu().numpy()
                hc = (torch.index_select(hc_copy[0], 0, selections), torch.index_select(hc_copy[1], 0, selections))
                current_input = torch.index_select(inputs, 0, selections)
                cur_last_t = torch.index_select(last_t_copy, 0, selections)
                print('cur_last_t.shape', cur_last_t.shape)

                #regio DEBBUG
                if debug_generator:
                    print(f"indices shape: {indices.shape}, time_indices shape: {time_indices.shape}")
                #endregio
                # Generate the probability distribution of the first node
                rw_step = 0
                hh, cc = self.lstmcell(current_input, hc)
                hc = (hh, cc)
                print('hh.shape', hh.shape)
                # print('cc.shape', cc.shape)
                # print(hc[0][7][73])
                #regio DEBBUG
                # if debug_generator:
                #     print(f"hh shape: {hh.shape}, cc shape: {cc.shape}")
                #endregio
                H_graph, H_time = hh, hh
                # p = self.W_up(H_graph)  
                # if debug_generator:
                #     print(f"p shape: {p.shape}")
                v_true = F.one_hot(indices[:, rw_step], num_classes=self.N).to(self.device).double()
                print('v_true.shape', v_true.shape)
                t = self.time_decoder(H_time) 
                print('t.shape', t.shape)
                #regio DEBBUG
                if debug_generator:
                    print(f"t shape: {t.shape}")
                #endregio
                t = torch.cat([self.time_decoder.constraints(t[step], cur_last_t[step]) for step in torch.arange(num_batches)], dim=0)                  # (n, 1)
                print('t.shape', t.shape)
                # print(t)
                #regio DEBBUG
                if debug_generator:
                    print(f"t constrained shape: {t.shape}")
                #endregio
                t_true = time_indices[:, rw_step].double()
                print('t_true.shape', t_true.shape)

                H_graph = self.W_down(v_true) 
                print('H_graph.shape', H_graph.shape)
                #regio DEBBUG
                if debug_generator:
                    print(f"H_graph shape: {H_graph.shape}")
                #endregio
                H_time = self.Wt_up(t_true) 
                print(f"H_time shape: {H_time.shape}")

                #regio DEBBUG
                if debug_generator:
                    print(f"H_time shape: {H_time.shape}")
                #endregio
                vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                print(f"vt shape: {vt.shape}")
                #regio DEBBUG
                if debug_generator:
                    print(f"vt shape: {vt.shape}")
                #endregio
                current_input = self.W_vt(vt)
                print(f"current_input shape: {current_input.shape}")
                cur_last_t = time_indices[:, rw_step]
                print(f"cur_last_t shape: {cur_last_t.shape}")
                #regio DEBBUG
                if debug_generator:
                    print(f"v_true shape: {v_true.shape}, t_true shape: {t_true.shape}, vt shape: {vt.shape}, current_input shape: {current_input.shape}")
                #endregio
                for rw_step in torch.arange(1, self.rw_len, device=self.device): #KEY FORLOOP HERE COPIED 2 OR 3 TIMES
                    hh, cc = self.lstmcell(current_input, hc)
                    hc = (hh, cc)

                    H_graph, H_time = hh, hh

                    v_true = F.one_hot(indices[:, rw_step], num_classes=self.N).to(self.device).double()
                    print('v_true.shape', v_true.shape)
                    t = self.time_decoder(H_time)                                     # (n, 1)
                    print('t.shape', t.shape)

                    t = torch.cat([self.time_decoder.constraints(t[step], cur_last_t[step]) for step in torch.arange(num_batches)], dim=0)                  # (n, 1)
                    print('t.shape', t.shape)

                    t_true = time_indices[:, rw_step].double()
                    print('t_true.shape', t_true.shape)

                    v, top_candidate, rm_th = self.up_project(H_graph.detach(), indices[:, rw_step])

                    comparison_result = (v[torch.arange(num_batches), top_candidate] > self.add_th).detach().cpu().numpy()
                    print('comparison_result.shape', comparison_result.shape)
                    top_candidate = top_candidate[comparison_result].detach()
                    indices_cpu_filtered = indices_cpu[comparison_result, rw_step-1:rw_step+1]
                    time_indices_cpu_filtered = time_indices_cpu[comparison_result, rw_step-1:rw_step+1]
                    for i, val in enumerate(indices_cpu_filtered):
                        add_rw.append((indices_cpu_filtered[i, 0], top_candidate[i], time_indices_cpu_filtered[i, 0].item()))
                    # print(add_rw)
                    
                    comparison_result = (v[torch.arange(num_batches), indices[:, rw_step]] < rm_th).detach().cpu().numpy()

                    indices_cpu_filtered = indices_cpu[comparison_result, rw_step-1:rw_step+1]
                    time_indices_cpu_filtered = time_indices_cpu[comparison_result, rw_step-1:rw_step+1]
                    for i, val in enumerate(indices_cpu_filtered):
                        rm_rw.append((indices_cpu_filtered[i, 0], indices_cpu_filtered[i, 1], time_indices_cpu_filtered[i, 0].item()))
                    # print(set(rm_rw))

                    # if 1 == 1:
                    #     quit()
                    s = time.time()
                    with torch.autograd.profiler.record_function("gen forward: W_down in graph editing"):
                        H_graph = self.W_down(v_true) 
                    # logging.info(f'{rw_step} W_down {time.time() - s}')
                    #regio DEBBUG
                    if debug_generator:
                        print(f"H_graph shape: {H_graph.shape}")
                    #endregio
                    H_time = self.Wt_up(t_true) 
                    #regio DEBBUG
                    if debug_generator:
                        print(f"H_time shape: {H_time.shape}")
                    #endregio
                    vt = torch.cat((H_graph, H_time), dim=1)                           # (n, H + H_t)
                    #regio DEBBUG
                    if debug_generator:
                        print(f"vt shape: {vt.shape}")
                    #endregio
                    current_input = self.W_vt(vt)
                    cur_last_t = time_indices[:, rw_step]
                    print('cur_last_t.shape', cur_last_t.shape)
                    #regio DEBBUG
                    if debug_generator:
                        print(f"current_input shape: {current_input.shape}, cur_last_t shape: {cur_last_t.shape}")
                    #endregio
                hh, cc = self.lstmcell(current_input, hc)
                verified_rw_prob = torch.sigmoid(self.prob(hh))
                # print(verified_rw_prob)
                # quit()
                #regio DEBBUG
                if debug_generator:
                    print(f"verified_rw_prob value: {verified_rw_prob}")
                #endregio
                if True: #verified_rw_prob > real_rw_prob[step]:
                    #regio DEBBUG
                    if debug_generator:
                        print("accepted for the better prob:", verified_rw_prob, " vs ", real_rw_prob[step])
                    #endregio
                    
                    overall_rm_rw = overall_rm_rw + rm_rw
                    overall_add_rw = overall_add_rw + add_rw
            # quit()
                # logging.info(set(overall_rm_rw))
            return overall_rm_rw, overall_add_rw
        
    def up_project(self, H_graph, real_indices):
        if self.index:
            v, top_candidate, rm_th = self.index.up_project(H_graph, real_indices)
        else:
            p = self.W_up(H_graph)  
            v = self.gumbel_softmax_sample(p, self.temp).squeeze()
            rm_th = v.sort().values[:, int(v.shape[1] * 0.05)]
            top_candidate = torch.argmax(v, dim=1)  # index
        return v, top_candidate, rm_th
    
class Discriminator(nn.Module):
    def __init__(self, H_inputs, H, H_t, N, device='cpu'):
        '''
            H_inputs: input dimension
            H:        hidden state dimension
            H_t:      dimension of the time hidden embedding
            N:        Number of nodes in the graph to generate
            rw_len:   Length of random walks to generate
        '''
        super(Discriminator, self).__init__()
        self.W_vt = nn.Linear(H_inputs + H_t, H_inputs, bias=False, device=device).type(torch.float64)#.to(device)
        torch.nn.init.xavier_uniform_(self.W_vt.weight)
        self.lstmcell = nn.LSTMCell(H_inputs, H, device=device).type(torch.float64)#.to(device)
        self.lin_out = nn.Linear(H, 1, bias=True, device=device).type(torch.float64)#.to(device)
        torch.nn.init.xavier_uniform_(self.lin_out.weight)
        torch.nn.init.zeros_(self.lin_out.bias)
        self.Wt_up = nn.Linear(1, H_t, device=device).type(torch.float64)#.to(device)
        torch.nn.init.xavier_uniform_(self.Wt_up.weight)
        torch.nn.init.zeros_(self.Wt_up.bias)

        self.W_down = nn.Linear(N, H_inputs, bias=False, device=device).type(torch.float64)#.to(device)
        torch.nn.init.xavier_uniform_(self.W_down.weight)
        self.H = H
        self.H_t = H_t
        self.N = N
        self.H_inputs = H_inputs
        self.device = device


    def forward(self, v, t):
        with torch.autograd.profiler.record_function("disc forward: rw_len_calc"):
            rw_len = v.shape[1]
        # with torch.autograd.profiler.record_function("disc forward: reshape_v_to_nrw_len_N"):
            # v = v.view(-1, self.N)                             # [n*rw_len, N]
        with torch.autograd.profiler.record_function("disc forward: apply_W_down"):
            v = self.W_down(v)                                 # [n*rw_len, H_inputs]
        # with torch.autograd.profiler.record_function("disc forward: reshape_v_to_n_rw_len_H_inputs"):
            # v = v.view(-1, rw_len, self.H_inputs)              # [n, rw_len, H_inputs]
        with torch.autograd.profiler.record_function("disc forward: reshape_t_to_n_rw_len_1"):
            t = t.view(-1, rw_len, 1)                          # [n, rw_len, 1]
        with torch.autograd.profiler.record_function("disc forward: apply_Wt_up"):
            t = self.Wt_up(t)                                  # [n, rw_len, H_t]
        with torch.autograd.profiler.record_function("disc forward: concat_v_t"):
            vt = torch.cat((v, t), dim=2)                      # [n, rw_len, H_inputs + H_t]
        with torch.autograd.profiler.record_function("disc forward: apply_W_vt"):
            inputs = self.W_vt(vt)                             # [n, rw_len, H_inputs]
        with torch.autograd.profiler.record_function("disc forward: init_hidden"):
            hc = self.init_hidden(v.size(0))
        with torch.autograd.profiler.record_function("disc forward: lstmcell_loop"):
            for i in range(rw_len):
                hc = self.lstmcell(inputs[:, i, :], hc)
        with torch.autograd.profiler.record_function("disc forward: apply_lin_out"):
            pred = self.lin_out(hc[0])
        return pred

    def init_inputs(self, num_samples):
        weight = next(self.parameters()).data
        return weight.new(num_samples, self.H_inputs).zero_().type(torch.float64)

    def init_hidden(self, num_samples):
        weight = next(self.parameters()).data
        return (
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64),
            weight.new(num_samples, self.H).zero_().contiguous().type(torch.float64))



class TimeDecoder(nn.Module):
    def __init__(self, H, H_t, dropout_p=0.2, device='cpu'):
        super(TimeDecoder, self).__init__()
        self.Wt_down = nn.Linear(H, H_t).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_down.weight)
        torch.nn.init.zeros_(self.Wt_down.bias)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.Wt_pred = nn.Linear(H_t, 1, bias=False).type(torch.float64)
        torch.nn.init.xavier_uniform_(self.Wt_pred.weight)
        self.dropout_p = dropout_p
        self.device = device

    def forward(self, x):                                  # (n, H)
        self.dropout.train()
        x = torch.tanh(self.Wt_down(x))                    # (n, H_t)
        x = self.dropout(x)                                # (n, H_t)
        x = self.Wt_pred(x)                                # (n, 1)
        max_t = torch.ones(x.shape[0], 1).type(torch.float64).to(x.device)
        x = torch.clamp(x, max=max_t)                      # make sure x is less than 1
        return x

    def constraints(self, x, last_t, epsilon=1e-1):
        # print('x.shape', x.shape)
        # print('last_t.shape', last_t.shape)
        epsilon = torch.tensor(epsilon).type(torch.float64).to(x.device)
        max_t = torch.ones(x.shape[0], 1).type(torch.float64).to(x.device)
        # print('max_t.shape', max_t.shape)
        min_ = torch.min(x)
        # print('min_', min_)
        # print('min_.shape', min_.shape)
        # quit()
        x = torch.where(min_ < epsilon, x - min_, x)      # (n, 1)
        max_ = torch.max(x)
        x = torch.where(1. < max_, x / max_, x)
        x = torch.clamp(x, min=last_t, max=max_t)
        return x

class DyGAN_trainer():
    def __init__(
        self, data, max_iterations=20000, rw_len=16, induced=True, batch_size=128, H_gen=40, H_disc=30, H_node=128,
        H_t=12, disten=False, H_inp=128, z_dim=16, lr=0.0003, n_critic=2, gp_weight=10.0,
        betas=(.5, .9), l2_penalty_disc=5e-5, l2_penalty_gen=1e-7, temp_start=5.0,
        temp_decay=1-5e-5, min_temp=0.5, baselines_stats=None, frac_edits=0.01, device='cpu'):
        """Initialize DyGAN.
        :param data: CTDNE temporal data
        :param N: Number of nodes in the graph to generate
        :param max_iterations: Maximal iterations if the stopping_criterion is not fulfilled, defaults to 20000
        :param rw_len: Length of random walks to generate, defaults to 16
        :param batch_size: The batch size, defaults to 128
        :param H_gen: The hidden_size of the generator, defaults to 40
        :param H_disc: The hidden_size of the discriminator, defaults to 30
        :param H_t: The hidden_size of the time embedding, defaults to 12
        :param H_inp: Input size of the LSTM, defaults to 128
        :param z_dim: The dimension of the random noise that is used as input to the generator, defaults to 16
        :param lr: Learning rate for both generator and discriminator, defaults to 0.0003
        :param n_critic:  The number of discriminator iterations per generator training iteration, defaults to 3
        :param gp_weight: Gradient penalty weight for the Wasserstein GAN, defaults to 10.0
        :param betas: Decay rates of the Adam Optimizers, defaults to (.5, .9)
        :param l2_penalty_disc:  L2 penalty for the generator weights, defaults to 5e-5
        :param l2_penalty_gen: L2 penalty on the di10scriminator weights, defaults to 1e-7
        :param temp_start: The initial temperature for the Gumbel softmax, defaults to 5.0
        :param temp_decay: After each evaluation, the current temperature is updated as
                        `current_temp := max(temp_decay * current_temp, min_temp)`, defaults to 1-5e-5
        :param min_temp: The minimal temperature for the Gumbel softmax, defaults to 0.5
        """ 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)
        self.max_iterations = max_iterations
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.disten = disten
        """
        " change N to feature dimension for decoding
        """
        self.N = data.num_nodes
        logging.info(f"self.N = data.num_nodes:, {data.num_nodes}")
        self.E = data.num_edges
        self.frac_edits = frac_edits
        self.generator = DyGANGenerator(
            H_inputs=H_inp, batch_size=batch_size,H=H_gen, H_t=H_t, N=self.N, rw_len=rw_len,
            z_dim=z_dim, temp=temp_start, disten=disten, device=self.device).to(self.device)
        # self.discriminator = Discriminator(
        #     H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N,
        #     rw_len=rw_len).to(self.device)
        self.discriminator = Discriminator(
            H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N, device=self.device).to(self.device)
        self.G_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.l2_penalty_disc = l2_penalty_disc
        self.l2_penalty_gen =l2_penalty_gen
        self.temp_start = temp_start
        self.temp_decay = temp_decay
        self.min_temp = min_temp
        self.data = data
        # self.data.df_data = self.data.df_data.drop_duplicates(['src','tar', 't'], keep='last').drop(['label'], axis=1)
        self.data.df_data = self.data.df_data.drop_duplicates(['src','tar', 't'], keep='last')
        self.walker = CTDNE_random_walker(self.data, rw_len, batch_size)
        self.induced = induced
        self.critic_loss = []
        self.generator_loss = []
        self.running = True
        # self.gold_stats = utils.eval_temp_graph(self.data.df_data, self.N, self.data.unique_ts)
        self.baselines_stats = baselines_stats
        # for name in self.baselines_stats:
        #     print(f"{name} score: {utils.temp_score(self.gold_stats, self.baselines_stats[name]):.4f}")
        self.best_gen = DyGANGenerator(
            H_inputs=H_inp,batch_size=batch_size, H=H_gen, H_t=H_t, N=self.N, rw_len=rw_len,
            z_dim=z_dim, temp=temp_start, disten=disten, device=self.device).to(self.device)
        # self.best_disc = Discriminator(
        #     H_inputs=H_inp, H=H_disc, H_t=H_t, N=self.N,
        #     rw_len=rw_len).to(self.device)
        self.best_disc = Discriminator(
            H_inputs=H_inp,H=H_disc, H_t=H_t, N=self.N, device=self.device).to(self.device)
        self.best_gen.eval()
        self.best_disc.eval()
        self.best_score = 1000000

    def l2_regularization_G(self, G):
        total_l2 = 0.0

        param_list = [
            G.W_down.weight,
            G.W_up.weight,
            # G.W_up.bias,
            G.shared_intermediate.weight,
            G.shared_intermediate.bias,
            G.h_intermediate.weight,
            G.h_intermediate.bias,
            G.c_intermediate.weight,
            G.c_intermediate.bias,
            G.lstmcell.weight_ih,
            G.lstmcell.weight_hh,
            G.lstmcell.bias_ih,
            G.lstmcell.bias_hh,
            G.W_vt.weight,
            G.time_decoder.Wt_down.weight,
            G.time_decoder.Wt_down.bias,
            G.time_decoder.Wt_pred.weight,
            G.Wt_up.weight,
            G.Wt_up.bias,
        ]

        if self.disten:
            param_list += [
                G.time_adapter.l1.weight,
                G.time_adapter.l1.bias,
                G.time_adapter.l2.weight,
                G.time_adapter.l2.bias,
            ]

        for param in param_list:
            total_l2 += (param ** 2).sum() / 2

        return self.l2_penalty_gen * total_l2
    
    ### Keep both versions and assert at the bottom that they are equal.

        # regularizaation for the generator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in G.W_down.weight.to(self.device)]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.weight.to(self.device)]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in G.W_up.bias.to(self.device)]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in G.shared_intermediate.weight.to(self.device)]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in G.shared_intermediate.bias.to(self.device)]) ** 2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in G.h_intermediate.weight.to(self.device)]) ** 2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in G.h_intermediate.bias.to(self.device)]) ** 2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in G.c_intermediate.weight.to(self.device)]) ** 2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in G.c_intermediate.bias.to(self.device)]) ** 2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.weight_ih.to(self.device)]) ** 2 / 2)
        l2_11 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.weight_hh.to(self.device)]) ** 2 / 2)
        l2_12 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.bias_ih.to(self.device)]) ** 2 / 2)
        l2_13 = torch.sum(torch.cat([x.view(-1) for x in G.lstmcell.bias_hh.to(self.device)]) ** 2 / 2)
        l2_14 = torch.sum(torch.cat([x.view(-1) for x in G.W_vt.weight.to(self.device)]) ** 2 / 2)
        l2_15 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_down.weight.to(self.device)]) ** 2 / 2)
        l2_16 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_down.bias.to(self.device)]) ** 2 / 2)
        l2_17 = torch.sum(torch.cat([x.view(-1) for x in G.time_decoder.Wt_pred.weight.to(self.device)]) ** 2 / 2)
        l2_18 = torch.sum(torch.cat([x.view(-1) for x in G.Wt_up.weight.to(self.device)]) ** 2 / 2)
        l2_19 = torch.sum(torch.cat([x.view(-1) for x in G.Wt_up.bias.to(self.device)]) ** 2 / 2)
        l_rw = l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10 + l2_11 + l2_12 + l2_13
        l_gen = l2_14 + l2_15 + l2_16 + l2_17 + l2_18 + l2_19

        del l2_1
        del l2_2
        del l2_3
        del l2_4
        del l2_5
        del l2_6
        del l2_7
        del l2_8
        del l2_9
        del l2_10
        del l2_11
        del l2_12
        del l2_13
        del l2_14
        del l2_15
        del l2_16
        del l2_17
        del l2_18
        del l2_19

        if self.disten:
            l2_20 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l1.weight.to(self.device)]) ** 2 / 2)
            l2_21 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l1.bias.to(self.device)]) ** 2 / 2)
            l2_22 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l2.weight.to(self.device)]) ** 2 / 2)
            l2_23 = torch.sum(torch.cat([x.view(-1) for x in G.time_adapter.l2.bias.to(self.device)]) ** 2 / 2)
            l_gen += l2_20 + l2_21 + l2_22 + l2_23
            del l2_20
            del l2_21
            del l2_22
            del l2_23
        l2 = self.l2_penalty_gen * (l_rw + l_gen)

        assert total_l2 == l_rw + l_gen, "total_l2 != l_rw + l_gen"

        return l2

    def l2_regularization_D(self, D):
        param_list = [
            D.W_down.weight,
            D.lstmcell.weight_ih,
            D.lstmcell.weight_hh,
            D.lstmcell.bias_ih,
            D.lstmcell.bias_hh,
            D.lin_out.weight,
            D.lin_out.bias,
            D.W_vt.weight,
            D.Wt_up.weight,
            D.Wt_up.bias,
        ]
        total_l2 = 0.0
        for param in param_list:
            total_l2 += (param ** 2).sum() / 2
        return self.l2_penalty_disc * total_l2

        # regularizaation for the discriminator. W_down will not be regularized.
        l2_1 = torch.sum(torch.cat([x.view(-1) for x in D.W_down.weight.to(self.device)]) ** 2 / 2)
        l2_2 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.weight_ih.to(self.device)]) ** 2 / 2)
        l2_3 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.weight_hh.to(self.device)]) ** 2 / 2)
        l2_4 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.bias_ih.to(self.device)]) ** 2 / 2)
        l2_5 = torch.sum(torch.cat([x.view(-1) for x in D.lstmcell.bias_hh.to(self.device)]) ** 2 / 2)
        l2_6 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.weight.to(self.device)]) ** 2 / 2)
        l2_7 = torch.sum(torch.cat([x.view(-1) for x in D.lin_out.bias.to(self.device)]) ** 2 / 2)
        l2_8 = torch.sum(torch.cat([x.view(-1) for x in D.W_vt.weight.to(self.device)]) ** 2 / 2)
        l2_9 = torch.sum(torch.cat([x.view(-1) for x in D.Wt_up.weight.to(self.device)]) ** 2 / 2)
        l2_10 = torch.sum(torch.cat([x.view(-1) for x in D.Wt_up.bias.to(self.device)]) ** 2 / 2)
        l2 = self.l2_penalty_disc * (l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10)
        assert total_l2 == l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10, "total_l2 != l2_1 + l2_2 + l2_3 + l2_4 + l2_5 + l2_6 + l2_7 + l2_8 + l2_9 + l2_10"
        del l2_1
        del l2_2
        del l2_3
        del l2_4
        del l2_5
        del l2_6
        del l2_7
        del l2_8
        del l2_9
        del l2_10
        return l2

    def calc_gp(self, fake_inputs, real_inputs):
        # calculate the gradient penalty. For more details see the paper 'Improved Training of Wasserstein GANs'.
        alpha = torch.rand((self.batch_size, 1, 1), dtype=torch.float64, device=self.device)
        fake_ns, fake_ts= fake_inputs
        real_ns, real_ts = real_inputs
        # print(fake_inputs, real_inputs)
        ns_difference = fake_ns - real_ns
        ts_difference = fake_ts - real_ts
        # fs_difference = fake_labels - real_labels
        ns_interpolates = real_ns + alpha * ns_difference
        ts_interpolates = real_ts + alpha * ts_difference
        # fs_interpolates = real_labels + alpha * fs_difference
        interpolates = (ns_interpolates, ts_interpolates)

        # for tensor in interpolates:
            # tensor.requires_grad = True

        y_pred_interpolates = self.discriminator(*interpolates)
        gradients = grad(outputs=y_pred_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(y_pred_interpolates),
            create_graph=True,
            retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=[1, 2]))
        gradient_penalty = torch.mean((slopes - 1) ** 2)
        gradient_penalty = gradient_penalty * self.gp_weight
        return gradient_penalty

    def critic_train_iteration(self):
        with record_function("critic training"):
            self.discriminator.train()
            start = time.time()
            with record_function("critic: zero_grad"):
                self.D_optimizer.zero_grad()
            #we generate walks first for label info and feed such info to the generator matching cgan
            with record_function("critic: generate_walks"):
                real_ns, real_ts, real_labels = self.walker.walk()
            # create fake and real inputs
            with record_function("critic: process_real_inputs"):
                real_ns = one_hot(torch.tensor(real_ns, dtype=torch.long, device=self.device), num_classes=self.N).to(torch.float64)
                real_ts = torch.tensor(real_ts, device=self.device)
                real_ts = real_ts.unsqueeze(dim=2).type(torch.float64).to(self.device)
                real_labels = torch.tensor(real_labels).to(self.device)
                real_inputs = (real_ns, real_ts)
            with record_function("critic: sample_fake_inputs"):
                fake_inputs = self.generator.sample(self.batch_size)
            # (n, rw_len), # (n, rw_len)
            with record_function("critic: discriminator_fake"):
                y_pred_fake = self.discriminator(*fake_inputs)
            with record_function("critic: discriminator_real"):
                y_pred_real = self.discriminator(*real_inputs)
            with record_function("critic: calculate_gradient_penalty"):
                gp = self.calc_gp(fake_inputs, real_inputs)                  # gradient penalty
            with record_function("critic: compute_disc_cost"):
                disc_cost = torch.mean(y_pred_fake) - torch.mean(y_pred_real) + gp + self.l2_regularization_D(self.discriminator)
            with record_function("critic: backward"):
                disc_cost.backward()
            with record_function("critic: optimizer_step"):
                self.D_optimizer.step()
            end = time.time()
            self.discriminator.eval()
        print(f"took {end-start} seconds for one critic train iteration")
        return disc_cost.item()
        

    def generator_train_iteration(self):
        with record_function("generator training"):
            with record_function("gen: set_train_mode"):
                self.generator.train()
            with record_function("gen: zero_grad"):
                self.G_optimizer.zero_grad()
            with record_function("gen: sample_fake_inputs"):
                fake_inputs = self.generator.sample(self.batch_size)
            with record_function("gen: discriminator_fake"):
                y_pred_fake = self.discriminator(*fake_inputs)
            with record_function("gen: compute_gen_cost"):
                gen_cost = -torch.mean(y_pred_fake) + self.l2_regularization_G(self.generator)
            with record_function("gen: backward"):
                gen_cost.backward()
            with record_function("gen: optimizer_step"):
                self.G_optimizer.step()
            self.generator.eval()
        return gen_cost.item()

    def save_model(self):
        self.best_gen.load_state_dict(copy.deepcopy(self.generator.state_dict()))
        self.best_disc.load_state_dict(copy.deepcopy(self.discriminator.state_dict()))
        self.best_gen.eval()
        self.best_disc.eval()

    def load_model_from_best(self):
        self.generator.load_state_dict(copy.deepcopy(self.best_gen.state_dict()))
        self.discriminator.load_state_dict(copy.deepcopy(self.best_disc.state_dict()))

    def save_model_parameters(self, save_path):
        save_data = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            # Add any other relevant information here
        }
        torch.save(save_data, save_path)
        print(f"Model parameters saved to {save_path}")

    def load_model_parameters(self, load_path):
        loaded_data = torch.load(load_path)
        self.generator.load_state_dict(loaded_data['generator_state_dict'])
        self.discriminator.load_state_dict(loaded_data['discriminator_state_dict'])
        print(f"Model parameters loaded from {load_path}")
    
    def create_graph(self, data, i=0, visualize=False, update=False, edges_only=False, num_iterations=3, debug=False):
        self.generator.eval()
        self.generator.temp = 0.5
        rm_nodes = []
        rm_times = []
        
        add_nodes = []
        add_times = []
        
        # edges_per_batch = self.batch_size * (self.rw_len - 1)
        # num_iterations = int(int(num_edges / 2) / edges_per_batch) + 1
        edges = data#.drop(columns=['label'])
        ori_len = len(edges)
        i = 0
        add_all = []
        rm_all = []
        save_vals = [int(100 * 1.2**j) for j in range(100)]
        while (i < num_iterations) and ((len(edges) < ori_len * (1 + self.frac_edits)) and (len(edges) > ori_len * (1 - self.frac_edits))):
            i += 1
            rw, ts, _ = self.walker.walk()
            rw = one_hot(torch.tensor(rw).type(torch.long), num_classes=self.N).type(torch.float64).to(self.device)
            ts = torch.tensor(ts)
            ts = ts.unsqueeze(dim=2).type(torch.float64).to(self.device)
            rm_rw, add_rw = self.generator.graph_edit(self.batch_size,(rw, ts))
            
            if debug:
                print('Iteration', i, 'rm_rw:', rm_rw)
                print('Iteration', i, 'add_rw:', add_rw)
            
            print('start:', len(edges))
            # edges_add = pd.DataFrame(rm_rw, columns=['src', 'tar', 't'])
            add_all += add_rw.copy()
            rm_all += rm_rw.copy()
            if len(rm_rw) > 0:
                edges_rm = pd.DataFrame(rm_rw, columns=['src', 'tar', 't'])
                edges_rm_copy = edges_rm.copy()
                edges_rm_copy['src'] = edges_rm['tar']
                edges_rm_copy['tar'] = edges_rm['src']
                edges_rm = pd.concat([edges_rm, edges_rm_copy], axis=0).drop_duplicates(keep='last')
                edges_rm['label'] = 0
                print('edges')
                print(edges)
                print('edges_rm')
                print(edges_rm)
                print(len(edges_rm))
                edges = without(edges, edges_rm)
                # print("removed ", len(without(edges, edges_rm)), "edges")
                print(edges)
                print('after removal pre drop:', len(edges))
                edges = edges.drop_duplicates(keep="first")
                edges.reset_index(drop=True, inplace=True)
                print('after removal:', len(edges))
            if len(add_rw) > 0:
                edges_add = pd.DataFrame(add_rw, columns=['src', 'tar', 't'])
                edges_add['label'] = 0
                edges = pd.concat([edges, edges_add]).drop_duplicates(keep="first")
                print("added ", len(edges_add), "edges")
                print('after adding', len(edges))
                edges.reset_index(drop=True, inplace=True)
            # import sys
            # sys.exit(0)

            # if i in save_vals:
            #     edges.to_csv(f'/nobackup/users/sammit/edges_{ori_len}_{i}_{len(edges)}.csv', index=False)

            if i % 100 == 1:
                logging.info(f"Edit cycle {i}, len(edges): {len(edges)}")
                # import pickle
                # with open('my_list.pkl', 'wb') as file:
                #     pickle.dump([add_all, rm_all], file)
                
        if visualize:
            self.data.visualize(edges)
        

        
        # if len(edges) < ori_len:
        #     dummy = [(0,0,0)]*(ori_len - len(edges))
        #     dummy = pd.DataFrame(dummy, columns=['src', 'tar', 't'])
        #     edges = pd.concat([edges, dummy])
        # elif len(edges) > ori_len:
        #     dummy = [(0,0,0)]*(ori_len - len(edges))
        #     dummy = pd.DataFrame(dummy, columns=['src', 'tar', 't'])
        #     edges = pd.concat([edges, dummy])
        if edges_only:

            edges_copy = edges.copy()
            edges_copy['src'] = edges['tar']
            edges_copy['tar'] = edges['src']
            edges_bi = pd.concat([edges, edges_copy], axis=0).drop_duplicates(keep='last')
            edges_bi.reset_index(drop=True, inplace=True)

            return edges, edges_bi
        # if update:
        #     self.generator.temp = np.maximum(self.temp_start * np.exp(-(1 - self.temp_decay) * i), self.min_temp)
        #     if generated_score < self.best_score:
        #         self.best_score = generated_score
        #         self.save_model()
        # return edges

    def convert_edges(self, rw, ts):
        
        edges = utils.temp_walk2edge(rw, ts)
        edges = edges[:int(num_edges / 2), :]
        
        
        edges = pd.DataFrame(edges, columns=['src', 'tar', 't'])
        edges_copy = edges.copy()
        edges_copy['src'] = edges['tar']
        edges_copy['tar'] = edges['src']
        edges = pd.concat([edges, edges_copy], axis=0)
        # assert len(edges) == num_edges
        edges.reset_index(drop=True, inplace=True)
    
    
        edges = pd.DataFrame(edges, columns=['src', 'tar', 't'])
        edges_copy = edges_rm.copy()
        edges_copy['src'] = edges['tar']
        edges_copy['tar'] = edges['src']
        edges = pd.concat([edges, edges_copy], axis=0)
        # assert len(edges) == num_edges
        edges.reset_index(drop=True, inplace=True)
        return edges

    def eval_model(self, num_eval=20):
        self.load_model_from_best()
        scores = []
        for _ in range(num_eval):
            temp_stats, edges = self.create_graph(num_edges=self.E, i=0, visualize=False, update=False)
            score = utils.temp_score(self.gold_stats, temp_stats)
            scores.append(score)
        print(f"Average: {np.mean(scores):.4f}")
        print(f"Max    : {np.max(scores):.4f}")
        print(f"Min    : {np.min(scores):.4f}")
        print(f"Median : {np.median(scores):.4f}")
        return scores

    def plot_graph(self):
        if len(self.critic_loss) > 10:
            plt.plot(self.critic_loss[9::], label="Critic loss")
            plt.plot(self.generator_loss[9::], label="Generator loss")
        else:
            plt.plot(self.critic_loss, label="Critic loss")
            plt.plot(self.generator_loss, label="Generator loss")
        plt.legend()
        plt.show()

    
    def train(self, create_graph_every=2000, plot_graph_every=200):
        """
        create_graph_every: int, default: 2000
            Creates every nth iteration a graph from randomwalks.
        plot_graph_every: int, default: 2000
            Plots the lost functions of the generator and discriminator.
        """
        starting_time = time.time()
        # Start Training
        print("start training")
        # total_prob_weights = []
        for i in range(self.max_iterations):
            if self.running:
                # if i == 52:
                #     print('total_prob_weights:', total_prob_weights)
                #     import sys
                #     sys.exit(0)
                self.critic_loss.append(np.mean([self.critic_train_iteration() for _ in range(self.n_critic)]))
                self.generator_loss.append(self.generator_train_iteration())
                # total_prob_weight = torch.sum(self.generator.prob.weight).item()
                # total_prob_weights.append(self.generator.prob.weight)
                # print(f"Total probability weight: {total_prob_weight}")
                if i % 2 == 1:
                    update = f'iteration: {i}    critic: {self.critic_loss[-1]:.5f}    gen: {self.generator_loss[-1]:.5f}'
                    print(update)
                    logging.info(update)
                # if i % create_graph_every == create_graph_every - 1:
                #     self.create_graph(self.E, i , visualize=True, update=True)
                #     print(f'Took {(time.time() - starting_time)/60} minutes so far..')
                if plot_graph_every > 0 and (i + 1) % plot_graph_every == 0:
                    self.plot_graph()


