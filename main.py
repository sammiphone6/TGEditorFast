import argparse
import logging
import os
import sys
import time
import yaml
import torch
import pandas as pd
import pickle
import subprocess
import random
import threading
from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd import profiler as autograd_profiler
from collections.abc import Mapping, Container
from pympler import asizeof
from HTNE import *

### PARAMS
data_folder_path = '/data/scratch'

# task = 'Chase'
# task = 'HI-Tiny'
# task = 'LI-Tiny'
task = 'HI-Small'
# task = 'HI-Medium'
# task = 'HI-Large'
do_preprocessing = False
do_data_formatting = False

####
debug_train = False

do_data_pre = True
do_HTNE_pre = True

do_dygan_training = True
do_editing = True

do_data_post = False
do_HTNE_post = False
####

kerb = 'sammit'
t = [
    # 'chronological',
    'seconds',
][0]
timestamp = int(time.time())


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object with all its contents."""
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = sys.getsizeof(o)
    ids.add(id(o))

    if isinstance(o, bytes) or isinstance(o, bytearray):
        return r

    if isinstance(o, Container):
        if isinstance(o, Mapping):
            return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())
        return r + sum(d(x, ids) for x in o)

    return r

def logger_setup(args):
    # Setup logging
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, f"logs{timestamp} {args.desc}.log")),     ## log to local log file
            logging.StreamHandler(sys.stdout)          ## log also to stdout (i.e., print to screen)
        ]
    )
    logging.info('logger is set up')

def log_setup_variables(args):
    logging.info(f"Running in mode: {args.mode}")
    logging.info(f"Using embedding: {args.emb}")
    logging.info(f'task: {task}')
    logging.info(f'debug_train: {debug_train}')
    logging.info(f'do_preprocessing: {do_preprocessing}')
    logging.info(f'do_data_formatting: {do_data_formatting}')
    logging.info(f'do_data_pre: {do_data_pre}')
    logging.info(f'do_HTNE_pre: {do_HTNE_pre}')
    logging.info(f'do_dygan_training: {do_dygan_training}')
    logging.info(f'do_data_post: {do_data_post}')
    logging.info(f'do_HTNE_post: {do_HTNE_post}')
    logging.info(f'kerb: {kerb}')
    logging.info(f'faster: {faster}')
    logging.info(f't: {t}')
    logging.info(f'timestamp: {timestamp}')
    args.mode = 'slow'

def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_preliminary_files():
    config = load_config(f"config-{task}.yml")

    # Access the variables
    device = auto_select_device(config['device'], strategy='greedy')
    data_path = f'{data_folder_path}/{kerb}/{task}_Trans.csv' #config['data_path']
    if 'local_env' in str(os.environ) and task == 'Chase':
        data_path = '~/TGEditor/edges_lab.csv'
        
    # For Train_HTNE
    HTNE_args = argparse.Namespace(**config['HTNE'])

    # For Dynamic_Graph_Experiments
    TGEditor_args = argparse.Namespace(**config['TGEditor'])

    logging.info(f'HTNE_args: {HTNE_args}')
    logging.info(f'TGEditor_args: {TGEditor_args}')

    return config, device, data_path, HTNE_args, TGEditor_args

def reset_seeds(TGEditor_args):
    # set seeds        
    random.seed(TGEditor_args.seed)
    np.random.seed(TGEditor_args.seed)
    torch.manual_seed(TGEditor_args.seed)
    torch.cuda.manual_seed_all(TGEditor_args.seed)

def preprocess_df(data_path):
    if task == 'Chase':
        cleaned_df = pd.read_csv(data_path, sep=',')
    else:
        if do_preprocessing:
        
            start_time = time.time()
            df = pd.read_csv(data_path, sep=',')
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype('int64') // 10**9
        
            #####################
            if t == 'chronological':
                ## This does chronological timestamp ##
                unique_dates = sorted(df['Timestamp'].unique())
                date_mapping = {date: i for i, date in enumerate(unique_dates)}
                df['Timestamp'] = df['Timestamp'].map(date_mapping)
            if t == 'seconds':
                ## This does seconds timestamp (like in Multi-GNN format_kaggle_files.py) ##
                min_t = df['Timestamp'].min() - 10
                df['Timestamp'] = df['Timestamp'].map(lambda x: x - min_t)
            #####################
            
            df['fromAccIdStr'] = df["From Bank"].astype(str) + df['Account'].astype(str)
            df['toAccIdStr'] = df["To Bank"].astype(str) + df['Account.1'].astype(str)
            account_mapping = {value: idx for idx, value in enumerate(pd.concat([df['fromAccIdStr'], df['toAccIdStr']]).unique())}
            df['fromAccIdStr'] = df['fromAccIdStr'].map(account_mapping)
            df['toAccIdStr'] = df['toAccIdStr'].map(account_mapping)
        
            # Select only the desired columns
            df[['Timestamp', 'fromAccIdStr', 'toAccIdStr', 'Is Laundering']]
            cleaned_df = df[['fromAccIdStr', 'toAccIdStr', 'Timestamp', 'Is Laundering']]
            
            # Rename columns
            cleaned_df.columns = ['src', 'tar', 't', 'label']
            cleaned_df = cleaned_df[:-1] ##Last row sometimes has a NaN value
            cleaned_df.to_csv(f'{data_folder_path}/{kerb}/{task}_{t}_Trans_Preprocessed.csv', index=False)
            cleaned_df
            end_time = time.time()
            logging.info(f"Data preprocessed. Elapsed time: {int(end_time - start_time)} seconds")
        else:
            cleaned_df = pd.read_csv(f'{data_folder_path}/{kerb}/{task}_{t}_Trans_Preprocessed.csv', sep=',')
    
    logging.info('cleaned_df loaded')
    logging.info(f'len(cleaned_df) = {len(cleaned_df)}')

    if debug_train:
        # Keep all rows where 'label' == 1
        condition_met_df = cleaned_df[cleaned_df['label'] == 1]
        # Keep every 1/250th row where 'label' != 1
        condition_not_met_df = cleaned_df[cleaned_df['label'] != 1].iloc[::250, :]
        # Concatenate the two DataFrames and ignore the index
        final_df = pd.concat([condition_met_df, condition_not_met_df])
        # Sort the DataFrame by its index in ascending order
        final_df = final_df.sort_index()
        # Reset the index to ensure it's sequential and start from 0
        cleaned_df = final_df.reset_index(drop=True)

    cleaned_df = cleaned_df[cleaned_df['src'] != cleaned_df['tar']]
    cleaned_df.drop_duplicates(inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)
    return cleaned_df

def get_data_pre_augmentation(df, device, HTNE_args, pre=True, prev_idx=None):
    do = do_data_pre if pre else do_data_post
    if do:
        logging.info(f"Started {'Pre' if pre else 'Post'} data processing")
        start_time = time.time()
        data = HTNE_dataset(edge_list=df, neg_size=HTNE_args.neg_size, hist_len=HTNE_args.hist_len, node2idx=prev_idx, device=device)
        end_time = time.time()

        file_path = f'{data_folder_path}/{kerb}/model/Data_{"Pre" if pre else "Post"}_{args.emb}_{task}_{t}{"_Debug" if debug_train else ""}.txt'
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as output_file:
            pickle.dump(data, output_file)
        logging.info(f"Finished {'Pre' if pre else 'Post'} data processing")
        logging.info(f"Total data processing time is: {end_time-start_time}")
        # logging.info("data for HTNE_dataset computed and saved")
    else:
        with open(f'{data_folder_path}/{kerb}/model/Data_{"Pre" if pre else "Post"}_{args.emb}_{task}_{t}{"_Debug" if debug_train else ""}.txt', 'rb') as input_file:
            data = pickle.load(input_file)
        logging.info(f"data for HTNE_dataset {'Pre' if pre else 'Post'} loaded")

    return data

def train_embedding(df, device, TGEditor_args, HTNE_args, args, pre=True, prev_idx=None, prev_nodes=None):
    reset_seeds(TGEditor_args)
    data = get_data_pre_augmentation(df, device, HTNE_args, pre=pre, prev_idx=prev_idx)

    # print('size of data', asizeof.asizeof(data))
    dataloader_lp = load_data(data, batch_size=HTNE_args.batch_size)

    emb_path = f'{data_folder_path}/{kerb}/model/HTNE_{"Pre" if pre else "Post"}_{args.emb}_{task}_{t}{"_Debug" if debug_train else ""}.ckpt'
    do = do_HTNE_pre if pre else do_HTNE_post
    if do:
        logging.info(f"Started training {'Pre' if pre else 'Post'} embedding")
        start_time = time.time()
        model_htne = Htne(emb_size=int(HTNE_args.emb_size/(2 if args.emb == 'new' else 1)), node_dim=prev_nodes if prev_nodes else data.num_nodes, new=(args.emb == 'new')).to(device)
        optimizer = SparseAdam(model_htne.parameters(), lr=HTNE_args.lr, eps=1e-8)
        patience = HTNE_args.patience
        max_loss = -float('inf')
        for i in range(HTNE_args.epochs):
            if True: #with torch.autograd.profiler.record_function("train emb: setup epoch"):
                total_loss = 0
                total_samples = 0 
                model_htne.train()
                print(f"Now at {i + 1} out of {HTNE_args.epochs}.")
            if True: #with torch.autograd.profiler.record_function("train emb: train batch"):
                if faster:
                    indices_list = list(range(data.data_size))
                    random.shuffle(indices_list)
                    for j in range(0, data.data_size, HTNE_args.batch_size):
                        indices = indices_list[j:j+HTNE_args.batch_size]
                        # batch_size = min(HTNE_args.batch_size, data.data_size - j)
                        if True: #with torch.autograd.profiler.record_function("train emb: zero grad"):
                            optimizer.zero_grad()
                        if True: #with torch.autograd.profiler.record_function("train emb: load batch"): ## For LARGE DATASET if this takes a while check and see if the data is already on the device or needs to be moved to it. If it needs to be moved to it, maybe try doing it in larger batches?
                            src = data.idx2src_id[indices].to(device).type(torch.long).unsqueeze(dim=1)
                            tar = data.hist[indices, 0, 0].to(device).type(torch.long).unsqueeze(dim=1)
                            dat = data.hist[indices, 0, 1].to(device).type(torch.float).unsqueeze(dim=1)
                            hist_nodes = data.hist[indices, 1:, 0].to(device).type(torch.long)
                            hist_times = data.hist[indices, 1:, 1].to(device).type(torch.float)
                            hist_masks = data.hist[indices, 1:, 2].to(device).type(torch.long)
                            negs = data.negative_sampling(len(indices)).to(device).type(torch.long)

                        if True: #with torch.autograd.profiler.record_function("train emb: forward pass"):
                            loss = model_htne(src, tar, dat, hist_nodes,
                                hist_times, hist_masks, negs)
                            loss = loss.sum()
                        if True: #with torch.autograd.profiler.record_function("train emb: update loss"):
                            total_loss += loss.item()
                            total_samples += src.size(0)
                            avg_loss = total_loss / total_samples
                            if (j//HTNE_args.batch_size + 1) % 20 == 0:
                                print(f"epoch {i + 1}, batch {j//HTNE_args.batch_size + 1}: {avg_loss:.04f}")
                        if True: #with torch.autograd.profiler.record_function("train emb: backward pass"):
                            loss.backward()
                        if True: #with torch.autograd.profiler.record_function("train emb: optimizer step"):
                            optimizer.step()
                        if True: #with torch.autograd.profiler.record_function("train emb: check patience"):
                            if avg_loss < max_loss:
                                max_loss = avg_loss
                                # torch.save(model2.state_dict(), f'./models/htne_{args2.data}_{exp_ID}.pt')
                                patience = HTNE_args.patience
                            if patience == 0:
                                print("early stopping")
                                # break
                if not faster:
                    for j, batch in enumerate(dataloader_lp):
                        batch_size = min(HTNE_args.batch_size, data.data_size - j)
                        if True: #with torch.autograd.profiler.record_function("train emb: zero grad"):
                            optimizer.zero_grad()
                        if True: #with torch.autograd.profiler.record_function("train emb: load batch"):
                            src = batch['source'].to(device, non_blocking=True)
                            tar = batch['target'].to(device, non_blocking=True)
                            dat = batch['date'].to(device, non_blocking=True)
                            hist_nodes = batch['hist_nodes'].to(device, non_blocking=True)
                            hist_times = batch['hist_times'].to(device, non_blocking=True)
                            hist_masks = batch['hist_masks'].to(device, non_blocking=True)
                            negs = batch['negs'].to(device, non_blocking=True)
                            # sys.exit()

                            # print('src', src.shape)
                            # print('tar', tar.shape)
                            # print('dat', dat.shape)
                            # print('hist_nodes', hist_nodes.shape)
                            # print('hist_times', hist_times.shape)
                            # print('hist_masks', hist_masks.shape)
                            # print('negs', negs.shape)
                            # print('src', src)
                            # print('tar', tar)
                            # print('dat', dat)
                            # print('hist_nodes', hist_nodes)
                            # print('hist_times', hist_times)
                            # print('hist_masks', hist_masks)
                            # print('negs', negs)
                        if True: #with torch.autograd.profiler.record_function("train emb: forward pass"):
                            loss = model_htne(src, tar, dat, hist_nodes,
                                hist_times, hist_masks, negs)
                            loss = loss.sum()
                            # break
                        if True: #with torch.autograd.profiler.record_function("train emb: update loss"):
                            total_loss += loss.item()
                            total_samples += src.size(0)
                            avg_loss = total_loss / total_samples
                            if (j + 1) % 20 == 0:
                                print(f"epoch {i + 1}, batch {j + 1}: {avg_loss:.04f}")
                        if True: #with torch.autograd.profiler.record_function("train emb: backward pass"):
                            loss.backward()
                        if True: #with torch.autograd.profiler.record_function("train emb: optimizer step"):
                            optimizer.step()
                        if True: #with torch.autograd.profiler.record_function("train emb: check patience"):
                            if avg_loss < max_loss:
                                max_loss = avg_loss
                                # torch.save(model2.state_dict(), f'./models/htne_{args2.data}_{exp_ID}.pt')
                                patience = HTNE_args.patience
                            if patience == 0:
                                print("early stopping")
                                break
            model_htne.normalize()
            logging.info(f"epoch {i + 1}, loss: {avg_loss:.04f}")
        if not args.debug and not pre:
            emb_path = emb_path[:-5] + str(timestamp) + emb_path[-5:]
        if not (args.debug and not pre): ## if not (debugging and post)
            torch.save(model_htne.state_dict(), emb_path)

        end_time = time.time()
        logging.info(f"Finished training {'Pre' if pre else 'Post'} embedding")
        logging.info(f"Total embedding training time is: {end_time-start_time}")
    else:
        model_htne = Htne(emb_size=int(HTNE_args.emb_size/(2 if args.emb == 'new' else 1)), node_dim=prev_nodes if prev_nodes else data.num_nodes, new=(args.emb == 'new')).to(device)
        model_htne.load_state_dict(torch.load(emb_path))
        model_htne.eval()
        logging.info(f"data for HTNE_embedding {'Pre' if pre else 'Post'} loaded")
    
    model_htne.eval()

    if not args.debug and not pre:
        logging.info(f'emb_path: {emb_path}')
    return data, model_htne, emb_path

def convert_indices(cleaned_df, data):
    saved_converted_path = f'{data_folder_path}/{kerb}/{task}_old_{t}_Trans_Preprocessed_removed_idx_mapped.csv'
    # saved_converted_path = f'{data_folder_path}/{kerb}/{task}_{args.emb}_{t}_Trans_Preprocessed_removed_idx_mapped.csv'
    cleaned_df['src'] = cleaned_df['src'].map(data.node2idx)
    cleaned_df['tar'] = cleaned_df['tar'].map(data.node2idx)
    min_t = cleaned_df['t'].min()
    max_t = cleaned_df['t'].max()
    cleaned_df['t'] = cleaned_df['t'] - min_t
    cleaned_df['t'] = cleaned_df['t'] / (max_t - min_t)
    logging.info(f"cleaned_df converted with data.node2idx map")
    cleaned_df.to_csv(saved_converted_path, index=False)
    return cleaned_df, saved_converted_path

def dygan_training(data, device, args, TGEditor_args, model_htne_pre):
    reset_seeds(TGEditor_args)
    if do_dygan_training:
        logging.info(f"do_dygan_training started")
                
        # exp_ID = int(SystemRandom().random() * 100000)
        # model_name = f'{TGEditor_args.model}_{TGEditor_args.data}_{exp_ID}.pt'
        # save_path = os.path.join(TGEditor_args.model, model_name)
        # loading the dataset
        # print("Original graph:")
        # # data.visualize()
        start_time = time.time()

        if TGEditor_args.model == 'DyGAN':
            if args.mode == 'slow':
                trainer = DyGAN_trainer(
                    data, max_iterations=TGEditor_args.max_iteration,
                    rw_len=TGEditor_args.rw_len, batch_size=TGEditor_args.batch_size, H_gen=TGEditor_args.H_gen, H_t=TGEditor_args.H_t,
                    disten=TGEditor_args.disten, H_disc=TGEditor_args.H_disc, H_inp=TGEditor_args.H_inp, z_dim=TGEditor_args.z_dim, lr=TGEditor_args.lr,
                    n_critic=TGEditor_args.n_critic, gp_weight=TGEditor_args.gp_weight, betas=TGEditor_args.betas,
                    l2_penalty_disc=TGEditor_args.l2_penalty_disc, l2_penalty_gen=TGEditor_args.l2_penalty_gen,
                    temp_start=TGEditor_args.temp_start, baselines_stats={}, frac_edits=args.frac_edits, device=device)
            elif args.mode == 'fast':
                trainer = DyGAN_trainer(
                    data, node_embs=model_htne_pre.node_emb, max_iterations=TGEditor_args.max_iteration,
                    rw_len=TGEditor_args.rw_len, batch_size=TGEditor_args.batch_size, H_gen=TGEditor_args.H_gen, H_t=TGEditor_args.H_t,
                    disten=TGEditor_args.disten, H_disc=TGEditor_args.H_disc, H_inp=TGEditor_args.H_inp, z_dim=TGEditor_args.z_dim, lr=TGEditor_args.lr,
                    n_critic=TGEditor_args.n_critic, gp_weight=TGEditor_args.gp_weight, betas=TGEditor_args.betas,
                    l2_penalty_disc=TGEditor_args.l2_penalty_disc, l2_penalty_gen=TGEditor_args.l2_penalty_gen,
                    temp_start=TGEditor_args.temp_start, baselines_stats={}, frac_edits=args.frac_edits, device=device)
        end_time = time.time()
        logging.info(f"finished intialization, took {end_time-start_time} seconds to finish")
        start_time = time.time()

        trainer.train(
            create_graph_every=TGEditor_args.create_every,
            plot_graph_every=TGEditor_args.plot_every,)
        # trainer.eval_model(num_eval=args.num_eval)
        
        end_time = time.time()
        torch.save(trainer, f'{data_folder_path}/sammit/DyGAN_{task}_{args.mode}_{args.emb}_{t}{"_Debug" if debug_train else ""}.ckpt')
        logging.info(f"Dygan Training completed, took: {end_time - start_time} seconds")
    
    else:
        trainer = torch.load(f'{data_folder_path}/sammit/DyGAN_{task}_{args.mode}_{args.emb}_{t}{"_Debug" if debug_train else ""}.ckpt')

    logging.info(f"do_dygan_training completed. do_dygan_training was {do_dygan_training}")
    return trainer

def graph_editing(trainer, cleaned_df, TGEditor_args):
    # TGEditor_args.seed = 140000
    reset_seeds(TGEditor_args)
    if do_editing:
        if True:#try:
            start = time.time()
            logging.info("Graph editing started")
            edited_graph, edited_graph_bi = trainer.create_graph(cleaned_df.drop_duplicates(keep="first", inplace=False), i=0, visualize=False, update=False, edges_only=True, num_iterations=TGEditor_args.num_edits)
            print(f"{TGEditor_args.num_edits} edits made in {int(time.time() - start)} seconds")
            logging.info(f"{TGEditor_args.num_edits} edits made in {int(time.time() - start)} seconds")

            edited_graph = edited_graph.drop_duplicates(keep="first", inplace=False)
            # edited_graph = edited_graph.sort_values(by='t')
            edited_graph = edited_graph.reset_index(drop = True, inplace = False)

            edited_graph.to_csv(f'{data_folder_path}/{kerb}/{task}_{args.mode}_{args.emb}_{t}{"_Debug" if debug_train else ""}_Trans_Edited.csv', index=False)
            logging.info("edited_graph saved")

        # except Exception as e:
        #     logging.info(f"Failed due to exception: {e}")
    else:
        edited_graph = pd.read_csv(f'{data_folder_path}/{kerb}/{task}_{args.mode}_{args.emb}_{t}{"_Debug" if debug_train else ""}_Trans_Edited.csv')
        logging.info("edited_graph loaded")

    logging.info(f"cleaned_df: {len(cleaned_df)}, edited_graph: {len(edited_graph)}, num_edits: {TGEditor_args.num_edits}")
    return edited_graph

def save_profiler_data_csv(prof, filename="profiler_output.csv"):
    averages = prof.key_averages()
    with open(filename, "w") as f:
        f.write(averages.table(sort_by="self_cpu_time_total", row_limit=-1))  # row_limit=-1 to capture all rows
    # with open(filename, "w") as f:
        f.write(averages.table(sort_by="self_cuda_time_total", row_limit=-1))  # row_limit=-1 to capture all rows
  
def data_formatting(saved_converted_path, outpath_folder, filename, task):
    if do_data_formatting:
        logging.info("Formatting Data for Multi-GNN")
        command = [
            "python", "format.py",
            "--in_path", saved_converted_path, #f"{task}_TestTGEPre_1",
            "--out_path_folder", outpath_folder,
            "--out_path_filename", filename,
            "--task", task,
        ]
        try:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Subprocess output:", result.stdout)
            print("Subprocess errors:", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit status 1")
            print("Error message:", e.stderr)
            print("Output message:", e.stdout)
    else:
        pass

def main(args):


    # Setup logging
    logger_setup(args)
    log_setup_variables(args)

    config, device, data_path, HTNE_args, TGEditor_args = load_preliminary_files()

    # all_res = []
    # for j in range(20):
    #     x = torch.empty(int(4e8), device=device)
    #     res = []
    #     for i in range(20):
    #         candidates = torch.randint(0, int(4e8), (int(1e7),), device=device)
    #         candidates = torch.unique(candidates)
    #         y = x[candidates]
    #         # x = torch.empty(int(1e8), device=device)
    #         # x = x.clone()
    #         print(f'{i} subprocess', subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    #         res.append(torch.nn.functional.softmax(y[0], dim=0))
    #         # del x
    #         # torch.cuda.empty_cache()
    
    # TGEditor_args.num_edits = int(TGEditor_args.num_edits/16)

    if debug_train:
        HTNE_args.epochs = 1
        TGEditor_args.max_iteration = 3
    # TGEditor_args.num_edits = 100
        # TGEditor_args.max_iteration = 2
        # TGEditor_args.num_edits = 2
    # HTNE_args.epochs = 1
    # edge_thresh = [2539172, 1269586, 634793, 317396, 158698, 79349, 39674, 19837, 9918]

    cleaned_df = preprocess_df(data_path)
    # cleaned_df = pd.read_csv(f'/data/scratch/sammit/filtered_edges_threshold_{edge_thresh[0]}.csv', sep=',')
    
    data_pre, model_htne_pre, emb_path_pre = train_embedding(cleaned_df, device, TGEditor_args, HTNE_args, args, pre=True)
    cleaned_df, saved_converted_path = convert_indices(cleaned_df, data_pre)
    trainer = dygan_training(data_pre, device, args, TGEditor_args, model_htne_pre)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    edited_graph = graph_editing(trainer, cleaned_df, TGEditor_args)
    print(f"len(edited_graph): {len(edited_graph)}")
    # quit()

    # edited_graph = pd.read_csv('{data_folder_path}/sammit/edges_4390017_176372_4380834.csv')
    # edited_graph = pd.read_csv('{data_folder_path}/sammit/edges_4390017_70880_4385031.csv')
    # edited_graph = pd.read_csv('{data_folder_path}/sammit/edges_tiny_741493_146977_739992.csv')
    # edited_graph = pd.read_csv('{data_folder_path}/sammit/edges_tiny_741493_211647_739540.csv')
        # logging.info('At bottom of with profile as prof')  

    # logging.info("Profiling complete, preparing to log results...")
    # save_profiler_data_csv(prof, filename=f"profiler_output_{timestamp}.csv")
    # logging.info('profile data saved')

    # # time.sleep(1000)

    idx = {i: i for i in data_pre.node2idx.values()}
    num_nodes = data_pre.num_nodes

    if do_data_post or do_HTNE_post:
        data_post, model_htne_post, emb_path_post = train_embedding(edited_graph, device, TGEditor_args, HTNE_args, args, pre=False, prev_idx = idx, prev_nodes=num_nodes)

    ### Just add the path if you need to do more experiments with one of your past embeddings
    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Post_old_HI-Tiny_seconds1718233726.ckpt"
    # emb_path_post = '{data_folder_path}/sammit/model/HTNE_Post_old_HI-Tiny_seconds1718234356.ckpt'
    # emb_path_post = '{data_folder_path}/sammit/model/HTNE_Post_old_HI-Tiny_seconds1718303204.ckpt'
    # emb_path_pre = f"{data_folder_path}/sammit/model/HTNE_Pre_old_HI-Tiny_seconds.ckpt"
    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Post_old_HI-Tiny_seconds1719166867.ckpt"
    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Post_old_HI-Small_seconds1719166847.ckpt"

    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Post_old_HI-Tiny_seconds1719166867.ckpt"
    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Post_old_HI-Small_seconds1719166847.ckpt"

    # emb_path_post = emb_path_pre
    # emb_path_post = '/data/scratch/sammit/model/HTNE_Post_new_HI-Tiny_seconds1721650125.ckpt'
    # do_HTNE_pre = True
    # emb_path_pre = "/data/scratch/sammit/model/HTNE_Pre_old_HI-Small_seconds.ckpt"
    # emb_path_post = "/data/scratch/sammit/model/HTNE_Post_old_HI-Small_seconds1721990459.ckpt"
    # emb_path_post = "/home/sammit/temp_small.ckpt"
    # do_HTNE_pre = True
    # emb_path_post = f"{data_folder_path}/sammit/model/HTNE_Pre_old_{task}_seconds.ckpt"

    # emb_path_pre = "{data_folder_path}/sammit/model/HTNE_Pre_old_{task}_seconds.ckpt"
    # emb_path_post = "{data_folder_path}/sammit/model/HTNE_Pre_old_HI-Small_seconds.ckpt"
    if True:#not args.debug:
        if do_HTNE_pre:
            logging.info(f"emb_path_pre: {emb_path_pre}")
        logging.info(f"emb_path_post: {emb_path_post}")
        outpath_folder = saved_converted_path.split('.')[0]
        filename = 'formatted_transactions.csv'

        data_formatting(saved_converted_path, outpath_folder, filename, task)

        logging.info("Running Multi-GNN Experiments")
        # Run inference on multi-gnn to see
        os.chdir(os.path.expanduser('~/Multi-GNN'))

        # for pre_or_post in ['post']*args.tests:
        tests = ['pre', 'post'] if do_HTNE_pre else ['post']
        def run_command():
            # Start the subprocess and store the process object
            for pre_or_post in tests*args.tests:
                command = [
                    "python", "main.py",
                    "--data", f"{outpath_folder}/{filename}", #f"{task}_TestTGEPre_1",
                    "--model", "gin",
                    "--emb", args.emb,
                    "--emb_path", emb_path_post if pre_or_post == 'post' else emb_path_pre,
                    "--n_epochs", "250",
                    "--desc", f"{pre_or_post} {args.desc}",
                    "--timestamp", str(timestamp),
                    # "--ports", 
                    # "--ego",
                ]
                process = subprocess.Popen(command)
                # Wait for the process to complete
                process.wait()

        # Create threads to run the command in parallel
        threads = []
        processes = []
        for i in range(args.n_threads):
            thread = threading.Thread(target=run_command)
            threads.append(thread)
            thread.start()
            time.sleep(5)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Handle Ctrl+C by terminating all subprocesses
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            for process in processes:
                process.terminate()
            print("Terminated all subprocesses.")


    logging.info(f'Made it to end no problem. Args: {args}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--mode", type=str, default="fast", help="TGEditor fast or slow")
    parser.add_argument("--emb", type=str, default="old", help="Type of embedding to use")
    parser.add_argument("--desc", type=str, default="no desc given", help="Description for run")
    parser.add_argument("--debug", type=bool, default=False, help="Reduces iterations if debug is True")
    parser.add_argument("--tests", type=int, default=2, help="Number of tests to run")
    parser.add_argument("--n_threads", type=int, default=20, help="Number of threads to run when doing Multi-GNN experiments")
    parser.add_argument("--frac_edits", type=float, default=1, help="Fraction of total edges to edit")
    args = parser.parse_args()

    if args.mode == 'working':
        from TGEditor_normal_working import *
        # args.mode = 'slow'
    elif args.mode == 'slow':
        from TGEditor_normal import *
        # args.mode = 'slow'
    elif args.mode == 'par':
        from TGEditor_par import *
        # args.mode = 'slow'
    elif args.mode == 'refactored':
        from TGEditor_par_refactored import *
        # args.mode = 'slow'
    elif args.mode == 'fast':
        from TGEditor_fast import *
        # args.mode = 'slow'
    else:
        raise Exception("Incorrect args.mode: neither 'fast' nor 'slow'")

    main(args)



