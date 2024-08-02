import torch
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging
import datetime
import os
import subprocess
import numpy as np

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            # logging.info(f"mask: {mask}")

            # logging.info(f"batch.x: {batch.x}")

            # logging.info(f"batch.edge_index: {batch.edge_index}")

            # logging.info(f"batch.edge_attr: {batch.edge_attr}")

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            # logging.info(f"batch.edge_attr2: {batch.edge_attr}")

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)

            # logging.info(f"out: {out}")
            pred = out[mask]
            # logging.info(f"pred: {pred}")
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)
            # logging.info(f"loss: {loss}")

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        precision = precision_score(ground_truth, pred, average='macro')
        recall = recall_score(ground_truth, pred, average='macro')
        wandb.log({"f1/train": f1, "precision/train": precision, "recall/train": recall, "loss/train": total_loss / total_examples}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

        #evaluate
        val_metrics = evaluate_homo(val_loader, val_inds, model, val_data, device, args, loss_fn)
        te_metrics = evaluate_homo(te_loader, te_inds, model, te_data, device, args, loss_fn)

        wandb.log({"f1/validation": val_metrics['f1'], "precision/validation": val_metrics['precision'], "recall/validation": val_metrics['recall'], "loss/validation": val_metrics['loss']}, step=epoch)
        wandb.log({"f1/test": te_metrics['f1'], "precision/test": te_metrics['precision'], "recall/test": te_metrics['recall'], "loss/test": te_metrics['loss']}, step=epoch)
        logging.info(f'Validation - F1: {val_metrics["f1"]:.4f}, Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}')
        logging.info(f'Test - F1: {te_metrics["f1"]:.4f}, Precision: {te_metrics["precision"]:.4f}, Recall: {te_metrics["recall"]:.4f}')

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            wandb.log({"best_test_f1": te_metrics['f1'], "best_test_precision": te_metrics['precision'], "best_test_recall": te_metrics['recall']}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    
    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model




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
            # memory_raw[0] = 1000000
            # memory_raw[1] = 1000000
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



def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config, debug_directory = False):
    #set device
    device = auto_select_device('auto', strategy='greedy')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging
        name=f"{device} {args.model} {args.timestamp} {args.desc} {args.emb}_emb_style {args.data} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, debug_directory)
        
    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if debug_directory:
        # torch.save(tr_loader, os.path.join(debug_directory, 'tr_loader.pt'))
        # torch.save(val_loader, os.path.join(debug_directory, 'val_loader.pt'))
        # torch.save(te_loader, os.path.join(debug_directory, 'te_loader.pt'))
        
        # torch.save(tr_inds, os.path.join(debug_directory, 'tr_inds.pt'))
        # torch.save(val_inds, os.path.join(debug_directory, 'val_inds.pt'))
        # torch.save(te_inds, os.path.join(debug_directory, 'te_inds.pt'))
        
        # torch.save(model, os.path.join(debug_directory, 'model.pt'))
        # torch.save(optimizer, os.path.join(debug_directory, 'optimizer.pt'))
        # torch.save(loss_fn, os.path.join(debug_directory, 'loss_fn.pt'))
        
        # torch.save(args, os.path.join(debug_directory, 'args.pt'))
        # torch.save(config, os.path.join(debug_directory, 'config.pt'))
        # # torch.save(device, os.path.join(debug_directory, 'device.pt'))
        # logging.info("000")
        # logging.info(000)
        # sampled_indices = []

        # for batch in val_loader:
        #     # Your training code here
        #     # ...
        #     # Log the indices of the samples in the current batch
        #     sampled_indices.append(batch.indices)  # Adapt this line based on how your DataLoader exposes indices
        
        # torch.save(sampled_indices, os.path.join(debug_directory, 'val_data.pt'))
        # logging.info(111)
        # for batch in te_loader:
        #     # Your training code here
        #     # ...
        #     # Log the indices of the samples in the current batch
        #     sampled_indices.append(batch.indices)  # Adapt this line based on how your DataLoader exposes indices
            
        # torch.save(sampled_indices, os.path.join(debug_directory, 'te_data.pt'))
        # logging.info(222)
        # for batch in tr_loader:
        #     # Your training code here
        #     # ...
        #     # Log the indices of the samples in the current batch
        #     sampled_indices.append(batch.indices)  # Adapt this line based on how your DataLoader exposes indices

        # torch.save(sampled_indices, os.path.join(debug_directory, 'tr_data.pt'))
        # logging.info(333)
        # torch.save(sampled_indices, os.path.join(debug_directory, 'data_config.pt'))
        
        logging.info(f"Done saving to debug_directory: {debug_directory}")
        
    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()