import time
import logging
from util import create_parser, set_seed, logger_setup
from data_loading import get_data
from training import train_gnn
from inference import infer_gnn
import json
import torch
import os
import numpy as np

def main():
    debug_directory = False#f'/nobackup/users/sammit/data_{time.time()}' #or set debug_directory = False to turn off
    if debug_directory:
        import os
        os.makedirs(debug_directory, exist_ok=True)
        
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config, debug_directory)

    if debug_directory:
        torch.save(tr_data, os.path.join(debug_directory, 'tr_data.pt'))
        torch.save(val_data, os.path.join(debug_directory, 'val_data.pt'))
        torch.save(te_data, os.path.join(debug_directory, 'te_data.pt'))
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        #Training
        logging.info(f"Running Training")        
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config, debug_directory)
    else:
        #Inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)

if __name__ == "__main__":
    main()
