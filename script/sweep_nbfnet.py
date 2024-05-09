import os
import sys
import copy
import math
import pprint
from itertools import islice
from functools import partial
import easydict

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import NBFNet, NBFNetInv, NBFNetEig, NBFNetDirect
from script.train_nbfnet import test, train_and_validate, get_model, separator 
from sheaf_theory.model_translator import translate_to_graph_rep_inv

import wandb

if __name__ == "__main__":

    with wandb.init(
        # set the wandb project where this run will be logged
        project="ultra",
    ) as wdb:
        cfg = easydict.EasyDict(wdb.config)

        torch.manual_seed(cfg.seed + util.get_rank())

        logger = util.get_root_logger()
        if util.get_rank() == 0:
            logger.warning("Random seed: %d" % cfg.seed)
            logger.warning("Config file: %s" % cfg)
            logger.warning(pprint.pformat(cfg))
        task_name = cfg.task["name"]
        dataset = util.build_dataset(cfg)
        device = util.get_device(cfg)
        
        train_data, valid_data, test_data = [dataset[0]], [dataset[1]], [dataset[2]]
        
        if "fast_test" in cfg.train:
            num_val_edges = cfg.train.fast_test
            if util.get_rank() == 0:
                logger.warning(f"Fast evaluation on {num_val_edges} samples in validation")
            short_valid = [copy.deepcopy(vd) for vd in valid_data]
            for graph in short_valid:
                mask = torch.randperm(graph.target_edge_index.shape[1])[:num_val_edges]
                graph.target_edge_index = graph.target_edge_index[:, mask]
                graph.target_edge_type = graph.target_edge_type[mask]
            
            short_valid = [sv.to(device) for sv in short_valid]

        train_data = [td.to(device) for td in train_data]
        valid_data = [vd.to(device) for vd in valid_data]
        test_data = [tst.to(device) for tst in test_data]

        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = [
            Data(
                edge_index=torch.cat([trg.target_edge_index, valg.target_edge_index, testg.target_edge_index], dim=1), 
                edge_type=torch.cat([trg.target_edge_type, valg.target_edge_type, testg.target_edge_type,]),
                num_nodes=trg.num_nodes).to(device)
            for trg, valg, testg in zip(train_data, valid_data, test_data)
        ]

        cfg.model['num_relation'] = train_data[0].num_relations[0]
        model = get_model(cfg, device, train_data[0])

        assert task_name == "MultiGraphPretraining", "Only the MultiGraphPretraining task is allowed for this script"

        cfg_copy = copy.deepcopy(cfg)

        logger.warning(separator)
        logger.warning("Evaluate on valid")
        test(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger)
        train_and_validate(cfg, model, train_data, valid_data if "fast_test" not in cfg.train else short_valid, 
                           filtered_data=filtered_data, batch_per_epoch=cfg.train.batch_per_epoch,
                           logger=logger)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        test(cfg, model, valid_data, filtered_data=filtered_data, dataset='valid', device=device, logger=logger)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        test(cfg, model, test_data, filtered_data=filtered_data, dataset='test', device=device, logger=logger)