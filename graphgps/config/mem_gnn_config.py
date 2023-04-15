from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def mem_gnn_cfg(cfg):
    # Use residual connections between the GNN layers.
    cfg.memory = 'memory'

    # example argument group
    cfg.memory = CN()

    cfg.memory.module = "priority_queue_v0"
    cfg.memory.output_size = 300
    cfg.memory.embedding_size = 300
    cfg.memory.memory_size = 8
    cfg.memory.nb_heads = 8
    cfg.memory.aggregation_technique = "max"
    cfg.memory.nb_z_fts = 300
    cfg.memory.send_to_all = True


register_config("mem_gnn", mem_gnn_cfg)
