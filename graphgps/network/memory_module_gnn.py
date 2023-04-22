import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.init import init_weights

from graphgps.layer.mem_processors import MemoryGCNConv
from graphgps.layer.mem_processors import MemoryGCNConv_2
from graphgps.layer.mem_processors import MemoryGINEConv
from graphgps.layer.memory_module import memory_module_factory


class MemoryGNN(torch.nn.Module):
    """
    General GNN model: encoder + stage + head
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.pre_mp = None
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        gnn_layers = []

        memory_module = memory_module_factory(
            memory_module=cfg.memory.module,
            output_size=cfg.memory.output_size,
            embedding_size=cfg.memory.embedding_size,
            memory_size=cfg.memory.memory_size,
            nb_heads=cfg.memory.nb_heads,
            aggregation_technique=cfg.memory.aggregation_technique,
            nb_z_fts=cfg.memory.nb_z_fts,
        )

        if cfg.gnn.layer_type == "gcnconv":
            layer_model = MemoryGCNConv
        elif cfg.gnn.layer_type == "gineconv":
            layer_model = MemoryGINEConv
        else:
            raise ValueError("Got unexpected layer type: {}".format(cfg.gnn.layer_type))

        for _ in range(cfg.gnn.layers_mp):
            gnn_layers.append(
                layer_model(
                    in_channels=dim_in,
                    out_channels=cfg.gnn.dim_inner,
                    memory_module=memory_module,
                    nb_z_fts=cfg.memory.nb_z_fts,
                    mem_msg_fts=cfg.memory.output_size,
                    send_to_all=cfg.memory.send_to_all,
                )
            )
            dim_in = cfg.gnn.dim_inner

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)

        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        batch = self.encoder(batch)
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)
        mem_state = None
        for module in self.gnn_layers:
            batch, mem_state = module(batch=batch, prev_mem_state=mem_state)
        batch = self.post_mp(batch)
        return batch


register_network("mem_gnn", MemoryGNN)


class MemoryGNN2(torch.nn.Module):
    """
    General GNN model: encoder + stage + head
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.pre_mp = None
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        gnn_layers = []

        memory_module = memory_module_factory(
            memory_module=cfg.memory.module,
            output_size=cfg.memory.output_size,
            embedding_size=cfg.memory.embedding_size,
            memory_size=cfg.memory.memory_size,
            nb_heads=cfg.memory.nb_heads,
            aggregation_technique=cfg.memory.aggregation_technique,
            nb_z_fts=cfg.memory.nb_z_fts,
        )

        for _ in range(cfg.gnn.layers_mp):
            gnn_layers.append(
                MemoryGCNConv_2(
                    in_channels=dim_in,
                    out_channels=cfg.gnn.dim_inner,
                    memory_module=memory_module,
                    nb_z_fts=cfg.memory.nb_z_fts,
                    mem_msg_fts=cfg.memory.output_size,
                    send_to_all=cfg.memory.send_to_all,
                )
            )
            dim_in = cfg.gnn.dim_inner

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)

        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        batch = self.encoder(batch)
        if self.pre_mp is not None:
            batch = self.pre_mp(batch)
        mem_state = None
        for module in self.gnn_layers:
            batch, mem_state = module(batch=batch, prev_mem_state=mem_state)
        batch = self.post_mp(batch)
        return batch


register_network("mem_gnn_2", MemoryGNN2)
