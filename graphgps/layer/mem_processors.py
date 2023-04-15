import torch

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.data import Batch
from torch import Tensor
from torch_sparse import SparseTensor
from graphgps.layer.memory_module import MemoryModule
from graphgps.layer.memory_module import MemoryState

from torch_geometric.nn.conv.gcn_conv import GCNConv, gcn_norm
from torch_sparse import matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from graphgps.layer.memory_module import update_using_memory
from torch.nn.functional import leaky_relu


class MemoryGCNConv(GCNConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        memory_module: MemoryModule,
        nb_z_fts: int,
        mem_msg_fts: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = False,
        send_to_all: bool = False,
        **kwargs
    ):
        super().__init__(
            in_channels,
            out_channels,
            improved,
            cached,
            add_self_loops,
            normalize,
            bias,
            **kwargs
        )
        self.memory_module = memory_module
        self.fts_projection = torch.nn.Linear(in_channels, nb_z_fts)
        self.mem_msg_projection = torch.nn.Linear(mem_msg_fts, out_channels)
        self.send_to_all = send_to_all

    def forward(
        self,
        batch: Batch,
        edge_weight: OptTensor = None,
        prev_mem_state: Optional[MemoryState] = None,
    ) -> Tuple[Batch, MemoryState]:
        x = batch.x
        edge_index = batch.edge_index
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        self.improved,
                        self.add_self_loops,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        z = self.fts_projection(batch.x)
        z = leaky_relu(z)
        read_values, msg_recepients, next_mem_state = self.memory_module(
            z=z, batch=batch.batch, prev_state=prev_mem_state
        )
        read_values = leaky_relu(read_values)
        read_values = self.mem_msg_projection(read_values)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            read_values=read_values,
            msg_recepients=msg_recepients,
            nb_nodes=x.shape[0],
            size=None,
        )

        if self.bias is not None:
            out += self.bias

        batch.x = out

        return batch, next_mem_state

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        read_values: Tensor,
        msg_recepients: Tensor,
        nb_nodes: int,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        if ptr is not None:
            raise ValueError("Sparse tensors are not supported yet.")
        else:
            inputs, index = update_using_memory(
                msgs=inputs,
                index=index,
                read_values=read_values,
                message_recepients=msg_recepients,
                nb_nodes=nb_nodes,
                send_to_all=self.send_to_all,
            )
            return scatter(
                inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr
            )

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        raise NotImplementedError
