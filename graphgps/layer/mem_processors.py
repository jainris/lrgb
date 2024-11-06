import torch

from torch_geometric.graphgym.config import cfg
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor
from torch_geometric.data import Batch
from torch import Tensor
from torch_sparse import SparseTensor
from graphgps.layer.memory_module import MemoryModule
from graphgps.layer.memory_module import MemoryState

from torch.nn import functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv, gcn_norm
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_sparse import matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from graphgps.layer.memory_module import update_using_memory
from torch.nn.functional import leaky_relu
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.models.layer import LayerConfig
import torch_geometric.graphgym.register as register


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


class MemoryGCNConv_2(GCNConv):
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

        z = batch.x
        read_values, msg_recepients, next_mem_state = self.memory_module(
            z=z, batch=batch.batch, prev_state=prev_mem_state
        )

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


class MemoryGCNConv_3(torch.nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()

        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn

        self.layer = MemoryGCNConv_2(
            in_channels=layer_config.dim_in,
            out_channels=layer_config.dim_out,
            bias=layer_config.has_bias,
            **kwargs
        )

        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                torch.nn.BatchNorm1d(
                    layer_config.dim_out,
                    eps=layer_config.bn_eps,
                    momentum=layer_config.bn_mom,
                )
            )
        if layer_config.dropout > 0:
            layer_wrapper.append(
                torch.nn.Dropout(
                    p=layer_config.dropout, inplace=layer_config.mem_inplace
                )
            )
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act])
        self.post_layer = torch.nn.Sequential(*layer_wrapper)

    def forward(
        self,
        batch: Batch,
        prev_mem_state: Optional[MemoryState] = None,
    ) -> Tuple[Batch, MemoryState]:
        batch, mem_state = self.layer(batch, prev_mem_state=prev_mem_state)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch, mem_state


class MemoryGINEConv(GINEConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        memory_module: MemoryModule,
        nb_z_fts: int,
        mem_msg_fts: int,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        send_to_all: bool = False,
        **kwargs
    ):
        nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
        )
        super().__init__(nn, eps, train_eps, edge_dim, **kwargs)
        self.memory_module = memory_module
        self.fts_projection = torch.nn.Linear(in_channels, nb_z_fts)
        self.mem_msg_projection = torch.nn.Linear(mem_msg_fts, in_channels)
        self.send_to_all = send_to_all
        self.dropout = cfg.gnn.dropout
        self.residual = cfg.gnn.residual

    def forward(
        self,
        batch: Batch,
        prev_mem_state: Optional[MemoryState] = None,
    ) -> Tuple[Batch, MemoryState]:
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        z = self.fts_projection(batch.x)
        z = leaky_relu(z)
        read_values, msg_recepients, next_mem_state = self.memory_module(
            z=z, batch=batch.batch, prev_state=prev_mem_state
        )
        read_values = leaky_relu(read_values)
        read_values = self.mem_msg_projection(read_values)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            read_values=read_values,
            msg_recepients=msg_recepients,
            nb_nodes=batch.x.shape[0],
        )

        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        if self.residual:
            out = out + batch.x  # residual connection

        # return batch
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


class MemoryGATv2Conv(GATv2Conv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        memory_module: MemoryModule,
        nb_z_fts: int,
        mem_msg_fts: int,
        send_to_all: bool = False,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=cfg.gnn.heads,
            concat=False,
            **kwargs
        )
        self.memory_module = memory_module
        self.fts_projection = torch.nn.Linear(in_channels, nb_z_fts)
        self.mem_msg_projection = torch.nn.Linear(mem_msg_fts, in_channels)
        self.send_to_all = send_to_all

    def forward(
        self,
        batch: Batch,
        prev_mem_state: Optional[MemoryState] = None,
    ) -> Tuple[Batch, MemoryState]:
        x = batch.x
        edge_index = batch.edge_index
        return_attention_weights = False
        size = None

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        z = self.fts_projection(batch.x)
        z = leaky_relu(z)
        read_values, msg_recepients, next_mem_state = self.memory_module(
            z=z, batch=batch.batch, prev_state=prev_mem_state
        )
        read_values = leaky_relu(read_values)
        read_values = self.mem_msg_projection(read_values)
        read_values = torch.unsqueeze(read_values, 1)
        read_values = read_values.repeat((1, self.heads, 1))

        # propagate_type: (x: PairTensor)
        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            size=size,
            read_values=read_values,
            msg_recepients=msg_recepients,
            nb_nodes=batch.x.shape[0],
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        # if isinstance(return_attention_weights, bool):
        #     assert alpha is not None
        #     if isinstance(edge_index, Tensor):
        #         return out, (edge_index, alpha)
        #     elif isinstance(edge_index, SparseTensor):
        #         return out, edge_index.set_value(alpha, layout="coo")
        # else:
        #     return out

        # return batch
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


class GATv2Conv_graphgym(torch.nn.Module):
    """
    Graph Attention Network (GAT) layer
    """

    def __init__(self, layer_config, **kwargs):
        super().__init__()
        self.model = GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            heads=cfg.gnn.heads,
            concat=False,
            bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


register_layer("gatv2conv", GATv2Conv_graphgym)
