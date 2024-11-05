import torch
import abc
import torch.nn.functional as F

from typing import Any, Callable, List, Optional, Tuple, NamedTuple, Dict, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch import Tensor
from torch_sparse import SparseTensor

from torch_scatter import scatter

MemoryState = NamedTuple
SMALL_NUMBER = 1e-6


class MemoryModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def initial_state(
        self, batch_size: int, nb_nodes: int, hiddens: Tensor, **kwargs
    ) -> MemoryState:
        """Memory module method to intiialize the state

        Returns:
          A memory state.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self, z: Tensor, batch: Tensor, prev_state: MemoryState, **kwargs
    ) -> Tuple[Tensor, Tensor, MemoryState]:
        """Memory module inference step.

        Args:
          z: Hidden state of nodes of the processor.
          batch: Batch of each node
          prev_state: Previous state of the Memory Module.
          **kwargs: Extra kwargs.

        Returns:
          Output of memory module inference step as a 2-tuple of
          (node embeddings, next state of memory module).
        """
        pass


class PriorityQueueState(MemoryState):
    memory_values: Tensor
    # read_strengths: Tensor
    write_mask: Tensor
    bias_mask: Tensor


class PriorityQueueV0(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str,
        nb_z_fts: int,
        skip_output_proj: bool,
        skip_value_proj: bool,
    ):
        super().__init__()
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

        if skip_value_proj:
            assert nb_z_fts == embedding_size, \
                "Expecting embedding size and input size to be the same for skipping value projection"
            self.value_proj = torch.nn.Identity()
        else:
            self.value_proj = torch.nn.Linear(nb_z_fts, self._embedding_size)
        if skip_output_proj:
            assert embedding_size == output_size, \
                "Expecting embedding size and output size to be the same for skipping output projection"
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Linear(self._embedding_size, self._output_size)

        self.a_1 = torch.nn.Linear(nb_z_fts, self.nb_heads)
        self.a_2 = torch.nn.Linear(nb_z_fts, self.nb_heads)

        self.heads_agg = None
        if self.nb_heads > 1:
            self.heads_agg = torch.nn.Linear(self.nb_heads, 1)

    def initial_state(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> PriorityQueueState:
        memory_values = torch.zeros(
            (batch_size, self._memory_size, self._embedding_size), device=device
        )
        write_mask = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=device), self._memory_size
        ).float()  # [B, S]
        bias_mask = torch.zeros((batch_size, self._memory_size), device=device)
        return PriorityQueueState(
            memory_values=memory_values, write_mask=write_mask, bias_mask=bias_mask
        )

    def __call__(
        self,
        z: Tensor,
        batch: Tensor,
        prev_state: Optional[PriorityQueueState],
        **kwargs
    ) -> Tuple[Tensor, Tensor, PriorityQueueState]:
        # z.shape: [N, F]
        nb_nodes, nb_z_fts = z.shape
        batch_size = torch.max(batch) + 1
        if prev_state is None:
            prev_state = self.initial_state(batch_size=batch_size, device=z.device)

        write_values = self.value_proj(z)  # [N, F']
        write_values = scatter(
            write_values, batch, dim=0, dim_size=batch_size, reduce="add"
        )
        # write_values = torch.sum(write_values, dim=1)
        write_values = torch.tanh(write_values)

        att_1 = torch.unsqueeze(self.a_1(z), dim=-1)  # [N, H, 1]
        att_2 = self.a_2(prev_state.memory_values)  # [B, S, H]
        att_2 = att_2[batch]  # [N, S, H]

        # [B, H, N, 1] + [B, H, 1, S]  = [B, H, N, S]
        # logits = torch.permute(att_1, (0, 2, 1, 3)) + torch.permute(att_2, (0, 2, 3, 1))
        # [H, N, 1] + [H, N, S]  = [H, N, S]
        logits = torch.permute(att_1, (1, 0, 2)) + torch.permute(att_2, (2, 0, 1))

        # Masking out the unwritten memory cells
        bias_mat = (prev_state.bias_mask - 1) * 1e9  # [B, S]
        bias_mat = bias_mat[batch]  # [N, S]
        bias_mat = torch.unsqueeze(bias_mat, 0)  # [1, N, S]

        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)  # [H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = torch.permute(coefs, (1, 2, 0))  # [N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = self.heads_agg(coefs)  # [N, S, 1]
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]
            coefs = F.softmax(coefs, dim=-1)  # [N, S]
        else:
            # Single-head
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]

        if self.aggregation_technique == "max":
            max_att_idx = torch.argmax(
                coefs,
                dim=-1,
            )  # [N]
            output = prev_state.memory_values[batch, max_att_idx]  # [N, F']
        elif self.aggregation_technique == "weighted":
            val = prev_state.memory_values[batch]  # [N, S, F']
            coefs = torch.unsqueeze(coefs, dim=2)  # [N, S, 1]
            output = coefs * val  # [N, S, F']
            output = torch.sum(output, dim=1)  # [N, F']
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        output = self.output_proj(output)
        message_recepients = torch.arange(nb_nodes, device=output.device)  # [N]

        new_memory_values = prev_state.memory_values + torch.unsqueeze(
            write_values, dim=1
        ) * torch.unsqueeze(prev_state.write_mask, dim=2)
        new_write_mask = torch.roll(prev_state.write_mask, shifts=1, dims=1)

        new_bias_mask = torch.minimum(
            (prev_state.bias_mask + prev_state.write_mask),
            torch.ones((1), device=prev_state.bias_mask.device),
        )

        return (
            output,
            message_recepients,
            PriorityQueueState(
                memory_values=new_memory_values,
                write_mask=new_write_mask,
                bias_mask=new_bias_mask,
            ),
        )


class PriorityQueueV1(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str,
        nb_z_fts: int,
        skip_output_proj: bool,
        skip_value_proj: bool,
    ):
        super().__init__()
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

        if skip_value_proj:
            assert nb_z_fts == embedding_size, \
                "Expecting embedding size and input size to be the same for skipping value projection"
            self.value_proj = torch.nn.Identity()
        else:
            self.value_proj = torch.nn.Linear(nb_z_fts, self._embedding_size)
        if skip_output_proj:
            assert embedding_size == output_size, \
                "Expecting embedding size and output size to be the same for skipping output projection"
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Linear(self._embedding_size, self._output_size)

        self.a_1 = torch.nn.Linear(nb_z_fts, self.nb_heads)
        self.a_2 = torch.nn.Linear(nb_z_fts, self.nb_heads)

        self.heads_agg = None
        if self.nb_heads > 1:
            self.heads_agg = torch.nn.Linear(self.nb_heads, 1)

    def initial_state(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> PriorityQueueState:
        memory_values = torch.zeros(
            (batch_size, self._memory_size, self._embedding_size), device=device
        )
        write_mask = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=device), self._memory_size
        ).float()  # [B, S]
        bias_mask = torch.zeros((batch_size, self._memory_size), device=device)
        return PriorityQueueState(
            memory_values=memory_values, write_mask=write_mask, bias_mask=bias_mask
        )

    def __call__(
        self,
        z: Tensor,
        batch: Tensor,
        prev_state: Optional[PriorityQueueState],
        **kwargs
    ) -> Tuple[Tensor, Tensor, PriorityQueueState]:
        # z.shape: [N, F]
        nb_nodes, nb_z_fts = z.shape
        batch_size = torch.max(batch) + 1
        if prev_state is None:
            prev_state = self.initial_state(batch_size=batch_size, device=z.device)

        write_values = self.value_proj(z)  # [N, F']
        write_values = scatter(
            write_values, batch, dim=0, dim_size=batch_size, reduce="add"
        )  # [S, F']
        # write_values = torch.sum(write_values, dim=1)
        write_values = torch.tanh(write_values)

        att_1 = torch.unsqueeze(self.a_1(z), dim=-1)  # [N, H, 1]
        att_2 = self.a_2(prev_state.memory_values)  # [B, S, H]
        att_2 = att_2[batch]  # [N, S, H]

        # [B, H, N, 1] + [B, H, 1, S]  = [B, H, N, S]
        # logits = torch.permute(att_1, (0, 2, 1, 3)) + torch.permute(att_2, (0, 2, 3, 1))
        # [H, N, 1] + [H, N, S]  = [H, N, S]
        logits = torch.permute(att_1, (1, 0, 2)) + torch.permute(att_2, (2, 0, 1))

        # Masking out the unwritten memory cells
        bias_mat = (prev_state.bias_mask - 1) * 1e9  # [B, S]
        bias_mat = bias_mat[batch]  # [N, S]
        bias_mat = torch.unsqueeze(bias_mat, 0)  # [1, N, S]

        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)  # [H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = torch.permute(coefs, (1, 2, 0))  # [N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = self.heads_agg(coefs)  # [N, S, 1]
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]
            coefs = F.softmax(coefs, dim=-1)  # [N, S]
        else:
            # Single-head
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]

        # Aggregating the attention values
        coefs = scatter(
            coefs, batch, dim=0, dim_size=batch_size, reduce="add"
        )  # [B, S]
        coefs = F.softmax(coefs, dim=-1)  # [B, S]
        coefs = coefs[batch]  # [N, S]

        if self.aggregation_technique == "max":
            max_att_idx = torch.argmax(
                coefs,
                dim=-1,
            )  # [N]
            output = prev_state.memory_values[batch, max_att_idx]  # [N, F']
        elif self.aggregation_technique == "weighted":
            val = prev_state.memory_values[batch]  # [N, S, F']
            coefs = torch.unsqueeze(coefs, dim=2)  # [N, S, 1]
            output = coefs * val  # [N, S, F']
            output = torch.sum(output, dim=1)  # [N, F']
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        output = self.output_proj(output)
        message_recepients = torch.arange(nb_nodes, device=output.device)  # [N]

        new_memory_values = prev_state.memory_values + torch.unsqueeze(
            write_values, dim=1
        ) * torch.unsqueeze(prev_state.write_mask, dim=2)
        new_write_mask = torch.roll(prev_state.write_mask, shifts=1, dims=1)

        new_bias_mask = torch.minimum(
            (prev_state.bias_mask + prev_state.write_mask),
            torch.ones((1), device=prev_state.bias_mask.device),
        )

        return (
            output,
            message_recepients,
            PriorityQueueState(
                memory_values=new_memory_values,
                write_mask=new_write_mask,
                bias_mask=new_bias_mask,
            ),
        )


class PriorityQueueStateV2(MemoryState):
    memory_values: Tensor
    read_strengths: Tensor
    write_mask: Tensor


class PriorityQueueV2(MemoryModule):
    def __init__(
        self,
        output_size: int,
        embedding_size: int,
        memory_size: int,
        nb_heads: int,
        aggregation_technique: str,
        nb_z_fts: int,
        skip_output_proj: bool,
        skip_value_proj: bool,
    ):
        super().__init__()
        self._output_size = output_size
        self._embedding_size = embedding_size
        self._memory_size = memory_size
        self.nb_heads = nb_heads
        self.aggregation_technique = aggregation_technique

        self.push_proj = torch.nn.Linear(nb_z_fts, 1)
        self.pop_proj = torch.nn.Linear(nb_z_fts, 1)
        if skip_value_proj:
            assert nb_z_fts == embedding_size, \
                "Expecting embedding size and input size to be the same for skipping value projection"
            self.value_proj = torch.nn.Identity()
        else:
            self.value_proj = torch.nn.Linear(nb_z_fts, self._embedding_size)
        if skip_output_proj:
            assert embedding_size == output_size, \
                "Expecting embedding size and output size to be the same for skipping output projection"
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Linear(self._embedding_size, self._output_size)

        self.a_1 = torch.nn.Linear(nb_z_fts, self.nb_heads)
        self.a_2 = torch.nn.Linear(self._embedding_size, self.nb_heads)

        self.heads_agg = None
        if self.nb_heads > 1:
            self.heads_agg = torch.nn.Linear(self.nb_heads, 1)

    def initial_state(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> PriorityQueueStateV2:
        memory_values = torch.zeros(
            (batch_size, self._memory_size, self._embedding_size), device=device
        )
        read_strengths = torch.zeros((batch_size, self._memory_size), device=device)
        write_mask = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=device), self._memory_size
        ).float()  # [B, S]
        return PriorityQueueStateV2(
            memory_values=memory_values,
            read_strengths=read_strengths,
            write_mask=write_mask,
        )

    def __call__(
        self,
        z: Tensor,
        batch: Tensor,
        prev_state: Optional[PriorityQueueStateV2],
        **kwargs
    ) -> Tuple[Tensor, Tensor, PriorityQueueStateV2]:
        # z.shape: [N, F]
        nb_nodes, nb_z_fts = z.shape
        if prev_state is None:
            batch_size = torch.max(batch) + 1
            prev_state = self.initial_state(batch_size=batch_size, device=z.device)
        batch_size = prev_state.memory_values.shape[0]

        push_strengths = self.push_proj(z)  # [N, 1]
        push_strengths = scatter(
            push_strengths, batch, dim=0, dim_size=batch_size, reduce="add"
        )  # [B, 1]
        push_strengths = F.sigmoid(push_strengths)  # [B, 1]

        pop_strengths = self.pop_proj(z)  # [N, 1]
        pop_strengths = F.sigmoid(pop_strengths)  # [N, 1]
        node_wise_pop_strengths = pop_strengths  # [N, 1]

        write_values = self.value_proj(z)  # [N, F']
        write_values = scatter(
            write_values, batch, dim=0, dim_size=batch_size, reduce="add"
        )
        write_values = torch.tanh(write_values)

        att_1 = torch.unsqueeze(self.a_1(z), dim=-1)  # [N, H, 1]
        att_2 = self.a_2(prev_state.memory_values)  # [B, S, H]
        att_2 = att_2[batch]  # [N, S, H]

        # [H, N, 1] + [H, N, S]  = [H, N, S]
        logits = torch.permute(att_1, (1, 0, 2)) + torch.permute(att_2, (2, 0, 1))

        # Masking out the unwritten memory cells
        unwritten_mask = prev_state.read_strengths < SMALL_NUMBER
        bias_mat = unwritten_mask * -1e9  # [B, S]
        bias_mat = bias_mat[batch]  # [N, S]
        bias_mat = torch.unsqueeze(bias_mat, 0)  # [1, N, S]

        coefs = F.softmax(F.leaky_relu(logits) + bias_mat, dim=-1)  # [H, N, S]
        # Projecting from the different heads into a single importance value
        coefs = torch.permute(coefs, (1, 2, 0))  # [N, S, H]
        if self.nb_heads > 1:
            # Multi-head
            coefs = self.heads_agg(coefs)  # [N, S, 1]
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]
            coefs = F.softmax(coefs, dim=-1)  # [N, S]
        else:
            # Single-head
            coefs = torch.squeeze(coefs, dim=2)  # [N, S]

        new_read_strengths = prev_state.read_strengths
        # Calculating the proportion of each element popped
        # (and their contribute to individual outputs)
        if self.aggregation_technique == "max":
            max_att_idx = torch.argmax(
                coefs,
                dim=-1,
            )  # [N]

            one_hot_max_att = F.one_hot(max_att_idx, self._memory_size)  # [N, S]
            pop_requested = node_wise_pop_strengths * one_hot_max_att  # [N, S]
        elif self.aggregation_technique == "weighted":
            pop_requested = node_wise_pop_strengths * coefs  # [N, S]
        else:
            raise ValueError(
                "Unknown memory aggregation technique " + self.aggregation_technique
            )
        total_pop_requested = scatter(
            pop_requested, batch, dim=0, dim_size=batch_size, reduce="add"
        )  # [B, S]
        recp_total_pop_requested = 1 / (total_pop_requested + SMALL_NUMBER)
        pop_proportions = pop_requested * recp_total_pop_requested[batch]  # [N, S]

        pop_given = torch.minimum(
            prev_state.read_strengths, total_pop_requested
        )  # [B, S]
        output_coefs = pop_proportions * pop_given[batch]  # [N, S]
        output = (
            torch.unsqueeze(output_coefs, dim=2) * prev_state.memory_values[batch]
        )  # [N, S, F']
        output = torch.sum(output, dim=1)  # [N, F']
        output = self.output_proj(output)
        message_recepients = torch.arange(nb_nodes, device=output.device)  # [N]

        new_read_strengths = new_read_strengths - pop_given

        new_memory_values = prev_state.memory_values + torch.unsqueeze(
            write_values, dim=1
        ) * torch.unsqueeze(prev_state.write_mask, dim=2)
        written_mask = new_read_strengths > SMALL_NUMBER
        new_read_strengths = new_read_strengths * written_mask
        new_read_strengths = new_read_strengths + push_strengths * prev_state.write_mask

        # Assuming memory_size > depth of GNN stack
        new_write_mask = torch.roll(prev_state.write_mask, shifts=1, dims=1)

        return (
            output,
            message_recepients,
            PriorityQueueStateV2(
                memory_values=new_memory_values,
                write_mask=new_write_mask,
                read_strengths=new_read_strengths,
            ),
        )


def memory_module_factory(
    memory_module: str,
    output_size: int,
    embedding_size: int,
    memory_size: int,
    nb_heads: int,
    aggregation_technique: str,
    nb_z_fts: int,
    skip_output_proj: bool,
    skip_value_proj: bool,
):
    if memory_module == "priority_queue_v0":
        return PriorityQueueV0(
            output_size=output_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            nb_heads=nb_heads,
            aggregation_technique=aggregation_technique,
            nb_z_fts=nb_z_fts,
            skip_output_proj=skip_output_proj,
            skip_value_proj=skip_value_proj,
        )
    elif memory_module == "priority_queue_v1":
        return PriorityQueueV1(
            output_size=output_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            nb_heads=nb_heads,
            aggregation_technique=aggregation_technique,
            nb_z_fts=nb_z_fts,
            skip_output_proj=skip_output_proj,
            skip_value_proj=skip_value_proj,
        )
    elif memory_module == "priority_queue_v2":
        return PriorityQueueV2(
            output_size=output_size,
            embedding_size=embedding_size,
            memory_size=memory_size,
            nb_heads=nb_heads,
            aggregation_technique=aggregation_technique,
            nb_z_fts=nb_z_fts,
            skip_output_proj=skip_output_proj,
            skip_value_proj=skip_value_proj,
        )
    else:
        raise ValueError("Unsupported memory module: {}".format(memory_module))


def update_using_memory(
    msgs: Tensor,
    index: Tensor,
    read_values: Tensor,
    message_recepients: Tensor,
    nb_nodes: int,
    send_to_all: bool = False,
) -> Tuple[Tensor, Tensor]:
    if send_to_all:
        assert read_values.shape[0] == nb_nodes
        # assert read_values == torch.arange(nb_nodes)
        read_values = torch.repeat_interleave(
            read_values, repeats=nb_nodes, dim=0
        )  # [N * N, F']
        message_recepients = torch.arange(nb_nodes).repeat(nb_nodes)  # [N * N]

    msgs = torch.cat([msgs, read_values], dim=0)
    index = torch.cat([index, message_recepients], dim=0)

    return msgs, index
