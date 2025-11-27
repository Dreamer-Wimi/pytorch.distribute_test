# tp_linear.py
import math
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert out_features % world_size == 0, "out_features must be divisible by world_size"
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.world_size == 1:
            return output
        tensor_list = [torch.empty_like(output) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, output)
        return torch.cat(tensor_list, dim=-1)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert in_features % world_size == 0, "in_features must be divisible by world_size"
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        start = self.rank * self.in_features_per_partition
        end = (self.rank + 1) * self.in_features_per_partition
        x_part = x[..., start:end]
        output = torch.nn.functional.linear(x_part, self.weight)
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            output = output + self.bias
        return output
