# tp_linear.py
import math
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert out_features % world_size == 0, f"{out_features} not divisible by {world_size}"
        self.out_features_per_rank = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features_per_rank)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        tensor_list = [torch.empty_like(output) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, output)
        return torch.cat(tensor_list, dim=-1)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        assert in_features % world_size == 0, f"{in_features} not divisible by {world_size}"
        self.in_features_per_rank = in_features // world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        start = dist.get_rank() * self.in_features_per_rank
        end = (dist.get_rank() + 1) * self.in_features_per_rank
        x_part = x[..., start:end]
        output = torch.nn.functional.linear(x_part, self.weight)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            output += self.bias
        return output
