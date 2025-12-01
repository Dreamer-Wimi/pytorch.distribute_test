import math
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        # out_features 是总维度，这里计算当前 rank 分到的维度
        self.out_features_per_rank = out_features // world_size
        self.world_size = world_size
        self.rank = rank
        
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [batch, seq, in_features]
        # output: [batch, seq, out_features_per_rank]
        return torch.nn.functional.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, bias=True):
        super().__init__()
        # in_features 是总输入维度，RowParallel 对输入维度进行切分
        self.in_features_per_rank = in_features // world_size
        self.world_size = world_size
        self.rank = rank
        
        # 权重形状: [out_features, in_features_per_rank]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # 关键修正：假设 x 已经是切分后的形状 [batch, seq, in_features_per_rank]
        # 因为它通常来自上一个 ColumnParallel 层的输出
        
        # 1. 本地矩阵乘法
        # input: [batch, seq, in_features_per_rank]
        # weight: [out_features, in_features_per_rank]
        # output_parallel: [batch, seq, out_features]
        output_parallel = torch.nn.functional.linear(x, self.weight)
        
        # 2. All-Reduce 求和
        dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
        
        # 3. 加上 Bias (Bias 是在 All-Reduce 之后加，因为它没有被切分)
        if self.bias is not None:
            output_parallel += self.bias
            
        return output_parallel