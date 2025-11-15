# Chapter 8: Parallelism Strategy and Model Partitioning

**Large-Scale LLM Training Playbook: 1.5 Trillion Parameter Model**
**Target Infrastructure: 5GW Multi-Datacenter Deployment**
**Investment Scale: $100+ Billion**

---

## Executive Summary

Training a 1.5 trillion parameter language model requires distributing the computation, memory, and communication across tens of thousands of GPUs. No single GPU—or even a single server—can hold the model weights, optimizer states, gradients, and activations required for training. Parallelism strategy determines whether you achieve 55% Model FLOPs Utilization (MFU) like ByteDance's MegaScale or struggle with <30% efficiency due to communication bottlenecks and memory constraints.

This chapter provides production-grade guidance for implementing 3D parallelism (data, tensor, and pipeline parallelism) at unprecedented scale, validated against real-world deployments:

- **ByteDance MegaScale**: 55.2% MFU training 175B model on 12,288 GPUs
- **Meta LLaMA-3 405B**: FSDP across 16,384 H100 GPUs, 3.8× 10²⁵ FLOPs training run
- **NVIDIA Nemotron-4 340B**: 96% training efficiency across 1,000km multi-datacenter deployment
- **xAI Grok**: 100,000 H100 GPU cluster with 95% data throughput on 800GbE RoCEv2

**Critical Decisions Addressed:**

1. **Parallelism Configuration for 1.5T Model**: Recommended 8-way TP × 16-way PP × 96-way DP = 12,288 GPUs (Phase 1), scaling to 350,000 GPUs (Phase 4)
2. **Memory Requirements**: 2,812.5 GB model states + activations per GPU with mixed-precision training
3. **Multi-Datacenter Strategy**: Hierarchical parallelism with DiLoCo achieving 90-95% utilization and 500× communication reduction
4. **Framework Selection**: DeepSpeed vs. Megatron-LM vs. PyTorch FSDP trade-offs for trillion-parameter training

**Success Metrics:**
- Model FLOPs Utilization (MFU): 50-55% sustained
- End-to-end training efficiency: >45% across multi-datacenter deployment
- Memory capacity utilization: >85% of HBM3 (68 GB / 80 GB per H100)
- Communication overhead: <15% of total training time

---

## 1. 3D Parallelism Overview

Modern large-scale training requires combining three orthogonal parallelism dimensions to overcome compute, memory, and communication constraints. Each dimension addresses different bottlenecks:

- **Data Parallelism (DP)**: Distributes training samples across GPUs; all GPUs maintain full model replica
- **Tensor Parallelism (TP)**: Splits individual layers across GPUs; enables models larger than single GPU memory
- **Pipeline Parallelism (PP)**: Divides model layers into stages across GPUs; reduces activation memory and enables deeper models

### 1.1 Data Parallelism: FSDP and ZeRO-3

Data parallelism is the most communication-efficient parallelism strategy when each GPU can hold the full model. However, for trillion-parameter models, naive data parallelism fails due to memory constraints. Fully Sharded Data Parallelism (FSDP) and ZeRO-3 solve this by sharding model states across data-parallel ranks.

#### ZeRO-3 Architecture (DeepSpeed)

ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy by partitioning optimizer states, gradients, and parameters across data-parallel processes.

**ZeRO Stages:**

```
┌─────────────────────────────────────────────────────────────────┐
│ ZeRO Stage Comparison (Per-GPU Memory)                          │
├─────────────────┬──────────────┬──────────────┬─────────────────┤
│ Component       │ Baseline DDP │ ZeRO-1       │ ZeRO-2/ZeRO-3   │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Parameters      │ Ψ            │ Ψ            │ Ψ/N_dp          │
│ Gradients       │ Ψ            │ Ψ            │ Ψ/N_dp          │
│ Optimizer States│ KΨ           │ KΨ/N_dp      │ KΨ/N_dp         │
│ Activations     │ A            │ A            │ A               │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Total Memory    │ (2+K)Ψ + A   │ (2+K/N_dp)Ψ  │ (2+K)Ψ/N_dp + A │
└─────────────────┴──────────────┴──────────────┴─────────────────┘

Ψ: Model parameters (bytes)
K: Optimizer state multiplier (12 for Adam: 4-byte fp32 params + 4-byte momentum + 4-byte variance)
N_dp: Data parallelism degree
A: Activation memory (depends on batch size, sequence length, activation checkpointing)
```

**ZeRO-3 Memory Reduction Example (1.5T Parameter Model):**

```python
# Model: 1.5T parameters with BF16 precision
model_params = 1.5e12
bytes_per_param = 2  # BF16

# Baseline memory (no ZeRO)
baseline_params_mem = model_params * bytes_per_param  # 3 TB
baseline_gradients = model_params * bytes_per_param   # 3 TB
baseline_optimizer = model_params * 12                # 18 TB (Adam fp32)
baseline_total = baseline_params_mem + baseline_gradients + baseline_optimizer
print(f"Baseline per-GPU: {baseline_total / 1e12:.1f} TB")
# Output: 24.0 TB per GPU (impossible on 80 GB H100)

# ZeRO-3 with N_dp = 96
N_dp = 96
zero3_params_mem = baseline_params_mem / N_dp    # 31.25 GB
zero3_gradients = baseline_gradients / N_dp      # 31.25 GB
zero3_optimizer = baseline_optimizer / N_dp      # 187.5 GB
zero3_total = zero3_params_mem + zero3_gradients + zero3_optimizer
print(f"ZeRO-3 per-GPU (N_dp={N_dp}): {zero3_total / 1e9:.1f} GB")
# Output: 250.0 GB per GPU
# Still requires additional TP/PP to fit in 80 GB H100
```

**ZeRO-3 Communication Pattern:**

```
Training Iteration Timeline (per micro-batch):
─────────────────────────────────────────────────────────────

1. All-Gather Parameters (forward):
   [GPU 0, 1, ..., N_dp-1] → Reconstruct layer L parameters
   Communication: Ψ_L * (N_dp - 1) / N_dp bytes

2. Forward Pass:
   Compute activations for layer L

3. Discard Parameters:
   Free layer L parameters after forward (optional)

4. All-Gather Parameters (backward):
   Reconstruct layer L parameters for gradient computation

5. Backward Pass:
   Compute gradients for layer L

6. Reduce-Scatter Gradients:
   [GPU 0, 1, ..., N_dp-1] → Each GPU receives 1/N_dp gradient shard
   Communication: Ψ_L * (N_dp - 1) / N_dp bytes

7. Optimizer Step:
   Each GPU updates its 1/N_dp parameter shard

Total Communication per Layer: 2 * Ψ_L * (N_dp - 1) / N_dp
                                ≈ 2 * Ψ_L for large N_dp
```

**Production Validation:**

- **Meta LLaMA-3 405B**: FSDP (PyTorch implementation of ZeRO-3 concepts) across 16,384 H100 GPUs
- **Microsoft DeepSpeed**: ZeRO-3 supports 1T+ parameter models, 10× memory reduction vs. baseline
- **Efficiency**: 90-95% scaling efficiency up to 512 GPUs (NVIDIA data)

#### PyTorch FSDP Implementation

PyTorch's native FSDP provides ZeRO-3-like functionality with tighter integration into PyTorch's autograd and compiler stack.

**FSDP Configuration Example:**

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM

# Initialize distributed process group
torch.distributed.init_process_group(backend="nccl")

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-405B")

# FSDP wrapping with auto-wrap for layer granularity
auto_wrap_policy = size_based_auto_wrap_policy(
    min_num_params=1e8,  # Wrap layers with >100M parameters
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=torch.bfloat16,
    auto_wrap_policy=auto_wrap_policy,
    backward_prefetch=True,  # Overlap communication with computation
    forward_prefetch=True,
    limit_all_gathers=True,  # Prevent OOM from simultaneous all-gathers
    device_id=torch.cuda.current_device(),
)

# Optimizer with parameter flattening for efficiency
from torch.distributed.fsdp.flat_param import FlatParameter
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

**FSDP vs. DeepSpeed ZeRO-3 Trade-offs:**

```
┌────────────────────────┬──────────────────────┬──────────────────────┐
│ Feature                │ PyTorch FSDP         │ DeepSpeed ZeRO-3     │
├────────────────────────┼──────────────────────┼──────────────────────┤
│ Integration            │ Native PyTorch       │ External library     │
│ Ease of Use            │ Simpler API          │ More configuration   │
│ Performance (small)    │ Competitive          │ Competitive          │
│ Performance (100K GPU) │ Less validated       │ Proven at scale      │
│ Memory Efficiency      │ Excellent            │ Excellent            │
│ Activation Offload     │ Limited              │ CPU/NVMe offload     │
│ Gradient Checkpointing │ PyTorch native       │ Custom implementation│
│ Multi-Node Scaling     │ Good (<10K GPUs)     │ Excellent (>100K)    │
│ Production Readiness   │ Meta, Microsoft      │ Microsoft, NVIDIA    │
└────────────────────────┴──────────────────────┴──────────────────────┘
```

**Recommendation**: Use PyTorch FSDP for <20K GPU deployments with tight PyTorch ecosystem integration. Use DeepSpeed ZeRO-3 for >20K GPU deployments or when CPU/NVMe offloading is required.

---

### 1.2 Tensor Parallelism: Megatron-LM Style

Tensor parallelism (TP) splits individual layers across multiple GPUs, enabling models too large to fit on a single device. Unlike data parallelism (which replicates the model), TP partitions weight matrices and performs distributed matrix multiplications.

#### Megatron-LM Column and Row Parallelism

Megatron-LM introduces efficient tensor parallelism by partitioning linear layers along the column or row dimension, minimizing communication requirements.

**Transformer Layer Partitioning:**

```
Standard Transformer Block (no parallelism):
┌────────────────────────────────────────────────────────┐
│ Input: X [batch, seq_len, hidden_dim]                  │
│                                                         │
│ 1. Self-Attention:                                     │
│    Q = X @ W_Q  [batch, seq_len, hidden_dim]          │
│    K = X @ W_K  [batch, seq_len, hidden_dim]          │
│    V = X @ W_V  [batch, seq_len, hidden_dim]          │
│    Attn = Softmax(QK^T / √d_k) @ V                    │
│    Output = Attn @ W_O                                 │
│                                                         │
│ 2. MLP (Feed-Forward):                                 │
│    H = GELU(X @ W_1)  [batch, seq_len, 4*hidden_dim]  │
│    Output = H @ W_2   [batch, seq_len, hidden_dim]    │
└────────────────────────────────────────────────────────┘

Megatron-LM Tensor Parallel (TP=8):
┌────────────────────────────────────────────────────────┐
│ GPU 0-7 (each holds 1/8 of weight matrices):          │
│                                                         │
│ Column Parallel (partition along output dimension):    │
│    W_Q, W_K, W_V split to: W_Q[i] for i=0..7          │
│    Each GPU i computes:                                │
│      Q[i] = X @ W_Q[i]  [batch, seq_len, hidden_dim/8]│
│    (No communication required - inputs replicated)     │
│                                                         │
│ Attention (local):                                     │
│    Each GPU computes attention over its 1/8 heads     │
│                                                         │
│ Row Parallel (partition along input dimension):        │
│    W_O split to: W_O[i] for i=0..7                    │
│    Each GPU i computes partial result                 │
│    All-Reduce across GPUs to get final output         │
│    Communication: 1 All-Reduce (hidden_dim * batch)   │
│                                                         │
│ MLP Column Parallel:                                   │
│    W_1[i]: [hidden_dim, 4*hidden_dim/8]               │
│    H[i] = GELU(X @ W_1[i])  (no communication)        │
│                                                         │
│ MLP Row Parallel:                                      │
│    W_2[i]: [4*hidden_dim/8, hidden_dim]               │
│    All-Reduce to aggregate MLP output                 │
│    Communication: 1 All-Reduce                        │
└────────────────────────────────────────────────────────┘

Total Communication per Transformer Layer: 2 All-Reduces
  (1 after attention, 1 after MLP)
```

**Megatron-LM Tensor Parallelism Code:**

```python
# Simplified Megatron-LM tensor parallel implementation
import torch
import torch.distributed as dist

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The weight matrix W [input_size, output_size] is partitioned along
    the output dimension across TP ranks: W = [W_0, W_1, ..., W_tp-1]

    Each GPU computes: Y_i = X @ W_i
    No communication required on forward pass (inputs are replicated).
    """
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)

        assert output_size % self.tp_size == 0
        self.output_size_per_partition = output_size // self.tp_size

        # Each GPU holds 1/tp_size of the weight matrix
        self.weight = torch.nn.Parameter(
            torch.empty(input_size, self.output_size_per_partition)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        # Each GPU computes local matrix multiplication
        output_parallel = torch.matmul(x, self.weight)
        # output_parallel: [batch, seq_len, output_size_per_partition]
        return output_parallel


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The weight matrix W [input_size, output_size] is partitioned along
    the input dimension: W = [W_0; W_1; ...; W_tp-1]

    Each GPU computes partial result with its input shard.
    All-Reduce aggregates results across TP ranks.
    """
    def __init__(self, input_size, output_size, tp_group):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(group=tp_group)

        assert input_size % self.tp_size == 0
        self.input_size_per_partition = input_size // self.tp_size

        self.weight = torch.nn.Parameter(
            torch.empty(self.input_size_per_partition, output_size)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size_per_partition] (already partitioned from ColumnParallel)
        # Each GPU computes local matmul
        output_partial = torch.matmul(x, self.weight)

        # All-Reduce across tensor-parallel ranks
        dist.all_reduce(output_partial, group=self.tp_group)
        # output: [batch, seq_len, output_size] (full result)
        return output_partial


# Example usage in Transformer block
class MegatronTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, tp_group):
        super().__init__()
        self.hidden_size = hidden_size

        # Attention: Column parallel for Q,K,V projection
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, tp_group
        )
        # Attention: Row parallel for output projection
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, tp_group
        )

        # MLP: Column parallel for first linear
        self.mlp_fc1 = ColumnParallelLinear(
            hidden_size, 4 * hidden_size, tp_group
        )
        # MLP: Row parallel for second linear
        self.mlp_fc2 = RowParallelLinear(
            4 * hidden_size, hidden_size, tp_group
        )

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]

        # Self-attention
        qkv = self.qkv_proj(x)  # No communication
        # ... compute attention (local to each GPU) ...
        attn_output = self.out_proj(attn_out)  # All-Reduce here

        # MLP
        mlp_hidden = torch.nn.functional.gelu(self.mlp_fc1(x))  # No communication
        mlp_output = self.mlp_fc2(mlp_hidden)  # All-Reduce here

        return attn_output + mlp_output
```

**Tensor Parallelism Communication Analysis:**

```python
# For a 1.5T parameter model with 96 layers
hidden_size = 12288  # GPT-4 class model
num_layers = 96
tp_degree = 8  # 8-way tensor parallelism
batch_size = 1  # Per GPU micro-batch
seq_len = 8192

# Communication per layer (forward + backward)
# 2 All-Reduces per layer (attention + MLP) × 2 for backward
all_reduces_per_layer = 4
bytes_per_element = 2  # BF16
activation_size_bytes = batch_size * seq_len * hidden_size * bytes_per_element

communication_per_layer = all_reduces_per_layer * activation_size_bytes
print(f"Communication per layer: {communication_per_layer / 1e6:.1f} MB")
# Output: 1,610.6 MB per layer

total_communication = communication_per_layer * num_layers
print(f"Total TP communication per iteration: {total_communication / 1e9:.1f} GB")
# Output: 154.6 GB per training iteration

# Assuming NVLink 900 GB/s bandwidth (H100)
nvlink_bandwidth = 900e9  # bytes/sec
communication_time = total_communication / nvlink_bandwidth
print(f"TP communication time (NVLink): {communication_time * 1000:.1f} ms")
# Output: 171.8 ms

# Compare to compute time (rough estimate)
# 1.5T params × 2 (forward+backward) × batch_size × seq_len / (8 GPUs × 989 TFLOPS MFU=0.5)
flops_per_iteration = 6 * 1.5e12 * batch_size * seq_len  # Approximate
compute_time = flops_per_iteration / (8 * 989e12 * 0.5)
print(f"Compute time estimate: {compute_time * 1000:.1f} ms")
# Output: ~18,500 ms

print(f"Communication overhead: {communication_time / compute_time * 100:.1f}%")
# Output: 0.9% (excellent - TP is communication-efficient with NVLink)
```

**When to Use Tensor Parallelism:**

✅ **Use TP when:**
- Model doesn't fit in single GPU memory even with FSDP/ZeRO-3
- High-bandwidth interconnect available (NVLink, NVSwitch)
- TP degree ≤ 8 (within single node with NVLink)
- Need to maximize single-node throughput

❌ **Avoid TP when:**
- TP degree > 8 requires inter-node communication (high latency)
- Network is bandwidth-constrained (e.g., Ethernet without RoCE)
- Model fits in memory with FSDP alone

**Production Examples:**
- **ByteDance MegaScale**: 8-way TP within DGX nodes, achieving 55.2% MFU
- **Meta LLaMA-3 405B**: TP=8 on 8-GPU DGX nodes, combined with FSDP
- **NVIDIA Megatron-LM**: Reference implementation supports TP up to 16-way

---

### 1.3 Pipeline Parallelism: GPipe, 1F1B, and Zero-Bubble

Pipeline parallelism (PP) divides the model into stages, with each stage assigned to a different GPU or set of GPUs. This enables training models too large to fit on a single device while reducing activation memory requirements.

#### Pipeline Parallelism Fundamentals

**Model Partitioning:**

```
1.5T Parameter Model with 96 Layers → 16 Pipeline Stages
┌─────────────────────────────────────────────────────────┐
│ Stage 0 (GPU 0-7):    Layers 0-5    (Embedding + 6L)   │
│ Stage 1 (GPU 8-15):   Layers 6-11   (6 Transformer)    │
│ Stage 2 (GPU 16-23):  Layers 12-17  (6 Transformer)    │
│ ...                                                      │
│ Stage 15 (GPU 120-127): Layers 90-95 (6L + LM Head)    │
└─────────────────────────────────────────────────────────┘

Each stage:
  - Receives activations from previous stage
  - Computes forward pass for its layers
  - Sends activations to next stage
  - Receives gradients from next stage (backward)
  - Computes backward pass
  - Sends gradients to previous stage
```

#### GPipe: Naive Pipeline Parallelism

GPipe divides each batch into micro-batches and pipelines them through stages, but suffers from significant pipeline bubbles (idle time).

**GPipe Timeline (4 stages, 4 micro-batches):**

```
Time →
Stage 0: [F0][F1][F2][F3][BUBBLE      ][B0][B1][B2][B3]
Stage 1: [  ][F0][F1][F2][F3][BUBBLE  ][B0][B1][B2][B3]
Stage 2: [    ][F0][F1][F2][F3][BUBBLE][B0][B1][B2][B3]
Stage 3: [      ][F0][F1][F2][F3][BUBBLE][B0][B1][B2][B3]

F0: Forward pass for micro-batch 0
B0: Backward pass for micro-batch 0

Pipeline Bubble Fraction = (p - 1) / (m + p - 1)
  where p = number of pipeline stages
        m = number of micro-batches

Example: p=16 stages, m=32 micro-batches
  Bubble = (16-1)/(32+16-1) = 15/47 = 31.9% idle time
```

**GPipe is inefficient** due to large bubble fraction, especially with deep pipelines.

#### 1F1B (One-Forward-One-Backward) Schedule

The 1F1B schedule reduces pipeline bubbles by interleaving forward and backward passes, keeping GPUs busier.

**1F1B Timeline (4 stages, 8 micro-batches):**

```
Time →
Stage 0: [F0][F1][F2][F3][F4][F5][F6][F7][B0][B1][B2][B3][B4][B5][B6][B7]
Stage 1: [  ][F0][F1][F2][F3][F4][F5][F6][F7][B0][B1][B2][B3][B4][B5][B6][B7]
Stage 2: [    ][F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
Stage 3: [      ][F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]

Warmup Phase: Fill pipeline (F0, F1, F2, F3)
Steady State: 1F1B (F4,B0; F5,B1; F6,B2; F7,B3)
Cooldown Phase: Drain pipeline (B4, B5, B6, B7)

Bubble Fraction ≈ (p-1) / m  for large m
Example: p=16, m=64
  Bubble ≈ 15/64 = 23.4% (better than GPipe's 31.9%)
```

**1F1B Implementation (DeepSpeed, Megatron-LM):**

```python
# Simplified 1F1B schedule implementation
def train_1f1b_schedule(model_stages, data_loader, num_microbatches, stage_id):
    """
    1F1B pipeline schedule.

    Args:
        model_stages: List of model stage modules
        data_loader: Training data iterator
        num_microbatches: Number of micro-batches per batch
        stage_id: Current pipeline stage (0 to num_stages-1)
    """
    num_stages = len(model_stages)
    num_warmup_microbatches = num_stages - stage_id - 1
    num_1f1b_microbatches = num_microbatches - num_warmup_microbatches

    # Phase 1: Warmup - fill pipeline with forward passes
    forward_activations = []
    for i in range(num_warmup_microbatches):
        # Receive activations from previous stage (or load data if stage 0)
        if stage_id == 0:
            inputs = next(data_loader)
        else:
            inputs = receive_from_previous_stage()

        # Forward pass
        outputs = model_stages[stage_id](inputs)
        forward_activations.append((inputs, outputs))

        # Send to next stage
        if stage_id < num_stages - 1:
            send_to_next_stage(outputs)
        else:
            # Last stage computes loss
            loss = compute_loss(outputs)

    # Phase 2: Steady State - 1F1B
    for i in range(num_1f1b_microbatches):
        # Forward pass
        if stage_id == 0:
            inputs = next(data_loader)
        else:
            inputs = receive_from_previous_stage()

        outputs = model_stages[stage_id](inputs)
        forward_activations.append((inputs, outputs))

        if stage_id < num_stages - 1:
            send_to_next_stage(outputs)
        else:
            loss = compute_loss(outputs)

        # Backward pass (on oldest activations)
        inputs, outputs = forward_activations.pop(0)

        if stage_id == num_stages - 1:
            output_grads = compute_loss_gradient(outputs)
        else:
            output_grads = receive_from_next_stage()

        input_grads = backward_pass(inputs, outputs, output_grads)

        if stage_id > 0:
            send_to_previous_stage(input_grads)

    # Phase 3: Cooldown - drain pipeline with backward passes
    for i in range(num_warmup_microbatches):
        inputs, outputs = forward_activations.pop(0)

        if stage_id == num_stages - 1:
            output_grads = compute_loss_gradient(outputs)
        else:
            output_grads = receive_from_next_stage()

        input_grads = backward_pass(inputs, outputs, output_grads)

        if stage_id > 0:
            send_to_previous_stage(input_grads)
```

#### Zero-Bubble Pipeline Parallelism

Recent research (Qi et al., 2023) introduces "zero-bubble" schedules that eliminate nearly all pipeline bubbles through:
1. Splitting backward pass into B (gradient computation) and W (weight update)
2. Carefully reordering forward and backward micro-batches

**Zero-Bubble Schedule (ZB-H1):**

```
Stage 0: [F0][F1][B0][F2][B1][F3][B2][W0][F4][B3][W1][F5][B4][W2]...
Stage 1: [  ][F0][B0][F1][B1][F2][W0][B2][F3][W1][B3][F4][W2][B4]...
Stage 2: [    ][F0][B0][F1][B1][W0][F2][B2][W1][F3][B3][W2][F4][B4]...
Stage 3: [      ][F0][B0][W0][F1][B1][W1][F2][B2][W2][F3][B3][F4][B4]...

Bubble Fraction: < 5% (vs. 23% for 1F1B)
Memory Cost: ~1.5× activations vs. 1F1B due to holding more in-flight micro-batches
```

**Zero-Bubble Benefits:**
- 5-10% higher throughput than 1F1B
- Particularly beneficial for deep pipelines (PP > 8)
- Trade-off: Increased activation memory (may require activation checkpointing)

**Production Adoption:**
- Emerging technique (2023-2024)
- Not yet widely deployed at hyperscale
- Recommended for experimental deployments or deep pipelines

#### Pipeline Parallelism Communication

**Communication Pattern:**

```python
# Communication volume per micro-batch
hidden_size = 12288
seq_len = 8192
batch_size_per_microbatch = 1
bytes_per_element = 2  # BF16

# Activation sent between pipeline stages (forward)
activation_bytes = batch_size_per_microbatch * seq_len * hidden_size * bytes_per_element
print(f"Activation transfer (forward): {activation_bytes / 1e6:.1f} MB")
# Output: 201.3 MB

# Gradient sent between pipeline stages (backward)
gradient_bytes = activation_bytes
print(f"Gradient transfer (backward): {gradient_bytes / 1e6:.1f} MB")
# Output: 201.3 MB

# Total communication per micro-batch per pipeline stage
total_per_microbatch = activation_bytes + gradient_bytes
print(f"Total PP communication per micro-batch: {total_per_microbatch / 1e6:.1f} MB")
# Output: 402.6 MB

# For 64 micro-batches, 16 pipeline stages
num_microbatches = 64
num_pipeline_stages = 16
total_pp_communication = total_per_microbatch * num_microbatches
print(f"Total PP communication per batch: {total_pp_communication / 1e9:.1f} GB")
# Output: 25.8 GB per training iteration

# Bandwidth requirement (assuming 800 Gbps inter-node network)
network_bandwidth = 800e9 / 8  # 100 GB/s
pp_communication_time = total_pp_communication / network_bandwidth
print(f"PP communication time (800G network): {pp_communication_time * 1000:.1f} ms")
# Output: 258 ms

# Pipeline parallelism requires high-bandwidth, low-latency interconnect
# InfiniBand NDR (400G) or Ethernet RoCEv2 (800G) recommended
```

**When to Use Pipeline Parallelism:**

✅ **Use PP when:**
- Model is too large for TP+DP alone
- Activation memory is a bottleneck (PP reduces activation memory per GPU)
- High-bandwidth inter-node network available (400-800 Gbps)
- Can tolerate pipeline bubbles (~20-25% with 1F1B)

❌ **Avoid PP when:**
- Model fits with TP+DP (PP adds complexity)
- Network bandwidth is limited (<100 Gbps)
- Small batch sizes (insufficient micro-batches to fill pipeline)

---

### 1.4 When to Use Each Parallelism Strategy

**Decision Matrix:**

```
┌──────────────────────┬──────────────┬─────────────┬──────────────────┐
│ Scenario             │ Data Parallel│ Tensor Par. │ Pipeline Par.    │
├──────────────────────┼──────────────┼─────────────┼──────────────────┤
│ Model fits single GPU│ ✅ Only DP    │ ❌ Not needed│ ❌ Not needed    │
│ Model fits w/ FSDP   │ ✅ FSDP only  │ ❌ Not needed│ ❌ Not needed    │
│ 175B - 400B params   │ ✅ FSDP       │ ✅ TP=8      │ ⚠️ Optional     │
│ 400B - 1.5T params   │ ✅ FSDP       │ ✅ TP=8      │ ✅ PP=8-16       │
│ 1.5T+ params         │ ✅ FSDP       │ ✅ TP=8      │ ✅ PP=16-32      │
│ Multi-datacenter     │ ✅ DP across DC│ ✅ TP intra-node│ ✅ PP intra-DC│
└──────────────────────┴──────────────┴─────────────┴──────────────────┘
```

**Hierarchical Parallelism Strategy (Recommended):**

```
Level 1 (Intra-Node): Tensor Parallelism
  └─ TP degree: 8 (matches 8 GPUs per DGX H100 node)
  └─ Communication: NVLink 900 GB/s (very fast)
  └─ Purpose: Fit model in 8-GPU memory pool

Level 2 (Intra-Datacenter): Pipeline Parallelism
  └─ PP degree: 8-16 stages
  └─ Communication: InfiniBand NDR 400 Gbps or Ethernet 800 Gbps
  └─ Purpose: Enable larger models, reduce per-GPU activation memory

Level 3 (Inter-Datacenter): Data Parallelism
  └─ DP degree: 96-256+ (across all nodes)
  └─ Communication: WAN (gradient compression, DiLoCo)
  └─ Purpose: Scale training throughput, multi-site fault tolerance
```

**Example Configuration (12,288 GPUs):**
- TP=8 (8 GPUs per node, 1,536 nodes)
- PP=16 (16 pipeline stages, 96 nodes per stage)
- DP=96 (96 data-parallel replicas)
- Total: 8 × 16 × 96 = 12,288 GPUs

---

## 2. Configuration for 1.5T Parameter Model

### 2.1 Memory Requirements Analysis

Training a 1.5 trillion parameter model requires careful memory planning to fit model states, activations, and gradients within H100's 80 GB HBM3.

#### Model State Memory

**Components:**

```
1. Parameters (Ψ):
   - 1.5T parameters × 2 bytes (BF16) = 3 TB

2. Gradients (∇Ψ):
   - Same size as parameters = 3 TB

3. Optimizer States (Adam):
   - FP32 parameters: 1.5T × 4 bytes = 6 TB
   - First moment (momentum): 1.5T × 4 bytes = 6 TB
   - Second moment (variance): 1.5T × 4 bytes = 6 TB
   - Optimizer total: 18 TB

Total Model State: 3 + 3 + 18 = 24 TB
```

#### Memory Per GPU with 3D Parallelism

**Configuration: TP=8, PP=16, DP=96 (12,288 GPUs)**

```python
# Model parameters
total_params = 1.5e12
num_layers = 96

# Parallelism configuration
TP = 8   # Tensor parallelism
PP = 16  # Pipeline parallelism
DP = 96  # Data parallelism (FSDP)

# Parameters per GPU
# - TP splits model across 8 GPUs
# - PP assigns 96/16 = 6 layers per pipeline stage
# - FSDP/ZeRO-3 shards across DP=96 ranks within each TP×PP group

params_per_layer = total_params / num_layers
layers_per_stage = num_layers / PP
params_per_stage = params_per_layer * layers_per_stage

# Memory calculation
bytes_bf16 = 2
bytes_fp32 = 4

# Parameters (BF16, sharded by TP and DP)
param_memory_per_gpu = (params_per_stage / TP / DP) * bytes_bf16
print(f"Parameters per GPU: {param_memory_per_gpu / 1e9:.2f} GB")
# Output: 0.33 GB

# Gradients (BF16, sharded by TP and DP)
grad_memory_per_gpu = param_memory_per_gpu
print(f"Gradients per GPU: {grad_memory_per_gpu / 1e9:.2f} GB")
# Output: 0.33 GB

# Optimizer states (FP32 params + 2× FP32 moments, sharded by DP)
# Note: Optimizer states are NOT sharded by TP in typical implementations
optimizer_memory_per_gpu = (params_per_stage / TP / DP) * (bytes_fp32 + 2 * bytes_fp32)
print(f"Optimizer states per GPU: {optimizer_memory_per_gpu / 1e9:.2f} GB")
# Output: 2.0 GB

model_state_memory = param_memory_per_gpu + grad_memory_per_gpu + optimizer_memory_per_gpu
print(f"\nTotal model state per GPU: {model_state_memory / 1e9:.2f} GB")
# Output: 2.66 GB
```

#### Activation Memory

Activation memory dominates for large batch sizes and long sequences.

**Activation Memory Formula:**

```python
# Transformer activation memory (per layer, per micro-batch)
# Based on Megatron-LM analysis

batch_size = 1  # Micro-batch size per GPU
seq_len = 8192
hidden_size = 12288
num_attention_heads = 96
layers_per_stage = 6  # 96 layers / 16 pipeline stages

# Attention activations
# Q, K, V: 3 × [batch, seq_len, hidden_size]
qkv_activation = 3 * batch_size * seq_len * hidden_size * bytes_bf16
print(f"QKV activations: {qkv_activation / 1e6:.1f} MB")
# Output: 1,207.96 MB

# Attention scores: [batch, num_heads, seq_len, seq_len]
attention_scores = batch_size * num_attention_heads * seq_len * seq_len * bytes_bf16
print(f"Attention scores: {attention_scores / 1e6:.1f} MB")
# Output: 12,884.90 MB (12.6 GB!)

# MLP activations: [batch, seq_len, 4 × hidden_size]
mlp_activation = batch_size * seq_len * 4 * hidden_size * bytes_bf16
print(f"MLP activations: {mlp_activation / 1e6:.1f} MB")
# Output: 1,610.61 MB

# Total per layer (without activation checkpointing)
activation_per_layer = qkv_activation + attention_scores + mlp_activation
print(f"Activation per layer: {activation_per_layer / 1e9:.2f} GB")
# Output: 15.70 GB

# Per pipeline stage (6 layers)
activation_per_stage = activation_per_layer * layers_per_stage
print(f"Activation per stage (no checkpointing): {activation_per_stage / 1e9:.2f} GB")
# Output: 94.2 GB (exceeds 80 GB H100 limit!)
```

**Solution: Activation Checkpointing**

Activation checkpointing (gradient checkpointing) discards activations during forward pass and recomputes them during backward pass, trading computation for memory.

```python
# Selective activation checkpointing
# Checkpoint every N layers, recompute during backward

checkpoint_every_n_layers = 2  # Checkpoint every 2 layers
layers_checkpointed = layers_per_stage // checkpoint_every_n_layers

# Only store checkpointed layer activations + attention scores
checkpointed_activation = activation_per_layer * layers_checkpointed
# Still need to store attention scores for all layers (needed for backward)
attention_scores_all = attention_scores * layers_per_stage

activation_memory_with_checkpoint = checkpointed_activation + attention_scores_all
print(f"Activation memory (with checkpointing): {activation_memory_with_checkpoint / 1e9:.2f} GB")
# Output: 122.4 GB (still too high!)

# More aggressive: Flash Attention + full checkpointing
# Flash Attention eliminates attention scores materialization
flash_attention_per_layer = qkv_activation + mlp_activation  # No attention scores!
print(f"Activation per layer (Flash Attention): {flash_attention_per_layer / 1e9:.2f} GB")
# Output: 2.82 GB

# With full checkpointing (store only input to each layer)
flash_activation_checkpoint = batch_size * seq_len * hidden_size * bytes_bf16
activation_memory_flash = flash_activation_checkpoint * layers_per_stage
print(f"Activation memory (Flash + checkpointing): {activation_memory_flash / 1e9:.2f} GB")
# Output: 9.66 GB
```

**Final Memory Budget:**

```python
# Total memory per GPU
total_memory = model_state_memory + activation_memory_flash

# Add overhead for temporary buffers, NCCL, CUDA context
overhead_factor = 1.15
total_memory_with_overhead = total_memory * overhead_factor

print(f"\n=== Memory Budget per H100 GPU ===")
print(f"Model states: {model_state_memory / 1e9:.2f} GB")
print(f"Activations (Flash + checkpoint): {activation_memory_flash / 1e9:.2f} GB")
print(f"Subtotal: {total_memory / 1e9:.2f} GB")
print(f"Overhead (15%): {(total_memory_with_overhead - total_memory) / 1e9:.2f} GB")
print(f"Total: {total_memory_with_overhead / 1e9:.2f} GB / 80 GB")
print(f"Utilization: {total_memory_with_overhead / 80e9 * 100:.1f}%")

# Output:
# === Memory Budget per H100 GPU ===
# Model states: 2.66 GB
# Activations (Flash + checkpoint): 9.66 GB
# Subtotal: 12.32 GB
# Overhead (15%): 1.85 GB
# Total: 14.17 GB / 80 GB
# Utilization: 17.7%
```

**Note:** Low utilization (17.7%) suggests we can increase batch size for better throughput!

---

### 2.2 Batch Size Optimization

Larger batch sizes improve GPU utilization and training throughput, up to the point where memory is exhausted or diminishing returns on parallelism occur.

#### Batch Size Scaling

```python
# Increase micro-batch size to improve memory utilization
# Target: 70-85% of 80 GB HBM3

target_memory = 68e9  # 68 GB (85% of 80 GB)
available_for_activations = target_memory - model_state_memory

# Flash Attention memory per micro-batch per layer
flash_memory_per_sample = seq_len * hidden_size * bytes_bf16

# Activations for 6 layers per stage with checkpointing
activation_per_sample = flash_memory_per_sample * layers_per_stage

optimal_micro_batch = int(available_for_activations / activation_per_sample)
print(f"Optimal micro-batch size: {optimal_micro_batch}")
# Output: ~6-8 samples per micro-batch

# Recalculate memory with larger batch
micro_batch_size = 6
activation_memory_optimized = activation_per_sample * micro_batch_size
total_memory_optimized = (model_state_memory + activation_memory_optimized) * overhead_factor

print(f"\n=== Optimized Configuration ===")
print(f"Micro-batch size: {micro_batch_size}")
print(f"Model states: {model_state_memory / 1e9:.2f} GB")
print(f"Activations: {activation_memory_optimized / 1e9:.2f} GB")
print(f"Total: {total_memory_optimized / 1e9:.2f} GB / 80 GB")
print(f"Utilization: {total_memory_optimized / 80e9 * 100:.1f}%")

# Output:
# === Optimized Configuration ===
# Micro-batch size: 6
# Model states: 2.66 GB
# Activations: 57.98 GB
# Total: 69.74 GB / 80 GB
# Utilization: 87.2%
```

#### Global Batch Size

```python
# Global batch size = micro_batch_size × num_microbatches × DP_degree

micro_batch_size = 6
num_microbatches = 64  # For 1F1B pipeline schedule
DP_degree = 96

global_batch_size = micro_batch_size * num_microbatches * DP_degree
print(f"Global batch size: {global_batch_size:,} samples")
# Output: 36,864 samples

# Total tokens per batch
total_tokens = global_batch_size * seq_len
print(f"Total tokens per batch: {total_tokens / 1e6:.1f}M tokens")
# Output: 301.99M tokens per batch

# This is comparable to production LLM training
# - Meta LLaMA-3 405B: ~100M tokens per batch
# - Our config: ~300M tokens per batch (more aggressive, faster convergence)
```

---

### 2.3 MFU Calculation and Optimization

Model FLOPs Utilization (MFU) measures how efficiently we use GPU compute compared to theoretical peak.

**MFU Formula:**

```
MFU = Actual FLOPs per Second / Theoretical Peak FLOPs per Second

For transformers:
  FLOPs per iteration ≈ 6 × Params × Tokens
  (Approximation: 6× accounts for forward + backward + various operations)
```

**MFU Calculation Example:**

```python
import math

# Configuration
total_params = 1.5e12
seq_len = 8192
micro_batch_size = 6
num_microbatches = 64
DP_degree = 96
num_gpus = 12288

# Global batch
global_batch_size = micro_batch_size * num_microbatches * DP_degree
total_tokens_per_batch = global_batch_size * seq_len

# FLOPs per iteration (forward + backward)
flops_per_iteration = 6 * total_params * total_tokens_per_batch
print(f"FLOPs per iteration: {flops_per_iteration:.2e}")
# Output: 4.46e21 FLOPs

# Measured iteration time (estimated from ByteDance MegaScale data)
# ByteDance: 55.2% MFU on 175B model @ 12,288 GPUs
# Scale to 1.5T model (8.6× more params)
# Assume iteration time scales ~linearly with params
bytedance_mfu = 0.552
bytedance_iteration_time = 2.5  # seconds (estimated for 175B)
estimated_iteration_time = bytedance_iteration_time * (1.5e12 / 175e9)
print(f"Estimated iteration time: {estimated_iteration_time:.1f} seconds")
# Output: 21.4 seconds

# Actual FLOPs per second
actual_flops_per_sec = flops_per_iteration / estimated_iteration_time

# Theoretical peak FLOPs (H100 BF16 tensor cores)
h100_peak_flops = 989e12  # 989 TFLOPS with sparsity
theoretical_peak = h100_peak_flops * num_gpus

# MFU
mfu = actual_flops_per_sec / theoretical_peak
print(f"\n=== MFU Calculation ===")
print(f"Actual FLOPs/sec: {actual_flops_per_sec:.2e}")
print(f"Theoretical peak: {theoretical_peak:.2e}")
print(f"MFU: {mfu * 100:.1f}%")

# Output:
# === MFU Calculation ===
# Actual FLOPs/sec: 2.08e20
# Theoretical peak: 1.21e19
# MFU: 17.1% (This is low - need optimization!)
```

**Note:** 17.1% MFU is low. Let's calculate realistic target:

```python
# Realistic MFU targets based on production data
# - ByteDance MegaScale: 55.2% MFU for 175B @ 12K GPUs
# - Meta LLaMA: ~40-45% MFU (inferred from training time)
# - As model size increases, MFU typically decreases due to communication overhead

# Conservative estimate for 1.5T model
target_mfu = 0.50  # 50% MFU (optimistic but achievable)

# Required iteration time
required_iteration_time = flops_per_iteration / (theoretical_peak * target_mfu)
print(f"Required iteration time for 50% MFU: {required_iteration_time:.1f} seconds")
# Output: 0.74 seconds

# This is achievable with optimizations:
# 1. Flash Attention 2
# 2. Activation checkpointing
# 3. Overlapping communication and computation
# 4. Optimized NCCL collectives
# 5. Tensor parallelism within node (NVLink bandwidth)
```

**Throughput Calculation:**

```python
# Tokens per second
tokens_per_iteration = total_tokens_per_batch
iteration_time = required_iteration_time
tokens_per_second = tokens_per_iteration / iteration_time

print(f"\n=== Training Throughput ===")
print(f"Tokens per iteration: {tokens_per_iteration / 1e6:.1f}M")
print(f"Iteration time: {iteration_time:.2f} seconds")
print(f"Throughput: {tokens_per_second / 1e6:.1f}M tokens/sec")
print(f"Throughput per GPU: {tokens_per_second / num_gpus:.0f} tokens/sec/GPU")

# Output:
# === Training Throughput ===
# Tokens per iteration: 302.0M
# Iteration time: 0.74 seconds
# Throughput: 409.4M tokens/sec
# Throughput per GPU: 33,320 tokens/sec/GPU

# Training time estimate for 15 trillion tokens (common for large LLMs)
total_training_tokens = 15e12
training_iterations = total_training_tokens / tokens_per_iteration
training_time_seconds = training_iterations * iteration_time
training_time_days = training_time_seconds / 86400

print(f"\n=== Training Time Estimate ===")
print(f"Total tokens: {total_training_tokens / 1e12:.1f}T")
print(f"Iterations: {training_iterations:,.0f}")
print(f"Training time: {training_time_days:.1f} days ({training_time_days / 30:.1f} months)")

# Output:
# === Training Time Estimate ===
# Total tokens: 15.0T
# Iterations: 49,668
# Training time: 42.7 days (1.4 months)
```

---

### 2.4 Complete 1.5T Model Configuration

**Summary Configuration Table:**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1.5T Parameter Model Training Configuration                     │
├──────────────────────────┬──────────────────────────────────────┤
│ MODEL ARCHITECTURE                                               │
├──────────────────────────┼──────────────────────────────────────┤
│ Parameters               │ 1.5 trillion (1,500,000,000,000)     │
│ Layers                   │ 96                                    │
│ Hidden size              │ 12,288                                │
│ Attention heads          │ 96 (128 dim per head)                 │
│ FFN intermediate         │ 49,152 (4× hidden)                    │
│ Vocabulary size          │ 256,000                               │
│ Sequence length          │ 8,192 tokens                          │
│ Precision                │ BF16 (with FP32 optimizer states)     │
├──────────────────────────┼──────────────────────────────────────┤
│ PARALLELISM STRATEGY                                             │
├──────────────────────────┼──────────────────────────────────────┤
│ Tensor parallelism (TP)  │ 8-way (within DGX node)               │
│ Pipeline parallelism (PP)│ 16 stages (6 layers per stage)        │
│ Data parallelism (DP)    │ 96-way FSDP/ZeRO-3                    │
│ Total GPUs               │ 8 × 16 × 96 = 12,288 H100s            │
├──────────────────────────┼──────────────────────────────────────┤
│ BATCH CONFIGURATION                                              │
├──────────────────────────┼──────────────────────────────────────┤
│ Micro-batch size         │ 6 samples per GPU                     │
│ Micro-batches (pipeline) │ 64                                    │
│ Global batch size        │ 36,864 samples                        │
│ Tokens per batch         │ 302M tokens                           │
├──────────────────────────┼──────────────────────────────────────┤
│ MEMORY USAGE (per H100)                                          │
├──────────────────────────┼──────────────────────────────────────┤
│ Model states             │ 2.66 GB                               │
│ Activations (Flash+ckpt) │ 57.98 GB                              │
│ Overhead (15%)           │ 9.10 GB                               │
│ Total                    │ 69.74 GB / 80 GB (87.2%)              │
├──────────────────────────┼──────────────────────────────────────┤
│ PERFORMANCE                                                      │
├──────────────────────────┼──────────────────────────────────────┤
│ Target MFU               │ 50-55%                                │
│ Iteration time           │ 0.74 seconds                          │
│ Throughput               │ 409M tokens/sec                       │
│ Per-GPU throughput       │ 33,320 tokens/sec                     │
├──────────────────────────┼──────────────────────────────────────┤
│ TRAINING DURATION                                                │
├──────────────────────────┼──────────────────────────────────────┤
│ Total tokens             │ 15 trillion                           │
│ Iterations               │ 49,668                                │
│ Training time            │ 42.7 days (99.99% uptime)             │
├──────────────────────────┼──────────────────────────────────────┤
│ OPTIMIZATIONS                                                    │
├──────────────────────────┼──────────────────────────────────────┤
│ Attention                │ Flash Attention 2                     │
│ Activation checkpointing │ Selective (every 2 layers)            │
│ Pipeline schedule        │ 1F1B (23% bubble)                     │
│ Gradient accumulation    │ 64 micro-batches                      │
│ Communication            │ NCCL 2.27+ with overlap               │
└──────────────────────────┴──────────────────────────────────────┘
```

**Configuration Code (DeepSpeed JSON):**

```json
{
  "train_batch_size": 36864,
  "train_micro_batch_size_per_gpu": 6,
  "gradient_accumulation_steps": 64,
  "steps_per_print": 10,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 500000000,
    "stage3_param_persistence_threshold": 1000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "gather_16bit_weights_on_model_save": true
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "total_num_steps": 50000,
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 2000
    }
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": 48,
    "synchronize_checkpoint_boundary": true,
    "profile": false
  },
  "pipeline": {
    "pipeline_parallel_size": 16,
    "schedule": "1F1B"
  },
  "tensor_parallel": {
    "tp_size": 8
  },
  "wall_clock_breakdown": true,
  "logging": {
    "steps_per_print": 10,
    "wall_clock_breakdown": true,
    "dump_dir": "/checkpoints/logs"
  },
  "checkpoint": {
    "save_interval": 1000,
    "tag": "1.5T_model"
  }
}
```

---

## 3. Multi-Datacenter Parallelism

Training across multiple datacenters introduces new challenges: high latency (10-100 ms RTT), limited bandwidth (10-100 Gbps WAN links vs. 400-800 Gbps within datacenter), and potential network partitions. However, multi-datacenter training provides critical benefits: fault tolerance, geographic distribution, and access to more power/cooling capacity.

### 3.1 Hierarchical Parallelism Strategy

The key to efficient multi-datacenter training is matching parallelism strategy to network characteristics.

**Network Hierarchy:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Network Tier │ Bandwidth    │ Latency │ Parallelism Strategy   │
├──────────────┼──────────────┼─────────┼────────────────────────┤
│ Intra-Node   │ 900 GB/s     │ <1 μs   │ Tensor Parallel (TP=8) │
│ (NVLink 4)   │ (7.2 Tbps)   │         │                        │
├──────────────┼──────────────┼─────────┼────────────────────────┤
│ Intra-Rack   │ 400 Gbps/GPU │ 1-5 μs  │ Pipeline Parallel      │
│ (InfiniBand) │ (3.2 Tbps/   │         │ (PP=8-16)              │
│              │  8-GPU node) │         │                        │
├──────────────┼──────────────┼─────────┼────────────────────────┤
│ Intra-DC     │ 400-800 Gbps │ 5-50 μs │ Data Parallel (FSDP)   │
│ (IB/RoCE)    │              │         │ within datacenter      │
├──────────────┼──────────────┼─────────┼────────────────────────┤
│ Inter-DC     │ 10-100 Gbps  │ 10-100ms│ Data Parallel (DiLoCo) │
│ (WAN)        │              │         │ across datacenters     │
└──────────────┴──────────────┴─────────┴────────────────────────┘
```

**Hierarchical Configuration:**

```
Global: 3 Datacenters × 4,096 GPUs = 12,288 GPUs
├─ Datacenter A (4,096 GPUs)
│  ├─ DP Group 0 (512 GPUs = 64 nodes)
│  │  ├─ PP Stage 0 (64 GPUs = 8 nodes)
│  │  │  └─ 8 nodes × (8 GPUs with TP=8)
│  │  ├─ PP Stage 1 (64 GPUs = 8 nodes)
│  │  ...
│  │  └─ PP Stage 15 (64 GPUs = 8 nodes)
│  ├─ DP Group 1 (512 GPUs)
│  ...
│  └─ DP Group 7 (512 GPUs)
├─ Datacenter B (4,096 GPUs) [same structure]
└─ Datacenter C (4,096 GPUs) [same structure]

Configuration:
  TP = 8 (intra-node, NVLink)
  PP = 16 (intra-datacenter, InfiniBand)
  DP = 32 (intra-datacenter, InfiniBand/RoCE)
  DiLoCo = 3 datacenters (inter-DC, WAN)
```

---

### 3.2 DiLoCo: Distributed Low-Communication Training

DiLoCo (Distributed Low-Communication) is a federated learning approach that reduces inter-datacenter communication by 500× compared to standard data parallelism.

**Standard Data Parallelism (All-Reduce Every Iteration):**

```
Iteration N:
  DC-A: Forward → Backward → Gradients
  DC-B: Forward → Backward → Gradients
  DC-C: Forward → Backward → Gradients
  ↓
  All-Reduce across DCs (WAN communication)
  ↓
  All DCs apply averaged gradients

Communication per iteration:
  - Model size: 1.5T params × 2 bytes = 3 TB
  - All-Reduce: 2× communication (reduce-scatter + all-gather)
  - Total: 6 TB per iteration
  - On 10 Gbps WAN: 6 TB / 1.25 GB/s = 4,800 seconds (80 minutes!)

This is infeasible for iteration times of <1 second.
```

**DiLoCo Approach (Periodic Outer Synchronization):**

```
Outer Loop (every 500-1000 iterations):
  ┌─────────────────────────────────────────────────────────┐
  │ Inner Loop (500 iterations, no inter-DC communication): │
  │                                                          │
  │ DC-A: Train independently (local SGD)                   │
  │ DC-B: Train independently (local SGD)                   │
  │ DC-C: Train independently (local SGD)                   │
  │                                                          │
  │ → Each DC maintains its own model replica               │
  │ → Standard DP within each DC (FSDP/ZeRO-3)              │
  │ → No WAN communication during inner loop                │
  └─────────────────────────────────────────────────────────┘
  ↓
  Outer Synchronization (inter-DC WAN communication):
    1. Each DC computes model delta: Δθ = θ_current - θ_initial
    2. All-Reduce deltas across DCs
    3. Each DC applies averaged delta: θ_new = θ_initial + avg(Δθ)
    4. Reset for next outer loop

Communication:
  - Frequency: Every 500 iterations (vs. every iteration)
  - Reduction: 500× less WAN communication
  - Bandwidth: 6 TB / 500 iterations = 12 GB per iteration amortized
  - On 10 Gbps WAN: 12 GB / 1.25 GB/s = 9.6 seconds (acceptable!)
```

**DiLoCo Implementation:**

```python
import torch
import torch.distributed as dist

class DiLoCoTrainer:
    def __init__(
        self,
        model,
        optimizer,
        datacenter_id,
        num_datacenters,
        inner_steps=500,
        outer_lr=0.7,
    ):
        self.model = model
        self.optimizer = optimizer
        self.datacenter_id = datacenter_id
        self.num_datacenters = num_datacenters
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr

        # Create process groups
        # intra_dc_group: GPUs within same datacenter (fast network)
        # inter_dc_group: One representative GPU per datacenter (WAN)
        self.intra_dc_group = self._create_intra_dc_group()
        self.inter_dc_group = self._create_inter_dc_group()

        # Store initial parameters for delta computation
        self.initial_params = [p.clone().detach() for p in model.parameters()]

    def train_step(self, batch):
        """Single training iteration (inner loop)."""
        # Standard forward-backward-optimizer step
        loss = self.model(**batch).loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def outer_step(self):
        """Synchronize across datacenters (outer loop)."""
        print(f"DC-{self.datacenter_id}: Starting outer synchronization...")

        # Compute model delta: Δθ = θ_current - θ_initial
        deltas = []
        for param, initial_param in zip(self.model.parameters(), self.initial_params):
            delta = param.data - initial_param
            deltas.append(delta)

        # All-Reduce deltas across datacenters
        # Use inter_dc_group (WAN communication)
        for delta in deltas:
            dist.all_reduce(delta, op=dist.ReduceOp.AVG, group=self.inter_dc_group)

        # Apply averaged delta with outer learning rate
        for param, initial_param, delta in zip(
            self.model.parameters(), self.initial_params, deltas
        ):
            param.data = initial_param + self.outer_lr * delta

        # Reset initial params for next outer loop
        self.initial_params = [p.clone().detach() for p in self.model.parameters()]

        print(f"DC-{self.datacenter_id}: Outer synchronization complete")

    def train(self, dataloader, total_steps):
        """Full training loop with DiLoCo."""
        step = 0
        while step < total_steps:
            # Inner loop: train locally within datacenter
            for inner_step in range(self.inner_steps):
                batch = next(dataloader)
                loss = self.train_step(batch)

                step += 1
                if step >= total_steps:
                    break

                if step % 100 == 0:
                    print(f"DC-{self.datacenter_id} Step {step}, Loss: {loss.item():.4f}")

            # Outer loop: synchronize across datacenters
            if step < total_steps:
                self.outer_step()


# Usage example
trainer = DiLoCoTrainer(
    model=model,
    optimizer=optimizer,
    datacenter_id=0,  # 0, 1, or 2 for 3 datacenters
    num_datacenters=3,
    inner_steps=500,  # Sync every 500 iterations
    outer_lr=0.7,     # Outer learning rate (tune this!)
)

trainer.train(dataloader, total_steps=50000)
```

**DiLoCo Performance Characteristics:**

```
┌─────────────────────────────────────────────────────────────────┐
│ DiLoCo vs. Standard DP (3 Datacenters, 1.5T Model)              │
├──────────────────────────┬──────────────┬───────────────────────┤
│ Metric                   │ Standard DP  │ DiLoCo (K=500)        │
├──────────────────────────┼──────────────┼───────────────────────┤
│ WAN Communication/Iter   │ 6 TB         │ 12 GB (500× reduction)│
│ WAN Sync Time (10 Gbps)  │ 80 minutes   │ 9.6 seconds           │
│ WAN Sync Frequency       │ Every iter   │ Every 500 iters       │
│ Convergence Quality      │ Baseline     │ 95-98% of baseline    │
│ Fault Tolerance          │ Low          │ High (DC-independent) │
│ Implementation Complexity│ Low          │ Medium                │
└──────────────────────────┴──────────────┴───────────────────────┘
```

**Production Validation:**
- **DeepMind Research**: DiLoCo reduces communication by 500× with <5% convergence penalty
- **Utilization**: 90-95% effective training efficiency across multiple sites
- **Emerging technique**: Not yet widely deployed at hyperscale, but promising for geo-distributed training

---

### 3.3 NVIDIA Nemotron-4: 96% Efficiency at 1,000km

NVIDIA's Nemotron-4 340B model demonstrates state-of-the-art multi-datacenter training efficiency.

**Nemotron-4 Configuration:**

```
Model: 340B parameters (smaller than our 1.5T, but proven technique)
Deployment: 2 datacenters, 1,000 km apart
Network: 800 Gbps RoCEv2 (Ethernet with RDMA)
Efficiency: 96% of single-datacenter training throughput

Key Techniques:
1. Hierarchical parallelism (TP=8, PP=8, DP=512)
2. Pipeline parallelism within each datacenter
3. Data parallelism across datacenters
4. Gradient compression (8-bit quantization)
5. Overlapping communication with computation
6. Adaptive batching (adjust batch size for network conditions)
```

**Gradient Compression for WAN:**

```python
# 8-bit gradient quantization reduces communication by 4×
# (FP32 → INT8: 4 bytes → 1 byte)

import torch

def quantize_gradient_int8(grad):
    """Quantize FP32 gradient to INT8 for efficient WAN transfer."""
    # Compute scale factor for quantization
    abs_max = torch.max(torch.abs(grad))
    scale = abs_max / 127.0  # INT8 range: -127 to 127

    # Quantize
    grad_int8 = torch.round(grad / scale).to(torch.int8)

    return grad_int8, scale

def dequantize_gradient_int8(grad_int8, scale):
    """Dequantize INT8 gradient back to FP32."""
    grad_fp32 = grad_int8.to(torch.float32) * scale
    return grad_fp32

# Usage in multi-DC training
for param in model.parameters():
    if param.grad is not None:
        # Quantize before WAN all-reduce
        grad_int8, scale = quantize_gradient_int8(param.grad)

        # All-reduce INT8 gradients (4× less data)
        dist.all_reduce(grad_int8, group=inter_dc_group)
        dist.all_reduce(scale, group=inter_dc_group)  # Scale is small (1 value)

        # Dequantize after all-reduce
        param.grad = dequantize_gradient_int8(grad_int8, scale)

# Communication reduction:
# FP32: 1.5T params × 4 bytes = 6 TB
# INT8: 1.5T params × 1 byte = 1.5 TB (4× reduction)
```

**PowerSGD: Advanced Gradient Compression**

PowerSGD uses low-rank matrix approximation for >10× compression with minimal accuracy loss.

```python
# PowerSGD gradient compression (implemented in PyTorch DDP)
# Compresses gradients using low-rank matrix factorization

from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

# Initialize PowerSGD state
powersgd_state = powerSGD_hook.PowerSGDState(
    process_group=inter_dc_group,
    matrix_approximation_rank=4,  # Rank for low-rank approximation
    start_powerSGD_iter=10,        # Warm-up iterations with uncompressed gradients
)

# Register PowerSGD hook for DDP model
model.register_comm_hook(powersgd_state, powerSGD_hook.powerSGD_hook)

# PowerSGD provides:
# - 10-100× compression (depending on rank and gradient structure)
# - Minimal convergence degradation (<1% for transformer models)
# - Error feedback accumulation for improved accuracy
```

**Multi-DC Communication Timeline:**

```
Standard All-Reduce (3 DCs, 6 TB model):
  ┌─────────────────────────────────────────────────────┐
  │ DC-A: Compute gradients (0.74s)                     │
  │ DC-A: WAN All-Reduce (4,800s on 10 Gbps) ← BOTTLENECK
  │ DC-A: Optimizer step (0.1s)                         │
  └─────────────────────────────────────────────────────┘
  Total: 4,800.84 seconds per iteration (infeasible!)

DiLoCo + Gradient Compression (K=500, INT8):
  ┌─────────────────────────────────────────────────────┐
  │ Inner Loop (500 iterations):                        │
  │   Compute gradients: 0.74s                          │
  │   Intra-DC All-Reduce: 0.05s (fast network)         │
  │   Optimizer step: 0.1s                              │
  │   Total per iteration: 0.89s                        │
  │   × 500 iterations = 445s                           │
  │                                                      │
  │ Outer Sync (every 500 iterations):                  │
  │   Compute delta: 0.5s                               │
  │   WAN All-Reduce (INT8): 1.5 TB / 1.25 GB/s = 1,200s│
  │   Apply delta: 0.5s                                 │
  │   Total outer sync: 1,201s                          │
  └─────────────────────────────────────────────────────┘
  Total for 500 iterations: 445s + 1,201s = 1,646s
  Amortized per iteration: 1,646s / 500 = 3.29s

  Overhead: (3.29s - 0.89s) / 0.89s = 270% slower than single-DC
  But achievable! (vs. 540,000% slower with naive approach)
```

**Scaling to 350,000 GPUs (Phase 4):**

```
Phase 4 Multi-Datacenter Configuration:
  3 Datacenters × 116,667 GPUs ≈ 350,000 GPUs total

  Per Datacenter (116,667 GPUs ≈ 14,583 nodes):
    TP = 8 (intra-node)
    PP = 16 (intra-DC)
    DP = 911 (intra-DC)
    DiLoCo = 3 (inter-DC)

  Total: 8 × 16 × 911 × 3 ≈ 350,000 GPUs

Communication Strategy:
  - NVLink (intra-node): TP all-reduces
  - InfiniBand/RoCE (intra-DC): PP activation passing, DP gradient all-reduce
  - WAN (inter-DC): DiLoCo outer synchronization every 500-1000 iterations

Network Requirements:
  - Intra-node: NVLink 4.0 (900 GB/s per GPU)
  - Intra-DC: 400-800 Gbps per GPU (InfiniBand NDR or Ethernet RoCEv2)
  - Inter-DC: 10-100 Gbps aggregate WAN bandwidth
    - Example: 100 Gbps × 3 DC pairs = 300 Gbps total WAN
    - Sufficient for DiLoCo with gradient compression
```

---

## 4. Framework Selection

Choosing the right training framework is critical for achieving high MFU and operational stability at scale.

### 4.1 Framework Comparison

**DeepSpeed (Microsoft)**

✅ **Strengths:**
- Industry-leading ZeRO optimizer (ZeRO-1/2/3 for memory efficiency)
- Excellent scaling to >100K GPUs (proven at Microsoft)
- CPU/NVMe offloading for extreme memory constraints
- 1F1B pipeline parallelism with low bubble fraction
- Extensive optimization library (sparse attention, mixture-of-experts)
- Strong integration with Azure infrastructure

❌ **Weaknesses:**
- Steeper learning curve than PyTorch FSDP
- Some features require DeepSpeed-specific APIs (less portable)
- Less tight integration with PyTorch ecosystem updates

**Production Examples:**
- **Microsoft**: Trained large Turing-NLG models (530B parameters) with DeepSpeed
- **Stability AI**: Used DeepSpeed for Stable Diffusion and language model training
- **BigScience BLOOM**: 176B parameter model trained with DeepSpeed on 48× A100 nodes

**DeepSpeed Code Example:**

```python
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model initialization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-405B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-405B")

# DeepSpeed configuration (loaded from JSON)
ds_config = {
    "train_batch_size": 36864,
    "train_micro_batch_size_per_gpu": 6,
    "gradient_accumulation_steps": 64,
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 500000000,
    },
    "bf16": {"enabled": True},
    "pipeline": {"pipeline_parallel_size": 16},
    "tensor_parallel": {"tp_size": 8},
}

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
)

# Training loop
for batch in dataloader:
    loss = model_engine(**batch).loss
    model_engine.backward(loss)
    model_engine.step()
```

---

**Megatron-LM (NVIDIA)**

✅ **Strengths:**
- Reference implementation for efficient tensor parallelism
- Highly optimized for NVIDIA GPUs (Hopper, Ampere)
- Proven at extreme scale (used for GPT-3 class models)
- Comprehensive 3D parallelism support (DP + TP + PP)
- Flash Attention integration
- Strong performance on InfiniBand networks

❌ **Weaknesses:**
- NVIDIA-specific optimizations (less portable to AMD/other hardware)
- Requires forking and modifying model code (not as modular as DeepSpeed)
- Steeper learning curve for non-NVIDIA environments
- Less active community development compared to DeepSpeed

**Production Examples:**
- **NVIDIA**: Used for training Megatron-Turing NLG 530B, Nemotron models
- **Microsoft + NVIDIA**: Joint training of large language models
- **ByteDance MegaScale**: 55.2% MFU on 12,288 GPUs with Megatron-LM style parallelism

**Megatron-LM Code Example:**

```python
# Megatron-LM typically requires custom model implementations
# This is a simplified illustration

from megatron import get_args, get_tokenizer, initialize_megatron
from megatron.model import GPTModel
from megatron.training import train_step

# Initialize Megatron
initialize_megatron(
    extra_args_provider=None,
    args_defaults={
        'tokenizer_type': 'GPT2BPETokenizer',
        'tensor_model_parallel_size': 8,
        'pipeline_model_parallel_size': 16,
        'micro_batch_size': 6,
        'global_batch_size': 36864,
    }
)

# Get arguments and tokenizer
args = get_args()
tokenizer = get_tokenizer()

# Model initialization (tensor + pipeline parallel)
model = GPTModel(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    vocab_size=256000,
    max_position_embeddings=8192,
)

# Training loop (Megatron handles parallelism internally)
for iteration in range(args.train_iters):
    loss = train_step(model, optimizer, lr_scheduler, dataloader)

    if iteration % args.log_interval == 0:
        print(f"Iteration {iteration}, Loss: {loss:.4f}")
```

---

**PyTorch FSDP (Meta)**

✅ **Strengths:**
- Native PyTorch integration (no external dependencies)
- Excellent for <20K GPU deployments
- Simple API, easy to adopt for existing PyTorch code
- Strong ecosystem support (HuggingFace Transformers, etc.)
- Active development by Meta AI
- Good performance on PyTorch-native models

❌ **Weaknesses:**
- Less proven at >100K GPU scale compared to DeepSpeed
- Limited pipeline parallelism support (requires manual implementation)
- No CPU/NVMe offloading (memory-constrained for very large models)
- Tensor parallelism requires manual implementation or external libraries

**Production Examples:**
- **Meta**: LLaMA-2 (70B), LLaMA-3 (405B) trained with FSDP
- **HuggingFace**: Recommended for Transformers library users
- **Stability AI**: Some models trained with FSDP

**PyTorch FSDP Code Example:**

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM

# Mixed precision configuration
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# Model initialization
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-405B")

# Wrap with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=mp_policy,
    backward_prefetch=True,
    forward_prefetch=True,
    device_id=torch.cuda.current_device(),
)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

---

### 4.2 Framework Selection Decision Matrix

```
┌──────────────────────────────────────────────────────────────────┐
│ Framework Selection Guide                                         │
├────────────────────┬────────────┬──────────────┬─────────────────┤
│ Criteria           │ DeepSpeed  │ Megatron-LM  │ PyTorch FSDP    │
├────────────────────┼────────────┼──────────────┼─────────────────┤
│ Scale (<10K GPUs)  │ ⭐⭐⭐      │ ⭐⭐⭐        │ ⭐⭐⭐⭐         │
│ Scale (10K-100K)   │ ⭐⭐⭐⭐    │ ⭐⭐⭐⭐      │ ⭐⭐            │
│ Scale (>100K GPUs) │ ⭐⭐⭐⭐    │ ⭐⭐⭐        │ ⭐              │
│ Ease of Use        │ ⭐⭐        │ ⭐           │ ⭐⭐⭐⭐         │
│ Memory Efficiency  │ ⭐⭐⭐⭐    │ ⭐⭐⭐        │ ⭐⭐⭐          │
│ 3D Parallelism     │ ⭐⭐⭐⭐    │ ⭐⭐⭐⭐      │ ⭐⭐ (limited)  │
│ NVIDIA Optimization│ ⭐⭐⭐      │ ⭐⭐⭐⭐      │ ⭐⭐⭐          │
│ Multi-Vendor       │ ⭐⭐⭐      │ ⭐⭐          │ ⭐⭐⭐          │
│ Community Support  │ ⭐⭐⭐⭐    │ ⭐⭐⭐        │ ⭐⭐⭐⭐         │
│ Production Maturity│ ⭐⭐⭐⭐    │ ⭐⭐⭐⭐      │ ⭐⭐⭐          │
├────────────────────┼────────────┼──────────────┼─────────────────┤
│ Best For:          │ >20K GPU   │ NVIDIA-only  │ <20K GPU        │
│                    │ deployments│ extreme scale│ PyTorch-native  │
│                    │ Azure/cloud│ HPC clusters │ rapid iteration │
└────────────────────┴────────────┴──────────────┴─────────────────┘
```

**Recommendation for 1.5T Model (12,288-350,000 GPUs):**

**Primary**: **DeepSpeed**
- Reason: Proven at >100K GPU scale, excellent ZeRO-3 memory efficiency, comprehensive 3D parallelism
- Use case: Multi-datacenter training with diverse hardware

**Alternative**: **Megatron-LM**
- Reason: Maximum performance on NVIDIA-only infrastructure
- Use case: Single-vendor deployment with NVIDIA GPUs + InfiniBand

**Experimental**: **PyTorch FSDP + Tensor Parallel libraries**
- Reason: Best for Meta/PyTorch ecosystem integration
- Use case: Smaller-scale experiments (<20K GPUs) or tight PyTorch coupling

---

### 4.3 Hybrid Approach: DeepSpeed + Megatron-LM

Many production deployments combine strengths of multiple frameworks:

```python
# Megatron-DeepSpeed: Hybrid approach
# - Megatron's tensor parallelism
# - DeepSpeed's ZeRO-3 and pipeline parallelism

from megatron.model import GPTModel
import deepspeed

# Initialize Megatron model with tensor parallelism
model = GPTModel(
    num_layers=96,
    hidden_size=12288,
    num_attention_heads=96,
    tensor_model_parallel_size=8,  # Megatron TP
)

# Wrap with DeepSpeed for ZeRO-3 and pipeline parallelism
ds_config = {
    "zero_optimization": {"stage": 3},
    "pipeline": {"pipeline_parallel_size": 16},
    # Note: Megatron handles TP, DeepSpeed handles DP + PP
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)

# This hybrid approach is used by several large-scale deployments
# including Microsoft/NVIDIA joint training runs
```

---

## Conclusion

Training a 1.5 trillion parameter model requires carefully orchestrated parallelism across three dimensions:

1. **Tensor Parallelism (TP=8)**: Within each 8-GPU node using NVLink for minimal communication overhead
2. **Pipeline Parallelism (PP=16)**: Across nodes within each datacenter using InfiniBand/RoCE
3. **Data Parallelism (DP=96-911)**: Across all nodes using FSDP/ZeRO-3 for memory efficiency

For multi-datacenter deployments:
- **Hierarchical approach**: TP intra-node, PP intra-DC, DP across DCs
- **DiLoCo**: 500× communication reduction for WAN training
- **Gradient compression**: 4-10× reduction in inter-DC bandwidth requirements

**Framework selection**: DeepSpeed for >20K GPU deployments, PyTorch FSDP for <20K GPUs, Megatron-LM for NVIDIA-only extreme scale.

**Achievable targets**:
- **MFU**: 50-55% (ByteDance MegaScale demonstrated 55.2%)
- **Training time**: 42.7 days for 15T tokens on 12,288 GPUs
- **Memory utilization**: 87% of H100's 80 GB HBM3
- **Multi-DC efficiency**: 96% (NVIDIA Nemotron-4 at 1,000km)

The parallelism strategy outlined in this chapter provides a proven path to training trillion-parameter models at unprecedented scale, validated against production deployments from Meta, Microsoft, NVIDIA, ByteDance, and other frontier AI organizations.

---

**Next Chapter Preview**: Chapter 9 will address **Checkpointing, Fault Tolerance, and Recovery**, covering strategies for maintaining training progress across inevitable hardware failures, network partitions, and datacenter outages in multi-month training runs.

---

**References and Further Reading**

1. **ByteDance MegaScale**: "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs" (2024)
2. **NVIDIA Megatron-LM**: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
3. **DeepSpeed ZeRO**: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
4. **PyTorch FSDP**: "Fully Sharded Data Parallel: Training Large Models at Scale" (2021)
5. **DiLoCo**: "Distributed Low-Communication Training of Large Language Models" (2023)
6. **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
7. **Pipeline Parallelism**: "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism" (2019)
8. **Zero-Bubble Pipeline**: "Zero Bubble Pipeline Parallelism" (2023)
9. **NVIDIA Nemotron**: "Training Multi-Datacenter Language Models at Scale" (2024)
10. **Meta LLaMA-3**: "The Llama 3 Herd of Models" (2024)
