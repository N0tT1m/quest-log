import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

const db = drizzle(sqlite, { schema });

interface TaskData {
	title: string;
	description: string;
	details: string;
}

interface ModuleData {
	name: string;
	description: string;
	tasks: TaskData[];
}

interface PathData {
	name: string;
	description: string;
	language: string;
	color: string;
	skills: string;
	startHint: string;
	difficulty: string;
	estimatedWeeks: number;
	schedule: string;
	modules: ModuleData[];
}

const transformersPath: PathData = {
	name: 'Transformers, LLMs & Generative AI Deep Dive',
	description: 'Comprehensive technical guide to understanding and building transformer-based models from the ground up. Master attention mechanisms, LLM architectures, training techniques, fine-tuning methods, and deployment optimization.',
	language: 'Python',
	color: 'cyan',
	skills: 'transformers, attention mechanisms, LLMs, GPT architecture, BERT, fine-tuning, LoRA, RLHF, inference optimization, distributed training',
	startHint: 'Start by implementing scaled dot-product attention from scratch',
	difficulty: 'advanced',
	estimatedWeeks: 12,
	schedule: `## 12-Week Transformers & LLMs Mastery

### Weeks 1-2: Attention Mechanisms

#### Week 1: Core Attention
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Environment setup, review linear algebra |
| Tue | Scaled Attention | Implement scaled dot-product attention |
| Wed | Multi-Head | Build multi-head attention module |
| Thu | Causal Masking | Add causal masking for autoregressive models |
| Fri | Variants | Explore Flash Attention, GQA concepts |
| Weekend | Practice | Implement attention visualizations |

#### Week 2: Position Encodings
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Sinusoidal | Implement original position encoding |
| Wed-Thu | RoPE | Build Rotary Position Embeddings |
| Fri | ALiBi | Implement Attention with Linear Biases |
| Weekend | Comparison | Compare different position encoding methods |

### Weeks 3-5: Transformer Architecture

#### Week 3: Building Blocks
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Feed-Forward | Implement FFN and SwiGLU |
| Tue | Normalization | Build LayerNorm and RMSNorm |
| Wed | Residual | Add residual connections |
| Thu | Transformer Block | Combine all components |
| Fri | Testing | Unit tests for each component |
| Weekend | Integration | Build complete transformer block |

#### Week 4: Full Models
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Decoder-Only | Build GPT-style model from scratch |
| Wed-Thu | Encoder-Only | Implement BERT-style architecture |
| Fri | Encoder-Decoder | Build full T5-style transformer |
| Weekend | Training | Train small model on toy dataset |

#### Week 5: Modern Architectures
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Llama | Study Llama architecture details |
| Tue | Mistral | Implement sliding window attention |
| Wed | Mixture of Experts | Build MoE routing |
| Thu | GQA | Implement Grouped Query Attention |
| Fri | Comparison | Benchmark different architectures |
| Weekend | Analysis | Compare memory/speed tradeoffs |

### Weeks 6-7: Training LLMs

#### Week 6: Pretraining
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Data Pipeline | Build tokenization pipeline |
| Tue | CLM Loss | Implement causal language modeling |
| Wed | MLM Loss | Implement masked language modeling |
| Thu | Training Loop | Build complete training pipeline |
| Fri | Mixed Precision | Add FP16/BF16 training |
| Weekend | Experiments | Train on larger dataset |

#### Week 7: Optimization
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Optimizer | Implement AdamW properly |
| Tue | Scheduler | Add cosine annealing with warmup |
| Wed | Gradient Clipping | Implement proper gradient handling |
| Thu | Gradient Accumulation | Handle large effective batches |
| Fri | Checkpointing | Add model checkpointing |
| Weekend | Long Run | Multi-hour training run |

### Weeks 8-9: Fine-Tuning Techniques

#### Week 8: Parameter-Efficient Methods
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | LoRA | Implement Low-Rank Adaptation |
| Wed-Thu | QLoRA | Add 4-bit quantization |
| Fri | Prefix Tuning | Implement prefix tuning |
| Weekend | Comparison | Compare methods on metrics |

#### Week 9: Advanced Fine-Tuning
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Instruction Tuning | Format instruction datasets |
| Wed | Reward Model | Build reward model for RLHF |
| Thu | DPO | Implement Direct Preference Optimization |
| Fri | Chat Format | Implement ChatML formatting |
| Weekend | Full Pipeline | End-to-end instruction tuning |

### Weeks 10-11: Inference Optimization

#### Week 10: Speed Optimization
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | KV Cache | Implement KV caching |
| Tue | Batching | Build dynamic batching |
| Wed | Quantization | Apply INT8/INT4 quantization |
| Thu | Flash Attention | Integrate Flash Attention |
| Fri | Benchmarking | Measure throughput/latency |
| Weekend | Optimization | Profile and optimize bottlenecks |

#### Week 11: Production Deployment
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | TorchScript | Export to TorchScript |
| Tue | ONNX | Convert to ONNX format |
| Wed | vLLM | Set up vLLM for serving |
| Thu | API Server | Build FastAPI inference server |
| Fri | Load Testing | Stress test deployment |
| Weekend | Monitoring | Add metrics and logging |

### Week 12: Advanced Topics & Projects

#### Distributed Training
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Data Parallel | Implement DDP |
| Tue | Model Parallel | Pipeline and tensor parallelism |
| Wed | ZeRO | Understand ZeRO optimization |
| Thu | Integration | Multi-GPU training setup |
| Fri | Capstone | Build and deploy complete LLM |
| Weekend | Documentation | Write comprehensive guide |

### Daily Commitment: 3-4 hours
### Hardware: GPU with 16GB+ VRAM recommended`,
	modules: [
		{
			name: 'Foundations: Attention Mechanisms',
			description: 'Deep understanding of attention - the core mechanism powering transformers',
			tasks: [
				{
					title: 'Implement scaled dot-product attention from scratch',
					description: 'Build the fundamental attention mechanism with masking support',
					details: `## Scaled Dot-Product Attention

### Why Attention?

Before transformers, sequence models (RNNs, LSTMs) had fundamental problems:

1. **Vanishing gradients** - Information from early tokens degrades
2. **No parallelization** - Sequential processing prevents GPU utilization

Attention solves both by allowing every token to directly attend to every other token in parallel.

### The Core Formula

\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) V
\`\`\`

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **d_k**: Dimension of keys (scaling factor prevents softmax saturation)

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Scaled dot-product attention mechanism.

    Args:
        query: (batch, heads, seq_len, d_k)
        key: (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Returns:
        output: (batch, heads, seq_len, d_v)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale by sqrt(d_k) to prevent softmax saturation
    scores = scores / math.sqrt(d_k)

    # Apply mask (for causal attention or padding)
    if mask is not None:
        # Mask positions with -inf so softmax gives 0 probability
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over the last dimension (attending positions)
    attention_weights = F.softmax(scores, dim=-1)

    # Handle NaN from -inf in softmax (e.g., fully masked rows)
    attention_weights = attention_weights.masked_fill(
        attention_weights != attention_weights, 0.0
    )

    # Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# Example usage
def example_attention():
    batch_size = 2
    num_heads = 8
    seq_len = 10
    d_k = 64

    # Random Q, K, V
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Causal mask (for autoregressive models)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

    # Apply attention
    output, weights = scaled_dot_product_attention(Q, K, V, causal_mask)

    print(f"Output shape: {output.shape}")  # (2, 8, 10, 64)
    print(f"Attention weights shape: {weights.shape}")  # (2, 8, 10, 10)

    # Verify attention weights sum to 1
    print(f"Weights sum: {weights.sum(dim=-1)[0, 0]}")  # Should be all 1s

    return output, weights


# Visualize attention patterns
def visualize_attention(attention_weights, tokens=None):
    """
    Visualize attention patterns.

    Args:
        attention_weights: (seq_len, seq_len)
        tokens: list of token strings
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        cmap='viridis',
        xticklabels=tokens,
        yticklabels=tokens,
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.xlabel('Key/Value Position')
    plt.ylabel('Query Position')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    plt.show()


# Testing different mask types
def test_masks():
    seq_len = 5
    Q = K = V = torch.randn(1, 1, seq_len, 64)

    # 1. No mask (full attention)
    out1, w1 = scaled_dot_product_attention(Q, K, V)
    print("Full attention:")
    print(w1[0, 0])

    # 2. Causal mask (can only attend to past)
    causal = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    out2, w2 = scaled_dot_product_attention(Q, K, V, causal)
    print("\\nCausal attention:")
    print(w2[0, 0])

    # 3. Padding mask (ignore padding tokens)
    # Say last 2 tokens are padding
    padding_mask = torch.ones(1, 1, 1, seq_len)
    padding_mask[0, 0, 0, -2:] = 0
    out3, w3 = scaled_dot_product_attention(Q, K, V, padding_mask)
    print("\\nPadding mask attention:")
    print(w3[0, 0])


if __name__ == "__main__":
    example_attention()
\`\`\`

### Why Scaling Matters

Without scaling (dividing by √d_k):
- Dot products grow with dimension size
- Softmax saturates (near 0 or 1)
- Gradients vanish
- Training becomes unstable

**Example:**
\`\`\`python
# Without scaling
d_k = 512
q = torch.randn(1, 1, 1, d_k)
k = torch.randn(1, 1, 10, d_k)
scores = q @ k.transpose(-2, -1)
print(f"Score range without scaling: {scores.min():.1f} to {scores.max():.1f}")
# Output: -50 to 50 (extreme values!)

# With scaling
scores_scaled = scores / math.sqrt(d_k)
print(f"Score range with scaling: {scores_scaled.min():.1f} to {scores_scaled.max():.1f}")
# Output: -2 to 2 (reasonable for softmax)
\`\`\`

### Attention Complexity

**Time Complexity:** O(n²·d) where n = sequence length, d = dimension
**Space Complexity:** O(n²) for attention matrix

This is why long sequences are expensive!

### Practice Exercises

- [ ] Implement attention from scratch
- [ ] Test with different mask types
- [ ] Visualize attention patterns
- [ ] Measure memory usage for different sequence lengths
- [ ] Compare with/without scaling
- [ ] Implement attention dropout`
				},
				{
					title: 'Build multi-head attention with projection layers',
					description: 'Extend single attention to multiple parallel attention heads',
					details: `## Multi-Head Attention

### Concept

Instead of one attention function, project Q, K, V multiple times with different learned linear projections. Each "head" learns different aspects of relationships.

**Why multiple heads?**
- Different heads learn different patterns (syntax, semantics, etc.)
- Increases model capacity without much added cost
- Empirically works much better than single large head

### Architecture

\`\`\`
Input → [Linear Q, Linear K, Linear V] → Split into H heads
    → H × Attention(Q_h, K_h, V_h) → Concat → Linear Out
\`\`\`

### Implementation

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Split d_model into num_heads, each with dimension d_k = d_model / num_heads
    """
    def __init__(self, d_model, num_heads, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Combined for efficiency (single matrix multiply)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # For PyTorch 2.0+ Flash Attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
            return_attention: whether to return attention weights

        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, d_model = x.size()

        # Linear projections in batch: (batch, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into Q, K, V: each (batch, seq_len, d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (batch, num_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        if self.flash and not return_attention:
            # Use PyTorch's optimized Flash Attention (2x faster!)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=(mask is not None)
            )
            attn_weights = None
        else:
            # Manual attention
            attn_output, attn_weights = self.scaled_dot_product_attention(
                q, k, v, mask
            )

        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final output projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Manual implementation of scaled dot-product attention."""
        d_k = q.size(-1)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


# Separate Q, K, V version (for cross-attention)
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention.

    Used in encoder-decoder models where Q comes from decoder, K and V from encoder.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Separate projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, target_len, d_model) - from decoder
            key: (batch, source_len, d_model) - from encoder
            value: (batch, source_len, d_model) - from encoder
            mask: (batch, 1, target_len, source_len)
        """
        batch_size = query.size(0)

        # Project Q, K, V
        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output)


# Testing
def test_multihead_attention():
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(d_model, num_heads)

    # Input
    x = torch.randn(batch_size, seq_len, d_model)

    # Causal mask
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))

    # Forward
    output, attn = mha(x, mask, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in mha.parameters()):,}")

    # Verify output shape matches input
    assert output.shape == x.shape
    # Verify attention is normalized
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))

    print("\\nAll tests passed!")


if __name__ == "__main__":
    test_multihead_attention()
\`\`\`

### Why It Works

Each head can specialize:
- **Head 1**: Syntactic relationships (subject-verb)
- **Head 2**: Semantic similarity
- **Head 3**: Long-range dependencies
- **Head 4**: Local patterns
- etc.

**Empirical observation:** Different heads learn interpretable patterns!

### Efficiency Notes

**Combined QKV Projection:**
\`\`\`python
# Less efficient: 3 separate matmuls
q = self.W_q(x)
k = self.W_k(x)
v = self.W_v(x)

# More efficient: 1 matmul, then split
qkv = self.qkv_proj(x)  # (batch, seq, 3*d_model)
q, k, v = qkv.chunk(3, dim=-1)
\`\`\`

**Flash Attention:**
- 2-4x faster than naive implementation
- Same math, better memory access patterns
- Use when available (PyTorch 2.0+)

### Parameter Count

For d_model=512, num_heads=8:
- QKV projections: 512 × (3 × 512) = 786,432
- Output projection: 512 × 512 = 262,144
- **Total: ~1M parameters**

### Practice Exercises

- [ ] Implement multi-head attention from scratch
- [ ] Visualize what different heads attend to
- [ ] Compare speed with/without Flash Attention
- [ ] Implement cross-attention variant
- [ ] Test on real text sequences
- [ ] Profile memory usage`
				}
			]
		},
		{
			name: 'Transformer Architecture',
			description: 'Build complete transformer models from components',
			tasks: [
				{
					title: 'Implement complete GPT-style decoder-only transformer',
					description: 'Build a full autoregressive language model from scratch',
					details: String.raw`## Complete GPT Architecture

### Decoder-Only Transformer

Used by: GPT series, Llama, Mistral, Claude

**Key features:**
- Causal (autoregressive) attention
- Generate text left-to-right
- Simple architecture, scales well

### Full Implementation

` + `\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler and faster than LayerNorm, used in Llama.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit feed-forward network.

    Used in Llama, PaLM - better than ReLU empirically.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Note: d_ff typically 8/3 * d_model for SwiGLU
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x, W1, W2, W3) = (Swish(x·W1) ⊙ (x·W3))·W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support."""

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0

        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads

        # Combined QKV projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        if not self.flash:
            # Causal mask
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                .view(1, 1, config.max_seq_len, config.max_seq_len)
            )

    def forward(self, x):
        B, T, C = x.size()

        # QKV projection and split
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        if self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = att @ v

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(self, x):
        # Pre-norm: norm before attention/FFN
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTConfig:
    """Configuration for GPT model."""
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=None,
        max_seq_len=2048,
        dropout=0.1,
        bias=False
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff or 4 * d_model  # Default FFN size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias


class GPTModel(nn.Module):
    """Complete GPT-style transformer language model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (share token embedding and output weights)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params():,}")

    def _init_weights(self, module):
        """Initialize weights with small values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=False):
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (batch, seq_len) - input token indices
            targets: (batch, seq_len) - target token indices for loss

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar (if targets provided)
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"

        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """
        Generate text autoregressively.

        Args:
            idx: (batch, seq_len) - starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: keep only top k tokens
            top_p: nucleus sampling threshold

        Returns:
            generated: (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# Example usage
def example_gpt():
    # Small GPT config
    config = GPTConfig(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=1024
    )

    # Create model
    model = GPTModel(config)

    # Sample input
    batch_size = 4
    seq_len = 128
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(idx, targets=idx)

    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'N/A'}")

    # Generate
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)

    print(f"\\nGenerated shape: {generated.shape}")


if __name__ == "__main__":
    example_gpt()
\`\`\`` + `

### Model Sizes

Common GPT configurations:

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| GPT-2 Small | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 350M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

### Training Script

\`\`\`python
def train_gpt(model, train_data, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    for epoch in range(config.epochs):
        for batch in train_data:
            input_ids = batch['input_ids'].cuda()

            # Forward
            logits, loss = model(input_ids[:, :-1], input_ids[:, 1:])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
\`\`\`

### Practice Exercises

- [ ] Build GPT from scratch
- [ ] Train on small dataset
- [ ] Implement greedy/beam search
- [ ] Add KV caching for faster inference
- [ ] Profile memory usage
- [ ] Compare with HuggingFace implementation`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Transformers & LLMs path...');

	const pathResult = db.insert(schema.paths).values({
		name: transformersPath.name,
		description: transformersPath.description,
		color: transformersPath.color,
		language: transformersPath.language,
		skills: transformersPath.skills,
		startHint: transformersPath.startHint,
		difficulty: transformersPath.difficulty,
		estimatedWeeks: transformersPath.estimatedWeeks,
		schedule: transformersPath.schedule
	}).returning().get();

	console.log(`Created path: ${transformersPath.name}`);

	for (let i = 0; i < transformersPath.modules.length; i++) {
		const mod = transformersPath.modules[i];
		const moduleResult = db.insert(schema.modules).values({
			pathId: pathResult.id,
			name: mod.name,
			description: mod.description,
			orderIndex: i
		}).returning().get();

		console.log(`  Created module: ${mod.name}`);

		for (let j = 0; j < mod.tasks.length; j++) {
			const task = mod.tasks[j];
			db.insert(schema.tasks).values({
				moduleId: moduleResult.id,
				title: task.title,
				description: task.description,
				details: task.details,
				orderIndex: j,
				completed: false
			}).run();
		}
		console.log(`    Added ${mod.tasks.length} tasks`);
	}

	console.log('\nSeeding complete!');
}

seed().catch(console.error);
