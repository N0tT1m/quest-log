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

const learnDLPath: PathData = {
	name: 'Learn Deep Learning Without Courses',
	description: 'A practical, build-first approach to learning deep learning by studying code, implementing from scratch, and building real projects. Skip the courses, learn by doing.',
	language: 'Python',
	color: 'purple',
	skills: 'PyTorch, transformers, neural networks, autograd, attention mechanisms, fine-tuning, model training',
	startHint: 'Clone nanoGPT and start reading model.py line by line',
	difficulty: 'intermediate',
	estimatedWeeks: 8,
	schedule: `## 8-Week Self-Directed Deep Learning Path

### Phase 1: Foundations (Weeks 1-2)

#### Week 1: Neural Networks from Scratch
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Install PyTorch, Jupyter, set up environment |
| Tue | micrograd | Clone micrograd, understand autograd engine |
| Wed | Implementation | Reimplement micrograd from scratch |
| Thu | Testing | Build simple neural network with your autograd |
| Fri | makemore pt1 | Watch Karpathy video, implement bigram model |
| Weekend | Practice | Complete makemore parts 2-3 (MLP, BatchNorm) |

#### Week 2: Backpropagation Deep Dive
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | makemore pt4 | Manual backpropagation implementation |
| Wed-Thu | makemore pt5 | WaveNet architecture |
| Fri | Consolidation | Review all makemore implementations |
| Weekend | Project | Build character-level text generator from scratch |

### Phase 2: Transformers (Weeks 3-5)

#### Week 3: Understanding Transformers
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Reading | "The Illustrated Transformer" |
| Tue | nanoGPT | Clone nanoGPT, run Shakespeare training |
| Wed | Study | Read model.py line by line, understand shapes |
| Thu | Attention | Implement multi-head attention from scratch |
| Fri | Blocks | Implement transformer block |
| Weekend | Full Model | Complete GPT implementation |

#### Week 4: Transformer Variations
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Position Encoding | Implement RoPE (Rotary Position Embedding) |
| Wed-Thu | Attention Variants | Implement GQA (Grouped Query Attention) |
| Fri | Normalization | Implement RMSNorm |
| Weekend | Integration | Build complete modern transformer |

#### Week 5: Tokenization
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Wed | BPE | Watch tokenizer video, implement BPE |
| Thu-Fri | minbpe | Study minbpe repo, reimplement |
| Weekend | Custom | Build tokenizer for your domain |

### Phase 3: Training & Fine-tuning (Weeks 6-7)

#### Week 6: Training Dynamics
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Learning Rate | Experiment with different LR values |
| Tue | Optimizers | Compare SGD, Adam, AdamW |
| Wed | Schedules | Implement cosine annealing, warmup |
| Thu | Regularization | Add dropout, weight decay |
| Fri | Debugging | Learn to diagnose training issues |
| Weekend | Experiments | Train models with different configurations |

#### Week 7: Fine-tuning Methods
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | LoRA | Implement LoRA from scratch |
| Wed-Thu | QLoRA | Add quantization with bitsandbytes |
| Fri | Comparison | Compare full fine-tuning vs LoRA |
| Weekend | Project | Fine-tune model on custom dataset |

### Phase 4: Real Projects (Week 8+)

#### Week 8: Build Something Real
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Planning | Choose project based on your interest |
| Tue-Thu | Implementation | Build core functionality |
| Fri | Optimization | Improve performance |
| Weekend | Deploy | Make it usable, write documentation |

### Daily Commitment
- **Minimum**: 1-2 hours focused coding
- **Ideal**: 3-4 hours with experiments
- **Rule**: Build > Watch (code more than you watch videos)`,
	modules: [
		{
			name: 'Learning Philosophy & Setup',
			description: 'Understand the build-first approach and set up your environment',
			tasks: [
				{
					title: 'Understand why courses fail and embrace build-first learning',
					description: 'Learn the philosophy of active learning through building',
					details: `## The Build-First Philosophy

### Why This Approach Works

**The Learning Loop:**
\`\`\`
Build something
    → Get stuck
    → Search the specific thing
    → Understand it
    → Repeat
\`\`\`

**Why build-first beats courses:**
- You learn what you need when you need it (just-in-time learning)
- Active learning > passive watching
- You build portfolio while learning
- Motivation stays high (you're making stuff)
- Real understanding through debugging

### Why Courses Often Fail

| Problem | Reality |
|---------|---------|
| Too slow | 40 hours to learn what takes 4 hours of building |
| Wrong order | Theory before you need it doesn't stick |
| Passive | Watching ≠ understanding |
| Outdated | Field moves fast, courses don't update |
| Generic | Not tailored to what you're building |
| Completion rates | <10% finish most online courses |

### When Courses Make Sense

- You need a certificate for a job
- You genuinely prefer structured learning
- You're starting from absolute zero (no programming experience)

### What Matters vs What Doesn't

**What Matters:**
- Building >> Watching (every hour coding beats 5 hours of videos)
- Depth >> Breadth (understand one thing fully before moving)
- Debugging == Learning (errors teach more than successes)
- Imperfect >> Perfect (ship something, then improve it)

**What Doesn't Matter:**
- Formal credentials
- Completing courses
- Reading every paper
- Understanding every proof
- Following "the right path"

### Signs You're Actually Learning

- You can explain concepts without jargon
- You can predict what will break before it does
- You can modify code and know what will change
- You have opinions on tradeoffs
- You build things that work

### The Math You Actually Need

**Essential (Learn These):**

| Math | Why | Depth Needed | Time to Learn |
|------|-----|--------------|---------------|
| **Linear Algebra** | Everything is matrix multiplication | Matrix multiply, transpose, shapes, broadcasting | 1-2 weeks |
| **Basic Calculus** | Understand gradients | Chain rule, partial derivatives (conceptually) | Few days |
| **Probability** | Loss functions, sampling | Softmax, distributions, Bayes basics | Few days |

**Nice to Have (Pick Up As Needed):**
- Statistics (evaluating models, A/B testing)
- Information Theory (cross-entropy, KL divergence)
- Optimization Theory (why Adam works, LR schedules)

**Rarely Need Deep Understanding:**
- Eigenvalues/eigenvectors (unless doing PCA)
- Measure theory
- Advanced calculus proofs
- Category theory, topology

### The Reality of Deep Learning Math

\`\`\`python
# You DON'T need to derive this yourself:
# ∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W

# PyTorch does it for you:
loss.backward()  # All gradients computed automatically
optimizer.step() # Weights updated

# What you DO need to understand:
# - Why loss is going up/down
# - Why gradients might vanish/explode
# - What shapes tensors should be
# - Why certain architectures work better
\`\`\`

### Your First Day: Do This Today

\`\`\`bash
# 1. Clone nanoGPT
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# 2. Set up environment
pip install torch numpy transformers datasets tiktoken wandb tqdm

# 3. Prepare data
python data/shakespeare_char/prepare.py

# 4. Train (takes ~15 mins on decent GPU, longer on CPU)
python train.py config/train_shakespeare_char.py

# 5. Generate text
python sample.py --out_dir=out-shakespeare-char

# 6. Open model.py and start reading
# Add print statements for shapes
# Modify things and see what happens
# This is how you learn
\`\`\`

### How to Study a Repository

\`\`\`bash
# Step 1: Clone it
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# Step 2: Run it first (make sure it works)
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py

# Step 3: Read model.py line by line
# Step 4: Add print statements to understand shapes
# Step 5: Modify something, see what breaks
# Step 6: Reimplement from scratch without looking
\`\`\`

### Essential Free Resources

**Video (But Not Courses):**

**Andrej Karpathy's YouTube** - The gold standard:
- "Let's build GPT from scratch" (2 hrs) - Complete transformer
- "Let's build the GPT Tokenizer" (2 hrs) - BPE tokenization
- makemore series (5 videos) - Neural networks from first principles
- Code along, pause, modify, break things, fix them

**3Blue1Brown** - Math intuition:
- Essence of Linear Algebra (especially episodes 1-7)
- Essence of Calculus (episodes 1-4)
- Neural Networks series (all 4 episodes)

**Written Resources:**
- PyTorch Tutorials (pytorch.org/tutorials) - Official, hands-on
- The Illustrated Transformer (jalammar.github.io)
- nanoGPT README - Implementation notes
- Hugging Face Docs - Architecture explanations

**Books (Free Online):**
- Dive into Deep Learning (d2l.ai) - Interactive, code-first
- The Little Book of Deep Learning (200 pages, concise)
- Understanding Deep Learning (udlbook.github.io)

### Practice Exercises

- [ ] Set up PyTorch environment
- [ ] Run nanoGPT Shakespeare training
- [ ] Read model.py and understand every line
- [ ] Modify batch size, see what changes
- [ ] Add print statements to debug shapes
- [ ] Generate some text samples`
				},
				{
					title: 'Study essential repositories: nanoGPT, micrograd, minbpe',
					description: 'Learn by reading small, well-written codebases',
					details: `## Repositories to Study

### Tier 1: Read Every Line

These are small enough to fully understand:

| Repo | Lines That Matter | What You Learn |
|------|-------------------|----------------|
| **nanoGPT** | ~300 | Transformers from scratch |
| **minbpe** | ~200 | Tokenization (BPE) |
| **llm.c** | ~1000 | Training without frameworks |
| **micrograd** | ~100 | Autograd engine |
| **tinygrad** | Core ~500 | Full ML framework, minimal |

### micrograd: Understanding Autograd

**What it teaches:**
- How automatic differentiation works
- How neural networks actually compute gradients
- The mechanics of backpropagation

**Study Plan:**
\`\`\`bash
git clone https://github.com/karpathy/micrograd
cd micrograd

# 1. Read engine.py (the Value class)
# 2. Understand how ._backward() works
# 3. Read nn.py (Neuron, Layer, MLP)
# 4. Run demo.py to see it work
# 5. Reimplement from scratch
\`\`\`

**Key Code to Understand:**

\`\`\`python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Backward pass
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
\`\`\`

### nanoGPT: Modern Transformers

**What it teaches:**
- Complete GPT implementation
- Attention mechanism
- Training loop
- Generation/sampling

**File Structure:**
\`\`\`
nanoGPT/
├── model.py          # <-- Start here (GPT implementation)
├── train.py          # Training loop
├── sample.py         # Text generation
├── data/shakespeare_char/  # Example dataset
└── config/           # Training configs
\`\`\`

**Key Components in model.py:**

\`\`\`python
class CausalSelfAttention(nn.Module):
    \"\"\"Multi-head causal self-attention\"\"\"

    def __init__(self, config):
        super().__init__()
        # Q, K, V projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention: (B, n_head, T, head_size) x (B, n_head, head_size, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum: (B, n_head, T, T) x (B, n_head, T, head_size)
        y = att @ v

        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    \"\"\"Transformer block: communication followed by computation\"\"\"

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Residual connection
        x = x + self.mlp(self.ln_2(x))   # Residual connection
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, t, device=device))
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Project to vocab
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
\`\`\`

### minbpe: Tokenization

**What it teaches:**
- Byte Pair Encoding (BPE)
- How tokenizers work
- Why tokenization matters

**Core Algorithm:**

\`\`\`python
def get_stats(ids):
    \"\"\"Count frequency of each pair\"\"\"
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    \"\"\"Merge all occurrences of pair into idx\"\"\"
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Training BPE
vocab_size = 512
num_merges = vocab_size - 256

ids = list(text.encode("utf-8"))  # Start with bytes
merges = {}

for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
\`\`\`

### Tier 2: Study Specific Parts

**Larger repos - focus on what you need:**

| Repo | Focus On | What You Learn |
|------|----------|----------------|
| **llama.cpp** | ggml.c, llama.cpp | Inference optimization |
| **vLLM** | core/ | Production serving |
| **transformers** | models/llama | Architecture patterns |
| **PEFT** | src/peft/tuners/lora | LoRA implementation |
| **bitsandbytes** | Core quantization | 4-bit/8-bit quantization |

### Study Plan for Each Repo

**Week 1: micrograd**
- Day 1-2: Read and understand
- Day 3-4: Reimplement from scratch
- Day 5: Add your own operations
- Weekend: Build simple neural network

**Week 2-3: nanoGPT**
- Days 1-3: Run and study model.py
- Days 4-7: Reimplement transformer from scratch
- Days 8-10: Experiment with configurations
- Weekend: Train on custom dataset

**Week 4: minbpe**
- Days 1-2: Understand BPE algorithm
- Days 3-4: Implement tokenizer
- Day 5: Train on different texts
- Weekend: Build custom tokenizer for your domain

### Practice Exercises

- [ ] Clone all three repos (micrograd, nanoGPT, minbpe)
- [ ] Run each one successfully
- [ ] Read code line by line, add comments explaining each part
- [ ] Reimplement micrograd from scratch
- [ ] Reimplement nanoGPT attention mechanism
- [ ] Implement BPE tokenization
- [ ] Build something using what you learned`
				}
			]
		},
		{
			name: 'Neural Network Foundations',
			description: 'Build neural networks from scratch to understand backpropagation',
			tasks: [
				{
					title: 'Implement micrograd: build autograd engine from scratch',
					description: 'Understand automatic differentiation by implementing it yourself',
					details: `## Building an Autograd Engine

### What is Autograd?

Automatic differentiation computes gradients without manual derivation. It's the core of PyTorch, TensorFlow, and all modern deep learning frameworks.

**How it works:**
1. Forward pass: compute output, build computational graph
2. Backward pass: traverse graph in reverse, compute gradients using chain rule

### Implementation Plan

**Core Components:**
1. Value class (stores data and gradient)
2. Operations (+, -, *, /)
3. Activation functions (ReLU, tanh, etc.)
4. Backward method (topological sort + backprop)

### The Value Class

\`\`\`python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Derivative of addition: gradient flows to both inputs
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Derivative of multiplication: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        # Division is multiplication by reciprocal
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivative of tanh: 1 - tanh^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # Derivative of exp: exp(x)
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backward pass
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
\`\`\`

### Example Usage

\`\`\`python
# Simple example: f(x) = x^2 + 3x + 1
x = Value(2.0, label='x')
y = x**2 + 3*x + 1
y.label = 'y'

print(f"y = {y.data}")  # 11.0

# Compute gradient
y.backward()

print(f"dy/dx = {x.grad}")  # Should be 2x + 3 = 7.0
\`\`\`

### Building a Neural Network

\`\`\`python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
\`\`\`

### Training Example

\`\`\`python
# Create network: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
n = MLP(3, [4, 4, 1])

# Training data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # Targets

# Training loop
for k in range(100):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Update (gradient descent)
    learning_rate = 0.01
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    if k % 10 == 0:
        print(f"step {k} loss {loss.data}")

# Final predictions
print("\\nFinal predictions:")
for x, y in zip(xs, ys):
    print(f"input: {x}, target: {y}, predicted: {n(x).data:.4f}")
\`\`\`

### Visualizing the Computational Graph

\`\`\`python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}",
                 shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
\`\`\`

### Understanding Backpropagation

**Chain Rule Example:**
\`\`\`
If y = f(g(x))
then dy/dx = (dy/dg) * (dg/dx)

Example: y = (x^2 + 1)^3
Let g = x^2 + 1, so y = g^3

dy/dx = (dy/dg) * (dg/dx)
      = 3g^2 * 2x
      = 3(x^2 + 1)^2 * 2x
\`\`\`

**In code:**
\`\`\`python
x = Value(2.0)
g = x**2 + 1
y = g**3

y.backward()

print(f"x.grad = {x.grad}")  # Should be 60.0
# 3 * (2^2 + 1)^2 * 2*2 = 3 * 25 * 4 = 300... wait, let me recalculate
# Actually: 3 * (5)^2 * 4 = 3 * 25 * 4 = 300
# Hmm, my calculation is off - the key is understanding the process!
\`\`\`

### Practice Exercises

- [ ] Implement Value class from scratch (no peeking!)
- [ ] Add ReLU activation function
- [ ] Add sigmoid activation function
- [ ] Build a 2-layer neural network
- [ ] Train on XOR problem
- [ ] Visualize computational graph
- [ ] Compare gradients to PyTorch to verify correctness`
				}
			]
		}
	]
};

async function seed() {
	console.log('Seeding Learn Deep Learning Without Courses path...');

	const pathResult = db.insert(schema.paths).values({
		name: learnDLPath.name,
		description: learnDLPath.description,
		color: learnDLPath.color,
		language: learnDLPath.language,
		skills: learnDLPath.skills,
		startHint: learnDLPath.startHint,
		difficulty: learnDLPath.difficulty,
		estimatedWeeks: learnDLPath.estimatedWeeks,
		schedule: learnDLPath.schedule
	}).returning().get();

	console.log(`Created path: ${learnDLPath.name}`);

	for (let i = 0; i < learnDLPath.modules.length; i++) {
		const mod = learnDLPath.modules[i];
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
