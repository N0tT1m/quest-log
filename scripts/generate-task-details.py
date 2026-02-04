#!/usr/bin/env python3
"""Generate comprehensive task details for all tasks missing details."""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "quest-log.db"
OUTPUT_DIR = Path(__file__).parent / "task-details"


def generate_details(task: dict) -> str:
    """Generate comprehensive markdown details for a task."""
    title = task["title"].lower()
    desc = (task["description"] or "").lower()
    path = task["path_name"].lower()

    details = []

    # Overview section
    details.append("## Overview")

    # Generate context-aware overview
    if "attention" in title or "self-attention" in title:
        details.append("Attention mechanisms allow models to focus on relevant parts of the input when producing output. Self-attention computes relationships between all positions in a sequence, enabling context-aware representations.")
    elif "transformer" in title:
        details.append("Transformers are the foundation of modern NLP, using self-attention to process sequences in parallel. Understanding the architecture is essential for working with LLMs.")
    elif "lora" in title:
        details.append("LoRA (Low-Rank Adaptation) enables efficient fine-tuning by adding small trainable matrices to frozen pretrained weights, reducing memory and compute requirements by 10-100x.")
    elif "quantiz" in title:
        details.append("Quantization reduces model precision (FP32→INT8→INT4) to decrease memory usage and increase inference speed while maintaining acceptable accuracy.")
    elif "kv cache" in title or "kv-cache" in title:
        details.append("KV caching stores computed Key and Value tensors from previous tokens during autoregressive generation, avoiding redundant computation and dramatically speeding up inference.")
    elif "softmax" in title:
        details.append("Softmax converts logits to probabilities that sum to 1. It's used in attention weights and classification outputs. Numerical stability requires subtracting the max before exponentiating.")
    elif "matrix" in title and "multiplic" in title:
        details.append("Matrix multiplication is fundamental to neural networks. The shape rule (m×n)@(n×p)=(m×p) must be internalized for understanding attention and layer operations.")
    elif "broadcast" in title:
        details.append("Broadcasting allows operations on tensors of different shapes by automatically expanding dimensions. Rules: align from right, dimensions match if equal or one is 1.")
    elif "gradient" in title:
        details.append("Gradients indicate the direction and magnitude of steepest increase for a function. In optimization, we move opposite to gradients (descent) to minimize loss.")
    elif "backprop" in title or "chain rule" in title:
        details.append("Backpropagation applies the chain rule to compute gradients through a computation graph. Each operation's gradient is multiplied along the path from output to parameters.")
    elif "loss" in title or "cross-entropy" in title:
        details.append("Loss functions measure the difference between predictions and targets. Cross-entropy loss is standard for classification: CE = -log(probability of correct class).")
    elif "embedding" in title:
        details.append("Embeddings map discrete tokens to dense vectors in continuous space. Position embeddings inject sequence order information since attention is position-invariant.")
    elif "sampling" in title or "top-k" in title or "top-p" in title:
        details.append("Sampling strategies control generation diversity. Greedy picks highest probability, top-k samples from k best, top-p (nucleus) samples from smallest set with cumulative probability ≥p.")
    elif "fine-tun" in title:
        details.append("Fine-tuning adapts pretrained models to specific tasks. Full fine-tuning updates all parameters; parameter-efficient methods like LoRA update only a small subset.")
    elif "rlhf" in title:
        details.append("RLHF aligns models with human preferences through: 1) supervised fine-tuning on demonstrations, 2) reward model training on preferences, 3) RL optimization with KL penalty.")
    elif "dpo" in title:
        details.append("Direct Preference Optimization simplifies RLHF by directly optimizing on preference pairs without a separate reward model, using a clever reparameterization of the RL objective.")
    elif "flash attention" in title:
        details.append("Flash Attention computes attention in tiles without materializing the full N×N attention matrix, reducing memory from O(N²) to O(N) and improving speed through better memory access patterns.")
    elif "speculative" in title:
        details.append("Speculative decoding uses a fast draft model to generate candidate tokens, then verifies them in parallel with the large model, achieving 2-3x speedup for autoregressive generation.")
    elif "vllm" in title or "serving" in title:
        details.append("High-throughput LLM serving requires PagedAttention for efficient KV cache management, continuous batching to maximize GPU utilization, and optimized CUDA kernels.")
    elif "rag" in title or "retrieval" in title:
        details.append("RAG augments LLMs with external knowledge by embedding documents, retrieving relevant chunks for each query, and injecting them into the prompt context.")
    elif "agent" in title:
        details.append("LLM agents combine reasoning with tool use. They plan actions, execute tools, observe results, and iterate until the task is complete. Common patterns include ReAct and chain-of-thought.")
    # Security-specific
    elif "exploit" in title or "payload" in title:
        details.append("Exploit development requires understanding target vulnerabilities, crafting payloads that achieve code execution, and bypassing security controls like ASLR, DEP, and CFI.")
    elif "shellcode" in title:
        details.append("Shellcode is position-independent machine code that performs actions like spawning shells. It must avoid null bytes and other bad characters, and account for the target architecture.")
    elif "injection" in title:
        details.append("Injection attacks insert malicious input that's interpreted as code or commands. Types include SQL injection, command injection, LDAP injection, and XSS.")
    elif "privilege" in title and "escal" in title:
        details.append("Privilege escalation exploits misconfigurations or vulnerabilities to gain higher access levels. Common vectors: SUID binaries, sudo rules, kernel exploits, service misconfigurations.")
    elif "lateral" in title or "pivot" in title:
        details.append("Lateral movement techniques allow attackers to spread through a network after initial access. Methods include pass-the-hash, token manipulation, and remote service exploitation.")
    elif "c2" in title or "command and control" in title:
        details.append("C2 frameworks provide remote control over compromised systems. They handle communication, task execution, and data exfiltration while evading detection.")
    elif "active directory" in title or "kerberos" in title:
        details.append("Active Directory attacks target Windows domain infrastructure. Key techniques: Kerberoasting, AS-REP roasting, DCSync, Golden/Silver tickets, and delegation abuse.")
    elif "password" in title and ("crack" in title or "hash" in title):
        details.append("Password cracking recovers plaintext from hashes using dictionaries, rules, and brute force. GPU acceleration (hashcat) is essential for modern hash types.")
    elif "phishing" in title or "social engineer" in title:
        details.append("Social engineering manipulates people into revealing information or performing actions. Technical controls are bypassed through human psychology and trust exploitation.")
    elif "evasion" in title or "bypass" in title:
        details.append("Evasion techniques avoid detection by security controls. Methods include obfuscation, encryption, living-off-the-land binaries, and timing-based detection bypass.")
    elif "forensic" in title:
        details.append("Digital forensics investigates security incidents through artifact analysis. Key areas: memory forensics, disk forensics, network forensics, and malware analysis.")
    # Systems programming
    elif "tcp" in title or "socket" in title:
        details.append("TCP sockets provide reliable, ordered byte streams. Implementation requires handling: connection establishment, buffering, flow control, and graceful shutdown.")
    elif "http" in title and "server" in title:
        details.append("HTTP servers parse requests, route to handlers, and format responses. Key considerations: keep-alive connections, chunked transfer encoding, and proper header handling.")
    elif "dns" in title:
        details.append("DNS translates domain names to IP addresses. Implementation covers: packet parsing, recursive resolution, caching, and handling various record types (A, AAAA, CNAME, MX).")
    elif "container" in title or "namespace" in title:
        details.append("Linux containers use namespaces for isolation (PID, network, mount, UTS, user) and cgroups for resource limits. The OCI spec defines container runtime standards.")
    elif "compiler" in title or "parser" in title:
        details.append("Compilers transform source code through: lexing (tokens), parsing (AST), semantic analysis, optimization, and code generation. Each phase has well-defined responsibilities.")
    elif "debug" in title and ("er" in title or "ging" in title):
        details.append("Debuggers use ptrace to control process execution, set breakpoints via INT3 instructions, and inspect memory/registers. DWARF format provides debug symbol information.")
    elif "memory" in title and "allocat" in title:
        details.append("Memory allocators manage heap memory through strategies like free lists, buddy allocation, or slab allocation. Key concerns: fragmentation, thread safety, and cache efficiency.")
    elif "thread" in title and "pool" in title:
        details.append("Thread pools maintain worker threads that process tasks from a queue. Benefits: amortized thread creation cost, bounded concurrency, and controlled resource usage.")
    elif "async" in title or "runtime" in title:
        details.append("Async runtimes use event loops and non-blocking I/O to handle many concurrent operations with few threads. Key components: executor, reactor, and Future abstraction.")
    # Database
    elif "redis" in title:
        details.append("Redis is an in-memory data structure store. Reimplementation covers: RESP protocol parsing, data structures (strings, lists, sets, hashes), persistence, and pub/sub.")
    elif "sqlite" in title:
        details.append("SQLite is a self-contained database engine. Key components: SQL parser, query planner, B-tree storage engine, pager for disk I/O, and WAL for transactions.")
    elif "b-tree" in title or "btree" in title:
        details.append("B-trees are self-balancing trees optimized for disk access. Operations maintain sorted order with O(log n) complexity. Node splits and merges handle insertions/deletions.")
    elif "lsm" in title:
        details.append("LSM trees optimize write performance by buffering writes in memory, then flushing to sorted runs on disk. Compaction merges runs to maintain read performance.")
    else:
        # Generic but useful overview
        if task["description"]:
            details.append(f"{task['description']}")
        else:
            details.append(f"This task covers essential concepts for {task['path_name']}. Complete it to build foundational skills.")

    # Implementation section with code examples
    details.append("\n### Implementation")

    if "attention" in title and "scratch" in desc:
        details.append("""```python
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```""")
    elif "lora" in title:
        details.append("""```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scale = alpha / rank

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scale
```""")
    elif "softmax" in title:
        details.append("""```python
def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)  # Numerical stability
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```""")
    elif "kv cache" in title or "kv-cache" in title:
        details.append("""```python
def generate_with_cache(model, input_ids, max_new_tokens):
    past_kv = None
    for _ in range(max_new_tokens):
        out = model(input_ids[:, -1:] if past_kv else input_ids, past_kv=past_kv)
        past_kv = out.past_kv
        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```""")
    elif "tcp" in title and "listen" in title:
        details.append("""```go
listener, err := net.Listen("tcp", ":8080")
if err != nil { log.Fatal(err) }
defer listener.Close()

for {
    conn, err := listener.Accept()
    if err != nil { continue }
    go handleConnection(conn)
}
```""")
    elif "http" in title and "server" in title:
        details.append("""```python
import socket

def handle_request(conn):
    request = conn.recv(4096).decode()
    method, path, _ = request.split('\\r\\n')[0].split(' ')
    response = f"HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\n\\r\\nHello"
    conn.send(response.encode())
    conn.close()
```""")
    else:
        details.append("Follow the implementation steps in the task description. Start with a minimal working version, then iterate.")

    # Key concepts
    details.append("\n### Key Concepts")
    details.append("- Understand the core algorithm before coding")
    details.append("- Handle edge cases explicitly")
    details.append("- Test against reference implementations")
    details.append("- Profile for performance bottlenecks")

    # Practice exercises
    details.append("\n### Practice")
    details.append("- [ ] Implement from scratch without references")
    details.append("- [ ] Test with edge cases")
    details.append("- [ ] Compare output with established libraries")
    details.append("- [ ] Optimize for production use")

    # Completion criteria
    details.append("\n### Completion Criteria")
    details.append("- [ ] Code produces correct output")
    details.append("- [ ] Can explain the implementation to others")
    details.append("- [ ] Edge cases handled properly")
    details.append("- [ ] Performance is acceptable")

    return "\n".join(details)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all tasks needing details
    cursor.execute("""
        SELECT t.id, t.title, t.description, p.name as path_name, p.id as path_id
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE t.details IS NULL OR length(t.details) < 100
        ORDER BY p.id, m.order_index, t.order_index
    """)

    tasks = [dict(row) for row in cursor.fetchall()]
    conn.close()

    print(f"Processing {len(tasks)} tasks...")

    # Group by path
    paths = defaultdict(lambda: {"pathName": "", "pathId": 0, "tasks": []})

    for task in tasks:
        path_id = task["path_id"]
        if not paths[path_id]["pathName"]:
            paths[path_id]["pathName"] = task["path_name"]
            paths[path_id]["pathId"] = path_id

        paths[path_id]["tasks"].append({
            "id": task["id"],
            "title": task["title"],
            "details": generate_details(task)
        })

    # Categorize into output files
    categories = {
        "ai-ml": [],
        "security-redteam": [],
        "systems": [],
        "networking": [],
        "databases": [],
        "projects": [],
        "specialized": []
    }

    for path_id, path_data in paths.items():
        name = path_data["pathName"].lower()

        if any(x in name for x in ["ai", "ml", "deep learning", "transformer", "llm", "neural"]):
            categories["ai-ml"].append(path_data)
        elif any(x in name for x in ["red team", "security", "hacking", "ctf", "exploit", "malware", "offensive", "defensive"]):
            categories["security-redteam"].append(path_data)
        elif any(x in name for x in ["build your own", "compiler", "shell", "debugger", "container", "memory", "thread", "async"]):
            categories["systems"].append(path_data)
        elif any(x in name for x in ["http", "tcp", "dns", "network", "packet", "load balancer"]):
            categories["networking"].append(path_data)
        elif any(x in name for x in ["redis", "sqlite", "database", "lsm", "b-tree", "key-value"]):
            categories["databases"].append(path_data)
        elif any(x in name for x in ["reimplement"]):
            categories["specialized"].append(path_data)
        else:
            categories["projects"].append(path_data)

    # Write output files
    OUTPUT_DIR.mkdir(exist_ok=True)

    total_tasks = 0
    for category, data in categories.items():
        if data:
            output_path = OUTPUT_DIR / f"{category}.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            task_count = sum(len(p["tasks"]) for p in data)
            total_tasks += task_count
            print(f"Wrote {task_count} tasks to {output_path}")

    print(f"\nTotal: {total_tasks} tasks with details generated")


if __name__ == "__main__":
    main()
