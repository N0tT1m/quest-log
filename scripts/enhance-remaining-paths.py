#!/usr/bin/env python3
"""
Enhance tasks in paths with below-average details (under 900 chars).
Targets 23 paths including AI/ML, Red Team Tooling, Build Your Own, and Project paths.
"""

import sqlite3
import re

DB_PATH = "data/quest-log.db"

# Task-specific detailed content keyed by lowercase task title keywords
TASK_DETAILS = {
    # ============== AI/ML PATHS ==============
    "error handling, retries": """## Overview
Production AI applications need robust error handling for API failures, rate limits, and timeouts.

### Implementation
```python
import time
from functools import wraps
from typing import TypeVar, Callable
import openai

T = TypeVar('T')

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (openai.RateLimitError, openai.APITimeoutError)
):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_openai(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        timeout=30
    )
    return response.choices[0].message.content

# Circuit breaker pattern for cascading failures
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

### Key Concepts
- Exponential backoff prevents thundering herd on rate limits
- Circuit breaker prevents cascading failures
- Distinguish retryable vs non-retryable errors
- Set appropriate timeouts for each operation

### Completion Criteria
- [ ] Implement retry decorator with exponential backoff
- [ ] Handle rate limits, timeouts, and API errors gracefully
- [ ] Add circuit breaker for downstream service protection
- [ ] Log all retry attempts for debugging""",

    "prompt engineering": """## Overview
Effective prompt engineering dramatically improves AI output quality through structured prompts, examples, and output parsing.

### Implementation
```python
from pydantic import BaseModel
from typing import Optional
import json
import openai

# Structured output with Pydantic
class ProductReview(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_points: list[str]
    suggested_response: Optional[str]

def analyze_review(review_text: str) -> ProductReview:
    system_prompt = '''You are a product review analyzer.
    Analyze reviews and return structured JSON with:
    - sentiment: "positive", "negative", or "neutral"
    - confidence: 0.0 to 1.0
    - key_points: list of main points
    - suggested_response: optional response if negative'''

    # Few-shot examples improve consistency
    few_shot = '''Example input: "Great product but shipping was slow"
    Example output: {"sentiment": "positive", "confidence": 0.7,
    "key_points": ["product quality praised", "shipping complaint"],
    "suggested_response": "Thank you! We're working on faster shipping."}'''

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": few_shot},
            {"role": "user", "content": f"Analyze: {review_text}"}
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return ProductReview(**data)

# Chain of thought prompting
def solve_with_reasoning(problem: str) -> dict:
    prompt = f'''Solve this step by step:
    Problem: {problem}

    Think through each step:
    1. Identify what we know
    2. Identify what we need to find
    3. Choose an approach
    4. Execute step by step
    5. Verify the answer

    Format: {{"reasoning": [...steps...], "answer": "final answer"}}'''

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

### Key Concepts
- System prompts set behavior and constraints
- Few-shot examples improve output consistency
- JSON mode ensures parseable structured output
- Chain-of-thought improves reasoning tasks

### Completion Criteria
- [ ] Design system prompts for your use case
- [ ] Add 2-3 few-shot examples per task type
- [ ] Implement JSON output parsing with validation
- [ ] Test prompt variations and measure quality""",

    "usage tracking": """## Overview
Track API usage, costs, and performance metrics to optimize AI applications and control spending.

### Implementation
```python
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3

@dataclass
class APICall:
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None

class UsageTracker:
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, db_path: str = "usage.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            model TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            latency_ms REAL,
            cost_usd REAL,
            success INTEGER,
            error TEXT
        )''')
        self.conn.commit()

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000

    def track_call(self, call: APICall):
        self.conn.execute('''INSERT INTO api_calls
            (timestamp, model, prompt_tokens, completion_tokens, latency_ms, cost_usd, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (call.timestamp.isoformat(), call.model, call.prompt_tokens,
             call.completion_tokens, call.latency_ms, call.cost_usd, call.success, call.error))
        self.conn.commit()

    def get_daily_stats(self, date: str) -> dict:
        cursor = self.conn.execute('''
            SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens),
                   SUM(cost_usd), AVG(latency_ms), SUM(CASE WHEN success THEN 1 ELSE 0 END)
            FROM api_calls WHERE date(timestamp) = ?
        ''', (date,))
        row = cursor.fetchone()
        return {
            "total_calls": row[0],
            "total_prompt_tokens": row[1],
            "total_completion_tokens": row[2],
            "total_cost_usd": row[3],
            "avg_latency_ms": row[4],
            "success_rate": row[5] / row[0] if row[0] > 0 else 0
        }
```

### Key Concepts
- Track every API call with tokens and costs
- Store metrics in database for analysis
- Set up alerts for cost thresholds
- Monitor latency and error rates

### Completion Criteria
- [ ] Implement usage tracking for all API calls
- [ ] Calculate costs based on model pricing
- [ ] Build dashboard showing daily/weekly metrics
- [ ] Set up budget alerts and rate limiting""",

    "cli tool that uses": """## Overview
Build a command-line interface that leverages LLM APIs for practical tasks like code review, documentation, or data analysis.

### Implementation
```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import openai

def analyze_code(file_path: str, task: str) -> str:
    code = Path(file_path).read_text()

    prompts = {
        "review": f"Review this code for bugs, security issues, and improvements:\\n```\\n{code}\\n```",
        "document": f"Generate docstrings and comments for this code:\\n```\\n{code}\\n```",
        "explain": f"Explain what this code does step by step:\\n```\\n{code}\\n```",
        "test": f"Generate unit tests for this code:\\n```\\n{code}\\n```"
    }

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert code reviewer."},
            {"role": "user", "content": prompts.get(task, prompts["review"])}
        ]
    )
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="AI-powered code assistant")
    parser.add_argument("file", help="File to analyze")
    parser.add_argument("-t", "--task", choices=["review", "document", "explain", "test"],
                        default="review", help="Task to perform")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--model", default="gpt-4", help="Model to use")

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: {args.file} not found", file=sys.stderr)
        sys.exit(1)

    result = analyze_code(args.file, args.task)

    if args.output:
        Path(args.output).write_text(result)
        print(f"Output written to {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()
```

### Key Concepts
- Use argparse for clean CLI interface
- Support multiple task types via subcommands
- Handle file I/O and errors gracefully
- Allow model selection and output redirection

### Completion Criteria
- [ ] Build CLI with argparse or click
- [ ] Implement at least 3 useful commands
- [ ] Add proper error handling and help text
- [ ] Package for easy installation (pip install)""",

    "web ui": """## Overview
Build a simple but functional web interface using Streamlit or Gradio to make your AI prototype accessible.

### Implementation
```python
import streamlit as st
import openai
from typing import Generator

st.set_page_config(page_title="AI Assistant", layout="wide")

# Session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_response(prompt: str) -> Generator[str, None, None]:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            *st.session_state.messages,
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

st.title("AI Assistant")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_response(prompt):
            full_response += chunk
            response_placeholder.write(full_response + "▌")
        response_placeholder.write(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

### Key Concepts
- Streamlit provides rapid prototyping
- Session state maintains conversation history
- Streaming responses improve perceived latency
- Sidebar for configuration options

### Completion Criteria
- [ ] Build working chat interface with history
- [ ] Add streaming responses for better UX
- [ ] Implement settings/configuration panel
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces""",

    # ============== CONTAINER RUNTIME ==============
    "uts namespace": """## Overview
UTS (UNIX Time-sharing System) namespace isolates hostname and domain name, allowing containers to have their own identity.

### Implementation
```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "syscall"
)

func main() {
    switch os.Args[1] {
    case "run":
        run()
    case "child":
        child()
    default:
        panic("unknown command")
    }
}

func run() {
    cmd := exec.Command("/proc/self/exe", append([]string{"child"}, os.Args[2:]...)...)
    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    cmd.SysProcAttr = &syscall.SysProcAttr{
        Cloneflags: syscall.CLONE_NEWUTS | syscall.CLONE_NEWPID | syscall.CLONE_NEWNS,
        Unshareflags: syscall.CLONE_NEWNS,
    }

    if err := cmd.Run(); err != nil {
        fmt.Printf("Error: %v\\n", err)
        os.Exit(1)
    }
}

func child() {
    // Set container hostname
    if err := syscall.Sethostname([]byte("container")); err != nil {
        panic(err)
    }

    // Verify isolation
    hostname, _ := os.Hostname()
    fmt.Printf("Container hostname: %s\\n", hostname)

    // Run the actual command
    cmd := exec.Command(os.Args[2], os.Args[3:]...)
    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    cmd.Run()
}
```

### Key Concepts
- UTS namespace created with CLONE_NEWUTS flag
- syscall.Sethostname changes hostname in namespace
- Changes don't affect host system
- Combined with other namespaces for full isolation

### Completion Criteria
- [ ] Create process with UTS namespace isolation
- [ ] Set custom hostname inside container
- [ ] Verify host hostname is unchanged
- [ ] Combine with PID and mount namespaces""",

    "image layer": """## Overview
Container images use layered filesystems where each layer represents a set of file changes, enabling efficient storage and caching.

### Implementation
```go
package main

import (
    "archive/tar"
    "compress/gzip"
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "io"
    "os"
    "path/filepath"
)

type Layer struct {
    Digest  string
    Size    int64
    TarPath string
}

type Image struct {
    Layers []Layer
    Config ImageConfig
}

type ImageConfig struct {
    Cmd        []string
    Env        []string
    WorkingDir string
}

func extractLayer(tarPath, targetDir string) error {
    file, err := os.Open(tarPath)
    if err != nil {
        return err
    }
    defer file.Close()

    gzr, err := gzip.NewReader(file)
    if err != nil {
        return err
    }
    defer gzr.Close()

    tr := tar.NewReader(gzr)

    for {
        header, err := tr.Next()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        target := filepath.Join(targetDir, header.Name)

        // Handle whiteout files (deletions)
        if filepath.Base(header.Name)[:4] == ".wh." {
            deletePath := filepath.Join(filepath.Dir(target), filepath.Base(header.Name)[4:])
            os.RemoveAll(deletePath)
            continue
        }

        switch header.Typeflag {
        case tar.TypeDir:
            os.MkdirAll(target, os.FileMode(header.Mode))
        case tar.TypeReg:
            f, _ := os.OpenFile(target, os.O_CREATE|os.O_WRONLY, os.FileMode(header.Mode))
            io.Copy(f, tr)
            f.Close()
        case tar.TypeSymlink:
            os.Symlink(header.Linkname, target)
        }
    }
    return nil
}

func buildRootfs(image Image, targetDir string) error {
    // Extract layers in order (base first)
    for _, layer := range image.Layers {
        fmt.Printf("Extracting layer %s\\n", layer.Digest[:12])
        if err := extractLayer(layer.TarPath, targetDir); err != nil {
            return err
        }
    }
    return nil
}
```

### Key Concepts
- Layers are tar archives with file changes
- Whiteout files (.wh.*) mark deletions
- Layers extracted in order, later layers override
- Content-addressable storage by SHA256 digest

### Completion Criteria
- [ ] Parse and extract tar.gz layer files
- [ ] Handle whiteout files for deletions
- [ ] Apply layers in correct order
- [ ] Compute and verify layer digests""",

    "image pull": """## Overview
Pulling container images involves interacting with registries using the OCI Distribution Spec to download manifests and layers.

### Implementation
```go
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "path/filepath"
)

type Manifest struct {
    SchemaVersion int    `json:"schemaVersion"`
    MediaType     string `json:"mediaType"`
    Config        Descriptor `json:"config"`
    Layers        []Descriptor `json:"layers"`
}

type Descriptor struct {
    MediaType string `json:"mediaType"`
    Digest    string `json:"digest"`
    Size      int64  `json:"size"`
}

type Registry struct {
    BaseURL string
    Client  *http.Client
}

func NewRegistry(host string) *Registry {
    return &Registry{
        BaseURL: fmt.Sprintf("https://%s/v2", host),
        Client:  &http.Client{},
    }
}

func (r *Registry) GetManifest(repo, tag string) (*Manifest, error) {
    url := fmt.Sprintf("%s/%s/manifests/%s", r.BaseURL, repo, tag)

    req, _ := http.NewRequest("GET", url, nil)
    req.Header.Set("Accept", "application/vnd.oci.image.manifest.v1+json")

    resp, err := r.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var manifest Manifest
    json.NewDecoder(resp.Body).Decode(&manifest)
    return &manifest, nil
}

func (r *Registry) DownloadBlob(repo, digest, targetPath string) error {
    url := fmt.Sprintf("%s/%s/blobs/%s", r.BaseURL, repo, digest)

    resp, err := r.Client.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    os.MkdirAll(filepath.Dir(targetPath), 0755)
    file, _ := os.Create(targetPath)
    defer file.Close()

    _, err = io.Copy(file, resp.Body)
    return err
}

func PullImage(registry *Registry, repo, tag, cacheDir string) error {
    manifest, err := registry.GetManifest(repo, tag)
    if err != nil {
        return err
    }

    // Download config
    configPath := filepath.Join(cacheDir, "config.json")
    registry.DownloadBlob(repo, manifest.Config.Digest, configPath)

    // Download layers
    for i, layer := range manifest.Layers {
        layerPath := filepath.Join(cacheDir, fmt.Sprintf("layer%d.tar.gz", i))
        fmt.Printf("Downloading layer %d/%d: %s\\n", i+1, len(manifest.Layers), layer.Digest[:12])
        registry.DownloadBlob(repo, layer.Digest, layerPath)
    }

    return nil
}
```

### Key Concepts
- OCI Distribution Spec defines registry API
- Manifest lists config and layer digests
- Blobs downloaded by content digest
- Authentication via Bearer tokens

### Completion Criteria
- [ ] Implement registry API client
- [ ] Parse OCI image manifests
- [ ] Download layers with progress reporting
- [ ] Implement local layer caching""",

    "network namespace": """## Overview
Network namespaces isolate network resources (interfaces, routing tables, firewall rules) so containers have their own network stack.

### Implementation
```go
package main

import (
    "fmt"
    "net"
    "os"
    "os/exec"
    "syscall"

    "github.com/vishvananda/netlink"
)

func createNetworkNamespace() error {
    // Create new process with network namespace
    cmd := exec.Command("/proc/self/exe", "child")
    cmd.SysProcAttr = &syscall.SysProcAttr{
        Cloneflags: syscall.CLONE_NEWNET | syscall.CLONE_NEWNS | syscall.CLONE_NEWUTS,
    }
    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    return cmd.Run()
}

func setupContainerNetwork(pid int, containerIP, bridgeIP string) error {
    // Create veth pair
    vethHost := &netlink.Veth{
        LinkAttrs: netlink.LinkAttrs{Name: fmt.Sprintf("veth%d", pid)},
        PeerName:  "eth0",
    }

    if err := netlink.LinkAdd(vethHost); err != nil {
        return fmt.Errorf("failed to create veth: %v", err)
    }

    // Get the peer (container side)
    vethContainer, _ := netlink.LinkByName("eth0")

    // Move peer to container's network namespace
    if err := netlink.LinkSetNsPid(vethContainer, pid); err != nil {
        return fmt.Errorf("failed to move veth to ns: %v", err)
    }

    // Get bridge
    bridge, _ := netlink.LinkByName("docker0")

    // Attach host side to bridge
    netlink.LinkSetMaster(vethHost, bridge)
    netlink.LinkSetUp(vethHost)

    return nil
}

func configureContainerInterface(containerIP string) error {
    // Inside container namespace
    eth0, _ := netlink.LinkByName("eth0")

    // Set IP address
    addr, _ := netlink.ParseAddr(containerIP + "/24")
    netlink.AddrAdd(eth0, addr)

    // Bring interface up
    netlink.LinkSetUp(eth0)

    // Add default route
    route := &netlink.Route{
        Scope:     netlink.SCOPE_UNIVERSE,
        LinkIndex: eth0.Attrs().Index,
        Gw:        net.ParseIP("172.17.0.1"),
    }
    netlink.RouteAdd(route)

    return nil
}
```

### Key Concepts
- CLONE_NEWNET creates isolated network stack
- Virtual ethernet (veth) pairs connect namespaces
- Bridge connects multiple container networks
- Each container gets its own interfaces and routes

### Completion Criteria
- [ ] Create process with network namespace
- [ ] Set up veth pairs for connectivity
- [ ] Configure IP addresses and routes
- [ ] Verify network isolation between containers""",

    "pid namespace": """## Overview
PID namespaces isolate process ID number spaces so containers see their own init process (PID 1) independent of the host.

### Implementation
```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "syscall"
)

func main() {
    switch os.Args[1] {
    case "run":
        run()
    case "child":
        child()
    }
}

func run() {
    fmt.Printf("Running %v as PID %d\\n", os.Args[2:], os.Getpid())

    cmd := exec.Command("/proc/self/exe", append([]string{"child"}, os.Args[2:]...)...)
    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    cmd.SysProcAttr = &syscall.SysProcAttr{
        Cloneflags: syscall.CLONE_NEWPID | syscall.CLONE_NEWNS | syscall.CLONE_NEWUTS,
        Unshareflags: syscall.CLONE_NEWNS,
    }

    if err := cmd.Run(); err != nil {
        fmt.Printf("Error: %v\\n", err)
        os.Exit(1)
    }
}

func child() {
    fmt.Printf("Running %v as PID %d\\n", os.Args[2:], os.Getpid())

    // Mount new proc filesystem for container
    syscall.Mount("proc", "/proc", "proc", 0, "")
    defer syscall.Unmount("/proc", 0)

    // Now ps will only show container processes
    cmd := exec.Command(os.Args[2], os.Args[3:]...)
    cmd.Stdin = os.Stdin
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    if err := cmd.Run(); err != nil {
        fmt.Printf("Error running %v: %v\\n", os.Args[2:], err)
    }
}

// Helper to reap zombie processes (important for PID 1)
func reapChildren() {
    var status syscall.WaitStatus
    for {
        pid, err := syscall.Wait4(-1, &status, syscall.WNOHANG, nil)
        if pid <= 0 || err != nil {
            break
        }
        fmt.Printf("Reaped child PID %d\\n", pid)
    }
}
```

### Key Concepts
- CLONE_NEWPID creates new PID namespace
- First process becomes PID 1 (init)
- PID 1 must reap zombie processes
- Mount new /proc to see only container processes

### Completion Criteria
- [ ] Create process with PID namespace isolation
- [ ] Verify container process sees itself as PID 1
- [ ] Mount container-specific /proc filesystem
- [ ] Implement zombie process reaping""",

    "mount namespace": """## Overview
Mount namespaces isolate the filesystem mount table, allowing containers to have their own view of the filesystem hierarchy.

### Implementation
```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "syscall"
)

func setupRootfs(rootfsPath string) error {
    // Ensure rootfs has required directories
    dirs := []string{"proc", "sys", "dev", "etc", "tmp"}
    for _, dir := range dirs {
        os.MkdirAll(filepath.Join(rootfsPath, dir), 0755)
    }

    // Pivot root requires the new root to be a mount point
    if err := syscall.Mount(rootfsPath, rootfsPath, "", syscall.MS_BIND|syscall.MS_REC, ""); err != nil {
        return fmt.Errorf("bind mount rootfs: %v", err)
    }

    // Create put_old directory for pivot_root
    putOld := filepath.Join(rootfsPath, ".pivot_root")
    os.MkdirAll(putOld, 0700)

    // Change to new root
    if err := syscall.PivotRoot(rootfsPath, putOld); err != nil {
        return fmt.Errorf("pivot_root: %v", err)
    }

    // Change working directory to new root
    if err := os.Chdir("/"); err != nil {
        return fmt.Errorf("chdir: %v", err)
    }

    // Unmount old root
    putOld = "/.pivot_root"
    if err := syscall.Unmount(putOld, syscall.MNT_DETACH); err != nil {
        return fmt.Errorf("unmount old root: %v", err)
    }
    os.RemoveAll(putOld)

    // Mount essential filesystems
    syscall.Mount("proc", "/proc", "proc", 0, "")
    syscall.Mount("sysfs", "/sys", "sysfs", 0, "")
    syscall.Mount("tmpfs", "/tmp", "tmpfs", 0, "")

    // Mount /dev with minimal devices
    syscall.Mount("tmpfs", "/dev", "tmpfs", syscall.MS_NOSUID|syscall.MS_STRICTATIME, "mode=755")

    // Create essential device nodes
    devices := []struct {
        path  string
        major uint32
        minor uint32
    }{
        {"/dev/null", 1, 3},
        {"/dev/zero", 1, 5},
        {"/dev/random", 1, 8},
        {"/dev/urandom", 1, 9},
    }

    for _, dev := range devices {
        syscall.Mknod(dev.path, syscall.S_IFCHR|0666, int(dev.major<<8|dev.minor))
    }

    return nil
}
```

### Key Concepts
- CLONE_NEWNS creates isolated mount table
- pivot_root switches to new root filesystem
- Mount proc/sys/dev for container functionality
- Create minimal /dev with required devices

### Completion Criteria
- [ ] Create process with mount namespace
- [ ] Set up container rootfs with pivot_root
- [ ] Mount proc, sys, and dev filesystems
- [ ] Create essential device nodes""",

    "virtual ethernet": """## Overview
Virtual ethernet (veth) pairs act as a virtual network cable connecting two network namespaces, enabling container networking.

### Implementation
```go
package main

import (
    "fmt"
    "net"
    "os/exec"

    "github.com/vishvananda/netlink"
)

func createVethPair(hostName, containerName string) error {
    veth := &netlink.Veth{
        LinkAttrs: netlink.LinkAttrs{
            Name: hostName,
            MTU:  1500,
        },
        PeerName: containerName,
    }

    if err := netlink.LinkAdd(veth); err != nil {
        return fmt.Errorf("failed to create veth pair: %v", err)
    }

    return nil
}

func moveToNamespace(ifaceName string, pid int) error {
    link, err := netlink.LinkByName(ifaceName)
    if err != nil {
        return err
    }

    return netlink.LinkSetNsPid(link, pid)
}

func configureInterface(ifaceName, ipAddr, gateway string) error {
    link, err := netlink.LinkByName(ifaceName)
    if err != nil {
        return err
    }

    // Parse and add IP address
    addr, err := netlink.ParseAddr(ipAddr)
    if err != nil {
        return err
    }

    if err := netlink.AddrAdd(link, addr); err != nil {
        return err
    }

    // Bring interface up
    if err := netlink.LinkSetUp(link); err != nil {
        return err
    }

    // Add default route if gateway specified
    if gateway != "" {
        route := &netlink.Route{
            LinkIndex: link.Attrs().Index,
            Gw:        net.ParseIP(gateway),
        }
        if err := netlink.RouteAdd(route); err != nil {
            return err
        }
    }

    return nil
}

func setupContainerNetworking(containerPID int, containerIP string) error {
    hostVeth := fmt.Sprintf("veth%d", containerPID)
    containerVeth := "eth0"

    // Create veth pair
    if err := createVethPair(hostVeth, containerVeth); err != nil {
        return err
    }

    // Move container end to container namespace
    if err := moveToNamespace(containerVeth, containerPID); err != nil {
        return err
    }

    // Configure host end - attach to bridge
    bridge, _ := netlink.LinkByName("docker0")
    hostLink, _ := netlink.LinkByName(hostVeth)
    netlink.LinkSetMaster(hostLink, bridge)
    netlink.LinkSetUp(hostLink)

    // Configure container end (must run in container namespace)
    // This would be done by the container process

    return nil
}
```

### Key Concepts
- Veth pairs connect two network namespaces
- One end stays on host, other moves to container
- Host end typically attached to bridge
- Container end configured with IP and routes

### Completion Criteria
- [ ] Create veth pairs programmatically
- [ ] Move veth end to container namespace
- [ ] Configure IP addresses on both ends
- [ ] Connect host end to bridge network""",

    "cgroups": """## Overview
Control groups (cgroups) v2 limit and account for resource usage (CPU, memory, I/O) of process groups.

### Implementation
```go
package main

import (
    "fmt"
    "os"
    "path/filepath"
    "strconv"
)

const cgroupRoot = "/sys/fs/cgroup"

type CgroupLimits struct {
    MemoryMax    int64  // bytes
    CPUMax       string // "quota period" e.g., "50000 100000" for 50%
    PidsMax      int64
    IOMax        string // "major:minor rbps=X wbps=Y"
}

func createCgroup(name string) (string, error) {
    path := filepath.Join(cgroupRoot, name)

    if err := os.MkdirAll(path, 0755); err != nil {
        return "", fmt.Errorf("failed to create cgroup: %v", err)
    }

    // Enable controllers
    controllers := "+cpu +memory +io +pids"
    subtreeControl := filepath.Join(cgroupRoot, "cgroup.subtree_control")
    os.WriteFile(subtreeControl, []byte(controllers), 0644)

    return path, nil
}

func setCgroupLimits(cgroupPath string, limits CgroupLimits) error {
    if limits.MemoryMax > 0 {
        memFile := filepath.Join(cgroupPath, "memory.max")
        if err := os.WriteFile(memFile, []byte(strconv.FormatInt(limits.MemoryMax, 10)), 0644); err != nil {
            return fmt.Errorf("set memory limit: %v", err)
        }
    }

    if limits.CPUMax != "" {
        cpuFile := filepath.Join(cgroupPath, "cpu.max")
        if err := os.WriteFile(cpuFile, []byte(limits.CPUMax), 0644); err != nil {
            return fmt.Errorf("set cpu limit: %v", err)
        }
    }

    if limits.PidsMax > 0 {
        pidsFile := filepath.Join(cgroupPath, "pids.max")
        if err := os.WriteFile(pidsFile, []byte(strconv.FormatInt(limits.PidsMax, 10)), 0644); err != nil {
            return fmt.Errorf("set pids limit: %v", err)
        }
    }

    return nil
}

func addProcessToCgroup(cgroupPath string, pid int) error {
    procsFile := filepath.Join(cgroupPath, "cgroup.procs")
    return os.WriteFile(procsFile, []byte(strconv.Itoa(pid)), 0644)
}

func getCgroupStats(cgroupPath string) map[string]string {
    stats := make(map[string]string)

    files := []string{"memory.current", "memory.peak", "cpu.stat", "pids.current"}
    for _, file := range files {
        data, err := os.ReadFile(filepath.Join(cgroupPath, file))
        if err == nil {
            stats[file] = string(data)
        }
    }

    return stats
}

func main() {
    cgPath, _ := createCgroup("mycontainer")

    limits := CgroupLimits{
        MemoryMax: 100 * 1024 * 1024, // 100MB
        CPUMax:    "50000 100000",    // 50% of one CPU
        PidsMax:   100,
    }

    setCgroupLimits(cgPath, limits)
    addProcessToCgroup(cgPath, os.Getpid())

    fmt.Println("Cgroup configured:", cgPath)
}
```

### Key Concepts
- Cgroups v2 uses unified hierarchy
- Controllers: cpu, memory, io, pids
- Limits set via filesystem interface
- Process added by writing PID to cgroup.procs

### Completion Criteria
- [ ] Create cgroup with enabled controllers
- [ ] Set memory, CPU, and PID limits
- [ ] Add processes to cgroup
- [ ] Read and display resource statistics""",

    # ============== SHELL ==============
    "path searching": """## Overview
PATH searching resolves command names to executable paths by searching directories in the PATH environment variable.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAX_PATH 4096

// Check if file exists and is executable
int is_executable(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (!S_ISREG(st.st_mode)) return 0;
    return (st.st_mode & S_IXUSR) || (st.st_mode & S_IXGRP) || (st.st_mode & S_IXOTH);
}

// Search PATH for command, return full path or NULL
char *find_in_path(const char *cmd) {
    // If cmd contains /, don't search PATH
    if (strchr(cmd, '/')) {
        if (is_executable(cmd)) {
            return strdup(cmd);
        }
        return NULL;
    }

    char *path_env = getenv("PATH");
    if (!path_env) return NULL;

    char *path_copy = strdup(path_env);
    char *dir = strtok(path_copy, ":");
    static char full_path[MAX_PATH];

    while (dir) {
        snprintf(full_path, sizeof(full_path), "%s/%s", dir, cmd);

        if (is_executable(full_path)) {
            free(path_copy);
            return strdup(full_path);
        }

        dir = strtok(NULL, ":");
    }

    free(path_copy);
    return NULL;
}

// Cache for frequently used commands
typedef struct {
    char *cmd;
    char *path;
} PathCacheEntry;

#define CACHE_SIZE 64
PathCacheEntry path_cache[CACHE_SIZE];
int cache_count = 0;

char *cached_find_in_path(const char *cmd) {
    // Check cache first
    for (int i = 0; i < cache_count; i++) {
        if (strcmp(path_cache[i].cmd, cmd) == 0) {
            return path_cache[i].path;
        }
    }

    // Not in cache, search PATH
    char *path = find_in_path(cmd);
    if (path && cache_count < CACHE_SIZE) {
        path_cache[cache_count].cmd = strdup(cmd);
        path_cache[cache_count].path = path;
        cache_count++;
    }

    return path;
}

int main() {
    char *cmds[] = {"ls", "cat", "nonexistent", "/bin/echo"};

    for (int i = 0; i < 4; i++) {
        char *path = find_in_path(cmds[i]);
        printf("%s -> %s\\n", cmds[i], path ? path : "not found");
        free(path);
    }

    return 0;
}
```

### Key Concepts
- PATH is colon-separated list of directories
- Commands with / bypass PATH search
- Check executable permission, not just existence
- Cache results for performance

### Completion Criteria
- [ ] Parse PATH environment variable
- [ ] Search directories in order
- [ ] Check file is executable
- [ ] Handle absolute/relative paths correctly""",

    "background execution": """## Overview
Background execution allows running processes without blocking the shell, managed through job control.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <string.h>

#define MAX_JOBS 64

typedef struct {
    pid_t pid;
    int job_id;
    char *command;
    int running;  // 1 = running, 0 = stopped
} Job;

Job jobs[MAX_JOBS];
int next_job_id = 1;

void sigchld_handler(int sig) {
    int status;
    pid_t pid;

    // Reap all terminated children
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        for (int i = 0; i < MAX_JOBS; i++) {
            if (jobs[i].pid == pid) {
                printf("\\n[%d]  Done    %s\\n", jobs[i].job_id, jobs[i].command);
                free(jobs[i].command);
                jobs[i].pid = 0;
                break;
            }
        }
    }
}

int add_job(pid_t pid, const char *command) {
    for (int i = 0; i < MAX_JOBS; i++) {
        if (jobs[i].pid == 0) {
            jobs[i].pid = pid;
            jobs[i].job_id = next_job_id++;
            jobs[i].command = strdup(command);
            jobs[i].running = 1;
            return jobs[i].job_id;
        }
    }
    return -1;
}

void execute_command(char **args, int background) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        if (background) {
            // Create new process group
            setpgid(0, 0);
        }
        execvp(args[0], args);
        perror("exec failed");
        exit(1);
    } else if (pid > 0) {
        if (background) {
            int job_id = add_job(pid, args[0]);
            printf("[%d] %d\\n", job_id, pid);
        } else {
            // Foreground: wait for completion
            int status;
            waitpid(pid, &status, 0);
        }
    }
}

void list_jobs() {
    for (int i = 0; i < MAX_JOBS; i++) {
        if (jobs[i].pid != 0) {
            printf("[%d]  %s    %s\\n",
                   jobs[i].job_id,
                   jobs[i].running ? "Running" : "Stopped",
                   jobs[i].command);
        }
    }
}

int main() {
    // Set up SIGCHLD handler
    struct sigaction sa;
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Example: run "sleep 5 &"
    char *args[] = {"sleep", "5", NULL};
    execute_command(args, 1);  // background = 1

    printf("Shell continues...\\n");
    list_jobs();

    // Wait for demo
    sleep(6);
    return 0;
}
```

### Key Concepts
- Fork process and don't wait for background jobs
- SIGCHLD handler reaps terminated processes
- Job table tracks background processes
- Process groups for job control

### Completion Criteria
- [ ] Detect & in command line
- [ ] Fork without waiting for background jobs
- [ ] Track jobs in job table
- [ ] Handle SIGCHLD to report completion""",

    "command history": """## Overview
Command history stores and recalls previous commands, supporting navigation and search.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <readline/readline.h>
#include <readline/history.h>

#define HISTORY_FILE ".shell_history"
#define MAX_HISTORY 1000

void init_history() {
    char *home = getenv("HOME");
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", home, HISTORY_FILE);

    // Load existing history
    read_history(path);

    // Set history limits
    stifle_history(MAX_HISTORY);
}

void save_history() {
    char *home = getenv("HOME");
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", home, HISTORY_FILE);

    write_history(path);
}

void add_to_history(const char *line) {
    // Don't add empty lines or duplicates
    if (!line || !*line) return;

    HIST_ENTRY *last = history_get(history_length);
    if (last && strcmp(last->line, line) == 0) return;

    add_history(line);
}

// Custom history expansion (e.g., !!, !n, !string)
char *expand_history(const char *line) {
    char *expanded = NULL;
    int result = history_expand((char *)line, &expanded);

    if (result == 0 || result == 1) {
        // No expansion or successful expansion
        return expanded ? expanded : strdup(line);
    } else {
        // Error
        fprintf(stderr, "History expansion error\\n");
        free(expanded);
        return NULL;
    }
}

// Search history with Ctrl+R style interface
void reverse_search() {
    printf("(reverse-i-search): ");
    char search[256];
    fgets(search, sizeof(search), stdin);
    search[strcspn(search, "\\n")] = 0;

    // Search backwards through history
    HIST_ENTRY **hist = history_list();
    if (!hist) return;

    for (int i = history_length - 1; i >= 0; i--) {
        if (strstr(hist[i]->line, search)) {
            printf("Found: %s\\n", hist[i]->line);
            return;
        }
    }
    printf("Not found\\n");
}

int main() {
    init_history();

    char *line;
    while ((line = readline("$ ")) != NULL) {
        if (*line) {
            // Expand history references
            char *expanded = expand_history(line);
            if (expanded) {
                add_to_history(expanded);
                printf("Execute: %s\\n", expanded);
                free(expanded);
            }
        }
        free(line);
    }

    save_history();
    return 0;
}
```

### Key Concepts
- readline library provides history support
- History persisted to file across sessions
- Expansion: !!, !n, !string, ^old^new
- Reverse search with Ctrl+R

### Completion Criteria
- [ ] Store commands in history list
- [ ] Persist history to file
- [ ] Implement history expansion (!!, !n)
- [ ] Add reverse search functionality""",

    "here documents": """## Overview
Here documents (heredocs) allow multi-line input redirection, useful for embedding large text blocks in scripts.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINE 4096

typedef struct {
    char *content;
    size_t size;
    size_t capacity;
} StringBuffer;

void buffer_init(StringBuffer *buf) {
    buf->capacity = 1024;
    buf->content = malloc(buf->capacity);
    buf->content[0] = '\\0';
    buf->size = 0;
}

void buffer_append(StringBuffer *buf, const char *str) {
    size_t len = strlen(str);
    while (buf->size + len + 1 > buf->capacity) {
        buf->capacity *= 2;
        buf->content = realloc(buf->content, buf->capacity);
    }
    strcpy(buf->content + buf->size, str);
    buf->size += len;
}

// Parse heredoc: cmd <<DELIMITER or cmd <<-DELIMITER (strip tabs)
int parse_heredoc(const char *line, char *delimiter, int *strip_tabs) {
    char *pos = strstr(line, "<<");
    if (!pos) return 0;

    pos += 2;
    *strip_tabs = (*pos == '-');
    if (*strip_tabs) pos++;

    // Skip whitespace
    while (*pos == ' ' || *pos == '\\t') pos++;

    // Extract delimiter (may be quoted)
    int quoted = 0;
    if (*pos == '\\'' || *pos == '"') {
        quoted = 1;
        pos++;
    }

    char *end = pos;
    while (*end && *end != ' ' && *end != '\\n' && *end != '\\'' && *end != '"') end++;

    strncpy(delimiter, pos, end - pos);
    delimiter[end - pos] = '\\0';

    return 1;
}

char *read_heredoc(const char *delimiter, int strip_tabs) {
    StringBuffer buf;
    buffer_init(&buf);

    char line[MAX_LINE];
    printf("> ");  // Secondary prompt

    while (fgets(line, sizeof(line), stdin)) {
        // Remove trailing newline for comparison
        char *nl = strchr(line, '\\n');
        if (nl) *nl = '\\0';

        // Check for delimiter (possibly with leading tabs stripped)
        char *check = line;
        if (strip_tabs) {
            while (*check == '\\t') check++;
        }

        if (strcmp(check, delimiter) == 0) {
            break;
        }

        // Restore newline and append
        if (nl) *nl = '\\n';

        // Strip leading tabs if <<- was used
        if (strip_tabs) {
            char *content = line;
            while (*content == '\\t') content++;
            buffer_append(&buf, content);
        } else {
            buffer_append(&buf, line);
        }

        printf("> ");
    }

    return buf.content;
}

int setup_heredoc_pipe(const char *content) {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid = fork();
    if (pid == 0) {
        // Child: write content to pipe
        close(pipefd[0]);
        write(pipefd[1], content, strlen(content));
        close(pipefd[1]);
        exit(0);
    }

    // Parent: return read end
    close(pipefd[1]);
    return pipefd[0];
}

int main() {
    char delimiter[256];
    int strip_tabs;

    char *test = "cat <<EOF";
    if (parse_heredoc(test, delimiter, &strip_tabs)) {
        printf("Delimiter: '%s', strip_tabs: %d\\n", delimiter, strip_tabs);

        char *content = read_heredoc(delimiter, strip_tabs);
        printf("Heredoc content:\\n%s", content);
        free(content);
    }

    return 0;
}
```

### Key Concepts
- <<DELIM reads until DELIM on its own line
- <<-DELIM strips leading tabs (for indented scripts)
- Content passed via pipe to command's stdin
- Delimiter can be quoted to prevent expansion

### Completion Criteria
- [ ] Parse << and <<- heredoc syntax
- [ ] Read content until delimiter
- [ ] Implement tab stripping for <<-
- [ ] Pipe content to command stdin""",

    "tab completion": """## Overview
Tab completion helps users by suggesting or completing command names, file paths, and arguments.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <readline/readline.h>
#include <readline/history.h>

// Custom completion generator for commands
char *command_generator(const char *text, int state) {
    static int list_index, len;
    static DIR *dir;
    static char *path_copy, *current_dir;
    struct dirent *entry;

    if (!state) {
        // First call: initialize
        list_index = 0;
        len = strlen(text);
        path_copy = strdup(getenv("PATH"));
        current_dir = strtok(path_copy, ":");
        dir = current_dir ? opendir(current_dir) : NULL;
    }

    while (dir || current_dir) {
        if (dir) {
            while ((entry = readdir(dir)) != NULL) {
                if (strncmp(entry->d_name, text, len) == 0) {
                    return strdup(entry->d_name);
                }
            }
            closedir(dir);
            dir = NULL;
        }

        current_dir = strtok(NULL, ":");
        if (current_dir) {
            dir = opendir(current_dir);
        }
    }

    free(path_copy);
    return NULL;
}

// Custom completion generator for filenames
char *filename_generator(const char *text, int state) {
    static DIR *dir;
    static int len;
    static char dirname[512], prefix[256];
    struct dirent *entry;

    if (!state) {
        // Parse directory and prefix from text
        char *slash = strrchr(text, '/');
        if (slash) {
            strncpy(dirname, text, slash - text + 1);
            dirname[slash - text + 1] = '\\0';
            strcpy(prefix, slash + 1);
        } else {
            strcpy(dirname, ".");
            strcpy(prefix, text);
        }
        len = strlen(prefix);
        dir = opendir(dirname);
    }

    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.' && prefix[0] != '.') continue;

            if (strncmp(entry->d_name, prefix, len) == 0) {
                char *match = malloc(strlen(dirname) + strlen(entry->d_name) + 2);
                sprintf(match, "%s%s%s",
                        strcmp(dirname, ".") ? dirname : "",
                        strcmp(dirname, ".") ? "" : "",
                        entry->d_name);
                return match;
            }
        }
        closedir(dir);
        dir = NULL;
    }

    return NULL;
}

// Main completion function
char **shell_completion(const char *text, int start, int end) {
    rl_attempted_completion_over = 1;  // Don't fall back to default

    // First word: command completion
    if (start == 0) {
        return rl_completion_matches(text, command_generator);
    }

    // Otherwise: filename completion
    return rl_completion_matches(text, filename_generator);
}

int main() {
    // Set up custom completion
    rl_attempted_completion_function = shell_completion;

    char *line;
    while ((line = readline("$ ")) != NULL) {
        if (*line) {
            printf("You typed: %s\\n", line);
        }
        free(line);
    }

    return 0;
}
```

### Key Concepts
- readline provides completion infrastructure
- Generator function called repeatedly
- Context-aware: commands vs files vs options
- Return matches one at a time

### Completion Criteria
- [ ] Implement command name completion
- [ ] Implement file path completion
- [ ] Handle partial paths correctly
- [ ] Show multiple matches when ambiguous""",

    "signals": """## Overview
Signal handling allows the shell to respond to keyboard interrupts, job control signals, and child process events.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <termios.h>

static pid_t foreground_pid = 0;
static struct termios shell_tmodes;

void sigint_handler(int sig) {
    // Ctrl+C: interrupt foreground process
    if (foreground_pid > 0) {
        kill(-foreground_pid, SIGINT);  // Send to process group
    }
    printf("\\n");
}

void sigtstp_handler(int sig) {
    // Ctrl+Z: suspend foreground process
    if (foreground_pid > 0) {
        kill(-foreground_pid, SIGTSTP);
        printf("\\n[Stopped]\\n");
    }
}

void sigchld_handler(int sig) {
    int status;
    pid_t pid;

    while ((pid = waitpid(-1, &status, WNOHANG | WUNTRACED)) > 0) {
        if (WIFEXITED(status)) {
            printf("Process %d exited with status %d\\n", pid, WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("Process %d killed by signal %d\\n", pid, WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
            printf("Process %d stopped\\n", pid);
        }
    }
}

void setup_signal_handlers() {
    struct sigaction sa;

    // SIGINT (Ctrl+C)
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL);

    // SIGTSTP (Ctrl+Z)
    sa.sa_handler = sigtstp_handler;
    sigaction(SIGTSTP, &sa, NULL);

    // SIGCHLD (child state change)
    sa.sa_handler = sigchld_handler;
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    sigaction(SIGCHLD, &sa, NULL);

    // Ignore these in the shell itself
    signal(SIGTTOU, SIG_IGN);
    signal(SIGTTIN, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
}

void run_foreground(char **args) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child: reset signal handlers
        signal(SIGINT, SIG_DFL);
        signal(SIGTSTP, SIG_DFL);
        signal(SIGTTOU, SIG_DFL);
        signal(SIGTTIN, SIG_DFL);

        // Create new process group
        setpgid(0, 0);

        // Take control of terminal
        tcsetpgrp(STDIN_FILENO, getpid());

        execvp(args[0], args);
        perror("exec");
        exit(1);
    } else {
        // Parent: track foreground process
        foreground_pid = pid;
        setpgid(pid, pid);
        tcsetpgrp(STDIN_FILENO, pid);

        // Wait for foreground process
        int status;
        waitpid(pid, &status, WUNTRACED);

        // Shell takes back terminal control
        tcsetpgrp(STDIN_FILENO, getpid());
        foreground_pid = 0;
    }
}

int main() {
    // Save shell's terminal modes
    tcgetattr(STDIN_FILENO, &shell_tmodes);

    setup_signal_handlers();

    // Put shell in its own process group
    setpgid(0, 0);
    tcsetpgrp(STDIN_FILENO, getpid());

    printf("Shell ready. Try Ctrl+C, Ctrl+Z with foreground processes.\\n");

    // Demo: run a command
    char *args[] = {"sleep", "100", NULL};
    run_foreground(args);

    return 0;
}
```

### Key Concepts
- Shell ignores SIGINT/SIGTSTP for itself
- Forward signals to foreground process group
- Reset handlers to default in child processes
- Terminal control transferred with tcsetpgrp

### Completion Criteria
- [ ] Handle SIGINT (Ctrl+C) properly
- [ ] Handle SIGTSTP (Ctrl+Z) for job suspension
- [ ] Handle SIGCHLD for background job notification
- [ ] Manage terminal control between shell and jobs""",

    # ============== RED TEAM TOOLING ==============
    "memory allocation": """## Overview
Understanding memory allocation in C is fundamental for security tools that need to manipulate memory, analyze heap layouts, or develop exploits.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Simple custom allocator for learning
#define HEAP_SIZE 65536
static char heap[HEAP_SIZE];
static size_t heap_offset = 0;

typedef struct block_header {
    size_t size;
    int free;
    struct block_header *next;
} block_header_t;

static block_header_t *free_list = NULL;

void *my_malloc(size_t size) {
    // Align to 8 bytes
    size = (size + 7) & ~7;

    // First-fit search
    block_header_t *curr = free_list;
    block_header_t *prev = NULL;

    while (curr) {
        if (curr->free && curr->size >= size) {
            curr->free = 0;
            return (void *)(curr + 1);
        }
        prev = curr;
        curr = curr->next;
    }

    // Allocate from heap
    if (heap_offset + sizeof(block_header_t) + size > HEAP_SIZE) {
        return NULL;  // Out of memory
    }

    block_header_t *new_block = (block_header_t *)(heap + heap_offset);
    new_block->size = size;
    new_block->free = 0;
    new_block->next = NULL;

    heap_offset += sizeof(block_header_t) + size;

    // Add to free list
    if (prev) prev->next = new_block;
    else free_list = new_block;

    return (void *)(new_block + 1);
}

void my_free(void *ptr) {
    if (!ptr) return;

    block_header_t *header = (block_header_t *)ptr - 1;
    header->free = 1;

    // Coalesce adjacent free blocks
    block_header_t *curr = free_list;
    while (curr && curr->next) {
        if (curr->free && curr->next->free) {
            curr->size += sizeof(block_header_t) + curr->next->size;
            curr->next = curr->next->next;
        } else {
            curr = curr->next;
        }
    }
}

// Heap analysis for security research
void dump_heap() {
    printf("Heap dump:\\n");
    block_header_t *curr = free_list;
    int block_num = 0;

    while (curr) {
        printf("Block %d: addr=%p, size=%zu, free=%d\\n",
               block_num++, (void *)curr, curr->size, curr->free);
        curr = curr->next;
    }
}

int main() {
    void *a = my_malloc(32);
    void *b = my_malloc(64);
    void *c = my_malloc(32);

    printf("Allocated: a=%p, b=%p, c=%p\\n", a, b, c);
    dump_heap();

    my_free(b);
    printf("\\nAfter freeing b:\\n");
    dump_heap();

    void *d = my_malloc(48);
    printf("\\nAfter allocating d (48 bytes):\\n");
    printf("d=%p (reused b's space)\\n", d);
    dump_heap();

    return 0;
}
```

### Key Concepts
- Heap managed through block headers
- Free list tracks available memory
- Coalescing prevents fragmentation
- Understanding allocation patterns aids exploitation

### Completion Criteria
- [ ] Implement basic malloc/free
- [ ] Add block coalescing
- [ ] Create heap visualization
- [ ] Study glibc malloc internals""",

    "pointer arithmetic": """## Overview
Pointer arithmetic is essential for low-level memory manipulation, buffer handling, and understanding how exploits work.

### Implementation
```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Basic pointer arithmetic
void demonstrate_pointer_math() {
    int arr[] = {10, 20, 30, 40, 50};
    int *ptr = arr;

    printf("Array base: %p\\n", (void *)ptr);
    printf("ptr[0] = %d, ptr[2] = %d\\n", ptr[0], ptr[2]);
    printf("*(ptr + 2) = %d\\n", *(ptr + 2));

    // Pointer subtraction gives element count
    int *end = &arr[4];
    printf("Elements between: %ld\\n", end - ptr);

    // Increment moves by sizeof(type)
    printf("ptr = %p, ptr+1 = %p (diff: %ld bytes)\\n",
           (void *)ptr, (void *)(ptr+1), (char *)(ptr+1) - (char *)ptr);
}

// Casting and byte-level access
void byte_level_access() {
    uint32_t value = 0xDEADBEEF;
    uint8_t *bytes = (uint8_t *)&value;

    printf("\\nValue: 0x%08X\\n", value);
    printf("Bytes (little-endian): ");
    for (int i = 0; i < 4; i++) {
        printf("%02X ", bytes[i]);
    }
    printf("\\n");

    // Modify individual bytes
    bytes[0] = 0x41;
    printf("After modifying first byte: 0x%08X\\n", value);
}

// Function pointer manipulation
typedef int (*operation)(int, int);

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

void function_pointers() {
    operation ops[] = {add, sub};

    printf("\\nadd(5,3) via pointer: %d\\n", ops[0](5, 3));
    printf("sub(5,3) via pointer: %d\\n", ops[1](5, 3));

    // Print function addresses
    printf("add address: %p\\n", (void *)add);
    printf("sub address: %p\\n", (void *)sub);
}

// Buffer manipulation (common in exploits)
void buffer_manipulation() {
    char buffer[64];
    memset(buffer, 0, sizeof(buffer));

    // Write at specific offsets
    *(uint32_t *)(buffer + 0) = 0x41414141;   // "AAAA"
    *(uint32_t *)(buffer + 4) = 0x42424242;   // "BBBB"
    *(uint64_t *)(buffer + 8) = 0x0000000012345678;

    printf("\\nBuffer contents:\\n");
    for (int i = 0; i < 24; i++) {
        printf("%02X ", (unsigned char)buffer[i]);
        if ((i + 1) % 8 == 0) printf("\\n");
    }
}

// Void pointer arithmetic (cast first!)
void void_pointer_demo() {
    char data[] = "Hello, World!";
    void *vptr = data;

    // Must cast before arithmetic
    char *cptr = (char *)vptr + 7;
    printf("\\nOffset 7: %s\\n", cptr);
}

int main() {
    demonstrate_pointer_math();
    byte_level_access();
    function_pointers();
    buffer_manipulation();
    void_pointer_demo();
    return 0;
}
```

### Key Concepts
- Pointer arithmetic respects type size
- Cast to char* for byte-level access
- Function pointers enable dynamic calls
- Void pointers require casting for arithmetic

### Completion Criteria
- [ ] Master pointer increment/decrement
- [ ] Implement byte-level memory access
- [ ] Work with function pointers
- [ ] Build buffer manipulation utilities""",

    "structs and unions": """## Overview
Structs and unions are essential for parsing binary protocols, file formats, and network packets in security tools.

### Implementation
```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h>

// Packed struct for network parsing
#pragma pack(push, 1)
typedef struct {
    uint8_t  version_ihl;
    uint8_t  tos;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t checksum;
    uint32_t src_addr;
    uint32_t dst_addr;
} ip_header_t;

typedef struct {
    uint16_t src_port;
    uint16_t dst_port;
    uint32_t seq_num;
    uint32_t ack_num;
    uint8_t  data_offset;
    uint8_t  flags;
    uint16_t window;
    uint16_t checksum;
    uint16_t urgent_ptr;
} tcp_header_t;
#pragma pack(pop)

// Union for type punning
typedef union {
    uint32_t addr;
    uint8_t octets[4];
} ip_addr_t;

void parse_ip_header(const uint8_t *packet) {
    ip_header_t *ip = (ip_header_t *)packet;

    printf("IP Header:\\n");
    printf("  Version: %d\\n", (ip->version_ihl >> 4) & 0xF);
    printf("  IHL: %d words\\n", ip->version_ihl & 0xF);
    printf("  Total Length: %d\\n", ntohs(ip->total_length));
    printf("  TTL: %d\\n", ip->ttl);
    printf("  Protocol: %d\\n", ip->protocol);

    ip_addr_t src = { .addr = ip->src_addr };
    ip_addr_t dst = { .addr = ip->dst_addr };

    printf("  Source: %d.%d.%d.%d\\n",
           src.octets[0], src.octets[1], src.octets[2], src.octets[3]);
    printf("  Dest: %d.%d.%d.%d\\n",
           dst.octets[0], dst.octets[1], dst.octets[2], dst.octets[3]);
}

// Flexible array member for variable-length data
typedef struct {
    uint32_t type;
    uint32_t length;
    uint8_t data[];  // Flexible array member
} tlv_t;

tlv_t *create_tlv(uint32_t type, const void *data, uint32_t len) {
    tlv_t *tlv = malloc(sizeof(tlv_t) + len);
    tlv->type = type;
    tlv->length = len;
    memcpy(tlv->data, data, len);
    return tlv;
}

// Bit fields for flags
typedef struct {
    uint8_t fin : 1;
    uint8_t syn : 1;
    uint8_t rst : 1;
    uint8_t psh : 1;
    uint8_t ack : 1;
    uint8_t urg : 1;
    uint8_t ece : 1;
    uint8_t cwr : 1;
} tcp_flags_t;

void print_tcp_flags(uint8_t flags) {
    tcp_flags_t *f = (tcp_flags_t *)&flags;
    printf("TCP Flags: ");
    if (f->syn) printf("SYN ");
    if (f->ack) printf("ACK ");
    if (f->fin) printf("FIN ");
    if (f->rst) printf("RST ");
    if (f->psh) printf("PSH ");
    printf("\\n");
}

int main() {
    // Sample IP packet (simplified)
    uint8_t packet[] = {
        0x45, 0x00, 0x00, 0x28,  // Version, IHL, TOS, Total Length
        0x00, 0x00, 0x40, 0x00,  // ID, Flags, Fragment
        0x40, 0x06, 0x00, 0x00,  // TTL, Protocol (TCP), Checksum
        0xC0, 0xA8, 0x01, 0x01,  // Source: 192.168.1.1
        0xC0, 0xA8, 0x01, 0x02   // Dest: 192.168.1.2
    };

    parse_ip_header(packet);

    print_tcp_flags(0x12);  // SYN+ACK

    return 0;
}
```

### Key Concepts
- #pragma pack ensures no padding
- Unions allow type punning
- Bit fields for flag parsing
- Flexible array members for variable data

### Completion Criteria
- [ ] Parse network packet headers
- [ ] Use unions for IP address handling
- [ ] Implement bit field flag parsing
- [ ] Build TLV (Type-Length-Value) parser""",

    "file i/o": """## Overview
Low-level file I/O is essential for security tools that read binary files, parse executables, or interact with special files.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

// Read entire file into buffer
uint8_t *read_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    fstat(fd, &st);
    *size = st.st_size;

    uint8_t *buffer = malloc(*size);
    read(fd, buffer, *size);
    close(fd);

    return buffer;
}

// Memory-mapped file access (efficient for large files)
void *mmap_file(const char *path, size_t *size, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) return NULL;

    struct stat st;
    fstat(*fd, &st);
    *size = st.st_size;

    void *mapped = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (mapped == MAP_FAILED) {
        close(*fd);
        return NULL;
    }

    return mapped;
}

// Write binary data
int write_binary(const char *path, const void *data, size_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;

    ssize_t written = write(fd, data, size);
    close(fd);

    return written == size ? 0 : -1;
}

// Hex dump utility
void hexdump(const uint8_t *data, size_t size, size_t offset) {
    for (size_t i = 0; i < size; i += 16) {
        printf("%08zx  ", offset + i);

        // Hex bytes
        for (int j = 0; j < 16; j++) {
            if (i + j < size) printf("%02x ", data[i + j]);
            else printf("   ");
            if (j == 7) printf(" ");
        }

        // ASCII
        printf(" |");
        for (int j = 0; j < 16 && i + j < size; j++) {
            uint8_t c = data[i + j];
            printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        printf("|\\n");
    }
}

// Parse ELF header example
typedef struct {
    uint8_t  e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
} elf_header_partial_t;

void parse_elf(const char *path) {
    size_t size;
    uint8_t *data = read_file(path, &size);
    if (!data) {
        printf("Failed to read file\\n");
        return;
    }

    if (size < sizeof(elf_header_partial_t) ||
        memcmp(data, "\\x7fELF", 4) != 0) {
        printf("Not an ELF file\\n");
        free(data);
        return;
    }

    elf_header_partial_t *elf = (elf_header_partial_t *)data;

    printf("ELF Header:\\n");
    printf("  Class: %d-bit\\n", elf->e_ident[4] == 2 ? 64 : 32);
    printf("  Endian: %s\\n", elf->e_ident[5] == 1 ? "Little" : "Big");
    printf("  Type: %d\\n", elf->e_type);
    printf("  Machine: %d\\n", elf->e_machine);

    printf("\\nFirst 64 bytes:\\n");
    hexdump(data, 64, 0);

    free(data);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <file>\\n", argv[0]);
        return 1;
    }

    parse_elf(argv[1]);
    return 0;
}
```

### Key Concepts
- Low-level I/O: open, read, write, close
- Memory mapping for efficient large file access
- Binary parsing with struct overlays
- Hex dump for debugging binary data

### Completion Criteria
- [ ] Implement file read/write helpers
- [ ] Use memory mapping for large files
- [ ] Build hexdump utility
- [ ] Parse binary file headers (ELF, PE)""",

    "socket programming": """## Overview
Socket programming enables network communication for security tools like scanners, proxies, and custom protocol implementations.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>

// TCP client connection
int tcp_connect(const char *host, int port) {
    struct sockaddr_in addr;
    struct hostent *he;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    he = gethostbyname(host);
    if (!he) {
        close(sock);
        return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    memcpy(&addr.sin_addr, he->h_addr, he->h_length);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    return sock;
}

// TCP server
int tcp_server(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr.s_addr = INADDR_ANY
    };

    bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    listen(sock, 10);

    return sock;
}

// Non-blocking connect with timeout
int tcp_connect_timeout(const char *host, int port, int timeout_sec) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // Set non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    struct hostent *he = gethostbyname(host);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    memcpy(&addr.sin_addr, he->h_addr, he->h_length);

    connect(sock, (struct sockaddr *)&addr, sizeof(addr));

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(sock, &fdset);

    struct timeval tv = { .tv_sec = timeout_sec, .tv_usec = 0 };

    if (select(sock + 1, NULL, &fdset, NULL, &tv) == 1) {
        int error;
        socklen_t len = sizeof(error);
        getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &len);

        if (error == 0) {
            // Restore blocking mode
            fcntl(sock, F_SETFL, flags);
            return sock;
        }
    }

    close(sock);
    return -1;
}

// Send and receive helpers
ssize_t send_all(int sock, const void *buf, size_t len) {
    const char *ptr = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t sent = send(sock, ptr, remaining, 0);
        if (sent <= 0) return -1;
        ptr += sent;
        remaining -= sent;
    }

    return len;
}

ssize_t recv_until(int sock, char *buf, size_t maxlen, const char *delim) {
    size_t total = 0;
    size_t delim_len = strlen(delim);

    while (total < maxlen - 1) {
        ssize_t n = recv(sock, buf + total, 1, 0);
        if (n <= 0) break;
        total++;

        if (total >= delim_len &&
            memcmp(buf + total - delim_len, delim, delim_len) == 0) {
            break;
        }
    }

    buf[total] = '\\0';
    return total;
}

// Example: HTTP request
void http_get(const char *host, const char *path) {
    int sock = tcp_connect(host, 80);
    if (sock < 0) {
        printf("Connection failed\\n");
        return;
    }

    char request[1024];
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\\r\\n"
             "Host: %s\\r\\n"
             "Connection: close\\r\\n"
             "\\r\\n", path, host);

    send_all(sock, request, strlen(request));

    char response[4096];
    ssize_t n = recv(sock, response, sizeof(response) - 1, 0);
    response[n] = '\\0';

    printf("Response:\\n%s\\n", response);
    close(sock);
}

int main() {
    http_get("example.com", "/");
    return 0;
}
```

### Key Concepts
- TCP: reliable, connection-oriented
- Non-blocking I/O with select/poll
- Proper error handling for network code
- Send/receive helpers for complete transfers

### Completion Criteria
- [ ] Implement TCP client and server
- [ ] Add timeout handling for connections
- [ ] Build HTTP client from scratch
- [ ] Handle partial sends/receives correctly""",

    "process injection": """## Overview
Process injection techniques allow code execution within another process's address space - essential for understanding malware and building security tools.

### Implementation
```c
// Windows process injection example (for educational purposes)
#include <windows.h>
#include <stdio.h>

// Classic DLL injection
BOOL InjectDLL(DWORD pid, const char *dllPath) {
    HANDLE hProcess = OpenProcess(
        PROCESS_CREATE_THREAD | PROCESS_VM_OPERATION |
        PROCESS_VM_WRITE | PROCESS_QUERY_INFORMATION,
        FALSE, pid);

    if (!hProcess) return FALSE;

    // Allocate memory in target process
    size_t pathLen = strlen(dllPath) + 1;
    LPVOID remotePath = VirtualAllocEx(hProcess, NULL, pathLen,
                                        MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (!remotePath) {
        CloseHandle(hProcess);
        return FALSE;
    }

    // Write DLL path to target
    WriteProcessMemory(hProcess, remotePath, dllPath, pathLen, NULL);

    // Get LoadLibraryA address (same in all processes)
    LPVOID loadLibAddr = GetProcAddress(GetModuleHandle("kernel32.dll"), "LoadLibraryA");

    // Create remote thread to load DLL
    HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0,
                                        (LPTHREAD_START_ROUTINE)loadLibAddr,
                                        remotePath, 0, NULL);

    if (hThread) {
        WaitForSingleObject(hThread, INFINITE);
        CloseHandle(hThread);
    }

    VirtualFreeEx(hProcess, remotePath, 0, MEM_RELEASE);
    CloseHandle(hProcess);

    return hThread != NULL;
}

// Shellcode injection
BOOL InjectShellcode(DWORD pid, const unsigned char *shellcode, size_t size) {
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    if (!hProcess) return FALSE;

    // Allocate executable memory
    LPVOID remoteAddr = VirtualAllocEx(hProcess, NULL, size,
                                        MEM_COMMIT | MEM_RESERVE,
                                        PAGE_EXECUTE_READWRITE);

    if (!remoteAddr) {
        CloseHandle(hProcess);
        return FALSE;
    }

    // Write shellcode
    WriteProcessMemory(hProcess, remoteAddr, shellcode, size, NULL);

    // Execute via CreateRemoteThread
    HANDLE hThread = CreateRemoteThread(hProcess, NULL, 0,
                                        (LPTHREAD_START_ROUTINE)remoteAddr,
                                        NULL, 0, NULL);

    if (hThread) {
        CloseHandle(hThread);
    }

    CloseHandle(hProcess);
    return hThread != NULL;
}

// Process hollowing (RunPE)
BOOL HollowProcess(const char *targetPath, const char *payloadPath) {
    STARTUPINFO si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    // Create suspended process
    if (!CreateProcess(targetPath, NULL, NULL, NULL, FALSE,
                       CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        return FALSE;
    }

    // Get thread context (contains EAX = entry point)
    CONTEXT ctx;
    ctx.ContextFlags = CONTEXT_FULL;
    GetThreadContext(pi.hThread, &ctx);

    // Read payload PE...
    // Unmap original executable...
    // Map payload sections...
    // Update entry point in context...
    // Resume thread

    ResumeThread(pi.hThread);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return TRUE;
}
```

### Key Concepts
- DLL injection via LoadLibrary
- Shellcode injection to executable memory
- Process hollowing replaces legitimate code
- Requires appropriate process permissions

### Completion Criteria
- [ ] Implement DLL injection
- [ ] Implement shellcode injection
- [ ] Study process hollowing technique
- [ ] Understand detection methods""",
}

# Domain-specific templates for tasks without exact matches
DOMAIN_TEMPLATES = {
    "ai_ml": """## Overview
{task_name} is a key concept in machine learning that involves {brief_description}.

### Implementation
```python
import torch
import torch.nn as nn
import numpy as np

# {task_name} implementation
class {class_name}:
    def __init__(self):
        pass

    def forward(self, x):
        # Implementation details
        pass

    def train_step(self, batch):
        # Training logic
        pass

# Example usage
def main():
    model = {class_name}()
    # Training loop
    for epoch in range(100):
        loss = model.train_step(batch)
        print(f"Epoch {{epoch}}: Loss = {{loss:.4f}}")
```

### Key Concepts
- Understanding the theory behind {task_name}
- Implementation considerations and trade-offs
- Common pitfalls and best practices
- Integration with existing ML pipelines

### Completion Criteria
- [ ] Understand theoretical foundations
- [ ] Implement core functionality
- [ ] Test with sample data
- [ ] Optimize for performance""",

    "security": """## Overview
{task_name} is an important security concept that involves {brief_description}.

### Implementation
```python
import socket
import struct
from typing import Optional

class {class_name}:
    def __init__(self, target: str, port: int):
        self.target = target
        self.port = port

    def execute(self) -> bool:
        # Implementation details
        try:
            # Core logic
            return True
        except Exception as e:
            print(f"Error: {{e}}")
            return False

# Example usage
if __name__ == "__main__":
    tool = {class_name}("192.168.1.1", 443)
    result = tool.execute()
    print(f"Result: {{result}}")
```

### Key Concepts
- Understanding the attack surface
- Defense mechanisms and bypasses
- Detection and logging considerations
- Legal and ethical implications

### Completion Criteria
- [ ] Understand the technique thoroughly
- [ ] Implement working proof-of-concept
- [ ] Test in controlled environment
- [ ] Document findings and mitigations""",

    "systems": """## Overview
{task_name} involves {brief_description} at the systems programming level.

### Implementation
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// {task_name} implementation
typedef struct {{
    // Data structures
}} context_t;

int initialize(context_t *ctx) {{
    // Initialization logic
    return 0;
}}

int process(context_t *ctx, void *data, size_t len) {{
    // Core processing
    return 0;
}}

void cleanup(context_t *ctx) {{
    // Resource cleanup
}}

int main(int argc, char **argv) {{
    context_t ctx;

    if (initialize(&ctx) != 0) {{
        fprintf(stderr, "Initialization failed\\n");
        return 1;
    }}

    // Main logic

    cleanup(&ctx);
    return 0;
}}
```

### Key Concepts
- Low-level system interactions
- Memory management considerations
- Error handling and edge cases
- Performance optimization techniques

### Completion Criteria
- [ ] Understand system calls involved
- [ ] Implement core functionality
- [ ] Handle errors gracefully
- [ ] Test thoroughly with edge cases""",

    "project": """## Overview
{task_name} is a component of this project that handles {brief_description}.

### Implementation
```python
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class {class_name}:
    def __init__(self, config: dict):
        self.config = config
        self._initialize()

    def _initialize(self):
        # Setup logic
        pass

    def process(self, data: any) -> Optional[any]:
        try:
            # Main processing logic
            result = self._transform(data)
            return result
        except Exception as e:
            logger.error(f"Processing error: {{e}}")
            return None

    def _transform(self, data):
        # Transformation logic
        return data

# Usage
if __name__ == "__main__":
    processor = {class_name}({{"key": "value"}})
    result = processor.process("input")
    print(f"Result: {{result}}")
```

### Key Concepts
- Component architecture and interfaces
- Error handling and logging
- Configuration management
- Testing strategies

### Completion Criteria
- [ ] Design component interface
- [ ] Implement core functionality
- [ ] Add comprehensive error handling
- [ ] Write unit tests""",
}

def get_class_name(title: str) -> str:
    """Convert task title to class name."""
    words = title.replace('-', ' ').replace('_', ' ').split()
    return ''.join(word.capitalize() for word in words if word.isalpha())[:30]

def get_brief_description(title: str) -> str:
    """Generate brief description from title."""
    return title.lower()

def find_matching_details(title: str) -> str:
    """Find matching details from TASK_DETAILS or generate from template."""
    title_lower = title.lower()

    # Check for exact or partial matches in TASK_DETAILS
    for key, details in TASK_DETAILS.items():
        if key in title_lower or title_lower in key:
            return details

    # Check for keyword matches
    keywords = title_lower.split()
    for key, details in TASK_DETAILS.items():
        key_words = key.split()
        if any(kw in key_words for kw in keywords if len(kw) > 3):
            return details

    return None

def determine_domain(path_name: str) -> str:
    """Determine the domain based on path name."""
    path_lower = path_name.lower()

    if any(x in path_lower for x in ['ai', 'ml', 'deep learning', 'transformer', 'neural']):
        return 'ai_ml'
    elif any(x in path_lower for x in ['red team', 'security', 'hacking', 'exploit', 'offensive']):
        return 'security'
    elif any(x in path_lower for x in ['container', 'shell', 'system', 'c/', 'rust', 'c#']):
        return 'systems'
    else:
        return 'project'

def generate_from_template(title: str, domain: str) -> str:
    """Generate details from domain template."""
    template = DOMAIN_TEMPLATES.get(domain, DOMAIN_TEMPLATES['project'])

    class_name = get_class_name(title)
    brief_desc = get_brief_description(title)

    return template.format(
        task_name=title,
        class_name=class_name or "Implementation",
        brief_description=brief_desc
    )

def enhance_tasks():
    """Enhance all tasks with light details."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get tasks needing enhancement
    cursor.execute("""
        SELECT t.id, t.title, p.name, LENGTH(t.details) as len
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE LENGTH(t.details) < 900
        ORDER BY p.name, t.title
    """)

    tasks = cursor.fetchall()
    print(f"Found {len(tasks)} tasks needing enhancement")

    enhanced = 0
    for task_id, title, path_name, current_len in tasks:
        # Try to find matching details
        details = find_matching_details(title)

        if not details:
            # Generate from template
            domain = determine_domain(path_name)
            details = generate_from_template(title, domain)

        # Update task
        cursor.execute("UPDATE tasks SET details = ? WHERE id = ?", (details, task_id))
        enhanced += 1

        if enhanced % 50 == 0:
            print(f"  Enhanced {enhanced}/{len(tasks)} tasks...")
            conn.commit()

    conn.commit()
    conn.close()

    print(f"Done! Enhanced {enhanced} tasks.")

if __name__ == "__main__":
    enhance_tasks()
