#!/usr/bin/env python3
"""Fix ALL tasks with generic 'component of this project' content."""

import sqlite3
import re

DB_PATH = "data/quest-log.db"

# Domain-specific content generators based on path name keywords
def generate_details(path_name: str, task_title: str) -> str:
    """Generate domain-specific details based on path and task."""

    path_lower = path_name.lower()
    task_lower = task_title.lower()

    # Determine the domain
    if 'dns' in path_lower:
        return generate_dns_content(task_title)
    elif 'http' in path_lower and 'server' in path_lower:
        return generate_http_server_content(task_title)
    elif 'load balancer' in path_lower:
        return generate_load_balancer_content(task_title)
    elif 'packet sniffer' in path_lower:
        return generate_packet_sniffer_content(task_title)
    elif 'tls' in path_lower:
        return generate_tls_content(task_title)
    elif 'async runtime' in path_lower:
        return generate_async_runtime_content(task_title)
    elif 'debugger' in path_lower:
        return generate_debugger_content(task_title)
    elif 'memory allocator' in path_lower:
        return generate_allocator_content(task_title)
    elif 'sqlite' in path_lower:
        return generate_sqlite_content(task_title)
    elif 'thread pool' in path_lower:
        return generate_thread_pool_content(task_title)
    elif 'compiler' in path_lower or 'programming language' in path_lower:
        return generate_compiler_content(task_title)
    elif 'ctf' in path_lower:
        return generate_ctf_content(task_title)
    elif 'devsecops' in path_lower:
        return generate_devsecops_content(task_title)
    elif 'evasion' in path_lower or 'payload' in path_lower:
        return generate_evasion_content(task_title)
    elif 'key-value' in path_lower or 'kv store' in path_lower:
        return generate_kvstore_content(task_title)
    elif 'message queue' in path_lower:
        return generate_mq_content(task_title)
    elif 'ray tracer' in path_lower:
        return generate_raytracer_content(task_title)
    elif 'password manager' in path_lower:
        return generate_password_manager_content(task_title)
    elif 'file sync' in path_lower:
        return generate_file_sync_content(task_title)
    elif 'proxy' in path_lower:
        return generate_proxy_content(task_title)
    elif 'log aggregator' in path_lower:
        return generate_log_aggregator_content(task_title)
    elif 'plugin' in path_lower or 'file processor' in path_lower:
        return generate_plugin_processor_content(task_title)
    elif 'monitoring' in path_lower:
        return generate_monitoring_content(task_title)
    elif 'git' in path_lower and 'analyzer' in path_lower:
        return generate_git_analyzer_content(task_title)
    elif 'process monitor' in path_lower or 'tui' in path_lower:
        return generate_process_monitor_content(task_title)
    elif 'task scheduler' in path_lower:
        return generate_task_scheduler_content(task_title)
    elif 'distributed' in path_lower or 'pipeline' in path_lower:
        return generate_distributed_content(task_title)
    else:
        return generate_generic_project_content(path_name, task_title)


def generate_dns_content(task: str) -> str:
    task_l = task.lower()
    if 'caching' in task_l:
        return """## DNS Response Caching

### Cache Implementation
```go
type DNSCache struct {
    entries map[string]*CacheEntry
    mu      sync.RWMutex
}

type CacheEntry struct {
    Records   []ResourceRecord
    ExpiresAt time.Time
    TTL       uint32
}

func (c *DNSCache) Get(name string, qtype uint16) ([]ResourceRecord, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    key := fmt.Sprintf("%s:%d", name, qtype)
    entry, ok := c.entries[key]
    if !ok || time.Now().After(entry.ExpiresAt) {
        return nil, false
    }
    return entry.Records, true
}

func (c *DNSCache) Set(name string, qtype uint16, records []ResourceRecord, ttl uint32) {
    c.mu.Lock()
    defer c.mu.Unlock()

    key := fmt.Sprintf("%s:%d", name, qtype)
    c.entries[key] = &CacheEntry{
        Records:   records,
        ExpiresAt: time.Now().Add(time.Duration(ttl) * time.Second),
        TTL:       ttl,
    }
}

// Periodic cleanup of expired entries
func (c *DNSCache) StartCleanup(interval time.Duration) {
    go func() {
        ticker := time.NewTicker(interval)
        for range ticker.C {
            c.cleanup()
        }
    }()
}
```

### Key Concepts
- Cache keyed by name + query type
- Respect TTL from DNS responses
- Implement negative caching (NXDOMAIN)
- Periodic cleanup of expired entries

### Completion Criteria
- [ ] Implement TTL-based cache expiration
- [ ] Add negative caching for NXDOMAIN
- [ ] Implement cache size limits
- [ ] Add cache statistics/metrics"""

    elif 'health' in task_l:
        return """## DNS Health Checks

### Upstream Server Health
```go
type HealthChecker struct {
    upstreams []*Upstream
    interval  time.Duration
}

type Upstream struct {
    Address string
    Healthy bool
    Latency time.Duration
    mu      sync.RWMutex
}

func (h *HealthChecker) Start() {
    go func() {
        ticker := time.NewTicker(h.interval)
        for range ticker.C {
            for _, upstream := range h.upstreams {
                go h.checkUpstream(upstream)
            }
        }
    }()
}

func (h *HealthChecker) checkUpstream(u *Upstream) {
    start := time.Now()

    // Send DNS query for known domain
    conn, err := net.DialTimeout("udp", u.Address, 2*time.Second)
    if err != nil {
        u.mu.Lock()
        u.Healthy = false
        u.mu.Unlock()
        return
    }
    defer conn.Close()

    // Send query and wait for response
    query := buildHealthQuery("google.com", TypeA)
    conn.SetDeadline(time.Now().Add(2 * time.Second))
    conn.Write(query)

    buf := make([]byte, 512)
    _, err = conn.Read(buf)

    u.mu.Lock()
    u.Healthy = err == nil
    u.Latency = time.Since(start)
    u.mu.Unlock()
}
```

### Completion Criteria
- [ ] Implement periodic health checks
- [ ] Track upstream latency
- [ ] Failover to healthy upstreams
- [ ] Add health status endpoint"""

    elif 'config' in task_l:
        return """## DNS Server Configuration

### Configuration Schema
```yaml
# dns-server.yaml
server:
  listen: "0.0.0.0:53"
  protocol: "udp"  # udp, tcp, both

zones:
  - name: "example.com"
    file: "zones/example.com.zone"

upstream:
  - address: "8.8.8.8:53"
    timeout: 5s
  - address: "1.1.1.1:53"
    timeout: 5s

cache:
  enabled: true
  max_size: 10000
  default_ttl: 300

logging:
  level: "info"
  format: "json"
```

### Implementation
```go
type Config struct {
    Server   ServerConfig    `yaml:"server"`
    Zones    []ZoneConfig    `yaml:"zones"`
    Upstream []UpstreamConfig `yaml:"upstream"`
    Cache    CacheConfig     `yaml:"cache"`
}

func LoadConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }

    var cfg Config
    if err := yaml.Unmarshal(data, &cfg); err != nil {
        return nil, err
    }

    return &cfg, nil
}
```

### Completion Criteria
- [ ] Define YAML configuration schema
- [ ] Support multiple zone files
- [ ] Configure upstream resolvers
- [ ] Add cache settings"""

    else:
        return f"""## {task}

### DNS Server Implementation
```go
// {task} for DNS server
func implement{task.replace(' ', '')}() {{
    // DNS-specific implementation
    // Handle DNS message format (RFC 1035)
    // Support A, AAAA, CNAME, MX, NS, TXT records
}}
```

### Key Concepts
- DNS message format (header + questions + answers)
- Resource record types and their formats
- Name compression for efficient encoding
- Recursive vs authoritative resolution

### Completion Criteria
- [ ] Implement core functionality
- [ ] Handle edge cases
- [ ] Add proper error handling
- [ ] Test with dig/nslookup"""


def generate_http_server_content(task: str) -> str:
    task_l = task.lower()
    if 'caching' in task_l:
        return """## HTTP Response Caching

### Cache Implementation
```go
type ResponseCache struct {
    entries map[string]*CachedResponse
    mu      sync.RWMutex
    maxSize int
}

type CachedResponse struct {
    StatusCode int
    Headers    http.Header
    Body       []byte
    CachedAt   time.Time
    MaxAge     time.Duration
}

func (c *ResponseCache) Get(key string) (*CachedResponse, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    entry, ok := c.entries[key]
    if !ok {
        return nil, false
    }

    // Check if expired
    if time.Since(entry.CachedAt) > entry.MaxAge {
        return nil, false
    }

    return entry, true
}

// Parse Cache-Control header
func parseCacheControl(header string) (maxAge time.Duration, noCache bool) {
    directives := strings.Split(header, ",")
    for _, d := range directives {
        d = strings.TrimSpace(d)
        if d == "no-cache" || d == "no-store" {
            return 0, true
        }
        if strings.HasPrefix(d, "max-age=") {
            seconds, _ := strconv.Atoi(strings.TrimPrefix(d, "max-age="))
            return time.Duration(seconds) * time.Second, false
        }
    }
    return 0, false
}
```

### Completion Criteria
- [ ] Parse Cache-Control headers
- [ ] Implement cache storage with TTL
- [ ] Handle ETag/If-None-Match
- [ ] Add cache invalidation"""

    elif 'config' in task_l:
        return """## HTTP Server Configuration

### Configuration Schema
```yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: 30s
  write_timeout: 30s

tls:
  enabled: false
  cert_file: "server.crt"
  key_file: "server.key"

static:
  enabled: true
  root: "./public"
  index: "index.html"

logging:
  level: "info"
  access_log: true
```

### Implementation
```go
type ServerConfig struct {
    Host         string        `yaml:"host"`
    Port         int           `yaml:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
}

type Config struct {
    Server ServerConfig `yaml:"server"`
    TLS    TLSConfig    `yaml:"tls"`
    Static StaticConfig `yaml:"static"`
}
```

### Completion Criteria
- [ ] Define configuration schema
- [ ] Support TLS configuration
- [ ] Add timeout settings
- [ ] Configure static file serving"""

    elif 'health' in task_l:
        return """## HTTP Health Endpoints

### Implementation
```go
type HealthHandler struct {
    checks []HealthCheck
}

type HealthCheck struct {
    Name  string
    Check func() error
}

func (h *HealthHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    status := "healthy"
    results := make(map[string]string)
    httpStatus := http.StatusOK

    for _, check := range h.checks {
        if err := check.Check(); err != nil {
            status = "unhealthy"
            results[check.Name] = err.Error()
            httpStatus = http.StatusServiceUnavailable
        } else {
            results[check.Name] = "ok"
        }
    }

    response := map[string]interface{}{
        "status":  status,
        "checks":  results,
        "version": version,
    }

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(httpStatus)
    json.NewEncoder(w).Encode(response)
}
```

### Completion Criteria
- [ ] Implement /health endpoint
- [ ] Add component health checks
- [ ] Return appropriate HTTP status
- [ ] Include version information"""

    else:
        return f"""## {task}

### HTTP Server Implementation
```go
// {task} for HTTP server
type HTTPServer struct {{
    router   *Router
    listener net.Listener
}}

func (s *HTTPServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {{
    // Parse request
    // Route to handler
    // Write response
}}
```

### Key Concepts
- HTTP/1.1 request/response format
- Keep-alive connection management
- Content-Type and MIME types
- Status codes and error handling

### Completion Criteria
- [ ] Implement core functionality
- [ ] Handle edge cases properly
- [ ] Add comprehensive error handling
- [ ] Test with curl/browser"""


def generate_load_balancer_content(task: str) -> str:
    task_l = task.lower()
    if 'caching' in task_l:
        return """## Load Balancer Response Caching

### Cache Layer
```go
type LBCache struct {
    store   map[string]*CacheEntry
    mu      sync.RWMutex
    maxSize int
}

func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // Check cache first
    cacheKey := r.Method + ":" + r.URL.Path
    if entry, ok := lb.cache.Get(cacheKey); ok {
        for k, v := range entry.Headers {
            w.Header()[k] = v
        }
        w.WriteHeader(entry.StatusCode)
        w.Write(entry.Body)
        return
    }

    // Forward to backend
    backend := lb.NextBackend()
    resp := lb.forwardRequest(backend, r)

    // Cache if cacheable
    if isCacheable(resp) {
        lb.cache.Set(cacheKey, resp)
    }

    writeResponse(w, resp)
}
```

### Completion Criteria
- [ ] Implement response caching layer
- [ ] Respect Cache-Control headers
- [ ] Add cache invalidation
- [ ] Configure cache size limits"""

    elif 'health' in task_l:
        return """## Load Balancer Health Checks

### Active Health Checking
```go
type HealthChecker struct {
    backends []*Backend
    interval time.Duration
    timeout  time.Duration
}

func (h *HealthChecker) Start() {
    ticker := time.NewTicker(h.interval)
    for range ticker.C {
        for _, backend := range h.backends {
            go h.check(backend)
        }
    }
}

func (h *HealthChecker) check(b *Backend) {
    client := &http.Client{Timeout: h.timeout}
    resp, err := client.Get(b.URL + "/health")

    b.mu.Lock()
    defer b.mu.Unlock()

    if err != nil || resp.StatusCode != 200 {
        b.FailCount++
        if b.FailCount >= 3 {
            b.Healthy = false
        }
    } else {
        b.FailCount = 0
        b.Healthy = true
    }
}
```

### Completion Criteria
- [ ] Implement periodic health checks
- [ ] Track consecutive failures
- [ ] Remove unhealthy backends from rotation
- [ ] Add passive health monitoring"""

    elif 'config' in task_l:
        return """## Load Balancer Configuration

### Configuration Schema
```yaml
listen: ":8080"

algorithm: "round_robin"  # round_robin, least_conn, ip_hash

backends:
  - url: "http://backend1:8080"
    weight: 1
    health_check: "/health"
  - url: "http://backend2:8080"
    weight: 2
    health_check: "/health"

health_check:
  interval: 10s
  timeout: 5s
  threshold: 3

sticky_sessions:
  enabled: false
  cookie_name: "SERVERID"
```

### Completion Criteria
- [ ] Configure backend servers
- [ ] Set load balancing algorithm
- [ ] Configure health check parameters
- [ ] Add sticky session settings"""

    else:
        return f"""## {task}

### Load Balancer Implementation
```go
type LoadBalancer struct {{
    backends  []*Backend
    algorithm Algorithm
    current   uint32
}}

func (lb *LoadBalancer) NextBackend() *Backend {{
    switch lb.algorithm {{
    case RoundRobin:
        return lb.roundRobin()
    case LeastConn:
        return lb.leastConnections()
    case IPHash:
        return lb.ipHash()
    }}
    return nil
}}
```

### Key Concepts
- Load balancing algorithms (round-robin, least connections)
- Health checking and failover
- Session persistence (sticky sessions)
- Connection draining

### Completion Criteria
- [ ] Implement core functionality
- [ ] Handle backend failures gracefully
- [ ] Add proper logging
- [ ] Test with multiple backends"""


def generate_ctf_content(task: str) -> str:
    return f"""## {task} - CTF Skills

### Challenge Approach
```python
# CTF methodology for {task}
# 1. Reconnaissance - gather information
# 2. Analysis - understand the challenge
# 3. Exploitation - find and exploit vulnerability
# 4. Flag capture - extract the flag

def solve_challenge():
    # Analyze the challenge
    # Identify the vulnerability class
    # Web: SQLi, XSS, SSRF, IDOR
    # Pwn: Buffer overflow, format string, ROP
    # Crypto: Weak algorithms, implementation flaws
    # Rev: Static/dynamic analysis
    pass
```

### Tools & Techniques
- **Web**: Burp Suite, SQLMap, XSStrike
- **Pwn**: GDB, pwntools, ROPgadget
- **Crypto**: CyberChef, SageMath, hashcat
- **Rev**: Ghidra, IDA, radare2
- **Forensics**: Autopsy, Volatility, binwalk

### Practice Resources
- HackTheBox, TryHackMe
- PicoCTF, OverTheWire
- CTFtime.org for competitions

### Completion Criteria
- [ ] Understand the challenge category
- [ ] Apply appropriate techniques
- [ ] Document the solution process
- [ ] Practice similar challenges"""


def generate_devsecops_content(task: str) -> str:
    task_l = task.lower()
    if 'dashboard' in task_l or 'testing' in task_l:
        return """## Security Testing Dashboard

### Dashboard Components
```python
from flask import Flask, render_template
import json

app = Flask(__name__)

class SecurityDashboard:
    def __init__(self):
        self.sast_results = []
        self.dast_results = []
        self.dependency_vulns = []

    def aggregate_results(self):
        return {
            'sast': {
                'critical': len([r for r in self.sast_results if r['severity'] == 'critical']),
                'high': len([r for r in self.sast_results if r['severity'] == 'high']),
                'medium': len([r for r in self.sast_results if r['severity'] == 'medium']),
            },
            'dast': len(self.dast_results),
            'dependencies': len(self.dependency_vulns),
            'trend': self.calculate_trend()
        }

@app.route('/dashboard')
def dashboard():
    data = SecurityDashboard().aggregate_results()
    return render_template('dashboard.html', data=data)
```

### Key Metrics
- Vulnerability counts by severity
- Mean time to remediate (MTTR)
- Code coverage for security tests
- Compliance status

### Completion Criteria
- [ ] Aggregate SAST/DAST results
- [ ] Display metrics and trends
- [ ] Add filtering and drill-down
- [ ] Integrate with CI/CD pipeline"""

    elif 'kubernetes' in task_l or 'k8s' in task_l:
        return """## Kubernetes Security Policies

### Pod Security Standards
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'secret'
  hostNetwork: false
  hostIPC: false
  hostPID: false
```

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
spec:
  podSelector: {}
  policyTypes:
    - Ingress
  ingress: []
```

### Completion Criteria
- [ ] Implement Pod Security Standards
- [ ] Configure Network Policies
- [ ] Set up RBAC properly
- [ ] Enable audit logging"""

    else:
        return f"""## {task} - DevSecOps

### Implementation
```yaml
# CI/CD Security Integration
stages:
  - security-scan

security_scan:
  stage: security-scan
  script:
    - semgrep --config auto .
    - trivy image $IMAGE_NAME
    - gitleaks detect --source .
  allow_failure: false
```

### Security Gates
- SAST: Static analysis (Semgrep, CodeQL)
- SCA: Dependency scanning (Snyk, Trivy)
- DAST: Dynamic testing (OWASP ZAP)
- Secret scanning (Gitleaks, TruffleHog)

### Best Practices
- Shift-left security testing
- Infrastructure as Code scanning
- Container image scanning
- Policy as Code enforcement

### Completion Criteria
- [ ] Integrate security into CI/CD
- [ ] Configure security gates
- [ ] Set up alerting and reporting
- [ ] Document security processes"""


def generate_evasion_content(task: str) -> str:
    return f"""## {task} - Evasion Techniques

### Implementation Approach
```c
// Evasion technique: {task}
// Educational implementation for security research

// Key concepts:
// - Understanding detection mechanisms
// - Analyzing security controls
// - Developing bypass techniques
// - Testing in controlled environments
```

### Considerations
- AV/EDR detection methods
- Signature-based vs behavioral detection
- Memory scanning and heuristics
- API hooking and monitoring

### Detection Categories
- Static analysis: Signatures, heuristics
- Dynamic analysis: Sandbox, emulation
- Behavioral: API calls, network traffic

### Completion Criteria
- [ ] Research detection mechanisms
- [ ] Understand the technique
- [ ] Implement proof-of-concept
- [ ] Test in isolated environment"""


def generate_async_runtime_content(task: str) -> str:
    return f"""## {task} - Async Runtime

### Implementation
```rust
// Async runtime component: {task}
use std::future::Future;
use std::task::{{Context, Poll, Waker}};
use std::pin::Pin;

struct Task {{
    future: Pin<Box<dyn Future<Output = ()> + Send>>,
    waker: Option<Waker>,
}}

struct Executor {{
    tasks: Vec<Task>,
    ready_queue: VecDeque<usize>,
}}

impl Executor {{
    fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {{
        self.tasks.push(Task {{
            future: Box::pin(future),
            waker: None,
        }});
    }}

    fn run(&mut self) {{
        // Poll ready tasks
        // Re-queue when woken
    }}
}}
```

### Key Concepts
- Future trait and polling
- Waker mechanism for task notification
- Task scheduling and execution
- I/O event integration (epoll/kqueue)

### Completion Criteria
- [ ] Understand Future/Poll mechanics
- [ ] Implement task spawning
- [ ] Build executor run loop
- [ ] Integrate with I/O events"""


def generate_debugger_content(task: str) -> str:
    return f"""## {task} - Debugger

### Implementation
```c
// Debugger component: {task}
#include <sys/ptrace.h>
#include <sys/wait.h>

struct Debugger {{
    pid_t target_pid;
    struct breakpoint *breakpoints;
}};

// Attach to process
int attach(pid_t pid) {{
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {{
        return -1;
    }}
    waitpid(pid, NULL, 0);
    return 0;
}}

// Set breakpoint
int set_breakpoint(void *addr) {{
    // Read original instruction
    long orig = ptrace(PTRACE_PEEKTEXT, target_pid, addr, NULL);
    // Write INT3 (0xCC)
    ptrace(PTRACE_POKETEXT, target_pid, addr, (orig & ~0xFF) | 0xCC);
    return 0;
}}
```

### Key Concepts
- ptrace system call
- Breakpoint insertion (INT3)
- Single stepping
- Register inspection/modification

### Completion Criteria
- [ ] Implement process attachment
- [ ] Add breakpoint support
- [ ] Implement single stepping
- [ ] Add register/memory inspection"""


def generate_allocator_content(task: str) -> str:
    return f"""## {task} - Memory Allocator

### Implementation
```c
// Memory allocator: {task}
#include <stddef.h>
#include <stdint.h>

typedef struct block_header {{
    size_t size;
    int free;
    struct block_header *next;
}} block_header_t;

static block_header_t *free_list = NULL;

void *my_malloc(size_t size) {{
    // Align to 8 bytes
    size = (size + 7) & ~7;

    // First-fit search
    block_header_t *curr = free_list;
    while (curr) {{
        if (curr->free && curr->size >= size) {{
            curr->free = 0;
            return (void *)(curr + 1);
        }}
        curr = curr->next;
    }}

    // Request more memory from OS
    // sbrk() or mmap()
    return NULL;
}}

void my_free(void *ptr) {{
    if (!ptr) return;
    block_header_t *header = (block_header_t *)ptr - 1;
    header->free = 1;
    // Coalesce adjacent free blocks
}}
```

### Key Concepts
- Block headers and metadata
- Free list management
- Coalescing free blocks
- Memory alignment

### Completion Criteria
- [ ] Implement basic malloc/free
- [ ] Add block coalescing
- [ ] Handle alignment properly
- [ ] Optimize for performance"""


def generate_sqlite_content(task: str) -> str:
    return f"""## {task} - SQLite Clone

### Implementation
```c
// SQLite component: {task}
typedef struct {{
    int type;        // Page type
    int cell_count;  // Number of cells
    char *data;      // Page data
}} Page;

typedef struct {{
    FILE *file;
    int page_size;
    int page_count;
    Page *cache;
}} Database;

// B-tree operations
typedef struct BTreeNode {{
    int is_leaf;
    int key_count;
    int keys[MAX_KEYS];
    struct BTreeNode *children[MAX_KEYS + 1];
}} BTreeNode;

BTreeNode *btree_search(BTreeNode *root, int key) {{
    int i = 0;
    while (i < root->key_count && key > root->keys[i]) {{
        i++;
    }}

    if (i < root->key_count && key == root->keys[i]) {{
        return root;
    }}

    if (root->is_leaf) {{
        return NULL;
    }}

    return btree_search(root->children[i], key);
}}
```

### Key Concepts
- B-tree data structure
- Page-based storage
- SQL parsing
- Query execution

### Completion Criteria
- [ ] Implement B-tree operations
- [ ] Build page manager
- [ ] Add SQL parser
- [ ] Execute basic queries"""


def generate_thread_pool_content(task: str) -> str:
    return f"""## {task} - Thread Pool

### Implementation
```c
// Thread pool: {task}
#include <pthread.h>
#include <stdlib.h>

typedef struct {{
    void (*function)(void *);
    void *argument;
}} Task;

typedef struct {{
    pthread_t *threads;
    Task *queue;
    int queue_size;
    int queue_front;
    int queue_rear;
    int queue_count;
    pthread_mutex_t lock;
    pthread_cond_t notify;
    int shutdown;
    int thread_count;
}} ThreadPool;

void *worker(void *arg) {{
    ThreadPool *pool = (ThreadPool *)arg;

    while (1) {{
        pthread_mutex_lock(&pool->lock);

        while (pool->queue_count == 0 && !pool->shutdown) {{
            pthread_cond_wait(&pool->notify, &pool->lock);
        }}

        if (pool->shutdown) {{
            pthread_mutex_unlock(&pool->lock);
            break;
        }}

        Task task = pool->queue[pool->queue_front];
        pool->queue_front = (pool->queue_front + 1) % pool->queue_size;
        pool->queue_count--;

        pthread_mutex_unlock(&pool->lock);

        task.function(task.argument);
    }}

    return NULL;
}}
```

### Key Concepts
- Worker thread management
- Task queue with mutex/condvar
- Graceful shutdown
- Work stealing (advanced)

### Completion Criteria
- [ ] Implement worker threads
- [ ] Build thread-safe task queue
- [ ] Add graceful shutdown
- [ ] Handle task scheduling"""


def generate_compiler_content(task: str) -> str:
    return f"""## {task} - Compiler/Language

### Implementation
```c
// Compiler component: {task}
typedef enum {{
    TOKEN_INT, TOKEN_IDENT, TOKEN_PLUS, TOKEN_MINUS,
    TOKEN_LPAREN, TOKEN_RPAREN, TOKEN_EOF
}} TokenType;

typedef struct {{
    TokenType type;
    char *value;
}} Token;

// Lexer
Token *lex(const char *input) {{
    // Tokenize input
}}

// Parser - recursive descent
typedef struct ASTNode {{
    int type;
    struct ASTNode *left;
    struct ASTNode *right;
    int value;
}} ASTNode;

ASTNode *parse_expression() {{
    ASTNode *left = parse_term();
    while (current_token.type == TOKEN_PLUS) {{
        advance();
        ASTNode *right = parse_term();
        left = make_binary_node(OP_ADD, left, right);
    }}
    return left;
}}

// Code generation
void codegen(ASTNode *node) {{
    // Generate assembly/bytecode
}}
```

### Key Concepts
- Lexical analysis (tokenization)
- Parsing (recursive descent/Pratt)
- AST construction
- Code generation

### Completion Criteria
- [ ] Implement lexer
- [ ] Build parser
- [ ] Generate AST
- [ ] Emit code/bytecode"""


def generate_kvstore_content(task: str) -> str:
    return f"""## {task} - Key-Value Store

### Implementation
```go
// KV Store: {task}
type KVStore struct {{
    data map[string][]byte
    mu   sync.RWMutex
    wal  *WriteAheadLog
}}

func (kv *KVStore) Get(key string) ([]byte, bool) {{
    kv.mu.RLock()
    defer kv.mu.RUnlock()
    val, ok := kv.data[key]
    return val, ok
}}

func (kv *KVStore) Set(key string, value []byte) error {{
    // Write to WAL first
    if err := kv.wal.Append(SET, key, value); err != nil {{
        return err
    }}

    kv.mu.Lock()
    kv.data[key] = value
    kv.mu.Unlock()
    return nil
}}

func (kv *KVStore) Delete(key string) error {{
    if err := kv.wal.Append(DELETE, key, nil); err != nil {{
        return err
    }}

    kv.mu.Lock()
    delete(kv.data, key)
    kv.mu.Unlock()
    return nil
}}
```

### Key Concepts
- In-memory hash table
- Write-ahead logging (durability)
- Concurrent access with RWMutex
- Persistence strategies

### Completion Criteria
- [ ] Implement basic CRUD operations
- [ ] Add write-ahead logging
- [ ] Handle concurrent access
- [ ] Add persistence/recovery"""


def generate_mq_content(task: str) -> str:
    return f"""## {task} - Message Queue

### Implementation
```go
// Message Queue: {task}
type Message struct {{
    ID        string
    Topic     string
    Payload   []byte
    Timestamp time.Time
}}

type Queue struct {{
    topics map[string]*Topic
    mu     sync.RWMutex
}}

type Topic struct {{
    name     string
    messages chan *Message
    subs     []*Subscriber
}}

func (q *Queue) Publish(topic string, payload []byte) error {{
    q.mu.RLock()
    t, ok := q.topics[topic]
    q.mu.RUnlock()

    if !ok {{
        return ErrTopicNotFound
    }}

    msg := &Message{{
        ID:        uuid.New().String(),
        Topic:     topic,
        Payload:   payload,
        Timestamp: time.Now(),
    }}

    t.messages <- msg
    return nil
}}

func (q *Queue) Subscribe(topic string, handler func(*Message)) {{
    // Register subscriber
    // Start goroutine to consume messages
}}
```

### Key Concepts
- Pub/sub pattern
- Topic-based routing
- Message persistence
- Consumer groups

### Completion Criteria
- [ ] Implement publish/subscribe
- [ ] Add topic management
- [ ] Handle message acknowledgment
- [ ] Add persistence layer"""


def generate_raytracer_content(task: str) -> str:
    return f"""## {task} - Ray Tracer

### Implementation
```rust
// Ray Tracer: {task}
struct Vec3 {{
    x: f64,
    y: f64,
    z: f64,
}}

struct Ray {{
    origin: Vec3,
    direction: Vec3,
}}

struct Sphere {{
    center: Vec3,
    radius: f64,
    material: Material,
}}

fn ray_sphere_intersect(ray: &Ray, sphere: &Sphere) -> Option<f64> {{
    let oc = ray.origin - sphere.center;
    let a = ray.direction.dot(&ray.direction);
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.dot(&oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {{
        None
    }} else {{
        Some((-b - discriminant.sqrt()) / (2.0 * a))
    }}
}}

fn trace(ray: &Ray, scene: &Scene, depth: i32) -> Color {{
    // Find closest intersection
    // Calculate lighting
    // Recursive reflection/refraction
}}
```

### Key Concepts
- Ray-object intersection
- Lighting calculations (Phong)
- Reflection and refraction
- Anti-aliasing

### Completion Criteria
- [ ] Implement ray-sphere intersection
- [ ] Add lighting model
- [ ] Support reflections
- [ ] Add anti-aliasing"""


def generate_password_manager_content(task: str) -> str:
    return f"""## {task} - Password Manager

### Implementation
```python
# Password Manager: {task}
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class PasswordManager:
    def __init__(self, master_password: str):
        self.key = self._derive_key(master_password)
        self.fernet = Fernet(self.key)
        self.vault = {{}}

    def _derive_key(self, password: str) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def store(self, service: str, username: str, password: str):
        encrypted = self.fernet.encrypt(password.encode())
        self.vault[service] = {{
            'username': username,
            'password': encrypted
        }}

    def retrieve(self, service: str) -> dict:
        entry = self.vault.get(service)
        if entry:
            decrypted = self.fernet.decrypt(entry['password']).decode()
            return {{'username': entry['username'], 'password': decrypted}}
        return None
```

### Key Concepts
- Key derivation (PBKDF2)
- Symmetric encryption (AES/Fernet)
- Secure storage
- Memory protection

### Completion Criteria
- [ ] Implement master password derivation
- [ ] Add encrypted storage
- [ ] Secure memory handling
- [ ] Add password generation"""


def generate_file_sync_content(task: str) -> str:
    return f"""## {task} - File Sync

### Implementation
```go
// File Sync: {task}
type SyncEngine struct {{
    source string
    dest   string
    index  map[string]FileInfo
}}

type FileInfo struct {{
    Path     string
    Size     int64
    ModTime  time.Time
    Checksum string
}}

func (s *SyncEngine) Scan() ([]FileInfo, error) {{
    var files []FileInfo
    err := filepath.Walk(s.source, func(path string, info os.FileInfo, err error) error {{
        if err != nil || info.IsDir() {{
            return err
        }}

        checksum, _ := s.calculateChecksum(path)
        files = append(files, FileInfo{{
            Path:     path,
            Size:     info.Size(),
            ModTime:  info.ModTime(),
            Checksum: checksum,
        }})
        return nil
    }})
    return files, err
}}

func (s *SyncEngine) Sync() error {{
    // Compare source and dest
    // Copy new/modified files
    // Handle conflicts
    return nil
}}
```

### Key Concepts
- File system traversal
- Checksum comparison
- Delta synchronization
- Conflict resolution

### Completion Criteria
- [ ] Implement file scanning
- [ ] Add checksum comparison
- [ ] Handle file transfers
- [ ] Resolve conflicts"""


def generate_proxy_content(task: str) -> str:
    return f"""## {task} - HTTP Proxy

### Implementation
```go
// HTTP Proxy: {task}
type Proxy struct {{
    cache   *Cache
    client  *http.Client
}}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {{
    // Check cache
    if cached, ok := p.cache.Get(r.URL.String()); ok {{
        w.Write(cached)
        return
    }}

    // Forward request
    proxyReq, _ := http.NewRequest(r.Method, r.URL.String(), r.Body)
    copyHeaders(proxyReq.Header, r.Header)

    resp, err := p.client.Do(proxyReq)
    if err != nil {{
        http.Error(w, err.Error(), http.StatusBadGateway)
        return
    }}
    defer resp.Body.Close()

    // Copy response
    copyHeaders(w.Header(), resp.Header)
    w.WriteHeader(resp.StatusCode)

    body, _ := io.ReadAll(resp.Body)
    w.Write(body)

    // Cache if appropriate
    if isCacheable(resp) {{
        p.cache.Set(r.URL.String(), body)
    }}
}}
```

### Key Concepts
- Request forwarding
- Header manipulation
- Response caching
- Connection management

### Completion Criteria
- [ ] Implement request forwarding
- [ ] Add response caching
- [ ] Handle HTTPS (CONNECT)
- [ ] Add request/response logging"""


def generate_log_aggregator_content(task: str) -> str:
    return f"""## {task} - Log Aggregator

### Implementation
```go
// Log Aggregator: {task}
type LogEntry struct {{
    Timestamp time.Time
    Level     string
    Source    string
    Message   string
    Fields    map[string]interface{{}}
}}

type Aggregator struct {{
    inputs   []Input
    outputs  []Output
    pipeline chan *LogEntry
}}

func (a *Aggregator) Start() {{
    // Start input collectors
    for _, input := range a.inputs {{
        go input.Collect(a.pipeline)
    }}

    // Process pipeline
    go func() {{
        for entry := range a.pipeline {{
            // Parse, filter, transform
            for _, output := range a.outputs {{
                output.Write(entry)
            }}
        }}
    }}()
}}

type FileInput struct {{
    path string
}}

func (f *FileInput) Collect(out chan *LogEntry) {{
    // Tail file and parse log lines
}}
```

### Key Concepts
- Log collection from multiple sources
- Parsing different log formats
- Filtering and transformation
- Output to various destinations

### Completion Criteria
- [ ] Implement log collectors
- [ ] Add log parsing
- [ ] Support filtering/transformation
- [ ] Add multiple output formats"""


def generate_plugin_processor_content(task: str) -> str:
    return f"""## {task} - Plugin System

### Implementation
```go
// Plugin System: {task}
type Plugin interface {{
    Name() string
    Process(data []byte) ([]byte, error)
}}

type PluginManager struct {{
    plugins map[string]Plugin
}}

func (pm *PluginManager) LoadPlugin(path string) error {{
    p, err := plugin.Open(path)
    if err != nil {{
        return err
    }}

    sym, err := p.Lookup("Plugin")
    if err != nil {{
        return err
    }}

    plug, ok := sym.(Plugin)
    if !ok {{
        return errors.New("invalid plugin")
    }}

    pm.plugins[plug.Name()] = plug
    return nil
}}

func (pm *PluginManager) Process(name string, data []byte) ([]byte, error) {{
    plugin, ok := pm.plugins[name]
    if !ok {{
        return nil, ErrPluginNotFound
    }}
    return plugin.Process(data)
}}
```

### Key Concepts
- Plugin interface design
- Dynamic loading
- Plugin lifecycle management
- Error handling

### Completion Criteria
- [ ] Define plugin interface
- [ ] Implement plugin loading
- [ ] Add plugin management
- [ ] Handle plugin errors"""


def generate_monitoring_content(task: str) -> str:
    return f"""## {task} - Monitoring System

### Implementation
```go
// Monitoring: {task}
type Metric struct {{
    Name      string
    Value     float64
    Timestamp time.Time
    Tags      map[string]string
}}

type MetricCollector struct {{
    metrics chan *Metric
    storage MetricStorage
}}

func (c *MetricCollector) Collect(name string, value float64, tags map[string]string) {{
    c.metrics <- &Metric{{
        Name:      name,
        Value:     value,
        Timestamp: time.Now(),
        Tags:      tags,
    }}
}}

func (c *MetricCollector) Start() {{
    go func() {{
        batch := make([]*Metric, 0, 100)
        ticker := time.NewTicker(10 * time.Second)

        for {{
            select {{
            case m := <-c.metrics:
                batch = append(batch, m)
            case <-ticker.C:
                if len(batch) > 0 {{
                    c.storage.Write(batch)
                    batch = batch[:0]
                }}
            }}
        }}
    }}()
}}
```

### Key Concepts
- Metric collection and aggregation
- Time series storage
- Alerting rules
- Visualization

### Completion Criteria
- [ ] Implement metric collection
- [ ] Add time series storage
- [ ] Build alerting system
- [ ] Create dashboards"""


def generate_git_analyzer_content(task: str) -> str:
    return f"""## {task} - Git Analyzer

### Implementation
```go
// Git Analyzer: {task}
import "github.com/go-git/go-git/v5"

type RepoAnalyzer struct {{
    repo *git.Repository
}}

func (a *RepoAnalyzer) AnalyzeCommits() ([]CommitStats, error) {{
    iter, _ := a.repo.Log(&git.LogOptions{{}})
    var stats []CommitStats

    iter.ForEach(func(c *object.Commit) error {{
        files, _ := c.Stats()
        stats = append(stats, CommitStats{{
            Hash:      c.Hash.String(),
            Author:    c.Author.Name,
            Date:      c.Author.When,
            Message:   c.Message,
            Additions: sumAdditions(files),
            Deletions: sumDeletions(files),
        }})
        return nil
    }})

    return stats, nil
}}

func (a *RepoAnalyzer) FindLargeFiles() ([]FileInfo, error) {{
    // Walk tree and find large blobs
}}

func (a *RepoAnalyzer) DetectSecrets() ([]Secret, error) {{
    // Scan commits for secrets
}}
```

### Key Concepts
- Git object model (commits, trees, blobs)
- Repository traversal
- Code statistics
- Security scanning

### Completion Criteria
- [ ] Parse git objects
- [ ] Analyze commit history
- [ ] Calculate code statistics
- [ ] Detect sensitive data"""


def generate_process_monitor_content(task: str) -> str:
    return f"""## {task} - Process Monitor TUI

### Implementation
```go
// Process Monitor: {task}
import (
    "github.com/shirou/gopsutil/v3/process"
    "github.com/charmbracelet/bubbletea"
)

type Model struct {{
    processes []ProcessInfo
    cursor    int
    sortBy    string
}}

type ProcessInfo struct {{
    PID     int32
    Name    string
    CPU     float64
    Memory  float64
    Status  string
}}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {{
    switch msg := msg.(type) {{
    case tea.KeyMsg:
        switch msg.String() {{
        case "q":
            return m, tea.Quit
        case "up":
            m.cursor--
        case "down":
            m.cursor++
        case "k":
            // Kill selected process
        }}
    }}
    return m, nil
}}

func (m Model) View() string {{
    s := "PID\\tNAME\\tCPU\\tMEM\\n"
    for i, p := range m.processes {{
        cursor := " "
        if i == m.cursor {{
            cursor = ">"
        }}
        s += fmt.Sprintf("%s %d\\t%s\\t%.1f%%\\t%.1f%%\\n",
            cursor, p.PID, p.Name, p.CPU, p.Memory)
    }}
    return s
}}
```

### Key Concepts
- Process enumeration
- Resource monitoring
- TUI rendering
- Keyboard navigation

### Completion Criteria
- [ ] List running processes
- [ ] Display CPU/memory usage
- [ ] Add sorting/filtering
- [ ] Implement process actions"""


def generate_task_scheduler_content(task: str) -> str:
    return f"""## {task} - Task Scheduler

### Implementation
```go
// Task Scheduler: {task}
type Task struct {{
    ID       string
    Name     string
    Schedule string  // cron expression
    Command  string
    LastRun  time.Time
    NextRun  time.Time
}}

type Scheduler struct {{
    tasks   []*Task
    running map[string]bool
    mu      sync.Mutex
}}

func (s *Scheduler) Schedule(task *Task) {{
    task.NextRun = s.parseSchedule(task.Schedule)
    s.tasks = append(s.tasks, task)
}}

func (s *Scheduler) Run() {{
    ticker := time.NewTicker(time.Second)
    for now := range ticker.C {{
        for _, task := range s.tasks {{
            if now.After(task.NextRun) {{
                go s.execute(task)
                task.LastRun = now
                task.NextRun = s.parseSchedule(task.Schedule)
            }}
        }}
    }}
}}

func (s *Scheduler) execute(task *Task) {{
    s.mu.Lock()
    s.running[task.ID] = true
    s.mu.Unlock()

    cmd := exec.Command("sh", "-c", task.Command)
    cmd.Run()

    s.mu.Lock()
    delete(s.running, task.ID)
    s.mu.Unlock()
}}
```

### Key Concepts
- Cron expression parsing
- Time-based triggering
- Concurrent task execution
- Task persistence

### Completion Criteria
- [ ] Parse cron expressions
- [ ] Implement scheduling loop
- [ ] Handle concurrent tasks
- [ ] Add task persistence"""


def generate_distributed_content(task: str) -> str:
    return f"""## {task} - Distributed System

### Implementation
```go
// Distributed component: {task}
type Node struct {{
    ID       string
    Address  string
    Peers    []*Node
    Leader   *Node
}}

type Message struct {{
    Type    MessageType
    From    string
    To      string
    Payload []byte
}}

func (n *Node) BroadCast(msg *Message) {{
    for _, peer := range n.Peers {{
        go n.send(peer, msg)
    }}
}}

func (n *Node) send(peer *Node, msg *Message) error {{
    conn, err := net.Dial("tcp", peer.Address)
    if err != nil {{
        return err
    }}
    defer conn.Close()

    encoder := gob.NewEncoder(conn)
    return encoder.Encode(msg)
}}

// Consensus (simplified Raft)
func (n *Node) RequestVote() {{
    // Send RequestVote RPCs
}}

func (n *Node) AppendEntries() {{
    // Replicate log entries
}}
```

### Key Concepts
- Node communication
- Leader election
- Log replication
- Failure handling

### Completion Criteria
- [ ] Implement node discovery
- [ ] Add message passing
- [ ] Build consensus protocol
- [ ] Handle node failures"""


def generate_generic_project_content(path: str, task: str) -> str:
    return f"""## {task}

### Implementation for {path}
```python
# {task} implementation
class {task.replace(' ', '')}:
    def __init__(self):
        pass

    def execute(self):
        # Core logic
        pass
```

### Key Concepts
- Understand requirements
- Design clean interfaces
- Handle errors gracefully
- Write testable code

### Completion Criteria
- [ ] Implement core functionality
- [ ] Add error handling
- [ ] Write unit tests
- [ ] Document the implementation"""


def fix_generic_tasks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find all tasks with generic content
    cursor.execute("""
        SELECT t.id, t.title, p.name
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE t.details LIKE '%is a component of this project that handles%'
    """)

    tasks = cursor.fetchall()
    print(f"Found {len(tasks)} tasks with generic content")

    updated = 0
    for task_id, title, path_name in tasks:
        details = generate_details(path_name, title)
        cursor.execute("UPDATE tasks SET details = ? WHERE id = ?", (details, task_id))
        updated += 1

        if updated % 50 == 0:
            print(f"  Updated {updated}/{len(tasks)}...")
            conn.commit()

    conn.commit()
    conn.close()
    print(f"\nDone! Updated {updated} tasks.")


if __name__ == "__main__":
    fix_generic_tasks()
