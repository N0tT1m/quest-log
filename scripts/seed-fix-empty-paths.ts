import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

function addTasks(pathId: number, modules: { name: string; desc: string; tasks: [string, string][] }[]) {
	modules.forEach((mod, i) => {
		const m = insertModule.run(pathId, mod.name, mod.desc, i, now);
		mod.tasks.forEach(([title, desc], j) => {
			insertTask.run(m.lastInsertRowid, title, desc, '', j, now);
		});
	});
}

// Find all paths with 0 tasks
const emptyPaths = db.prepare(`
	SELECT p.id, p.name FROM paths p
	LEFT JOIN modules m ON m.path_id = p.id
	LEFT JOIN tasks t ON t.module_id = m.id
	GROUP BY p.id
	HAVING COUNT(t.id) = 0
`).all() as { id: number; name: string }[];

console.log(`Found ${emptyPaths.length} paths with no tasks`);

for (const path of emptyPaths) {
	console.log(`Adding tasks to: ${path.name}`);

	// Add generic but relevant tasks based on path name
	if (path.name.includes('Compiler') || path.name.includes('Programming Language')) {
		addTasks(path.id, [
			{ name: 'Lexical Analysis', desc: 'Build the lexer', tasks: [
				['Design token types', 'Define tokens: keywords (if, while, fn), operators (+, -, ==), literals (123, "string"), identifiers, punctuation. Create enum/union for token types with associated data.'],
				['Implement lexer', 'Read source character by character. Match patterns: digits→number, letters→identifier/keyword, quotes→string. Return token stream with position info for errors.'],
				['Handle whitespace and comments', 'Skip spaces, tabs, newlines (track line numbers). Handle // line comments and /* block comments */. Decide if newlines are significant (like Python).'],
				['Add error reporting', 'On invalid character, report: "Error at line 5, column 12: unexpected character \'@\'". Store source location in tokens for later error messages in parser.'],
				['Write lexer tests', 'Test each token type: numbers (42, 3.14), strings ("hello"), operators (==, !=), keywords. Test edge cases: empty input, unclosed strings, nested comments.'],
			]},
			{ name: 'Parsing', desc: 'Build the parser', tasks: [
				['Define grammar', 'Write formal grammar in BNF/EBNF. Example: expr → term (("+"|"-") term)*. Document precedence: multiplicative > additive > comparison > logical.'],
				['Implement recursive descent parser', 'One function per grammar rule. parseExpr() calls parseTerm() in a loop. Consume tokens with expect(TOKEN_PLUS). Return AST nodes.'],
				['Build AST nodes', 'Design node types: BinaryExpr(left, op, right), IfStmt(cond, then, else), FnDecl(name, params, body). Use tagged union or class hierarchy.'],
				['Handle operator precedence', 'Pratt parser or precedence climbing. Higher precedence binds tighter: 1+2*3 → 1+(2*3). Handle right-associative ops like assignment: a=b=c → a=(b=c).'],
				['Add parser error recovery', 'On error, skip to synchronization point (semicolon, closing brace). Report error, continue parsing. Collect multiple errors before stopping.'],
			]},
			{ name: 'Code Generation', desc: 'Generate output', tasks: [
				['Design IR or output format', 'Choose target: bytecode (custom VM), LLVM IR, x86 assembly, JavaScript/C transpilation. Consider: debugging support, optimization opportunities.'],
				['Implement code generator', 'Visitor pattern over AST. generate(BinaryExpr) emits: generate(left), generate(right), emit(ADD). Track registers or stack positions.'],
				['Handle control flow', 'If: generate condition, jump-if-false to else label, generate then, jump to end, else label, generate else. While: loop label, condition, jump-if-false to end, body, jump to loop.'],
				['Implement functions', 'Prologue: save frame pointer, allocate locals. Call: push args, call instruction, pop return value. Return: restore frame, return instruction. Handle calling conventions.'],
				['Add optimizations', 'Constant folding: 2+3 → 5 at compile time. Dead code elimination. Common subexpression elimination. Strength reduction: x*2 → x<<1.'],
			]},
		]);
	} else if (path.name.includes('Deep Learning') || path.name.includes('ML') || path.name.includes('Transformer') || path.name.includes('AI')) {
		addTasks(path.id, [
			{ name: 'Foundations', desc: 'Core concepts', tasks: [
				['Understand neural network basics', 'Layers transform inputs via weights+bias+activation. Activations: ReLU(x)=max(0,x), sigmoid, tanh. Backprop: chain rule computes gradients layer by layer.'],
				['Implement gradient descent', 'Compute loss gradient w.r.t. parameters. Update: w = w - lr * gradient. Variants: SGD with momentum, Adam (adaptive learning rates). Implement from scratch in NumPy.'],
				['Build simple neural network', 'Forward: h = relu(x @ W1 + b1), y = h @ W2 + b2. Backward: compute dL/dW2, dL/dW1 using chain rule. Update weights. Test on XOR or MNIST.'],
				['Learn PyTorch/TensorFlow basics', 'Tensors, autograd, nn.Module. Define model as class, forward() method. Loss functions, optimizers. Training loop: zero_grad, forward, loss, backward, step.'],
				['Understand loss functions', 'MSE for regression: (y-ŷ)². Cross-entropy for classification: -Σ y*log(ŷ). BCE for binary. Softmax + cross-entropy for multi-class. Label smoothing for regularization.'],
			]},
			{ name: 'Architecture', desc: 'Model architectures', tasks: [
				['Implement attention mechanism', 'Attention(Q,K,V) = softmax(QK^T/√d)V. Q=query (what I want), K=keys (what\'s available), V=values (what I return). Scaled dot-product prevents gradient issues.'],
				['Build transformer block', 'MultiHeadAttention → Add&Norm → FFN → Add&Norm. FFN: two linear layers with GELU. Residual connections enable gradient flow through deep networks.'],
				['Add positional encoding', 'Attention is permutation-invariant, needs position info. Sinusoidal: PE(pos,2i)=sin(pos/10000^(2i/d)). Or learned embeddings. RoPE for modern models.'],
				['Implement layer normalization', 'Normalize across features: (x-μ)/σ * γ + β. Stabilizes training, enables higher learning rates. Pre-LN (before sublayer) more stable than post-LN.'],
				['Build full model', 'Stack N transformer blocks (6-96 depending on scale). Add embedding layer, final linear head. GPT: decoder-only. BERT: encoder-only. T5: encoder-decoder.'],
			]},
			{ name: 'Training & Deployment', desc: 'Production ML', tasks: [
				['Set up training loop', 'for epoch: for batch: forward, loss, backward, step. Log metrics to wandb/tensorboard. Evaluate on validation set. Early stopping if no improvement.'],
				['Implement learning rate scheduling', 'Warmup: linearly increase LR over first N steps (e.g., 2000). Then decay: cosine annealing, linear decay, or constant. Prevents early training instability.'],
				['Add model checkpointing', 'Save: torch.save({model_state, optimizer_state, epoch, loss}, path). Load to resume. Save best model based on validation metric. Keep last N checkpoints.'],
				['Build inference pipeline', 'Load model, set eval mode (disables dropout). Batch inputs for efficiency. Use torch.no_grad() to save memory. Implement KV-cache for autoregressive generation.'],
				['Deploy model', 'Export to ONNX or TorchScript. Serve with FastAPI/Flask. Use TensorRT/vLLM for optimization. Handle batching, load balancing. Monitor latency and throughput.'],
			]},
		]);
	} else if (path.name.includes('Redis') || path.name.includes('SQLite') || path.name.includes('LSM') || path.name.includes('Key-Value')) {
		addTasks(path.id, [
			{ name: 'Data Structures', desc: 'Core structures', tasks: [
				['Implement hash table', 'Open addressing or chaining for collisions. Resize when load factor > 0.75. Hash function: murmur3 or xxhash. O(1) average GET/SET. Handle key comparison for strings.'],
				['Build skip list or B-tree', 'Skip list: probabilistic layers, O(log n) search. B-tree: balanced, good for disk. For LSM: memtable (skip list) + sorted string tables (SST) on disk.'],
				['Add expiration support', 'Store expiry timestamp with keys. Passive expiry: check on access. Active expiry: background thread scans for expired keys. Lazy + periodic combination.'],
				['Implement persistence', 'RDB: periodic full snapshots (fork + serialize). AOF: append every write command. On restart, replay AOF or load RDB. Configurable fsync policy.'],
				['Build WAL', 'Write-ahead log: write to log before applying to memory. On crash, replay log. Ensures durability. Truncate log after checkpoint. Group commit for performance.'],
			]},
			{ name: 'Protocol & API', desc: 'Client interface', tasks: [
				['Design wire protocol', 'RESP format: *3\\r\\n$3\\r\\nSET\\r\\n$3\\r\\nkey\\r\\n$5\\r\\nvalue\\r\\n. Or custom binary: length-prefixed messages. Consider: simplicity, parseability, extensibility.'],
				['Implement parser', 'State machine: read type byte, parse based on type. Handle partial reads (TCP framing). Buffer incomplete messages. Return parsed command with arguments.'],
				['Add command handlers', 'Map command names to handler functions. GET(key): lookup in hash table. SET(key, value): insert. INCR: atomic increment. LPUSH/RPUSH for lists.'],
				['Build client library', 'Connect, send commands, parse responses. Connection pooling for concurrency. Retry with backoff. Convenient API: client.set("key", "value"), client.get("key").'],
				['Add pipelining', 'Send multiple commands without waiting for responses. Read all responses after. Reduces round-trip latency. Client queues commands, server processes in order.'],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement transactions', 'MULTI: start queueing. Commands queued (not executed). EXEC: execute all atomically. WATCH: optimistic locking, abort if watched key changed. DISCARD: cancel.'],
				['Add pub/sub', 'SUBSCRIBE channel: client receives messages. PUBLISH channel message: send to all subscribers. Pattern subscriptions: PSUBSCRIBE news.*. Separate connection for pub/sub.'],
				['Build replication', 'Replica connects to primary, requests full sync (RDB). Then streams commands. Replica applies commands to stay in sync. Promote replica if primary fails.'],
				['Add clustering', 'Partition keyspace into 16384 slots. Hash slot = CRC16(key) % 16384. Each node owns slots. Client redirected: MOVED 3999 127.0.0.1:6380. Resharding moves slots.'],
				['Implement monitoring', 'Track: ops/sec, memory usage, connected clients, keyspace hits/misses. INFO command returns stats. Slow log for long-running commands. CLIENT LIST for connections.'],
			]},
		]);
	} else if (path.name.includes('Packet') || path.name.includes('DNS') || path.name.includes('Load Balancer') || path.name.includes('HTTP Server') || path.name.includes('TLS')) {
		addTasks(path.id, [
			{ name: 'Network Basics', desc: 'Foundation', tasks: [
				['Understand the protocol', 'Read the RFC (e.g., RFC 1035 for DNS, RFC 7230 for HTTP). Understand message format, headers, status codes. Use Wireshark to observe real traffic.'],
				['Set up raw sockets or library', 'Raw sockets for packet-level control (requires root). Or use library: net.Listen("tcp", ":80") in Go. Understand TCP vs UDP, blocking vs non-blocking.'],
				['Implement packet parsing', 'Read bytes, decode headers. DNS: 12-byte header, then questions/answers. HTTP: parse request line, headers (until \\r\\n\\r\\n), body. Handle variable-length fields.'],
				['Build packet construction', 'Serialize to bytes for sending. Pack header fields in correct order and endianness (network byte order = big endian). Calculate checksums if required.'],
				['Add error handling', 'Validate input: correct lengths, valid values. Handle partial reads, connection resets, timeouts. Return appropriate error responses (HTTP 400, DNS FORMERR).'],
			]},
			{ name: 'Core Implementation', desc: 'Main functionality', tasks: [
				['Implement server/client', 'Server: bind port, accept connections, spawn goroutine per client. Client: connect, send request, read response. Handle connection lifecycle properly.'],
				['Add request handling', 'Parse request, route to handler based on path/query type. Extract parameters. Validate input. Call business logic. DNS: lookup records. HTTP: serve files or API.'],
				['Build response generation', 'Construct valid response: status, headers, body. Set Content-Length or use chunked encoding. DNS: copy query ID, set response flag, add answer records.'],
				['Implement connection management', 'Limit max concurrent connections. Timeouts: read, write, idle. Keep-alive for HTTP. Connection pooling for client. Graceful shutdown: stop accepting, finish in-flight.'],
				['Add logging', 'Log: timestamp, client IP, request summary, response code, duration. Structured logging (JSON) for parsing. Log levels: debug, info, warn, error. Correlation IDs.'],
			]},
			{ name: 'Advanced Features', desc: 'Production ready', tasks: [
				['Add TLS support', 'Use tls.Listen() or wrap connection. Load certificate and key. Support SNI for multiple domains. Configure cipher suites and TLS versions. Let\'s Encrypt for auto-renewal.'],
				['Implement caching', 'Cache responses by key (URL, query). Respect Cache-Control headers. TTL expiration. LRU eviction when full. Conditional requests: If-None-Match, 304 Not Modified.'],
				['Build health checks', 'Endpoint: GET /health returns 200 if healthy. Check dependencies (database, cache). Kubernetes: liveness (restart if unhealthy) and readiness (remove from LB) probes.'],
				['Add configuration', 'Config file (YAML, TOML) or environment variables. Settings: port, timeouts, TLS paths, log level. Reload config without restart (SIGHUP). Validate config on load.'],
				['Performance testing', 'Benchmark with wrk, hey, or ab: wrk -t4 -c100 -d30s http://localhost:8080. Profile with pprof. Optimize hot paths. Target: requests/sec, latency p50/p99.'],
			]},
		]);
	} else if (path.name.includes('Ray Tracer')) {
		addTasks(path.id, [
			{ name: 'Ray Tracing Basics', desc: 'Core algorithm', tasks: [
				['Implement ray-sphere intersection', 'Solve quadratic: |o + td - c|² = r². Discriminant < 0: miss. Else: t = (-b ± sqrt(discriminant)) / 2a. Return closest positive t and compute normal.'],
				['Build camera model', 'Define: position, look-at, up vector, FOV. For each pixel, compute ray direction from camera through pixel on virtual screen. Aspect ratio correction.'],
				['Add basic shading', 'Diffuse (Lambertian): color * max(0, N·L) where N=normal, L=light direction. Attenuate by distance squared. Accumulate contribution from all lights.'],
				['Implement shadows', 'Before adding light contribution, cast shadow ray from hit point toward light. If it hits any object before reaching light, point is in shadow for that light.'],
				['Add reflections', 'Compute reflection direction: R = D - 2(D·N)N. Cast new ray from hit point in R direction. Recursively trace, blend result with surface color. Limit recursion depth.'],
			]},
			{ name: 'Materials & Lighting', desc: 'Realistic rendering', tasks: [
				['Implement materials', 'Diffuse: matte surfaces. Specular: mirror-like reflection. Glossy: blurred reflections. Dielectric: glass with refraction (Snell\'s law). Fresnel: angle-dependent reflection.'],
				['Add multiple light sources', 'Point lights: position, color, intensity. Directional: direction, color (sun). Area lights: sample multiple points for soft shadows. Sum contributions from all lights.'],
				['Build BVH acceleration', 'Bounding Volume Hierarchy: recursively split objects by spatial extent. Each node has AABB bounding box. Traverse tree: if ray misses box, skip subtree. O(log n) vs O(n) intersections.'],
				['Implement textures', 'UV mapping: sphere uses spherical coordinates. Cube uses face + local coords. Load image, sample at UV. Bilinear filtering for smooth results. Normal maps for surface detail.'],
				['Add anti-aliasing', 'Supersampling: cast multiple rays per pixel (4x4 grid or random jitter). Average colors. Reduces jaggies on edges. Stratified sampling for better distribution.'],
			]},
		]);
	} else if (path.name.includes('Terraform') || path.name.includes('DevOps')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Basics', desc: 'Core concepts', tasks: [
				['Understand IaC principles', 'Declare desired state in code. Version control infrastructure changes. Review in PR. Reproducible environments. Drift detection: compare actual vs desired state.'],
				['Design resource model', 'Resource has: type (aws_instance), name, attributes (ami, instance_type). State file tracks: resource ID, attributes, dependencies. JSON or custom format.'],
				['Implement provider interface', 'Provider implements CRUD: Create(config) → id, Read(id) → config, Update(id, config), Delete(id). Wraps cloud API. Returns errors for user feedback.'],
				['Build plan/apply workflow', 'Plan: diff desired (config) vs actual (state) → changes needed. Show: + create, ~ update, - destroy. Apply: execute plan, update state. Require approval for destructive changes.'],
				['Add state management', 'Store resource IDs and attributes. Lock during operations (prevent concurrent apply). Refresh: read actual state from cloud, update state file. Detect drift.'],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement modules', 'Reusable, parameterized resource groups. module "vpc" { source = "./modules/vpc", cidr = "10.0.0.0/16" }. Encapsulate complexity. Publish to registry.'],
				['Add variables and outputs', 'variable "region" { default = "us-east-1" }. Reference: var.region. Outputs export values: output "ip" { value = aws_instance.main.public_ip }. Pass between modules.'],
				['Build dependency graph', 'Parse references: aws_instance.main.id creates dependency. Topological sort for execution order. Create independent resources in parallel. Handle circular dependency errors.'],
				['Implement import', 'Adopt existing resources: import aws_instance.main i-1234567890. Read attributes from cloud, write to state. Generate config suggestion. Bring unmanaged resources under IaC.'],
				['Add remote state', 'Store state in S3/GCS/Terraform Cloud. Locking with DynamoDB/similar prevents concurrent writes. Team shares state. Encrypt at rest. Backend configuration block.'],
			]},
		]);
	} else if (path.name.includes('Password Manager')) {
		addTasks(path.id, [
			{ name: 'Cryptography', desc: 'Security foundation', tasks: [
				['Implement key derivation', 'Derive encryption key from master password using Argon2id (memory-hard, resists GPU attacks) or PBKDF2-SHA256 (100k+ iterations). Salt per vault. Output 256-bit key.'],
				['Add encryption', 'AES-256-GCM: authenticated encryption. Encrypt each entry with unique nonce. GCM provides integrity (detects tampering). Never reuse nonce with same key.'],
				['Build secure storage', 'Encrypted vault file: header (salt, params) + encrypted entries. Decrypt to memory on unlock, clear on lock. Protect against memory dumps if possible.'],
				['Implement master password', 'Single password unlocks vault. Verify by decrypting test block or using MAC. Lock after timeout. Clear password from memory after key derivation.'],
				['Add password generation', 'Cryptographically random: crypto/rand or secrets module. Configurable: length (16+), character sets (upper, lower, digits, symbols). Pronounceable option. Check against common passwords.'],
			]},
			{ name: 'User Interface', desc: 'Usability', tasks: [
				['Build CLI interface', 'Commands: init, unlock, add, get, list, edit, delete, lock. Prompt for master password securely (no echo). Tab completion for entry names. JSON output option.'],
				['Add search and filter', 'Search by: title, username, URL, tags. Fuzzy matching for typos. Filter by category/folder. Sort by last used, name, date added. Quick access to favorites.'],
				['Implement clipboard integration', 'Copy password to clipboard: auto-clear after 30 seconds. On macOS: pbcopy. Linux: xclip. Windows: clip.exe. Warn if clipboard manager might store history.'],
				['Add import/export', 'Import from: CSV, 1Password, LastPass, Bitwarden exports. Export: encrypted backup or CSV (warn about security). Include all fields: title, username, password, URL, notes.'],
				['Build browser extension', 'Detect login forms (input type=password). Match URL to entries. Auto-fill credentials. Communicate with main app via native messaging. Secure against XSS.'],
			]},
		]);
	} else if (path.name.includes('Hacking') || path.name.includes('Security') || path.name.includes('Red Team') || path.name.includes('CTF')) {
		addTasks(path.id, [
			{ name: 'Reconnaissance', desc: 'Information gathering', tasks: [
				['Learn network scanning', 'Nmap: -sn (host discovery), -sS (SYN scan), -sV (version detection). Identify open ports, running services, OS fingerprinting. Scan ranges: nmap 192.168.1.0/24.'],
				['Understand enumeration', 'Extract details from services: SMB shares (smbclient -L), SNMP (snmpwalk), DNS zone transfers (dig axfr), LDAP queries. Banner grabbing for versions.'],
				['Practice OSINT', 'theHarvester for emails, subdomains. Shodan for internet-connected devices. LinkedIn for employee info. GitHub for leaked credentials, API keys in code.'],
				['Study web recon', 'Subdomain enum: subfinder, amass. Directory brute: gobuster, feroxbuster. Tech detection: wappalyzer, whatweb. Check robots.txt, .git exposure, backup files.'],
				['Build target profile', 'Compile: IP ranges, subdomains, technologies, employees, email format, exposed services. Identify attack surface. Prioritize targets by likelihood of success.'],
			]},
			{ name: 'Exploitation', desc: 'Attack techniques', tasks: [
				['Learn common vulnerabilities', 'OWASP Top 10: injection, broken auth, XSS, insecure deserialization, SSRF. Understand root cause, impact, and remediation for each.'],
				['Practice privilege escalation', 'Linux: SUID binaries, sudo misconfig, cron jobs, kernel exploits. Windows: service misconfig, unquoted paths, token impersonation (Potato), AlwaysInstallElevated.'],
				['Understand Active Directory attacks', 'Kerberoasting, AS-REP roasting, pass-the-hash, DCSync, golden/silver tickets, delegation abuse, NTLM relay. BloodHound for attack path mapping.'],
				['Study web exploitation', 'XSS: steal cookies, keylog. SQLi: union-based extraction, blind time-based. SSRF: access internal services, cloud metadata. Chaining vulns for impact.'],
				['Practice post-exploitation', 'Persistence: scheduled tasks, registry, services. Credential harvesting: mimikatz, browser passwords. Pivoting: SSH tunnels, chisel, SOCKS proxy to internal network.'],
			]},
			{ name: 'Tooling', desc: 'Build your own', tasks: [
				['Build a scanner', 'Concurrent port scanner in Go/Python. Service detection with banner grabbing. Output: open ports, service versions. Add vulnerability checks for common issues.'],
				['Create an exploit', 'Pick a CVE with public PoC. Understand the vulnerability. Adapt exploit for target. Add reliability: check version, handle errors. Document usage.'],
				['Develop a payload', 'Reverse shell: connect back, spawn shell, redirect I/O. Stager: small first stage downloads larger payload. Obfuscation to evade basic detection.'],
				['Write a parser', 'Parse Nmap XML, Burp logs, or BloodHound JSON. Extract actionable data: high-value targets, credentials, attack paths. Output to database or report.'],
				['Automate workflow', 'Chain tools: recon → enumeration → exploitation. Handle errors, log results. Parallel execution where possible. Config-driven target specification.'],
			]},
		]);
	} else if (path.name.includes('Homelab')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Setup', desc: 'Build foundation', tasks: [
				['Set up hypervisor', 'Proxmox VE (free, KVM-based) or ESXi. Allocate CPU/RAM for VMs. Set up storage: local SSD, NFS share, or Ceph for clustering. Template VMs for quick deployment.'],
				['Configure networking', 'Create VLANs: management (10), servers (20), users (30), IoT (40), security lab (100). pfSense/OPNsense firewall: inter-VLAN rules, NAT, VPN. Document IP scheme.'],
				['Deploy domain controller', 'Windows Server with AD DS role. Create domain: lab.local. Add DNS, DHCP. Create OUs, users, groups. GPOs for security settings. Second DC for redundancy.'],
				['Add monitoring', 'Prometheus: scrape metrics from node_exporter, windows_exporter. Grafana dashboards: CPU, memory, disk, network. Alerts: Alertmanager → email/Slack when thresholds exceeded.'],
				['Set up logging', 'ELK (Elasticsearch, Logstash, Kibana) or Loki+Grafana. Collect: Windows Event Logs (Winlogbeat), syslog (rsyslog), application logs. Dashboards for security events, search interface.'],
			]},
			{ name: 'Security Lab', desc: 'Practice environment', tasks: [
				['Deploy vulnerable machines', 'Metasploitable 2/3, DVWA, VulnHub boxes, HackTheBox/TryHackMe VPN. Isolated VLAN. Document intentional vulnerabilities. Rotate machines for variety.'],
				['Set up attack machine', 'Kali or Parrot Linux VM. Install additional tools: BloodHound, CrackMapExec, Sliver. Access to vulnerable VLAN. Separate from production network.'],
				['Configure C2 framework', 'Sliver or Mythic: generate implants, set up listeners (HTTP, HTTPS, DNS). Practice: deploy implant to vulnerable VM, execute commands, lateral movement. Understand C2 traffic.'],
				['Add detection tools', 'Elastic Security (SIEM): ingest Windows/Linux logs, detect attacks. Wazuh: HIDS, file integrity. Suricata: NIDS. Create detection rules, tune false positives.'],
				['Practice attack/defense', 'Purple team exercises: attack from Kali, detect in SIEM. Document: attack technique, logs generated, detection logic. Build playbooks. Atomic Red Team for tests.'],
			]},
		]);
	} else {
		// Generic development tasks
		addTasks(path.id, [
			{ name: 'Foundation', desc: 'Core implementation', tasks: [
				['Research the domain', 'Read documentation, RFCs, existing implementations. Understand: problem being solved, constraints, typical approaches. List requirements: functional and non-functional.'],
				['Design architecture', 'Identify components and responsibilities. Define interfaces between components. Choose data structures and algorithms. Consider: scalability, maintainability, testability.'],
				['Set up project structure', 'Initialize repo, package manager, build system. Organize: src/, tests/, docs/. Configure linting, formatting. CI pipeline for automated checks. README with setup instructions.'],
				['Implement core logic', 'Build minimum viable functionality first. Focus on correctness, then performance. Iterate: implement, test, refine. Keep functions small and focused.'],
				['Add error handling', 'Define error types for different failure modes. Return errors, don\'t panic/throw unexpectedly. Log errors with context. Provide actionable error messages to users.'],
			]},
			{ name: 'Features', desc: 'Build out functionality', tasks: [
				['Implement main features', 'Prioritize by user value. Build incrementally: basic version, then enhance. Each feature: design, implement, test, document. Keep scope manageable.'],
				['Add configuration', 'Config file (YAML, TOML, JSON) and/or environment variables. Sensible defaults. Validate config on load. Document all options. Support config reload if applicable.'],
				['Build CLI or API', 'CLI: use argparse/clap/cobra for argument parsing. API: REST or gRPC, versioned endpoints. Consistent interface design. Help text and examples.'],
				['Add logging', 'Structured logging (JSON) with levels: debug, info, warn, error. Include: timestamp, component, message, relevant data. Configure output destination and level.'],
				['Write tests', 'Unit tests for functions, integration tests for components. Cover: happy path, edge cases, error conditions. Aim for high coverage of critical paths. Run in CI.'],
			]},
			{ name: 'Polish', desc: 'Production ready', tasks: [
				['Optimize performance', 'Profile to find bottlenecks: CPU (pprof), memory. Optimize hot paths. Benchmark before/after changes. Balance: performance vs code clarity vs development time.'],
				['Add documentation', 'README: what, why, quick start. API docs: all public functions/endpoints. Architecture doc for contributors. Examples and tutorials for common use cases.'],
				['Handle edge cases', 'Empty input, very large input, malformed data, concurrent access, resource exhaustion. Add validation and graceful degradation. Document limitations.'],
				['Package for distribution', 'Build scripts for all platforms. Installers: brew, apt, npm, pip. Docker image. Release process: version bump, changelog, tag, publish. Signed binaries if applicable.'],
				['Final testing', 'End-to-end tests simulating real usage. Test on clean machine. Verify documentation accuracy. Security review. Performance benchmarks. User acceptance testing.'],
			]},
		]);
	}
}

console.log('Done adding tasks to empty paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
