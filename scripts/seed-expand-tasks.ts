import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

// Helper to add tasks to a path
function addModuleWithTasks(pathId: number, moduleName: string, moduleDesc: string, orderIndex: number, tasks: [string, string, string][]) {
	const mod = insertModule.run(pathId, moduleName, moduleDesc, orderIndex, now);
	tasks.forEach(([title, desc, details], i) => {
		insertTask.run(mod.lastInsertRowid, title, desc, details, i, now);
	});
	return mod.lastInsertRowid;
}

// Get paths that need more tasks
const pathsNeedingTasks = db.prepare(`
	SELECT p.id, p.name FROM paths p
	LEFT JOIN modules m ON m.path_id = p.id
	LEFT JOIN tasks t ON t.module_id = m.id
	WHERE p.schedule IS NOT NULL
	GROUP BY p.id
	HAVING COUNT(t.id) < 10
`).all() as { id: number; name: string }[];

console.log(`Found ${pathsNeedingTasks.length} paths needing more tasks`);

// Delete existing modules/tasks for these paths (cascade will delete tasks)
const deleteModules = db.prepare('DELETE FROM modules WHERE path_id = ?');

for (const path of pathsNeedingTasks) {
	console.log(`Expanding: ${path.name}`);
	deleteModules.run(path.id);
}

// ============================================================================
// Reimplement: Chisel (path 60)
// ============================================================================
const chiselPath = pathsNeedingTasks.find(p => p.name.includes('Chisel'));
if (chiselPath) {
	addModuleWithTasks(chiselPath.id, 'Week 1-2: TCP Tunneling Core', 'Build the foundation of TCP forwarding', 0, [
		['Implement basic TCP listener', 'Create a TCP server that accepts connections', `## TCP Listener Foundation

### Goals
- Create TCP listener on configurable port
- Handle multiple concurrent connections
- Implement graceful shutdown

### Implementation
\`\`\`go
listener, err := net.Listen("tcp", ":8080")
// Accept loop with goroutines
\`\`\`

### Testing
- Test with netcat: \`nc localhost 8080\`
- Verify concurrent connections work`],
		['Build TCP client connector', 'Connect to remote TCP endpoints', `## TCP Client

### Implementation
- Dial remote endpoints
- Handle connection timeouts
- Implement retry logic with backoff`],
		['Implement bidirectional data forwarding', 'Forward data between connections', `## Data Forwarding

Use io.Copy in goroutines for bidirectional streaming:
\`\`\`go
go io.Copy(dst, src)
go io.Copy(src, dst)
\`\`\``],
		['Add connection multiplexing', 'Handle multiple tunneled connections', `## Multiplexing

### Protocol Design
- Frame format: [length][channel_id][data]
- Channel management
- Flow control basics`],
		['Implement port forwarding mode', 'Forward local port to remote', `## Local Port Forward

chisel client --local 8080:internal:80

Maps localhost:8080 -> internal:80 through tunnel`],
		['Add reverse port forwarding', 'Expose remote port locally', `## Reverse Forward

chisel client --reverse 8080:localhost:80

Expose client's port 80 on server's 8080`],
	]);

	addModuleWithTasks(chiselPath.id, 'Week 3-4: HTTP/WebSocket Transport', 'Tunnel over HTTP for firewall bypass', 1, [
		['Implement HTTP CONNECT tunnel', 'Use HTTP CONNECT for tunneling', `## HTTP CONNECT

Standard proxy method:
\`\`\`
CONNECT target:port HTTP/1.1
Host: target:port
\`\`\``],
		['Build WebSocket transport layer', 'Upgrade HTTP to WebSocket', `## WebSocket Tunnel

### Why WebSocket?
- Bidirectional communication
- Works through proxies
- Looks like normal web traffic`],
		['Add HTTP request/response framing', 'Encapsulate tunnel data in HTTP', `## HTTP Framing

Encode binary data in HTTP bodies
Handle chunked transfer encoding`],
		['Implement connection keepalive', 'Maintain persistent connections', `## Keepalive

- Ping/pong frames
- Reconnection logic
- Session resumption`],
		['Add TLS support', 'Secure the tunnel transport', `## TLS Configuration

- Generate/load certificates
- Verify server certificates
- Support custom CA`],
		['Build proxy authentication', 'Support authenticated proxies', `## Proxy Auth

- Basic authentication
- NTLM support for corporate proxies`],
	]);

	addModuleWithTasks(chiselPath.id, 'Week 5-6: SOCKS5 & Polish', 'Add SOCKS proxy and production features', 2, [
		['Implement SOCKS5 server', 'Full SOCKS5 proxy protocol', `## SOCKS5 Protocol

### Handshake
1. Client sends supported auth methods
2. Server selects method
3. Authentication (if required)
4. Connection request

### Commands
- CONNECT
- BIND
- UDP ASSOCIATE`],
		['Add SOCKS5 authentication', 'Username/password auth', `## SOCKS5 Auth

Method 0x02: Username/Password
\`\`\`
[version][ulen][username][plen][password]
\`\`\``],
		['Implement UDP relay', 'SOCKS5 UDP associate', `## UDP Relay

For DNS and other UDP protocols through SOCKS`],
		['Add fingerprint resistance', 'Evade traffic analysis', `## Evasion

- Randomize packet sizes
- Add jitter to timing
- Mimic browser TLS fingerprint`],
		['Build CLI interface', 'User-friendly command line', `## CLI Design

\`\`\`
chisel server --port 8080 --auth user:pass
chisel client server:8080 --socks 1080
\`\`\``],
		['Add logging and metrics', 'Observability features', `## Monitoring

- Connection statistics
- Bandwidth tracking
- Error logging`],
	]);
}

// ============================================================================
// Reimplement: Web Proxy / Burp Suite (path 75)
// ============================================================================
const burpPath = pathsNeedingTasks.find(p => p.name.includes('Web Proxy') || p.name.includes('Burp'));
if (burpPath) {
	addModuleWithTasks(burpPath.id, 'Week 1: HTTP Proxy Core', 'MITM proxy foundation', 0, [
		['Build basic HTTP proxy', 'Intercept and forward HTTP requests', `## HTTP Proxy Foundation

### Architecture
\`\`\`
Browser -> Proxy -> Target Server
         (intercept)
\`\`\`

### Implementation
1. Listen on localhost:8080
2. Parse incoming HTTP requests
3. Forward to destination
4. Return response to client`],
		['Implement HTTPS interception', 'Generate certs for MITM', `## HTTPS MITM

### Steps
1. Generate root CA certificate
2. On CONNECT, generate cert for target domain
3. Establish TLS with client using generated cert
4. Establish TLS with server
5. Decrypt, inspect, re-encrypt`],
		['Add request/response parsing', 'Parse HTTP headers and body', `## HTTP Parsing

Parse:
- Request line (method, path, version)
- Headers (handle multi-line)
- Body (Content-Length, chunked)`],
		['Build request modification', 'Edit requests before forwarding', `## Request Editing

Allow modification of:
- URL/path
- Headers
- Body/parameters
- Method`],
		['Implement response modification', 'Edit responses before returning', `## Response Editing

Modify:
- Status code
- Headers
- Body content`],
		['Add WebSocket proxying', 'Handle WebSocket upgrades', `## WebSocket Support

1. Detect Upgrade header
2. Forward handshake
3. Proxy frames bidirectionally`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 2: Interceptor UI', 'Terminal UI for inspection', 1, [
		['Set up TUI framework', 'Use textual or rich for terminal UI', `## TUI Setup

Use Python textual for rich terminal interface:
\`\`\`python
from textual.app import App
from textual.widgets import DataTable, TextArea
\`\`\``],
		['Build request list view', 'Display intercepted requests', `## Request List

Columns:
- ID
- Method
- URL
- Status
- Size
- Time`],
		['Implement request detail view', 'Show full request/response', `## Detail View

Split pane:
- Request headers + body
- Response headers + body
- Hex view for binary`],
		['Add intercept toggle', 'Pause and modify requests', `## Intercept Mode

Toggle interception on/off
Queue requests when intercepted
Edit before forwarding`],
		['Build search and filter', 'Filter request history', `## Filtering

Filter by:
- URL pattern (regex)
- Method
- Status code
- Content type
- Size range`],
		['Add keyboard shortcuts', 'Vim-style navigation', `## Shortcuts

- j/k: navigate list
- Enter: view details
- e: edit request
- f: forward
- d: drop`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 3: History & Repeater', 'Request storage and replay', 2, [
		['Implement SQLite storage', 'Persist request history', `## Database Schema

\`\`\`sql
CREATE TABLE requests (
  id INTEGER PRIMARY KEY,
  method TEXT,
  url TEXT,
  headers TEXT,
  body BLOB,
  response_status INTEGER,
  response_headers TEXT,
  response_body BLOB,
  timestamp INTEGER
);
\`\`\``],
		['Build history browser', 'Browse and search past requests', `## History Features

- Pagination for large histories
- Full-text search
- Date range filtering`],
		['Create Repeater tool', 'Resend modified requests', `## Repeater

1. Select request from history
2. Modify as needed
3. Send and view response
4. Compare with original`],
		['Add response comparison', 'Diff two responses', `## Response Diff

Highlight differences:
- Status code changes
- Header changes
- Body changes (unified diff)`],
		['Implement export formats', 'Export as cURL, Python, etc.', `## Export Formats

- cURL command
- Python requests
- Raw HTTP
- HAR format`],
		['Build session management', 'Save/load proxy sessions', `## Sessions

Save complete state:
- Request history
- Settings
- Scope configuration`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 4: Active Scanner', 'Automated vulnerability scanning', 3, [
		['Build endpoint crawler', 'Discover endpoints from history', `## Crawler

Extract from history:
- All unique URLs
- Form actions
- JavaScript endpoints
- API paths`],
		['Implement payload injection', 'Insert test payloads', `## Injection Points

Identify:
- URL parameters
- POST body parameters
- Headers (Cookie, User-Agent, etc.)
- JSON fields`],
		['Add XSS detection', 'Find cross-site scripting', `## XSS Scanner

### Payloads
- \`<script>alert(1)</script>\`
- \`"><img src=x onerror=alert(1)>\`
- Event handlers

### Detection
Check if payload reflected unencoded`],
		['Implement SQLi detection', 'Find SQL injection', `## SQLi Detection

### Payloads
- Single quote: \`'\`
- Boolean: \`' OR '1'='1\`
- Time-based: \`'; WAITFOR DELAY '0:0:5'--\`

### Detection
- Error messages
- Response differences
- Time delays`],
		['Add SSRF/LFI checks', 'Server-side vulnerabilities', `## SSRF Detection

### Payloads
- \`http://localhost/admin\`
- \`http://169.254.169.254/\`
- \`file:///etc/passwd\`

### Out-of-band
Use callback server to detect blind SSRF`],
		['Build scan reports', 'Generate vulnerability reports', `## Reports

Include:
- Executive summary
- Vulnerability details
- Risk ratings
- Remediation advice
- Request/response evidence`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 5: Intruder', 'Automated attack tool', 4, [
		['Build payload position marker', 'Mark injection points', `## Position Marking

Use markers like §param§ to identify injection points
Support multiple positions per request`],
		['Implement Sniper attack', 'Single position, multiple payloads', `## Sniper Mode

One position at a time
Iterate through payload list
Good for: fuzzing parameters`],
		['Add Battering Ram attack', 'Same payload everywhere', `## Battering Ram

Same payload in all positions simultaneously
Good for: username=§x§&password=§x§`],
		['Build Pitchfork attack', 'Parallel payload lists', `## Pitchfork

Multiple lists, iterate in parallel:
- usernames[0] + passwords[0]
- usernames[1] + passwords[1]
Good for: credential stuffing`],
		['Implement Cluster Bomb', 'All combinations', `## Cluster Bomb

All combinations of all payload lists
usernames × passwords
Good for: brute force`],
		['Add result analysis', 'Identify interesting responses', `## Analysis

Flag responses with:
- Different status code
- Different length
- Matching pattern
- Different response time`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 6: Extensions & Polish', 'Plugin system and utilities', 5, [
		['Design plugin architecture', 'Extensible proxy system', `## Plugin System

Hooks:
- onRequest(req) -> modified req
- onResponse(resp) -> modified resp
- onScanResult(vuln) -> custom handling`],
		['Build Decoder utility', 'Encode/decode transformations', `## Decoder

Transformations:
- Base64 encode/decode
- URL encode/decode
- HTML entities
- Hex encode/decode
- Gzip compress/decompress`],
		['Create Comparer tool', 'Visual diff utility', `## Comparer

Side-by-side comparison
Highlight differences
Support binary comparison`],
		['Add Sequencer', 'Token randomness analysis', `## Sequencer

Analyze session tokens:
- Collect samples
- Statistical analysis
- Entropy calculation
- Identify patterns`],
		['Implement scope configuration', 'Target scope management', `## Scope

Define target scope:
- Include patterns
- Exclude patterns
- Only scan in-scope items`],
		['Final polish and documentation', 'Error handling and docs', `## Polish

- Comprehensive error handling
- User documentation
- Example workflows
- Performance optimization`],
	]);
}

// ============================================================================
// Reimplement: Metasploit Framework (path 77)
// ============================================================================
const msfPath = pathsNeedingTasks.find(p => p.name.includes('Metasploit'));
if (msfPath) {
	addModuleWithTasks(msfPath.id, 'Week 1-2: Core Framework', 'Module system and architecture', 0, [
		['Design module architecture', 'Plugin-based exploit system', `## Module System

### Module Types
- Exploits: deliver payloads
- Auxiliary: scanning, fuzzing
- Post: post-exploitation
- Payloads: shellcode/agents
- Encoders: payload encoding
- Nops: NOP sleds`],
		['Implement module loader', 'Dynamic module loading', `## Module Loading

\`\`\`ruby
# modules/exploits/windows/smb/ms17_010.rb
class MetasploitModule < Msf::Exploit::Remote
  def exploit
    # exploitation logic
  end
end
\`\`\``],
		['Build datastore system', 'Module options/configuration', `## Datastore

Options:
- RHOSTS: target hosts
- RPORT: target port
- PAYLOAD: selected payload
- LHOST/LPORT: listener`],
		['Create module mixins', 'Shared functionality', `## Mixins

- Msf::Exploit::Remote::Tcp
- Msf::Exploit::Remote::HttpClient
- Msf::Exploit::Remote::SMB
- Msf::Auxiliary::Scanner`],
		['Implement target handling', 'Multi-target support', `## Targets

Define multiple targets:
- OS versions
- Software versions
- Architecture (x86, x64)
- Each with specific offsets`],
		['Add payload compatibility', 'Match payloads to exploits', `## Payload Compat

Filter payloads by:
- Platform (windows, linux)
- Architecture
- Connection type (reverse, bind)
- Size constraints`],
	]);

	addModuleWithTasks(msfPath.id, 'Week 3-4: Payload System', 'Shellcode and staged payloads', 1, [
		['Build payload generator', 'Generate raw shellcode', `## Payload Generation

Assemble shellcode from stagers + stages:
1. Stager: small, establishes connection
2. Stage: full agent, loaded by stager`],
		['Implement stagers', 'Initial connection code', `## Stagers

reverse_tcp: connect back to attacker
bind_tcp: listen for connection
reverse_http: HTTP-based callback`],
		['Create Meterpreter stage', 'Full-featured agent', `## Meterpreter

Features:
- File system access
- Process management
- Network pivoting
- Keystroke capture
- Screenshot`],
		['Add payload encoding', 'Evade basic detection', `## Encoders

- XOR encoding
- Shikata ga nai (polymorphic)
- Alpha-numeric
- Custom encoders`],
		['Implement payload templates', 'Executable generation', `## Templates

Inject payload into:
- Windows PE
- Linux ELF
- macOS Mach-O
- Script files`],
		['Build handler system', 'Listener management', `## Handlers

\`\`\`
use exploit/multi/handler
set PAYLOAD windows/meterpreter/reverse_tcp
set LHOST 10.0.0.1
exploit -j
\`\`\``],
	]);

	addModuleWithTasks(msfPath.id, 'Week 5-6: Console & Database', 'CLI and session management', 2, [
		['Build interactive console', 'msfconsole interface', `## Console

- Tab completion
- Command history
- Module context switching
- Resource scripts`],
		['Implement session management', 'Track active sessions', `## Sessions

\`\`\`
sessions -l          # list
sessions -i 1        # interact
sessions -k 1        # kill
sessions -u 1        # upgrade shell
\`\`\``],
		['Add database backend', 'Store results in PostgreSQL', `## Database

Tables:
- hosts: discovered hosts
- services: ports/services
- vulns: vulnerabilities
- creds: credentials
- loots: exfiltrated data`],
		['Create workspace support', 'Organize engagements', `## Workspaces

Separate data by engagement:
\`\`\`
workspace -a client1
workspace client1
\`\`\``],
		['Build reporting engine', 'Generate reports', `## Reports

Export formats:
- HTML report
- XML export
- JSON dump
- Custom templates`],
		['Implement RPC API', 'Remote control interface', `## MSFRPC

\`\`\`python
from pymetasploit3.msfrpc import MsfRpcClient
client = MsfRpcClient('password')
exploit = client.modules.use('exploit', 'windows/smb/ms17_010')
\`\`\``],
	]);
}

// ============================================================================
// Reimplement: Hashcat (path 81)
// ============================================================================
const hashcatPath = pathsNeedingTasks.find(p => p.name.includes('Hashcat'));
if (hashcatPath) {
	addModuleWithTasks(hashcatPath.id, 'Week 1-2: Hash Identification & CPU Cracking', 'Foundation of password cracking', 0, [
		['Implement hash identification', 'Auto-detect hash types', `## Hash Detection

Identify by:
- Length (MD5=32, SHA1=40, SHA256=64)
- Character set
- Format markers ($1$, $6$, $2a$)
- Context clues`],
		['Build MD5 cracker', 'Basic hash cracking', `## MD5 Implementation

\`\`\`python
import hashlib
def crack_md5(hash, wordlist):
    for word in wordlist:
        if hashlib.md5(word.encode()).hexdigest() == hash:
            return word
\`\`\``],
		['Add SHA family support', 'SHA1, SHA256, SHA512', `## SHA Support

Same pattern as MD5:
- Read wordlist
- Hash each candidate
- Compare to target`],
		['Implement bcrypt cracking', 'Slow hash support', `## bcrypt

Much slower - requires optimization:
- Cost factor awareness
- Early termination
- Parallel processing critical`],
		['Build dictionary attack', 'Wordlist-based cracking', `## Dictionary Attack

\`\`\`
./cracker -m 0 -a 0 hash.txt wordlist.txt
\`\`\`

Modes: straight, combination, rule-based`],
		['Add rule engine', 'Word mangling rules', `## Rules

Common transformations:
- l: lowercase
- u: uppercase
- c: capitalize
- $1: append 1
- ^!: prepend !
- r: reverse`],
	]);

	addModuleWithTasks(hashcatPath.id, 'Week 3-4: GPU Acceleration', 'CUDA/OpenCL implementation', 1, [
		['Set up CUDA environment', 'GPU programming basics', `## CUDA Setup

- Install CUDA toolkit
- Understand GPU architecture
- Blocks, threads, warps
- Memory hierarchy`],
		['Implement GPU MD5 kernel', 'Parallel hash computation', `## GPU MD5

\`\`\`cuda
__global__ void md5_kernel(char* wordlist, char* hashes, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute MD5 of wordlist[idx]
    // compare against target hashes
}
\`\`\``],
		['Add batch processing', 'Process millions of candidates', `## Batching

- Load wordlist in chunks
- Transfer to GPU memory
- Process in parallel
- Retrieve results`],
		['Implement multiple hash attack', 'Crack many hashes at once', `## Multi-Hash

Compare each candidate against ALL target hashes
Amortizes computation cost
Much faster for large hash lists`],
		['Add OpenCL support', 'Cross-platform GPU', `## OpenCL

Platform-independent:
- AMD GPUs
- Intel GPUs
- NVIDIA GPUs
- CPU fallback`],
		['Optimize memory transfers', 'Minimize GPU<->CPU copies', `## Memory Optimization

- Pinned memory
- Async transfers
- Double buffering
- Stream concurrency`],
	]);

	addModuleWithTasks(hashcatPath.id, 'Week 5-6: Advanced Attacks', 'Masks, combinator, and brain', 2, [
		['Build mask attack', 'Pattern-based brute force', `## Mask Attack

?l = lowercase
?u = uppercase
?d = digit
?s = special

Example: ?u?l?l?l?d?d?d?d
Matches: Pass1234`],
		['Implement combinator attack', 'Combine wordlists', `## Combinator

word1 + word2 from two lists:
admin + 123 = admin123
super + user = superuser`],
		['Add hybrid attacks', 'Wordlist + mask', `## Hybrid

Wordlist + brute force:
- password?d?d?d
- ?d?d?d?dpassword`],
		['Build markov generator', 'Statistical candidate generation', `## Markov Chains

Learn password patterns from leaks
Generate statistically likely candidates
Much faster than brute force`],
		['Implement distributed cracking', 'Multi-machine support', `## Distributed

- Split keyspace across machines
- Central coordinator
- Progress synchronization
- Result aggregation`],
		['Add restore/session support', 'Resume interrupted attacks', `## Sessions

Save state:
- Current position
- Attack configuration
- Found passwords
- Runtime statistics`],
	]);
}

// ============================================================================
// Build Your Own Shell (path 94)
// ============================================================================
const shellPath = pathsNeedingTasks.find(p => p.name === 'Build Your Own Shell');
if (shellPath) {
	addModuleWithTasks(shellPath.id, 'Week 1-2: Core Shell', 'REPL and command execution', 0, [
		['Implement read-eval-print loop', 'Basic shell loop', `## REPL

\`\`\`c
while (1) {
    print_prompt();
    char* line = read_line();
    char** args = parse_line(line);
    execute(args);
}
\`\`\``],
		['Build command parser', 'Tokenize input', `## Parsing

Handle:
- Whitespace splitting
- Quoted strings
- Escape characters
- Special characters`],
		['Implement fork/exec', 'Run external commands', `## Process Execution

\`\`\`c
pid_t pid = fork();
if (pid == 0) {
    execvp(args[0], args);
    exit(EXIT_FAILURE);
}
waitpid(pid, &status, 0);
\`\`\``],
		['Add PATH searching', 'Find executables', `## PATH Lookup

Search directories in PATH:
/usr/local/bin
/usr/bin
/bin`],
		['Build built-in commands', 'cd, exit, export, etc.', `## Builtins

Must be built-in (can't fork):
- cd: change directory
- exit: terminate shell
- export: set env var
- alias: command aliases`],
		['Implement environment variables', 'getenv/setenv', `## Environment

- $HOME, $PATH, $USER
- Custom variables
- Variable expansion: $VAR, \${VAR}`],
	]);

	addModuleWithTasks(shellPath.id, 'Week 3-4: I/O and Pipes', 'Redirection and pipelines', 1, [
		['Add input redirection', '< operator', `## Input Redirect

\`\`\`
command < input.txt
\`\`\`

Open file, dup2 to stdin before exec`],
		['Implement output redirection', '> and >> operators', `## Output Redirect

\`\`\`
command > output.txt   # overwrite
command >> output.txt  # append
\`\`\`

dup2 file to stdout`],
		['Build stderr redirection', '2> operator', `## Stderr

\`\`\`
command 2> errors.txt
command > all.txt 2>&1
\`\`\``],
		['Implement pipes', '| operator', `## Pipes

\`\`\`c
int pipefd[2];
pipe(pipefd);
// fork, dup2 write end to stdout
// fork, dup2 read end to stdin
\`\`\``],
		['Add multiple pipes', 'cmd1 | cmd2 | cmd3', `## Pipeline

Handle arbitrary length pipelines
Connect stdout -> stdin between processes`],
		['Implement here documents', '<< operator', `## Here Doc

\`\`\`
cat << EOF
multiple
lines
EOF
\`\`\``],
	]);

	addModuleWithTasks(shellPath.id, 'Week 5-6: Job Control & Polish', 'Background jobs and signals', 2, [
		['Add background execution', '& operator', `## Background Jobs

\`\`\`
long_command &
\`\`\`

Don't wait for process, print job number`],
		['Implement job listing', 'jobs command', `## Jobs

Track background processes:
[1]+ Running    sleep 100 &
[2]- Running    vim &`],
		['Build fg/bg commands', 'Job control', `## Foreground/Background

fg %1: bring job 1 to foreground
bg %1: continue job 1 in background`],
		['Handle signals', 'SIGINT, SIGTSTP, etc.', `## Signals

- Ctrl+C: SIGINT (interrupt)
- Ctrl+Z: SIGTSTP (suspend)
- Ctrl+D: EOF (exit)

Setup signal handlers properly`],
		['Add command history', 'Arrow keys, history', `## History

- Up/Down arrow navigation
- !n: run command n
- !!: repeat last command
- history: show history`],
		['Implement tab completion', 'Autocomplete paths/commands', `## Completion

Complete:
- Commands from PATH
- File paths
- Variable names`],
	]);
}

// ============================================================================
// Build Your Own SQLite (path 177 or similar)
// ============================================================================
const sqlitePath = pathsNeedingTasks.find(p => p.name.includes('SQLite'));
if (sqlitePath) {
	addModuleWithTasks(sqlitePath.id, 'Week 1-2: SQL Parser', 'Parse SQL statements', 0, [
		['Build SQL tokenizer', 'Lexical analysis', `## Tokenizer

Token types:
- Keywords: SELECT, FROM, WHERE
- Identifiers: table/column names
- Literals: strings, numbers
- Operators: =, <, >, AND, OR`],
		['Implement expression parser', 'Parse WHERE clauses', `## Expression Parsing

Build AST for:
- Comparisons: a = 1
- Boolean: a AND b OR c
- Arithmetic: a + b * c`],
		['Parse SELECT statements', 'Full SELECT support', `## SELECT Parser

\`\`\`sql
SELECT col1, col2
FROM table
WHERE condition
ORDER BY col
LIMIT n
\`\`\``],
		['Parse INSERT/UPDATE/DELETE', 'Data modification', `## DML Parsing

INSERT INTO table (cols) VALUES (vals)
UPDATE table SET col = val WHERE cond
DELETE FROM table WHERE cond`],
		['Add CREATE TABLE parsing', 'DDL statements', `## DDL Parser

\`\`\`sql
CREATE TABLE name (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
)
\`\`\``],
		['Implement query planner', 'Generate execution plan', `## Query Planning

Convert AST to execution plan:
1. Table scan or index lookup
2. Filter rows
3. Sort if needed
4. Project columns`],
	]);

	addModuleWithTasks(sqlitePath.id, 'Week 3-4: Storage Engine', 'B-tree and file format', 1, [
		['Design page format', 'Fixed-size pages', `## Page Layout

4KB pages containing:
- Header (type, count, pointers)
- Cell array (rows/keys)
- Free space
- Cell content area`],
		['Implement B-tree structure', 'Balanced tree for indexing', `## B-Tree

- Internal nodes: keys + child pointers
- Leaf nodes: keys + data
- Balanced: O(log n) operations`],
		['Build B-tree insertion', 'Add keys with splitting', `## B-Tree Insert

1. Find correct leaf
2. Insert key
3. If overflow, split node
4. Propagate splits up`],
		['Add B-tree deletion', 'Remove with rebalancing', `## B-Tree Delete

1. Find and remove key
2. If underflow, borrow or merge
3. Propagate changes up`],
		['Implement table storage', 'Row format', `## Row Storage

Variable-length rows:
[header][col1_len][col1_data][col2_len][col2_data]...`],
		['Add free list management', 'Reclaim deleted space', `## Free List

Track free pages for reuse
Vacuum to reclaim space`],
	]);

	addModuleWithTasks(sqlitePath.id, 'Week 5-6: Transactions & Polish', 'ACID properties', 2, [
		['Implement write-ahead log', 'WAL for durability', `## WAL

Before modifying page:
1. Write old page to WAL
2. Modify page in memory
3. Checkpoint: flush to main file`],
		['Add transaction support', 'BEGIN/COMMIT/ROLLBACK', `## Transactions

BEGIN: start transaction
COMMIT: make changes permanent
ROLLBACK: undo changes`],
		['Implement locking', 'Concurrency control', `## Locking

- Shared lock: readers
- Exclusive lock: writers
- Lock escalation`],
		['Build REPL interface', 'Interactive SQL shell', `## REPL

\`\`\`
sqlite> SELECT * FROM users;
id | name  | email
1  | alice | alice@example.com
\`\`\``],
		['Add index support', 'CREATE INDEX', `## Indexes

Secondary B-trees:
- Key: indexed column value
- Value: row pointer`],
		['Implement EXPLAIN', 'Show query plan', `## EXPLAIN

Show execution plan:
\`\`\`
EXPLAIN SELECT * FROM users WHERE id = 1;
SEARCH TABLE users USING PRIMARY KEY (id=1)
\`\`\``],
	]);
}

// ============================================================================
// Reimplement: pspy (path 76)
// ============================================================================
const pspyPath = pathsNeedingTasks.find(p => p.name.includes('pspy'));
if (pspyPath) {
	addModuleWithTasks(pspyPath.id, 'Week 1: Process Monitoring', 'Core process snooping', 0, [
		['Parse /proc filesystem', 'Read process information', `## /proc Parsing

For each PID in /proc/:
- /proc/[pid]/cmdline: command line
- /proc/[pid]/exe: executable path
- /proc/[pid]/stat: process state
- /proc/[pid]/status: detailed info`],
		['Implement process enumeration', 'List all processes', `## Process List

\`\`\`go
files, _ := ioutil.ReadDir("/proc")
for _, f := range files {
    if pid, err := strconv.Atoi(f.Name()); err == nil {
        // process PID
    }
}
\`\`\``],
		['Build delta detection', 'Find new/exited processes', `## Delta Detection

Compare process lists:
- New PIDs = spawned processes
- Missing PIDs = exited processes
- Track spawn time for ordering`],
		['Add command line extraction', 'Get full command', `## Command Line

Read /proc/[pid]/cmdline
Replace null bytes with spaces
Handle processes with empty cmdline`],
		['Implement polling loop', 'Continuous monitoring', `## Polling

\`\`\`go
ticker := time.NewTicker(100 * time.Millisecond)
for range ticker.C {
    detectNewProcesses()
}
\`\`\``],
		['Add UID/GID tracking', 'Who ran the command', `## User Tracking

From /proc/[pid]/status:
Uid: real effective saved fs
Gid: real effective saved fs`],
	]);

	addModuleWithTasks(pspyPath.id, 'Week 2: File System Events', 'inotify for file access', 1, [
		['Set up inotify watches', 'Monitor file events', `## inotify

\`\`\`go
fd, _ := syscall.InotifyInit()
syscall.InotifyAddWatch(fd, "/tmp", syscall.IN_CREATE|IN_MODIFY)
\`\`\``],
		['Watch key directories', 'Common locations', `## Directories to Watch

- /tmp, /var/tmp
- /dev/shm
- Cron directories
- User home directories`],
		['Handle inotify events', 'Process notifications', `## Event Handling

Events:
- IN_CREATE: file created
- IN_DELETE: file deleted
- IN_MODIFY: file modified
- IN_OPEN: file opened`],
		['Correlate files to processes', 'Who touched what', `## Correlation

When file event occurs:
- Check /proc/*/fd/* for open handles
- Check fuser output
- Match timing with process spawns`],
		['Add recursive watching', 'Monitor subdirectories', `## Recursive Watch

inotify doesn't recurse automatically
Add watches for new directories
Handle watch limit`],
		['Implement event filtering', 'Reduce noise', `## Filtering

Ignore:
- Known noisy processes
- Specific file patterns
- Configurable exclusions`],
	]);

	addModuleWithTasks(pspyPath.id, 'Week 3: Output & Stealth', 'Display and evasion', 2, [
		['Build colored output', 'Highlight important events', `## Colored Output

Color by type:
- Red: root processes
- Yellow: cron jobs
- Green: user processes
- Blue: file system events`],
		['Add timestamp formatting', 'Precise timing', `## Timestamps

2024-01-15 03:14:15 | UID=0 | /usr/sbin/cron
2024-01-15 03:14:15 | UID=0 | /bin/sh -c /root/backup.sh`],
		['Implement log output', 'Save to file', `## Logging

-o output.log: write to file
Append mode for long runs
Rotate large logs`],
		['Minimize footprint', 'Reduce detectability', `## Stealth

- Static binary (no deps)
- Minimal memory usage
- Low CPU impact
- No disk writes by default`],
		['Add network connection tracking', 'Detect network activity', `## Network Tracking

Parse /proc/[pid]/net/*
Detect new connections
Track listening ports`],
		['Build summary statistics', 'Aggregate information', `## Statistics

- Most frequent commands
- Most active users
- Process spawn rate
- File access patterns`],
	]);
}

// Continue with more paths...

console.log('Done expanding tasks!');

// Count results
const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
