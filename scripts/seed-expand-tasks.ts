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
		['Implement basic TCP listener', 'Create TCP server using net.Listen("tcp", ":8080") that accepts connections in a loop. Spawn goroutine per connection for concurrent handling. Implement graceful shutdown with context cancellation. Test with netcat: nc localhost 8080.', `## TCP Listener Foundation

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
		['Build TCP client connector', 'Implement net.Dial to connect to remote TCP endpoints with configurable timeout via net.DialTimeout. Add retry logic with exponential backoff (1s, 2s, 4s) for transient failures. Handle connection refused, timeout, and DNS resolution errors gracefully.', `## TCP Client

### Implementation
- Dial remote endpoints
- Handle connection timeouts
- Implement retry logic with backoff`],
		['Implement bidirectional data forwarding', 'Use io.Copy in two goroutines for full-duplex forwarding: one copies client→server, other copies server→client. Use sync.WaitGroup or errgroup to wait for both directions. Handle EOF properly - close write side when read side ends.', `## Data Forwarding

Use io.Copy in goroutines for bidirectional streaming:
\`\`\`go
go io.Copy(dst, src)
go io.Copy(src, dst)
\`\`\``],
		['Add connection multiplexing', 'Design framing protocol for multiple logical connections over single TCP connection: each frame has length prefix (2-4 bytes), channel ID (unique per tunneled connection), and payload data. Implement channel open/close handshake and flow control to prevent fast sender overwhelming slow receiver.', `## Multiplexing

### Protocol Design
- Frame format: [length][channel_id][data]
- Channel management
- Flow control basics`],
		['Implement port forwarding mode', 'Local port forward: listen on localhost:8080, when connection arrives, open channel through tunnel to internal:80 on remote network, forward bidirectionally. Syntax: chisel client --local 8080:internal:80. Enables accessing internal services through tunnel.', `## Local Port Forward

chisel client --local 8080:internal:80

Maps localhost:8080 -> internal:80 through tunnel`],
		['Add reverse port forwarding', 'Reverse port forward: tell server to listen on its port 8080, forward connections back through tunnel to client localhost:80. Syntax: chisel client --reverse 8080:localhost:80. Useful for exposing internal services or receiving callbacks through firewalls.', `## Reverse Forward

chisel client --reverse 8080:localhost:80

Expose client's port 80 on server's 8080`],
	]);

	addModuleWithTasks(chiselPath.id, 'Week 3-4: HTTP/WebSocket Transport', 'Tunnel over HTTP for firewall bypass', 1, [
		['Implement HTTP CONNECT tunnel', 'Implement HTTP CONNECT method for tunneling: client sends "CONNECT target:port HTTP/1.1", proxy establishes TCP connection to target, returns "200 Connection Established", then blindly forwards bytes. Standard way to tunnel through corporate proxies.', `## HTTP CONNECT

Standard proxy method:
\`\`\`
CONNECT target:port HTTP/1.1
Host: target:port
\`\`\``],
		['Build WebSocket transport layer', 'Upgrade HTTP connection to WebSocket for bidirectional tunnel: send Upgrade header, complete handshake, then use WebSocket frames for tunnel data. Works through HTTP proxies, looks like normal web traffic to firewalls, supports binary data natively.', `## WebSocket Tunnel

### Why WebSocket?
- Bidirectional communication
- Works through proxies
- Looks like normal web traffic`],
		['Add HTTP request/response framing', 'Encapsulate tunnel data in HTTP for environments that inspect traffic: encode binary as base64 in request/response bodies, use chunked transfer encoding for streaming, rotate between GET and POST requests to look like normal browsing.', `## HTTP Framing

Encode binary data in HTTP bodies
Handle chunked transfer encoding`],
		['Implement connection keepalive', 'Maintain persistent tunnel connections: send WebSocket ping frames periodically (every 30s), detect dead connections via pong timeout, implement automatic reconnection with backoff. Support session resumption to continue interrupted transfers.', `## Keepalive

- Ping/pong frames
- Reconnection logic
- Session resumption`],
		['Add TLS support', 'Secure tunnel transport with TLS: generate self-signed CA and server certificates, or load existing certs. Client verifies server certificate to prevent MITM. Support custom CA for enterprise environments. Use crypto/tls in Go.', `## TLS Configuration

- Generate/load certificates
- Verify server certificates
- Support custom CA`],
		['Build proxy authentication', 'Support authenticated corporate proxies: implement HTTP Basic auth (base64 user:pass in Proxy-Authorization header), add NTLM authentication for Windows environments (3-message challenge-response), handle 407 Proxy Authentication Required responses.', `## Proxy Auth

- Basic authentication
- NTLM support for corporate proxies`],
	]);

	addModuleWithTasks(chiselPath.id, 'Week 5-6: SOCKS5 & Polish', 'Add SOCKS proxy and production features', 2, [
		['Implement SOCKS5 server', 'Full SOCKS5 proxy (RFC 1928): handle version/auth negotiation, support CONNECT command (TCP proxy), BIND command (accept incoming connections), UDP ASSOCIATE (UDP relay). Parse destination as IPv4, IPv6, or domain name. Return success/failure status.', `## SOCKS5 Protocol

### Handshake
1. Client sends supported auth methods
2. Server selects method
3. Authentication (if required)
4. Connection request

### Commands
- CONNECT
- BIND
- UDP ASSOCIATE`],
		['Add SOCKS5 authentication', 'Implement username/password auth (method 0x02): after method selection, client sends version, username length, username, password length, password. Verify against configured credentials. Return success (0x00) or failure (0x01). Support no-auth (0x00) as fallback.', `## SOCKS5 Auth

Method 0x02: Username/Password
\`\`\`
[version][ulen][username][plen][password]
\`\`\``],
		['Implement UDP relay', 'SOCKS5 UDP ASSOCIATE for UDP traffic: client requests UDP relay, server binds UDP port and returns address. Client sends UDP datagrams with SOCKS5 header (domain/port), server relays to destination. Essential for tunneling DNS queries.', `## UDP Relay

For DNS and other UDP protocols through SOCKS`],
		['Add fingerprint resistance', 'Evade traffic analysis: pad packets to random sizes (break length fingerprinting), add random timing jitter between packets, mimic browser TLS fingerprints (JA3) by matching cipher suites and extensions order. Blend with normal HTTPS traffic.', `## Evasion

- Randomize packet sizes
- Add jitter to timing
- Mimic browser TLS fingerprint`],
		['Build CLI interface', 'Create user-friendly CLI with cobra/urfave: server mode (chisel server --port 8080 --auth user:pass), client mode (chisel client server:8080 --socks 1080). Support config files, environment variables, verbose logging flags, version info.', `## CLI Design

\`\`\`
chisel server --port 8080 --auth user:pass
chisel client server:8080 --socks 1080
\`\`\``],
		['Add logging and metrics', 'Production observability: structured JSON logging with levels (debug/info/warn/error), connection statistics (active, total, bytes transferred), Prometheus metrics endpoint for monitoring, configurable log output (file, stdout, syslog).', `## Monitoring

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
		['Build basic HTTP proxy', 'Create HTTP proxy listening on localhost:8080: accept browser connections, parse HTTP request (method, path, headers, body), extract Host header for destination, forward request to target server, stream response back to browser. Handle Connection: keep-alive.', `## HTTP Proxy Foundation

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
		['Implement HTTPS interception', 'MITM HTTPS traffic: generate root CA cert (user installs in browser), when CONNECT request arrives for domain, dynamically generate certificate signed by CA for that domain, establish TLS with client, separately establish TLS with real server, decrypt/re-encrypt all traffic.', `## HTTPS MITM

### Steps
1. Generate root CA certificate
2. On CONNECT, generate cert for target domain
3. Establish TLS with client using generated cert
4. Establish TLS with server
5. Decrypt, inspect, re-encrypt`],
		['Add request/response parsing', 'Parse HTTP/1.1 format: request line (GET /path HTTP/1.1), headers until blank line (handle multi-line folded headers), body via Content-Length or chunked Transfer-Encoding. Preserve exact formatting for faithful forwarding. Support gzip/deflate decompression.', `## HTTP Parsing

Parse:
- Request line (method, path, version)
- Headers (handle multi-line)
- Body (Content-Length, chunked)`],
		['Build request modification', 'Allow intercepting and editing requests before forwarding: modify URL/path, add/remove/change headers, edit body parameters (form data, JSON), change HTTP method. Store original for comparison. Queue requests when in intercept mode.', `## Request Editing

Allow modification of:
- URL/path
- Headers
- Body/parameters
- Method`],
		['Implement response modification', 'Edit responses before returning to browser: change status code, modify headers (remove security headers like CSP for testing), inject or modify body content, handle compressed responses by decompressing first then recompressing.', `## Response Editing

Modify:
- Status code
- Headers
- Body content`],
		['Add WebSocket proxying', 'Handle WebSocket connections: detect "Upgrade: websocket" header, forward handshake (101 Switching Protocols), then proxy WebSocket frames bidirectionally. Parse frame format to display messages in UI. Support binary and text frames.', `## WebSocket Support

1. Detect Upgrade header
2. Forward handshake
3. Proxy frames bidirectionally`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 2: Interceptor UI', 'Terminal UI for inspection', 1, [
		['Set up TUI framework', 'Use Python textual for modern terminal UI: create App subclass, define layout with containers (Horizontal/Vertical), add widgets (DataTable for request list, TextArea for editing). Handle resize, color themes, async updates from proxy thread.', `## TUI Setup

Use Python textual for rich terminal interface:
\`\`\`python
from textual.app import App
from textual.widgets import DataTable, TextArea
\`\`\``],
		['Build request list view', 'Display intercepted requests in scrollable table: columns for ID (auto-increment), Method (colored by type), URL (truncate long paths), Status (colored 2xx/3xx/4xx/5xx), Size (human-readable bytes), Time (request duration). Sort by any column.', `## Request List

Columns:
- ID
- Method
- URL
- Status
- Size
- Time`],
		['Implement request detail view', 'Show full request/response in split panes: left pane shows request headers and body (syntax highlighted for JSON/XML), right pane shows response. Tab between raw/pretty/hex views. Copy to clipboard support. Wrap long lines optionally.', `## Detail View

Split pane:
- Request headers + body
- Response headers + body
- Hex view for binary`],
		['Add intercept toggle', 'Toggle interception mode: when enabled, requests queue and wait for user action. Display pending count. For each intercepted request: Forward (send as-is), Edit (modify then forward), or Drop (cancel request). Show timeout warning for long waits.', `## Intercept Mode

Toggle interception on/off
Queue requests when intercepted
Edit before forwarding`],
		['Build search and filter', 'Filter request history: regex pattern matching on URL, filter by HTTP method checkboxes, status code ranges (2xx, 4xx, etc.), content-type (HTML, JSON, images), size range (min/max bytes). Combine filters with AND logic. Save filter presets.', `## Filtering

Filter by:
- URL pattern (regex)
- Method
- Status code
- Content type
- Size range`],
		['Add keyboard shortcuts', 'Vim-style keyboard navigation: j/k move up/down in list, Enter opens detail view, q closes panels, e enters edit mode, f forwards request, d drops request, / opens search, n/N for next/prev match. Show shortcut hints in status bar.', `## Shortcuts

- j/k: navigate list
- Enter: view details
- e: edit request
- f: forward
- d: drop`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 3: History & Repeater', 'Request storage and replay', 2, [
		['Implement SQLite storage', 'Persist all requests to SQLite database: store method, full URL, headers as JSON, body as BLOB (handle binary), response status/headers/body, timestamp. Index URL and timestamp for fast queries. Use WAL mode for concurrent access from proxy and UI threads.', `## Database Schema

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
		['Build history browser', 'Browse stored requests with pagination (load 100 at a time for performance). Full-text search across URLs and response bodies using SQLite FTS5. Filter by date range picker. Show total count and current page. Virtual scrolling for large histories.', `## History Features

- Pagination for large histories
- Full-text search
- Date range filtering`],
		['Create Repeater tool', 'Send modified requests repeatedly: select request from history, edit any part (method, path, headers, body), send to server, view response side-by-side with original. Save variations as named tabs. Track request/response history per tab.', `## Repeater

1. Select request from history
2. Modify as needed
3. Send and view response
4. Compare with original`],
		['Add response comparison', 'Diff two responses visually: compare status codes, highlight added/removed headers, show unified diff of bodies. Support comparing any two requests from history. Useful for detecting behavior changes between requests.', `## Response Diff

Highlight differences:
- Status code changes
- Header changes
- Body changes (unified diff)`],
		['Implement export formats', 'Export requests in multiple formats: cURL command (with -H headers, -d body), Python requests code, raw HTTP format, HAR (HTTP Archive) for browser import. Copy to clipboard or save to file. Include response optionally.', `## Export Formats

- cURL command
- Python requests
- Raw HTTP
- HAR format`],
		['Build session management', 'Save/restore complete proxy state: export database to file with all request history, save settings (proxy port, intercept rules, scope), load previous session on startup. Auto-save periodically to prevent data loss.', `## Sessions

Save complete state:
- Request history
- Settings
- Scope configuration`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 4: Active Scanner', 'Automated vulnerability scanning', 3, [
		['Build endpoint crawler', 'Discover all endpoints from proxy history: extract unique URLs, parse HTML for form actions and links, analyze JavaScript for fetch/XMLHttpRequest URLs, identify API paths from JSON responses. Build sitemap of testable endpoints.', `## Crawler

Extract from history:
- All unique URLs
- Form actions
- JavaScript endpoints
- API paths`],
		['Implement payload injection', 'Identify injection points in requests: URL query parameters (?id=1), POST body parameters (form-urlencoded and multipart), HTTP headers (Cookie, User-Agent, Referer, X-Forwarded-For), JSON fields (nested objects too). Generate test requests for each point.', `## Injection Points

Identify:
- URL parameters
- POST body parameters
- Headers (Cookie, User-Agent, etc.)
- JSON fields`],
		['Add XSS detection', 'Test for XSS vulnerabilities: inject payloads like <script>alert(1)</script>, break out of attributes with "><img onerror=alert(1)>, try event handlers (onmouseover, onfocus). Check if payload appears unencoded in response. Test different contexts (HTML, attribute, JavaScript).', `## XSS Scanner

### Payloads
- \`<script>alert(1)</script>\`
- \`"><img src=x onerror=alert(1)>\`
- Event handlers

### Detection
Check if payload reflected unencoded`],
		['Implement SQLi detection', 'Test for SQL injection: send single quote and detect SQL error messages in response, use boolean payloads (OR 1=1 vs OR 1=2) and compare response differences, measure response time with SLEEP/WAITFOR for blind detection. Fingerprint database type from errors.', `## SQLi Detection

### Payloads
- Single quote: \`'\`
- Boolean: \`' OR '1'='1\`
- Time-based: \`'; WAITFOR DELAY '0:0:5'--\`

### Detection
- Error messages
- Response differences
- Time delays`],
		['Add SSRF/LFI checks', 'Test server-side vulnerabilities: try localhost URLs (http://127.0.0.1/admin), cloud metadata endpoints (169.254.169.254), file:// protocol for LFI. Use out-of-band callback server (like Burp Collaborator) to detect blind SSRF where response not visible.', `## SSRF Detection

### Payloads
- \`http://localhost/admin\`
- \`http://169.254.169.254/\`
- \`file:///etc/passwd\`

### Out-of-band
Use callback server to detect blind SSRF`],
		['Build scan reports', 'Generate professional vulnerability reports: executive summary with risk overview, detailed findings with severity ratings (Critical/High/Medium/Low), affected endpoints, request/response evidence, remediation recommendations with code examples. Export as HTML/PDF.', `## Reports

Include:
- Executive summary
- Vulnerability details
- Risk ratings
- Remediation advice
- Request/response evidence`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 5: Intruder', 'Automated attack tool', 4, [
		['Build payload position marker', 'Mark injection points in request template using delimiters like §param§. Support multiple positions per request. Auto-detect common positions (parameters, headers). Visual editor to add/remove markers. Preserve markers across request edits.', `## Position Marking

Use markers like §param§ to identify injection points
Support multiple positions per request`],
		['Implement Sniper attack', 'Single position attack mode: iterate through payload list, inject each payload into one position at a time while leaving others at baseline. If 3 positions and 100 payloads, sends 300 requests. Best for testing individual parameters for vulnerabilities.', `## Sniper Mode

One position at a time
Iterate through payload list
Good for: fuzzing parameters`],
		['Add Battering Ram attack', 'Same payload in all positions simultaneously: use single payload list, inject same value into all marked positions for each request. Useful for testing username=§x§&password=§x§ where same value needed in both. Simpler than Cluster Bomb for this case.', `## Battering Ram

Same payload in all positions simultaneously
Good for: username=§x§&password=§x§`],
		['Build Pitchfork attack', 'Parallel payload lists: position 1 uses list 1, position 2 uses list 2, iterate in lockstep (first items together, then second items). For credential stuffing with known username:password pairs. Sends N requests where N is length of shortest list.', `## Pitchfork

Multiple lists, iterate in parallel:
- usernames[0] + passwords[0]
- usernames[1] + passwords[1]
Good for: credential stuffing`],
		['Implement Cluster Bomb', 'All combinations attack: try every combination of all payload lists. With 100 usernames and 1000 passwords, sends 100,000 requests. Exponential growth with positions. Classic brute force attack. Add rate limiting to avoid lockouts.', `## Cluster Bomb

All combinations of all payload lists
usernames × passwords
Good for: brute force`],
		['Add result analysis', 'Identify interesting responses: flag different status codes (200 vs 403), different response lengths (indicates different behavior), grep for patterns (Welcome, Invalid, Error), highlight response time anomalies. Sort and filter results table by any column.', `## Analysis

Flag responses with:
- Different status code
- Different length
- Matching pattern
- Different response time`],
	]);

	addModuleWithTasks(burpPath.id, 'Week 6: Extensions & Polish', 'Plugin system and utilities', 5, [
		['Design plugin architecture', 'Create extensible hook system: plugins implement onRequest(req) to modify/log requests, onResponse(resp) for responses, onScanResult(vuln) for custom vulnerability handling. Load plugins from directory at startup. Provide API for accessing proxy state, sending requests.', `## Plugin System

Hooks:
- onRequest(req) -> modified req
- onResponse(resp) -> modified resp
- onScanResult(vuln) -> custom handling`],
		['Build Decoder utility', 'Encode/decode transformation tool: Base64 encode/decode, URL encoding (%20, +), HTML entities (&amp;, &#60;), hex encoding (\\x41), gzip/deflate compression. Chain multiple transforms. Smart decode attempts auto-detection. Copy output to clipboard.', `## Decoder

Transformations:
- Base64 encode/decode
- URL encode/decode
- HTML entities
- Hex encode/decode
- Gzip compress/decompress`],
		['Create Comparer tool', 'Visual diff utility for comparing requests/responses: side-by-side view, highlight added/removed lines (green/red), character-level diff within lines, skip whitespace option. Support comparing binary content with hex view. Load any two items from history.', `## Comparer

Side-by-side comparison
Highlight differences
Support binary comparison`],
		['Add Sequencer', 'Analyze randomness of session tokens: collect N samples automatically (configurable count), calculate entropy (bits of randomness), perform chi-square and other statistical tests, identify predictable patterns. Flag weak tokens that might be guessable.', `## Sequencer

Analyze session tokens:
- Collect samples
- Statistical analysis
- Entropy calculation
- Identify patterns`],
		['Implement scope configuration', 'Define target scope for scanning: include patterns (*.example.com, example.com/api/*), exclude patterns (*.google.com, /logout). Apply scope to scanner, intruder, history highlighting. Out-of-scope requests shown in gray or hidden.', `## Scope

Define target scope:
- Include patterns
- Exclude patterns
- Only scan in-scope items`],
		['Final polish and documentation', 'Production readiness: comprehensive error handling with user-friendly messages, graceful degradation when components fail, user documentation with screenshots and examples, tutorial workflows for common tasks, performance profiling and optimization for large histories.', `## Polish

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
		['Design module architecture', 'Build plugin-based module system with types: Exploits (deliver payloads to vulnerabilities), Auxiliary (scanning, fuzzing, DoS), Post (post-exploitation actions), Payloads (shellcode and agents), Encoders (obfuscation), Nops (NOP sled generation). Each type has specific base class and interface.', `## Module System

### Module Types
- Exploits: deliver payloads
- Auxiliary: scanning, fuzzing
- Post: post-exploitation
- Payloads: shellcode/agents
- Encoders: payload encoding
- Nops: NOP sleds`],
		['Implement module loader', 'Dynamically load modules from filesystem: scan modules/ directory tree, parse Ruby/Python files, extract metadata (name, description, author, references), instantiate module classes on demand. Support hot-reload during development. Handle syntax errors gracefully.', `## Module Loading

\`\`\`ruby
# modules/exploits/windows/smb/ms17_010.rb
class MetasploitModule < Msf::Exploit::Remote
  def exploit
    # exploitation logic
  end
end
\`\`\``],
		['Build datastore system', 'Module configuration via datastore: define options with types (string, integer, address, port), required vs optional, default values, validation. Common options: RHOSTS (targets), RPORT (port), PAYLOAD (selected payload), LHOST/LPORT (callback listener). Inherit options from parent classes.', `## Datastore

Options:
- RHOSTS: target hosts
- RPORT: target port
- PAYLOAD: selected payload
- LHOST/LPORT: listener`],
		['Create module mixins', 'Shared functionality via mixins: Msf::Exploit::Remote::Tcp (socket handling), HttpClient (HTTP requests with cookies/auth), SMB (SMB protocol), Msf::Auxiliary::Scanner (multi-target scanning with threading). Mixins provide connect(), send_request_cgi(), etc.', `## Mixins

- Msf::Exploit::Remote::Tcp
- Msf::Exploit::Remote::HttpClient
- Msf::Exploit::Remote::SMB
- Msf::Auxiliary::Scanner`],
		['Implement target handling', 'Define multiple exploit targets: each target specifies OS version, software version, architecture (x86/x64), and memory offsets (return addresses, gadgets). User selects target or auto-detect. Different targets may need different shellcode, stack pivots, or ROP chains.', `## Targets

Define multiple targets:
- OS versions
- Software versions
- Architecture (x86, x64)
- Each with specific offsets`],
		['Add payload compatibility', 'Match payloads to exploits: filter by platform (windows, linux, osx), architecture (x86, x64, arm), connection type (reverse_tcp, bind_tcp, reverse_http), space constraints (exploit may limit payload size). Show only compatible payloads in UI.', `## Payload Compat

Filter payloads by:
- Platform (windows, linux)
- Architecture
- Connection type (reverse, bind)
- Size constraints`],
	]);

	addModuleWithTasks(msfPath.id, 'Week 3-4: Payload System', 'Shellcode and staged payloads', 1, [
		['Build payload generator', 'Generate position-independent shellcode: combine stager (small code that establishes connection, ~300 bytes) with stage (full agent loaded over connection, ~100KB+). Stager fits in exploits tight space constraints, pulls larger stage over network.', `## Payload Generation

Assemble shellcode from stagers + stages:
1. Stager: small, establishes connection
2. Stage: full agent, loaded by stager`],
		['Implement stagers', 'Build connection stagers: reverse_tcp (connect back to attacker LHOST:LPORT), bind_tcp (listen on victim port for attacker connection), reverse_http (callback via HTTP to blend with web traffic), reverse_https (encrypted variant). Each ~200-500 bytes of shellcode.', `## Stagers

reverse_tcp: connect back to attacker
bind_tcp: listen for connection
reverse_http: HTTP-based callback`],
		['Create Meterpreter stage', 'Full-featured post-exploitation agent: filesystem operations (upload, download, search), process management (list, migrate, inject), network pivoting (route traffic through victim), credential harvesting (hashdump, mimikatz), surveillance (keylogger, screenshot, webcam).', `## Meterpreter

Features:
- File system access
- Process management
- Network pivoting
- Keystroke capture
- Screenshot`],
		['Add payload encoding', 'Evade signature detection: XOR encoding with random key, shikata_ga_nai polymorphic encoder (different output each generation), alpha-numeric (A-Za-z0-9 only for restricted inputs). Chain multiple encoders. Iterate to avoid bad characters (null bytes, newlines).', `## Encoders

- XOR encoding
- Shikata ga nai (polymorphic)
- Alpha-numeric
- Custom encoders`],
		['Implement payload templates', 'Inject shellcode into executable templates: Windows PE (modify .text section, fix entry point), Linux ELF (similar technique), macOS Mach-O, script wrappers (PowerShell, Python, VBScript). Preserve original functionality while adding payload execution.', `## Templates

Inject payload into:
- Windows PE
- Linux ELF
- macOS Mach-O
- Script files`],
		['Build handler system', 'Manage payload listeners: exploit/multi/handler catches incoming connections from any payload type. Run as background job (-j flag), handle multiple simultaneous sessions, auto-migrate for stability, integrate sessions with post modules.', `## Handlers

\`\`\`
use exploit/multi/handler
set PAYLOAD windows/meterpreter/reverse_tcp
set LHOST 10.0.0.1
exploit -j
\`\`\``],
	]);

	addModuleWithTasks(msfPath.id, 'Week 5-6: Console & Database', 'CLI and session management', 2, [
		['Build interactive console', 'Create msfconsole interface using readline: tab completion for commands, modules, and options. Command history with persistence across sessions. Context switching (use module changes prompt). Resource scripts (.rc files) for automation. Color-coded output.', `## Console

- Tab completion
- Command history
- Module context switching
- Resource scripts`],
		['Implement session management', 'Track all active sessions: list sessions with ID, type, target info. Interact with specific session (sessions -i 1). Kill sessions (sessions -k 1). Upgrade basic shell to Meterpreter (sessions -u 1). Background sessions (Ctrl+Z) to return to console.', `## Sessions

\`\`\`
sessions -l          # list
sessions -i 1        # interact
sessions -k 1        # kill
sessions -u 1        # upgrade shell
\`\`\``],
		['Add database backend', 'PostgreSQL storage for engagement data: hosts table (IP, OS, name), services (port, protocol, state, banner), vulns (references, severity), creds (username, password/hash, realm), loots (files, screenshots, data extracted). Enable correlation across findings.', `## Database

Tables:
- hosts: discovered hosts
- services: ports/services
- vulns: vulnerabilities
- creds: credentials
- loots: exfiltrated data`],
		['Create workspace support', 'Organize data by engagement: workspace -a creates new workspace, workspace name switches to it. Each workspace has isolated hosts, services, vulns, creds. Prevents data mixing between clients. Default workspace for general use.', `## Workspaces

Separate data by engagement:
\`\`\`
workspace -a client1
workspace client1
\`\`\``],
		['Build reporting engine', 'Generate engagement reports: HTML with executive summary, findings, and evidence. XML for tool import. JSON for custom processing. Custom templates with ERB/Jinja. Include screenshots, request/response data, remediation recommendations.', `## Reports

Export formats:
- HTML report
- XML export
- JSON dump
- Custom templates`],
		['Implement RPC API', 'Remote control via MessagePack-RPC: authenticate with token, list/use modules, set options, execute exploits, manage sessions. Enables GUI clients, automation scripts, CI/CD integration. Python library pymetasploit3 simplifies access.', `## MSFRPC

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
		['Implement hash identification', 'Auto-detect hash type from format: length (MD5=32 hex, SHA1=40, SHA256=64, SHA512=128), character set (hex only vs base64), format markers ($1$ for MD5crypt, $6$ for SHA512crypt, $2a$ for bcrypt). Handle multiple possible matches.', `## Hash Detection

Identify by:
- Length (MD5=32, SHA1=40, SHA256=64)
- Character set
- Format markers ($1$, $6$, $2a$)
- Context clues`],
		['Build MD5 cracker', 'Basic hash cracking: read wordlist line by line, compute MD5 hash of each word, compare hex digest to target hash. Return matching plaintext. Baseline implementation before optimization. Measure words/second for benchmarking.', `## MD5 Implementation

\`\`\`python
import hashlib
def crack_md5(hash, wordlist):
    for word in wordlist:
        if hashlib.md5(word.encode()).hexdigest() == hash:
            return word
\`\`\``],
		['Add SHA family support', 'Extend to SHA variants: SHA1 (40 chars), SHA256 (64 chars), SHA512 (128 chars). Same algorithm structure - hash candidate, compare to target. Use hashlib.sha1(), sha256(), sha512(). Support raw binary hash input as well as hex.', `## SHA Support

Same pattern as MD5:
- Read wordlist
- Hash each candidate
- Compare to target`],
		['Implement bcrypt cracking', 'Handle computationally expensive bcrypt: honor cost factor from hash ($2a$10$ means 2^10 iterations). Much slower than MD5 (maybe 100/sec vs millions/sec). Parallel processing across CPU cores essential. Consider early termination on partial match.', `## bcrypt

Much slower - requires optimization:
- Cost factor awareness
- Early termination
- Parallel processing critical`],
		['Build dictionary attack', 'Wordlist-based cracking mode (-a 0): load wordlist into memory or stream from disk, hash each word, compare to target(s). Support multiple hash types via mode flag (-m). Track progress, display speed (H/s), estimate time remaining.', `## Dictionary Attack

\`\`\`
./cracker -m 0 -a 0 hash.txt wordlist.txt
\`\`\`

Modes: straight, combination, rule-based`],
		['Add rule engine', 'Word mangling rules to expand wordlist: l (lowercase), u (uppercase), c (capitalize first), $1 (append "1"), ^! (prepend "!"), r (reverse), d (duplicate). Apply rules to each wordlist word generating variants. Chain multiple rules.', `## Rules

Common transformations:
- l: lowercase
- u: uppercase
- c: capitalize
- $1: append 1
- ^!: prepend !
- r: reverse`],
	]);

	addModuleWithTasks(hashcatPath.id, 'Week 3-4: GPU Acceleration', 'CUDA/OpenCL implementation', 1, [
		['Set up CUDA environment', 'Learn GPU programming basics: install CUDA toolkit, understand GPU architecture (thousands of simple cores), execution model (blocks contain threads, 32 threads = warp execute in lockstep), memory hierarchy (global slow, shared fast, registers fastest).', `## CUDA Setup

- Install CUDA toolkit
- Understand GPU architecture
- Blocks, threads, warps
- Memory hierarchy`],
		['Implement GPU MD5 kernel', 'Parallel MD5 on GPU: each thread computes hash of one candidate. Implement MD5 algorithm in CUDA (no standard library). Copy wordlist batch to GPU, launch kernel with thousands of threads, each compares result against target hashes. Retrieve matches.', `## GPU MD5

\`\`\`cuda
__global__ void md5_kernel(char* wordlist, char* hashes, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // compute MD5 of wordlist[idx]
    // compare against target hashes
}
\`\`\``],
		['Add batch processing', 'Process millions of candidates: load wordlist chunk into host memory, transfer to GPU (cudaMemcpy), launch kernel over batch, retrieve results. Overlap transfer of next batch with computation of current batch using CUDA streams.', `## Batching

- Load wordlist in chunks
- Transfer to GPU memory
- Process in parallel
- Retrieve results`],
		['Implement multiple hash attack', 'Attack many hashes simultaneously: store all target hashes in GPU memory, each candidate compares against entire hash list. Cost of hashing amortized across all targets - cracking 1000 hashes only slightly slower than cracking 1. Massive efficiency gain.', `## Multi-Hash

Compare each candidate against ALL target hashes
Amortizes computation cost
Much faster for large hash lists`],
		['Add OpenCL support', 'Cross-platform GPU support via OpenCL: write kernels once, run on AMD GPUs, Intel integrated graphics, NVIDIA GPUs, or CPU fallback. Abstract device selection, memory management, kernel compilation. Performance may vary by vendor.', `## OpenCL

Platform-independent:
- AMD GPUs
- Intel GPUs
- NVIDIA GPUs
- CPU fallback`],
		['Optimize memory transfers', 'Minimize CPU-GPU transfer overhead: use pinned (page-locked) memory for faster DMA transfers, async transfers with cudaMemcpyAsync, double buffering (compute batch N while transferring batch N+1), multiple CUDA streams for concurrency.', `## Memory Optimization

- Pinned memory
- Async transfers
- Double buffering
- Stream concurrency`],
	]);

	addModuleWithTasks(hashcatPath.id, 'Week 5-6: Advanced Attacks', 'Masks, combinator, and brain', 2, [
		['Build mask attack', 'Pattern-based brute force: define character sets per position using masks. ?l=lowercase, ?u=uppercase, ?d=digit, ?s=special, ?a=all. Example: ?u?l?l?l?d?d?d?d matches Pass1234. Calculate keyspace size and estimated time. Custom charsets with -1.', `## Mask Attack

?l = lowercase
?u = uppercase
?d = digit
?s = special

Example: ?u?l?l?l?d?d?d?d
Matches: Pass1234`],
		['Implement combinator attack', 'Combine two wordlists: concatenate each word from list1 with each word from list2. admin+123=admin123, super+user=superuser. Keyspace is len(list1) × len(list2). Can add rules to modify left and right words before combining.', `## Combinator

word1 + word2 from two lists:
admin + 123 = admin123
super + user = superuser`],
		['Add hybrid attacks', 'Combine wordlist with mask: append or prepend brute force to dictionary words. password?d?d?d tests password000 through password999. ?d?d?d?dpassword for prefix. Efficient for common patterns like word+numbers.', `## Hybrid

Wordlist + brute force:
- password?d?d?d
- ?d?d?d?dpassword`],
		['Build markov generator', 'Statistical password generation using Markov chains: train on password leaks to learn character transition probabilities. Generate candidates in probability order (most likely first). Much more efficient than pure brute force for human-created passwords.', `## Markov Chains

Learn password patterns from leaks
Generate statistically likely candidates
Much faster than brute force`],
		['Implement distributed cracking', 'Multi-machine cracking: split keyspace across nodes (machine 1 gets 0-25%, machine 2 gets 25-50%, etc.). Central coordinator assigns work, collects results, tracks progress. Handle node failures gracefully. Aggregate found passwords.', `## Distributed

- Split keyspace across machines
- Central coordinator
- Progress synchronization
- Result aggregation`],
		['Add restore/session support', 'Save and resume attacks: periodically checkpoint current position in keyspace, attack configuration (mode, wordlist, rules, mask), found passwords, runtime statistics. On restart, load session file and continue from last position. Essential for long attacks.', `## Sessions

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
		['Implement read-eval-print loop', 'Core shell loop: print prompt (user@host:cwd$), read input line with readline or getline, parse into command and arguments, execute command, repeat. Handle EOF (Ctrl+D) to exit. Clear errno between iterations.', `## REPL

\`\`\`c
while (1) {
    print_prompt();
    char* line = read_line();
    char** args = parse_line(line);
    execute(args);
}
\`\`\``],
		['Build command parser', 'Tokenize input line: split on whitespace, handle quoted strings (preserve spaces inside quotes), process escape characters (\\n, \\t, \\\\), recognize special characters (|, <, >, &, ;) as separate tokens. Return array of argument strings.', `## Parsing

Handle:
- Whitespace splitting
- Quoted strings
- Escape characters
- Special characters`],
		['Implement fork/exec', 'Execute external commands: fork() creates child process, child calls execvp(command, args) to replace itself with command, parent calls waitpid() to wait for completion. Check exit status via WIFEXITED/WEXITSTATUS macros. Handle execvp failure.', `## Process Execution

\`\`\`c
pid_t pid = fork();
if (pid == 0) {
    execvp(args[0], args);
    exit(EXIT_FAILURE);
}
waitpid(pid, &status, 0);
\`\`\``],
		['Add PATH searching', 'Find executables in PATH: if command not absolute path, search each directory in PATH environment variable (/usr/local/bin:/usr/bin:/bin). Check file exists and is executable (access with X_OK). Use first match. execvp does this automatically.', `## PATH Lookup

Search directories in PATH:
/usr/local/bin
/usr/bin
/bin`],
		['Build built-in commands', 'Commands that must run in shell process (cant fork): cd (chdir changes shells cwd), exit (terminate shell), export (modify shells environment), alias (command substitution), source (execute script in current shell). Check for builtins before fork/exec.', `## Builtins

Must be built-in (can't fork):
- cd: change directory
- exit: terminate shell
- export: set env var
- alias: command aliases`],
		['Implement environment variables', 'Manage environment: getenv("PATH") retrieves value, setenv("VAR","value",1) sets it. Expand $VAR and ${VAR} in command lines before execution. Support $HOME, $PATH, $USER, $PWD, and custom variables.', `## Environment

- $HOME, $PATH, $USER
- Custom variables
- Variable expansion: $VAR, \${VAR}`],
	]);

	addModuleWithTasks(shellPath.id, 'Week 3-4: I/O and Pipes', 'Redirection and pipelines', 1, [
		['Add input redirection', 'Redirect stdin from file: parse < filename in command, open file for reading in child process before exec, use dup2(fd, STDIN_FILENO) to replace stdin with file descriptor. Close original fd. Command reads from file instead of terminal.', `## Input Redirect

\`\`\`
command < input.txt
\`\`\`

Open file, dup2 to stdin before exec`],
		['Implement output redirection', 'Redirect stdout to file: > truncates/creates file (O_WRONLY|O_CREAT|O_TRUNC), >> appends (O_WRONLY|O_CREAT|O_APPEND). Open file in child, dup2(fd, STDOUT_FILENO). Set permissions 0644 on create.', `## Output Redirect

\`\`\`
command > output.txt   # overwrite
command >> output.txt  # append
\`\`\`

dup2 file to stdout`],
		['Build stderr redirection', 'Redirect stderr: 2> redirects fd 2 (stderr) to file. 2>&1 duplicates stdout to stderr (order matters - do after stdout redirect). &> redirects both stdout and stderr to same file. dup2(fd, STDERR_FILENO).', `## Stderr

\`\`\`
command 2> errors.txt
command > all.txt 2>&1
\`\`\``],
		['Implement pipes', 'Connect commands with pipes: pipe() creates fd pair (read and write ends). Fork first process, dup2 write end to stdout. Fork second process, dup2 read end to stdin. Close unused ends in each process. Wait for both children.', `## Pipes

\`\`\`c
int pipefd[2];
pipe(pipefd);
// fork, dup2 write end to stdout
// fork, dup2 read end to stdin
\`\`\``],
		['Add multiple pipes', 'Handle cmd1 | cmd2 | cmd3: create N-1 pipes for N commands. Each command (except last) writes to pipe, each command (except first) reads from pipe. Fork all processes, connect pipe ends appropriately, close unused fds, wait for all children.', `## Pipeline

Handle arbitrary length pipelines
Connect stdout -> stdin between processes`],
		['Implement here documents', 'Here-doc input (<<): read lines until delimiter (EOF), write to pipe, child reads from pipe as stdin. Support <<- for stripped leading tabs. Expand variables in content unless delimiter quoted (<<"EOF"). Useful for inline multi-line input.', `## Here Doc

\`\`\`
cat << EOF
multiple
lines
EOF
\`\`\``],
	]);

	addModuleWithTasks(shellPath.id, 'Week 5-6: Job Control & Polish', 'Background jobs and signals', 2, [
		['Add background execution', 'Run command in background: if line ends with &, fork but dont waitpid. Print job number and PID: [1] 12345. Add to job list with Running status. Use WNOHANG with waitpid to reap finished background jobs without blocking.', `## Background Jobs

\`\`\`
long_command &
\`\`\`

Don't wait for process, print job number`],
		['Implement job listing', 'Track background jobs: maintain list of job structs (pid, command, status). jobs command prints list with job numbers, status (Running, Stopped, Done), and command. Update status when jobs complete (check in SIGCHLD handler or before prompt).', `## Jobs

Track background processes:
[1]+ Running    sleep 100 &
[2]- Running    vim &`],
		['Build fg/bg commands', 'Job control builtins: fg %N brings job N to foreground (tcsetpgrp to give terminal control, waitpid to wait). bg %N continues stopped job in background (kill with SIGCONT). Support %% for current job, %- for previous.', `## Foreground/Background

fg %1: bring job 1 to foreground
bg %1: continue job 1 in background`],
		['Handle signals', 'Signal handling: install handlers for SIGINT (Ctrl+C interrupt - forward to foreground job, dont kill shell), SIGTSTP (Ctrl+Z suspend - stop foreground job), SIGCHLD (child exited - update job status). Shell ignores signals when running foreground job.', `## Signals

- Ctrl+C: SIGINT (interrupt)
- Ctrl+Z: SIGTSTP (suspend)
- Ctrl+D: EOF (exit)

Setup signal handlers properly`],
		['Add command history', 'Command history with readline library or custom implementation: store commands in list, Up/Down arrow navigates history, !n runs command n from history, !! repeats last command, !string runs most recent command starting with string. Persist to ~/.history.', `## History

- Up/Down arrow navigation
- !n: run command n
- !!: repeat last command
- history: show history`],
		['Implement tab completion', 'Tab completion: on Tab keypress, get partial word, find matches. For commands, search PATH directories. For files, search current directory. For variables, search environment. Display multiple matches, complete unique prefix. Use readline library or handle raw terminal input.', `## Completion

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
		['Build SQL tokenizer', 'Lexical analysis: scan SQL string character by character, produce tokens. Token types: keywords (SELECT, FROM, WHERE - case insensitive), identifiers (table/column names), literals (strings in quotes, numbers), operators (=, <>, <=, AND, OR). Track line/column for errors.', `## Tokenizer

Token types:
- Keywords: SELECT, FROM, WHERE
- Identifiers: table/column names
- Literals: strings, numbers
- Operators: =, <, >, AND, OR`],
		['Implement expression parser', 'Recursive descent parser for expressions: handle operator precedence (AND before OR, comparison before boolean), parse comparisons (a = 1, b > 5), arithmetic expressions (a + b * c respects precedence), parentheses for grouping. Build AST nodes.', `## Expression Parsing

Build AST for:
- Comparisons: a = 1
- Boolean: a AND b OR c
- Arithmetic: a + b * c`],
		['Parse SELECT statements', 'Parse full SELECT: column list (*, specific columns, expressions with aliases), FROM clause (table name, joins), WHERE clause (filter expression), ORDER BY (column list with ASC/DESC), LIMIT/OFFSET. Build SelectStatement AST node.', `## SELECT Parser

\`\`\`sql
SELECT col1, col2
FROM table
WHERE condition
ORDER BY col
LIMIT n
\`\`\``],
		['Parse INSERT/UPDATE/DELETE', 'Data modification statements: INSERT INTO table (columns) VALUES (values) or INSERT INTO table SELECT. UPDATE table SET column=value, column2=value2 WHERE condition. DELETE FROM table WHERE condition. Validate column counts match.', `## DML Parsing

INSERT INTO table (cols) VALUES (vals)
UPDATE table SET col = val WHERE cond
DELETE FROM table WHERE cond`],
		['Add CREATE TABLE parsing', 'DDL statements: CREATE TABLE name (column_def, column_def...). Column definitions include name, type (INTEGER, TEXT, REAL, BLOB), constraints (PRIMARY KEY, NOT NULL, UNIQUE, DEFAULT value, FOREIGN KEY). Parse and validate schema.', `## DDL Parser

\`\`\`sql
CREATE TABLE name (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
)
\`\`\``],
		['Implement query planner', 'Convert AST to execution plan: analyze WHERE clause for index usage, choose table scan vs index lookup, plan join order for multi-table queries, add sort step if ORDER BY on non-indexed column, project only needed columns. Optimize by pushing filters down.', `## Query Planning

Convert AST to execution plan:
1. Table scan or index lookup
2. Filter rows
3. Sort if needed
4. Project columns`],
	]);

	addModuleWithTasks(sqlitePath.id, 'Week 3-4: Storage Engine', 'B-tree and file format', 1, [
		['Design page format', 'Fixed 4KB pages: header (page type, cell count, free space pointer, rightmost child), cell pointer array (offsets to cells), free space in middle, cell content area growing from end. Page types: internal node, leaf node, overflow, free.', `## Page Layout

4KB pages containing:
- Header (type, count, pointers)
- Cell array (rows/keys)
- Free space
- Cell content area`],
		['Implement B-tree structure', 'B+tree for indexing: internal nodes store keys and child page pointers (no data), leaf nodes store keys and row data (or rowids), leaves linked for range scans. Balanced by construction - O(log n) for search, insert, delete. Typical fanout 100-200 keys per node.', `## B-Tree

- Internal nodes: keys + child pointers
- Leaf nodes: keys + data
- Balanced: O(log n) operations`],
		['Build B-tree insertion', 'Insert into B-tree: traverse from root to find correct leaf, insert key in sorted position. If leaf overflows (too many keys), split into two leaves and push middle key up to parent. Recursively split parents as needed. May create new root.', `## B-Tree Insert

1. Find correct leaf
2. Insert key
3. If overflow, split node
4. Propagate splits up`],
		['Add B-tree deletion', 'Delete from B-tree: find and remove key from leaf. If leaf underflows (too few keys), borrow key from sibling or merge with sibling. Update parent keys. Recursively fix underflows up tree. May reduce tree height if root becomes empty.', `## B-Tree Delete

1. Find and remove key
2. If underflow, borrow or merge
3. Propagate changes up`],
		['Implement table storage', 'Row format: variable-length records with header (column count, column types), then column data (length-prefixed for variable types). NULL stored as type flag, not as data. Overflow pages for rows exceeding page size. Rowid as implicit primary key.', `## Row Storage

Variable-length rows:
[header][col1_len][col1_data][col2_len][col2_data]...`],
		['Add free list management', 'Track free pages: maintain linked list of free pages in file header. When page deleted, add to free list. When page needed, pop from free list before extending file. VACUUM command rewrites database to reclaim fragmented space.', `## Free List

Track free pages for reuse
Vacuum to reclaim space`],
	]);

	addModuleWithTasks(sqlitePath.id, 'Week 5-6: Transactions & Polish', 'ACID properties', 2, [
		['Implement write-ahead log', 'WAL for durability: before modifying main database page, write original page content to WAL file. Modifications go to memory/WAL. On checkpoint, flush WAL changes to main database file. Crash recovery replays WAL. Enables readers during writes.', `## WAL

Before modifying page:
1. Write old page to WAL
2. Modify page in memory
3. Checkpoint: flush to main file`],
		['Add transaction support', 'ACID transactions: BEGIN starts transaction (implicit for single statements), COMMIT writes changes permanently, ROLLBACK undoes all changes since BEGIN. Ensure atomicity (all or nothing), consistency (valid state), isolation (transactions dont interfere), durability (committed = permanent).', `## Transactions

BEGIN: start transaction
COMMIT: make changes permanent
ROLLBACK: undo changes`],
		['Implement locking', 'Concurrency control: shared lock for readers (multiple allowed), exclusive lock for writers (one at a time, no readers). Lock escalation from shared to exclusive on write. SQLite uses file-level locking; more sophisticated: row-level locks, MVCC.', `## Locking

- Shared lock: readers
- Exclusive lock: writers
- Lock escalation`],
		['Build REPL interface', 'Interactive SQL shell: read line, parse SQL, execute query, display results in formatted table. Handle multiple statements separated by semicolon. Special commands: .tables, .schema, .quit. Tab completion for table/column names. History support.', `## REPL

\`\`\`
sqlite> SELECT * FROM users;
id | name  | email
1  | alice | alice@example.com
\`\`\``],
		['Add index support', 'Secondary indexes via CREATE INDEX: build separate B-tree where key is indexed column value, value is rowid pointing to main table row. Query planner chooses index when WHERE clause matches indexed column. Maintain index on INSERT/UPDATE/DELETE.', `## Indexes

Secondary B-trees:
- Key: indexed column value
- Value: row pointer`],
		['Implement EXPLAIN', 'Show query execution plan: EXPLAIN SELECT shows steps like SCAN TABLE (full scan), SEARCH TABLE USING INDEX (index lookup), SORT (ORDER BY without index). Helps understand query performance and optimize with indexes.', `## EXPLAIN

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
		['Parse /proc filesystem', 'Read Linux /proc entries: /proc/[pid]/cmdline contains null-separated command line arguments, /proc/[pid]/exe symlink points to executable path, /proc/[pid]/stat has process state and times, /proc/[pid]/status has readable UID/GID and memory info.', `## /proc Parsing

For each PID in /proc/:
- /proc/[pid]/cmdline: command line
- /proc/[pid]/exe: executable path
- /proc/[pid]/stat: process state
- /proc/[pid]/status: detailed info`],
		['Implement process enumeration', 'List all processes: read /proc directory, filter entries that are numeric (PIDs). For each PID, attempt to read cmdline and status. Handle race conditions where process exits between directory listing and file reading. Skip kernel threads.', `## Process List

\`\`\`go
files, _ := ioutil.ReadDir("/proc")
for _, f := range files {
    if pid, err := strconv.Atoi(f.Name()); err == nil {
        // process PID
    }
}
\`\`\``],
		['Build delta detection', 'Detect process spawns and exits: maintain set of known PIDs, compare each scan to previous. New PIDs are spawned processes (log immediately), missing PIDs are exited processes. Sort by spawn time (from /proc/[pid]/stat) to show chronologically.', `## Delta Detection

Compare process lists:
- New PIDs = spawned processes
- Missing PIDs = exited processes
- Track spawn time for ordering`],
		['Add command line extraction', 'Get full command: read /proc/[pid]/cmdline, replace null bytes (argument separators) with spaces. Some processes (kernel threads) have empty cmdline - read comm or exe instead. Handle permission denied errors for other users processes.', `## Command Line

Read /proc/[pid]/cmdline
Replace null bytes with spaces
Handle processes with empty cmdline`],
		['Implement polling loop', 'Continuous monitoring loop: poll every 100ms to catch short-lived processes (cron jobs, scripts). Use time.Ticker for consistent intervals. Balance frequency vs CPU usage. Consider adaptive polling (faster when activity detected).', `## Polling

\`\`\`go
ticker := time.NewTicker(100 * time.Millisecond)
for range ticker.C {
    detectNewProcesses()
}
\`\`\``],
		['Add UID/GID tracking', 'Identify process owner: parse /proc/[pid]/status for Uid line (real, effective, saved, filesystem UIDs) and Gid line. Look up username from /etc/passwd. Highlight processes run as root (UID 0) - especially interesting for privilege escalation.', `## User Tracking

From /proc/[pid]/status:
Uid: real effective saved fs
Gid: real effective saved fs`],
	]);

	addModuleWithTasks(pspyPath.id, 'Week 2: File System Events', 'inotify for file access', 1, [
		['Set up inotify watches', 'Use Linux inotify API: InotifyInit() creates instance, InotifyAddWatch() registers directory with event mask (IN_CREATE, IN_MODIFY, IN_DELETE, IN_OPEN). Read from fd to receive event structs with filename and event type.', `## inotify

\`\`\`go
fd, _ := syscall.InotifyInit()
syscall.InotifyAddWatch(fd, "/tmp", syscall.IN_CREATE|IN_MODIFY)
\`\`\``],
		['Watch key directories', 'Monitor security-relevant paths: /tmp and /var/tmp (world-writable), /dev/shm (RAM-based, often used for malware), cron directories (/etc/cron.d, /var/spool/cron), user home directories for config changes. Add watches to each.', `## Directories to Watch

- /tmp, /var/tmp
- /dev/shm
- Cron directories
- User home directories`],
		['Handle inotify events', 'Process event stream: read events from inotify fd (blocking read), parse struct (wd, mask, cookie, filename). IN_CREATE = new file, IN_DELETE = removed, IN_MODIFY = content changed, IN_OPEN = file accessed. Log with timestamp and path.', `## Event Handling

Events:
- IN_CREATE: file created
- IN_DELETE: file deleted
- IN_MODIFY: file modified
- IN_OPEN: file opened`],
		['Correlate files to processes', 'Link file events to processes: when file event occurs, scan /proc/*/fd/* for symlinks to that file to find which process has it open. Match event timing with recently spawned processes. Challenging for short-lived file access.', `## Correlation

When file event occurs:
- Check /proc/*/fd/* for open handles
- Check fuser output
- Match timing with process spawns`],
		['Add recursive watching', 'Monitor subdirectories: inotify watches are not recursive by default. Watch for IN_CREATE of directories, add new watch on created directories. Handle watch descriptor limit (default 8192) - may need to increase via /proc/sys/fs/inotify/max_user_watches.', `## Recursive Watch

inotify doesn't recurse automatically
Add watches for new directories
Handle watch limit`],
		['Implement event filtering', 'Reduce noise in output: ignore known noisy processes (systemd, snapd, journal), filter by file patterns (ignore .swp editor files), configurable exclusion lists. Show only security-relevant events to focus attention on suspicious activity.', `## Filtering

Ignore:
- Known noisy processes
- Specific file patterns
- Configurable exclusions`],
	]);

	addModuleWithTasks(pspyPath.id, 'Week 3: Output & Stealth', 'Display and evasion', 2, [
		['Build colored output', 'Color-code events for quick scanning: red for root (UID=0) processes (potential privilege escalation), yellow for cron-triggered processes (scheduled tasks), green for user processes, blue for filesystem events. Use ANSI escape codes.', `## Colored Output

Color by type:
- Red: root processes
- Yellow: cron jobs
- Green: user processes
- Blue: file system events`],
		['Add timestamp formatting', 'Precise timestamps: show ISO format (2024-01-15 03:14:15) with milliseconds for ordering. Include UID and username. Format: timestamp | UID=N (username) | command. Align columns for readability. Optional relative timestamps (seconds ago).', `## Timestamps

2024-01-15 03:14:15 | UID=0 | /usr/sbin/cron
2024-01-15 03:14:15 | UID=0 | /bin/sh -c /root/backup.sh`],
		['Implement log output', 'Save output to file: -o flag writes to specified file. Append mode for long monitoring sessions. Consider log rotation for multi-day runs. Strip ANSI color codes when writing to file (or keep with -c flag).', `## Logging

-o output.log: write to file
Append mode for long runs
Rotate large logs`],
		['Minimize footprint', 'Stealth operation: compile as static binary (no dynamic library dependencies), minimize memory allocations, keep CPU usage under 1%, avoid disk writes (no temp files). Dont show in obvious places (rename binary). Single binary deployment.', `## Stealth

- Static binary (no deps)
- Minimal memory usage
- Low CPU impact
- No disk writes by default`],
		['Add network connection tracking', 'Monitor network activity: parse /proc/[pid]/net/tcp and /proc/[pid]/net/udp for connections. Detect new listening ports. Track outbound connections (potential C2 callbacks). Correlate with process spawns to identify what initiated connection.', `## Network Tracking

Parse /proc/[pid]/net/*
Detect new connections
Track listening ports`],
		['Build summary statistics', 'Aggregate insights after monitoring: most frequently run commands (identify patterns), most active users, process spawn rate over time (detect bursts), file access patterns (which directories most active). Display on exit or signal.', `## Statistics

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
