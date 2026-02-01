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

// Expand all remaining paths with <10 tasks
const needsExpansion = [
	// Reimplement: Authentication Coercion (67)
	{ id: 67, modules: [
		{ name: 'Week 1-2: Coercion Techniques', desc: 'Force authentication', tasks: [
			['Implement PrinterBug/SpoolSample', 'Printer service coercion', '## PrinterBug\n\nMS-RPRN RpcRemoteFindFirstPrinterChangeNotification\nForce target to auth to attacker\nWorks on any domain-joined machine'],
			['Build PetitPotam', 'EFS coercion', '## PetitPotam\n\nMS-EFSRPC EfsRpcOpenFileRaw\nCoerce auth without credentials\nEffective against DCs'],
			['Add DFSCoerce', 'DFS coercion', '## DFSCoerce\n\nMS-DFSNM NetrDfsAddStdRoot\nAnother coercion vector\nWorks when others patched'],
			['Implement ShadowCoerce', 'VSS coercion', '## ShadowCoerce\n\nMS-FSRVP IsPathShadowCopied\nVolume Shadow Copy abuse\nRequires specific config'],
			['Build coercion detector', 'Check for vulns', '## Detection\n\nCheck which coercion methods work\nTest without triggering alerts\nReport vulnerable services'],
			['Add relay integration', 'Combine with relay', '## Integration\n\nCoerce → NTLM auth → Relay\nTo LDAP, SMB, HTTP, AD CS\nComplete attack chain'],
		] as [string, string, string][] },
	]},

	// Reimplement: BloodHound Collector (64)
	{ id: 64, modules: [
		{ name: 'Week 1-2: Data Collection', desc: 'Gather AD information', tasks: [
			['Query domain users', 'User enumeration', '## Users\n\nLDAP query all users\nAttributes: name, SID, UAC flags\nGroup memberships'],
			['Enumerate groups', 'Group membership', '## Groups\n\nAll domain groups\nNested membership resolution\nHighlight privileged groups'],
			['Collect computer objects', 'Machine accounts', '## Computers\n\nAll domain computers\nOS versions\nDelegation settings'],
			['Query ACLs', 'Permission enumeration', '## ACLs\n\nSecurity descriptors\nDACL parsing\nIdentify dangerous permissions'],
			['Enumerate sessions', 'Who logged where', '## Sessions\n\nNetSessionEnum\nNetWkstaUserEnum\nMap user-to-machine'],
			['Collect GPO data', 'Group Policy', '## GPOs\n\nAll GPOs and links\nWhere they apply\nSettings that affect security'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Output & Analysis', desc: 'Format for BloodHound', tasks: [
			['Build JSON output', 'BloodHound format', '## JSON Format\n\nSeparate files:\n- users.json\n- computers.json\n- groups.json\n- domains.json'],
			['Implement edge detection', 'Attack paths', '## Edges\n\nAdminTo\nCanRDP\nMemberOf\nGenericAll\nWriteDacl'],
			['Add stealth options', 'Avoid detection', '## Stealth\n\n- Throttle queries\n- Avoid certain queries\n- Use existing session'],
			['Build local collection', 'Local admin enum', '## Local Groups\n\nWho is local admin\nWho can RDP\nWho can PS remote'],
			['Implement trust enum', 'Domain trusts', '## Trusts\n\nTrust relationships\nTrust direction\nSID filtering status'],
			['Add certificate template enum', 'AD CS', '## Cert Templates\n\nVulnerable templates\nCA servers\nEnrollment permissions'],
		] as [string, string, string][] },
	]},

	// Reimplement: Network Tunneling Tools (73)
	{ id: 73, modules: [
		{ name: 'Week 1-2: TCP Tunneling', desc: 'Basic tunneling', tasks: [
			['Build local port forward', 'Bind and connect', '## Local Forward\n\n-L 8080:target:80\nListen locally, connect to target\nThrough intermediate host'],
			['Implement remote port forward', 'Reverse tunnel', '## Remote Forward\n\n-R 8080:localhost:80\nListen on remote, connect back\nBypass NAT/firewall'],
			['Add dynamic port forward', 'SOCKS proxy', '## Dynamic\n\n-D 1080\nSOCKS proxy through tunnel\nRoute any traffic'],
			['Build connection multiplexing', 'Share tunnel', '## Mux\n\nSingle SSH connection\nMultiple forwards\nControl socket'],
			['Implement SSH tunneling', 'Full SSH support', '## SSH\n\n- Password auth\n- Key auth\n- ProxyJump chains'],
			['Add auto-reconnect', 'Maintain tunnel', '## Resilience\n\nDetect disconnect\nAuto reconnect\nExponential backoff'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Tunneling', desc: 'Evasion and protocols', tasks: [
			['Build DNS tunnel', 'Tunnel over DNS', '## DNS Tunnel\n\nEncode data in queries\nExfil via TXT records\nSlow but covert'],
			['Implement ICMP tunnel', 'Tunnel over ping', '## ICMP Tunnel\n\nPayload in ICMP data\nOften allowed through FW\nRequires raw sockets'],
			['Add HTTP tunnel', 'Web-based tunnel', '## HTTP Tunnel\n\nLong-poll or WebSocket\nLooks like web traffic\nCDN-able'],
			['Build double pivot', 'Chain tunnels', '## Pivot Chain\n\nHost A → Host B → Host C\nNested tunnels\nRoute through network'],
			['Implement VPN-like mode', 'Full network access', '## VPN Mode\n\nTUN interface\nRoute traffic through tunnel\nTransparent to apps'],
			['Add traffic obfuscation', 'Hide tunnel traffic', '## Obfuscation\n\nMake traffic look normal\nRandomize patterns\nMimic protocols'],
		] as [string, string, string][] },
	]},

	// Reimplement: Nuclei Template Scanner (70)
	{ id: 70, modules: [
		{ name: 'Week 1-2: Template Engine', desc: 'Parse and run templates', tasks: [
			['Parse YAML templates', 'Template format', '## Template Format\n\n```yaml\nid: cve-2021-xxxx\ninfo:\n  name: Vuln Name\n  severity: high\nrequests:\n  - method: GET\n    path: /vuln\n```'],
			['Build HTTP executor', 'Send requests', '## HTTP Executor\n\n- Custom headers\n- Body templates\n- Follow redirects\n- Handle cookies'],
			['Implement matchers', 'Detect vulnerabilities', '## Matchers\n\n- Status code\n- Body contains\n- Regex match\n- Word match\n- Binary match'],
			['Add extractors', 'Pull data from response', '## Extractors\n\n- Regex capture groups\n- JSON path\n- XPath\n- DSL expressions'],
			['Build template variables', 'Dynamic values', '## Variables\n\n{{BaseURL}}\n{{Host}}\n{{randstr}}\nCustom variables'],
			['Implement workflows', 'Conditional scanning', '## Workflows\n\nIf template A matches\nThen run template B\nConditional logic'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Scanner Features', desc: 'Full scanner', tasks: [
			['Build target management', 'Multiple targets', '## Targets\n\n- URL list\n- CIDR ranges\n- File input\n- Stdin'],
			['Add rate limiting', 'Control speed', '## Rate Limiting\n\n- Requests per second\n- Concurrent hosts\n- Per-host limits'],
			['Implement severity filtering', 'Focus on critical', '## Filtering\n\n- By severity\n- By tag\n- By ID\n- Include/exclude'],
			['Build output formats', 'Report results', '## Output\n\n- JSON\n- JSONL\n- Markdown\n- SARIF'],
			['Add headless browser', 'JavaScript rendering', '## Headless\n\nFor JS-heavy sites\nChromium integration\nDOM-based checks'],
			['Implement interactsh', 'OOB detection', '## OOB Testing\n\nCallback server\nDetect blind vulns\nSSRF, XXE, etc.'],
		] as [string, string, string][] },
	]},

	// Reimplement: Password Cracker (71)
	{ id: 71, modules: [
		{ name: 'Week 1-2: Hash Cracking', desc: 'Core cracking', tasks: [
			['Implement hash functions', 'MD5, SHA, bcrypt', '## Hash Functions\n\nMD5, SHA1, SHA256\nNTLM, NetNTLM\nbcrypt, scrypt'],
			['Build dictionary attack', 'Wordlist-based', '## Dictionary\n\nRead wordlist\nHash each word\nCompare to target'],
			['Add rule engine', 'Word mangling', '## Rules\n\n$1 - append 1\n^! - prepend !\nc - capitalize\nr - reverse'],
			['Implement mask attack', 'Pattern brute force', '## Masks\n\n?l = lowercase\n?u = uppercase\n?d = digit\n?s = special'],
			['Build combinator attack', 'Word combinations', '## Combinator\n\nword1 + word2\nFrom two wordlists\nGenerates combinations'],
			['Add hybrid attack', 'Wordlist + mask', '## Hybrid\n\npassword?d?d?d\nWordlist base + pattern'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Performance', desc: 'Optimization', tasks: [
			['Implement multi-threading', 'Parallel cracking', '## Threading\n\nPartition keyspace\nWorker threads\nThread-safe queue'],
			['Add GPU support', 'CUDA/OpenCL', '## GPU\n\nMassively parallel\nThousands of cores\nMuch faster than CPU'],
			['Build resume support', 'Save progress', '## Resume\n\nCheckpoint state\nResume interrupted attacks\nTrack progress'],
			['Implement potfile', 'Store cracked hashes', '## Potfile\n\nhash:plaintext\nDon\'t recrack known\nShare across sessions'],
			['Add distributed mode', 'Multiple machines', '## Distributed\n\nSplit keyspace\nCoordinate workers\nAggregate results'],
			['Build benchmarking', 'Speed testing', '## Benchmark\n\nHashes per second\nPer hash type\nPer attack mode'],
		] as [string, string, string][] },
	]},

	// Reimplement: Privilege Escalation Enumeration (72)
	{ id: 72, modules: [
		{ name: 'Week 1-2: Linux Enumeration', desc: 'Linux privesc checks', tasks: [
			['Check SUID binaries', 'SUID abuse', '## SUID\n\n```bash\nfind / -perm -4000 2>/dev/null\n```\n\nCheck GTFOBins for exploits'],
			['Enumerate capabilities', 'Linux capabilities', '## Capabilities\n\n```bash\ngetcap -r / 2>/dev/null\n```\n\nCAP_SETUID, CAP_NET_RAW, etc.'],
			['Check sudo permissions', 'sudo -l', '## Sudo\n\nWhat can current user sudo?\nNOPASSWD entries\nWildcard abuse'],
			['Find writable paths', 'PATH hijacking', '## PATH\n\nWritable dirs in PATH?\nCron jobs with relative paths\nHijack commands'],
			['Enumerate cron jobs', 'Scheduled tasks', '## Cron\n\n/etc/crontab\n/etc/cron.d/*\n/var/spool/cron/\nWritable scripts?'],
			['Check kernel version', 'Kernel exploits', '## Kernel\n\nuname -a\nSearch for CVEs\nDirty COW, etc.'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Windows Enumeration', desc: 'Windows privesc checks', tasks: [
			['Check service permissions', 'Service abuse', '## Services\n\nUnquoted paths\nWeak service permissions\nWritable service binaries'],
			['Enumerate scheduled tasks', 'Task scheduler', '## Tasks\n\nschtasks /query\nWritable task files\nHigh-priv tasks'],
			['Check AlwaysInstallElevated', 'MSI abuse', '## MSI\n\nRegistry key check\nInstall MSI as SYSTEM\nEasy privesc'],
			['Find stored credentials', 'Credential hunting', '## Credentials\n\ncmdkey /list\nWireless passwords\nBrowser passwords\nRegistry autologon'],
			['Check token privileges', 'Token abuse', '## Tokens\n\nSeImpersonate → Potato\nSeBackup → SAM dump\nSeDebug → injection'],
			['Enumerate installed software', 'Vulnerable apps', '## Software\n\nFind installed versions\nCheck for CVEs\nWeak folder permissions'],
		] as [string, string, string][] },
	]},

	// Reimplement: Web Fuzzer (69)
	{ id: 69, modules: [
		{ name: 'Week 1-2: Fuzzing Core', desc: 'Directory and parameter fuzzing', tasks: [
			['Build URL fuzzer', 'Directory enumeration', '## URL Fuzzing\n\nhttp://target/FUZZ\nReplace FUZZ with wordlist\nFilter by status code'],
			['Implement parameter fuzzer', 'Query param fuzzing', '## Params\n\n?FUZZ=value\n?param=FUZZ\nDiscover hidden params'],
			['Add header fuzzing', 'HTTP header testing', '## Headers\n\nFuzz header names\nFuzz header values\nFind hidden functionality'],
			['Build response filtering', 'Reduce noise', '## Filtering\n\n- Filter by status\n- Filter by size\n- Filter by words\n- Filter by regex'],
			['Implement recursion', 'Follow discovered paths', '## Recursion\n\nFound /admin/\nFuzz /admin/FUZZ\nConfigurable depth'],
			['Add rate limiting', 'Control speed', '## Rate Limit\n\n- Requests per second\n- Concurrent connections\n- Per-host limits'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Features', desc: 'Full fuzzer', tasks: [
			['Build multiple wordlist support', 'Clusterbomb mode', '## Multi-wordlist\n\n/FUZZ1/FUZZ2\nTwo wordlists\nAll combinations'],
			['Implement POST fuzzing', 'Body fuzzing', '## POST\n\n-d "user=FUZZ&pass=FUZZ"\nJSON body fuzzing\nFile upload fuzzing'],
			['Add output formats', 'Save results', '## Output\n\n- JSON\n- CSV\n- HTML report'],
			['Build matcher system', 'Positive matching', '## Matchers\n\nMatch on status, size, regex\nInverse of filters\nFind specific responses'],
			['Implement auto-calibration', 'Smart filtering', '## Calibration\n\nSend baseline requests\nAuto-detect response patterns\nFilter dynamically'],
			['Add extension scanning', 'File extension fuzzing', '## Extensions\n\n/file.FUZZ\n.php, .asp, .bak, .old\nCombine with filenames'],
		] as [string, string, string][] },
	]},

	// Reimplement: ntlmrelayx (66)
	{ id: 66, modules: [
		{ name: 'Week 1-2: NTLM Relay Core', desc: 'Relay mechanics', tasks: [
			['Implement NTLM server', 'Accept auth', '## NTLM Server\n\nSMB, HTTP, LDAP server\nSend Type 2 challenge from target\nReceive Type 3, forward'],
			['Build SMB relay', 'Relay to SMB', '## SMB Relay\n\nCan\'t relay to same machine\nRelay to other servers\nExecute via SMBExec/Atexec'],
			['Add LDAP relay', 'Relay to LDAP', '## LDAP Relay\n\nAdd computer account\nModify ACLs\nResource-based delegation'],
			['Implement HTTP relay', 'Relay to HTTP', '## HTTP Relay\n\nAD CS web enrollment\nExchange (EWS, OWA)\nSharePoint'],
			['Build multi-relay', 'Spray to targets', '## Multi-Relay\n\nRelay to multiple targets\nTry each one\nFirst success wins'],
			['Add signing detection', 'Check SMB signing', '## Signing\n\nSMB signing = no relay\nLDAP signing = no relay\nDetect before trying'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Attack Chains', desc: 'Post-relay actions', tasks: [
			['Implement secretsdump', 'Dump creds via relay', '## Secrets\n\nRelay admin auth\nDump SAM, LSA, NTDS\nNo credentials needed'],
			['Build RBCD attack', 'Delegation abuse', '## RBCD\n\nRelay to LDAP\nSet msDS-AllowedToActOnBehalfOfOtherIdentity\nS4U2Self/S4U2Proxy'],
			['Add shadow credentials', 'Key credential attack', '## Shadow Creds\n\nRelay to LDAP\nAdd KeyCredentialLink\nPKINIT authentication'],
			['Implement AD CS attack', 'ESC8', '## ESC8\n\nRelay to CA web enrollment\nRequest cert for victim\nAuth as victim'],
			['Build SOCKs proxy', 'Pivot via relay', '## SOCKS\n\nMaintain relayed session\nExpose as SOCKS proxy\nRoute tools through'],
			['Add loot capture', 'Harvest data', '## Loot\n\nCapture NTLMv2 hashes\nSave for offline cracking\nLog all attempts'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Go (55)
	{ id: 55, modules: [
		{ name: 'Week 1-2: Go Fundamentals', desc: 'Go for security tools', tasks: [
			['Set up Go environment', 'Toolchain setup', '## Setup\n\nInstall Go\nSet GOPATH\ngo mod init'],
			['Learn concurrency', 'Goroutines and channels', '## Concurrency\n\ngo func() { ... }()\nchan for communication\nselect for multiplexing'],
			['Implement network operations', 'net package', '## Networking\n\nnet.Dial, net.Listen\nTCP/UDP sockets\nHTTP client/server'],
			['Build process execution', 'os/exec package', '## Execution\n\nexec.Command()\nCapture output\nPipes'],
			['Add file operations', 'os package', '## Files\n\nos.Open, os.Create\nDirectory walking\nPermission handling'],
			['Implement cross-compilation', 'Multi-platform builds', '## Cross-compile\n\nGOOS=windows GOARCH=amd64\nStatic binaries\nCGO considerations'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Build tools in Go', tasks: [
			['Build port scanner', 'Concurrent scanning', '## Scanner\n\nConcurrent connections\nService detection\nBanner grabbing'],
			['Implement reverse shell', 'Basic implant', '## Reverse Shell\n\nConnect back\nExecute commands\nReturn output'],
			['Add HTTP C2 client', 'Beacon in Go', '## C2 Client\n\nPeriodic checkin\nReceive tasks\nReturn results'],
			['Build file exfiltrator', 'Data transfer', '## Exfil\n\nFind interesting files\nCompress and encode\nUpload to C2'],
			['Implement keylogger', 'Input capture', '## Keylogger\n\nHook keyboard (Windows)\n/dev/input (Linux)\nLog keystrokes'],
			['Add persistence', 'Maintain access', '## Persistence\n\nStartup registration\nScheduled tasks\nService creation'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Python (56)
	{ id: 56, modules: [
		{ name: 'Week 1-2: Python Security Basics', desc: 'Python for security', tasks: [
			['Network programming', 'Socket operations', '## Sockets\n\n```python\nimport socket\ns = socket.socket()\ns.connect((host, port))\n```'],
			['HTTP operations', 'Requests library', '## HTTP\n\nrequests.get/post\nSession handling\nProxy support'],
			['Process execution', 'subprocess module', '## Subprocess\n\n```python\nsubprocess.run([\'cmd\'], capture_output=True)\n```'],
			['File operations', 'os and pathlib', '## Files\n\nPath handling\nWalking directories\nReading/writing'],
			['Implement threading', 'Concurrent operations', '## Threading\n\nThreadPoolExecutor\nQueue for tasks\nThread-safe operations'],
			['Add argument parsing', 'CLI interfaces', '## Argparse\n\n```python\nparser = argparse.ArgumentParser()\nparser.add_argument(\'-t\', \'--target\')\n```'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Python tools', tasks: [
			['Build web scanner', 'Dir/file bruteforce', '## Scanner\n\nConcurrent requests\nFilter responses\nFollow redirects'],
			['Implement password sprayer', 'Credential testing', '## Sprayer\n\nMultiple protocols\nLockout awareness\nRandom delays'],
			['Add OSINT tool', 'Information gathering', '## OSINT\n\nSubdomain enum\nEmail harvesting\nSocial media lookup'],
			['Build phishing toolkit', 'Campaign tools', '## Phishing\n\nClone pages\nTrack clicks\nCapture credentials'],
			['Implement log analyzer', 'Parse and search', '## Logs\n\nParse formats\nExtract IOCs\nTimeline analysis'],
			['Add automation framework', 'Chain tools', '## Automation\n\nPipeline tools\nError handling\nReporting'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Rust (54)
	{ id: 54, modules: [
		{ name: 'Week 1-2: Rust Fundamentals', desc: 'Rust for security', tasks: [
			['Set up Rust environment', 'Toolchain', '## Setup\n\nrustup install\ncargo new project\nRust analyzer'],
			['Learn ownership', 'Memory safety', '## Ownership\n\nOwnership rules\nBorrowing\nLifetimes'],
			['Implement networking', 'std::net', '## Networking\n\nTcpListener\nTcpStream\nasync with tokio'],
			['Process execution', 'Command API', '## Command\n\n```rust\nCommand::new(\"cmd\")\n  .arg(\"/c\")\n  .arg(\"dir\")\n  .output()\n```'],
			['Add error handling', 'Result and Option', '## Errors\n\nResult<T, E>\nOption<T>\n? operator'],
			['Build CLI interface', 'Clap library', '## Clap\n\n```rust\n#[derive(Parser)]\nstruct Args {\n  #[arg(short, long)]\n  target: String,\n}\n```'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Rust tools', tasks: [
			['Build port scanner', 'Async scanning', '## Scanner\n\nTokio async\nConcurrent connections\nTimeout handling'],
			['Implement implant', 'Rust payload', '## Implant\n\nSmall binary\nAnti-analysis\nCross-platform'],
			['Add file operations', 'System interaction', '## Files\n\nstd::fs\nwalkdir crate\nPermission handling'],
			['Build cryptographic tools', 'Encryption', '## Crypto\n\nring or rust-crypto\nAES encryption\nKey derivation'],
			['Implement shellcode loader', 'Memory execution', '## Shellcode\n\nVirtualAlloc (Windows)\nmmap (Linux)\nExecute in memory'],
			['Add evasion techniques', 'AV bypass', '## Evasion\n\nString encryption\nAPI hashing\nSyscalls'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: C# (53)
	{ id: 53, modules: [
		{ name: 'Week 1-2: C# Fundamentals', desc: 'C# for offense', tasks: [
			['Set up .NET environment', '.NET SDK', '## Setup\n\ndotnet new console\ndotnet build\nCross-platform'],
			['Learn P/Invoke', 'Windows API calls', '## P/Invoke\n\n```csharp\n[DllImport(\"kernel32.dll\")]\nstatic extern IntPtr VirtualAlloc(...);\n```'],
			['Implement process operations', 'System.Diagnostics', '## Processes\n\nProcess.Start()\nProcess enumeration\nProcess injection'],
			['Add registry operations', 'Windows Registry', '## Registry\n\nRegistry.GetValue()\nRegistry.SetValue()\nPersistence'],
			['Build WMI queries', 'Management classes', '## WMI\n\n```csharp\nManagementObjectSearcher\nQuery system info\nRemote execution\n```'],
			['Implement reflection', 'Dynamic loading', '## Reflection\n\nAssembly.Load()\nType.GetMethod()\nDynamic invocation'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Offensive Tools', desc: 'C# tools', tasks: [
			['Build process injector', 'Code injection', '## Injection\n\nVirtualAllocEx\nWriteProcessMemory\nCreateRemoteThread'],
			['Implement credential dumper', 'LSASS interaction', '## Creds\n\nMiniDumpWriteDump\nParse dump\nExtract hashes'],
			['Add lateral movement', 'Remote execution', '## Lateral\n\nWMI execution\nService creation\nPSExec-style'],
			['Build persistence tool', 'Maintain access', '## Persistence\n\nRegistry run keys\nScheduled tasks\nServices'],
			['Implement AD enumeration', 'LDAP queries', '## AD Enum\n\nDirectorySearcher\nQuery users, groups\nFind attack paths'],
			['Add execute-assembly', 'In-memory execution', '## Execute\n\nLoad .NET assembly\nInvoke entry point\nNo disk write'],
		] as [string, string, string][] },
	]},

	// Reimplement: Nmap (68)
	{ id: 68, modules: [
		{ name: 'Week 1-2: Port Scanning', desc: 'Scan techniques', tasks: [
			['Implement TCP connect scan', 'Full TCP handshake', '## TCP Connect\n\nComplete 3-way handshake\nReliable detection\nLogged by target'],
			['Build SYN scan', 'Half-open scan', '## SYN Scan\n\nSend SYN\nRST = closed, SYN-ACK = open\nDon\'t complete handshake'],
			['Add UDP scan', 'UDP port detection', '## UDP Scan\n\nSend UDP packet\nICMP unreachable = closed\nNo response = open|filtered'],
			['Implement service detection', 'Banner grabbing', '## Service Detection\n\nConnect to open ports\nSend probes\nMatch signatures'],
			['Build OS detection', 'TCP/IP fingerprint', '## OS Detection\n\nTTL values\nTCP window size\nOption order'],
			['Add timing controls', 'Scan speed', '## Timing\n\n-T0 to -T5\nParallelism\nTimeouts'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Features', desc: 'Full scanner', tasks: [
			['Implement host discovery', 'Find live hosts', '## Discovery\n\nPing scan\nARP scan (local)\nTCP/UDP probes'],
			['Build script engine', 'NSE equivalent', '## Scripts\n\nVuln detection\nBrute force\nService enumeration'],
			['Add output formats', 'Report results', '## Output\n\n- Normal\n- XML\n- Grepable\n- JSON'],
			['Implement evasion', 'IDS bypass', '## Evasion\n\nDecoy scans\nFragmentation\nSource port manipulation'],
			['Build target specification', 'Input handling', '## Targets\n\nCIDR notation\nIP ranges\nHostname resolution'],
			['Add IPv6 support', 'IPv6 scanning', '## IPv6\n\nFull IPv6 support\nScope handling\nNeighbor discovery'],
		] as [string, string, string][] },
	]},

	// Reimplement: Responder (65)
	{ id: 65, modules: [
		{ name: 'Week 1-2: Protocol Poisoning', desc: 'LLMNR/NBT-NS/mDNS', tasks: [
			['Implement LLMNR responder', 'Link-Local Multicast', '## LLMNR\n\nListen on 224.0.0.252:5355\nRespond to name queries\nProvide attacker IP'],
			['Build NBT-NS responder', 'NetBIOS Name Service', '## NBT-NS\n\nListen on UDP 137\nRespond to broadcasts\nSimilar to LLMNR'],
			['Add mDNS responder', 'Multicast DNS', '## mDNS\n\nListen on 224.0.0.251:5353\nApple/Linux name resolution\nPoison queries'],
			['Implement WPAD responder', 'Proxy auto-config', '## WPAD\n\nProxy config poisoning\nCapture HTTP traffic\nCredential theft'],
			['Build DHCPv6 server', 'IPv6 config', '## DHCPv6\n\nRogue DNS server\nMITM IPv6 traffic'],
			['Add DNS poisoning', 'Rogue DNS', '## DNS\n\nRespond to DNS queries\nPoint to attacker\nCapture traffic'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Authentication Capture', desc: 'Credential harvesting', tasks: [
			['Implement SMB server', 'Capture NTLM', '## SMB\n\nFake SMB server\nCapture NTLMv2 hashes\nLog credentials'],
			['Build HTTP server', 'Capture HTTP auth', '## HTTP\n\nRequest NTLM auth\nCapture hashes\nBasic auth capture'],
			['Add FTP server', 'FTP credentials', '## FTP\n\nRogue FTP server\nCapture plaintext creds'],
			['Implement LDAP server', 'Directory auth', '## LDAP\n\nFake LDAP server\nSimple bind = plaintext\nNTLM bind = hash'],
			['Build MSSQL server', 'Database auth', '## MSSQL\n\nFake SQL server\nCapture credentials\nNTLM hashes'],
			['Add analysis mode', 'Passive capture', '## Analysis\n\nNo poisoning\nJust capture broadcasts\nSafe reconnaissance'],
		] as [string, string, string][] },
	]},

	// Remaining paths with 4-6 tasks - expand with more modules
	// Evasion & Payload Tools (86)
	{ id: 86, modules: [
		{ name: 'Shellcode Development', desc: 'Custom shellcode', tasks: [
			['Write position-independent code', 'PIC fundamentals', '## PIC\n\nNo hardcoded addresses\nRelative addressing\nDynamic resolution'],
			['Implement API hashing', 'Resolve APIs dynamically', '## API Hash\n\nHash function names\nWalk PEB/EAT\nFind by hash'],
			['Build syscall stubs', 'Direct syscalls', '## Syscalls\n\nAvoid ntdll hooks\nDirect kernel calls\nSyswhispers approach'],
			['Add shellcode encoder', 'Basic obfuscation', '## Encoding\n\nXOR encoding\nCustom encoding\nDecoder stub'],
			['Implement shellcode loader', 'Execution methods', '## Loaders\n\nVirtualAlloc + memcpy\nCallback execution\nModule stomping'],
			['Build egg hunter', 'Find shellcode in memory', '## Egg Hunter\n\nSmall stub finds larger payload\nUseful for small buffer overflow'],
		] as [string, string, string][] },
		{ name: 'Payload Delivery', desc: 'Delivery mechanisms', tasks: [
			['Build dropper', 'Download and execute', '## Dropper\n\nFetch payload\nWrite and execute\nOr execute in memory'],
			['Implement stager', 'Multi-stage payload', '## Stager\n\nSmall first stage\nDownloads full payload\nMinimal detection surface'],
			['Add Office macros', 'Document-based delivery', '## Macros\n\nVBA download\nPowerShell execution\nAMSI bypass'],
			['Build HTA payload', 'HTML Application', '## HTA\n\nmshta.exe execution\nJScript/VBScript\nNo Mark-of-the-Web'],
			['Implement LNK payload', 'Shortcut file', '## LNK\n\nMalicious shortcut\nIcon spoofing\nCommand execution'],
			['Add ISO/IMG delivery', 'Container bypass', '## ISO\n\nBypass Mark-of-the-Web\nAuto-mount on Windows\nHide payload inside'],
		] as [string, string, string][] },
	]},

	// Exploit Development Tools (85)
	{ id: 85, modules: [
		{ name: 'Memory Corruption', desc: 'Buffer overflow basics', tasks: [
			['Understand stack layout', 'Stack fundamentals', '## Stack\n\nReturn address\nSaved frame pointer\nLocal variables\nFunction arguments'],
			['Implement fuzzer', 'Find crashes', '## Fuzzing\n\nGenerate test cases\nMonitor for crashes\nTriaging crashes'],
			['Build pattern generator', 'Find offset', '## Pattern\n\nUnique pattern generation\nFind EIP offset\ncyclic pattern'],
			['Control EIP/RIP', 'Redirect execution', '## Control Flow\n\nOverwrite return address\nJump to shellcode\nROP gadgets'],
			['Implement ROP chain', 'Return-oriented programming', '## ROP\n\nFind gadgets\nChain to disable DEP\nCall VirtualProtect'],
			['Add heap exploitation', 'Heap overflow', '## Heap\n\nHeap metadata corruption\nUse-after-free\nHeap spraying'],
		] as [string, string, string][] },
		{ name: 'Modern Mitigations', desc: 'Bypass protections', tasks: [
			['Bypass ASLR', 'Address randomization', '## ASLR\n\nInfo leak\nPartial overwrite\nJIT spray'],
			['Bypass DEP/NX', 'Non-executable memory', '## DEP\n\nROP to VirtualProtect\nROP to mprotect\nReturn-to-libc'],
			['Bypass stack canaries', 'Stack cookies', '## Canaries\n\nInfo leak\nBrute force (fork)\nOverwrite handler'],
			['Bypass CFG', 'Control flow guard', '## CFG\n\nFind allowed targets\nUse valid call sites\nAbuse CFG bitmap'],
			['Implement info leak', 'Defeat ASLR', '## Info Leak\n\nFormat string\nOut-of-bounds read\nUse-after-free'],
			['Build exploit framework', 'Reliable exploitation', '## Framework\n\nTarget detection\nPayload selection\nError handling'],
		] as [string, string, string][] },
	]},
];

// Expand all paths
for (const path of needsExpansion) {
	console.log(`Expanding: ${path.id}`);
	expandPath(path.id, path.modules);
}

console.log('Done!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
