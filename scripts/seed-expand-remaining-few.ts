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
		['Build network scanner', 'Host discovery', '## Scanner\n\nPing sweep\nARP scan\nTCP connect scan'],
		['Implement service detection', 'Banner grabbing', '## Services\n\nConnect to ports\nSend probes\nIdentify services'],
		['Add DNS enumeration', 'Zone transfers', '## DNS\n\nSubdomain brute\nZone transfer attempt\nDNS record lookup'],
		['Build SNMP scanner', 'Community string testing', '## SNMP\n\nTest default communities\nWalk MIB tree\nExtract system info'],
		['Implement SMB enumeration', 'Shares and users', '## SMB\n\nList shares\nNull session enum\nUser enumeration'],
		['Add LDAP enumeration', 'Directory queries', '## LDAP\n\nAnonymous bind\nUser/group enum\nGPO enumeration'],
	] as [string, string, string][] },
	{ name: 'Network Attacks', desc: 'Exploitation tools', tasks: [
		['Build ARP spoofer', 'MITM attack', '## ARP Spoof\n\nPoison ARP cache\nIntercept traffic\nForward packets'],
		['Implement packet capture', 'Traffic analysis', '## Capture\n\nPcap library\nFilter traffic\nExtract credentials'],
		['Add password sprayer', 'Credential testing', '## Sprayer\n\nSMB spray\nLDAP spray\nKerberos spray'],
		['Build relay tool', 'NTLM relay', '## Relay\n\nCapture NTLM\nRelay to target\nExecute commands'],
		['Implement coercer', 'Force authentication', '## Coercer\n\nPrinterbug\nPetitPotam\nDFSCoerce'],
		['Add Kerberos attacks', 'AS-REP/Kerberoast', '## Kerberos\n\nRequest TGTs\nCrack offline\nForge tickets'],
	] as [string, string, string][] },
]);

// Reimplement: Evil-WinRM & C2 (89)
expandPath(89, [
	{ name: 'WinRM Client', desc: 'PowerShell remoting', tasks: [
		['Implement WinRM protocol', 'HTTP/SOAP transport', '## WinRM\n\nHTTP on 5985\nHTTPS on 5986\nSOAP envelope format'],
		['Build shell interface', 'Interactive PS', '## Shell\n\nSend commands\nReceive output\nHandle errors'],
		['Add file transfer', 'Upload/download', '## Files\n\nBase64 encoding\nChunk large files\nStream transfer'],
		['Implement pass-the-hash', 'NTLM authentication', '## PtH\n\nAuth with hash\nNo password needed'],
		['Build script execution', 'Run PS scripts', '## Scripts\n\nLoad scripts\nBypass execution policy\nIn-memory execution'],
		['Add Kerberos auth', 'Ticket-based auth', '## Kerberos\n\nUse TGT/TGS\nPass-the-ticket\nS4U delegation'],
	] as [string, string, string][] },
	{ name: 'C2 Features', desc: 'Command and control', tasks: [
		['Build beacon client', 'Periodic checkin', '## Beacon\n\nSleep interval\nJitter\nReceive tasks'],
		['Implement task queue', 'Command queuing', '## Tasks\n\nQueue commands\nTrack execution\nReturn results'],
		['Add process injection', 'Code injection', '## Injection\n\nInject into process\nHide execution\nBypass detection'],
		['Build lateral movement', 'Spread access', '## Lateral\n\nWMI execution\nSMB execution\nPSRemoting'],
		['Implement persistence', 'Maintain access', '## Persistence\n\nRegistry\nScheduled tasks\nServices'],
		['Add credential harvesting', 'Extract creds', '## Creds\n\nMimikatz-style\nLSASS dump\nBrowser passwords'],
	] as [string, string, string][] },
]);

// Reimplement: Impacket Suite (87 - the second one)
expandPath(87, [
	{ name: 'Protocol Implementations', desc: 'Core protocols', tasks: [
		['Implement SMB2/3 client', 'Modern SMB', '## SMB\n\nNegotiation\nSession setup\nTree connect'],
		['Build MSRPC layer', 'DCE/RPC over SMB', '## MSRPC\n\nNamed pipe transport\nInterface binding\nRequest/response'],
		['Add Kerberos client', 'AS/TGS requests', '## Kerberos\n\nAS-REQ/AS-REP\nTGS-REQ/TGS-REP\nEncryption types'],
		['Implement LDAP client', 'Directory access', '## LDAP\n\nBind operation\nSearch queries\nModify operations'],
		['Build NTLM client', 'Challenge-response', '## NTLM\n\nNegotiate\nChallenge\nAuthenticate'],
		['Add WMI client', 'Windows Management', '## WMI\n\nDCOM connection\nQuery classes\nMethod execution'],
	] as [string, string, string][] },
	{ name: 'Attack Tools', desc: 'Security tools', tasks: [
		['Build secretsdump', 'Credential extraction', '## Secrets\n\nRemote registry\nSAM dump\nLSA secrets'],
		['Implement psexec', 'Remote execution', '## PSExec\n\nService creation\nCommand execution\nOutput capture'],
		['Add wmiexec', 'WMI execution', '## WMI\n\nWin32_Process.Create\nOutput via share\nStealthy execution'],
		['Build smbclient', 'File operations', '## SMB Client\n\nList shares\nUpload/download\nDelete files'],
		['Implement GetNPUsers', 'AS-REP roasting', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack offline'],
		['Add GetUserSPNs', 'Kerberoasting', '## Kerberoast\n\nFind SPNs\nRequest TGS\nCrack service passwords'],
	] as [string, string, string][] },
]);

// Reimplement: Pivoting & C2 (84)
expandPath(84, [
	{ name: 'Pivoting Techniques', desc: 'Network pivoting', tasks: [
		['Build SOCKS proxy', 'Dynamic port forwarding', '## SOCKS\n\nSOCKS5 server\nAuth support\nRoute through pivot'],
		['Implement port forwarding', 'Static tunnels', '## Forwards\n\nLocal forward\nRemote forward\nDynamic forward'],
		['Add chisel-like tunnel', 'HTTP tunnel', '## HTTP Tunnel\n\nTunnel over HTTP\nFirewall bypass\nWebSocket support'],
		['Build multi-hop pivot', 'Chain pivots', '## Multi-Hop\n\nPivot through multiple hosts\nNested tunnels\nRoute management'],
		['Implement VPN-like mode', 'Full network access', '## VPN\n\nTUN interface\nRoute entire subnet\nTransparent access'],
		['Add DNS tunneling', 'Covert channel', '## DNS\n\nEncode in queries\nSlow but covert\nFirewall bypass'],
	] as [string, string, string][] },
	{ name: 'C2 Infrastructure', desc: 'Command and control', tasks: [
		['Design C2 protocol', 'Communication design', '## Protocol\n\nEncryption\nAuthentication\nCommand format'],
		['Build implant', 'Agent software', '## Implant\n\nTask execution\nResult return\nStealth features'],
		['Implement server', 'C2 server', '## Server\n\nListener management\nImplant tracking\nOperator interface'],
		['Add redirectors', 'Traffic redirection', '## Redirectors\n\nHide real C2\nDomain fronting\nHTTP redirects'],
		['Build payload generator', 'Create implants', '## Generator\n\nConfigure implant\nEmbed config\nObfuscation'],
		['Implement team collaboration', 'Multi-operator', '## Team\n\nShared sessions\nRole-based access\nEvent logging'],
	] as [string, string, string][] },
]);

// Reimplement: Rubeus (63)
expandPath(63, [
	{ name: 'Ticket Operations', desc: 'Kerberos ticket handling', tasks: [
		['Request TGT', 'AS-REQ operation', '## TGT Request\n\nAS-REQ with password\nAS-REQ with hash\nAS-REQ with certificate'],
		['Request TGS', 'TGS-REQ operation', '## TGS Request\n\nService ticket for SPN\nS4U2Self\nS4U2Proxy'],
		['Implement pass-the-ticket', 'Use tickets', '## PTT\n\nInject into memory\nUse for authentication\nCross-realm'],
		['Build ticket renewal', 'Extend lifetime', '## Renewal\n\nRenew before expiry\nMaintain access\nMaximum lifetime'],
		['Add ticket export', 'Save tickets', '## Export\n\nDump from memory\nSave to file\nKirbi format'],
		['Implement ticket parsing', 'Analyze tickets', '## Parse\n\nDecode ticket\nShow PAC info\nExpiry times'],
	] as [string, string, string][] },
	{ name: 'Kerberos Attacks', desc: 'Attack techniques', tasks: [
		['Build Kerberoasting', 'Crack service accounts', '## Kerberoast\n\nFind SPNs\nRequest TGS\nExtract for cracking'],
		['Implement AS-REP roasting', 'No preauth attack', '## AS-REP\n\nFind vulnerable users\nRequest AS-REP\nCrack password hash'],
		['Add overpass-the-hash', 'Hash to ticket', '## OverPTH\n\nNTLM hash to TGT\nUse for authentication\nStealth movement'],
		['Build unconstrained delegation', 'TGT theft', '## Unconstrained\n\nMonitor for TGTs\nCapture admin tickets\nImpersonate users'],
		['Implement constrained delegation', 'S4U abuse', '## Constrained\n\nS4U2Self for ticket\nS4U2Proxy to target\nAlternate service name'],
		['Add resource-based delegation', 'RBCD attack', '## RBCD\n\nModify msDS-AllowedToActOnBehalfOfOtherIdentity\nS4U chain\nPrivilege escalation'],
	] as [string, string, string][] },
]);

// Reimplement Red Team Tools: Web & AD (82)
expandPath(82, [
	{ name: 'Web Attack Tools', desc: 'Web application testing', tasks: [
		['Build directory bruteforcer', 'Content discovery', '## Dirbusting\n\nWordlist enumeration\nFilter responses\nRecursive scanning'],
		['Implement parameter fuzzer', 'Find hidden params', '## Param Fuzz\n\nFuzz parameter names\nFuzz values\nDetect vulns'],
		['Add XSS scanner', 'Cross-site scripting', '## XSS\n\nContext detection\nPayload generation\nReflection checking'],
		['Build SQLi scanner', 'SQL injection', '## SQLi\n\nError-based\nBlind boolean\nTime-based'],
		['Implement subdomain enum', 'Attack surface discovery', '## Subdomains\n\nDNS brute force\nCertificate transparency\nArchive search'],
		['Add credential sprayer', 'Web app spray', '## Spray\n\nLogin form spray\nOAuth spray\nRate limit handling'],
	] as [string, string, string][] },
	{ name: 'AD Attack Tools', desc: 'Active Directory attacks', tasks: [
		['Build BloodHound ingestor', 'AD enumeration', '## BloodHound\n\nLDAP queries\nSession enum\nACL collection'],
		['Implement DCSync', 'Credential theft', '## DCSync\n\nDRSGetNCChanges\nReplicate secrets\nNTDS extraction'],
		['Add GPO abuse', 'Group Policy attacks', '## GPO\n\nFind writable GPOs\nAdd scheduled task\nDeploy malware'],
		['Build delegation abuse', 'Kerberos delegation', '## Delegation\n\nFind delegation\nAbuse unconstrained\nRBCD attack'],
		['Implement ADCS attacks', 'Certificate abuse', '## ADCS\n\nESC1-ESC8\nTemplate abuse\nCertificate theft'],
		['Add persistence mechanisms', 'Maintain access', '## Persistence\n\nGolden ticket\nSilver ticket\nSkeleton key'],
	] as [string, string, string][] },
]);

// Reimplement: Complete Aircrack-ng Suite (88)
expandPath(88, [
	{ name: 'Wireless Capture', desc: 'Packet capture', tasks: [
		['Set up monitor mode', 'Interface config', '## Monitor Mode\n\nairmon-ng start wlan0\nCapture all frames\nChannel hopping'],
		['Build packet capture', 'Capture traffic', '## Capture\n\nCapture to pcap\nFilter by BSSID\nFilter by channel'],
		['Implement deauth attack', 'Client disconnection', '## Deauth\n\nForge deauth frames\nDisconnect clients\nCapture handshakes'],
		['Add beacon injection', 'Fake AP', '## Beacon\n\nCreate fake AP\nEvil twin attack\nCapture credentials'],
		['Build handshake capture', 'WPA handshake', '## Handshake\n\nCapture EAPOL frames\nDeauth to trigger\nVerify handshake'],
		['Implement PMKID capture', 'Clientless attack', '## PMKID\n\nCapture from AP\nNo client needed\nFaster attack'],
	] as [string, string, string][] },
	{ name: 'Password Cracking', desc: 'Offline attacks', tasks: [
		['Build WPA cracker', 'Dictionary attack', '## WPA Crack\n\nParse handshake\nHash with PBKDF2\nCompare PMK'],
		['Implement PMKID cracker', 'PMKID attack', '## PMKID Crack\n\nExtract PMKID\nHashcat mode 16800\nFaster than WPA'],
		['Add rule-based cracking', 'Password mangling', '## Rules\n\nHashcat rules\nWord variations\nEfficient cracking'],
		['Build WEP cracker', 'Legacy protocol', '## WEP\n\nCollect IVs\nPTW attack\nFragmentation attack'],
		['Implement GPU acceleration', 'Fast cracking', '## GPU\n\nHashcat integration\nCUDA/OpenCL\nMassive speedup'],
		['Add distributed cracking', 'Multiple machines', '## Distributed\n\nSplit workload\nCoordinate nodes\nAggregate results'],
	] as [string, string, string][] },
]);

// Reimplement: Password & WiFi Cracking (83)
expandPath(83, [
	{ name: 'Password Cracking', desc: 'Offline hash cracking', tasks: [
		['Implement hash identification', 'Detect hash type', '## Identification\n\nByLength, format\nMagic numbers\nContext clues'],
		['Build dictionary attack', 'Wordlist cracking', '## Dictionary\n\nLoad wordlist\nHash and compare\nProgress tracking'],
		['Add mask attack', 'Pattern-based', '## Mask\n\n?l?u?d?s\nCustom charsets\nIncremental lengths'],
		['Implement rule engine', 'Word mangling', '## Rules\n\nJohn/Hashcat rules\nCombine with wordlist\nMultiple rules'],
		['Build hybrid attack', 'Wordlist + mask', '## Hybrid\n\nBase word + pattern\nWord + digits\nEfficient approach'],
		['Add markov chains', 'Statistical attack', '## Markov\n\nLearn from passwords\nGenerate likely guesses\nFaster than brute'],
	] as [string, string, string][] },
	{ name: 'WiFi Cracking', desc: 'Wireless attacks', tasks: [
		['Capture WPA handshake', 'Get authentication', '## Capture\n\nMonitor mode\nDeauth clients\nCapture EAPOL'],
		['Implement WPA cracking', 'PBKDF2-based', '## WPA Crack\n\nPBKDF2-SHA1\nCompare to MIC\nGPU acceleration'],
		['Add PMKID attack', 'Clientless crack', '## PMKID\n\nExtract from beacon\nNo handshake needed\nFaster attack'],
		['Build rainbow tables', 'Precomputation', '## Rainbow\n\nPre-compute for SSIDs\nInstant lookup\nStorage tradeoff'],
		['Implement WEP cracking', 'Legacy attacks', '## WEP\n\nCollect IVs\nStatistical attack\nPTW method'],
		['Add online attacks', 'Live attacks', '## Online\n\nWPS brute force\nPixie dust\nDefault passwords'],
	] as [string, string, string][] },
]);

// Red Team Tooling: C/C++ Fundamentals (52)
expandPath(52, [
	{ name: 'C/C++ Basics', desc: 'Language fundamentals', tasks: [
		['Set up development environment', 'Compiler and tools', '## Setup\n\nVisual Studio or MinGW\nCMake for builds\nDebugger setup'],
		['Learn memory management', 'Pointers and allocation', '## Memory\n\nmalloc/free\nnew/delete\nMemory layout'],
		['Implement file operations', 'File I/O', '## Files\n\nfopen, fread, fwrite\nBinary file handling\nMemory mapping'],
		['Add socket programming', 'Network operations', '## Sockets\n\nWinsock/BSD sockets\nTCP/UDP\nClient/server'],
		['Build process operations', 'System interaction', '## Processes\n\nCreateProcess\nProcess injection\nHandle manipulation'],
		['Implement threading', 'Concurrency', '## Threads\n\nCreateThread\nSynchronization\nThread pools'],
	] as [string, string, string][] },
	{ name: 'Offensive C/C++', desc: 'Security applications', tasks: [
		['Build shellcode loader', 'Execute shellcode', '## Loader\n\nVirtualAlloc\nmemcpy\nExecute in memory'],
		['Implement DLL injection', 'Process injection', '## DLL Inject\n\nCreateRemoteThread\nLoadLibrary\nManual mapping'],
		['Add API hooking', 'Function interception', '## Hooks\n\nIAT hooking\nInline hooking\nDetour library'],
		['Build PE parser', 'Executable analysis', '## PE\n\nParse headers\nSection enumeration\nImport/export tables'],
		['Implement syscalls', 'Direct kernel calls', '## Syscalls\n\nAvoid usermode hooks\nResolve SSN\nCall directly'],
		['Add anti-debug', 'Evasion techniques', '## Anti-Debug\n\nIsDebuggerPresent\nPEB checks\nTiming checks'],
	] as [string, string, string][] },
]);

// Reimplement: Mimikatz (62)
expandPath(62, [
	{ name: 'Credential Extraction', desc: 'Memory extraction', tasks: [
		['Understand LSASS', 'Security subsystem', '## LSASS\n\nLocal Security Authority\nStores credentials\nProtected process'],
		['Build LSASS accessor', 'Open LSASS', '## Access\n\nSeDebugPrivilege\nOpenProcess\nRead memory'],
		['Implement sekurlsa module', 'Extract credentials', '## Sekurlsa\n\nParse LSASS memory\nFind credential structures\nExtract passwords/hashes'],
		['Add wdigest extraction', 'Plaintext passwords', '## Wdigest\n\nReversible encryption\nFind in memory\nDecrypt passwords'],
		['Build DPAPI module', 'Data protection', '## DPAPI\n\nMaster keys\nCredential files\nDecryption'],
		['Implement Kerberos module', 'Ticket extraction', '## Kerberos\n\nFind tickets in memory\nExport to file\nPass-the-ticket'],
	] as [string, string, string][] },
	{ name: 'Advanced Attacks', desc: 'Kerberos attacks', tasks: [
		['Build golden ticket', 'Forge TGT', '## Golden\n\nkrbtgt hash\nForge TGT\n10-year validity'],
		['Implement silver ticket', 'Forge service ticket', '## Silver\n\nService account hash\nForge TGS\nTarget specific service'],
		['Add skeleton key', 'Backdoor DC', '## Skeleton\n\nPatch LSASS\nAny password works\nMaster key'],
		['Build DCSync', 'Replicate credentials', '## DCSync\n\nDRSGetNCChanges\nDump all hashes\nNo code on DC'],
		['Implement DCShadow', 'Rogue DC', '## DCShadow\n\nRegister rogue DC\nPush malicious changes\nStealthy persistence'],
		['Add token manipulation', 'Impersonation', '## Tokens\n\nDuplicate token\nImpersonate user\nPrivilege escalation'],
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
