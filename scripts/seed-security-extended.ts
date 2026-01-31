import Database from 'better-sqlite3';

const sqlite = new Database('data/quest-log.db');

const insertPath = sqlite.prepare(
	'INSERT INTO paths (name, description, color, created_at) VALUES (?, ?, ?, ?)'
);
const insertModule = sqlite.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = sqlite.prepare(
	'INSERT INTO tasks (module_id, title, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);

const now = Date.now();

// ============================================================================
// BLUE TEAM / DEFENSIVE SECURITY
// ============================================================================
const bluePath = insertPath.run(
	'Blue Team & Defensive Security',
	'Master defensive security: SIEM, threat detection, incident response, forensics, and malware analysis. Build detection capabilities and respond to real-world attacks.',
	'sky',
	now
);

// Module 1: SIEM & Log Analysis
const blueM1 = insertModule.run(bluePath.lastInsertRowid, 'SIEM & Log Analysis', 'Set up and master Security Information and Event Management', 0, now);
const blueM1Tasks = [
	['Set up ELK Stack (Elasticsearch, Logstash, Kibana)', 'Deploy a local ELK stack using Docker. Configure Elasticsearch for log storage, Logstash for parsing, and Kibana for visualization. This is the foundation for all log analysis work.'],
	['Configure Windows Event Log forwarding', 'Set up Windows Event Forwarding (WEF) to centralize logs from Windows endpoints. Focus on Security, System, and PowerShell logs. Configure subscriptions for critical event IDs.'],
	['Build detection rules for common attacks', 'Create Kibana alerts for: failed login brute force (4625), new service installation (7045), PowerShell suspicious commands (4104), and process creation (4688). Test each rule.'],
	['Ingest and analyze Sysmon logs', 'Install Sysmon with a detection-focused config (SwiftOnSecurity or Olaf). Parse process creation, network connections, and file creation events. Build a dashboard for visibility.'],
	['Create a threat hunting dashboard', 'Build a Kibana dashboard showing: top processes, network connections by destination, rare parent-child relationships, and unsigned binaries. Use this for proactive hunting.'],
	['Practice with BOTS dataset', 'Download Splunk Boss of the SOC dataset and import into ELK. Practice investigating a realistic attack scenario including lateral movement and data exfiltration.'],
];
blueM1Tasks.forEach((t, i) => insertTask.run(blueM1.lastInsertRowid, t[0], t[1], i, now));

// Module 2: Threat Detection & Hunting
const blueM2 = insertModule.run(bluePath.lastInsertRowid, 'Threat Detection & Hunting', 'Proactive threat hunting and detection engineering', 1, now);
const blueM2Tasks = [
	['Learn the MITRE ATT&CK framework', 'Study the ATT&CK matrix focusing on Initial Access, Execution, Persistence, and Lateral Movement. Map your detection capabilities to ATT&CK techniques.'],
	['Write Sigma rules for detection', 'Learn Sigma rule syntax and write rules for: mimikatz execution, DCSync, Kerberoasting, and scheduled task persistence. Convert to your SIEM format.'],
	['Implement YARA rules for file scanning', 'Write YARA rules to detect malware families. Start with simple string matching, progress to PE structure analysis. Integrate with your analysis workflow.'],
	['Build a threat hunting hypothesis workflow', 'Create a structured approach: hypothesis → data sources → analysis → findings. Document 5 hunts: unusual PowerShell, lateral movement, persistence, exfiltration, credential access.'],
	['Analyze network traffic with Zeek', 'Deploy Zeek to capture network metadata. Analyze conn.log, dns.log, http.log, and ssl.log. Build detections for beaconing, DNS tunneling, and suspicious TLS.'],
	['Practice with Atomic Red Team', 'Run Atomic Red Team tests in a lab. Verify your detections trigger. Document gaps and improve detection coverage.'],
];
blueM2Tasks.forEach((t, i) => insertTask.run(blueM2.lastInsertRowid, t[0], t[1], i, now));

// Module 3: Incident Response
const blueM3 = insertModule.run(bluePath.lastInsertRowid, 'Incident Response', 'Structured incident handling and containment', 2, now);
const blueM3Tasks = [
	['Create an IR playbook template', 'Build playbooks covering: phishing, malware infection, compromised account, ransomware, and data breach. Include detection, containment, eradication, and recovery steps.'],
	['Master memory forensics with Volatility', 'Use Volatility 3 to analyze memory dumps. Practice: listing processes, network connections, injected code, and extracting malware. Analyze a compromised system image.'],
	['Perform disk forensics with Autopsy', 'Use Autopsy to analyze disk images. Practice timeline analysis, file recovery, browser artifact analysis, and registry examination.'],
	['Collect and preserve evidence properly', 'Learn chain of custody, proper imaging with FTK Imager or dd, hash verification, and documentation requirements. Practice on a test system.'],
	['Analyze a complete attack chain', 'Work through a full incident: initial access → execution → persistence → lateral movement → objective. Document IOCs, TTPs, and remediation steps.'],
	['Write an incident report', 'Create a comprehensive report including: executive summary, timeline, technical details, IOCs, root cause, and recommendations. Practice clear communication for different audiences.'],
];
blueM3Tasks.forEach((t, i) => insertTask.run(blueM3.lastInsertRowid, t[0], t[1], i, now));

// Module 4: Malware Analysis
const blueM4 = insertModule.run(bluePath.lastInsertRowid, 'Malware Analysis', 'Static and dynamic malware analysis techniques', 3, now);
const blueM4Tasks = [
	['Set up a malware analysis lab', 'Build an isolated VM environment with FlareVM or REMnux. Configure network isolation, snapshots, and safe file sharing. Never analyze malware on production systems.'],
	['Perform static analysis with PE tools', 'Use PE-bear, pestudio, and CFF Explorer to analyze Windows executables. Identify imports, exports, sections, and suspicious indicators without execution.'],
	['Master dynamic analysis with x64dbg', 'Debug malware to understand runtime behavior. Set breakpoints on key APIs (VirtualAlloc, CreateProcess, WinHTTP). Trace execution and extract IOCs.'],
	['Analyze a dropper/loader', 'Reverse a multi-stage malware sample. Identify the unpacking routine, decrypt embedded payloads, and extract the final payload. Document the full chain.'],
	['Reverse engineer a RAT/backdoor', 'Analyze command-and-control malware. Identify C2 protocol, commands supported, and persistence mechanism. Extract network IOCs for detection.'],
	['Write a malware analysis report', 'Document a complete analysis: summary, static findings, dynamic behavior, network indicators, MITRE mappings, and detection recommendations.'],
];
blueM4Tasks.forEach((t, i) => insertTask.run(blueM4.lastInsertRowid, t[0], t[1], i, now));

// ============================================================================
// RED TEAM EXTENDED - WEB, CLOUD, MOBILE
// ============================================================================
const redExtPath = insertPath.run(
	'Red Team Extended: Web, Cloud & Mobile',
	'Expand offensive skills beyond Windows/AD. Master web application attacks, cloud security (AWS/Azure), mobile app testing, and social engineering.',
	'red',
	now
);

// Module 1: Web Application Attacks
const redM1 = insertModule.run(redExtPath.lastInsertRowid, 'Web Application Attacks', 'Master OWASP Top 10 and advanced web exploitation', 0, now);
const redM1Tasks = [
	['Set up a web hacking lab', 'Deploy DVWA, WebGoat, and Juice Shop locally with Docker. These provide safe, legal targets for practicing web attacks.'],
	['Master SQL injection techniques', 'Practice union-based, blind boolean, and time-based SQLi. Use sqlmap for automation but understand manual techniques. Extract data and escalate to OS command execution.'],
	['Exploit XSS vulnerabilities', 'Find and exploit reflected, stored, and DOM-based XSS. Progress from alert boxes to session hijacking and keylogging. Bypass common WAF filters.'],
	['Attack authentication mechanisms', 'Test for: weak passwords, credential stuffing, session fixation, JWT attacks, OAuth misconfigurations, and 2FA bypasses.'],
	['Exploit server-side vulnerabilities', 'Practice SSRF, XXE, SSTI, and insecure deserialization. Understand impact: internal network access, RCE, data exfiltration.'],
	['Chain vulnerabilities for maximum impact', 'Combine multiple lower-severity bugs into critical chains. Example: open redirect → OAuth token theft → account takeover.'],
	['Perform a full web app assessment', 'Conduct end-to-end pentest of a target app. Document methodology, findings, and provide actionable remediation.'],
];
redM1Tasks.forEach((t, i) => insertTask.run(redM1.lastInsertRowid, t[0], t[1], i, now));

// Module 2: Cloud Security (AWS/Azure)
const redM2 = insertModule.run(redExtPath.lastInsertRowid, 'Cloud Security & Attacks', 'Attack and defend cloud infrastructure', 1, now);
const redM2Tasks = [
	['Set up cloud pentesting lab', 'Create AWS free tier and Azure trial accounts. Deploy intentionally vulnerable environments: CloudGoat, flAWS, AzureGoat.'],
	['Enumerate cloud resources', 'Use tools like ScoutSuite, Prowler, and enumerate_iam. Find exposed S3 buckets, overly permissive IAM policies, and public resources.'],
	['Exploit IAM misconfigurations', 'Practice privilege escalation via: overly permissive policies, role assumption chains, and instance profile abuse. Use Pacu for automation.'],
	['Attack serverless functions', 'Exploit Lambda/Azure Functions: injection in event data, environment variable secrets, overprivileged execution roles.'],
	['Pivot through cloud networks', 'Exploit VPC peering, transit gateways, and hybrid connections. Move from cloud to on-prem and back.'],
	['Attack container orchestration', 'Exploit Kubernetes misconfigurations: exposed dashboards, privileged containers, mounted service accounts. Escape containers to host.'],
	['Perform a cloud security assessment', 'Conduct full assessment of a cloud environment. Review IAM, network security, data storage, logging, and provide recommendations.'],
];
redM2Tasks.forEach((t, i) => insertTask.run(redM2.lastInsertRowid, t[0], t[1], i, now));

// Module 3: Mobile Application Security
const redM3 = insertModule.run(redExtPath.lastInsertRowid, 'Mobile Application Security', 'Android and iOS application testing', 2, now);
const redM3Tasks = [
	['Set up mobile testing environment', 'Configure Android emulator with Frida and Objection. For iOS, set up a jailbroken device or use Corellium. Install Burp Suite for traffic interception.'],
	['Bypass SSL pinning', 'Use Frida scripts to bypass certificate pinning in Android and iOS apps. Intercept and modify HTTPS traffic.'],
	['Analyze Android APKs', 'Decompile APKs with jadx, analyze Smali with apktool. Find hardcoded secrets, API keys, and insecure storage.'],
	['Exploit insecure data storage', 'Find sensitive data in: SharedPreferences, SQLite databases, external storage, and backup files. Demonstrate impact.'],
	['Attack mobile APIs', 'Test the backend API for: broken authentication, IDOR, excessive data exposure, and rate limiting issues.'],
	['Reverse engineer mobile apps', 'Use Ghidra or IDA for native library analysis. Bypass root/jailbreak detection and other client-side protections.'],
	['Conduct a mobile app pentest', 'Full assessment following OWASP MASTG. Test both client-side and server-side components. Deliver a comprehensive report.'],
];
redM3Tasks.forEach((t, i) => insertTask.run(redM3.lastInsertRowid, t[0], t[1], i, now));

// ============================================================================
// DEVSECOPS
// ============================================================================
const devSecPath = insertPath.run(
	'DevSecOps Engineering',
	'Integrate security into CI/CD pipelines. Master SAST, DAST, container security, secrets management, and infrastructure as code security.',
	'violet',
	now
);

// Module 1: CI/CD Security
const devM1 = insertModule.run(devSecPath.lastInsertRowid, 'CI/CD Pipeline Security', 'Secure your build and deployment pipelines', 0, now);
const devM1Tasks = [
	['Audit a CI/CD pipeline for risks', 'Review GitHub Actions, GitLab CI, or Jenkins configs. Identify: exposed secrets, overprivileged tokens, untrusted inputs, and supply chain risks.'],
	['Implement branch protection rules', 'Configure required reviews, status checks, signed commits, and prevent force pushes. Protect main/release branches.'],
	['Secure CI/CD secrets management', 'Migrate hardcoded secrets to GitHub Secrets, Vault, or cloud secret managers. Implement secret rotation and audit logging.'],
	['Prevent dependency confusion attacks', 'Configure package managers to use private registries first. Implement namespace protection and verify package integrity.'],
	['Implement signed commits and artifacts', 'Set up GPG commit signing and artifact signing with Sigstore/cosign. Verify signatures in deployment pipelines.'],
	['Create a security gate in CI', 'Build a pipeline stage that fails builds on: critical vulnerabilities, secret detection, or failed security tests.'],
];
devM1Tasks.forEach((t, i) => insertTask.run(devM1.lastInsertRowid, t[0], t[1], i, now));

// Module 2: SAST & DAST
const devM2 = insertModule.run(devSecPath.lastInsertRowid, 'SAST & DAST Integration', 'Automated security testing in development', 1, now);
const devM2Tasks = [
	['Integrate Semgrep for SAST', 'Add Semgrep to CI pipeline. Configure rules for your languages. Handle false positives with ignore comments and baseline.'],
	['Set up dependency scanning', 'Implement Dependabot, Snyk, or Trivy for vulnerability scanning. Configure auto-PRs for patches and break builds on critical vulns.'],
	['Implement secret scanning', 'Add Gitleaks or TruffleHog to pre-commit hooks and CI. Scan git history for leaked secrets. Establish remediation process.'],
	['Configure DAST with OWASP ZAP', 'Run ZAP in CI against staging environment. Configure authenticated scanning, API scanning, and baseline comparisons.'],
	['Build a security testing dashboard', 'Aggregate results from all security tools. Track vulnerability trends, mean time to remediate, and coverage metrics.'],
	['Create developer security training', 'Build secure coding guidelines and training materials. Include examples of common vulnerabilities in your stack and how to prevent them.'],
];
devM2Tasks.forEach((t, i) => insertTask.run(devM2.lastInsertRowid, t[0], t[1], i, now));

// Module 3: Container Security
const devM3 = insertModule.run(devSecPath.lastInsertRowid, 'Container & Kubernetes Security', 'Secure containerized workloads', 2, now);
const devM3Tasks = [
	['Create hardened base images', 'Build minimal images from scratch or distroless. Remove shells, package managers, and unnecessary tools. Scan with Trivy.'],
	['Implement image signing and verification', 'Sign images with cosign, verify in Kubernetes with admission controllers. Prevent unsigned images from running.'],
	['Configure Kubernetes security policies', 'Implement Pod Security Standards (restricted). Configure: no root, no privileged, read-only fs, dropped capabilities.'],
	['Set up network policies', 'Implement zero-trust networking. Default deny all, explicitly allow required traffic. Isolate namespaces.'],
	['Deploy runtime security monitoring', 'Install Falco for runtime threat detection. Alert on: shell spawns, sensitive file access, unexpected network connections.'],
	['Implement secrets management in K8s', 'Use External Secrets Operator with Vault or cloud secret managers. Avoid native K8s secrets for sensitive data.'],
	['Perform a container security assessment', 'Full audit of container environment: images, runtime config, network, secrets, and RBAC. Provide hardening recommendations.'],
];
devM3Tasks.forEach((t, i) => insertTask.run(devM3.lastInsertRowid, t[0], t[1], i, now));

// ============================================================================
// CTF PRACTICE
// ============================================================================
const ctfPath = insertPath.run(
	'CTF Challenge Practice',
	'Structured practice for Capture The Flag competitions. Cover web, crypto, pwn, reverse engineering, and forensics challenges.',
	'amber',
	now
);

// Module 1: Web Challenges
const ctfM1 = insertModule.run(ctfPath.lastInsertRowid, 'Web Exploitation', 'Solve web-based CTF challenges', 0, now);
const ctfM1Tasks = [
	['Complete 10 easy web challenges on picoCTF', 'Start with picoCTF web challenges. Focus on: source code inspection, cookies, hidden paths, and basic injection.'],
	['Solve PortSwigger Web Academy SQLi labs', 'Complete all SQL injection labs. Progress from basic to blind to out-of-band. Document payloads that work.'],
	['Master SSTI exploitation', 'Practice template injection on HackTheBox or TryHackMe. Learn Jinja2, Twig, and Freemarker payloads.'],
	['Solve prototype pollution challenges', 'Find and solve JS prototype pollution CTF challenges. Understand the vuln and how to chain with other bugs.'],
	['Complete 5 hard web CTF challenges', 'Tackle harder challenges from recent CTFs (CTFtime.org). Time yourself and compare to writeups after.'],
];
ctfM1Tasks.forEach((t, i) => insertTask.run(ctfM1.lastInsertRowid, t[0], t[1], i, now));

// Module 2: Cryptography
const ctfM2 = insertModule.run(ctfPath.lastInsertRowid, 'Cryptography', 'Break crypto in CTF challenges', 1, now);
const ctfM2Tasks = [
	['Learn classical cipher attacks', 'Practice breaking: Caesar, Vigenere, substitution ciphers. Use frequency analysis and known plaintext attacks.'],
	['Attack weak RSA implementations', 'Solve RSA challenges: small e, common modulus, Wiener attack, Coppersmith. Use RsaCtfTool.'],
	['Exploit AES/block cipher weaknesses', 'Practice: ECB mode attacks, padding oracle, bit flipping in CBC. Implement attacks from scratch for understanding.'],
	['Solve hash-based challenges', 'Attack: length extension, hash collisions, weak MAC constructions. Practice with MD5 and SHA-1 challenges.'],
	['Complete 5 medium crypto challenges', 'Solve crypto challenges from CryptoHack.org. Focus on understanding the math behind each attack.'],
];
ctfM2Tasks.forEach((t, i) => insertTask.run(ctfM2.lastInsertRowid, t[0], t[1], i, now));

// Module 3: Binary Exploitation
const ctfM3 = insertModule.run(ctfPath.lastInsertRowid, 'Binary Exploitation (Pwn)', 'Memory corruption and binary attacks', 2, now);
const ctfM3Tasks = [
	['Set up pwn environment', 'Install pwntools, GDB with pwndbg/GEF, Ghidra. Configure for 32-bit and 64-bit targets.'],
	['Master buffer overflow basics', 'Solve stack buffer overflow challenges. Control EIP/RIP, build ROP chains, bypass NX with ret2libc.'],
	['Exploit format string vulnerabilities', 'Practice reading/writing memory with format strings. Leak addresses and overwrite GOT entries.'],
	['Learn heap exploitation', 'Study glibc malloc internals. Solve heap challenges: use-after-free, double free, house of techniques.'],
	['Complete 5 pwn challenges on pwnable.kr/tw', 'Work through progressive difficulty. Document your exploits and understand mitigations bypassed.'],
];
ctfM3Tasks.forEach((t, i) => insertTask.run(ctfM3.lastInsertRowid, t[0], t[1], i, now));

// Module 4: Reverse Engineering
const ctfM4 = insertModule.run(ctfPath.lastInsertRowid, 'Reverse Engineering', 'Analyze and understand binaries', 3, now);
const ctfM4Tasks = [
	['Learn x86/x64 assembly basics', 'Understand registers, calling conventions, stack operations. Practice reading disassembly before running binaries.'],
	['Master Ghidra for static analysis', 'Reverse crackme challenges. Use decompiler, rename variables, annotate functions. Make the code readable.'],
	['Defeat anti-debugging tricks', 'Bypass: IsDebuggerPresent, timing checks, self-modifying code. Patch binaries to remove protections.'],
	['Reverse obfuscated code', 'Tackle VMs and custom obfuscation. Trace execution, identify patterns, write deobfuscation scripts.'],
	['Complete 5 reversing challenges', 'Solve challenges from crackmes.one and CTFs. Progress from simple keygens to complex VMs.'],
];
ctfM4Tasks.forEach((t, i) => insertTask.run(ctfM4.lastInsertRowid, t[0], t[1], i, now));

// Module 5: Forensics
const ctfM5 = insertModule.run(ctfPath.lastInsertRowid, 'Forensics & Steganography', 'Extract hidden data and analyze artifacts', 4, now);
const ctfM5Tasks = [
	['Master file format analysis', 'Learn magic bytes, file carving with binwalk, and fixing corrupted files. Extract embedded files from various formats.'],
	['Analyze memory dumps', 'Use Volatility for CTF challenges. Extract passwords, process info, network connections, and hidden data.'],
	['Solve steganography challenges', 'Use: strings, exiftool, steghide, zsteg, stegsolve. Learn LSB extraction and various stego techniques.'],
	['Analyze network captures', 'Extract data from pcaps with Wireshark and tshark. Reassemble streams, find hidden channels, decode protocols.'],
	['Complete 5 forensics challenges', 'Solve forensics challenges from CTFs. Combine multiple techniques in single challenges.'],
];
ctfM5Tasks.forEach((t, i) => insertTask.run(ctfM5.lastInsertRowid, t[0], t[1], i, now));

console.log('Seeded 4 new security learning paths:');
console.log('  - Blue Team & Defensive Security (4 modules, 24 tasks)');
console.log('  - Red Team Extended: Web, Cloud & Mobile (3 modules, 21 tasks)');
console.log('  - DevSecOps Engineering (3 modules, 19 tasks)');
console.log('  - CTF Challenge Practice (5 modules, 25 tasks)');
console.log('Total: 89 new tasks added!');

sqlite.close();
