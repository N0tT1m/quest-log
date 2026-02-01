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
// NETWORK ANALYSIS & TRAFFIC FORENSICS
// ============================================================================
const netPath = insertPath.run(
	'Network Analysis & Traffic Forensics',
	'Master network traffic analysis, packet capture, protocol dissection, and network forensics. Essential skills for both offensive and defensive security.',
	'cyan',
	now
);

// Module 1: Packet Capture Fundamentals
const mod1 = insertModule.run(netPath.lastInsertRowid, 'Packet Capture Fundamentals', 'Master tcpdump, Wireshark, and capture techniques', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Master tcpdump for command-line capture', 'Learn tcpdump BPF filter syntax for capturing specific traffic, output formatting options, writing to pcap files, and real-time traffic analysis on remote servers via SSH for network troubleshooting and security monitoring', `## tcpdump Essentials

### Basic Capture Commands
\`\`\`bash
# Capture on specific interface
tcpdump -i eth0

# Capture with verbose output
tcpdump -i eth0 -v

# Capture and save to file
tcpdump -i eth0 -w capture.pcap

# Read from saved file
tcpdump -r capture.pcap

# Capture specific number of packets
tcpdump -i eth0 -c 100

# Don't resolve hostnames (faster)
tcpdump -i eth0 -n

# Show packet contents in hex and ASCII
tcpdump -i eth0 -X
\`\`\`

### Essential Filters
\`\`\`bash
# Filter by host
tcpdump host 192.168.1.100

# Filter by source/destination
tcpdump src 192.168.1.100
tcpdump dst 10.0.0.1

# Filter by port
tcpdump port 443
tcpdump src port 80

# Filter by protocol
tcpdump tcp
tcpdump udp
tcpdump icmp

# Combine filters with and/or/not
tcpdump 'src 192.168.1.100 and dst port 443'
tcpdump 'tcp and (port 80 or port 443)'
tcpdump 'not arp and not icmp'

# Filter by TCP flags
tcpdump 'tcp[tcpflags] & tcp-syn != 0'
tcpdump 'tcp[tcpflags] == tcp-syn'
\`\`\`

### Advanced Filters
\`\`\`bash
# Capture HTTP GET requests
tcpdump -i eth0 -A 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'

# Capture DNS queries
tcpdump -i eth0 'udp port 53'

# Capture packets larger than X bytes
tcpdump -i eth0 'greater 1000'

# Capture specific VLAN
tcpdump -i eth0 'vlan 100'
\`\`\`

### Practice Exercise
1. Capture all traffic to/from a specific website
2. Filter only HTTPS handshakes (SYN packets to port 443)
3. Save 5 minutes of traffic and analyze offline`, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Master Wireshark interface and navigation', 'Learn Wireshark navigation including column customization, display filter syntax, following TCP streams, expert info analysis, and using the Statistics menu for protocol hierarchy and endpoint analysis', `## Wireshark Mastery

### Display Filters (Different from Capture Filters!)
\`\`\`
# Filter by IP
ip.addr == 192.168.1.100
ip.src == 192.168.1.100
ip.dst == 10.0.0.1

# Filter by port
tcp.port == 443
tcp.srcport == 80
udp.dstport == 53

# Filter by protocol
http
dns
tls
ssh

# Filter HTTP methods
http.request.method == "GET"
http.request.method == "POST"

# Filter by TCP flags
tcp.flags.syn == 1
tcp.flags.reset == 1

# Contains specific string
http contains "password"
frame contains "secret"

# Filter by packet length
frame.len > 1000
tcp.len > 0

# Combine with && || !
ip.addr == 192.168.1.100 && tcp.port == 80
http || dns
!arp && !icmp
\`\`\`

### Essential Keyboard Shortcuts
- Ctrl+F: Find packet
- Ctrl+G: Go to packet number
- Ctrl+→: Next packet in conversation
- Ctrl+.: Next packet (same stream)
- Right-click → Follow → TCP/HTTP Stream

### Columns to Add
1. Edit → Preferences → Columns
2. Add: Delta Time, Source Port, Dest Port, TCP Stream Index
3. Right-click column → displayed as hex (for flags)

### Coloring Rules
Create custom coloring for:
- Red: TCP RST, HTTP 4xx/5xx
- Yellow: Retransmissions
- Green: HTTP 200 OK

### Practice
1. Open a pcap and identify all unique IP conversations
2. Find all HTTP requests and their response codes
3. Follow a complete TCP stream from SYN to FIN`, 1, now);

insertTask.run(mod1.lastInsertRowid, 'Capture traffic in different network positions', 'Understand network capture techniques including switch port mirroring (SPAN), network TAPs, hub-based capture, and active interception via ARP spoofing to gather traffic for security analysis', `## Network Capture Positions

### 1. SPAN/Mirror Port (Legitimate)
\`\`\`
# Cisco switch configuration
Switch(config)# monitor session 1 source interface Gi0/1
Switch(config)# monitor session 1 destination interface Gi0/24
\`\`\`
- Pros: No inline device, no latency
- Cons: May drop packets under load, half-duplex issues

### 2. Network TAP (Best for Forensics)
- Hardware device that copies traffic
- Full-duplex capture without drops
- Passive - no IP address, invisible
- Recommended: Dualcomm or Throwing Star LAN Tap

### 3. Inline Capture (IDS/IPS Position)
\`\`\`bash
# Linux bridge for inline capture
brctl addbr br0
brctl addif br0 eth0 eth1
ifconfig br0 up
tcpdump -i br0 -w capture.pcap
\`\`\`

### 4. ARP Spoofing (Pentest/Red Team)
\`\`\`bash
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# ARP spoof with arpspoof
arpspoof -i eth0 -t 192.168.1.100 192.168.1.1
arpspoof -i eth0 -t 192.168.1.1 192.168.1.100

# Or use bettercap
bettercap -iface eth0
> net.probe on
> set arp.spoof.targets 192.168.1.100
> arp.spoof on
> net.sniff on
\`\`\`

### 5. Wireless Capture
\`\`\`bash
# Put interface in monitor mode
airmon-ng start wlan0

# Capture on specific channel
airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon

# Or with tcpdump
tcpdump -i wlan0mon -w wireless.pcap
\`\`\`

### Practice Lab
1. Set up a Linux bridge and capture traffic
2. Use bettercap to intercept traffic between two VMs
3. Capture wireless traffic and identify SSIDs`, 2, now);

insertTask.run(mod1.lastInsertRowid, 'Extract files and objects from captures', 'Use Wireshark export objects, tcpflow, or NetworkMiner to carve transferred files including images, documents, executables, and archives from HTTP, SMB, and FTP traffic for forensic analysis', `## File Extraction from Packet Captures

### Wireshark Export Objects
1. File → Export Objects → HTTP/SMB/TFTP/IMF
2. Select files to save
3. Works for unencrypted protocols only

### NetworkMiner (Automated)
\`\`\`bash
# Install on Linux
sudo apt install networkminer

# Open pcap - auto extracts:
# - Files, images
# - Credentials
# - DNS queries
# - Host details
\`\`\`

### Foremost (File Carving)
\`\`\`bash
# Extract files from raw pcap
foremost -i capture.pcap -o output/

# Custom file types
foremost -t jpg,png,pdf,doc -i capture.pcap -o output/
\`\`\`

### Binwalk (Embedded Files)
\`\`\`bash
# Analyze pcap for embedded files
binwalk capture.pcap

# Extract all found files
binwalk -e capture.pcap
\`\`\`

### Scapy for Custom Extraction
\`\`\`python
from scapy.all import *

def extract_http_payload(pcap_file):
    packets = rdpcap(pcap_file)
    for pkt in packets:
        if pkt.haslayer(Raw):
            payload = pkt[Raw].load
            if b'PNG' in payload or b'JFIF' in payload:
                # Found image data
                print(f"Image found in packet {packets.index(pkt)}")

extract_http_payload('capture.pcap')
\`\`\`

### Tshark for Targeted Extraction
\`\`\`bash
# Extract HTTP URLs
tshark -r capture.pcap -Y http.request -T fields -e http.host -e http.request.uri

# Extract file hashes from SMB
tshark -r capture.pcap -Y smb2 -T fields -e smb2.filename

# Extract DNS queries
tshark -r capture.pcap -Y dns.qry.name -T fields -e dns.qry.name | sort -u
\`\`\`

### Practice
1. Download a pcap with HTTP file transfers
2. Extract all images using multiple methods
3. Compare results from Wireshark vs NetworkMiner`, 3, now);

// Module 2: Protocol Analysis
const mod2 = insertModule.run(netPath.lastInsertRowid, 'Protocol Deep Dives', 'Analyze HTTP, DNS, TLS, and other protocols in depth', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Analyze HTTP traffic in depth', 'Dissect HTTP headers, methods, status codes, and body content to identify web vulnerabilities, credential leakage, injection attacks, and suspicious user-agent patterns in network captures', `## HTTP Protocol Analysis

### HTTP Request Structure
\`\`\`
GET /api/users HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Cookie: session=abc123
Authorization: Bearer eyJ...

[Request Body for POST/PUT]
\`\`\`

### Key Headers to Examine
\`\`\`
# Request Headers
Host: Target server
User-Agent: Client identification (fingerprinting)
Cookie: Session tokens
Authorization: Credentials
X-Forwarded-For: Real client IP (when proxied)
Referer: Previous page (data leakage)

# Response Headers
Set-Cookie: Session management
Location: Redirects
X-Powered-By: Tech stack disclosure
Server: Server software
Content-Security-Policy: Security controls
\`\`\`

### Wireshark HTTP Analysis
\`\`\`
# Display filters
http.request.method == "POST"
http.response.code >= 400
http.cookie contains "session"
http.authorization
http.file_data contains "password"

# Follow HTTP stream
Right-click → Follow → HTTP Stream

# Export HTTP objects
File → Export Objects → HTTP
\`\`\`

### Detect Common Attacks
\`\`\`
# SQL Injection attempts
http.request.uri contains "UNION"
http.request.uri contains "SELECT"
http.request.uri contains "'"

# XSS attempts
http.request.uri contains "<script"
http.request.uri contains "javascript:"

# Path traversal
http.request.uri contains ".."
http.request.uri contains "/etc/passwd"

# Command injection
http.request.uri contains "|"
http.request.uri contains ";"
\`\`\`

### Tshark HTTP Extraction
\`\`\`bash
# Extract all URLs
tshark -r capture.pcap -Y http.request -T fields \\
  -e http.host -e http.request.uri

# Extract POST data
tshark -r capture.pcap -Y "http.request.method==POST" \\
  -T fields -e http.file_data

# Extract cookies
tshark -r capture.pcap -Y http.cookie -T fields \\
  -e ip.src -e http.cookie
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Master DNS traffic analysis', 'Analyze DNS queries and responses to detect reconnaissance patterns, identify DNS tunneling through entropy analysis and query volume, and recognize data exfiltration via long subdomain labels or TXT records', `## DNS Protocol Analysis

### DNS Record Types
\`\`\`
A      - IPv4 address
AAAA   - IPv6 address
CNAME  - Canonical name (alias)
MX     - Mail exchange
NS     - Name server
TXT    - Text records (SPF, DKIM, verification)
PTR    - Reverse lookup
SOA    - Start of authority
SRV    - Service records
\`\`\`

### Wireshark DNS Filters
\`\`\`
# All DNS traffic
dns

# DNS queries only
dns.flags.response == 0

# DNS responses only
dns.flags.response == 1

# Specific query types
dns.qry.type == 1   # A records
dns.qry.type == 28  # AAAA records
dns.qry.type == 16  # TXT records

# Failed lookups (NXDOMAIN)
dns.flags.rcode == 3

# Long domain names (potential tunneling)
dns.qry.name.len > 50
\`\`\`

### Detect DNS Tunneling
\`\`\`bash
# Signs of DNS tunneling:
# 1. Unusually long subdomain names
# 2. High volume of TXT queries
# 3. Base64/hex encoded subdomains
# 4. Queries to suspicious domains

# Tshark analysis
tshark -r capture.pcap -Y dns -T fields \\
  -e dns.qry.name -e dns.qry.type | \\
  awk '{print length($1), $0}' | sort -rn | head -20

# Look for entropy in subdomains
tshark -r capture.pcap -Y "dns.qry.type == 16" -T fields \\
  -e dns.qry.name | sort | uniq -c | sort -rn
\`\`\`

### DNS Exfiltration Detection
\`\`\`python
from scapy.all import *
import math

def entropy(s):
    prob = [s.count(c)/len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

packets = rdpcap('capture.pcap')
for pkt in packets:
    if pkt.haslayer(DNSQR):
        query = pkt[DNSQR].qname.decode()
        subdomain = query.split('.')[0]
        if len(subdomain) > 20 and entropy(subdomain) > 3.5:
            print(f"Suspicious: {query} (entropy: {entropy(subdomain):.2f})")
\`\`\`

### DNS Recon Indicators
\`\`\`
# Zone transfer attempts
dns.qry.type == 252  # AXFR

# Reverse lookups (network mapping)
dns.qry.type == 12   # PTR records

# High volume from single source
dns.qry.name | count by ip.src
\`\`\``, 1, now);

insertTask.run(mod2.lastInsertRowid, 'Decrypt and analyze TLS traffic', 'Decrypt TLS sessions using pre-master secrets or session keys, analyze certificate chains for anomalies, examine JA3/JA3S fingerprints for client identification, and detect TLS-based C2 patterns', `## TLS/SSL Traffic Analysis

### TLS Without Decryption (Metadata Analysis)
\`\`\`
# Wireshark filters
tls.handshake.type == 1    # Client Hello
tls.handshake.type == 2    # Server Hello
tls.handshake.type == 11   # Certificate
tls.alert_message          # TLS alerts

# Extract SNI (Server Name Indication)
tls.handshake.extensions_server_name

# JA3 fingerprinting
tls.handshake.ja3
\`\`\`

### JA3/JA3S Fingerprinting
\`\`\`bash
# JA3 = MD5 of TLS Client Hello parameters
# Used to fingerprint malware, bots, tools

# Zeek JA3 logging
zeek -r capture.pcap ja3

# Common malicious JA3 hashes:
# - Check against ja3er.com database
# - Compare to known tool fingerprints

# Tshark extraction
tshark -r capture.pcap -Y tls.handshake.type==1 \\
  -T fields -e ip.src -e tls.handshake.ja3
\`\`\`

### Decrypt TLS with SSLKEYLOGFILE
\`\`\`bash
# 1. Set environment variable before starting browser
export SSLKEYLOGFILE=/tmp/keys.log
google-chrome &

# 2. Capture traffic
tcpdump -i eth0 -w encrypted.pcap

# 3. Decrypt in Wireshark
# Edit → Preferences → Protocols → TLS
# Set "(Pre)-Master-Secret log filename" to /tmp/keys.log

# Now TLS traffic is decrypted!
\`\`\`

### Decrypt with Private Key (Server-side)
\`\`\`
# Wireshark: Edit → Preferences → Protocols → TLS → RSA Keys
# Add: IP, Port, Protocol (http), Key file path

# Only works for RSA key exchange (not ECDHE)
# Modern TLS uses ephemeral keys - need SSLKEYLOGFILE
\`\`\`

### TLS Attack Indicators
\`\`\`
# Downgrade attacks
tls.handshake.version < 0x0303  # Less than TLS 1.2

# Self-signed certificates
tls.handshake.certificate (inspect issuer)

# Certificate errors
tls.alert_message.description == 42  # Bad certificate
tls.alert_message.description == 48  # Unknown CA

# Expired certificates
# Check validity dates in certificate details
\`\`\`

### Practice
1. Capture HTTPS traffic with SSLKEYLOGFILE
2. Decrypt and analyze the HTTP inside
3. Extract JA3 hashes and look up on ja3er.com`, 2, now);

insertTask.run(mod2.lastInsertRowid, 'Analyze SMB and Active Directory traffic', 'Dissect SMB negotiation, NTLM authentication, file operations, and Active Directory protocols including LDAP queries, Kerberos exchanges, and DRSUAPI replication to understand Windows network behavior', `## SMB/CIFS and AD Protocol Analysis

### SMB Basics
\`\`\`
# SMB versions
SMB1 - Legacy, insecure (EternalBlue)
SMB2 - Windows Vista+
SMB3 - Windows 8+, encryption support

# Wireshark filters
smb || smb2
smb2.cmd == 5    # Create (file access)
smb2.cmd == 8    # Read
smb2.cmd == 9    # Write
smb2.filename    # File being accessed
\`\`\`

### Detect SMB Attacks
\`\`\`
# EternalBlue/MS17-010 indicators
smb.cmd == 0x72  # Negotiate
# Look for Trans2 SESSION_SETUP anomalies

# Brute force attempts
smb2.nt_status == 0xc000006d  # Wrong password

# Pass-the-Hash indicators
# NTLM authentication without prior Kerberos

# Lateral movement
smb2.filename contains "ADMIN$"
smb2.filename contains "IPC$"
smb2.filename contains ".exe"
\`\`\`

### Kerberos Analysis
\`\`\`
# Kerberos message types
kerberos.msg_type == 10  # AS-REQ (initial auth)
kerberos.msg_type == 11  # AS-REP
kerberos.msg_type == 12  # TGS-REQ (service ticket)
kerberos.msg_type == 13  # TGS-REP

# Extract SPNs being requested
kerberos.SNameString

# Kerberoasting detection
# High volume of TGS-REQ for service accounts
kerberos.msg_type == 12 | count by ip.src
\`\`\`

### NTLM Analysis
\`\`\`
# NTLM message types
ntlmssp.messagetype == 1  # NEGOTIATE
ntlmssp.messagetype == 2  # CHALLENGE
ntlmssp.messagetype == 3  # AUTHENTICATE

# Extract NTLM hashes (for cracking)
# From AUTHENTICATE message:
# - Domain, Username, Hostname
# - NTProofStr, NTLMv2 Response

# Tshark extraction
tshark -r capture.pcap -Y ntlmssp.messagetype==3 \\
  -T fields -e ntlmssp.auth.domain \\
  -e ntlmssp.auth.username -e ntlmssp.ntlmv2_response
\`\`\`

### DCSync Detection
\`\`\`
# Look for DRSUAPI replication requests
drsuapi
dcerpc.dg_flags

# From non-DC to DC = suspicious
# Normal: DC to DC replication only
\`\`\`

### Extract Credentials
\`\`\`bash
# PCredz - automatic credential extraction
python3 Pcredz.py -f capture.pcap

# Extracts:
# - NTLM hashes
# - Kerberos tickets
# - HTTP Basic auth
# - FTP/Telnet credentials
\`\`\``, 3, now);

// Module 3: Network Forensics
const mod3 = insertModule.run(netPath.lastInsertRowid, 'Network Forensics', 'Investigate incidents using network evidence', 2, now);

insertTask.run(mod3.lastInsertRowid, 'Detect C2 beaconing patterns', 'Analyze network traffic for C2 indicators including periodic callback intervals, jitter patterns, DNS tunneling, HTTP beaconing, and encrypted channel characteristics to identify compromised hosts', `## C2 Beacon Detection

### Beaconing Characteristics
\`\`\`
1. Regular intervals (jitter may vary ±10-20%)
2. Similar packet sizes
3. Long-running connections or repeated short ones
4. Communication to single external IP/domain
5. Often uses common ports (80, 443, 8080)
\`\`\`

### Manual Analysis in Wireshark
\`\`\`
# Filter to suspicious destination
ip.dst == x.x.x.x

# Add columns:
# - Delta Time Displayed (time between packets)
# - TCP payload length

# Look for:
# - Regular intervals (30s, 60s, 5min)
# - Consistent or similar payload sizes
\`\`\`

### RITA (Real Intelligence Threat Analytics)
\`\`\`bash
# Install RITA
# Analyzes Zeek logs for threat indicators

# Import logs
rita import /path/to/zeek/logs mydataset

# Show beacons
rita show-beacons mydataset

# Output shows:
# - Score (0-1, higher = more suspicious)
# - Source, Destination
# - Connections count
# - Timing regularity
\`\`\`

### Beacon Detection with Python
\`\`\`python
from scapy.all import *
from collections import defaultdict
import statistics

def detect_beacons(pcap_file, threshold=0.2):
    packets = rdpcap(pcap_file)

    # Group by connection
    connections = defaultdict(list)
    for pkt in packets:
        if IP in pkt:
            key = (pkt[IP].src, pkt[IP].dst)
            connections[key].append(pkt.time)

    # Analyze timing
    for conn, times in connections.items():
        if len(times) < 10:
            continue

        deltas = [times[i+1]-times[i] for i in range(len(times)-1)]
        mean_delta = statistics.mean(deltas)
        stdev = statistics.stdev(deltas)

        # Low standard deviation = regular beaconing
        if mean_delta > 0 and stdev/mean_delta < threshold:
            print(f"Potential beacon: {conn}")
            print(f"  Interval: {mean_delta:.1f}s ± {stdev:.1f}s")
            print(f"  Connections: {len(times)}")

detect_beacons('capture.pcap')
\`\`\`

### Zeek Beacon Analysis
\`\`\`bash
# Zeek script for beacon detection
# beacon-detection.zeek

zeek -r capture.pcap beacon-detection.zeek

# Analyze conn.log
cat conn.log | zeek-cut id.orig_h id.resp_h \\
  | sort | uniq -c | sort -rn | head -20
\`\`\`

### Known C2 Indicators
\`\`\`
# Cobalt Strike default
# - 60 second beacon interval
# - /submit.php, /pixel.gif URIs
# - Malleable C2 profiles vary

# Metasploit Meterpreter
# - Reverse HTTPS on 443
# - TLS with specific JA3

# Check against threat intel
# - Abuse.ch
# - AlienVault OTX
# - VirusTotal
\`\`\``, 0, now);

insertTask.run(mod3.lastInsertRowid, 'Investigate data exfiltration', 'Identify data exfiltration by analyzing unusual outbound traffic volumes, detecting encoded data in DNS or HTTP, examining connections to cloud storage services, and correlating upload activity with sensitive file access', `## Data Exfiltration Detection

### Common Exfil Methods
\`\`\`
1. HTTPS to cloud storage (Drive, Dropbox, S3)
2. DNS tunneling (encoded data in subdomains)
3. ICMP tunneling
4. Steganography in images
5. Encrypted channels to attacker infrastructure
6. Email attachments
7. FTP/SFTP to external servers
\`\`\`

### Detect Large Outbound Transfers
\`\`\`bash
# Zeek analysis - top talkers by bytes
cat conn.log | zeek-cut id.orig_h id.resp_h orig_bytes \\
  | sort -t$'\\t' -k3 -rn | head -20

# Tshark - bytes per destination
tshark -r capture.pcap -q -z conv,ip

# Look for:
# - Unusual destinations with high byte counts
# - Outbound >> inbound (upload heavy)
\`\`\`

### DNS Exfiltration
\`\`\`
# Wireshark filters
dns.qry.name.len > 50
dns.qry.type == 16  # TXT records

# Calculate data volume in DNS
tshark -r capture.pcap -Y dns.qry.name -T fields \\
  -e dns.qry.name | awk '{total += length($1)}
  END {print "Total bytes in queries:", total}'

# Compare query vs response sizes
# Exfil: Large queries, small responses
\`\`\`

### ICMP Tunneling
\`\`\`
# Normal ICMP echo: 64 bytes data
# Tunneling: Variable, often larger

# Detect oversized ICMP
icmp && data.len > 64

# Analyze ICMP data content
# Look for ASCII, base64, structure
\`\`\`

### Cloud Storage Detection
\`\`\`
# Filter cloud storage traffic
tls.handshake.extensions_server_name contains "dropbox"
tls.handshake.extensions_server_name contains "drive.google"
tls.handshake.extensions_server_name contains "s3.amazonaws"
tls.handshake.extensions_server_name contains "blob.core.windows"

# Without decryption:
# - Track upload byte volumes
# - Note timing (after-hours activity)
# - Correlate with endpoint events
\`\`\`

### Email Exfiltration
\`\`\`
# SMTP traffic
smtp

# Large attachments
smtp.data.fragment (check sizes)

# Webmail (harder without decryption)
# Look for TLS to mail providers
tls.handshake.extensions_server_name contains "mail"
tls.handshake.extensions_server_name contains "outlook"
\`\`\`

### Encrypted Exfil Analysis
\`\`\`
# Without decryption, look for:
# 1. Unusual destination IPs/domains
# 2. Self-signed certificates
# 3. Unusual JA3 fingerprints
# 4. Large upload volumes
# 5. Timing anomalies
# 6. Connections to known-bad infrastructure
\`\`\``, 1, now);

insertTask.run(mod3.lastInsertRowid, 'Analyze malware network behavior', 'Profile malware network signatures by examining C2 protocol structures, domain generation algorithm patterns, certificate anomalies, and distinctive packet timing or sizing that characterize specific malware families', `## Malware Network Analysis

### Sandbox Network Capture
\`\`\`bash
# Set up isolated analysis network
# Route malware VM through capture point

# INetSim - simulate internet services
inetsim --config /etc/inetsim/inetsim.conf

# Captures malware attempting:
# - DNS resolution
# - HTTP/HTTPS connections
# - SMTP (spam/exfil)
# - IRC (older C2)
\`\`\`

### Initial Beacon Analysis
\`\`\`
# First network connections reveal:
# 1. C2 infrastructure
# 2. Download of additional payloads
# 3. Connectivity checks

# Wireshark - first 60 seconds
frame.time_relative < 60

# Order by time, look for:
# - DNS queries (C2 domain resolution)
# - HTTP/HTTPS to external IPs
# - Raw TCP connections
\`\`\`

### Identify Malware Families
\`\`\`
# Network signatures

# Emotet
# - Heavy use of TLS
# - Traffic to residential IPs (infected hosts)
# - Specific JA3 fingerprints

# Cobalt Strike
# - Beacon profiles (malleable C2)
# - /submit.php, /pixel.gif (default)
# - Check JA3/JARM fingerprint databases

# RATs (Remote Access Trojans)
# - Reverse connections on startup
# - Heartbeat/keepalive patterns
# - Command/response structure
\`\`\`

### Extract IOCs from Traffic
\`\`\`bash
# Extract all contacted IPs
tshark -r malware.pcap -T fields -e ip.dst | sort -u

# Extract all DNS queries
tshark -r malware.pcap -Y dns.qry.name -T fields \\
  -e dns.qry.name | sort -u

# Extract URLs
tshark -r malware.pcap -Y http.request -T fields \\
  -e http.host -e http.request.uri

# Extract User-Agents
tshark -r malware.pcap -Y http.user_agent -T fields \\
  -e http.user_agent | sort -u

# Extract TLS certificates
tshark -r malware.pcap -Y tls.handshake.certificate \\
  -T fields -e x509sat.uTF8String
\`\`\`

### Zeek for Malware Analysis
\`\`\`bash
zeek -r malware.pcap

# Key log files:
# conn.log - All connections
# dns.log - DNS queries/responses
# http.log - HTTP transactions
# ssl.log - TLS connections
# files.log - File transfers
# notice.log - Anomalies detected

# Extract file hashes
cat files.log | zeek-cut md5 sha1 sha256 filename
\`\`\`

### Submit to Threat Intel
\`\`\`
# Check extracted IOCs against:
# - VirusTotal
# - Abuse.ch (URLhaus, MalwareBazaar)
# - AlienVault OTX
# - Shodan (for infrastructure)

# Automate with APIs
curl "https://www.virustotal.com/api/v3/ip_addresses/{ip}" \\
  -H "x-apikey: YOUR_API_KEY"
\`\`\``, 2, now);

insertTask.run(mod3.lastInsertRowid, 'Build network forensics timeline', 'Correlate packet timestamps with host logs to reconstruct attack sequences, mapping initial compromise, lateral movement, and data exfiltration phases into a coherent forensic timeline with evidence citations', `## Attack Timeline Construction

### Timeline Data Sources
\`\`\`
1. Packet captures (pcap)
2. Zeek/Bro logs
3. Firewall logs
4. IDS/IPS alerts
5. Netflow data
6. DNS logs
7. Proxy logs
\`\`\`

### Zeek Log Timeline
\`\`\`bash
# Combine relevant Zeek logs with timestamps
cat conn.log | zeek-cut -d ts id.orig_h id.resp_h \\
  id.resp_p proto service > timeline.txt

cat dns.log | zeek-cut -d ts id.orig_h query >> timeline.txt

cat http.log | zeek-cut -d ts id.orig_h host uri >> timeline.txt

# Sort by timestamp
sort -t$'\\t' -k1 timeline.txt > sorted_timeline.txt
\`\`\`

### Wireshark Timeline
\`\`\`
# Add absolute time column
Edit → Preferences → Columns → Add "Absolute Time"

# Export specific fields
File → Export Packet Dissections → As CSV

# Fields to include:
# - Absolute time
# - Source/Dest IP
# - Protocol
# - Info
\`\`\`

### Key Events to Identify
\`\`\`
## Initial Access
- First connection to attacker IP
- Malware download (HTTP GET for .exe, .dll)
- Phishing link clicked

## Execution
- First C2 beacon
- Additional payload downloads

## Persistence
- Scheduled task creation (network visible?)
- Additional malware drops

## Discovery
- Port scans from infected host
- LDAP queries (AD enumeration)
- SMB share enumeration

## Lateral Movement
- SMB connections to internal hosts
- RDP connections
- WMI/PSExec traffic

## Exfiltration
- Large outbound transfers
- DNS tunneling
- Cloud storage uploads

## Command and Control
- Ongoing beacon patterns
- Command execution patterns
\`\`\`

### Correlation Script
\`\`\`python
import pandas as pd
from datetime import datetime

# Load multiple log sources
conn = pd.read_csv('conn.log', sep='\\t')
dns = pd.read_csv('dns.log', sep='\\t')
http = pd.read_csv('http.log', sep='\\t')

# Normalize timestamps
# Combine sources
# Sort by time

# Filter to incident timeframe
start = datetime(2024, 1, 15, 8, 0, 0)
end = datetime(2024, 1, 15, 18, 0, 0)

# Group by source IP to follow attacker
# Create narrative timeline
\`\`\`

### Timeline Documentation
\`\`\`markdown
## Incident Timeline

| Time (UTC) | Source | Event | Evidence |
|------------|--------|-------|----------|
| 08:15:32 | 10.1.1.50 | Phishing link clicked | HTTP to evil.com |
| 08:15:45 | 10.1.1.50 | Malware downloaded | GET /payload.exe |
| 08:16:02 | 10.1.1.50 | First C2 beacon | TLS to 1.2.3.4:443 |
| 08:30:00 | 10.1.1.50 | AD enumeration | LDAP queries |
| 09:15:22 | 10.1.1.50 | Lateral movement | SMB to 10.1.1.100 |
| 09:45:00 | 10.1.1.100 | Second host infected | C2 beacon |
| 14:30:00 | 10.1.1.50 | Data exfiltration | 2GB to cloud |
\`\`\``, 3, now);

// Module 4: Zeek (Bro) Deep Dive
const mod4 = insertModule.run(netPath.lastInsertRowid, 'Zeek Network Monitoring', 'Deploy and use Zeek for comprehensive network visibility', 3, now);

insertTask.run(mod4.lastInsertRowid, 'Deploy Zeek for network monitoring', 'Install Zeek on a network tap or span port, configure interfaces and log rotation, enable protocol analyzers, and set up log shipping to a SIEM for real-time network security monitoring and threat detection', `## Zeek Deployment

### Installation
\`\`\`bash
# Ubuntu/Debian
sudo apt install zeek

# CentOS/RHEL
sudo yum install zeek

# From source (latest features)
git clone --recursive https://github.com/zeek/zeek
cd zeek && ./configure && make && sudo make install
\`\`\`

### Configuration
\`\`\`bash
# /opt/zeek/etc/node.cfg
[zeek]
type=standalone
host=localhost
interface=eth0

# For cluster deployment
[manager]
type=manager
host=manager-host

[proxy]
type=proxy
host=proxy-host

[worker-1]
type=worker
host=worker1-host
interface=eth0
\`\`\`

### Network Configuration
\`\`\`bash
# /opt/zeek/etc/networks.cfg
# Define your local networks
10.0.0.0/8      Private
172.16.0.0/12   Private
192.168.0.0/16  Private
\`\`\`

### Start Zeek
\`\`\`bash
# Deploy configuration
zeekctl deploy

# Check status
zeekctl status

# Process existing pcap
zeek -r capture.pcap

# Live capture
zeek -i eth0
\`\`\`

### Log Locations
\`\`\`
/opt/zeek/logs/current/   # Active logs
/opt/zeek/logs/YYYY-MM-DD/ # Archived logs

Key logs:
- conn.log      # All connections
- dns.log       # DNS queries
- http.log      # HTTP transactions
- ssl.log       # TLS/SSL details
- files.log     # File transfers
- notice.log    # Security notices
- weird.log     # Anomalies
\`\`\`

### Essential Zeek Scripts
\`\`\`bash
# /opt/zeek/share/zeek/site/local.zeek

# Load standard scripts
@load tuning/defaults
@load misc/scan
@load frameworks/files/extract-all-files
@load protocols/ssl/validate-certs

# Custom detections
@load policy/protocols/ssl/validate-certs
@load policy/protocols/ssh/detect-bruteforcing
\`\`\``, 0, now);

insertTask.run(mod4.lastInsertRowid, 'Master Zeek log analysis with zeek-cut', 'Process Zeek connection, DNS, HTTP, and SSL logs using zeek-cut for field extraction, combined with grep, awk, and sort for hunting suspicious connections, unusual protocols, and anomalous traffic patterns', `## Zeek Log Analysis

### zeek-cut Basics
\`\`\`bash
# Extract specific fields
cat conn.log | zeek-cut id.orig_h id.resp_h id.resp_p

# With timestamps human-readable
cat conn.log | zeek-cut -d ts id.orig_h id.resp_h

# Convert to CSV
cat conn.log | zeek-cut -c ts id.orig_h id.resp_h > output.csv
\`\`\`

### Connection Analysis
\`\`\`bash
# Top talkers by connection count
cat conn.log | zeek-cut id.orig_h | sort | uniq -c | sort -rn | head

# Top destinations
cat conn.log | zeek-cut id.resp_h | sort | uniq -c | sort -rn | head

# Connections by port
cat conn.log | zeek-cut id.resp_p | sort | uniq -c | sort -rn | head

# Long-duration connections
cat conn.log | zeek-cut id.orig_h id.resp_h duration | \\
  awk '$3 > 3600' | sort -t$'\\t' -k3 -rn

# High byte transfers
cat conn.log | zeek-cut id.orig_h id.resp_h orig_bytes resp_bytes | \\
  awk '{print $0, $3+$4}' | sort -k5 -rn | head
\`\`\`

### DNS Analysis
\`\`\`bash
# All queried domains
cat dns.log | zeek-cut query | sort -u

# Query types
cat dns.log | zeek-cut qtype_name | sort | uniq -c | sort -rn

# Failed lookups (NXDOMAIN)
cat dns.log | zeek-cut query rcode_name | grep NXDOMAIN

# Long domain names (tunneling indicator)
cat dns.log | zeek-cut query | awk 'length($1) > 50'

# TXT record queries (exfil indicator)
cat dns.log | zeek-cut query qtype_name | grep TXT
\`\`\`

### HTTP Analysis
\`\`\`bash
# All accessed hosts
cat http.log | zeek-cut host | sort -u

# HTTP methods distribution
cat http.log | zeek-cut method | sort | uniq -c

# Find downloads by extension
cat http.log | zeek-cut uri | grep -E '\\.(exe|dll|ps1|bat|vbs)$'

# User agents
cat http.log | zeek-cut user_agent | sort -u

# POST requests (potential exfil)
cat http.log | zeek-cut method host uri | grep POST
\`\`\`

### SSL/TLS Analysis
\`\`\`bash
# Certificate issuers
cat ssl.log | zeek-cut issuer | sort | uniq -c | sort -rn

# Self-signed certs
cat ssl.log | zeek-cut issuer subject | awk '$1 == $2'

# SSL versions
cat ssl.log | zeek-cut version | sort | uniq -c

# JA3 fingerprints
cat ssl.log | zeek-cut ja3 | sort | uniq -c | sort -rn
\`\`\`

### File Extraction
\`\`\`bash
# List extracted files
cat files.log | zeek-cut filename md5 sha1

# Find executables
cat files.log | zeek-cut mime_type filename | grep executable
\`\`\``, 1, now);

insertTask.run(mod4.lastInsertRowid, 'Write custom Zeek detection scripts', 'Develop Zeek scripts using its event-driven scripting language to detect custom threats, track connection state, extract protocol-specific fields, and generate alerts for organization-specific attack patterns', `## Custom Zeek Scripting

### Zeek Script Basics
\`\`\`zeek
# hello.zeek
event zeek_init()
{
    print "Zeek started!";
}

event zeek_done()
{
    print "Zeek finished!";
}

# Run: zeek -r capture.pcap hello.zeek
\`\`\`

### Connection Events
\`\`\`zeek
# detect-ports.zeek
event connection_established(c: connection)
{
    if (c$id$resp_p == 4444/tcp)
    {
        print fmt("Suspicious port 4444: %s -> %s",
                  c$id$orig_h, c$id$resp_h);
    }
}
\`\`\`

### HTTP Detection
\`\`\`zeek
# detect-suspicious-ua.zeek
event http_header(c: connection, is_orig: bool,
                  name: string, value: string)
{
    if (is_orig && name == "USER-AGENT")
    {
        if (/curl|wget|python/i in value)
        {
            print fmt("Script/Tool User-Agent: %s from %s",
                      value, c$id$orig_h);
        }
    }
}

# Detect suspicious file downloads
event http_reply(c: connection, version: string,
                 code: count, reason: string)
{
    if (c$http$uri == /.*\\.(exe|dll|scr|ps1)$/)
    {
        print fmt("Executable download: %s%s",
                  c$http$host, c$http$uri);
    }
}
\`\`\`

### DNS Detection
\`\`\`zeek
# detect-dns-tunnel.zeek
event dns_request(c: connection, msg: dns_msg,
                  query: string, qtype: count, qclass: count)
{
    # Long subdomain = potential tunneling
    local parts = split_string(query, /\\./);
    if (|parts| > 0 && |parts[0]| > 30)
    {
        print fmt("Possible DNS tunnel: %s", query);
    }
}
\`\`\`

### Using Notices (Alerts)
\`\`\`zeek
# custom-notices.zeek
module CustomDetect;

export {
    redef enum Notice::Type += {
        Suspicious_Connection,
        Potential_Exfil,
    };
}

event connection_established(c: connection)
{
    if (c$id$resp_p in set(4444/tcp, 5555/tcp, 1337/tcp))
    {
        NOTICE([
            $note=Suspicious_Connection,
            $msg=fmt("Connection to suspicious port: %s:%s",
                    c$id$resp_h, c$id$resp_p),
            $conn=c,
            $identifier=cat(c$id$orig_h)
        ]);
    }
}
\`\`\`

### Intelligence Framework
\`\`\`zeek
# Load threat intel
@load frameworks/intel/seen
@load frameworks/intel/do_notice

# intel.dat format:
#fields indicator indicator_type meta.source
# 1.2.3.4 Intel::ADDR MyThreatFeed
# evil.com Intel::DOMAIN MyThreatFeed
# abc123... Intel::FILE_HASH MyThreatFeed

redef Intel::read_files += { "/opt/zeek/intel/intel.dat" };
\`\`\`

### Deploy Custom Scripts
\`\`\`bash
# Add to /opt/zeek/share/zeek/site/local.zeek
@load ./custom-notices.zeek
@load ./detect-dns-tunnel.zeek

# Reload Zeek
zeekctl deploy
\`\`\``, 2, now);

console.log('Seeded: Network Analysis & Traffic Forensics');
console.log('  - 4 modules, 12 detailed tasks');

sqlite.close();
