#!/usr/bin/env npx tsx
/**
 * Seed: Scanning & Enumeration Tools
 * Nmap, Masscan, Nuclei, ffuf
 */

import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));
const now = Date.now();

const insertPath = db.prepare(`
	INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, created_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
	INSERT INTO modules (path_id, name, description, order_index, created_at)
	VALUES (?, ?, ?, ?, ?)
`);

const insertTask = db.prepare(`
	INSERT INTO tasks (module_id, title, description, details, order_index, created_at)
	VALUES (?, ?, ?, ?, ?, ?)
`);

// ============================================================================
// NMAP REIMPLEMENTATION
// ============================================================================
const nmapPath = insertPath.run(
	'Reimplement: Nmap Network Scanner',
	'Build a network scanner like Nmap. TCP/UDP scanning, service detection, OS fingerprinting, and NSE-like scripting engine.',
	'cyan',
	'Rust+Python',
	'advanced',
	14,
	'TCP/IP, raw sockets, service detection, OS fingerprinting, async I/O',
	now
);

const nmapMod1 = insertModule.run(nmapPath.lastInsertRowid, 'Port Scanning Engine', 'Multi-technique port scanning', 0, now);

insertTask.run(nmapMod1.lastInsertRowid, 'Build TCP SYN Scanner', 'Implement stealthy port scanning using raw sockets to send SYN packets and analyze responses (SYN-ACK for open, RST for closed) without completing the TCP handshake, avoiding full connection logging on targets', `## TCP SYN Scanner (Half-Open)

### Overview
Build a fast SYN scanner using raw sockets for stealthy port detection.

### Rust Implementation
\`\`\`rust
//! TCP SYN Scanner using raw sockets
//! Requires root/admin privileges

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;
use tokio::sync::mpsc;
use pnet::packet::tcp::{TcpPacket, MutableTcpPacket, TcpFlags};
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::ipv4::{Ipv4Packet, MutableIpv4Packet};
use pnet::transport::{
    transport_channel, TransportChannelType,
    TransportProtocol, TransportSender, TransportReceiver
};
use pnet::packet::Packet;

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub port: u16,
    pub state: PortState,
    pub service: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PortState {
    Open,
    Closed,
    Filtered,
}

pub struct SynScanner {
    source_ip: Ipv4Addr,
    source_port: u16,
    timeout: Duration,
    tx: TransportSender,
    rx: TransportReceiver,
}

impl SynScanner {
    pub fn new(source_ip: Ipv4Addr) -> Result<Self, Box<dyn std::error::Error>> {
        let protocol = TransportChannelType::Layer4(
            TransportProtocol::Ipv4(IpNextHeaderProtocols::Tcp)
        );

        let (tx, rx) = transport_channel(4096, protocol)?;

        Ok(Self {
            source_ip,
            source_port: rand::random::<u16>() | 0x8000, // Random high port
            timeout: Duration::from_secs(3),
            tx,
            rx,
        })
    }

    pub async fn scan_port(
        &mut self,
        target: Ipv4Addr,
        port: u16
    ) -> ScanResult {
        // Send SYN packet
        if let Err(e) = self.send_syn(target, port) {
            return ScanResult {
                port,
                state: PortState::Filtered,
                service: None,
            };
        }

        // Wait for response
        match self.receive_response(target, port).await {
            Some(flags) => {
                if flags & TcpFlags::SYN != 0 && flags & TcpFlags::ACK != 0 {
                    // SYN-ACK = Open
                    // Send RST to close connection
                    let _ = self.send_rst(target, port);
                    ScanResult {
                        port,
                        state: PortState::Open,
                        service: self.guess_service(port),
                    }
                } else if flags & TcpFlags::RST != 0 {
                    // RST = Closed
                    ScanResult {
                        port,
                        state: PortState::Closed,
                        service: None,
                    }
                } else {
                    ScanResult {
                        port,
                        state: PortState::Filtered,
                        service: None,
                    }
                }
            }
            None => {
                // No response = Filtered
                ScanResult {
                    port,
                    state: PortState::Filtered,
                    service: None,
                }
            }
        }
    }

    fn send_syn(&mut self, target: Ipv4Addr, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let mut tcp_buffer = [0u8; 20];
        let mut tcp_packet = MutableTcpPacket::new(&mut tcp_buffer).unwrap();

        tcp_packet.set_source(self.source_port);
        tcp_packet.set_destination(port);
        tcp_packet.set_sequence(rand::random::<u32>());
        tcp_packet.set_acknowledgement(0);
        tcp_packet.set_data_offset(5);
        tcp_packet.set_flags(TcpFlags::SYN);
        tcp_packet.set_window(65535);
        tcp_packet.set_urgent_ptr(0);

        // Calculate checksum
        let checksum = pnet::packet::tcp::ipv4_checksum(
            &tcp_packet.to_immutable(),
            &self.source_ip,
            &target
        );
        tcp_packet.set_checksum(checksum);

        self.tx.send_to(tcp_packet, IpAddr::V4(target))?;
        Ok(())
    }

    fn send_rst(&mut self, target: Ipv4Addr, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let mut tcp_buffer = [0u8; 20];
        let mut tcp_packet = MutableTcpPacket::new(&mut tcp_buffer).unwrap();

        tcp_packet.set_source(self.source_port);
        tcp_packet.set_destination(port);
        tcp_packet.set_sequence(rand::random::<u32>());
        tcp_packet.set_flags(TcpFlags::RST);
        tcp_packet.set_window(0);
        tcp_packet.set_data_offset(5);

        let checksum = pnet::packet::tcp::ipv4_checksum(
            &tcp_packet.to_immutable(),
            &self.source_ip,
            &target
        );
        tcp_packet.set_checksum(checksum);

        self.tx.send_to(tcp_packet, IpAddr::V4(target))?;
        Ok(())
    }

    async fn receive_response(&mut self, target: Ipv4Addr, port: u16) -> Option<u8> {
        use tokio::time::timeout;

        let result = timeout(self.timeout, async {
            loop {
                let mut buffer = [0u8; 1500];

                // This is simplified - real implementation would use async I/O
                // and filter for specific source IP/port
            }
        }).await;

        None
    }

    fn guess_service(&self, port: u16) -> Option<String> {
        match port {
            21 => Some("ftp".to_string()),
            22 => Some("ssh".to_string()),
            23 => Some("telnet".to_string()),
            25 => Some("smtp".to_string()),
            53 => Some("dns".to_string()),
            80 => Some("http".to_string()),
            110 => Some("pop3".to_string()),
            143 => Some("imap".to_string()),
            443 => Some("https".to_string()),
            445 => Some("microsoft-ds".to_string()),
            3306 => Some("mysql".to_string()),
            3389 => Some("ms-wbt-server".to_string()),
            5432 => Some("postgresql".to_string()),
            6379 => Some("redis".to_string()),
            8080 => Some("http-proxy".to_string()),
            _ => None,
        }
    }

    pub async fn scan_ports(
        &mut self,
        target: Ipv4Addr,
        ports: Vec<u16>
    ) -> Vec<ScanResult> {
        let mut results = Vec::new();

        for port in ports {
            let result = self.scan_port(target, port).await;
            if result.state == PortState::Open {
                println!("[+] {}:{} - OPEN ({})",
                    target, port,
                    result.service.as_deref().unwrap_or("unknown")
                );
            }
            results.push(result);
        }

        results
    }
}

// TCP Connect scanner (no raw sockets needed)
pub struct ConnectScanner {
    timeout: Duration,
    concurrency: usize,
}

impl ConnectScanner {
    pub fn new(timeout: Duration, concurrency: usize) -> Self {
        Self { timeout, concurrency }
    }

    pub async fn scan_port(&self, target: &str, port: u16) -> ScanResult {
        use tokio::net::TcpStream;
        use tokio::time::timeout;

        let addr = format!("{}:{}", target, port);

        match timeout(self.timeout, TcpStream::connect(&addr)).await {
            Ok(Ok(_stream)) => {
                ScanResult {
                    port,
                    state: PortState::Open,
                    service: None,
                }
            }
            Ok(Err(_)) => {
                ScanResult {
                    port,
                    state: PortState::Closed,
                    service: None,
                }
            }
            Err(_) => {
                ScanResult {
                    port,
                    state: PortState::Filtered,
                    service: None,
                }
            }
        }
    }

    pub async fn scan_range(
        &self,
        target: &str,
        ports: Vec<u16>
    ) -> Vec<ScanResult> {
        use futures::stream::{self, StreamExt};

        let results: Vec<ScanResult> = stream::iter(ports)
            .map(|port| async move {
                self.scan_port(target, port).await
            })
            .buffer_unordered(self.concurrency)
            .collect()
            .await;

        results
    }
}

#[tokio::main]
async fn main() {
    let scanner = ConnectScanner::new(
        Duration::from_secs(2),
        100  // Concurrent connections
    );

    let target = "scanme.nmap.org";
    let ports: Vec<u16> = (1..=1000).collect();

    println!("[*] Scanning {} ports on {}", ports.len(), target);

    let results = scanner.scan_range(target, ports).await;

    let open_ports: Vec<_> = results
        .iter()
        .filter(|r| r.state == PortState::Open)
        .collect();

    println!("\\n[+] Found {} open ports", open_ports.len());
    for result in open_ports {
        println!("  {}/tcp open", result.port);
    }
}
\`\`\`

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Async TCP Connect Scanner
"""

import asyncio
import socket
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class PortState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    FILTERED = "filtered"


@dataclass
class ScanResult:
    port: int
    state: PortState
    service: Optional[str] = None
    banner: Optional[str] = None


class AsyncScanner:
    def __init__(self, timeout: float = 2.0, concurrency: int = 100):
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)

    async def scan_port(self, host: str, port: int) -> ScanResult:
        async with self.semaphore:
            try:
                future = asyncio.open_connection(host, port)
                reader, writer = await asyncio.wait_for(future, self.timeout)

                # Try to grab banner
                banner = None
                try:
                    writer.write(b"\\r\\n")
                    await writer.drain()
                    data = await asyncio.wait_for(reader.read(1024), 1.0)
                    banner = data.decode('utf-8', errors='ignore').strip()
                except:
                    pass

                writer.close()
                await writer.wait_closed()

                return ScanResult(
                    port=port,
                    state=PortState.OPEN,
                    service=self._guess_service(port),
                    banner=banner
                )

            except asyncio.TimeoutError:
                return ScanResult(port=port, state=PortState.FILTERED)
            except ConnectionRefusedError:
                return ScanResult(port=port, state=PortState.CLOSED)
            except Exception:
                return ScanResult(port=port, state=PortState.FILTERED)

    async def scan_range(self, host: str, ports: List[int]) -> List[ScanResult]:
        tasks = [self.scan_port(host, port) for port in ports]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _guess_service(self, port: int) -> Optional[str]:
        services = {
            21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
            53: 'dns', 80: 'http', 110: 'pop3', 143: 'imap',
            443: 'https', 445: 'smb', 3306: 'mysql', 3389: 'rdp',
            5432: 'postgresql', 6379: 'redis', 8080: 'http-proxy'
        }
        return services.get(port)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Async Port Scanner')
    parser.add_argument('host', help='Target host')
    parser.add_argument('-p', '--ports', default='1-1000',
                        help='Port range (e.g., 1-1000 or 22,80,443)')
    parser.add_argument('-t', '--timeout', type=float, default=2.0)
    parser.add_argument('-c', '--concurrency', type=int, default=100)

    args = parser.parse_args()

    # Parse ports
    if '-' in args.ports:
        start, end = map(int, args.ports.split('-'))
        ports = list(range(start, end + 1))
    else:
        ports = [int(p) for p in args.ports.split(',')]

    scanner = AsyncScanner(args.timeout, args.concurrency)

    print(f"[*] Scanning {len(ports)} ports on {args.host}")

    results = await scanner.scan_range(args.host, ports)

    open_ports = [r for r in results if r.state == PortState.OPEN]

    print(f"\\n[+] Found {len(open_ports)} open ports:")
    for r in sorted(open_ports, key=lambda x: x.port):
        service = r.service or "unknown"
        print(f"  {r.port}/tcp open  {service}")
        if r.banner:
            print(f"    Banner: {r.banner[:50]}")


if __name__ == '__main__':
    asyncio.run(main())
\`\`\`
`, 0, now);

insertTask.run(nmapMod1.lastInsertRowid, 'Build Service Detection Engine', 'Send protocol-specific probe packets to open ports and analyze responses using signature matching to identify running services, versions, and underlying operating systems through banner grabbing and protocol fingerprinting', `## Service Detection Engine

### Overview
Identify services by sending probes and matching responses.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Service Detection Engine
Probe services and identify versions
"""

import socket
import ssl
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ServiceMatch:
    name: str
    product: Optional[str] = None
    version: Optional[str] = None
    extra_info: Optional[str] = None
    confidence: int = 100


@dataclass
class Probe:
    name: str
    protocol: str  # tcp/udp
    probe_string: bytes
    ports: List[int]
    rarity: int = 5
    matches: List[Tuple[str, str, str]] = None  # (pattern, name, versioninfo)


class ServiceDetector:
    """Detect services by probing and pattern matching"""

    def __init__(self):
        self.probes = self._load_probes()

    def _load_probes(self) -> List[Probe]:
        """Load service probes (like nmap-service-probes)"""

        return [
            # NULL probe - just connect and read
            Probe(
                name="NULL",
                protocol="tcp",
                probe_string=b"",
                ports=[],
                rarity=1,
                matches=[
                    (r"^SSH-([\\d.]+)-OpenSSH[_-]([\\w.]+)", "ssh", "OpenSSH \\\\2"),
                    (r"^SSH-([\\d.]+)-dropbear", "ssh", "Dropbear"),
                    (r"^220.*FTP", "ftp", None),
                    (r"^220.*SMTP", "smtp", None),
                    (r"^\\* OK.*IMAP", "imap", None),
                    (r"^\\+OK.*POP3", "pop3", None),
                ]
            ),

            # HTTP probe
            Probe(
                name="GetRequest",
                protocol="tcp",
                probe_string=b"GET / HTTP/1.0\\r\\n\\r\\n",
                ports=[80, 8080, 8000, 8888],
                rarity=1,
                matches=[
                    (r"^HTTP/1\\.[01] \\d+ ", "http", None),
                    (r"Server: Apache/([\\d.]+)", "http", "Apache \\\\1"),
                    (r"Server: nginx/([\\d.]+)", "http", "nginx \\\\1"),
                    (r"Server: Microsoft-IIS/([\\d.]+)", "http", "IIS \\\\1"),
                ]
            ),

            # HTTPS probe
            Probe(
                name="SSLSessionReq",
                protocol="tcp",
                probe_string=b"",  # SSL handshake
                ports=[443, 8443, 9443],
                rarity=1,
                matches=[
                    (r".*", "https", None),
                ]
            ),

            # MySQL probe
            Probe(
                name="MySQL",
                protocol="tcp",
                probe_string=b"",
                ports=[3306],
                rarity=2,
                matches=[
                    (r"^.\\x00\\x00\\x00\\x0a([\\d.]+)", "mysql", "MySQL \\\\1"),
                    (r"^.\\x00\\x00\\x00\\xffj\\x04", "mysql", "MySQL (access denied)"),
                ]
            ),

            # Redis probe
            Probe(
                name="Redis",
                protocol="tcp",
                probe_string=b"*1\\r\\n$4\\r\\nPING\\r\\n",
                ports=[6379],
                rarity=2,
                matches=[
                    (r"^\\+PONG", "redis", "Redis"),
                    (r"^-NOAUTH", "redis", "Redis (auth required)"),
                ]
            ),

            # SMB probe
            Probe(
                name="SMB",
                protocol="tcp",
                probe_string=self._build_smb_negotiate(),
                ports=[445],
                rarity=2,
                matches=[
                    (r"^\\x00\\x00", "microsoft-ds", "SMB"),
                ]
            ),

            # DNS probe
            Probe(
                name="DNSVersionBindReq",
                protocol="udp",
                probe_string=self._build_dns_version_query(),
                ports=[53],
                rarity=2,
                matches=[
                    (r"version\\.bind", "dns", None),
                ]
            ),
        ]

    def _build_smb_negotiate(self) -> bytes:
        """Build SMB negotiate request"""
        # Simplified SMB1 negotiate
        return (
            b"\\x00\\x00\\x00\\x85"  # NetBIOS
            b"\\xff\\x53\\x4d\\x42"  # SMB
            b"\\x72"                 # Negotiate
            b"\\x00\\x00\\x00\\x00"  # Status
            b"\\x18\\x53\\xc0"       # Flags
            b"\\x00\\x00"            # Flags2
            + bytes(12) +            # PID, UID, MID
            b"\\x00\\x62"            # Byte count
            b"\\x02NT LM 0.12\\x00"  # Dialects
        )

    def _build_dns_version_query(self) -> bytes:
        """Build DNS version.bind query"""
        return (
            b"\\x00\\x00"  # Transaction ID
            b"\\x00\\x00"  # Flags (standard query)
            b"\\x00\\x01"  # Questions
            b"\\x00\\x00"  # Answers
            b"\\x00\\x00"  # Authority
            b"\\x00\\x00"  # Additional
            b"\\x07version\\x04bind\\x00"  # version.bind
            b"\\x00\\x10"  # Type TXT
            b"\\x00\\x03"  # Class CHAOS
        )

    def detect_service(
        self,
        host: str,
        port: int,
        protocol: str = "tcp",
        timeout: float = 5.0
    ) -> Optional[ServiceMatch]:
        """Detect service on a port"""

        # Get applicable probes
        probes = sorted(
            [p for p in self.probes if p.protocol == protocol],
            key=lambda p: (port not in p.ports, p.rarity)
        )

        for probe in probes:
            try:
                response = self._send_probe(host, port, probe, timeout)

                if response:
                    match = self._match_response(response, probe)
                    if match:
                        return match

            except Exception as e:
                continue

        return None

    def _send_probe(
        self,
        host: str,
        port: int,
        probe: Probe,
        timeout: float
    ) -> Optional[bytes]:
        """Send probe and get response"""

        if probe.protocol == "tcp":
            return self._send_tcp_probe(host, port, probe, timeout)
        else:
            return self._send_udp_probe(host, port, probe, timeout)

    def _send_tcp_probe(
        self,
        host: str,
        port: int,
        probe: Probe,
        timeout: float
    ) -> Optional[bytes]:
        """Send TCP probe"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((host, port))

            # Check for SSL
            if probe.name == "SSLSessionReq" or port in [443, 8443]:
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

                sock = context.wrap_socket(sock, server_hostname=host)

            if probe.probe_string:
                sock.send(probe.probe_string)

            response = sock.recv(4096)
            return response

        except Exception:
            return None
        finally:
            sock.close()

    def _send_udp_probe(
        self,
        host: str,
        port: int,
        probe: Probe,
        timeout: float
    ) -> Optional[bytes]:
        """Send UDP probe"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)

        try:
            sock.sendto(probe.probe_string, (host, port))
            response, _ = sock.recvfrom(4096)
            return response
        except:
            return None
        finally:
            sock.close()

    def _match_response(
        self,
        response: bytes,
        probe: Probe
    ) -> Optional[ServiceMatch]:
        """Match response against probe patterns"""

        response_str = response.decode('utf-8', errors='ignore')

        for pattern, name, version_info in probe.matches:
            match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)

            if match:
                # Extract version info
                product = None
                version = None

                if version_info:
                    # Replace backreferences
                    version_str = version_info
                    for i, group in enumerate(match.groups(), 1):
                        if group:
                            version_str = version_str.replace(f"\\\\{i}", group)

                    parts = version_str.split()
                    if len(parts) >= 1:
                        product = parts[0]
                    if len(parts) >= 2:
                        version = parts[1]

                return ServiceMatch(
                    name=name,
                    product=product,
                    version=version
                )

        return None


async def main():
    import asyncio

    detector = ServiceDetector()

    targets = [
        ("scanme.nmap.org", 22),
        ("scanme.nmap.org", 80),
        ("google.com", 443),
    ]

    for host, port in targets:
        print(f"[*] Probing {host}:{port}")

        result = detector.detect_service(host, port)

        if result:
            print(f"  Service: {result.name}")
            if result.product:
                print(f"  Product: {result.product}")
            if result.version:
                print(f"  Version: {result.version}")
        else:
            print(f"  Service: unknown")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
\`\`\`
`, 1, now);

// ============================================================================
// FFUF / FEROXBUSTER
// ============================================================================
const ffufPath = insertPath.run(
	'Reimplement: Web Fuzzer (ffuf/feroxbuster)',
	'Build a fast web fuzzer for directory discovery, parameter fuzzing, and virtual host enumeration.',
	'green',
	'Rust+Go',
	'intermediate',
	8,
	'HTTP, async I/O, fuzzing, wordlists, recursion',
	now
);

const ffufMod1 = insertModule.run(ffufPath.lastInsertRowid, 'Web Fuzzing Engine', 'High-performance HTTP fuzzer', 0, now);

insertTask.run(ffufMod1.lastInsertRowid, 'Build Directory Buster', 'Implement fast directory and file discovery using concurrent HTTP requests, wordlist processing, response filtering by size/status/word count, and recursive scanning for hidden web content enumeration', `## Directory Buster

### Overview
Fast directory/file discovery using wordlists.

### Go Implementation
\`\`\`go
package main

import (
    "bufio"
    "crypto/tls"
    "flag"
    "fmt"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"
)

type Result struct {
    URL        string
    StatusCode int
    Length     int64
    Words      int
    Lines      int
}

type Fuzzer struct {
    client      *http.Client
    baseURL     string
    wordlist    string
    extensions  []string
    threads     int
    filterCodes []int
    results     chan Result
    wg          sync.WaitGroup
}

func NewFuzzer(baseURL, wordlist string, threads int) *Fuzzer {
    transport := &http.Transport{
        TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
        MaxIdleConns:    threads,
        MaxConnsPerHost: threads,
    }

    client := &http.Client{
        Transport: transport,
        Timeout:   10 * time.Second,
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            return http.ErrUseLastResponse
        },
    }

    return &Fuzzer{
        client:   client,
        baseURL:  strings.TrimRight(baseURL, "/"),
        wordlist: wordlist,
        threads:  threads,
        results:  make(chan Result, 1000),
    }
}

func (f *Fuzzer) SetExtensions(exts string) {
    if exts != "" {
        f.extensions = strings.Split(exts, ",")
    }
}

func (f *Fuzzer) SetFilterCodes(codes []int) {
    f.filterCodes = codes
}

func (f *Fuzzer) worker(words <-chan string) {
    defer f.wg.Done()

    for word := range words {
        paths := []string{word}

        // Add extensions
        for _, ext := range f.extensions {
            paths = append(paths, word+"."+ext)
        }

        for _, path := range paths {
            url := f.baseURL + "/" + path
            result := f.checkURL(url)

            if result != nil && !f.shouldFilter(result) {
                f.results <- *result
            }
        }
    }
}

func (f *Fuzzer) checkURL(url string) *Result {
    resp, err := f.client.Get(url)
    if err != nil {
        return nil
    }
    defer resp.Body.Close()

    // Read body for stats
    body := make([]byte, 0)
    buf := make([]byte, 4096)
    for {
        n, err := resp.Body.Read(buf)
        body = append(body, buf[:n]...)
        if err != nil {
            break
        }
    }

    bodyStr := string(body)
    words := len(strings.Fields(bodyStr))
    lines := strings.Count(bodyStr, "\\n")

    return &Result{
        URL:        url,
        StatusCode: resp.StatusCode,
        Length:     int64(len(body)),
        Words:      words,
        Lines:      lines,
    }
}

func (f *Fuzzer) shouldFilter(r *Result) bool {
    for _, code := range f.filterCodes {
        if r.StatusCode == code {
            return true
        }
    }
    return false
}

func (f *Fuzzer) Run() error {
    file, err := os.Open(f.wordlist)
    if err != nil {
        return err
    }
    defer file.Close()

    words := make(chan string, f.threads*10)

    // Start workers
    for i := 0; i < f.threads; i++ {
        f.wg.Add(1)
        go f.worker(words)
    }

    // Start result printer
    go func() {
        for result := range f.results {
            fmt.Printf("[%d] %-50s [Size: %d, Words: %d, Lines: %d]\\n",
                result.StatusCode,
                result.URL,
                result.Length,
                result.Words,
                result.Lines,
            )
        }
    }()

    // Read wordlist
    scanner := bufio.NewScanner(file)
    count := 0
    for scanner.Scan() {
        word := strings.TrimSpace(scanner.Text())
        if word != "" && !strings.HasPrefix(word, "#") {
            words <- word
            count++
        }
    }

    close(words)
    f.wg.Wait()
    close(f.results)

    fmt.Printf("\\n[+] Processed %d words\\n", count)
    return nil
}

func main() {
    url := flag.String("u", "", "Target URL")
    wordlist := flag.String("w", "", "Wordlist path")
    threads := flag.Int("t", 50, "Number of threads")
    extensions := flag.String("e", "", "Extensions (comma-separated)")
    filterCodes := flag.String("fc", "404", "Filter status codes")

    flag.Parse()

    if *url == "" || *wordlist == "" {
        fmt.Println("Usage: gobuster -u <URL> -w <wordlist>")
        flag.PrintDefaults()
        os.Exit(1)
    }

    fuzzer := NewFuzzer(*url, *wordlist, *threads)
    fuzzer.SetExtensions(*extensions)

    // Parse filter codes
    if *filterCodes != "" {
        var codes []int
        for _, c := range strings.Split(*filterCodes, ",") {
            var code int
            fmt.Sscanf(c, "%d", &code)
            codes = append(codes, code)
        }
        fuzzer.SetFilterCodes(codes)
    }

    fmt.Printf("[*] Target: %s\\n", *url)
    fmt.Printf("[*] Wordlist: %s\\n", *wordlist)
    fmt.Printf("[*] Threads: %d\\n", *threads)
    fmt.Println()

    if err := fuzzer.Run(); err != nil {
        fmt.Printf("Error: %v\\n", err)
        os.Exit(1)
    }
}
\`\`\`

### Rust Implementation
\`\`\`rust
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::fs::File;
use futures::stream::{self, StreamExt};

#[derive(Debug)]
struct Result {
    url: String,
    status: u16,
    length: usize,
}

struct Fuzzer {
    client: Client,
    base_url: String,
    semaphore: Arc<Semaphore>,
}

impl Fuzzer {
    fn new(base_url: &str, concurrency: usize) -> Self {
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            semaphore: Arc::new(Semaphore::new(concurrency)),
        }
    }

    async fn check(&self, word: &str) -> Option<Result> {
        let _permit = self.semaphore.acquire().await.ok()?;

        let url = format!("{}/{}", self.base_url, word);

        match self.client.get(&url).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp.bytes().await.unwrap_or_default();

                if status != 404 {
                    Some(Result {
                        url,
                        status,
                        length: body.len(),
                    })
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    async fn run(&self, wordlist: &str) -> Vec<Result> {
        let file = File::open(wordlist).await.unwrap();
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut words = Vec::new();
        while let Ok(Some(line)) = lines.next_line().await {
            let word = line.trim().to_string();
            if !word.is_empty() && !word.starts_with('#') {
                words.push(word);
            }
        }

        println!("[*] Loaded {} words", words.len());

        let results: Vec<Option<Result>> = stream::iter(words)
            .map(|word| async move {
                self.check(&word).await
            })
            .buffer_unordered(100)
            .collect()
            .await;

        results.into_iter().flatten().collect()
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <url> <wordlist>", args[0]);
        return;
    }

    let fuzzer = Fuzzer::new(&args[1], 100);

    println!("[*] Starting fuzzer on {}", args[1]);

    let results = fuzzer.run(&args[2]).await;

    println!("\\n[+] Found {} results:", results.len());
    for r in results {
        println!("[{}] {} (size: {})", r.status, r.url, r.length);
    }
}
\`\`\`
`, 0, now);

// ============================================================================
// NUCLEI
// ============================================================================
const nucleiPath = insertPath.run(
	'Reimplement: Nuclei Template Scanner',
	'Build a template-based vulnerability scanner like Nuclei. YAML templates, multi-protocol support, and automatic detection.',
	'yellow',
	'Go+Python',
	'advanced',
	12,
	'YAML, HTTP, templates, vulnerability detection, DSL',
	now
);

const nucleiMod1 = insertModule.run(nucleiPath.lastInsertRowid, 'Template Engine', 'YAML template parsing and execution', 0, now);

insertTask.run(nucleiMod1.lastInsertRowid, 'Build Template Parser', 'Parse YAML-based vulnerability templates supporting HTTP request definitions, matchers (status, regex, word), extractors, conditional logic, and template variables for flexible security scanning workflows', `## Template Parser

### Overview
Parse YAML templates for vulnerability detection.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Nuclei-style Template Scanner
"""

import yaml
import re
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import concurrent.futures


@dataclass
class TemplateInfo:
    name: str
    author: str
    severity: str
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Matcher:
    type: str  # word, regex, status, size
    part: str = "body"  # body, header, all
    words: List[str] = field(default_factory=list)
    regex: List[str] = field(default_factory=list)
    status: List[int] = field(default_factory=list)
    condition: str = "or"  # and, or
    negative: bool = False


@dataclass
class Extractor:
    type: str  # regex, kval, json
    name: str = ""
    part: str = "body"
    regex: List[str] = field(default_factory=list)
    group: int = 0


@dataclass
class HTTPRequest:
    method: str
    path: List[str]
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    matchers: List[Matcher] = field(default_factory=list)
    extractors: List[Extractor] = field(default_factory=list)
    matchers_condition: str = "or"
    redirects: bool = True
    max_redirects: int = 10


@dataclass
class Template:
    id: str
    info: TemplateInfo
    requests: List[HTTPRequest]


@dataclass
class Finding:
    template_id: str
    template_name: str
    severity: str
    url: str
    matched_at: str
    extracted: Dict[str, str] = field(default_factory=dict)


class TemplateParser:
    """Parse Nuclei YAML templates"""

    def parse(self, path: str) -> Template:
        with open(path) as f:
            data = yaml.safe_load(f)

        info = TemplateInfo(
            name=data['info'].get('name', ''),
            author=data['info'].get('author', ''),
            severity=data['info'].get('severity', 'info'),
            description=data['info'].get('description', ''),
            tags=data['info'].get('tags', '').split(',') if data['info'].get('tags') else []
        )

        requests = []
        for req_data in data.get('requests', []) or data.get('http', []):
            matchers = []
            for m in req_data.get('matchers', []):
                matchers.append(Matcher(
                    type=m.get('type', 'word'),
                    part=m.get('part', 'body'),
                    words=m.get('words', []),
                    regex=m.get('regex', []),
                    status=m.get('status', []),
                    condition=m.get('condition', 'or'),
                    negative=m.get('negative', False)
                ))

            extractors = []
            for e in req_data.get('extractors', []):
                extractors.append(Extractor(
                    type=e.get('type', 'regex'),
                    name=e.get('name', ''),
                    part=e.get('part', 'body'),
                    regex=e.get('regex', []),
                    group=e.get('group', 0)
                ))

            path = req_data.get('path', [])
            if isinstance(path, str):
                path = [path]

            requests.append(HTTPRequest(
                method=req_data.get('method', 'GET'),
                path=path,
                headers=req_data.get('headers', {}),
                body=req_data.get('body', ''),
                matchers=matchers,
                extractors=extractors,
                matchers_condition=req_data.get('matchers-condition', 'or'),
                redirects=req_data.get('redirects', True),
                max_redirects=req_data.get('max-redirects', 10)
            ))

        return Template(
            id=data.get('id', ''),
            info=info,
            requests=requests
        )


class TemplateExecutor:
    """Execute templates against targets"""

    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.verify = False
        self.timeout = timeout

    def execute(self, template: Template, target: str) -> List[Finding]:
        findings = []

        for request in template.requests:
            for path in request.path:
                # Replace variables
                url = urljoin(target, path)
                url = self._replace_variables(url, target)

                try:
                    response = self._make_request(request, url)

                    if self._check_matchers(request, response):
                        extracted = self._run_extractors(request, response)

                        findings.append(Finding(
                            template_id=template.id,
                            template_name=template.info.name,
                            severity=template.info.severity,
                            url=url,
                            matched_at=path,
                            extracted=extracted
                        ))

                except Exception as e:
                    continue

        return findings

    def _replace_variables(self, text: str, target: str) -> str:
        """Replace template variables"""
        from urllib.parse import urlparse

        parsed = urlparse(target)

        replacements = {
            '{{BaseURL}}': target.rstrip('/'),
            '{{RootURL}}': f"{parsed.scheme}://{parsed.netloc}",
            '{{Hostname}}': parsed.netloc,
            '{{Host}}': parsed.hostname or '',
            '{{Port}}': str(parsed.port or (443 if parsed.scheme == 'https' else 80)),
            '{{Path}}': parsed.path or '/',
            '{{Scheme}}': parsed.scheme,
        }

        for var, value in replacements.items():
            text = text.replace(var, value)

        return text

    def _make_request(
        self,
        request: HTTPRequest,
        url: str
    ) -> requests.Response:
        """Execute HTTP request"""

        headers = request.headers.copy()

        response = self.session.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.body if request.body else None,
            timeout=self.timeout,
            allow_redirects=request.redirects
        )

        return response

    def _check_matchers(
        self,
        request: HTTPRequest,
        response: requests.Response
    ) -> bool:
        """Check if response matches"""

        if not request.matchers:
            return True

        results = []

        for matcher in request.matchers:
            matched = self._check_matcher(matcher, response)
            if matcher.negative:
                matched = not matched
            results.append(matched)

        if request.matchers_condition == 'and':
            return all(results)
        else:
            return any(results)

    def _check_matcher(
        self,
        matcher: Matcher,
        response: requests.Response
    ) -> bool:
        """Check single matcher"""

        # Get content to match against
        if matcher.part == 'body':
            content = response.text
        elif matcher.part == 'header':
            content = str(response.headers)
        else:
            content = str(response.headers) + response.text

        if matcher.type == 'word':
            if matcher.condition == 'and':
                return all(word in content for word in matcher.words)
            else:
                return any(word in content for word in matcher.words)

        elif matcher.type == 'regex':
            for pattern in matcher.regex:
                if re.search(pattern, content, re.IGNORECASE):
                    return True
            return False

        elif matcher.type == 'status':
            return response.status_code in matcher.status

        return False

    def _run_extractors(
        self,
        request: HTTPRequest,
        response: requests.Response
    ) -> Dict[str, str]:
        """Extract data from response"""

        extracted = {}

        for extractor in request.extractors:
            if extractor.part == 'body':
                content = response.text
            elif extractor.part == 'header':
                content = str(response.headers)
            else:
                content = response.text

            if extractor.type == 'regex':
                for pattern in extractor.regex:
                    match = re.search(pattern, content)
                    if match:
                        if extractor.group and len(match.groups()) >= extractor.group:
                            value = match.group(extractor.group)
                        else:
                            value = match.group(0)
                        name = extractor.name or 'extracted'
                        extracted[name] = value
                        break

        return extracted


class Scanner:
    """Scan targets with templates"""

    def __init__(self, templates_dir: str, concurrency: int = 25):
        self.parser = TemplateParser()
        self.executor = TemplateExecutor()
        self.templates_dir = templates_dir
        self.concurrency = concurrency

    def scan(self, targets: List[str], template_files: List[str]) -> List[Finding]:
        all_findings = []

        # Parse templates
        templates = []
        for tf in template_files:
            try:
                template = self.parser.parse(tf)
                templates.append(template)
            except Exception as e:
                print(f"[-] Error parsing {tf}: {e}")

        print(f"[*] Loaded {len(templates)} templates")
        print(f"[*] Scanning {len(targets)} targets")

        # Scan
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []

            for target in targets:
                for template in templates:
                    futures.append(
                        executor.submit(self.executor.execute, template, target)
                    )

            for future in concurrent.futures.as_completed(futures):
                try:
                    findings = future.result()
                    for finding in findings:
                        print(f"[{finding.severity.upper()}] {finding.template_name} - {finding.url}")
                        all_findings.append(finding)
                except Exception as e:
                    pass

        return all_findings


# Example template YAML:
EXAMPLE_TEMPLATE = """
id: apache-status

info:
  name: Apache Status Page
  author: example
  severity: info
  tags: apache,config

requests:
  - method: GET
    path:
      - "{{BaseURL}}/server-status"
      - "{{BaseURL}}/server-info"

    matchers-condition: or
    matchers:
      - type: word
        words:
          - "Apache Server Status"
          - "Server Version:"
        condition: or

      - type: status
        status:
          - 200

    extractors:
      - type: regex
        name: version
        regex:
          - 'Server Version: Apache/([\\\\d.]+)'
        group: 1
"""

if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Template Scanner')
    parser.add_argument('-u', '--url', help='Target URL')
    parser.add_argument('-l', '--list', help='File with target URLs')
    parser.add_argument('-t', '--templates', required=True, help='Templates directory')
    parser.add_argument('-c', '--concurrency', type=int, default=25)

    args = parser.parse_args()

    # Get targets
    targets = []
    if args.url:
        targets = [args.url]
    elif args.list:
        with open(args.list) as f:
            targets = [line.strip() for line in f if line.strip()]

    # Get templates
    template_files = glob.glob(f"{args.templates}/**/*.yaml", recursive=True)

    scanner = Scanner(args.templates, args.concurrency)
    findings = scanner.scan(targets, template_files)

    print(f"\\n[+] Total findings: {len(findings)}")
\`\`\`
`, 0, now);

console.log('Seeded: Scanning & Enumeration Tools');
console.log('  - Nmap Port Scanner');
console.log('  - Service Detection');
console.log('  - Directory Buster (ffuf)');
console.log('  - Template Scanner (Nuclei)');
